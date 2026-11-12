# 加载训练好的权重

在 PyTorch 或 Hugging Face 中离线训练好探针（probe）、稀疏自编码器（SAE）、超网络（hypernetwork）、LoRA 适配层、引导向量（steering vector）或 n-gram 统计大表后，下一步是将这些权重加载到 `pllm.Workload` 状态中，供 vLLM 连续批处理引擎高效读取或在线训练。

由于现代推理引擎依赖 CUDA Graph 消除 CPU 发射开销，模型执行期间 Python 解释器并不介入。pLLM 通过声明式的状态初始化（`init=`）在引擎构建期解析张量、静态核算显存并验证契约。本指南提供六种常见场景的加载配方与排查建议。

---

## 选型指南

根据权重来源与使用方式选择对应的声明：

| 你手头有什么 | 适用场景 | 声明方式与参数 | 运行时更新与开销特征 |
| --- | --- | --- | --- |
| 离线导出的 `.safetensors` 文件 | 线性探针、分类头、先验概率大表、离线引导向量 | `pllm.InstanceState(fields, init=pllm.Pretrained(path, placement=...))` | 引擎启动时加载一次；文件哈希记入负载身份；大表可设 `placement="host"` |
| 现有 `nn.Module` 的 `state_dict` 权重 | 小型 MLP 探针、SAE 编码器、超网络 | `init=pllm.FromModule(factory, pretrained=path)` | 严格匹配模块参数（`strict=True`）；工厂函数仅捕获纯数值配置 |
| 服务模型检查点里 vLLM 用不到的额外张量 | In-Place TTT 一类检查点多出的快权重参数 | `pllm.InstanceState(fields, init=pllm.ModelCheckpoint({field: name}))` | 只读；支持 `layers=` 与 `{layer}` 模式；这些张量从 vLLM 的加载中排除，交给你的状态 |
| 宿主代码每次生成调用动态计算的值 | 逐批次实验引导向量、外部超网络单次生成的 LoRA | `init=pllm.Imported()` + `llm.generate(batch=pllm.io.StateImport(...))` | 实例状态更新并保留最新版本；请求状态每个 prompt 一行，首步前写入 |
| 另一个独立进程持续产生的数据流 | 实时视频流特征、外生传感器或强化学习控制器特征 | `pllm.InstanceState(fields, init=pllm.Fed("ordered" \| "latest"))` | 单负载限一个；有界背压环形管道；通过 `llm.feed` 异步跨进程通信 |
| 训练好的 LoRA 低秩分解权重 | 针对 attention/MLP 目标投影的 PEFT 适配增量 | `pllm.model.Adapter(state, targets, operator="low_rank", ...)` | 目标字段自动按 `a_0, b_0, a_1, b_1` 展开；请求级在 CUDA Graph 内运行 |

---

## 1. safetensors 文件中的命名张量 → pllm.Pretrained

### 你有已命名的张量字典 → 这样写

如果离线导出的权重已经存为 `.safetensors` 文件（例如键名为 `"w"` 与 `"b"`），使用 `pllm.Pretrained(path)` 初始化 `pllm.InstanceState` 或 `pllm.RequestState`。

```python
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)

# 准备一个离线训练保存的 safetensors 探针权重文件
tmp_dir = tempfile.TemporaryDirectory()
probe_path = str(Path(tmp_dir.name) / "probe.safetensors")
save_file({"w": torch.randn(896), "b": torch.zeros(1)}, probe_path)


class SentimentMonitor(pllm.Workload):
    """观测负载：使用离线训练的探针计算每个请求的最终隐层情感得分。"""

    probe = pllm.InstanceState(
        {"w": pllm.Field(lambda width, _: (width,)), "b": (1,)},
        init=pllm.Pretrained(probe_path),
    )

    @pllm.export(name="sentiment", max_rows=4, max_cols=1)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        w: torch.Tensor = pllm.parameter("probe", "w"),
        b: torch.Tensor = pllm.parameter("probe", "b"),
    ) -> torch.Tensor:
        return torch.sigmoid(h.float() @ w.float()[:, None] + b.float())


print(testing.check(SentimentMonitor(), CONFIG).passed)
```

### 规则与机制

1. **字段名与文件键名严格一致**：`fields` 字典中的每一个键必须与 safetensors 文件中保存的张量名称一一对应。文件里缺少某个字段时，构建时直接报错。
2. **文件哈希参与负载身份**：`pllm.Pretrained` 会在声明期读取并计算该文件的 SHA-256 哈希值，将其记入负载的静态指纹中。这保证了在多 worker 进程下所有实例使用完全同一份权重，并且一旦磁盘上的权重在启动后被篡改，校验机制会立即报错。
3. **自动转为存储精度**：张量加载时会自动转换为状态声明的物理存储精度（默认跟随底座模型精度，例如 `bfloat16`）。如果状态声明了 `optimizer=` 进行后续训练，存储精度默认为 `float32`。

### 超大查表与主机内存存放（placement="host"）

对于体量巨大（例如千万级 n-gram 词表先验大表、数十 GB 检索记忆）且仅需按行索引的只读状态，若全量放入 GPU 显存会导致 KV cache 空间被严重挤压。此时可以显式声明 `placement="host"`：

```python
table_path = str(Path(tmp_dir.name) / "table.safetensors")
save_file({"table": torch.randn(16, 151936)}, table_path)


class HostLookup(pllm.Workload):
    memory = pllm.InstanceState(
        {"table": lambda m: (16, m.vocab_size)},
        init=pllm.Pretrained(table_path, placement="host"),
    )

    def lookup(
        self,
        ids: torch.Tensor = pllm.input_tokens(history=1),
        table: torch.Tensor = pllm.parameter("memory", "table"),
    ) -> torch.Tensor:
        # 在 CPU 锁页内存中按 token id 查行，结果作为新张量返回
        return table[ids[:, -1].remainder(16)].float()

    @pllm.add_logits()
    def add(self, delta: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return delta


print(pllm.explain(HostLookup(), CONFIG).states[0].placement)
```

**为什么限制为只读 InstanceState？**
主机上的表只支持按行读取（gather）：每步只把用到的那些行经 PCIe 搬到 GPU。训练和写入要在显存里改整个字段，放在主机内存里每步都得整张来回搬；请求状态随请求创建和释放，本来就不是"一张大表"。所以 `placement="host"` 只用于没有优化器、没有 `carry=` 的只读 `InstanceState`。

同时，模型前向内部（如自定义算子位置）不能直接读取主机内存状态；必须先在辅助方法中 gather 所需行，再通过 `pllm.result` 传给模型位置（详见 [11 n-gram 先验/Engram 式查表](../examples/11_lookup_memory.ipynb) 与 [状态](../guides/state.md)）。

---

## 2. nn.Module 的 state_dict → pllm.FromModule

### 你有 PyTorch 模块及其 state_dict → 这样写

如果算法的核心结构是一个标准 PyTorch `nn.Module`（例如包含多层结构的 MLP 探针或稀疏自编码器 SAE），且离线保存了 `module.state_dict()`，使用 `pllm.FromModule(factory, pretrained=path)`：

```python
class SmallMLP(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, rank),
            nn.ReLU(),
            nn.Linear(rank, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 保存标准 state_dict 权重到 safetensors
mlp = SmallMLP(896, 4)
mlp_file = str(Path(tmp_dir.name) / "mlp.safetensors")
save_file(mlp.state_dict(), mlp_file)


class ModuleProbe(pllm.Workload):
    probe = pllm.InstanceState(
        init=pllm.FromModule(lambda m: SmallMLP(m.hidden_size, 4), pretrained=mlp_file)
    )

    @pllm.export(name="score", max_rows=4, max_cols=1)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        probe: nn.Module = pllm.module("probe"),
    ) -> torch.Tensor:
        return torch.sigmoid(probe(h.float()))


print(testing.check(ModuleProbe(), CONFIG).passed)
```

### 规则与机制

1. **严格加载（strict state_dict）**：底座使用 `strict=True` 加载权重字典，保存的 key 必须与模块参数命名完全匹配。如果 safetensors 文件中缺少参数或包含多余参数，会在绑定期直接抛出 `ValueError`。
2. **读取方式**：方法签名中通过 `probe: nn.Module = pllm.module("probe")` 获取可调用模块实例。若声明为 `RequestState`，每一行请求会自动使用属于该请求独立副本的参数运行。
3. **工厂函数只能捕获纯数值，不要闭包捕获模块或张量对象**：
   `factory` 只能是一个纯函数或仅引用普通标量（int、float、str）的可调用对象。
   **为什么？**
   在多进程服务引擎（如多卡张量并行或 `ModelConfig(multiprocessing=True)`）中，负载必须能够在各个独立的 worker 子进程中重新构建（rebuilt in worker processes）并在各子进程的独立 GPU 设备上下文中分配显存。如果工厂函数闭包捕获了父进程创建的 `nn.Module` 实例或 `torch.Tensor`，对象跨进程序列化会导致设备上下文错乱和内存逃逸。pLLM 的静态代码指纹分析会在分析阶段检测此类闭包，并显式拒绝。

**自测**：
在定义 `FromModule` 时，为什么写成 `mlp = MyModule(); init=pllm.FromModule(lambda m: mlp)` 会被拒绝？
<details><summary>答案</summary>
因为闭包直接引用了在主进程中提前实例化的 <code>nn.Module</code> 对象。当引擎启动多进程 worker 时，worker 子进程无法安全复用其他进程中带有本地设备绑定的模块对象。正确的做法是将初始化配置（如隐层维度、rank）作为标量传入工厂函数，让每个 worker 子进程在自身上下文中独立构建模块骨架，并通过 <code>pretrained=</code> 参数指定 safetensors 路径严格加载权重。
</details>

---

## 3. 底座模型自身的检查点张量 → pllm.ModelCheckpoint 与 pllm.io

### 你的检查点里有 vLLM 模型类用不到的张量 → 这样写

有些检查点在标准结构之外多存了一些张量，例如 In-Place TTT 社区检查点里每层的快权重卷积参数。vLLM 的模型类不认识它们，加载时会忽略；你的负载却需要它们。`pllm.ModelCheckpoint` 把这些张量从 vLLM 的加载中排除，直接读进你声明的状态：

```python
# check-docs: skip: needs an In-Place TTT checkpoint that stores these tensors
ttt_params = pllm.InstanceState(
    {"conv": pllm.Field(lambda d, _: (d, 1, 5))},
    layers=(0, 1, 2, 3),
    init=pllm.ModelCheckpoint({"conv": "model.layers.{layer}.mlp.ttt_conv.weight"}),
)
```

`layers=` 列出层号，名字里的 `{layer}` 按每一层展开。完整的用法见 `pllm.workloads.InPlaceTTTWorkload` 和 [示例 08](../examples/08_inplace_ttt.ipynb)。

### 规则与机制

1. **只读**：`pllm.ModelCheckpoint` 只用于没有优化器的 `InstanceState`。这些值来自模型检查点，是模型的一部分；需要在它们的基础上训练或调整时，把更新写进另一份状态。
2. **不要用它读 vLLM 自己要用的权重**：列在 `ModelCheckpoint` 里的张量会从 vLLM 的加载中排除。写 `model.embed_tokens.weight` 这样 vLLM 本身需要的名字，会让模型缺了这部分权重。想在负载里用 embedding、LM head 这类底座权重的一份副本，用下面的 `pllm.io.checkpoint_tensors` 在 CPU 上读出来，再经 `pllm.Pretrained` 或 `pllm.io.StateImport` 送进状态。

### 在宿主 CPU 端读取检查点工具

有时宿主代码需要在模型启动前离线读取底座张量（例如在 CPU 上根据词表嵌入构建索引表），可以使用 `pllm.io` 提供的两个轻量级工具：

```python
# 1. 定位底座模型在本地磁盘上的快照目录（无需额外网络下载）
checkpoint_dir = pllm.io.checkpoint_directory("Qwen/Qwen2.5-0.5B-Instruct")

# 2. 从分片 safetensors 或单文件中读取指定张量字典
tensors = pllm.io.checkpoint_tensors(checkpoint_dir, ("model.embed_tokens.weight",))
embed_weight = tensors["model.embed_tokens.weight"]
print(f"Loaded embed weight with shape: {tuple(embed_weight.shape)}")
```

---

## 4. 宿主代码单次调用动态传入的值 → pllm.Imported 与 StateImport

### 你每次调用 generate 都想更换权重 → 这样写

如果权重并不是全局固定的静态文件，而是由宿主在每次 `generate` 调用时动态计算生成的（例如单次实验尝试不同的激活引导向量，或者由外部超网络根据当前批次 prompt 动态生成的 LoRA 增量），声明 `init=pllm.Imported()`：

```python
class Steering(pllm.Workload):
    steer = pllm.InstanceState(
        {"vector": pllm.Field(lambda width, _: (width,))},
        init=pllm.Imported(),
    )
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


# 查看执行计划：ActivationPatch 在 CUDA Graph 内运行，不产生 eager 回退
print(repr(pllm.explain(Steering(), CONFIG).eager_reason))
```

在调用 `llm.generate` 时，通过 `batch=pllm.io.StateImport(...)` 将张量传入：

```python
# check-docs: skip: 模型推理与 generate 调用需要 GPU 引擎
from vllm import SamplingParams

vector = torch.randn(896)
greedy = SamplingParams(max_tokens=16, temperature=0.0)

with pllm.make_llm(CONFIG, workload=Steering()) as llm:
    # 第一次调用：注入特定的引导向量
    outputs_steered = llm.generate(
        ["Today is a great day"],
        greedy,
        batch=pllm.io.StateImport((vector,)),
    )

    # 后续调用：如果不传 batch=，将自动沿用上一次导入的版本
    outputs_cached = llm.generate(["Today is a great day"], greedy)
```

### 从 weights-only 检查点快速读取

如果宿主手头是 PyTorch 标准 `torch.save` 导出的 weights-only `.pt` 文件，使用 `pllm.io.load_tensor_fields` 安全提取张量：

```python
weights_path = Path(tmp_dir.name) / "weights.pt"
torch.save({"vector": torch.randn(896)}, weights_path)

# 按字段顺序读取张量元组，直接用于构造 StateImport
(imported_vector,) = pllm.io.load_tensor_fields(weights_path, ("vector",))
print(imported_vector.shape)
```

### 规则与机制

1. **InstanceState vs RequestState 的导入行为**：
   - **`InstanceState`**：传入的张量不带行维度（形状与声明完全一致）。本次调用会为该实例状态发布一个**新版本**，供该调用的所有请求读取；后续未提供 `batch=` 的 `generate` 调用会继续沿用该版本。
   - **`RequestState`**：传入的张量必须带有前导 prompt 轴（形状为 `[len(prompts), *shape]`）。该值会在每个请求的**第一次前向之前（request.prepare 阶段）**写入该请求专属的状态副本，同一个 prompt 生成的 `n` 个候选样本共享这同一行初值。
2. **为什么通过 generate(batch=...) 传入？**
   在连续批处理与 CUDA Graph 模式下，GPU 解码执行时不运行 Python 代码。如果在推理循环中动态传输张量，会强制触发 CPU 与 GPU 的同步（host sync），破坏图回放流水线。通过 `StateImport`，引擎在进入图执行前统一完成张量设备拷贝与版本注册，确保整图连续执行无阻塞。

---

## 5. 外部进程持续异步产生的值 → pllm.Fed 与 llm.feed

### 你有另一个独立进程在连续推流 → 这样写

如果数据不是单次离线生成的，而是由另一个独立进程（视频解码、音频特征提取、外部控制器）源源不断产生的，使用 `pllm.Fed`。外部进程送来的数据经一个 `@pllm.write` 转写进 `pllm.model.ActivationPatch` 读的状态，从而影响生成。现成的完整例子是 `pllm.workloads.video.VideoWorkload`：`features` 是 fed 状态，`encode` 方法把它写进 `inject`，激活补丁每个 decode 步读最新的 `inject`。

```python
from pllm.workloads.video import VideoWorkload

video_config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=1, max_model_len=64, max_num_batched_tokens=64)
print([state.name for state in pllm.explain(VideoWorkload(), video_config).states])  # ['features', 'inject']
```

启动引擎后，用 `llm.feed("features")` 拿到写入句柄，交给生产者进程按步 `publish`，结束时 `end()` 并 `close()`（完整程序见仓库里的 `examples/video_session.py`）：

```python
# check-docs: skip: needs a CUDA engine and a producer process
import multiprocessing as mp

with pllm.make_llm(video_config, workload=VideoWorkload()) as llm:
    feed = llm.feed("features")
    producer = mp.get_context("spawn").Process(target=produce, args=(feed,))  # produce() 调用 feed.publish(...)
    producer.start()
    llm.generate(["Describe the scene in this video frame:"], SamplingParams(max_tokens=8, temperature=0.0))
producer.join()
```

### 规则与机制

1. **一个负载最多声明一个 Fed 状态**：
   **为什么？** 每条输入流有一个有界的发布队列：生产者太快时 `publish` 会等待，而不是让显存无限增长；模型每步按顺序消费一次。多条流各自推进时，"这一步读的是哪一帧"不再唯一，当前实现只支持一条流。需要多路输入时，把它们作为同一个 fed 状态的多个字段一起发布。
2. **"ordered" 与 "latest" 消费模式**：
   - `"ordered"`：严格保序消费，引擎每步按序消费一个发布版本，适合视频帧解码、连续语音流等步步对齐的时序场景。
   - `"latest"`：只读最新，如果生产者推进快于模型消费，跳过堆积版本直接读取最新发布值，适合实时传感器监控或外部控制信号。
3. **不能有优化器或 carry**：
   Fed 状态是只读外部输入流，其生命周期完全由外部生产者推进，不能在引擎内进行反向传播或带跨步自累加。

---

## 6. 适配到 Adapter 的训练好 LoRA 权重

### 你有针对注意力或 MLP 投影的 LoRA 增量 → 这样写

如果手头有微调好的 PEFT / LoRA 权重（矩阵 $A$ 与 $B$），希望将其以权重增量 $\Delta W = \text{scale} \cdot B A$ 的形式应用到底座模型的线性投影层上，声明 `pllm.model.Adapter`：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLoRA(pllm.Workload):
    rank: int = 4

    @property
    def lora(self) -> pllm.RequestState:
        # 为每个目标投影声明 a 和 b 矩阵的形状函数
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Imported(),
        )

    @property
    def apply(self) -> pllm.model.Adapter:
        # 指定目标层投影，低秩算子自动计算 scale * b @ a
        targets = (
            pllm.model.Projection("q_proj", 12),
            pllm.model.Projection("v_proj", 12),
        )
        return pllm.model.Adapter("lora", targets, operator="low_rank", scale=1.0)


print(pllm.explain(PromptLoRA(), CONFIG).states[0].name)
```

### 规则与机制

1. **字段命名自动按 target 顺序展开为 a_0, b_0, a_1, b_1**：
   当 `Adapter` 绑定多个目标投影（如上面示例中的 `q_proj` 和 `v_proj`）时，`pllm.Field` 传入的形状计算函数会依次接收每个目标投影各自的 `(in_features, out_features)`。底层物理状态会自动为每个投影分配独立的字段副本，字段名按目标投影索引依次命名为：
   - 第 0 个 target（layer 12 的 `q_proj`）：对应字段 `a_0` 与 `b_0`；
   - 第 1 个 target（layer 12 的 `v_proj`）：对应字段 `a_1` 与 `b_1`。
   如果使用 `Pretrained` 加载 safetensors 文件，文件中的键名必须准确使用 `a_0, b_0, a_1, b_1` 等展开命名。
2. **全零 vs 非零初始化的动机（zero vs non-zero init）**：
   - **离线训练好的 LoRA**：直接通过 `init=pllm.Pretrained(...)` 或 `init=pllm.Imported()` 送入训练完成的非零张量，由于不含优化器，直接参与前向矩阵乘法。
   - **在线训练的 LoRA（声明了 `optimizer=`）**：
     **不要使用全零初始化（`pllm.Zeros()`）！**
     **为什么？**
     低秩增量为 $B A$。若 $A$ 和 $B$ 初值同时为 0，反向传播时 $A$ 的梯度正比于 $B^\top = 0$，$B$ 的梯度正比于 $A^\top = 0$。双方梯度全部恒等于 0，优化器永远无法迈出第一步！pLLM 会在构建期直接拦截并抛出错误。在线训练必须使用 `init=pllm.Computed(...)` 将一个因子（如 $A$）初始化为随机小值、另一个因子（$B$）初始化为 0。
3. **RequestState vs InstanceState 的执行代价**：
   - 挂载在 `RequestState` 上的 `Adapter`：不同请求在同一个批次中各自读取自己请求副本的因子，**完全保留在 CUDA Graph 内部运行**，没有任何 CPU 调度额外开销。
   - 挂载在 `InstanceState` 上的 `Adapter`：修改的是跨请求全局共享的投影权重，会导致模型前向**退回 eager 模式**运行（详见 [改模型](../guides/model-edits.md)）。

**自测**：
如果一个 LoRA 适配层包含了 4 个目标投影（`q_proj`, `k_proj`, `v_proj`, `down_proj`），且声明了 `operator="low_rank", factors=("a", "b")`，那么 safetensors 文件中应该存入哪些键？
<details><summary>答案</summary>
应该存入 8 个张量键，按 targets 传入顺序索引：<code>a_0, b_0, a_1, b_1, a_2, b_2, a_3, b_3</code>。其中 <code>a_i</code> 的形状为 <code>(rank, in_features_i)</code>，<code>b_i</code> 的形状为 <code>(out_features_i, rank)</code>。
</details>

---

## 常见报错与排查

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[加载与放置状态](../reference/errors.md#加载与放置状态)一节。

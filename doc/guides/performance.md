# 为什么几行算法会拖慢解码：pllm 性能与显存指南

写推理时算法（在线训练、记忆查表、残差改写、引导采样、稀疏注意力、自适应深度）时，最容易遇到的困惑是：**算法本身只有几行 PyTorch 张量运算，放进自回归解码（decode）循环后却慢了一大截，甚至直接撑爆显存。**

这篇指南面向正在用 `pllm` 写 workload 的开发者。我们将从自回归解码的物理特性出发，帮你建立判断性能与显存的心智模型，看清 `pllm` 已经在底层替你做了什么、不同写法各自的代价是多少，以及该怎么改代码或调参数才能跑得更快、更省显存。

### 目录

1. [心智模型：自回归 Decode 的三个物理瓶颈](#1-心智模型自回归-decode-的三个物理瓶颈)
2. [先看计划：在 CPU 上用 `pllm.explain` 算清账单](#2-先看计划在-cpu-上用-pllmexplain-算清账单)
3. [保住时间线：零同步、CUDA Graph 与旁路重叠](#3-保住时间线零同步cuda-graph-与旁路重叠)
4. [守住显存容量：分页状态、精度与预算控制](#4-守住显存容量分页状态精度与预算控制)
5. [五大算法场景的开销与写法选择](#5-五大算法场景的开销与写法选择)
   - [5.1 在线训练与引导采样](#51-在线训练与引导采样)
   - [5.2 外部大表与词表级查表](#52-外部大表与词表级查表)
   - [5.3 自定义注意力与 KV Cache 瘦身](#53-自定义注意力与-kv-cache-瘦身)
   - [5.4 模型层内残差改写](#54-模型层内残差改写)
   - [5.5 前向内训练、线性扫描与自适应深度](#55-前向内训练线性扫描与自适应深度)
6. [多卡张量并行（TP > 1）](#6-多卡张量并行tp--1)
7. [线上观测与故障排查速查表](#7-线上观测与故障排查速查表)

---

## 1. 心智模型：自回归 Decode 的三个物理瓶颈

在预填充（prefill）或离线训练中，GPU 一次处理几千个 token，瓶颈通常是算力（FLOPs）。但在自回归 decode 阶段，模型每一步只为每个并发请求生成 **1 个 token**。矩阵乘（GEMM）的算术强度（每从显存读一字节权重能做多少次乘加）约等于有效 batch 行数。在小到中等 batch 下，**GPU 算力几乎是过剩的，真正的瓶颈只有三个**：

1. **GPU 时间线有没有断掉（气泡）**：CPU 每步能不能把整串算子一口气喂给 GPU？只要有一次 CPU 等待 GPU（host sync），或者退回逐算子发射（eager 模式），GPU 就会大面积空转。
2. **每步多搬了多少字节（显存带宽与总线通信）**：权重、请求状态、词表级 logits 和主机内存查表每步要读写多少字节？
3. **给 KV Cache 剩了多少空间（显存容量）**：模型权重、CUDA Graph 私有内存池和 `pllm` 的状态预留先分走固定份额，**剩下的显存才归 vLLM 的 KV cache**。多预留几百 MiB，并发上限就会断崖式下跌。

### 一组值得记住的“标尺数字”

在评估自己的 workload 之前，先看一组基准数字。除非另有说明，本文的数字都在以下设置下测得：**单卡 RTX 4090（24 GiB）、`Qwen/Qwen2.5-0.5B-Instruct`（BF16）、128 条并发序列（8 条 prompt $\times$ 16 个采样分支）、生成 256 个 token、开启 CUDA Graph**；百分比都是相对同一次测量中原生 vLLM（开启 CUDA Graph）吞吐的比值。7B 单卡数字同样在 RTX 4090 上测得，多卡数字在 A100 上测得。

在这个设置下，原生 vLLM 跑一步 decode 约 **$3.30\text{ ms}$**（吞吐约 **$38,821\text{ tok/s}$**）：

| 操作或现象 | 实测耗时 / 吞吐代价 | 物理原因 |
| :--- | :--- | :--- |
| **退回 Eager 模式**（关掉 CUDA Graph） | 吞吐直接掉到图模式的 **45%** | 即使不加任何自定义算法，逐算子从 Python 发射的 CPU 开销也远慢于 GPU 执行 |
| **多算一次词表投影**（LM Head GEMM） | 约 **$+0.38\text{ ms}$ / 步**（超整步 10%） | 0.5B 模型的 LM Head 权重为 $896 \times 151936$（BF16），重达 **$272\text{ MB}$**，耗时几乎全花在从显存读这块权重上 |
| **原生采样器扫一遍词表** | temperature 约 **$120\ \mu\text{s}$**，top-k/top-p 约 **$360\ \mu\text{s}$**，Gumbel 采样约 **$95\ \mu\text{s}$** | $128 \times 151936$ 的词表张量在显存里反复读写 |
| **跨 PCIe 搬一次宽表**（Host $\rightarrow$ GPU） | 物理传输约 **$1.7\text{ ms}$ / 步**（超半步前向时间） | PCIe 4.0 x16 有效带宽约 **$22.4\text{ GB/s}$**（仅为 4090 显存带宽 $\sim 1000\text{ GB/s}$ 的 1/45）；128 行查表共需搬运 **$38.9\text{ MB}$** |
| **多卡 All-Reduce 被旁路任务撞上** | 4 张 A100 跑 7B（TP=4）时，若旁路训练每步与 vLLM 的 Custom All-Reduce 重叠约 **$193\text{–}205\ \mu\text{s}$**，吞吐只有原生的 **87.07%**（同设置单卡为 93.79%） | 旁路训练与 All-Reduce 争抢带宽和 SM，拖慢全卡同步 |

围绕这三个瓶颈，`pllm` 在底层替你做了三件事：
1. **保住时间线**：稳态 decode 每步 0 次主机同步，把前向改写和旁路训练录进 CUDA Graph，并把每步约 **$700\ \mu\text{s}$** 的 CPU 簿记藏进 GPU 执行阴影里。
2. **省下显存容量**：用分页状态池、按角色自动选精度和 CPU 静态代码追踪，把显存留给 KV cache。
3. **少搬字节、聪明重叠**：能合并读权重的绝不读两遍；必须搬运或训练的，只在 GPU 对应硬件资源空闲时塞进旁路 stream 重叠执行。

---

## 2. 先看计划：在 CPU 上用 `pllm.explain` 算清账单

`pllm.explain(workload, config)` 是写完 workload 后最重要的性能工具。它只读模型的 `config.json` 并在 CPU 上用 meta 张量做形状与生命周期追踪，**不加载权重、不初始化 CUDA、不占 GPU**（首次调用约 $2.9\text{ s}$，同进程内再次调用 $< 0.1\text{ s}$），就能直接告诉你：模型会不会退回 eager、每个方法排在哪条流、方法里有没有主机同步，以及 `pllm` 要预留多少字节显存。

### 代码示例

下面定义了一个在线蒸馏 + 引导采样的 workload，并在 CPU 上打印它的执行计划：

```python
from dataclasses import dataclass
import torch
from torch import nn
import pllm


@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        module=lambda m: nn.Sequential(
            nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size)
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    @pllm.guide_sampling(beta=pllm.Attr("beta"))
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=512,
    max_num_seqs=128,
    max_num_batched_tokens=8192,
)
print(pllm.explain(Distill(beta=0.5), config))
```

输出（节选）：

```text
forward  CUDA graphs: no effect needs eager
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0)
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
tap      model.layers.0[0], decode: 460800 bytes shared by loss, predict
state  scope  roots  bytes/root  reserved bytes [exact]
distiller  request  514  129104  66359516  field 0: float32, field 1: float32, ...
memory   peak 982708000 bytes funded by pllm [per-rank tensor reservations; excludes GPU-dependent BLAS workspace]
         below 982708000 bytes tasks run inline; below 932657440 bytes planning fails
host     0 syncs in traced workload math; exports: none
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

### 怎么从输出里看性能与显存

对照 [《写代码时就看到成本》](explain.md) 中的输出结构，重点盯住这五类行：

* **`forward`**：如果是 `CUDA graphs: no effect needs eager`，说明模型主干保住了图模式；如果出现 `eager: ...`，后面会直接点名是哪个声明导致了退回 eager。
* **`member`**：看每个方法的 `anchor / lane / graph`：
  * `graph` 为 `captured` 说明该方法本身也被录进了 CUDA Graph。
  * `lane` 为 `side` 说明它在旁路 CUDA stream 上运行，会与后续的模型前向重叠；`main` 说明在主 stream 上运行。
  * `anchor` 说明触发时机：`request.prepare` 在一步最开头（模型前向之前），`model.execute` 在模型前向时，`state.publish` 在前向结束、状态发布新版本时，`token.commit` 在这一步的 token 确定之后。
* **`tap`**：从模型层取出的激活缓冲大小。如果多个方法读同一层同一阶段（如 `loss` 和 `predict` 都读 `pllm.hidden(0)`），`pllm` 只建一份缓冲供它们共享（`460800 bytes shared by loss, predict`）。
* **`state` 与 `memory`**：
  * `state` 显示每个状态预留了多少份（`roots`）、每份多少字节、总预留多少字节，以及每个字段解析出的 `dtype`。
  * `memory` 显示 `pllm` 为任务计算预留的峰值字节数，以及两个关键阈值：`below X bytes tasks run inline; below Y bytes planning fails`（预算低于 X 时旁路任务失去重叠、退回主 stream 内联执行；低于 Y 时无法运行）。
* **`host`**：正常应显示 `0 syncs in traced workload math`。如果你在方法里写了 `.item()` 一类会触发 GPU 到 CPU 同步的操作，`explain` 根本不会走到这一行，而是在形状追踪时直接报错（见 [3.1 节](#31-零主机同步为什么一行-item-会拖垮吞吐)）。

> **实用建议与边界**
> * `pllm.explain` 不包含模型权重、vLLM 自身的激活和 KV cache 占用，也不包含与具体 GPU 相关的 BLAS 工作区大小（因此受影响的峰值是下界）。能不能装进 GPU 要到 `make_llm` 时才确定。
> * 作用在模型层内部或注意力里的计算（`pllm.model.ActivationPatch`、`@pllm.custom_residual`、`@pllm.custom_attention`）以及自定义算子，在 CPU 上无法追踪同步，`host` 行显示为 `unverified syncs`，需要上 GPU 跑 [`pllm.testing.check_on_gpu`](validation.md) 验证。

---

## 3. 保住时间线：零同步、CUDA Graph 与旁路重叠

在 4090 上跑 0.5B 模型，一步前向只有 $3.30\text{ ms}$。想在加了自定义算法后依然贴近原生吞吐，核心就是不让 GPU 停下来等 CPU，也不让自定义小算子打碎 CUDA Graph。

### 3.1 零主机同步：为什么一行 `.item()` 会拖垮吞吐？

在异步执行模型下，CPU 总是领先 GPU 一步准备元数据。`pllm` 每步的 CPU 簿记总计约 **$700\ \mu\text{s}$**，所需的槽位、掩码等元数据通过一张锁页 `int64` 表一次上传（H2D 约 **$6.24\ \mu\text{s}$**），完全藏在 $3.30\text{ ms}$ 的 GPU 前向阴影里，造成的步间 GPU 空闲仅约 **$4\text{–}6\ \mu\text{s}$**。

一旦在方法里写下 `.item()`、`.tolist()`、`.cpu()`、`print(tensor)` 或 `if tensor.any():`，CPU 就必须停下来等 GPU 把前面的所有算子跑完，流水线被彻底排空；而方法又运行在 CUDA Graph 和旁路 stream 里，这样的代码根本无法录制。所以 `pllm` 不让它跑起来：规划时用 meta 张量追踪你的方法，遇到这类操作直接报错，`pllm.explain` 和 `make_llm` 都过不去。

```python
@dataclass(frozen=True)
class BadBias(pllm.Workload):
    strength: float = 1.0
    bias = pllm.InstanceState({"b": lambda m: (m.vocab_size,)}, imported=True)

    # ❌ 错误写法：按张量的值走 Python 分支，需要 GPU -> CPU 同步
    @pllm.add_logits()
    def add(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        b: torch.Tensor = pllm.parameter("bias", "b"),
    ) -> torch.Tensor:
        if h.float().norm() > 10.0:  # explain 报错：Tensor.item() cannot be called on meta tensors
            return self.strength * b.float().expand(h.shape[0], -1)
        return torch.zeros(h.shape[0], b.shape[0], device=h.device)


@dataclass(frozen=True)
class GoodBias(pllm.Workload):
    strength: float = 1.0
    bias = pllm.InstanceState({"b": lambda m: (m.vocab_size,)}, imported=True)

    # ✅ 正确写法：用张量掩码和 torch.where 在 GPU 上逐行完成分支
    @pllm.add_logits()
    def add(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        b: torch.Tensor = pllm.parameter("bias", "b"),
    ) -> torch.Tensor:
        large = h.float().norm(dim=-1, keepdim=True) > 10.0  # [rows, 1]
        return torch.where(large, self.strength * b.float(), 0.0)  # [rows, vocab]
```

`pllm.explain(BadBias(), config)` 抛出 `RuntimeError: Tensor.item() cannot be called on meta tensors`，并附一条提示：`methods run inside CUDA graphs: use tensor operations such as torch.where, not .item() or Python branches on tensor values`。`.cpu()`、`torch.nonzero` 等同样在这一步被拦下。`GoodBias` 则正常规划，`host` 行为 `0 syncs`。注意改写后的语义也更对：原写法用整个 batch 的范数做一次判断，会让一个请求的结果受同批其他请求影响；改写后每一行只看自己。

**机制与验证**：
* `pllm` 将“稳态 decode 每步 0 次阻塞式主机同步”作为硬契约。在 RTX 4090 的 128 并发剖析中，`pllm` 产生的 Event/Stream 主机等待为 **0**，decode 窗口内 `cudaDeviceSynchronize` 空闲时间为 **0**。
* 遇到自适应深度分桶、计数 scan 完成判断等必须动态跳转的内部场景，`pllm` 使用 CUDA Graph 内的设备端条件节点（CUDA IF nodes，需 CUDA 12.4+ 驱动路径）直接在 GPU 上跳转，不把条件拷回 CPU。
* `explain` 只能追踪普通方法里的数学。模型层内和注意力里的计算要用 `pllm.testing.check_on_gpu(workload, config)` 验证：它在 `torch.cuda.set_sync_debug_mode("error")` 下实跑至少 8 个稳态 decode 步，任何隐式同步都会立即报错。

### 3.2 全链路 CUDA Graph：把训练、残差和自定义算子都录进图

CUDA Graph 会把一串 GPU kernel 的启动参数预先录制成一张图，之后每步只用一次极轻量的回放（replay）发射整串计算。`pllm` 在三个层面维持图模式：

1. **任务与在线训练整包捕获（`StepCapture`）**：在线训练（如 `@pllm.train`）包含前向、loss、掩码归约、`backward()`、写状态和分页 `AdamW` 等几十个小 kernel。`pllm` 按最大行数和状态容量把它们录进同一张图；输入使用固定缓冲，未满 batch 的空行指向不会污染真实状态的哑槽位（sink state）。因此即便每步活跃请求集合在变，也**不需要重录图**：稳态下每个 decode 步只回放 1 次训练图，0 次重捕获。在一次 128 并发的对比测试中，训练进图让每步从 $6.73\text{ ms}$ 降到 $5.25\text{ ms}$，吞吐提高约 **25%**。
2. **模型层内残差改写进图**：`@pllm.custom_residual` 会被编译为具名算子并直接录进 vLLM 的模型图里（详见 [5.4 节](#54-模型层内残差改写)）。
3. **自定义算子进图**：用 `torch.library.custom_op` 注册并附上 `register_fake` 的 Triton 或 PyTorch 算子，只要直接调用该 op 对象（不要调 `torch.ops.<namespace>`），就能被捕进图与旁路流（见[教程 09](../tutorials/09_custom_kernels.ipynb)）。在 RTX 4090（128 并发）下，自定义 Triton 算子吞吐达原生 vLLM 的 **95%**（$37,518\text{ tok/s}$），PyTorch 自定义算子达 **96%**（$37,976\text{ tok/s}$）。

**如何控制与检查图模式**：

```python
# 默认 cuda_graphs="auto"：能进图就进图，不能进图时退回 eager，
# 原因写在 explain 的 forward 行和运行时的 llm.diagnostics().eager_reason 里
config_auto = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", cuda_graphs="auto")

# cuda_graphs=True：workload 含有任何会导致 eager 的特性时，explain 和 make_llm 直接报错
config_strict = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", cuda_graphs=True)
```

`ActivationPatch` 通过 `custom_residual` 机制在 CUDA Graph 内执行，支持 `cuda_graphs=True`。

哪些特性会让模型前向退回 eager（或替换整图）？
* 共享可变 `InstanceState` 的 `pllm.model.Adapter`：退回 eager。
* `@pllm.custom_forward`：vLLM 的整模型图会被替换为 `pllm` 按层切片捕获的分段图（`eager_reason` 会注明 `@pllm.custom_forward`）。

### 3.3 旁路 Stream 重叠与 `publish_lag`：什么时候真能提速？

当你在 `pllm.explain` 中看到某个任务（比如 `@pllm.train`）的 lane 是 `side` 时，意味着它在旁路 CUDA stream 上执行。但你需要建立一个关键直觉：**旁路 stream 并不是多了一张显卡。同一张 GPU 的 SM 和显存带宽是固定的——只有当主 stream 没有把某项硬件资源吃满时，旁路任务塞进来才叫“免费重叠”；如果主 stream 已经把带宽吃满了，旁路任务挤进来只是在抢同一块蛋糕。**

#### 为什么默认 `publish_lag=1` 能提速？

如果要求第 $n$ 步采完样立刻算完梯度、第 $n+1$ 步前向开头必须读到最新权重（即 `publish_lag=0`），那么每步前向都必须干等反向传播和 AdamW 跑完。

`pllm` 默认采用 `publish_lag=1`：
* 第 $n$ 步 token 确定后，经过极短的拷贝栅栏，训练任务立刻放到 `side` stream 上异步执行；
* 第 $n+1$ 步前向直接读取第 $n-1$ 步的旧版本状态，不等待第 $n$ 步的训练；
* 第 $n$ 步算出的新权重在第 $n+2$ 步生效。

这样，旁路训练就能填进主干流在步间、小算子和非满载区段的空隙里。在 RTX 4090（0.5B，128 并发）测试中：
* **只训练不引导（$\beta=0$）**：`publish_lag=1` 达到原生 vLLM 吞吐的 **94.3%**（`publish_lag=0` 为 **91.4%**）。
* **训练 + 引导采样（$\beta=0.1$）**：`publish_lag=1` 达到原生的 **84.1%**（1,245 个旁路 kernel 中有 780 个与主干重叠；`publish_lag=0` 为 **73.7%**）。

如果你的算法在数学上严格要求下一步立即读到更新，可以通过 `ExecutionOptions(publish_lag=0)` 显式关闭滞后，并承担串行等待的代价：

```python
strict_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=128,
    max_num_batched_tokens=8192,
    execution=pllm.ExecutionOptions(publish_lag=0),
)
print(pllm.explain(Distill(), strict_config))
```

此时 `explain` 中 `loss` 的 lane 从 `side` 变为 `main`，`timing` 行写着 `publish_lag=0 (you set): updates at step n become readable at n + 1; updates serialize before the next forward`；状态的 `roots` 也随之变少（见 [4.1 节](#41-分页状态池与版本槽位公式roots)）。

> **实用建议与避坑**
> * **不要对 workload 方法手动套 `@torch.compile`**：`pllm` 已经在内部需要的地方（如词表重加权、只读查表的整图融合）使用了编译，并依靠 `StepCapture` 消除发射开销。在 `ESamp`（$\beta=0$）上测试过直接编译手写的 loss 前向与反向，吞吐（92.77%）与不编译相比没有可测收益。
> * **重叠不是越多越好**：`ESamp`（$\beta=0$）即使 CPU 几乎零气泡、开启 `publish_lag=1`，吞吐约为原生的 92%–94%。剩下的差距主要来自 GPU 上实打实的训练图回放（每步约 $100\text{–}140\ \mu\text{s}$）与主干流共用硬件，而不是 CPU 调度。强行提高重叠度（去掉采样器前的等待，让训练挤进正在满负荷扫词表的采样阶段）在测试中反而让吞吐从 94.49% 降到 92.73%：训练没有消失，只是去和采样器抢显存带宽了。

#### 自测小问题（第 1–3 节）

1. **问题**：你在 `@pllm.train` 方法里加了一句 `assert not torch.isnan(loss).any()` 来排查数值问题，会发生什么？
   <details><summary>点击查看答案</summary>
   <code>assert</code> 要把张量转成 Python <code>bool</code>，这需要一次 GPU 到 CPU 的阻塞式同步。<code>pllm</code> 在规划阶段用 meta 张量追踪方法，这一行会直接报错（<code>RuntimeError: Tensor.item() cannot be called on meta tensors</code>，附带“用 <code>torch.where</code> 等张量操作”的提示），<code>pllm.explain</code> 和 <code>make_llm</code> 都过不去。想观察数值，用 <code>@pllm.export</code> 导出你关心的统计量（第 7 节），或在运行时查询不产生每步同步的 <code>llm.diagnostics().numeric_health</code>，看状态字段里有没有 NaN/Inf。
   </details>

2. **问题**：既然 `lane: side` 可以在后台流上跑，如果把一个每步要读写几百 MB 显存的重度任务塞进 `side` 流，能不能白嫖时间？
   <details><summary>点击查看答案</summary>
   不能。Decode 阶段主干模型本身就是显存带宽受限的。如果旁路任务也在大量读写显存，它会和主干的投影或采样器争抢同一份显存带宽，导致主干变慢。只有当旁路任务占用的资源与主干互补（例如只用少量 SM 等待慢速 PCIe，或者填补步间空隙）时，重叠才能带来净收益。
   </details>

---

## 4. 守住显存容量：分页状态、精度与预算控制

在 vLLM 中，显存分配的顺序是：**模型权重 + CUDA Graph 捕获池 + `pllm` 状态与任务工作区 $\rightarrow$ 剩下的才分给 KV Cache**。一个真实的例子：在 RTX 4090（24 GiB，`gpu_memory_utilization=0.85`）上跑 7B 模型，如果把模型的 CUDA Graph 一直录到 8192 个 token 的批量，捕获池占掉的显存会让可用 KV Cache 从 76,144 个 token 降到 3,872 个，吞吐只剩原生的约 21%。`pllm` 按 vLLM 的常规尺寸录制模型图，并让下面每一字节预留都可见、可控。

### 4.1 分页状态池与版本槽位公式（`roots`）

当你声明 `pllm.RequestState` 或 `pllm.InstanceState` 时，`pllm` 用分页张量状态池（PTTS）管理显存。当状态更新时，框架执行**字段级写时复制（COW）**：只为被修改的字段分配新物理页，未修改字段直接与旧版本共享物理槽位。

为什么在 `pllm.explain` 里看到的 `roots`（预留槽位数）比 `max_num_seqs` 大得多？
因为为了支持 `publish_lag` 跨步重叠，当旁路流正在写入新版本时，主干流还在读取旧版本，多个版本必须同时驻留在显存里。精确槽位公式如下（其中 `lineages` 为并发请求槽位数，即 `max_num_seqs`）：

* **普通可写 `RequestState`**：$\text{roots} = (3 + \text{publish\_lag}) \times \text{lineages} + 2$
  * 例如第 2 节的 `Distill`：`max_num_seqs=128`、默认 `publish_lag=1` 时，$\text{roots} = 4 \times 128 + 2 = \mathbf{514}$，每份 $129,104\text{ B}$，总预留 $66,359,516\text{ B} \approx \mathbf{63.3\text{ MiB}}$。改成 `publish_lag=0` 后为 $3 \times 128 + 2 = 386$ 份。
* **前向内精确更新（`@pllm.train_in_forward(visibility="exact")` decode）**：$\text{roots} = 2 \times \text{lineages} + 2$
* **`InstanceState`**：可写实例状态最多按单条 lineage 计算（默认 `publish_lag=1` 时 $\text{roots} = 6$）；只读（不可变）实例状态只需 **`1` 个 root**。

### 4.2 自动精度选择：什么时候用 BF16，什么时候必须用 FP32？

声明状态字段时，默认 `dtype="auto"`。`pllm` 会静态分析每个字段在 workload 中的读写角色，自动平衡显存占用与数值稳定性（规约、梯度累加与优化器内部计算始终保持 $\ge$ FP32）：

| 字段在方法中的角色 | `dtype="auto"` 自动选择的精度 | 原因与实测影响 |
| :--- | :--- | :--- |
| **被优化器更新的参数**<br>或 **“读出再写回”的累加器/跨块进位（carry）** | `torch.float32` | 防止微小更新被吞噬或长序列误差发散。在 32k tokens（chunk 64）递推测试中，carry 用 BF16 的相对误差高达 **$1.437 \times 10^{-1}$**，而用 FP32 仅为 **$3.692 \times 10^{-4}$**（低三个数量级） |
| **只读字段**（如静态引导向量、冻结参数）<br>或 **只写字段** | 跟随模型精度（如 `torch.bfloat16`） | 省下一半状态显存与读写带宽。在 4090（128 并发）上，自动精度让 ReFT 每请求状态从 $3.50\text{ KiB}$ 降到 **$1.75\text{ KiB}$**，FactorAdapter 从 $22.00\text{ KiB}$ 降到 **$11.00\text{ KiB}$**。反之，若把 $9.00\text{ MiB}$/请求的稠密状态强行设为 FP32，吞吐会从 $3,156\text{ tok/s}$ 掉到 **$2,724\text{ tok/s}$（86.3%）** |

如果你需要在只读字段上直接做仅支持 FP32 的运算，可以在 `pllm.Field` 中显式指定 `dtype=torch.float32`，或在计算时转换：

```python
@dataclass(frozen=True)
class Probe(pllm.Workload):
    @property
    def probe(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {
                "w": pllm.Field(lambda width, _: (width,), dtype=torch.float32),  # 显式 FP32
                "v": pllm.Field(lambda width, _: (width,)),  # 只读，自动跟随模型精度
            },
            imported=True,
        )

    @pllm.export(name="score", max_rows=8, max_cols=2)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        w: torch.Tensor = pllm.parameter("probe", "w"),
        v: torch.Tensor = pllm.parameter("probe", "v"),
    ) -> torch.Tensor:
        return torch.stack((h.float() @ w, h.float() @ v.float()), dim=-1)
```

`explain` 的 `state` 行会显示 `field 0: float32, field 1: bfloat16`。

### 4.3 工作区自动降级与四个显存预算旋钮

`pllm` 在绑定引擎时，会对你的真实方法代码（含 autograd 保存的张量、优化器和 scan）做生命周期追踪，算出精确的峰值工作区（例如 `ESamp` 的 `predict` 工作区仅需 **$27.8\text{ MB}$**，`OverRIDE` 训练工作区需 **$267\text{ MB}$**）。

你可以通过 `pllm.ExecutionOptions` 上的参数控制显存分配与过载保护：

```python
exec_opts = pllm.ExecutionOptions(
    state_page_budget_bytes=256 << 20,  # 状态池（PTTS）上限：256 MiB
    execution_peak_budget_bytes=1024 << 20,  # 执行峰值上限（含状态页、任务与捕获池）：1024 MiB
    cuda_graph_pool_budget_bytes=128 << 20,  # 每个任务/模型的 CUDA Graph 池预算：128 MiB
)
config = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", execution=exec_opts)
```

这四个旋钮的行为如下：
1. **`execution_peak_budget_bytes`（两级优雅降级）**：包含 PTTS 状态页、任务工作区与存活结果、anchor 缓冲和捕获池预留；不含模型权重与 KV 池。旁路重叠（`side` lane）需要同时保留两步缓冲，因此峰值更高。以 128 并发的 `ESamp`（$\beta=0.5$）为例，旁路重叠计划峰值约为 **$1,041\text{ MiB}$**，内联（inline）下限约为 **$973\text{ MiB}$**。
   * 如果你设置的 `execution_peak_budget_bytes` 介于两者之间，规划器**不会报错，而是自动把旁路任务退回主 stream 内联执行**（`explain` 里该任务的 lane 从 `side` 变为 `main`），用一点重叠吞吐换取显存装得下。
   * 如果低于内联下限，规划失败，`explain(...).budget_errors` 会写明峰值需要多少字节、预算只有多少，并建议调小 `max_num_batched_tokens`、调大 `execution_peak_budget_bytes`，或把只读大表声明为 `placement="host"`。
2. **`state_page_budget_bytes`**：状态总预留（`roots × bytes/root`）超过该值时，`explain` 把缺口写进 `budget_errors`（建议调小 `max_num_seqs` 或调大预算），`make_llm` 在绑定阶段抛出 `ValueError`。默认 `0` 表示用后端实际可用的显存。
3. **`cuda_graph_pool_budget_bytes`**：控制每个任务/模型的 CUDA Graph 私有内存池预算，不是所有池的合计上限。默认 `None` 按负载追踪任务预留（模型图默认池为 **$64\text{ MiB}$**，exact decode overlay 另按形状计入工作区；像状态化 `@pllm.custom_residual` 子层或 `@pllm.custom_forward` 可能需要调大到几百 MiB 乃至 $1\text{ GiB}$）。**捕获预算不足时会直接报错并给出所需字节数，绝不静默退回 eager。**
4. **`max_host_training_rows`**：只用于主机批训练（`llm.generate(..., batch=...)` 传入的 `WeightedCEBatch`、`CausalTrainBatch`），表示每个状态每步最多训练多少行，训练工作区按它规划；默认 `1`，批次超过时会报错提示调大。它不限制 decode 时 `@pllm.train` 在请求行上的训练。

> **实用建议：生命周期与 Prefix Caching**
> * **用 `with pllm.make_llm(...) as llm:` 管理生命周期**：退出 `with` 块时，框架会按序清理 vLLM、cuBLAS 工作区、CUDA Graph 池与编译缓存，归还显存。唯一的例外是使用了 `@pllm.train_in_forward(visibility="exact")` 捕获的引擎，关闭时暂不能归还全部显存，后续引擎请在新的子进程中启动。
> * **Prefix Caching 兼容性**：`ModelConfig(enable_prefix_caching=True)` 会自动把 workload 的命名空间混入块哈希，保证数学一致时才复用前缀；但它与 `@pllm.custom_residual` 和 `@pllm.custom_attention` 互斥（这两者要求关闭 prefix caching）。
> * **量化模型上的在线训练**：在 GPTQ/Marlin 量化底座上做在线训练时，`pllm` 使用瞬态反量化：反量化后的稠密权重只活过一次反向传播，用完即释放，**峰值显存不增加**（14B GPTQ 在 A100 上每个训练步 **$0.193\text{ s}$**；4090 上 14B 的 QKV 反量化仅 **$0.104\text{ ms}$**）。预先缓存反量化权重虽然每步略快（$0.176\text{ s}$），却要多占 $24.63\text{ GiB}$ 常驻显存。

---

## 5. 五大算法场景的开销与写法选择

不同算法改动的是解码循环的不同阶段。下面按五类最常见的推理时算法，给出推荐写法、开销拆解和避坑指南。

### 5.1 在线训练与引导采样

**它是什么、什么时候用**：每个请求拥有自己的小网络（如 `RequestState(module=..., optimizer=pllm.AdamW(...))`），每步用 `@pllm.train` 在旁路更新权重，并用 `@pllm.guide_sampling` 或 `@pllm.add_logits` 调整下一步的采样分布（典型算法：`ESamp`、`OverRIDE`，见[教程 03](../tutorials/03_decode_training.ipynb)）。读多层词表分布的 `DoLa` 一类方法也在这里一并讨论。

#### 代码示例

训练加引导的写法参考 [第 2 节的 `Distill` 示例](#2-先看计划在-cpu-上用-pllmexplain-算清账单)。当只需要在部分行上施加效果时，可以用 `rows=` 指定一个选择器方法：它读取标记输入、返回布尔 `[rows]` 掩码，只有选中的行应用效果，其余行照常采样：

```python
@dataclass(frozen=True)
class SelectiveBias(pllm.Workload):
    strength: float = 1.0
    bias = pllm.InstanceState({"b": lambda m: (m.vocab_size,)}, imported=True)

    def later_samples(self, sample: torch.Tensor = pllm.sample_index()) -> torch.Tensor:
        return sample > 0  # 每个 prompt 的第 0 个样本保持原样，留作对照

    @pllm.add_logits(rows="later_samples")
    def add(
        self,
        sample: torch.Tensor = pllm.sample_index(),
        b: torch.Tensor = pllm.parameter("bias", "b"),
    ) -> torch.Tensor:
        return self.strength * b.float().expand(sample.shape[0], -1)  # [rows, vocab]
```

偏置向量在调用时传入：`llm.generate(..., batch=pllm.io.StateImport((vector,)))`。

#### Sampler epilogue：把词表操作放进已有遍历

框架把工作负载的 logits 操作并入 LM Head 和采样阶段，统一称为 **sampler epilogue**。`@pllm.add_logits` 的多个贡献按声明顺序加入，每次仍按输入 logits 的 dtype 舍入；`rows=` 只影响选中行，再照常执行 vLLM 的惩罚、掩码、温度和过滤。MRv2 在需要处理 logits 的步骤中，把这些加法与已有的 FP32 采样输入拷贝合成一次遍历；跳过处理的步骤保留 LM Head 的 dtype，省去每个贡献的 dtype 转换、独立加法和行子集的全量拷贝；随后可继续组合 `@pllm.guide_sampling`。逐步路径不读取设备值到主机。请求 raw logprobs 时仍保留独立原始分布；MRv1 使用同一有序加法内核，但输入已经是 FP32 时没有可省的上游转换遍历。

`explain` 将这次共享拷贝的输入读取、FP32 写回和一个内核归到第一个解码加法，后续加法只计各自贡献的读取字节；行选择按最大全批大小估计。这是逻辑流量估计，不是延迟预测。

#### 默认有界融合引导：容量为什么是 4096？

`ExecutionOptions.guidance_capacity` 默认是 **4096**。在支持的 MRv2、TP=1、未量化且未缩放或变换的 LM Head 上，框架在 vLLM 完成过滤后只投影保留下来的 token，并把引导与抽样融合；不支持的 runner 或 LM Head 会自动选择精确的 dense 路径。`guidance_capacity=None` 可显式选择精确 dense，`pllm.explain` 会报告实际路径、回退原因以及相应的 FLOP/字节估计。

**容量应让常见行不进入 overflow 路径；较小的 K 反而更慢，因为 overflow 行还需要 top-K 选择。** 保留下来的有限 logits 不超过 K 时，内核直接压紧、投影、抽样，无需排序。超过 K 时才选出 logits 最高的 K 个 token 进行引导，其余 tail 保持未引导并参与抽样（相同 logit 按 token ID 决胜）。若上游跳过了全词表行的过滤，且 top-1 概率达到 `top_p < 1`，则使用真正的单 token nucleus。

同一轮 RTX 4090、Qwen2.5-0.5B、128 并发、256 生成 token、`beta=0.5`、CUDA Graph、FlashInfer sampler 的七轮测量中，各容量的吞吐中位数如下：

| guidance capacity | 吞吐（tok/s） |
|---|---:|
| 256 | 33,737 |
| 1024 | 34,069 |
| **4096（默认）** | **34,742** |
| `None`（精确 dense） | 32,624 |

四组实测语料共 652,800 行中，最大的真实 nucleus 为 **2,785**，因此 K=4096 的真实支持集全部走简单路径；相对 dense 引导的策略误差 p99.9 TV 约 **$6\times10^{-14}$**，共享 Gumbel 抽样一致率为 **100%**。这些是该工作负载的测量结果，并非其他模型、过滤参数或 beta 的精度保证；有界路径在 overflow 时只引导 top-K，尖峰全词表行也会使用上述 nucleus 规则。需要精确 dense 行为时设置 `guidance_capacity=None`。

Dense 路径仍需额外的全词表 LM Head 投影和重加权：旧版 RTX 4090 测量中，开启引导比 `beta=0` 每步多约 $401\ \mu\text{s}$，其中投影约 $118\ \mu\text{s}$、重加权约 $187\ \mu\text{s}$。这些 dense 成本不代表默认有界融合路径；有界路径的 `explain` 投影与流量数字是容量上界，不能把更小的上界直接当作更高的吞吐。

旁路训练本身也很省：在 `beta=0` 时每步 $177.6\ \mu\text{s}$ 的旁路 GPU 时间里，分页 `AdamW` 只占 **$29.3\ \mu\text{s}$**——它直接从源物理槽位读取参数和一阶/二阶矩并写回目标槽位，省去了对 5 个优化器状态字段做写时复制的整页搬运。

> **实用建议与限制**
> * **优先用 `@pllm.guide_sampling`，慎用多层 `pllm.layer_logits`**：`pllm.layer_logits`（如 `DoLa` 读取多个中间层的词表分布）走独立的 GEMM，每多读一层就要多读一遍 $272\text{ MB}$ 的 LM Head 权重。在 0.5B 模型（32 请求 $\times n=4$）上，`DoLa` 的吞吐仅为原生的 **25%**（见[教程 06](../tutorials/06_reading_inside.ipynb)）。
> * **模型越大，引导采样的相对代价越小**：在 0.5B 小模型上，词表投影和重加权占整步时间的 $\sim 10\%$；但在 7B 模型上，主干每步要读十几 GB 权重，同样的词表开销占比很小——在 RTX 4090 上跑 7B 模型时，`ESamp`（$\beta=0.5$）吞吐约为原生 vLLM 的 **94%**（$\beta=0$ 约 **97%**）。
> * **逐行训练默认 `reduction="mean"`**：只平均本步每个状态选中的行；`reduction="sum"` 则求和。两者都不会跨状态归约。宿主批次自己归约，非默认 `reduction=` 会在声明期报错。

---

### 5.2 外部大表与词表级查表

**它是什么、什么时候用**：算法需要根据历史输入 token（`pllm.input_tokens`）从一张巨大的预训练查找表（如几百 MiB 到几 GiB 的 n-gram 先验表或 Engram 式记忆表）中取出向量，经掩码和缩放后叠加到词表分布或残差流上（见[教程 11](../tutorials/11_lookup_memory.ipynb)）。

#### 代码示例：把大表放在显存（`device`）或主机内存（`host`）

```python
from dataclasses import dataclass
from typing import Literal

import torch
import pllm


@dataclass(frozen=True)
class LookupPrior(pllm.Workload):
    path: str  # safetensors 文件，含形状为 [buckets, vocab] 的张量 "bonus"
    buckets: int = 2048
    scale: float = 0.1
    placement: Literal["device", "host"] = "device"  # 显存紧张时改为 "host"

    @property
    def table(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"bonus": lambda m: (self.buckets, m.vocab_size)},
            pretrained=self.path,
            placement=self.placement,
        )

    # 不带效果装饰器的查表方法：只读 token 历史和只读表
    def lookup(
        self,
        ids: torch.Tensor = pllm.input_tokens(history=1),  # [rows, 2]：当前 token 与前一个
        bonus: torch.Tensor = pllm.parameter("table", "bonus"),
    ) -> torch.Tensor:
        keys = (ids[:, 0] * 31 + ids[:, 1] + 1) % self.buckets
        return bonus[keys].float() * self.scale  # [rows, vocab]

    @pllm.add_logits()
    def prior(self, extra: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return extra
```

`explain` 中 `lookup` 一行为 `request.prepare / side / captured`：查表在一步最开头、在旁路 stream 上发射；`placement="host"` 时 `table` 一行显示 `host bytes`、`logical transfer bytes/row: 303872` 和 `decode bytes/step: 38895616`。

#### 原理与代价：如何把 $1.7\text{ ms}$ 的 PCIe 传输藏进前向里？

以 $2048 \times 151936$（BF16）的查找表为例，表本身占 **$594\text{ MiB}$**。128 个并发请求每步各查一行（每行 $303,872\text{ B}$），每步共需读取 **$38.9\text{ MB}$**。`pllm` 对这种查表做了两层优化：

1. **表在显存（`placement="device"`）：Inductor 整图融合**
   对于只依赖 `pllm.input_tokens` 和不可变状态的不可微查表方法，`pllm` 在绑定引擎时会自动将其中的 gather、类型转换、掩码（`where`）和缩放编译融合为**单次表读取 kernel**，省去把 $38.9\text{ MB}$ 宽向量写回显存再读出来做逐元素运算的往返流量，并把它提前到一步最开头的 `request.prepare` 发射。在 4090 上，设备端查表吞吐达到原生 vLLM 的 **82%**（**$31,750\text{ tok/s}$**，约 $4.03\text{ ms}$/步）。融合可能改变舍入；需要严格位级一致时用 `ExecutionOptions(allow_numerical_transforms=False)` 关闭。
2. **表在主机内存（`placement="host"`）：受限 SM 的旁路 UVA 拉取**
   把 $594\text{ MiB}$ 的表改为 `placement="host"` 后，`pllm` 在 GPU 上预留的显存峰值从 **$702\text{ MiB}$ 直降到 $109\text{ MiB}$**（$1\text{ GiB}$ 的 GPU 预算下也能挂载 $3\text{ GiB}$ 的主机表）。
   但每步 $38.9\text{ MB}$ 跨过 PCIe 4.0 x16（$\sim 22.4\text{ GB/s}$）需要实打实的 **$1.7\text{ ms}$**。如果直接用普通 GPU kernel 去主机内存拉数据，它会启动 37,984 个线程块占满 4090 的全部 128 个 SM——虽然这些线程块只是在等慢速的 PCIe，但主干流的模型前向根本排不上队：$1,753.7\ \mu\text{s}$ 的读表只有 $26.2\ \mu\text{s}$ 与主干重叠。
   **`pllm` 的做法**：跑满 PCIe 根本不需要多少 SM。`pllm` 拦截对主机表的按行读取，改用自定义常驻 Triton 算子，**把线程块数量限制在 GPU SM 总数的四分之一**（4090 上只用 32 个线程块），并在一步最开头的 `request.prepare / side` 发射。这样，1/4 的 SM 在后台等 PCIe，剩下 **3/4 的 SM 空出来跑主干模型前向**，剖析中读表时间的 **99.14%** 与主干 kernel 重叠。

| 配置（RTX 4090，0.5B，128 并发，594 MiB 表） | 吞吐（tok/s） | 相对原生 vLLM 比例 | 单步耗时 | `pllm` 显存峰值 |
| :--- | :--- | :--- | :--- | :--- |
| 原生 vLLM（无查表） | $38,821$ | 100% | $3.30\text{ ms}$ | — |
| 查表：`placement="device"` | $31,750$ | **82%** | $4.03\text{ ms}$ | $702\text{ MiB}$ |
| 查表：`placement="host"` | $30,174$ | **78%** | $4.24\text{ ms}$ | **$109\text{ MiB}$** |

可以看到，尽管每步跨 PCIe 搬运了 $38.9\text{ MB}$（物理耗时 $1.7\text{ ms}$），**表放 `host` 比放 `device` 每步只慢了约 $0.21\text{ ms}$**，却省下了近 $600\text{ MiB}$ 显存。

> **实用建议与限制**
> * **怎么让主机查表成功藏进前向里？** 把查表写成**不带效果装饰器的独立方法**，只读取 `pllm.input_tokens` 和不可变状态，再让 `@pllm.add_logits` 或 `@pllm.custom_residual` 通过 `pllm.result("lookup")` 读它的结果。这样规划器才能把它排到 `request.prepare / side`。如果把查表直接写在 `@pllm.add_logits` 方法体里，它会落在 `model.execute / main`；如果查表方法还读了 `pllm.hidden` 或 `pllm.final_hidden`，它要等前向结束（`state.publish / main`）。两种情况传输都无法与前向重叠。
> * **怎么估算 PCIe 传输能不能藏住？** 看 `pllm.explain` 输出的 `logical transfer bytes/row`（或 `decode bytes/step`），用 $\frac{\text{并发数} \times \text{bytes/row}}{22\text{ GB/s}}$ 估算 PCIe 耗时。只要它明显短于一次模型前向（如 0.5B 的 $3.3\text{ ms}$），绝大部分传输都能被藏住。
> * **`placement="host"` 支持的操作**：仅支持只读 `pretrained` 表的按行读取，包括直接索引 `table[idx]`（含切片视图如 `table[:, ::2][idx]`）、`torch.index_select`、`torch.nn.functional.embedding` 和 `torch.vmap`，其他操作在规划时被拒绝。模型层内的方法（如 `@pllm.custom_residual`）不能直接读主机表，要通过 `pllm.result` 读取独立查表方法的结果。原因是：模型层内的方法在模型图里、主 stream 上某一层的中间执行，在那里读主机内存，PCIe 搬运会直接卡住这一层之后的前向，藏不掉；拆成只依赖 token 和只读表的独立方法后，规划器才能把它提前到步首的旁路 stream，用少量 SM 读、与前向重叠（见上一条）。同一个查表结果也可以同时交给多个效果方法，而只搬一次。`pllm` 不会为了装下预算自动把表挪到主机内存，这个取舍由你显式声明。

---

### 5.3 自定义注意力与 KV Cache 瘦身

**它是什么、什么时候用**：使用 `@pllm.custom_attention` 改变某些层存储 KV 的物理布局（滑动窗口、保留整个 prompt 加最近窗口、跨层复用低层 KV），或者改变读取 KV 的方式（如 Quest 风格的按页打分 Top-K 稀疏读取，见[教程 10](../tutorials/10_custom_attention.ipynb)）。

#### 代码示例：静态 KV 布局瘦身 vs 动态 Top-K 读取

方法的前两个参数是这一层的 query 和分页 KV（不带默认值）；`kv.attend(q)` 返回注意力输出和 FP32 log-sum-exp，取 `[0]` 即可。

```python
from dataclasses import dataclass
import torch
import pllm


# 方案 A：静态 KV 几何布局（既省显存容量，又省读取带宽，且走原生 FA2 内核）
@dataclass(frozen=True)
class LocalGlobal(pllm.Workload):
    """偶数层只保留最近 256 个 token 的 KV；加 sink=True 则保留整个 prompt 加最近 256 个生成 token。"""

    @pllm.custom_attention(layers=tuple(range(0, 24, 2)), keep="window", window=256)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


# 方案 B：跨层共享 KV（奇数层不存 KV，直接读下面一层的 KV，省掉一半 KV 显存）
@dataclass(frozen=True)
class CrossLayerShared(pllm.Workload):
    @pllm.custom_attention(layers=tuple(range(1, 24, 2)), keep="none", source=1)
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        return kv.attend(q)[0]


# 方案 C：动态 Top-K 稀疏读取（省逻辑读取字节，但不省显存容量）
@dataclass(frozen=True)
class TopPages(pllm.Workload):
    pages: int = 8  # 每个头读几页

    @pllm.custom_attention(layers=(12,))
    def read(self, q: torch.Tensor, kv: pllm.model.PagedKV) -> torch.Tensor:
        block = kv.key_cache.shape[1]
        keys = kv.key_cache[kv.block_table[kv.requests].long()].float()  # [行, 页, 页内位置, kv 头, 头维]
        rows, heads, dim = q.shape
        g = q.float().reshape(rows, keys.shape[3], heads // keys.shape[3], dim)  # GQA 分组
        scores = torch.einsum("rkgd,rpkd->rkgp", g, keys.amax(2)).reshape(rows, heads, -1)  # 粗略的每页分数
        page = torch.arange(scores.shape[-1], device=q.device)
        scores = scores.masked_fill(page * block > kv.positions[:, None, None], -float("inf"))  # 还没写到的页
        chosen = scores.topk(min(self.pages, scores.shape[-1]), dim=-1).indices  # [行, 头, pages]
        tokens = (chosen[..., None] * block + torch.arange(block, device=q.device)).flatten(2)
        return kv.attend(q, topk=tokens)[0]  # 每个头只读自己选中的位置
```

方案 C 用的是简化打分；按 Quest 原文用每页 key 的最小/最大值求上界的完整写法见教程 10 的 `QuestRead`。

#### 原理与代价：分清“真省显存容量”和“只省逻辑读取”

写 `@pllm.custom_attention` 时，最重要的一条分界线是你的写法能否映射到底层 vLLM 的物理块管理器和原生 FlashAttention-2 / Triton 内核：

| 特性维度 | 静态几何布局（`keep="window"` / `sink=True` / `keep="none", source=k`） | 动态稀疏读取（`kv.attend(q, topk=...)`） |
| :--- | :--- | :--- |
| **省 KV 显存容量吗？** | **真省容量**：直接映射为 vLLM 原生 `SlidingWindowSpec`、`SinkSpec` 和跨层共享组，滑出窗口的物理块会被真正回收给池子 | **不省容量**：所有历史 token 的 KV 仍必须保留在显存中供打分和挑选 |
| **省读取带宽吗？** | **真省带宽**：直接调用原生 FA2 和 Triton `unified_attention` | **省逻辑读取字节**（`explain` 中教程 10 的 `QuestRead` 每 token 最多读 $458,752\text{ B}$，全量为 $1,048,576\text{ B}$），但打分和 gather 走 PyTorch 算子 |
| **实测单请求 KV 显存上限**<br>（0.5B，教程 10：`max_model_len=2048`，256 token 分块 prefill） | 全量 **$24.0\text{ MiB}$** $\rightarrow$ 偶数层窗口 256 降至 **$16.6\text{ MiB}$**；共享一半层降至 **$12.0\text{ MiB}$**（可同时容纳的 2048-token 请求从 358.8 个增至 717.6 个）。`sink=True` 保留整个 prompt，上限仍为 $24.0\text{ MiB}$，实际节省取决于生成长度 | 仍为 **$24.0\text{ MiB}$** |
| **实测吞吐**<br>（RTX 4090，0.5B，128 并发） | 原生读取（`NativeRead`）**97%**；偶数层窗口（`LocalGlobal`）**97%**；跨层共享（`CrossLayerShared`）**95%** | 教程 10 中 `QuestRead` 仅为原生的 **7%**（$2,727\text{ tok/s}$），因为 PyTorch 级打分与 gather 开销远大于省下的读取 |

> **实用建议与限制**
> * **布局改的是模型数学**：未经相应训练的模型直接换成窗口或跨层共享布局，回答质量可能明显变差（教程 10 中 `LocalGlobal` 和 `CrossLayerShared` 都答错了长文检索问题，`PromptPlusWindow` 答对）。先在自己的任务上验证质量，再看吞吐。
> * **不支持按注意力分数动态淘汰物理 KV 块（如 SnapKV / H2O）**：vLLM 对每个请求只维护单一的 `num_computed_tokens` 计数，位置编码与物理槽位均由它线性推导，且块表（block table）只能追加、不能在中间挖洞。如果你的目标是省 KV 显存，请改用静态 `keep="window"`（可加 `sink=True`）或 `keep="none", source=...`；如果只想少读，用 `kv.attend(q, topk=...)`。
> * **硬件与配置限制**：`@pllm.custom_attention` 要求 `tensor_parallel_size=1`、FlashAttention-2、非量化 KV 且关闭 prefix caching；静态布局还要求 MRv2 runner，且不能与 `pllm.PositionState` 或 `@pllm.custom_forward` 同时使用。
> * **不要设极小的滑动窗口**：窗口极小（如只有 3 个 token）时，FA2 处理窗口的耗时（$91.5\ \mu\text{s}$）反而比直接算全量注意力（$16.6\ \mu\text{s}$）慢 5 倍以上。
> * **复用模型自带的稀疏选择**：读取并复用 DSA 一类模型自带的 top-k 选择（`pllm.attention(layer, "topk")`）需要 vLLM 的 DSA 路径，要求 **SM90+（Hopper 架构，如 H100）**；在 RTX 4090（SM89）和 A100（SM80）上不可用，调用会报错说明尚未支持。

---

### 5.4 模型层内残差改写

**它是什么、什么时候用**：需要在 Transformer 的每一层或特定子层内部直接修改残差流、注意力 Value 或跨层连接（如 Value Residual、LAuReL、Hyper-Connections），同时不希望模型退回 eager 模式（见[教程 07](../tutorials/07_custom_residual.ipynb)）。

#### 代码示例

`site="mlp"` 时，方法的前两个参数是子层之前的残差流 `x` 和 MLP 分支输出 `y`（不带默认值），返回值替换原来的 `x + y`：

```python
from dataclasses import dataclass
import torch
import pllm


@dataclass(frozen=True)
class ScaleResidual(pllm.Workload):
    scale: float = 0.95  # 1.0 即原模型

    @pllm.custom_residual(site="mlp", layers=(8, 16))
    def dampen(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + self.scale * y
```

#### 原理与代价：比 `ActivationPatch` 快在哪里？

`pllm.model.ActivationPatch` 通过同一套 `custom_residual` 模型站点在 CUDA Graph 内修改层输出。
而 `@pllm.custom_residual` 会被 `pllm` 降级为具名算子 `pllm::model_site`，并用设备端索引读写私有状态页，使 vLLM 在录制整模型 CUDA Graph 时**把你的残差改写一并录进图里**。

在 RTX 4090（0.5B，8 条 prompt $\times n=4$，生成 256 tokens）下的实测吞吐占原生图模式比例：
* **Value Residual**：**85%**（同样的改写在 eager 下只有 34.4%）
* **Hyper-Connections（$n=4$）**：**65%–66.6%**（eager 下 19.2%）
* **带状态记忆子层**：在设置 `cuda_graph_pool_budget_bytes=1 << 30`（1 GiB）进图后为 **41%**（eager 下 11.6%）。

> **实用建议与限制**
> * `@pllm.custom_residual` 要求 `tensor_parallel_size=1` 并关闭 prefix caching。`explain` 中它的 `host` 行显示 `unverified`，CPU 上的 `pllm.testing.check` 也会跳过模型层内的执行；要用 `check_on_gpu` 验证。
> * 如果你的残差改写带有可读写状态（`state=` 的状态化子层），录制模型图时需要更多捕获池空间，请调大 `ExecutionOptions(cuda_graph_pool_budget_bytes=...)`，不够时报错信息会给出所需字节数。
> * 挂了 `custom_residual` 的模型图里，vLLM 录的是原始算子而不是 Inductor 融合后的算子，所以即使是恒等改写（`x + y`）也只有原生图模式的约 **66%**。只在真正需要改写的层上挂载（用 `layers=...` 指定），不要无故挂满所有层；评估改写本身的代价时，可以和 eager 模式的原生 vLLM 对比，或直接用 `check_on_gpu`（它会构建匹配的原生引擎）。

---

### 5.5 前向内训练、线性扫描与自适应深度

**它是什么、什么时候用**：需要在模型前向内部做快权重更新（`@pllm.train_in_forward`，见[教程 08](../tutorials/08_inplace_ttt.ipynb)）、用 `pllm.scan` 计算长序列线性递推（$S_t = A_t S_{t-1} + B_t$），或者用 `@pllm.custom_forward` 编排层循环与按 token 提前退出（`active=mask`，见[教程 12](../tutorials/12_adaptive_depth.ipynb)）。

#### 1) 线性递推 `pllm.scan`：自动并行前缀扫描

在 chunk 内做线性递推时，如果用 Python 循环串行步进，GPU 的 SM 根本填不满。用 `pllm.scan` 写递推后，`pllm` 的 Rust 规划器会自动识别形如 $S_t = A_t S_{t-1} + B_t$ 的仿射递推，在显存工作区预算允许且系数为 FP32 时，替换为**并行前缀扫描（Parallel Prefix Scan）**：
* **微基准**：单头 $64 \times 64$ 状态扫描从串行的 $1.439\text{ ms}$ 降到 **$0.219\text{ ms}$（6.57 倍）**；14 头 $64 \times 64$ 为 **1.85 倍**；完整状态转移从 $5.879\text{ ms}$ 降到 **$1.759\text{ ms}$（3.34 倍）**。
* **自动回退**：当状态维度很大（如 $d=896$）导致树形扫描不划算时，形状守卫会保留串行循环；CPU 上、部分扫描或需要求导的调用也保持串行。你也可以设 `ExecutionOptions(tuning="measured")` 让引擎在绑定时实测挑选最快实现，或用 `allow_numerical_transforms=False` 禁止并行重排。

#### 2) 前向内精确更新 `@pllm.train_in_forward(visibility="exact")`

默认情况下，`@pllm.train_in_forward` 使用 `visibility="forward_pinned"`（同一次前向的所有 chunk 读同一个起始状态）。对于 In-Place TTT 等要求“同一前向内后一个 chunk 必须读到前一个 chunk 更新后的快权重”的算法，需声明 `visibility="exact"`：
* **实测表现**：`pllm` 会把 `visibility="exact"` 的 decode 阶段捕获进 CUDA Graph（prefill 仍为 eager）。在 RTX 4090 上跑 In-Place TTT 4B 模型（4096 prompt + 256 生成），batch size = 1 达 **$61.61\text{ tok/s}$**（16k 上下文达 **$42.91\text{ tok/s}$**，均快于官方 HF 实现的 $54.8 / 34.9\text{ tok/s}$），batch size = 4 达 **$123.76\text{ tok/s}$**（官方实现只支持 batch size = 1）。

#### 3) 自适应深度跳层（`@pllm.custom_forward` + `active=mask`）：为什么只有“整批全跳”才省读权重？

在 `@pllm.custom_forward(tokens="decode")` 中，方法接收 `layers: pllm.model.Layers`；调用 `layers(start, stop, hidden=..., active=mask)` 运行一段层，`active` 为 `False` 的行跳过这一段：

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import pllm


@dataclass(frozen=True)
class SkipMiddle(pllm.Workload):
    threshold: float = 0.85  # 余弦相似度大于它就跳过第 16–19 层

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        h12 = layers(0, 12)[11]  # 第 0–11 层，所有 token 都跑
        h16 = layers(12, 16, hidden=h12)[15]  # 第 12–15 层
        moved = F.cosine_similarity(h16.float(), h12.float(), dim=-1).unsqueeze(-1) <= self.threshold
        h20 = layers(16, 20, hidden=h16, active=moved)[19]  # 只有 moved 为 True 的行计算
        return layers(20, hidden=h20)[-1]  # 第 20–23 层
```

**被跳过的层到底做了什么：状态传播。** 一行在某一段层上 `active=False` 时，它的残差流原样穿过这段层（等价于把这几层当作恒等映射）。但后面的 token 在每一层做 attention 时，仍然要读到这个位置在这一层的 K/V，所以 pllm 会为这行补写被跳过层的 K/V：拿它停下时的隐藏状态，经过每个被跳过层自己的输入 norm、K/V 投影和 RoPE 算出 K/V 写进 KV cache（第一次跳过时补写一次，约占一层计算量的 7%；之后再跳过同一层时保留已写入的值）。这就是 CALM 提出的“状态传播”：它给出的 K/V，正是“这些层输出等于输入”这个假设下该位置本应有的 K/V，误差只局限在被跳过的 token 自己的深层状态上。

为什么默认用状态传播，而且它能覆盖各种需求：
* **KV 的形状和语义都不变。** 每一层、每个位置都有 K/V，所以分页 KV cache、原生 FlashAttention、prefix cache、抢占后重算都照常工作，后面的 token 在任何一层都按正常方式做 attention。
* **与“怎么决定跳”无关。** 余弦相似度阈值、学出来的退出头、置信度、固定的跳层表、循环深度里每一圈是否停下，最后都归结为“这一行不跑这一段层”；早退（跳过剩下所有层）、跳过中间一段、循环模型提前停圈，都用同一个机制表达。
* **只依赖这一行自己。** 补写只用到这一行停下时的状态，不依赖同一 batch 里别的请求，所以一个请求的结果不会因为和谁一起调度而改变，也能进 CUDA graph、不需要 host 同步。

其他做法为什么不作默认：直接复制上一层的 K/V 不对（不同层的 K/V 投影在不同子空间里）；“留空洞”（被跳过的 token 不进这一层的 attention，Mixture-of-Depths 的做法）只适用于按这种方式训练过的模型，而且要求每层维护各自的 KV 长度，vLLM 的 KV 管理做不到；“推迟补算真实深层”（FREE 的做法）会让每个请求每步处理的 token 数变得不固定，和固定形状的 CUDA graph 冲突。

这里有一个极其重要的物理事实需要建立直觉：
* **为什么 128 个 token 里跳过 50% 并不会让这几层快一倍？**
  因为 decode 阶段的矩阵乘是**权重带宽受限**的（例如 7B 模型单层权重达 $466\text{ MB}$）。哪怕 128 行里只有 1 行 `active=True`，GPU 也必须把这一层几百 MB 的权重完整读进 SM 才能算这 1 行！被跳过的行只省下了它自己的注意力 KV 读取流量（且首次被跳过时仍需用原生算子补齐约占单层开销 7% 的 norm/QKV/RoPE，以维持后续 KV 连续）。
* **什么时候才能省下读权重的带宽？**
  `pllm` 使用 CUDA Graph 条件节点将活跃行压实（compact）到 2 的幂次桶里执行。**只有当整个 batch 在该层全部跳过（活跃行数为 0，落入 `bucket 0`）时，GPU 才会通过条件节点直接跳过整层算子，真正省下读权重的带宽。**
* **实测数字（RTX 4090，0.5B，8 条 prompt $\times n=4$，第 16–19 层可跳）**：
  * 从不跳过：原生图模式吞吐的 **93%**（对比原生 eager 仅 45%）
  * 约 14% token 跳过（阈值 0.85）：原生图吞吐的 **90%**
  * 整批全部跳过（落入 bucket 0）：原生图吞吐的 **102%**。
* **上下文更长、batch 更大时情况不同。** 上面说的是短上下文的 decode：时间主要花在读权重上。上下文很长时（例如几 k 到几十 k），每行要读自己全部历史的 K/V，这部分可能超过读一层权重的量；被跳过的行不做这一层的 attention，这部分读取会按跳过的行数成比例省下（原理如此，尚未实测）。batch 很大或在 prefill 那样受算力限制时，压紧活跃行直接省下矩阵乘的计算量。

> **实用建议与限制**
> * `active=` 仅支持 `@pllm.custom_forward(tokens="decode")`，跳层不会释放被跳层的 KV 显存。
> * **为什么 `active=` 只用于生成的 token？** 一个 token 跳过某些层时，pllm 用它“停下时”的隐藏状态补算这些层的 K/V。这些 K/V 只是近似值，后面的 token 读到的就是这份近似。对生成的 token，这份近似只影响同一请求接下来的几个 token，而且它跳层的前提是“已经算够了”（CALM、LayerSkip 等早退方法也只让生成的 token 退出）。prompt token 不一样：它不需要输出任何东西，唯一的作用就是给后面所有 token 提供 K/V；它的深层 K/V 被近似之后，影响的是整段回答，还会通过 prefix cache 和 n 个样本扩散出去。Ouro 论文的实测也是：prefill 阶段的 K/V 用近似值会掉 10 个以上的 GSM8K 点，decode 阶段却几乎不掉。另外，prompt 是整块并行计算的，被压紧后的行位置不连续，原生的 FlashAttention 无法直接计算它们，省下的矩阵乘大多会被额外的 attention 开销抵消。所以 prompt 默认按原模型完整深度运行，你不需要为它另写方法；只有当你要改变 prompt 的计算方式（例如固定的层循环）时，才写一个不带 `active=` 的 `tokens="prefill"` 方法，并优先用 `over="span"`（逐 token 的程序在 prefill 上会按位置串行执行）。
> * 使用 `@pllm.custom_forward` 时，同一个 workload 内不能再声明读取 `pllm.hidden` / `pllm.final_hidden` 的方法（请改用 `pllm.residual` 或 `pllm.layer_logits`），也不能用 `pllm.model.Adapter`。按层切片的图可能需要调大 `ExecutionOptions(cuda_graph_pool_budget_bytes=...)`（例如 0.5B 模型在 32 并发下捕获池约需 $748\text{ MiB}$，128 并发约需 $1,392\text{ MiB}$）；`explain` 不估算这部分，预算不够时报错信息会给出所需字节数。

---

## 6. 多卡张量并行（TP > 1）

当模型单卡装不下或需要更高带宽时，可以通过 `ModelConfig(tensor_parallel_size=...)` 开启多卡张量并行（需 vLLM 0.32.0 与 `runner="mrv2"`）。

### 原理与代价：为什么 `pllm` 选择“小状态按卡复制 + 流优先级错峰”？

在张量并行下，vLLM 每算完一层注意力或 MLP 都要做一次全卡同步（Custom All-Reduce）。如果：① `pllm` 自己也发起跨卡集合通信，或者 ② 旁路流上的在线训练刚好在某张卡上和 vLLM 的 All-Reduce 撞在一起抢带宽，全卡就会一起被卡住。

因此，`pllm` 在 TP 下采用两条机制：
1. **零 `pllm` 集合通信**：hidden tap、路由计算、`PositionState` 和请求级小状态在每个 TP rank 上各存一份完全相同的副本，各卡独立执行完全相同的确定性逻辑（实测各 TP 规模下与原生 vLLM 的 logits 均方根差为 0），`@pllm.export` 只在 rank 0 交付。
2. **流优先级与发布错峰**：将 TP 推理主 stream 设为高优先级（`-1`）、旁路训练 stream 设为普通优先级（`0`），并让自动状态更新在采样前写入私有状态、在 token 提交时发布，使**旁路训练与 vLLM Custom All-Reduce 的重叠时间降为 $0\ \mu\text{s}$/步**。

**实测表现（A100-SXM4-80GB，`Qwen/Qwen2.5-7B-Instruct`，`ESamp` $\beta=0$，128 并发）**：
* **TP = 1**：原生 vLLM 吞吐的 **95.84%**
* **TP = 2**：原生 vLLM 吞吐的 **95.63%**
* **TP = 4**：原生 vLLM 吞吐的 **97.60%**

> **实用建议与限制**
> * **TP 不会减少单卡上 `pllm` 状态的显存占用**：因为状态是按 rank 复制而非分片的（`pllm.explain` 会在输出中提示 `reservations are per rank`）。TP 省下的是底座模型权重在单卡上的占用。
> * **TP > 1 不支持的特性**：以下能力在 `tensor_parallel_size > 1` 时会被直接拒绝：`@pllm.custom_residual`、`@pllm.custom_attention`、除按分片的 dense / low-rank 改写（只由 `@pllm.train_in_forward` 在本卡分片上更新）之外的权重改写、经 `pllm.causal_model` 的整模型训练、状态投喂（`feed`），以及 `pllm.testing.check_on_gpu`（请先在 `tensor_parallel_size=1` 下跑通 `check_on_gpu`）。

---

## 7. 线上观测与故障排查速查表

### 7.1 无阻塞观测：`@pllm.export` 与 `llm.diagnostics()`

如果你想在运行时把激活或统计量导回 CPU 观察，**切勿在方法里写 `.cpu()` 或 `print()`**，请使用 `@pllm.export`：

```python
@dataclass(frozen=True)
class Monitor(pllm.Workload):
    # 每 8 步导出一次，最多 4 行 × 16 列（一行须装进一个 32 KiB 事件：max_cols * 8 <= 32768 B）
    @pllm.export(name="head", every=8, max_rows=4, max_cols=16)
    def peek(self, h: torch.Tensor = pllm.final_hidden()) -> torch.Tensor:
        return h[:, :16].float()

    def on_export(self, event: pllm.io.TraceEvent) -> None:
        print(event.step_id, event.tensor().shape)  # 在后台异步回调，不阻塞解码
```

* **`@pllm.export` 的零阻塞保证**：它使用 `non_blocking=True` 拷入有界锁页缓冲（队列上限 256 个事件，每个 $\le 32\text{ KiB}$），消费跟不上、队列满时**默认直接丢弃（`overflow="drop"`）而不阻塞 GPU 或 CPU**（丢弃计数见 `llm.trace_snapshot().export_dropped`；每步超出 `max_rows` 的行记在 `event.dropped_rows`）。
* **`llm.diagnostics()` 的有界归约**：查询 `numeric_health` 时，框架在 GPU 端分块归约状态快照，仅在调用时拷回标量（对 594 MiB 大表做健康检查的额外显存仅 **$19.0\text{ MiB}$**），在稳态 decode 期间产生 **0 次每步同步**。

### 7.2 症状排查速查表

遇到性能不达预期或显存报错时，按**先看 `pllm.explain`（静态） $\rightarrow$ 再调配置/改代码 $\rightarrow$ 最后查 `llm.diagnostics()`（运行时）**的顺序对照下表：

| 你遇到的症状或问题 | 第一步：看 `pllm.explain(workload, config)` | 第二步：可调的配置开关或代码改动 | 第三步：运行时查 `llm.diagnostics()` / `llm.trace_snapshot()` |
| :--- | :--- | :--- | :--- |
| **启动时报显存不足，或可用 KV 并发太低** | 看 `state` 行（`roots`、`bytes/root`、`dtype`）；看 `memory peak`；看 `budget_errors` | 大只读表改 `placement="host"`；注意力改静态 `keep="window"` 或 `source=k`；只读字段让 `dtype="auto"` 自动选 BF16；调 `state_page_budget_bytes` / `execution_peak_budget_bytes` / `cuda_graph_pool_budget_bytes`；主机批训练时调小 `max_host_training_rows` | 查 `state_usage`（对比逻辑存活字节与物理分配字节，以及 `admission_capacity_failures`）、`kv_cache_blocks`、`kv_cache_max_concurrency` |
| **步时明显变慢，怀疑退回了 eager 或有 CPU 同步** | 看 `forward` 行是否显示 `eager: ...`；看 `member` 行是否 `captured`；看 `host` 行是 `0 syncs` 还是 `unverified` | 方法里的 `.item()` / `if tensor:` 改用 `torch.where`；设 `ModelConfig(cuda_graphs=True)` 让任何 eager 原因直接报错；跑 `testing.check_on_gpu` 抓模型层内的隐式同步 | 查 `eager_reason`（为空字符串表示运行在整图模式） |
| **加了引导采样或额外计算后吞吐下降明显** | 看 `member` 行在哪个阶段运行；看每个方法的 `projection FLOP`、`sampler ... bytes`；检查是否声明了多层 `pllm.layer_logits` | 用 `@pllm.guide_sampling` 替代独立读多层 `layer_logits`（共用 LM Head GEMM）；用 `rows=` 选择器缩小 `@pllm.add_logits` 或 `@pllm.guide_sampling` 的作用行数 | 查 `trace_snapshot()` 中的任务耗时 |
| **大表设了 `placement="host"` 但吞吐依然很低** | 看查表方法的 `anchor / lane` 是否为 `request.prepare / side`；看 `logical transfer bytes/row` | 把查表写成只读 `pllm.input_tokens` 和不可变状态的独立方法，经 `pllm.result` 交给效果方法（不要混读 `pllm.hidden`，否则无法提前到步首与前向重叠）；检查每步总传输字节除以 $22\text{ GB/s}$ 是否已经超过了模型单步前向时间 | — |
| **旁路训练（`side`）没有和前向重叠** | 看 `member` 的 lane 是 `side` 还是退回了 `main`；看 `timing` 行的 `publish_lag` | 保持默认 `publish_lag=1`；如果预算介于 `memory` 行的两个阈值之间导致退回内联，调大 `ExecutionOptions(execution_peak_budget_bytes=...)` | 查 `trace_snapshot()` 中主干流与旁路流的耗时分布 |
| **长序列递推或在线训练出现数值发散 / NaN** | 看 `state` 行各字段解析出的 `dtype`（累加器/carry 是否为 `float32`） | 在 `pllm.Field(..., dtype=torch.float32)` 显式指定 FP32；检查 `pllm.scan` 系数是否为 FP32 | 查 `numeric_health` |

#### 自测小问题（第 4–7 节）

1. **问题**：你写了一个查表方法，既读取了 `pllm.input_tokens()` 去 `placement="host"` 的大表里查向量，又在同一个方法里读取了 `pllm.final_hidden()` 做门控。为什么加上这个表之后每步突然慢了将近 $2\text{ ms}$？
   <details><summary>点击查看答案</summary>
   因为同一个方法里读取了 <code>pllm.final_hidden()</code>，这个方法就必须等到模型前向跑完之后才能启动（<code>explain</code> 中它的 anchor/lane 是 <code>state.publish / main</code>），无法在一步最开头的 <code>request.prepare / side</code> 发射。结果 1.7 ms 的 PCIe 传输完全暴露在模型前向之外。解决办法是把只依赖 <code>input_tokens</code> 的主机查表拆成独立的方法，用 <code>pllm.result</code> 把结果交给读 <code>final_hidden</code> 的效果方法，让规划器能把查表提前到步首与前向重叠。
   </details>

2. **问题**：在 0.5B 模型上，你用 `@pllm.custom_forward(tokens="decode")` 实现了早退（Early Exit），发现在 128 并发下有 30% 的 token 跳过了后 4 层，但端到端吞吐几乎没有提升。这是为什么？
   <details><summary>点击查看答案</summary>
   Decode 阶段的层计算是权重带宽受限的。只要 128 个并发 token 里还有 70%（甚至哪怕只有 1 个 token）需要跑后 4 层，GPU 就必须把这 4 层的全部权重从显存读进 SM。部分跳过只省下了那 30% token 的 KV cache 读取流量；只有当整个 batch 在这 4 层全部跳过（落入 <code>bucket 0</code>）时，才会省下读权重的带宽。
   </details>

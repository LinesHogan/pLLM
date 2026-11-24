# 改模型：激活补丁、Adapter 与前向内训练

在 PyTorch 或 Hugging Face 里调试算法时，修改模型的激活值或有效权重通常只需要注册一个 `forward_hook`，或者直接改写 `linear.weight`。但在 vLLM 这类高并发推理引擎中，几十个请求的 token 被连续批处理（continuous batching）打平在同一个二维张量中，模型的前向传播也被捕获进了 CUDA graph——Python 层的 hook 既无法区分哪一行张量属于哪个请求，在 CUDA graph 重放时更不会被触发；直接修改共享的权重矩阵则会破坏请求隔离并导致显存冲突。

pLLM 提供了三种声明式的改模型机制，在不改动共享底座权重的前提下，让每个请求或全局实例按需定制计算：

1. [`pllm.model.ActivationPatch`](../reference/api.md#pllmmodelactivationpatch)：在指定层之后的残差流上叠加激活修正（例如激活引导 Steering、LoReFT），完全运行在模型的 CUDA graph 内。
2. [`pllm.model.Adapter`](../reference/api.md#pllmmodeladapter)：在指定的投影矩阵（如注意力 Q/K/V 或 MLP 投影）上叠加权重增量 $\Delta W$（例如每请求 LoRA、稠密增量、多头增量），自动按请求隔离并在 CUDA graph 内寻址。
3. [`@pllm.train_in_forward`](../reference/api.md#pllmtrain_in_forward)：在前向传播运行的过程中，直接利用投影层的输入与目标更新私有状态（例如 In-Place TTT 快权重），实现边前向边训练的闭环。

---

## ActivationPatch：在残差流上打补丁

**要解决的问题**：在某些层计算完成后，直接对其输出的残差流（residual stream）做加性干预或低秩仿射变换，用于表征引导（Steering）或表征微调（LoReFT），且不能拖慢推理速度。

先看一个最小示例：在第 12 层的残差流上叠加一个外部传入的引导向量，并在解码阶段的每个生成 step 持续生效：

```python
from dataclasses import dataclass

import torch

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class Steering(pllm.Workload):
    steer = pllm.InstanceState({"vector": pllm.Field(lambda width, _: (width,))}, init=pllm.Imported())
    patch = pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)


print(pllm.explain(Steering(), CONFIG).eager_reason)
```

调用 `pllm.explain(Steering(), CONFIG).eager_reason` 输出为空字符串 `""`，这表明 `ActivationPatch` 完全留在模型的 CUDA graph 中执行，不会让前向传播退回 eager 模式。

### 声明参数与算子

`pllm.model.ActivationPatch` 的构造参数如下：

```text
pllm.model.ActivationPatch(
    state: str,
    layers: tuple[int, ...],
    first_tokens: int = 0,
    last_tokens: int = 2,
    operator: Literal["low_rank_affine", "add"] = "low_rank_affine",
    *,
    decode: bool = False,
    refresh: Literal["admission", "step"] = "admission",
    on_missing: Literal["stop", "hold"] = "stop",
)
```

- **`state`**：存放干预参数的状态名称（支持 `InstanceState` 或 `RequestState`）。
- **`layers`**：要干预的层号元组（必须升序且唯一），干预点位于该层之后的残差流（对应 Hugging Face 输出中的 `hidden_states[layer + 1]`）。
- **`operator`**：
  - `"add"`：纯加法干预，状态中每层需要一个形状为 `(hidden_size,)` 的向量字段，执行 $h \leftarrow h + v$。
  - `"low_rank_affine"`：低秩仿射变换（LoReFT 风格），状态中每层需要三个字段：权重矩阵 `(rank, hidden_size)`、偏置 `(rank,)` 与旋转矩阵 `(hidden_size, rank)`，执行 $h \leftarrow h + R^T (W h + b - R h)$。
- **Token 作用范围**：
  - `first_tokens` 与 `last_tokens`：仅对 prompt（prefill 阶段）的前 $k$ 个 token 和后 $m$ 个 token 生效。
  - `decode=True`：关键字参数，声明该补丁是否对生成阶段（decode）的每一个 token 持续生效。若 `decode=False`，声明非默认的 `refresh` 或 `on_missing` 会在构建时直接报错。
- **动态更新策略**：
  - `refresh="admission"`（默认）：请求在整个生命周期内固定读取它准入时绑定的状态版本。
  - `refresh="step"`：每个 decode 步都切换读取当步发布的最新版本。
  - `on_missing="stop"`（默认）与 `"hold"`：配合外部流式输入 `pllm.io.Feed` 使用。若某一 step 没有新的输入到达，`"hold"` 沿用上一步成功使用的版本，`"stop"` 则停下该请求。

### Forward-only KV：旧 token 不会回改

`ActivationPatch` 遵循 **Forward-only KV** 语义：
在第 $L$ 层注入的激活变化，会影响当前 token 在第 $L$ 层之后的后续网络层（$> L$），并伴随后续注意力层写入 K/V 缓存，从而对未来生成的 token 产生影响；但它**绝不会回改已计算完成的旧 token 在第 $L$ 层及之前已固化的 KV 缓存**。

为什么不重新计算旧 token 的 KV？因为自回归解码的核心吞吐优势在于重用前缀缓存。若每次激活微调都要求旧 token 的 KV 随之更新，解码单步的开销就会从 $O(1)$ 退化为重新遍历整个上下文的 $O(T)$，使连续批处理的吞吐彻底崩溃。

---

## Adapter：投影矩阵上的权重修正

**要解决的问题**：在不复制整个底座模型的前提下，为特定投影层（例如注意力的 Q/K/V 投影或 MLP 的 Gate/Up/Down 投影）挂载每请求独立的 LoRA 或参数补丁，并在同一个 batch 中并发计算不同请求的增量。

下面以 `PromptLoRA` 为例：每个请求在准入时通过 `pllm.io.StateImport` 载入属于自己的秩为 4 的低秩矩阵 $A$ 与 $B$，挂载在第 12 层的 `q_proj` 与 `v_proj` 上：

```python
@dataclass(frozen=True)
class PromptLoRA(pllm.Workload):
    rank: int = 4

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Imported(),
        )

    @property
    def apply(self) -> pllm.model.Adapter:
        targets = (pllm.model.Projection("q_proj", 12), pllm.model.Projection("v_proj", 12))
        return pllm.model.Adapter("lora", targets, operator="low_rank")


print(pllm.explain(PromptLoRA(), CONFIG).eager_reason)
```

输出同样为空字符串 `""`：对 `RequestState` 施加的 `Adapter` 完全驻留在 CUDA graph 内。

```python
# check-docs: skip: 需要已加载模型与 GPU 运行时
from vllm import SamplingParams

with pllm.make_llm(CONFIG, workload=PromptLoRA()) as llm:
    # 外部为每个 prompt 准备一份 LoRA 因子：RequestState 的导入值带一个前导的 prompt 轴
    a_q = torch.randn(1, 4, 896) * 0.02  # q_proj：输入 896，输出 896
    b_q = torch.randn(1, 896, 4) * 0.02
    a_v = torch.randn(1, 4, 896) * 0.02  # v_proj：Qwen2.5-0.5B 用 GQA，输出只有 2 个 KV 头 × 64 = 128
    b_v = torch.randn(1, 128, 4) * 0.02
    # 字段顺序对应两个 target：a_0, b_0, a_1, b_1
    batch = pllm.io.StateImport((a_q, b_q, a_v, b_v))
    outputs = llm.generate(["Hello world!"], SamplingParams(max_tokens=16), batch=batch)
```

### 声明参数与目标命名

`pllm.model.Adapter` 的构造签名如下：

```text
pllm.model.Adapter(
    state: str,
    targets: tuple[pllm.model.Projection, ...],
    weight: str = "value",
    operator: Literal["dense", "low_rank", "per_head"] | Callable[..., torch.Tensor] = "dense",
    factors: tuple[str, ...] = ("a", "b"),
    scale: float = 1.0,
    *,
    share: Mapping[str, tuple[tuple[int, ...], ...]] = {},
)
```

- **`targets`**：目标投影元组。投影名称由 `pllm.model.Projection(name, layer=None)` 声明，可选名称为 Hugging Face 标准的 `"q_proj"`、`"k_proj"`、`"v_proj"`、`"o_proj"`、`"gate_proj"`、`"up_proj"`、`"down_proj"`；`layer=None` 表示对模型所有层生效。
- **`operator`**：
  - `"low_rank"`：低秩乘积 $\text{scale} \cdot (x A^T) B^T$，读取 `factors`（默认 `("a", "b")`）命名的两个字段。
  - `"dense"`：稠密矩阵加法 $\text{scale} \cdot x W_{\text{delta}}^T$，读取 `weight`（默认 `"value"`）命名的字段。
  - `"per_head"`：按注意力头独立施加形状为 `(head_dim, head_dim)` 的块变换，读取 `weight` 字段。
  - 自定义可调用对象 `fn(hidden, *weights) -> delta`：纯 PyTorch 函数或 `torch.library.custom_op`（须为 `mutates_args=()` 且注册 `register_fake`），按顺序读取 `factors` 命名的字段（声明期使用 `FakeTensorMode` 与 `fixed_shapes` 校验；要求 `tensor_parallel_size=1`）。
- **`share`**：可选字段共享映射，从字段名映射到层号分组，使组内同名投影共享同一份状态字段（例如 Concertina 折叠架构在相邻层间共享低秩因子 `a`/`b`）。
- **多目标字段展开规则**：
  当一个 `Adapter` 作用于多个 `targets` 时，状态中的 `pllm.Field` 形状回调 `lambda in_features, out_features: (...)` 会按目标投影的输入与输出宽度被依次调用。底层分配的私有字段将按照 targets 列表的顺序自动展开并命名为 `a_0`, `b_0`, `a_1`, `b_1`, ...（声明在 `share=` 中的字段每组只在领头层分配一份）。

### `RequestState` 与 `InstanceState` 都能进图

- **`RequestState`**：每个请求独享一份状态页。`Adapter` 在 CUDA graph 内部通过 batch 中每一行所属的请求索引找到对应的私有因子地址并计算增量，全程无需重新捕获图，**留在 CUDA graph 内**。
- **`InstanceState`**：所有请求共享同一份状态（或按请求准入版本隔离的状态快照）。`Adapter` 通过统一的版本槽表在 CUDA graph 内部直接索引当前步骤绑定的状态页面，同样**留在 CUDA graph 内**，不会触发回退到 eager 模式。
  （在开启 `cuda_graphs="auto"` 且存在其他不可进图写法如 `@pllm.custom_forward` 时，`explain` 与 `llm.diagnostics().eager_reason` 会给出对应原因。）

### 为什么可训练的低秩状态必须非零初始化

如果状态带有优化器（例如在线训练 LoRA），且算子为 `"low_rank"`，pLLM 禁止使用默认的 `init=pllm.Zeros()`，必须显式传入非零初值（例如 `init=pllm.Computed(...)`）：

**原因**：低秩输出为 $\Delta y = \text{scale} \cdot (x A^T) B^T$。损失函数对因子 $A$ 的梯度正比于 $B^T$，对 $B$ 的梯度正比于 $A^T$。如果两个因子初始全为 0，那么第一步反向传播计算出的两个梯度的数值严格恒等于 0，优化器永远无法更新参数。常见的做法是在 `pllm.Computed` 中将 $A$ 随机高斯初始化（例如标准差 0.02），将 $B$ 置零：这既保证了初始时的有效增量为 0（不破坏基座表现），又保证了反向传播时 $B$ 能够获得非零梯度。

### 量化基座模型上的行为

在 AWQ、GPTQ、FP8 等量化模型上使用 `Adapter` 时：
- 底座的线性变换依然使用原本的高性能量化算子计算。
- `Adapter` 的低秩或稠密增量以高精度（FP16/BF16/FP32）独立计算后，作为**高精度残差修正**加在量化算子的输出上：$y = \text{quant\_linear}(x) + \Delta W \cdot x$。
- 底座模型本身的量化权重不会解包。只有当你的训练方法显式声明了 `pllm.operand("weight")` 时，引擎才会为该投影解量化并常驻一份高精度的稠密底座权重副本，带来额外的显存占用。

### 张量并行（TP）分片规则

在 `tensor_parallel_size > 1`（vLLM 0.32）下，`"dense"` 与 `"low_rank"` 算子严格遵循底座投影的通信切分：
- **列并行投影（q_proj, k_proj, v_proj, gate_proj, up_proj）**：按输出特征切分。稠密权重按行切分；低秩补丁中 $B$ 按输出维度分片，$A$ 在各卡间复制。
- **行并行投影（o_proj, down_proj）**：按输入特征切分。稠密权重按列切分；低秩补丁中 $A$ 按输入维度分片，$B$ 在各卡间复制。
- 状态必须由 `pllm.Zeros` 或 `pllm.Computed` 初始化，且只能由 `@pllm.train_in_forward` 在单卡本地分片上更新（禁止跨卡从列切分输出反向训练复制的 $A$，或从行切分输入反向训练复制的 $B$）。
- `"per_head"` 算子、自定义投影算子可调用对象，以及携带优化器或其他外部导入（`Imported` / Checkpoint）的状态目前要求 `tensor_parallel_size=1`。

---

## 独立声明状态生产者：同一份 Adapter 的多种来源

`pllm.model.Adapter` 只是一个**消费者声明**，它只负责回答“从哪个状态读取数据并施加给哪些投影”，完全不关心状态内部的数据从何而来。这种生产者与消费者的正交解耦，使得同一份 Adapter 结构可以无缝适配多种算法演进：

1. **外部导入（StateImport）**：每个请求在准入时直接写入外部超网络产出的因子或预加载的 LoRA 权重（参见教程 [02 每请求 LoRA](../tutorials/02_request_adapters.ipynb)）。
2. **前向后写入（@pllm.write）**：由驻留在 `InstanceState` 中的共享轻量网络读取当前请求 prompt 最后一个 token 的表征，在前向结束后写入请求的因子（例如 Text-to-LoRA）。
3. **在线蒸馏与训练（@pllm.train）**：在旁路 stream 上利用后层隐藏状态作为监督信号，通过 AdamW 优化器更新因子（例如 TLM）。
4. **前向内训练（@pllm.train_in_forward）**：在模型推理的单次前向内直接利用操作数就地更新因子。

消费端代码保持完全一致：无论由哪种生产者驱动，只要状态名称与形状匹配，`pllm.model.Adapter("lora", targets, operator="low_rank")` 就能在底层自动应用。

---

## @pllm.train_in_forward：前向过程中的内联训练

**要解决的问题**：在线测试期训练（Test-Time Training / In-Place TTT）要求快权重在**前向传播的内部**根据当前已流过的 token 立即更新，使得后续的 token 或后续层能立即在修正后的权重下执行。

下面以 `PromptAdapters` 为例：在 prompt prefill 阶段，模型每经过 16 个 token（一个 chunk），就利用当前层的输入重建该层 MLP 的输入，并通过 SGD 更新快权重，最后在 CPU 上用契约检查器验证正确性：

```python
TARGETS = tuple(pllm.model.Projection("down_proj", layer) for layer in (0, 2))


def start_small(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["a"].normal_(std=0.02, generator=generator)
    fields["b"].zero_()


@dataclass(frozen=True)
class PromptAdapters(pllm.Workload):
    rank: int = 4

    @property
    def adapters(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (self.rank, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, self.rank)),
            },
            init=pllm.Computed(start_small),
            optimizer=pllm.SGD(1e-3),
        )

    @pllm.train_in_forward(state="adapters", targets=TARGETS, operator="low_rank", chunk=16, flush="prompt_end")
    def learn(
        self,
        x: torch.Tensor = pllm.operand("input"),
        target: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        a: torch.Tensor = pllm.parameter("adapters", "a"),
        b: torch.Tensor = pllm.parameter("adapters", "b"),
    ) -> torch.Tensor:
        prediction = x @ base.T + (x @ a.to(x.dtype).T) @ b.to(x.dtype).T
        return (prediction.float() - target.float()).square().mean(-1)


print(testing.check(PromptAdapters(), CONFIG).passed)
```

`testing.check(PromptAdapters(), CONFIG).passed` 在 CPU 上完整模拟了行置换、请求隔离、分块调度和抢占重放，返回验证通过的契约列表。

### 操作数（Operands）与参数约束

在 `@pllm.train_in_forward` 方法体内，参数通过默认值声明输入来源：
- `pllm.operand("input")`：目标投影算子当前收到的输入行。
- `pllm.operand("target")`：当前投影所属层 MLP 的输入激活值（标准 In-Place TTT 的目标特征）。
- `pllm.operand("weight")`：该投影的底座权重矩阵（量化模型上会按需物化稠密副本）。
- `pllm.operand("prompt_end")`：一个 0 维布尔标量张量，指示当前 chunk 是否已经到达 prompt 的末尾。
- `pllm.parameter(state, field)`：引用请求私有的状态缓冲区（如 $a, b$ 或 $\Delta W$）。

**修改权限**：`pllm.operand` 捕获的张量是底座前向的中间激活，**严格只读**；代码中唯一允许原地修改（in-place）的只有 `pllm.parameter` 授权的私有参数缓冲区。

### 分块优化（Chunked SGD）与尾部刷新

- **`chunk=`**：设置更新的 token 粒度。当状态绑定了 `optimizer=pllm.SGD(lr)` 时，方法只需返回一个逐行 loss 张量（形状为 `[rows]`），框架会在累积满一个 chunk 的行数后自动执行一次 SGD 步进。
- **`flush="prompt_end"`**：当 prompt 的总 token 数不能被 `chunk` 整除时（例如长度为 50，chunk 为 16），末尾余下的 2 个 token 通常会被丢弃。设置 `flush="prompt_end"` 后，框架会在遇到 prompt 边界时就地触发一次更新，确保所有 token 都参与适应。
- **`tokens=`**：指定哪些阶段参与更新，默认为 `tokens="prefill"`（只在 prompt 阶段训练，生成阶段冻结使用），亦可设为 `"decode"` 或 `"all"`（参见 [08 In-Place TTT](../examples/08_inplace_ttt.ipynb)）。

### 可见性策略：forward_pinned 与 exact 的代价

- **`visibility="forward_pinned"`（默认）**：整趟前向传播自始至终读取进入该步时固定的状态版本；前向内产生的更新只发布给后续的前向步骤读取。
- **`visibility="exact"`**：同一个前向传播内部，后一个 chunk 的投影计算能够**立即看到前一个 chunk 刚更新出来的最新参数**。
- **代价与权衡**：`"exact"` 打破了整批并行的独立性，在前向计算流内部插入了 chunk 级别的同步与参数依赖，需要框架维护私有的中间状态缓冲区；除非算法显式依赖块间递推（如 TTT），否则建议使用默认的 `"forward_pinned"`。

### 手动计算更新：返回 UpdateOutput

如果不使用框架的 SGD 优化器，而是自己推导了封闭解或特定算子，可以不给状态指定优化器，方法直接返回 `pllm.UpdateOutput`：

```python
targets = (pllm.model.Projection("down_proj", 0),)


class ManualFastWeights(pllm.Workload):
    factors = pllm.RequestState(
        {
            "a": pllm.Field(lambda in_features, out_features: (2, in_features)),
            "b": pllm.Field(lambda in_features, out_features: (out_features, 2)),
        }
    )

    @pllm.train_in_forward(state="factors", targets=targets, operator="low_rank", factors=("a", "b"), scale=1.0)
    def update(
        self,
        x: torch.Tensor = pllm.operand("input"),
        y: torch.Tensor = pllm.operand("target"),
        a: torch.Tensor = pllm.parameter("factors", "a"),
        b: torch.Tensor = pllm.parameter("factors", "b"),
    ) -> pllm.UpdateOutput:
        residual = (x @ a.to(x.dtype).T) @ b.to(x.dtype).T  # 只返回叠加在底座上的残差
        a.add_((0.1 * y[:, : a.shape[0]].T @ x).to(a))      # 原地更新私有参数
        b.add_((0.1 * (y.T @ y)[:, : b.shape[1]]).to(b))
        return pllm.UpdateOutput(
            residual,
            residual.new_ones((), dtype=torch.bool),              # changed: 是否产生更新
            residual.new_ones((), dtype=torch.int64),             # update_count: 更新步数计数
            residual.new_full((), x.shape[0], dtype=torch.int64), # trained_rows: 参与训练的行数
        )


print(pllm.explain(ManualFastWeights(), CONFIG).eager_reason)
```

`UpdateOutput` 包含四个字段：`residual` 为加到底座输出上的增量张量；其余三个均为与输出同设备的 0 维张量计数器，确保整个前向过程不需要任何 GPU 到 CPU 的主机同步。

---

## 在 @pllm.custom_forward 里使用 Adapter：修正跟着每一次投影调用走

在使用 [`@pllm.custom_forward`](../tutorials/04_custom_forward.ipynb) 重写模型的层循环逻辑时（例如循环深度 looped depth、动态跳层或层重放），标准前向中“每一层每个 token 恰好执行一次”的假设被打破了。此时挂载在模型上的 `Adapter` 遵循以下核心规则：

> **核心原则：权重修正跟着每一次实际的投影算子调用走，按当次调用的实际输入计算，并按每一行所属请求的状态施加。**

1. **按次触发，有调用就有修正**：只要目标投影被实际执行——无论是常规层调用、循环深度中的重复调用，还是重放——`Adapter` 都会基于当次调用的实际激活值施加一次权重增量。如果某几层在循环深度中被循环执行了 3 遍，那么这几层的目标投影就会被修正 3 次。这在物理语义上等价于“带着被修正后的权重跑了 3 圈”。
2. **状态归属进入前向时固定**：每一行对应哪个请求的状态，在进入 forward 时就已经锚定。即便在 `@pllm.custom_forward` 里通过 `active=` 压缩了活跃行，或者将请求重排成不同的 wave，底层索引始终保留每行的请求归属，确保每行计算时准确索引到自己请求的私有状态。
3. **跳过的不加，补写的照加**：完全跳过的投影不会发生调用，自然不施加修正；但在跳层执行时，如果触发了原生 QKV 传播来补齐 K/V 缓存，实际运行的 Q/V 投影依然会加上各自的权重增量。
4. **模型与深度调度解耦**：同一个带有 `Adapter` 的适配模型，既可以直接跑原生前向，也可以原封不动地交给不同的 `@pllm.custom_forward` 调度逻辑复用，无需为自定义控制流专门重写适配代码。
5. **限制**：`Adapter` 表达的是“该投影的物理权重改变了”，没有按循环圈数开关的选项。如果算法要求“仅在第一圈施加修正，后续各圈不再生效”，这属于隐藏层激活干预，应当在 `@pllm.custom_forward` 中通过 Python 代码显式加在第一圈的 hidden 张量上，而非使用 `Adapter`。

---

## 高级入口：运行时动态切层（`pllm.model.LayerRoute`）

在服务已经运行且捕获完 CUDA graph 后，如果希望动态切换探针捕获层、激活引导层或 LoRA 挂载层（例如将观察与引导目标从第 8 层切到第 20 层，或在同一连续批处理中让不同请求命中不同层，甚至由模型自身通过 `@pllm.write` 在 GPU 上计算下一步作用的层），可以使用显式的高级入口 [`pllm.model.LayerRoute`](../reference/api.md#pllmmodellayerroute)：

```python
class DynamicSteering(pllm.Workload):
    ctrl = pllm.InstanceState(
        {"layer": pllm.Field((), torch.int64), "vector": pllm.Field(lambda width, _: (width,))},
        init=pllm.Imported(),
    )
    route = pllm.model.LayerRoute("ctrl", over=(4, 12, 20))
    steer = route.patch(operator="add", decode=True)

    @pllm.export(name="probe", max_rows=8, max_cols=1)
    def capture(self, r: torch.Tensor = route.residual()) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


print(testing.check(DynamicSteering(), CONFIG).passed)
```

### 设计约束与代价说明

动态切层之所以独立为 `pllm.model.LayerRoute` 而非普通层号参数，是因为它会在候选层集合 `over` 上预埋门控图节点，并非零代价：

1. **必须显式指定候选层集合 `over=(...)`**：不存在隐式挂满全模型所有层的默认选项。候选层越少，图内门控节点的固定发射开销越低；在 CUDA 下，动态探针（`hidden` / `residual` / `layer_logits`）、`add` 与 `low_rank_affine`（`rank <= 256`）补丁、以及 `dense` 与 `low_rank`（`rank <= 256`）适配器会在 L2 中检查 `int64` 层号并跳过未命中候选层的参数读取与矩阵乘（补丁所在的边界归一化仍会传递残差与分支张量端口，`per_head`、`rank > 256` 或最后一维非连续的补丁输入则回退至掩码求值）。
2. **运行时整数编码与 `-1` 旁路开关**：控制字段必须声明为无优化器状态中的 0 维 `int64` 标量 `pllm.Field((), torch.int64)`，存放非负物理层号（须属于 `over`），或传入 `-1` 动态关闭（`patch` / `adapter` 退化为恒等映射，读取或未在 `@pllm.custom_forward` 中访问到的层返回全零张量并在 `event.layers` 记为 `-1`）。通过 `StateImport` / `import_state` 或 `Feed` 从宿主传入不在 `over` 且非 `-1` 的层号会在边界直接报错；由 `@pllm.write` 在 GPU 上算出的越界值按 `-1` 处理，全程保持 0 主机同步。
3. **默认逐步生效（`refresh="step"`，Forward-only KV）**：`route.adapter(...)` 与 `route.patch(decode=True)` 默认采用 `refresh="step"`（`decode=False` 时默认为 `refresh="admission"`），在分块 prefill 与 decode 各步均读取当前步最新版本而不钉住请求准入快照；在线调用 `llm.import_state(...)` 后，正在运行的请求在下一个前向步立即切换到新层，历史已写入的 KV 缓存保持不变。
4. **单槽路由与多层参数库自动适配**：默认 `layers=()` 时，状态只需存储单份干预向量或 LoRA 因子（$O(1)$ 显存），在 `over` 间动态路由（要求 `over` 内同名投影形状一致）；若目标 `InstanceState` 声明了 `layers=over`，则自动绑定每层独立参数槽 `{field}_{l}`，供 GPU `@pllm.write` 零拷贝跳层。

---

## 常见错误与运行约束：抢占、重算与前缀缓存

在将模型修改类算法推向实际部署时，底层的并发调度会引入一些边界约束。

### 抢占（Preemption）后的重算行为

当 vLLM 的 GPU 显存用尽且无法容纳新的 KV block 时，调度器会触发**抢占（preemption）**：驱逐某些低优先级请求的 KV cache，待显存空闲后再重新计算这些请求从 prompt 起始的历史。

- **能够无缝重算的负载**：
  若请求状态以 `request_id` 为键（`key_fields` 包含 `request_id`），负载中**没有可变的实例状态**，也**没有外部 feed、导入或宿主输入**，重算历史就能完全复现原来的状态，引擎直接重算。`pllm.explain` 的 `preempt` 行会写明你的负载属于哪一种。
- **会直接报错中断的负载**：
  任何依赖外部逐步输入、含有可变 `InstanceState`、或存在跨步状态依赖的 overlay / 激活修改负载，在发生抢占时**无法保证重算状态与历史严格一致**，此时引擎会直接抛出运行时错误，原文和改法见报错目录的[请求与运行时](../reference/errors.md#请求与运行时)一节。

> **历史 KV 的因子保留**：在因果自回归模型中，已写入的旧 KV 缓存保留的是计算当时所使用的有效权重；后续 step 更新 Adapter 因子不会回溯重算历史前缀的 KV。

### 为什么 Prefix Caching 与 Overlay / 激活补丁互斥

在 vLLM 配置中，`enable_prefix_caching=True` 会在不同的请求之间共享具有相同前缀文本的 KV 块。
**在 pLLM 中，除仅读取不可变 `InstanceState` 的 `Adapter` 与 `last_tokens=0` 的 `ActivationPatch` 外，其余 Overlay（`Adapter`、`train_in_forward`）与 `ActivationPatch` 均与 Prefix Caching 严格互斥，组合时会在构建期直接拒绝。**

**原因**：
Prefix Caching 的基本假设是“相同的 token 序列产生完全相同的 KV 张量”。但无论是每请求的 LoRA 权重，还是逐步更新的快权重，亦或是逐层注入的激活引导，都会改变前向传播的实际数学计算：
1. 具有相同 prompt 文本的两个请求，若挂载了不同的 LoRA 因子，其产生的 KV 缓存是完全不同的；若强行共享前缀缓存，后一个请求就会直接读取前一个请求在错误权重下生成的激活值。
因此，pLLM 采取安全关闭的原则：凡是改写模型激活或投影权重的负载，必须配置 `enable_prefix_caching=False`。

---

## 自测

<details>
<summary>1. 为什么在带优化器的低秩 LoRA 状态中，不能使用默认的 init=pllm.Zeros()？</summary>

低秩矩阵增量计算公式为 $\Delta y = \text{scale} \cdot (x A^T) B^T$。损失对 $A$ 的偏导数乘有因子 $B^T$，对 $B$ 的偏导数乘有因子 $A^T$。若 $A$ 和 $B$ 均初始化为全 0，则两个矩阵的第一步梯度数学上严格恒为 0，优化器永远无法学动参数。必须使用非零初始化（如 $A$ 高斯随机初始化，$B$ 置 0）。
</details>

<details>
<summary>2. 为什么 ActivationPatch 对第 12 层的修改不会改变旧 token 在第 12 层的 KV cache？</summary>

自回归解码遵循 Forward-only KV 机制：旧 token 的 KV 缓存在处理该 token 的当步前向时就已经计算并落盘。若要让过去的 KV 缓存也反映新的激活干预，引擎必须在每个生成步骤重新执行一遍整个历史前缀的前向传播，这会导致生成延迟从 $O(1)$ 骤增为 $O(T)$，严重破坏并发吞吐。
</details>

<details>
<summary>3. 挂载到 InstanceState 上的 pllm.model.Adapter 会退回 eager 模式吗？</summary>

不会。无论绑定 `RequestState` 还是共享 `InstanceState`，`pllm.model.Adapter` 都通过统一的可进图覆盖表按每行或版本槽地址解析增量，保留在 vLLM 的 CUDA graph 内。只有在主动关闭 CUDA graph（`cuda_graphs=False`）或使用了切片捕获图写法（如 `@pllm.custom_forward`）时才不会保留整模型 CUDA graph。
</details>

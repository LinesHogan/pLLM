# 状态

在推理期算法中，算法往往需要跨 decode 步或跨请求保留并演化数值：例如探针的权重矩阵、每个请求独有的适配器网络、跨请求共享的大表，或记录生成过程的滑动统计量。在普通 PyTorch 中，直觉通常是直接修改模块属性或给对象赋新值；但在 vLLM 与 pLLM 中，每一步连续批处理会重新组织张量行，方法作为录制好的 CUDA Graph 重放，借来的张量视图禁止原地修改。

**状态（State）是 pLLM 中唯一可变的东西**。只要某个值需要在步与步之间、请求与请求之间存活，就必须显式声明为状态。声明状态后，pLLM 负责分配显存、按版本安全发布、处理 CUDA Graph 捕获，并在后台重叠训练与前向。

---

## 1. 为什么状态是唯一可变的东西

在编写算法时，可能会尝试在方法内部写入 `self.mean += 1`，或者对输入张量直接执行原地操作 `w.add_(1)`。在 pLLM 体系下，这些做法会被拒绝或完全失效，其原因如下：

1. **CUDA Graph 捕获与回放**：在解码阶段，方法会被捕获进 CUDA Graph。在后续步骤的图回放期间，Python 解释器根本不会再次执行方法体中的 Python 语句。任何写在 Python 实例属性上的变量（如普通字典或列表）都不会随 GPU 上的每步前向而更新。
2. **借用视图为只读**：传给方法的输入参数（如 `pllm.hidden`）和参数视图（如 `pllm.parameter`）都是借来的只读张量视图。直接在原地修改它们（如 `w.add_(1)` 或 `h.mul_(2)`）会破坏 CUDA Graph 内部的张量复用，也会污染其他并发请求或后续步骤读取的数据，因此被创作契约严格禁止并在检查时报错。
3. **连续批处理动态排布**：每一步前向拼入批次的请求及其对应的张量行都在动态变化。没有框架的统一显存管理，普通张量无法知晓哪一行对应哪条请求。

因此，任何需要跨步演化的值，都必须显式声明为 pLLM 状态，并只能通过 `@pllm.train`（优化器更新）或 `@pllm.write`（自定义计算更新）来发布新版本。

---

## 2. 两种作用域与路由键

声明状态时，首要决策是选择其**生命周期与共享范围**。pLLM 提供两种作用域：

| 作用域 | 声明方式 | 语义与生命周期 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **请求作用域** | `pllm.RequestState(...)` | 每个请求（或每组请求）拥有一份独立的私有状态副本。随请求准入（admission）分配，随请求完成而释放。 | 每请求 LoRA、每个请求的在线蒸馏小网络、每个请求的 token 累加统计。 |
| **实例作用域** | `pllm.InstanceState(...)` | 整个推理引擎实例共享唯一一份状态。所有并发请求读取并操作同一组物理张量。 | 冻结探针权重、跨请求共享的嵌入词表、全局大表、模型级统计量。 |

### 用 `key_fields` 对请求状态分组

`RequestState` 默认按请求严格隔离，即每个请求独立拥有自己的状态副本。如果需要让一组请求共享同一份状态（例如同一 Prompt 的多个候选采样共享同一个适配器，或者同一租户的请求共享训练成果），可以通过 `key_fields` 声明路由键：

- `("request_id",)`（默认）：每个请求独立一份状态副本，互不干扰。
- `("prompt_index",)`：同一次 `generate` 中同一 Prompt 抽取的多个采样（`sample`）共享同一份状态副本。
- `("sample_index",)`：跨 Prompt 按采样的相对序号对齐共享。
- `("route",)`：按外部业务路由键共享。宿主调用 `generate(prompts, routes=...)` 传入路由值，路由值相同的请求加入同一状态副本。即使当前请求结束，该状态副本仍会在会话中保留，供后续具有相同路由值的请求复用。关于路由键的进阶用法与容量配置，详见 [让一组请求共享训练成果](../howto/share-training-across-requests.md)。

---

## 3. 声明字段：形状、Dimensions 与 Field

状态由一个或多个具名字段组成。字段可以通过单个形状或字典方式声明：

### 单字段与多字段

- **单个字段**：直接传 `shape=(...)`，该状态默认拥有一个名为 `"value"` 的字段。
  ```python
  from dataclasses import dataclass

  import torch
  from torch import nn

  import pllm

  bias = pllm.InstanceState(shape=(896,))
  ```
- **多字段**：通过 `fields={"name": spec, ...}`（或第一个位置参数传入字典）声明具名字段：
  ```python
  linear = pllm.RequestState({
      "weight": lambda m: (m.hidden_size, m.hidden_size),
      "bias": (896,),
  })
  ```

### 字段形状的声明形式

每个字段的形状规范支持三种形式：
1. **静态固定元组**：如 `(896,)`、`(4, 896)`。
2. **接收 `pllm.Dimensions` 的 Callable**：如 `lambda m: (m.vocab_size,)`。参数 `m` 暴露模型的静态事实：
   - `m.hidden_size`：隐藏层维度（例如 Qwen2.5-0.5B 为 896）；
   - `m.vocab_size`：词表大小（例如 151936）；
   - `m.layer_count`：层数（例如 24）；
   - `m.target_shapes`：状态被 `pllm.model.Adapter` 读取时，各目标投影的 `(in_features, out_features)`；
   - `m.dtype`：模型的精度。
3. **`pllm.Field`**：当字段需要按所附加的目标投影层动态确定形状时使用（例如配合 `pllm.model.Adapter`）：
   ```python
   lora_a = pllm.Field(lambda in_features, out_features: (4, in_features))
   ```
   同时，`pllm.Field` 还支持为单个字段显式指定独立的存储精度（如 `pllm.Field((896,), dtype=torch.float32)`）。
   同一个状态中可以自由混合普通形状元组和 `pllm.Field`（例如 `{"w": pllm.Field(lambda width, _: (width,)), "b": (1,)}`）。

### 类属性 vs 带类型的 property

- **类属性**：如果状态的形状和配置在类编写时即可确定，直接声明为类属性。
- **带类型的 property**：如果状态形状依赖于算法的初始化参数（例如在 `@dataclass(frozen=True)` 中由用户传入的 `rank`），必须声明为 `@property`，并且**必须标注返回类型**（`-> pllm.RequestState` 或 `-> pllm.InstanceState`）：

```python
@dataclass(frozen=True)
class DynamicProbe(pllm.Workload):
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )
```

**为什么必须显式标注返回类型？**
pLLM 在构建引擎和执行 [`pllm.explain`](explain.md) 规划时，通过静态反射检查类属性与方法。类型注解让规划器在不实例化未知代码或触发潜在副作用的情况下，快速准确地识别出哪些成员是状态以及它们的作用域。

---

## 4. 初始值来源：`init=`

每个状态在准入时（请求开始或引擎启动）都必须有明确的初值。初值由 `init=` 参数声明（每条请求的初始副本都拷贝自该来源）：

| `init=` 选项 | 作用域 | 初始值来源与行为 |
| :--- | :--- | :--- |
| `pllm.Zeros()` | Request / Instance | 所有字段初始化为 0（默认行为）。 |
| `pllm.Computed(function)` | Request / Instance | 引擎构建时用带种子的生成器调用 `function(fields, generator)` 原地填充一次，后续每次准入直接拷贝该初始值。 |
| `pllm.Pretrained(path, placement="device")` | Request / Instance | 从 `.safetensors` 文件加载与字段同名的张量（文件哈希计入负载身份）。`placement="host"` 可将只读大表固定在锁页主机内存中，默认 `"device"` 放不下时绝不隐式降级。 |
| `pllm.ModelCheckpoint(names)` | 仅限 InstanceState | 读服务模型检查点里 vLLM 模型类用不到的额外张量（`names` 映射 `{field: tensor_name}`，可含 `{layer}` 占位符），这些张量从 vLLM 的加载中排除；只读，无优化器。不要用它读 vLLM 自己要用的权重。 |
| `pllm.FromModule(factory, pretrained="")` | Request / Instance | 将 `factory(dimensions)` 构建的 `nn.Module` 的参数作为状态的唯一字段。按模块自身的机制初始化，或从 `pretrained` 路径加载权重。 |
| `pllm.Fed(order="ordered" \| "latest")` | 仅限 InstanceState | 由外部生产者进程通过 `llm.feed(state)` 动态推送更新；`"ordered"` 严格消费每次发布，`"latest"` 仅保留最新；每个负载至多声明一个。 |
| `pllm.Imported()` | Request / Instance | 每次发起推理时通过 `generate(batch=pllm.io.StateImport(...))` 在请求首个前向之前传入；RequestState 为每个 prompt 提供一行，InstanceState 提供整个张量。 |

关于从磁盘载入预训练权重或外部 safetensors 的具体操作与排查，详见 [加载训练好的权重](../howto/load-trained-weights.md)。

---

## 5. 冻结状态（只读状态）

如果一个状态不需要被 `@pllm.train` 或 `@pllm.write` 更新，只需省略 `optimizer=`（或显式写 `optimizer=None`）。这种状态称为**冻结状态（Frozen State）**。

冻结状态具有以下性质：
1. **零额外开销**：不分配梯度槽位与 Adam 动量槽位，物理显存仅等于字段张量本身的大小。
2. **不被优化器改变**：它的值只来自初值，或者来自 `pllm.io.StateImport` 导入和 `@pllm.write` 写入的新版本。
3. **全局自由可读**：可以在任何方法（包括观测导出 `@pllm.export`、采样引导 `@pllm.guide_sampling`、残差改写 `@pllm.custom_residual` 等）中通过 `pllm.parameter` 读取。

---

## 6. 存储精度自动规则（`dtype="auto"`）与显式覆盖

状态参数默认采用 `dtype="auto"`。pLLM 会在构建期静态分析每个字段在各个方法中的读写角色，自动权衡数值稳定性与显存开销：

| 字段在方法中的角色 | `dtype="auto"` 推导的精度 | 原因与实测影响 |
| :--- | :--- | :--- |
| **被优化器更新的参数**（`@pllm.train`） | `torch.float32` | 防止微小梯度与动量更新在低精度下下溢被吞噬。在 BF16 下长序列梯度的累加极易发散。 |
| **读出再写回的累加器**（`@pllm.write`） | `torch.float32` | 跨步累加（如滑动均值）每次更新幅度小，在低精度下误差会持续累积放大。 |
| **跨块/跨步进位字段**（carry 字段） | `torch.float32` | 跨步连续递推测试表明，carry 采用 BF16 相对误差可高达约 $1.4 \times 10^{-1}$，而采用 FP32 仅约 $3.7 \times 10^{-4}$。 |
| **只读字段**（如冻结探针、引导向量）<br>或 **只写字段** | 跟随模型精度（如 `torch.bfloat16`） | 省下一半状态显存与访存带宽，且与主干模型的 GEMM 计算精度自然对齐。 |

### 显式覆盖精度

如果需要在只读字段上直接执行仅支持单精度的算子，可以在 `InstanceState` / `RequestState` 上指定 `dtype=torch.float32`，或在 `pllm.Field` 中为单个字段覆盖：

```python
state = pllm.InstanceState({
    "w_fp32": pllm.Field(lambda width, _: (width,), dtype=torch.float32),  # 显式 FP32
    "w_model": pllm.Field(lambda width, _: (width,)),                     # 自动跟随模型精度（BF16）
})
```

---

## 7. 读取状态与可见版本

在算法方法中，通过以下四种输入标记读取状态及其元数据（作为参数默认值声明）：

| 读取标记 | 返回类型与形状 | 语义说明 |
| :--- | :--- | :--- |
| `pllm.parameter(state, field, *, per_row=False)` | `torch.Tensor` | 读取指定状态的某个具名字段。<br>• **RequestState**：自动带有领先行轴 `[rows, *S]`，对齐当前操作调度的行（每行对应自己请求的状态）。<br>• **InstanceState**：保持声明时的原始形状 `[*S]`；若指定 `per_row=True`，会自动展开一个广播行轴 `[rows, *S]` 以便进行按行矩阵运算。 |
| `pllm.module(state)` | `nn.Module` | 读取以 `init=pllm.FromModule(...)` 初始化的模块。<br>对 `RequestState`，每个请求行在计算时使用该请求私有的参数副本。 |
| `pllm.version(state)` | `torch.Tensor`（int64 `[rows]`） | 返回当前每条请求的状态已完成更新发布的次数（准入时为 0）。常用作“是否已训练过”的条件门控（例如 `torch.where(version > 0, ...)`）。 |
| `pllm.index(state)` | `torch.Tensor`（int64 `[rows]`） | 返回当前每条请求在底层状态池中的槽位编号，用于对外部池化张量做按行查表。 |

### 读取看到的是哪个版本？

任何方法在某一步中读取状态时，看到的**永远是当前步骤开始前已发布的版本**（发布前版本）。
- 即使当前步骤的旁路流正在训练并准备写入新版本，当前主干流的前向计算依然读取上一步发布的旧版本；
- 正在运行的请求始终保持自己的版本快照，绝不会读到只更新了一半的中间脏数据。关于版本发布的时序与流重叠模型，详见 [核心概念](../explain/core-concepts.md)。

---

## 8. 跨步记忆：carry 字段与 `pllm.carry`

如果需要在解码步骤之间传递非优化器参数的连续记忆（例如上一步的预测误差、上一步记录的浅层特征），可以使用 **carry 字段**：

1. **声明 carry 字段**：在状态中配置 `carry={"field_name": shape}`。carry 字段不分配梯度与动量槽位，不会被优化器更新。
2. **在训练中写入**：把 `@pllm.train` 方法的某个输入默认值用 `pllm.carry(source, state=..., field=...)` 包裹。在训练更新发布时，框架自动将该输入中每条请求最后一行（最新 token）的张量值写入对应的 carry 字段。
3. **在下一步读取**：使用 `pllm.parameter(state, "field_name")` 读取上一步发布的值。

**为什么第一步读取到的始终是零？**
状态的 carry 字段在请求准入时进行零初始化。根据“读取看到当前步开始前已发布版本”的时序规则，在第 0 步开始时，该状态尚未发生过任何更新发布，因此第 0 步读取到的初始版本只能是零。这正是版本发布语义的自然体现。

---

## 9. PositionState 与主机只读表

除了常规的张量状态外，pLLM 还支持两种特定拓扑的状态存储：

### PositionState

`pllm.PositionState(initial=0.0)` 每个 KV cache 块持有一个 hidden 宽度的向量，仅供 `@pllm.custom_forward` 内通过 `layers.load` / `layers.save` 读写（详见 [改写 forward 结构](forward-structure.md)）。

### 主机只读大表（`placement="host"`）

当算法需要使用超大规模的只读表（如百万级 n-gram 先验表、检索记忆库）时，将其全部常驻 GPU 显存会导致 KV Cache 空间严重缩减。通过 `pllm.InstanceState(..., init=pllm.Pretrained(path, placement="host"))`，可将只读表保留在 CPU 锁页内存中：
- **旁路重叠搬运**：仅依赖 token ID 的查表方法会被规划器调度在每步最初的 `request.prepare` 阶段、于独立的旁路 CUDA stream（`side` lane）上执行。它只占用约四分之一的 SM 算力通过 PCIe 异步搬运所需行，将传输时间完全隐藏在模型前向计算的背后。
- **允许的算子**：为了保证静态内存与传输边界，主机表仅支持按行抽取操作：切片索引（`table[idx]`，包含先做切片视图如 `table[:, ::2]` 再索引）、`torch.index_select`、`torch.nn.functional.embedding` 以及 `torch.vmap`。在规划期尝试对主机表执行其他算子会被直接拒绝。更详细的带宽模型见 [性能与显存](../explain/performance.md)。

---

## 10. 显存预留与预算控制

状态虽然按需分配，但必须在引擎启动前完成显存预留规划：

- **为什么预留的页数大于并发请求数？**
  在 [`pllm.explain`](explain.md) 的报告中，状态每个字段的页数（`x N pages`）通常显著大于 `max_num_seqs`。这是因为在默认的跨步重叠模式（`publish_lag=1`）下，旁路流正在写出第 $n+1$ 步的新版本，而主干流还在并发读取第 $n$ 步的旧版本，新旧多个版本必须在显存中同时并存。旧版本只保留读者会读的字段：AdamW 的步数和一、二阶矩只有优化器读，它们只为最新版本和正在写的版本预留。
- **预算控制旋钮**：
  `ExecutionOptions.state_page_budget_bytes` 约束状态池的总显存上限，超出可用空间时会在编译期抛出 `ValueError`。`ExecutionOptions.execution_peak_budget_bytes` 约束包括状态、方法工作区和 CUDA Graph 预留在内的整体执行峰值，并在显存吃紧时自动将旁路任务退回主流内联执行以防止 OOM。运行时的具体物理开销与数值健康状态可通过 `llm.diagnostics().state_usage` 和 `numeric_health` 查询，详见 [配置项](../reference/config.md) 与 [explain 与 diagnostics 字段](../reference/diagnostics.md)。

---

## 11. 代码示例

下面的代码示例均可在 CPU 上直接运行校验（无需 GPU）。

### 示例 1：带 `Computed` 初值的冻结探针（`InstanceState`）

该示例声明一个整个引擎共享的冻结探针分类器，在初始化时通过生成器原地填充权重，通过 `@pllm.export` 导出每个 token 的判别分数。

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


def init_probe(fields: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    fields["weight"].normal_(generator=generator)
    fields["bias"].zero_()


class FrozenProbe(pllm.Workload):
    # 实例作用域 + 冻结状态：准入时初始化一次，永不被优化器更新
    classifier = pllm.InstanceState(
        {"weight": lambda m: (m.hidden_size,), "bias": (1,)},
        init=pllm.Computed(init_probe),
    )

    @pllm.export(name="probe_score", max_rows=8, max_cols=1)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        w: torch.Tensor = pllm.parameter("classifier", "weight"),
        b: torch.Tensor = pllm.parameter("classifier", "bias"),
    ) -> torch.Tensor:
        # [rows, 896] * [896] -> [rows, 1]
        return torch.sigmoid((h.float() * w).sum(dim=-1, keepdim=True) + b)


print(testing.check(FrozenProbe(), CONFIG).passed)
```

### 示例 2：每请求独立的神经网络模块（`RequestState`）

该示例声明一个随请求动态分配的小型神经网络模块，每条请求拥有独立的参数，并在前向计算时逐行调用各自的私有网络。

```python
@dataclass(frozen=True)
class PerRequestHead(pllm.Workload):
    rank: int = 4

    # 形状依赖实例属性 rank，声明为带类型注解的 property
    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        head: nn.Module = pllm.module("head"),
    ) -> torch.Tensor:
        # head 内部自动为每条请求行绑定该请求独立的权重副本
        return head(h.float())


print(testing.check(PerRequestHead(), CONFIG).passed)
```

### 示例 3：用 `@pllm.write` 更新请求滑动均值

该示例演示如何使用自定义张量数学更新状态：每一步读取上一步的滑动均值与当前 token 的特征向量，计算加权更新后通过 `@pllm.write` 原子发布。

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        # 累加器读取并重写自身，自动推导存储精度为 float32
        return (0.9 * mean + 0.1 * h.float(),)

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(
        self,
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)


print(testing.check(RunningMean(), CONFIG).passed)
```

### 示例 4：带 `carry` 进位字段的跨步记忆

该示例演示在在线训练时跨步记录信息：训练方法计算预测误差，同时通过 `pllm.carry` 将浅层激活记录进 carry 字段，供下一步训练时作为递推输入读出。

```python
@dataclass(frozen=True)
class StepMemory(pllm.Workload):
    beta: float = 0.5

    head = pllm.RequestState(
        {"w": lambda m: (m.hidden_size, m.hidden_size)},
        optimizer=pllm.AdamW(lr=1e-3),
        carry={"prev_act": lambda m: (m.hidden_size,)},
    )

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> torch.Tensor:
        return torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)

    @pllm.train(state="head")
    def loss(
        self,
        # 发布更新时，将 shallow 中每个请求最新行的值存入 prev_act 字段
        shallow: torch.Tensor = pllm.carry(pllm.hidden(0), state="head", field="prev_act"),
        deep: torch.Tensor = pllm.hidden(-1),
        w: torch.Tensor = pllm.parameter("head", "w"),
        # 读取上一步发布的值（第一步读取为全 0）
        prev: torch.Tensor = pllm.parameter("head", "prev_act"),
    ) -> torch.Tensor:
        pred = torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)
        return (pred + prev - deep.float()).square().mean(-1)


print(pllm.explain(StepMemory(), CONFIG))
```

---

## 12. 常见错误

| 报错（节选） | 原因 | 改法 |
| :--- | :--- | :--- |
| `... mutates borrowed input ... in place; inputs and parameters are read-only views. ...` | 在方法里原地改了 `pllm.parameter` 读到的视图 | 返回新值，经 `@pllm.write` 或 `@pllm.train` 发布 |
| `@pllm.train reads ..., which trains with ...; declare ... with optimizer=...` | 训练的状态没有优化器 | 按报错给状态加优化器 |
| `carry fields ride on the training transition; give the state an optimizer` | carry 字段随训练更新一起发布 | 给状态加优化器 |

其余与状态有关的报错见[报错目录](../reference/errors.md)。

---

## 13. 自测

<details>
<summary>1. 为什么在带优化器的状态或累加器字段上，<code>dtype="auto"</code> 会自动选择 <code>float32</code> 而非更省显存的 <code>bfloat16</code>？</summary>

在自回归解码中，在线训练每步的梯度更新和滑动平均通常幅度微小。在 `bfloat16` 精度下，尾数仅有 7 位，微小的梯度增量或加权量在加到现有权重上时会因精度下溢而被直接截断归零（更新吞噬），导致模型无法学到任何变化甚至数值发散。实测表明，Carry 进位使用 `bfloat16` 的相对误差高达约 $1.4 \times 10^{-1}$，而使用 `float32` 仅为约 $3.7 \times 10^{-4}$。因此框架自动为需要微小累加的字段采用 `float32`，而对只读权重和只写前向字段保留模型精度（如 `bfloat16`）以节省显存。
</details>

<details>
<summary>2. 为什么在第 0 步读取 <code>carry</code> 字段时拿到的张量值始终是全零？</summary>

这是由 pLLM 的版本发布时序规则决定的：任何方法在某一步中读取状态时，看到的都是“当前步开始前已发布的版本”（即第 $n$ 步读取第 $n-1$ 步发布的结果）。而在请求准入的第 0 步开始时，该请求尚未执行过任何训练方法，不存在之前发布的版本，且 carry 字段在准入时默认进行零初始化，因此第 0 步读取到的数值必然是零。
</details>

<details>
<summary>3. 为什么读取 <code>RequestState</code> 的字段时张量自动带有领先的行轴（<code>[rows, *S]</code>），而读取 <code>InstanceState</code> 默认没有？</summary>

在连续批处理中，多个请求被拼接在同一个批次内并发执行。<code>RequestState</code> 为每个请求维护私有的参数副本，因此读取时框架自动根据当前操作所调度的行，为每一行请求对齐并切出其私有的参数切片，形成 <code>[rows, *S]</code> 的张量；而 <code>InstanceState</code> 在全实例范围内只有一份全局唯一的张量副本，不随并发请求数扩缩，因此默认保持声明时的原始维度 <code>[*S]</code>（若需参与按行批处理运算，可显式指定 <code>per_row=True</code> 展开广播行轴）。
</details>

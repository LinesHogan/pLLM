# 组合负载

在很多推理算法的研究中，你往往会同时开发多个独立的模块：例如一个在解码时在线训练轻量网络以引导采样的算法（如 ESamp），和一个在后台监控模型特定层残差范数的探针（如 StreamNorm）。如果要把它们放进同一个推理引擎，你不需要把所有状态定义与张量方法手动合并到一个庞大的类中——在 `pllm` 中，只需编写一个外层 `@dataclass(frozen=True)` 负载，将各个子负载声明为其字段。

`pllm` 会在构建引擎时遍历所有字段，把它们扁平化为一个统一的执行图：自动分配显存、捕获 CUDA graph、并在连续批处理中同时调度它们。每个组件保持命名空间与状态独立，互不污染，并且冲突会在构建期直接报错点名。

---

## 声明外层负载：dataclass 字段

动机：将两个独立的 `pllm.Workload` 组合在一起，同时保持子负载的代码完全不用修改。

写法：把外层负载定义为继承自 `pllm.Workload` 的 `@dataclass(frozen=True)` 类，每个子负载使用 `field(default_factory=...)` 声明为一个字段。

为什么使用 `default_factory`？因为 dataclass 字段如果是类实例，直接赋值默认实例会导致所有外层负载实例共享同一个子负载对象；`default_factory=MyWorkload` 保证每次实例化外层负载时，各个子负载都能获得独立的实例与配置。

```python
from dataclasses import dataclass, field

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)
    def norm(self, r: torch.Tensor = pllm.residual(12)) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)


@dataclass(frozen=True)
class ESampLite(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
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

    def strength(self) -> float:
        return self.beta

    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


@dataclass(frozen=True)
class Both(pllm.Workload):
    esamp: ESampLite = field(default_factory=ESampLite)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

---

## 命名空间隔离与导出路由

当你将 `ESampLite` 和 `StreamNorm` 组合进 `Both` 时，外层负载的字段名（`esamp` 和 `monitor`）成为各个独立组件（component）的命名空间前缀：

1. **状态与方法前缀**：
   在 [`pllm.explain`](explain.md) 的执行计划和显存报告中，所有状态、方法和任务名都会自动加上字段名前缀。例如：
   - `ESampLite` 的状态 `distiller` 变为 `esamp.distiller`；
   - 训练方法 `loss` 变为 `esamp.loss`，引导采样方法 `predict` 变为 `esamp.predict`；
   - 监控方法 `norm` 变为 `monitor.norm`。
   这样即使两个子负载内部恰好使用了同名的方法或状态（例如都叫 `loss` 或 `state`），也不会发生符号冲突。

2. **导出事件路由（`on_export`）**：
   `StreamNorm` 声明了 `@pllm.export(name="stream_norm", ...)`。在引擎内部，导出的全局张量名会被修饰为 `monitor.stream_norm`。但当导出事件从 GPU 传输到 CPU 宿主进程时，`pllm` 会根据事件来自哪个组件剥离掉字段名前缀 `monitor.`，将本地标签 `stream_norm` 传递给对应组件实例的 `monitor.on_export(event)`。
   **收益**：组件作者编写 `on_export` 回调时，无需知道这个组件未来会被挂载在哪个字段名下，原有代码直接复用。

下面在 CPU 上运行 `pllm.explain` 查看执行计划，并用 `pllm.testing.check` 验证调度契约：

```python
# 在 CPU 上查看计划：方法和状态带有 esamp.* 与 monitor.* 前缀
print(pllm.explain(Both(), CONFIG))

# 在 CPU 上运行调度契约检查（包含 component independence 验证）
print(testing.check(Both(), CONFIG, neutral=Both(esamp=ESampLite(beta=0.0))).passed)
```

---

## 组件独立性与状态私有

在 `Both` 中，`monitor` 能不能去读取 `esamp` 的状态？例如在 `monitor` 的方法里声明 `f: nn.Module = pllm.module("esamp.distiller")`？

**答案是严格禁止的。** 如果你在任何输入标记或状态引用里使用了包含点号 `.` 的名称，`pllm` 构建时会抛出异常：
`components are independent; to share state, put both methods in one workload`。

### 为什么强制状态私有？

这是由 `pllm` 的**组件独立性保证**决定的：
- **状态轨迹必须等价于单跑**：任何一个组件在组合负载里的计算过程、状态演化轨迹（每一步的状态张量取值、优化器步数与版本更新），必须与它单独在一个引擎里运行时严格逐位一致。
- **防止隐式时序耦合**：如果组件之间能够跨字段读写状态，就会引入隐式的执行时序依赖（谁先算、谁读到了谁的哪一个中间版本），破坏流水线与 CUDA graph 的重叠调度，使得单独测试组件时通过、拼在一起时却产生难以复现的数值漂移。

### 契约检查：`component independence`

当你对组合负载运行 [`pllm.testing.check`](checking.md) 时，测试器不仅会检查常规的批处理调度契约（行置换、请求隔离、抢占重放等），还会额外执行 **`component independence`（组件独立性）** 检查：
1. 在 CPU 上完整运行一次外层组合负载 `Both`；
2. 依次单独运行每个子组件（`ESampLite` 和 `StreamNorm`）；
3. 逐个请求、逐个 token 比较每个组件在两处的所有输出、状态与导出事件；若存在任何不匹配，抛出 `ContractError`。

> [!TIP]
> **如果两个方法确实需要共享状态怎么办？**
> 如果两个方法需要读写同一个状态（例如一个方法用 `@pllm.train` 训练小权重，另一个方法读这套权重去改写前向残差），说明它们在算法逻辑上本就是同一个算法的两个步骤，而不是两个正交的组件。此时应当将它们声明在同一个 `pllm.Workload` 类内部，而不是拆成两个组件跨界访问。详见[负载的组成](workload-anatomy.md)与[状态](state.md)。

---

## 效果的合并：多个 `@pllm.add_logits`

不同组件产生的结果如何作用到模型上？

- **`@pllm.add_logits` 可以并存**：
  若组件 A 和组件 B 都声明了 `@pllm.add_logits`，它们不会互相冲突。`pllm` 会在采样前处理阶段（sampler epilogue）将各组件输出的 logits 增量按外层 dataclass 的字段声明顺序依次累加到模型的 base logits 上。
- **先加 logits，再做引导采样**：
  全部 `@pllm.add_logits` 的累加在采样引导（`@pllm.guide_sampling`）之前完成。这意味着如果有组件修改了 logits，后续的采样引导方法依然可以在修改后的分布基础上执行过滤与对数重加权。详见[结果去哪](results.md)。

---

## 组合冲突与构建报错

并非所有算法效果都可以无条件叠加。当两个组件对底座引擎的同一物理资源提出互斥需求时，`pllm` **在构建期（`pllm.explain` 或 `pllm.make_llm`）直接拒绝启动并明确点名冲突的双方**，不会让后声明的效果悄悄覆盖先声明的效果。

先看一个故意制造冲突的例子：两个组件都声明了 `@pllm.guide_sampling`。

```python
@dataclass(frozen=True)
class ConflictingBoth(pllm.Workload):
    a: ESampLite = field(default_factory=ESampLite)
    b: ESampLite = field(default_factory=ESampLite)


try:
    pllm.explain(ConflictingBoth(), CONFIG)
except ValueError as e:
    print(f"捕获到预期的构建冲突: {e}")
```

运行这段代码会直接抛出 `ValueError`：
`a.predict and b.predict both declare @pllm.guide_sampling; keep at most one guidance method`。报错明确点名了冲突的双方 `a.predict` 和 `b.predict`。

### 常见组合冲突一览

下表列出常见的组合冲突、构建期报错信息以及背后的原因与解决建议：

| 常见报错（节选） | 冲突原因与解决办法 |
| --- | --- |
| `... and ... both declare @pllm.guide_sampling; keep at most one guidance method` | **每个负载最多声明一个采样引导方法**。采样器后置过滤只支持一套打分与挑选逻辑；若只需微调不同 token 的概率偏置，改用支持多个并存的 `@pllm.add_logits`。 |
| `... trains on ... but ... on ...; pllm funds one training mode per workload, so move one of them to a separate workload` | **每个负载只资助一种训练调度**。连续批处理中的逐请求行在线训练（`rows of running requests`）与全批次宿主训练（`WeightedRowBatch` / `CausalTrainBatch`）具有完全不同的执行图生命周期，不能混在同一个负载内；请将它们拆分到独立的引擎中运行。 |
| `a @pllm.train_in_forward method drives every projection hook of its workload, so it does not yet combine with a pllm.model.Adapter; train the adapter's state in that method or use two workloads` | **就地前向更新与 Adapter 互斥**。`@pllm.train_in_forward` 接管了其目标投影的所有钩子进行就地前向更新，无法与独立的 `Adapter` 叠加；直接在该方法内训练该状态，或拆为两个负载。 |
| `... components are independent; to share state, put both methods in one workload` | **组件之间严格禁止跨界读写状态**。组件状态必须私有以保证独立运行与组合运行的轨迹逐位一致；需要共享状态的方法必须写在同一个 `Workload` 类中。 |
| `... reads block or model outputs, which a @pllm.custom_forward does not produce; use pllm.residual or pllm.layer_logits to read each layer's last visited stream` | **自定义前向结构与块输出读取互斥**。`@pllm.custom_forward` 可能跳层或循环执行某些层，各层的固定块输出 `pllm.hidden` 没有唯一定义；改用 `pllm.residual` 或 `pllm.layer_logits` 读取每层最后一次访问的激活流。 |

完整的规则矩阵与所有被拒绝的边界组合详见[组合规则](../reference/composition-rules.md)与[报错目录](../reference/errors.md)。

---

## 自测

1. 组合负载 `Both` 中包含监控组件 `monitor: StreamNorm`，其内部方法声明了 `@pllm.export(name="stream_norm", ...)`。在宿主端消费事件时，`monitor.on_export(event)` 接收到的 `event.tensor_name` 是 `"monitor.stream_norm"` 还是 `"stream_norm"`？为什么？

<details>
<summary>答案</summary>

是 `"stream_norm"`。`pllm` 的导出分发机制在将 GPU 导出的事件送入组件自身的 `on_export` 回调前，会自动剥离外层字段名前缀 `monitor.`。这样设计的目的是实现组件复用：子组件编写和单测时无需感知外部挂载它的字段名。
</details>

2. 研究员想要让组件 A 计算出每一行的分类特征，并将其存入状态 `feat`，再让组件 B 读取 `feat` 生成对应的 logits 偏置。他们把 A 和 B 写成了两个独立的子负载拼入外层 dataclass，但在 B 中声明 `pllm.parameter("a.feat", "value")` 时报错。为什么 `pllm` 不支持这种写法？正确的实现方式是什么？

<details>
<summary>答案</summary>

`pllm` 强制组件独立性：每个组件的状态轨迹必须与它单独运行时完全一致，跨组件读写状态会引入隐式时序依赖并破坏单测契约（`component independence` 检查会失败）。

正确做法：由于状态 `feat` 本就是这两个方法紧密共享的核心中间量，它们在算法语义上属于同一个算法。应该把产生 `feat` 的方法和读取 `feat` 的方法写在同一个 `pllm.Workload` 类中。
</details>

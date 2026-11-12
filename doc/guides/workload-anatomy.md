# 负载的组成

pLLM 把**算法声明**与**引擎调度**拆开（为什么不用 hook，见[核心概念](../explain/core-concepts.md)）：你写一个 `pllm.Workload` 子类，用声明回答三个问题——**读什么**、**存什么**、**结果去哪**——再在方法体里写普通的 PyTorch 张量数学。pLLM 在构建引擎时只读一次这些声明，自动规划每个方法在什么时机、哪条 CUDA stream 上运行，替你管理每个请求的状态显存并将计算捕获进 CUDA graph。

先看一个最小的可运行负载：不存任何状态，只读取模型第 12 层之后的残差流，算出每个请求当前 token 的 L2 范数并导出到 CPU。

```python
from dataclasses import dataclass, field

import torch
from torch import nn

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


class StreamNorm(pllm.Workload):
    @pllm.export(name="stream_norm", max_rows=8, max_cols=1)  # 结果去哪：导出到宿主机的 on_export
    def norm(
        self,
        r: torch.Tensor = pllm.residual(12),                  # 读什么：第 12 层之后的残差流
    ) -> torch.Tensor:
        return r.float().norm(dim=-1, keepdim=True)           # [rows, 1]：按特征维（dim=-1）求范数


print(pllm.explain(StreamNorm(), CONFIG))
```

这段代码里没有注册句柄，也没有运行时回调。如果算法还需要每个请求独立维护一份可训练的状态（例如在线蒸馏 ESamp：每个请求边生成边用浅层激活预测深层激活，并用浅层预测引导采样），只需在同一个类里加上**存什么**的声明：

```python
@dataclass(frozen=True)
class Distill(pllm.Workload):
    beta: float = 0.5

    # 存什么：每个请求独立拥有一份小网络和 AdamW 优化器状态
    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # 读什么：默认值声明输入来源；结果去哪：@pllm.train 最小化返回的逐行 loss
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)  # [rows]

    # 从实例读取配置的方法，供装饰器引用
    def strength(self) -> float:
        return self.beta

    # 结果去哪：用预测的 hidden 向量按强度 beta 引导采样
    @pllm.guide_sampling(beta=strength)
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())  # [rows, hidden]
```

下面逐一拆解负载类的结构、三类声明、方法体的编写规则以及常见的配置与组合方式。如果你是第一次运行带负载的引擎，也可以先配合 [教程 00：快速上手](../tutorials/00_quickstart.ipynb) 边跑边看。

---

## 负载类与三类声明

一个负载是 `pllm.Workload` 的子类。当负载带有超参数或开关（如上面的 `beta: float = 0.5`）时，把它写成 `@dataclass(frozen=True)`。

为什么要冻结（`frozen=True`）？因为 pLLM 在构建引擎时一次性读取负载实例上的配置并生成执行计划；引擎启动后再去修改 Python 实例的属性，不会改变已经在 GPU 上编译好的执行图。使用冻结的 dataclass 既能防止运行期误改属性，又能随时用 `Distill(beta=0.0)` 或 `dataclasses.replace(workload, beta=0.0)` 构造一个零强度的对照实例，交给 `pllm.testing.check` 检查“关闭算法时是否严格等价于原模型”。

负载里的所有声明都归为三类：

| 声明回答的问题 | 写法 | 可用的公开名字 | 详细指南 |
| --- | --- | --- | --- |
| **读什么**（方法的数据来源） | 方法参数的默认值 | 模型激活与 logits：`pllm.hidden`、`pllm.residual`、`pllm.final_hidden`、`pllm.layer_logits`、`pllm.attention`<br>状态与元数据：`pllm.parameter`、`pllm.module`、`pllm.version`、`pllm.index`、`pllm.carry`<br>Token 与采样反馈：`pllm.input_tokens`、`pllm.sample_index`、`pllm.base_logits`、`pllm.sampled_tokens`<br>同一步其他方法的结果：`pllm.result`<br>模型层内位置与投影操作数：`pllm.depth`、`pllm.value`、`pllm.operand`<br>宿主训练批次与可微模型：`pllm.batch`、`pllm.causal_model` | [读取模型内部](reading.md)<br>[状态](state.md) |
| **存什么**（跨步或跨请求持有的张量） | 类属性，或标注了返回类型的 `@property` | 状态作用域：`pllm.RequestState`（每个请求一份独立副本）、`pllm.InstanceState`（全引擎共享一份副本）、`pllm.PositionState`<br>字段描述：`pllm.Field`、`pllm.Dimensions`<br>初值 `init=`：`pllm.Zeros`（默认）、`pllm.Computed`、`pllm.Pretrained`、`pllm.ModelCheckpoint`、`pllm.FromModule`、`pllm.Fed`、`pllm.Imported`<br>优化器 `optimizer=`：`pllm.AdamW`、`pllm.SGD` | [状态](state.md)<br>[在线训练](training.md) |
| **结果去哪**（返回值或状态如何生效） | 方法装饰器，或模型改动类属性 | 观测与采样：`@pllm.export`、`@pllm.add_logits`、`@pllm.guide_sampling`<br>状态更新与训练：`@pllm.write`、`@pllm.train`、`@pllm.differentiate`<br>模型层内改动：`pllm.model.ActivationPatch`、`pllm.model.Adapter`、`@pllm.train_in_forward`<br>前向结构改写：`@pllm.custom_residual`、`@pllm.custom_attention`、`@pllm.custom_forward` | [结果去哪](results.md)<br>[改模型](model-edits.md)<br>[改 forward 结构](forward-structure.md)<br>[宿主循环与批训练](host-loops.md) |

全部公开接口的签名与说明见 [API 参考](../reference/api.md)。写这三类声明时有三条基本规则：

1. **除 `self` 外的每个输入参数都必须写标记默认值。** Python 参数名（如 `shallow`、`deep`）只是方法体内部使用的局部变量名，pLLM 不会根据参数名猜测你想读哪一层；默认值（如 `= pllm.hidden(0)`）才是给规划器看的声明，类型注解（`: torch.Tensor`）则告诉 IDE 和类型检查器方法体拿到的值是什么类型。（唯一的例外是 `@pllm.custom_residual` 的流张量 `x, y`、`@pllm.custom_attention` 的 `q, kv` 以及 `@pllm.custom_forward` 的 `layers`，它们作为打头的位置参数传入，后面的参数依然使用标记默认值。）
2. **产出结果的方法必须挂效果装饰器。** 一个普通方法只返回张量、不挂 `@pllm.export` / `@pllm.train` 等装饰器时，pLLM 不知道返回值该送往采样器、导出队列还是优化器，因此不会单独调度它（除非它被另一个方法通过 `pllm.result("方法名")`、`rows="方法名"` 引用，或显式挂了 `@pllm.on`）。若整个负载没有任何效果声明，构建时会直接报错提醒。
3. **同一个方法上同一种类的 pLLM 装饰器最多挂一个。** 不要在一个方法上同时叠放 `@pllm.export` 和 `@pllm.train`——如果允许叠放，外层装饰器会覆盖内层装饰器的效果属性，导致其中一个效果静默丢失。因此 pLLM 在声明检查时直接拒绝叠放效果；需要把同一份计算结果用于两个效果时，拆成两个方法并用 `pllm.result("方法名")` 复用结果即可。

---

## 方法体里写什么

声明好输入和效果后，方法体内部就是普通的 PyTorch 张量运算（如果需要接入自写的 Triton 算子，可通过 `torch.library.custom_op` 调入，见[自写算子](custom-ops.md)）。由于方法在连续批处理和 CUDA graph 下运行，方法体需要遵守四条边界：

### 1. 第 0 维是请求行：只在特征维（`dim=-1`）上归约，不要跨行（`dim=0`）计算

方法拿到的激活张量、请求状态参数和采样行张量，第 0 维（`[rows, ...]`）都是**当前步的请求行**（每个运行中的请求一行，或采样器每抽取一个 token 一行）。

- **原因**：在连续批处理下，每一步有哪些请求同时在跑、哪个请求排在第几行，完全由引擎调度器根据请求的到达、完成和抢占动态决定。如果你在方法里写了 `h - h.mean(dim=0)`、`softmax(dim=0)`、BatchNorm 或按行排序，一个请求的计算结果就会混入同一批次里其他无关请求的激活值，批次大小或排列顺序一变，输出就会跟着变。
- **做法**：归约一律沿着特征维度写（如 `.mean(dim=-1, keepdim=True)`、`.norm(dim=-1)`），让每一行的输出只依赖该行自己的输入和自己的请求状态。`pllm.testing.check` 会通过打乱行顺序（`row permutation`）和扰动邻居请求（`request isolation`）专门检查这一条。

### 2. 输入与状态参数是借用的只读视图：不要原地修改

不要对方法参数做原地（in-place）写入，比如 `h.mul_(2)`、`w.add_(1)` 或 `torch.matmul(..., out=h)`。

- **原因**：方法收到的激活张量和状态张量是框架借给当前方法的只读视图，同一步里的其他方法可能也要读同一块激活，而状态池里的旧版本页面还被其他正在读取该版本的任务引用。原地修改会直接篡改共享内存，绕过状态的版本发布机制。
- **做法**：方法体里只写返回新张量的非原地运算（如 `h * 2`、`w + 1`）；如果确实需要在局部张量上做切片赋值，先 `.clone()` 一份副本。对状态的任何持久化修改都通过 `@pllm.write` 或 `@pllm.train` 发布成新版本（见[状态](state.md)与[结果去哪](results.md)）。

### 3. 不要做 GPU→CPU 同步，也不要根据张量的值写 Python 条件分支

方法体里不能出现 `.item()`、`.tolist()`、`.cpu()`、`print(tensor)`，也不能写 `if h.norm() > 1:` 或 `while not converged:` 这类依赖张量值的 Python 分支与循环。

- **原因**：有两个层面的原因。第一，在构建引擎（以及运行 `pllm.explain`）时，pLLM 会先用 PyTorch 的 `meta` 张量（只有形状和 dtype、没有实际数据存储）追踪一遍方法体来推导形状与显存预算；在 `meta` 张量上调用 `.item()` 或做布尔判断会立刻触发 `RuntimeError: Tensor.item() cannot be called on meta tensors`。第二，在 GPU 运行期，方法会被捕获进 CUDA graph，并且常常放在旁路 CUDA stream 上与模型前向重叠执行；CUDA graph 只能记录固定的算子序列，无法录制 Python 的动态分支，而 GPU→CPU 同步则会阻塞主机线程并破坏流捕获。
- **做法**：
  - 条件逻辑用布尔掩码和 `torch.where(mask, a, b)` 表达；
  - 想在运行期查看张量数值，用 `@pllm.export` 异步导出到 CPU 的 `on_export` 回调；
  - **允许的 Python 循环**：以负载配置字段为上界的有界 `for` 循环（例如 `for _ in range(self.loops):`）完全可以使用。因为 `self.loops` 在构建时已经是固定的 Python 整数，追踪时循环会直接展开成固定步数的算子序列。

### 4. 注意张量精度与返回形状

- **精度（dtype）**：模型内部的激活输入（`pllm.hidden`、`pllm.residual`、`pllm.final_hidden`）默认以模型的计算精度（通常为 `bfloat16`）存储和捕获，而 `pllm.layer_logits` 为 `float32`。做平方误差、范数或累加时，先调用 `.float()` 转成 `float32`，可以避免 `bfloat16` 下溢或精度损失。
- **返回形状**：
  - 按请求行训练的 `@pllm.train` 方法必须返回一维的 `[rows]` loss 张量（每行一个标量），**不要**在方法末尾自己做 `.mean()` 归约成 0 维标量。原因是同一个批次里的不同行可能属于不同请求的状态（且捕获表中的填充行或未选中行需要由框架先用掩码剔除），pLLM 会按状态对有效行做归约（默认 `reduction="mean"`，可选 `"sum"`）。相反，读取 `pllm.batch(...)` 的宿主批训练方法针对整个外部批次做一步更新，由方法自己返回单个标量 loss（见[在线训练](training.md)与[宿主循环与批训练](host-loops.md)）。
  - `@pllm.add_logits` 必须返回 `[rows, vocab_size]`；`@pllm.guide_sampling` 必须返回 `[rows, hidden_size]`（框架再用模型自己的 LM head 把它投影到词表空间）。

---

## 从实例读取配置

写可复用的负载时，我们常把层号、引导强度 `beta`、低秩维度 `rank` 等设为 dataclass 字段。把这些字段传给装饰器、输入标记或状态声明时，有两个高频易错点。

### 为什么不能在装饰器里直接写 `beta=beta`？

观察下面这种错误写法：

```text
@dataclass(frozen=True)
class BadDistill(pllm.Workload):
    beta: float = 0.5
    ...
    @pllm.guide_sampling(beta=beta)   # 这里的 beta 是类定义时的默认值 0.5
    def predict(self, ...): ...
```

- **原因**：Python 在执行 `class` 定义体时就会求值 `@pllm.guide_sampling(beta=beta)` 的参数，此时还没有任何 `self` 实例存在。名字 `beta` 拿到的是类属性默认值 `0.5`。当你之后实例化 `BadDistill(beta=0.0)` 传给 `testing.check(..., neutral=BadDistill(beta=0.0))` 时，装饰器里绑定的仍然是 `0.5`，导致零强度检查报错：`[neutral] guide_sampling still declares a sampler stage; set its beta to zero.`。
- **做法**：任何装饰器参数或输入标记参数（如 `@pllm.guide_sampling(beta=...)`、`pllm.hidden(layer)`、`pllm.layer_logits(layers)`、`@pllm.export(every=...)`、`@pllm.train(steps=...)`）只要依赖实例字段，就传一个**接收负载实例并返回该配置值的方法**（或者 `lambda self: self.beta`）。pLLM 会在绑定具体实例时调用一次这个方法。写成带返回类型注解的方法（如 `def strength(self) -> float:`）还能让 IDE 支持跳转定义，并让类型检查器校验返回类型。

### 为什么状态写成 `@property` 时必须标注返回类型？

当状态的形状依赖实例字段（如 `self.rank`）时，类属性拿不到 `self`，需要把状态写成 `@property`。此时**必须**给 `@property` 加上返回类型注解（如 `-> pllm.RequestState` 或 `-> pllm.InstanceState`；模型改动属性则标 `-> pllm.model.Adapter`、`-> pllm.model.ActivationPatch` 或 `-> pllm.PositionState`）。如果类里还定义了其他普通辅助 `@property`，同样要标上它们各自的返回类型（如 `-> int`）。

- **原因**：pLLM 在分析负载类时，需要在不盲目执行任意用户 getter 代码的前提下，先识别出哪些成员是状态或模型改动声明，并对方法与函数做依赖快照，随后才按顺序求值状态声明。通过检查 `@property` 的返回类型注解（包含 `from __future__ import annotations` 下的字符串注解），框架无需运行未知的 property 代码就能准确认出声明属性。

下面的例子同时展示了“用方法向输入标记传递层号 `self.layer`”和“用带返回类型注解的 `@property` 声明依赖 `self.rank` 的状态”：

```python
@dataclass(frozen=True)
class ConfigurableProbe(pllm.Workload):
    layer: int = 12
    rank: int = 4

    @property
    def head(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(lambda m: nn.Linear(m.hidden_size, self.rank))
        )

    def source_layer(self) -> int:
        return self.layer

    @pllm.export(name="probe", max_rows=8, max_cols=4)
    def score(
        self,
        r: torch.Tensor = pllm.residual(source_layer),
        head: nn.Module = pllm.module("head"),
    ) -> torch.Tensor:
        return head(r.float())


print(testing.check(ConfigurableProbe(layer=6, rank=4), CONFIG).passed)
```

---

## 方法什么时候运行

你通常不需要手动指定方法的运行时间，因为每个效果装饰器都有与它语义匹配的默认阶段（`pllm.Phase`）：

| 效果 | 默认阶段 | 在单步中的时间窗口 |
| --- | --- | --- |
| `@pllm.train_in_forward`、`@pllm.custom_residual`、`@pllm.custom_attention` | `Phase.IN_FORWARD` | 模型前向传播执行期间 |
| `@pllm.export` | `Phase.AFTER_FORWARD` | 模型前向完成、状态发布之后，采样之前 |
| `@pllm.add_logits`、`@pllm.guide_sampling` | `Phase.BEFORE_SAMPLE` | 模型前向完成之后、采样器抽取 token 之前 |
| `@pllm.write`、`@pllm.train` | `Phase.AUTOMATIC` | 由规划器按方法读的输入安排：输入一就绪就运行；按请求行训练在采样之后、于旁路 stream 上运行，与下一步前向重叠 |

### `pllm.Phase` 是步内的时间窗口，不是运行时钩子

`pllm.Phase` 共有六个枚举值：`BEFORE_FORWARD`、`IN_FORWARD`、`AFTER_FORWARD`、`BEFORE_SAMPLE`、`AFTER_SAMPLE`、`AUTOMATIC`。

它们表示引擎单步时间轴上的**窗口**，而不是运行时触发的回调函数。构建引擎时，pLLM 把每个方法放进所属窗口，并根据方法之间的数据依赖（例如方法 B 通过 `pllm.result("A")` 读取方法 A 的输出，或读取某块状态的发布版本）自动排出拓扑顺序。

这也是为什么**一个值只能在产出它的阶段或更晚的阶段被读取**：
- 当前步的 `pllm.hidden`、`pllm.residual`、`pllm.final_hidden` 只有在模型前向完成后才存在，不能在 `BEFORE_FORWARD` 中读取；
- `pllm.base_logits()`（未经负载修改的模型原始 logits）和 `pllm.sampled_tokens()`（本步采样抽出的 token）只有在采样完成后才存在，只能在 `@pllm.train` 或显式标了 `@pllm.on(pllm.Phase.AFTER_SAMPLE)` 的方法中读取。若要在采样前读取模型 logits（例如在 `@pllm.add_logits` 里），应使用 `pllm.layer_logits((-1,))`。

当你需要把一个方法移到非默认窗口，或者让一个不带效果装饰器的方法在指定窗口运行（供其他方法通过 `pllm.result` 读取）时，给方法加上 `@pllm.on(phase)`（只接收一个 `pllm.Phase` 参数，或返回 `pllm.Phase` 的设置方法）：

```python
class RunningMean(pllm.Workload):
    stats = pllm.RequestState({"mean": pllm.Field(lambda width, _: (width,))})

    @pllm.export(name="mean_norm", max_rows=8, max_cols=1)
    def report(self, mean: torch.Tensor = pllm.parameter("stats", "mean")) -> torch.Tensor:
        return mean.norm(dim=-1, keepdim=True)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.write(state="stats", fields=("mean",))
    def update(
        self,
        h: torch.Tensor = pllm.final_hidden(),
        mean: torch.Tensor = pllm.parameter("stats", "mean"),
    ) -> tuple[torch.Tensor]:
        return (0.9 * mean + 0.1 * h.float(),)


print(testing.check(RunningMean(), CONFIG).passed)
```

---

## 选择器：只让一部分行生效

回到开头的 `Distill` 负载：新请求刚进入解码的第 0 步时，它的 `distiller` 小网络还没有训练过任何一步（更新次数 `version == 0`），此时用随机初值去引导采样只会引入噪声。我们希望“只有当一个请求的状态至少更新过 1 次之后，才对它开启采样引导”。

因为方法体在 CUDA graph 里运行，不能写 `if version > 0:`，也不能动态删减张量的行（那会改变张量形状）。pLLM 的做法是**行选择器（`rows="方法名"`）**：定义一个读取标记输入、返回一维布尔掩码 `bool [rows]` 的方法，再把方法名传给效果装饰器的 `rows=`。

```python
@dataclass(frozen=True)
class GatedDistill(pllm.Workload):
    beta: float = 0.5

    distiller = pllm.RequestState(
        init=pllm.FromModule(
            lambda m: nn.Sequential(nn.Linear(m.hidden_size, 4), nn.Linear(4, m.hidden_size))
        ),
        optimizer=pllm.AdamW(1e-3),
    )

    # 选择器：返回 bool [rows] 掩码，只选中已经更新过至少一次的请求行
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

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

    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())
```

选择器方法（`trained`）同样被编译成图里的一个普通张量任务，掩码全程留在 GPU 上：在 `@pllm.guide_sampling` 和 `@pllm.add_logits` 中，掩码为 `False` 的行权重为 0，采样器保持该行原有的 logits 不变，无需任何 GPU→CPU 同步。

### `rows=` 在哪里可用，在哪里会被拒绝？

并非所有装饰器都支持 `rows=`。下表列出各效果对行选择器的支持情况与原因：

| 效果 | 是否支持 `rows=` | 行为与原因 |
| --- | --- | --- |
| `@pllm.add_logits(rows="...")`<br>`@pllm.guide_sampling(beta=..., rows="...")` | 支持 | 未选中的行在采样器端保持原始 logits 不变。 |
| 配合模型内效果（如 `Adapter`、`ActivationPatch`，且无 `@pllm.guide_sampling`）或配合 `@pllm.add_logits` 的按请求行 `@pllm.train` | 支持 `rows=` 与 `update_rows=` | 方法仍在所有行上求值，`rows=` 掩码决定哪些行的 loss 计入归约；`update_rows=` 决定哪些状态执行优化器更新并发布新版本。 |
| **独立**的按请求行 `@pllm.train`（无模型内效果且无 `@pllm.add_logits`），或**与 `@pllm.guide_sampling` 并存**的 `@pllm.train` | **不支持**（声明期报错） | 这两条路径走的是采样后的无条件更新路径（在旁路 stream 上与下一步模型前向重叠）：每一步都会无条件推进每个运行中请求的状态优化器并发布新版本，不实现按行掩码的条件跳过。 |
| 读取 `pllm.batch(...)` 的宿主批训练 | **不支持** | 宿主批训练在批次整体上做一步更新；需要跳过某些样本时，将 `WeightedRowBatch` 或 `WeightedCEBatch` 中对应行的 `weights` 设为 `0`。 |
| `@pllm.write` | **不支持** | `@pllm.write` 每步写入该状态的全部行；若想让部分行保持原值，可在方法内读入旧值并用 `torch.where(mask, new_val, old_val)` 返回。 |

> [!TIP]
> **在独立训练或配 `@pllm.guide_sampling` 的训练里，如何屏蔽某些行的影响？**
> 1. 如果目的是“不让未训练好的状态影响生成”，像上面的 `GatedDistill` 一样，把 `rows="trained"` 挂在 `@pllm.guide_sampling` 上；
> 2. 如果目的是“让某些行在当前步不产生梯度”，将训练装饰器设为 `@pllm.train(state=..., reduction="sum")`，并在 `loss` 方法内把这些行的 loss 乘以 `0`（注意不要用默认的 `reduction="mean"`，因为乘以 `0` 的行仍会被算进均值分母，稀释同一状态其余行的梯度）。

---

## 把多个负载组合成一个

如果你已经写好了在线蒸馏负载 `GatedDistill` 和激活监控负载 `StreamNorm`，想在同一个引擎里同时运行它们，不需要把两个类的方法复制粘贴到一起——只需在一个外层 dataclass 负载里把它们声明成字段：

```python
@dataclass(frozen=True)
class Combined(pllm.Workload):
    distill: GatedDistill = field(default_factory=GatedDistill)
    monitor: StreamNorm = field(default_factory=StreamNorm)
```

每个字段是一个独立的组件：状态、方法和导出名带上字段名前缀（`explain` 里显示为 `distill.loss`、`monitor.norm`），组件之间不能互读状态，真正冲突的组合（例如两个组件都引导采样）在构建时被拒绝并点名双方。为什么这样设计、哪些组合被拒绝，见[组合负载](composition.md)。

---

## 从写完到上卡：开发循环

写负载最快的方式不是每次都启动完整的 GPU 引擎，而是先在 CPU 上花几秒钟看完执行计划、跑完调度契约检查，最后再上 GPU：

1. **写负载**：脚本、notebook 单元格或 `.py` 模块里定义都可以；开启多进程引擎（`ModelConfig(multiprocessing=True)`）时，负载字段要能 pickle（不放锁、打开的文件等句柄），worker 子进程据此重建负载并重新校验声明（详见[上生产](../howto/production.md)）。
2. **在 CPU 上看成本（`pllm.explain`）**：只读取模型的 `config.json` 元数据，不加载模型权重、不分配 GPU 显存，直接打印模型是否保持 CUDA graph、每个方法在哪个时机和哪条 stream 上运行、状态预留多少字节、方法内是否有 GPU→CPU 同步。详见[看成本：pllm.explain](explain.md)。
3. **在 CPU 上查契约（`pllm.testing.check`）**：用真实的规划器和执行器在 CPU 上跑三个请求的固定调度，验证行重排、请求隔离、混合预填充与解码、分块预填充、抢占重算、版本可见性（`publish_lag`）、组件独立性以及零强度等价（`neutral=`）。详见[检查正确性](checking.md)。
4. **在 GPU 上运行（`pllm.make_llm`）**：加载模型生成，并通过 `llm.diagnostics()` 检查实际运行指标。
5. **在 GPU 上查完整契约（`pllm.testing.check_on_gpu`）**：验证连续解码步零同步、KV 驱逐重算以及 `neutral=` 与原生 vLLM 逐位一致（模型层内改动如 `ActivationPatch`、`Adapter`、`@pllm.custom_residual` 在 CPU 检查中会标记为 `SKIP`，需要这一步在 GPU 上验证）。

下面在 CPU 上对 `GatedDistill` 和组合后的 `Combined` 运行第 2、3 步：

```python
print(pllm.explain(GatedDistill(), CONFIG))
print(testing.check(GatedDistill(), CONFIG, neutral=GatedDistill(beta=0.0)))
print(testing.check(Combined(), CONFIG, neutral=Combined(distill=GatedDistill(beta=0.0))).passed)
```

CPU 检查通过后，在 GPU 上运行第 4、5 步：

```python
# check-docs: skip: model generation and GPU contract check require CUDA
from vllm import SamplingParams

GPU_CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8, gpu_memory_utilization=0.3)

with pllm.make_llm(GPU_CONFIG, workload=GatedDistill()) as llm:
    outputs = llm.generate(["Explain gradient descent:"], SamplingParams(n=4, max_tokens=32, seed=0))
    report = llm.diagnostics()
print(repr(report.eager_reason), report.update_count, report.trained_rows)

print(testing.check_on_gpu(GatedDistill(), GPU_CONFIG, neutral=GatedDistill(beta=0.0)))
```

---

## 常见报错

这一页的写法被拒绝时的报错原文、原因和改法，见报错目录的[声明与输入](../reference/errors.md#声明与输入)、[方法体](../reference/errors.md#方法体)、[testing.check 的 ContractError](../reference/errors.md#testingcheck-的-contracterror)几节。

---

## 自测

1. 为什么在方法体里写 `hidden - hidden.mean(dim=0, keepdim=True)` 能通过 Python 语法检查，却会在 `pllm.testing.check` 时报 `mixes rows`？
<details>
<summary>答案</summary>

张量的第 0 维是当前步打包在一起的不同请求行（`[rows, hidden]`）。`dim=0` 求均值会把同一批次里不同请求的激活混在一起，导致一个请求的输出取决于此时有哪些其他请求恰好和它同批运行。应改为沿着特征维归约：`hidden - hidden.mean(dim=-1, keepdim=True)`。
</details>

2. 在 `@dataclass(frozen=True)` 负载类中，为什么 `@pllm.guide_sampling(beta=beta)` 会让 `neutral=MyWorkload(beta=0.0)` 失效，应该怎么改？
<details>
<summary>答案</summary>

`class` 定义体执行时还没有实例 `self`，直接写 `beta=beta` 会把类属性的默认值（如 `0.5`）固定进装饰器里，之后构造 `MyWorkload(beta=0.0)` 也不会改变它。应在类里定义方法 `def strength(self) -> float: return self.beta`，并在装饰器里写 `@pllm.guide_sampling(beta=strength)`，让 pLLM 在绑定该实例时读取 `self.beta`。
</details>

3. 想在每步解码时把一个新张量存入每个请求自己的状态里，能不能在普通方法里直接对 `w: torch.Tensor = pllm.parameter("state", "w")` 调用 `w.copy_(new_val)`？为什么？
<details>
<summary>答案</summary>

不能。`pllm.parameter` 拿到的是只读视图，同一状态版本可能正被同一步的其他方法读取，原地调用 `w.copy_()` 会破坏版本隔离并在 `testing.check` 中触发 `mutates borrowed input` 报错。应当使用 `@pllm.write(state="state", fields=("w",))` 装饰方法并返回 `(new_val,)`，由框架原子地发布为状态的下一个版本。
</details>

---

## 下一步

- 了解模型每一层能读到哪些激活与 logits → [读取模型内部](reading.md)
- 声明每请求状态、全实例状态、初值来源与 `key_fields` 共享 → [状态](state.md)
- 导出观测、修改 logits、引导采样与写回状态 → [结果去哪](results.md)
- 在解码途中训练状态、控制梯度边与 `publish_lag` → [在线训练](training.md)
- 用状态直接改动层激活或线性投影权重 → [改模型](model-edits.md)
- 改写残差连接、注意力读法或循环层深度 → [改 forward 结构](forward-structure.md)

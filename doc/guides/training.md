# 在线训练

在标准推理框架中，大模型推理被严格设计为只读的前向传播。但在很多前沿算法中，模型需要在**生成的同时进行学习**：每个请求携带一小份私有参数（如浅层到深层的蒸馏网络、测试时训练 TTT 适配器、共享预测头等），每生成一个 token 就根据前向信号计算一次损失并更新参数，更新后的参数随即参与后续 token 的生成。

在连续批处理（continuous batching）和 CUDA Graph 深度优化的推理引擎中手动实现这件事极其困难：
- 连续批处理会将不同请求的当前 token 拼接在第 0 维，张量每一行的生命周期和归属各不相同；
- CUDA Graph 要求固定的显存地址和算子执行拓扑，传统的动态计算图无法直接重放；
- 推理阶段默认关闭自动求导（autograd），在模型外部手动调用反向传播会打断推理流水线；
- 如果反向传播与优化器更新串行阻塞在下一步前向之前，推理吞吐将严重下降。

pLLM 允许你以纯声明的方式定义在线训练：在状态上声明优化器（如 `optimizer=pllm.AdamW(...)`），在方法上挂 `@pllm.train(state=...)` 并返回逐行损失向量。pLLM 负责掩码填充、按状态求导归约、调用优化器以及发布新版本，并将整条训练管线捕获进 CUDA Graph，置于旁路 CUDA stream 上与模型主干前向异步重叠。

---

## 最小示例：在线浅层蒸馏

下面是一个完整的最小可运行负载 `ShallowDistiller`（源自教程与参考实现）：每个请求拥有一份私有的小网络 `distiller`，在 decode 阶段用第 0 层的隐藏状态预测最后一层的隐藏状态，并通过 `@pllm.guide_sampling` 将浅层预测作为采样引导。

```python
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

import pllm
from pllm import testing

CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)


@dataclass(frozen=True)
class ShallowDistiller(pllm.Workload):
    beta: float = 0.5
    rank: int = 4

    # 存什么：每个请求拥有一份私有小网络和独立的 AdamW 优化器
    @property
    def distiller(self) -> pllm.RequestState:
        return pllm.RequestState(
            init=pllm.FromModule(
                lambda m: nn.Sequential(nn.Linear(m.hidden_size, self.rank), nn.Linear(self.rank, m.hidden_size))
            ),
            optimizer=pllm.AdamW(1e-3),
        )

    # 状态资格判断：只对已接收过更新的状态开启采样引导
    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    # 训练方法：返回每一行的标量损失 [rows]
    @pllm.train(state="distiller")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (f(shallow.float()) - deep.float()).square().mean(-1)

    # 从实例动态读取参数
    def strength(self) -> float:
        return self.beta

    # 采样引导：将浅层预测投影到词表，推离已知特征
    @pllm.guide_sampling(beta=strength, rows="trained")
    def predict(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        f: nn.Module = pllm.module("distiller"),
    ) -> torch.Tensor:
        return f(shallow.float())


check_result = testing.check(ShallowDistiller(), CONFIG, neutral=ShallowDistiller(beta=0.0))
print("publish_lag" in check_result.passed)
```

当你运行上述代码时，pLLM 在背后自动完成了以下工作：
1. **自动掩码与隔离**：自动屏蔽未激活的填充槽位，确保每个请求仅根据自己的输入更新自己的参数；
2. **局部求导**：在方法内部构建一阶计算图，从损失反向传播至 `distiller` 参数；
3. **优化器步进**：针对每个请求的状态执行 AdamW 更新；
4. **旁路发布与重叠**：训练任务被安排在采样之后的旁路 CUDA stream 上运行，并整包录制入 CUDA Graph，不打断主干流的执行。

---

## 返回值约定：逐行损失 `[rows]`

在线训练方法的核心约定是：**必须返回形状为 `[rows]` 的一维张量，每一行代表对应请求在当前步的损失**。

不要在 Python 方法体里将行轴压平（例如调用 `.mean()` 或 `.sum()` 返回标量）。

- **原因**：在连续批处理中，输入张量的第 0 维动态拼接了当前批次所有活跃请求的 token。框架必须获取每一行独立的损失数值，才能根据当前的请求拓扑过滤无效填充行、对每个状态独立实施归约，并保持请求之间的隔离性。如果你在 Python 中直接求全局均值，不同请求的损失就会在求导前混在一起，彻底破坏请求间的数学隔离。
- **违反后果**：若方法返回标量，`testing.check` 和执行器会在形状追踪时直接抛出异常：`training methods return one loss per row; pLLM reduces the selected rows per state`。

---

## 状态上的优化器与执行步调

在 pLLM 中，**优化器直接声明在状态上，而不是装饰器上**：

```python
head = pllm.RequestState(
    {"w": lambda m: (m.hidden_size,)},
    optimizer=pllm.AdamW(1e-3),
)
```

- **原因**：优化器的一阶动量、二阶动量以及步数计时器在生命周期上严格隶属于状态本身。对于 `pllm.RequestState`，当请求结束退出时，其参数与优化器内部的动量槽位会一同被显存池释放回收；对于跨请求共享的 `pllm.InstanceState`，优化器动量则在引擎整个生命周期内全局保持。
- **匹配规则**：在运行中的请求行上训练时（不包含宿主批次字段），状态必须声明 `optimizer=pllm.AdamW(lr)`。若未声明优化器或声明类型不匹配，在负载构建期就会报错提示。
- **训练哪种状态**：请求行训练通常训练 `pllm.RequestState`（每个请求一份）。也可以在请求行上训练所有请求共享的 `pllm.InstanceState`（像 `pllm.workloads.OverRIDEWorkload` 那样）：每步所有请求的行一起给这一份状态做一次更新，所以一个请求的结果会受同批其他请求影响。

### 引擎步进与分块依赖

在执行步调上，pLLM 遵循严格的物理规则：**每个状态在每个引擎步（engine step）执行且仅执行一次优化器更新**。

- **Decode 阶段**：每个活跃请求每步生成 1 个 token，对应 1 行输入，因此每个请求状态在每一步推进一次优化器。
- **Prefill 阶段（Prompt 行）**：当开启分块预填充（chunked prefill）时，长 Prompt 会被调度器拆分成多个分块跨步处理。每个引擎步中处理的分块会作为一个整体参与当步的训练方法，该状态也随该分块更新一次。因此，Prompt 阶段的训练步数和更新轨迹天然依赖于调度器的分块边界。

> [!NOTE]
> 如果你的算法（例如 In-Place TTT）在数学上严格要求按模型序列位置划分固定大小的 token 块（如每 16 个 token 精确更新一步），不受外部调度器动态切块的影响，请改用 `@pllm.train_in_forward(chunk=...)` 将训练内联进模型的前向投影中，详见 [改模型](model-edits.md)。

---

## 训练的归约：mean 还是 sum

用 `@pllm.train(state=..., reduction=...)` 定义在线请求训练时，方法针对当前批次的请求行返回一个形状为 `[rows]` 的逐行损失向量。pLLM 的步进模型是：**每个状态在每个引擎步（engine step）执行一次优化器更新**，更新前先将该状态在本步中被选中的行归约为一个标量损失。

如果你写的是最常见的、按默认 `request_id` 隔离的 decode 阶段训练，每个请求在每个引擎步里只有 1 行（该状态本步选中行数 $n_s = 1$）。此时对 1 个数求均值和求和完全一样，`reduction="mean"` 与 `reduction="sum"` 没有任何区别，你不需要为此操心。

只有当**同一个状态在同一个引擎步内对应多行**时，`reduction` 才会真正决定同一状态内各行的相对权重，以及跨步之间的梯度幅度：
- 包含多个 token 的 **prompt 行**（prefill 阶段）；
- 多个请求通过共享 `key_fields` 映射到同一个状态；
- 所有请求共享同一个 `InstanceState`；
- 其它在单步内为同一状态产生多行的场景。

### 机制与怎么选

`@pllm.train` 的 `reduction` 只在每个状态内部归约本步选中的行，**两个选项都保持不同状态之间相互独立**：
- `reduction="mean"`（默认）：对每个状态 $s$ 在本步选中的 $n_s$ 行取均值，即 $\frac{1}{n_s}\sum_{r \in s} \ell_r$。
- `reduction="sum"`：对每个状态 $s$ 在本步选中的行直接求和，即 $\sum_{r \in s} \ell_r$。各状态归约后的标量损失**相加**后反向传播，绝不会把不同状态的行放在一起平均。

选择哪一个，取决于你使用的优化器以及多行之间的关系：

1. **使用 `pllm.AdamW` 时，默认选 `"mean"`**
   AdamW 对全局常数缩放不敏感，但会受**不同步之间梯度幅度跳变**的影响。在 prefill 步中，一个请求可能一次进入 $P$ 个 prompt 行；如果用 `"sum"`，prefill 步的梯度幅度会比单行 decode 步放大约 $P$ 倍，导致 AdamW 的二阶矩估计 $v$ 被放大约 $P^2$ 倍，进而压制后续大量 decode 步的有效步长。`"mean"` 消除了单步内行数 $n_s$ 的影响，让 prefill 步与 decode 步的梯度尺度保持一致。
2. **希望每个 token 的贡献恒定时，选 `"sum"`**
   `"mean"` 会把本步同一状态的行数 $n_s$ 除掉；如果你希望每个 token 对更新的贡献只取决于它自己的梯度，而不因本步恰好有多少行同批到达而被 $1/n_s$ 摊薄，应选 `"sum"`。
3. **共享 `key_fields` 的在线组训练，选 `"sum"`**
   当一组兄弟采样请求共享同一个状态时，不同样本往往在不同步提前结束。如果使用 `"mean"`，随着部分样本结束退出，分母 $n_s$ 变小，剩余存活样本中每个 token 的权重会被动放大；使用 `"sum"` 则保证每个存活 token 始终贡献恒定的一份梯度。

### 常用归一化配方

围绕 `@pllm.train` 的返回值与参数，可以直接组合出常见的训练变体：

- **逐行加权**（如优势值、置信度、门控权重 $w_r$）：直接在训练方法里返回 `weight * loss`（即 $w_r \cdot \ell_r$），配合 `reduction="mean"` 或 `"sum"` 即可。
- **Dr.GRPO 式的常数长度归一化**：希望除以固定常数 $L_{\max}$ 而不是当前步的动态行数 $n_s$ 时，使用 `reduction="sum"`，并在返回时直接将损失乘以常数 $1 / L_{\max}$。
- **排除特定行：优先用 `rows=`，不要用零权重配 mean**：在 `reduction="mean"` 下，返回 `0.0 * loss` 的行**依然计入分母 $n_s$**，会平白稀释其它有效行的梯度；`rows=` 会把这一行同时从分子和分母里去掉。注意 `rows=` 只在请求训练与模型效果或 `@pllm.add_logits` 组合时可用；独立的请求训练、以及与 `@pllm.guide_sampling` 并存的训练会训练每个选中状态的所有行。这两种场景下想排除某些行，可以用 `reduction="sum"` 加零权重：求和时零权重行不贡献，也不改变其它行的权重。
- **时间衰减与动量**：跨步的指数移动平均（EMA）或衰减属于优化器职责（如 AdamW 的 $\beta_1$、`weight_decay`）或显式正则项，不应试图通过步内归约来实现。

下面是一个采用 `reduction="sum"` 实现常数长度归一化（Dr.GRPO 风格）的完整示例：

```python
@dataclass(frozen=True)
class LengthNormalizedTrainer(pllm.Workload):
    max_len: float = 256.0

    predictor = pllm.RequestState(
        {"w": lambda m: (m.hidden_size,)},
        optimizer=pllm.AdamW(1e-3),
    )

    @pllm.train(state="predictor", reduction="sum")
    def loss(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        deep: torch.Tensor = pllm.hidden(-1),
        w: torch.Tensor = pllm.parameter("predictor", "w"),
    ) -> torch.Tensor:
        # 常数长度归一化：配合 reduction="sum"，每行损失除以固定常数 max_len
        pred = shallow.float() * w
        return (pred - deep.float()).square().mean(-1) / self.max_len


print("chunking / tokens=" in testing.check(LengthNormalizedTrainer(), CONFIG).passed)
```

---

## 零梯度不等于不更新

在 PyTorch 本地训练脚本中，若某参数梯度的值为 0，部分开发者会误以为该参数在当前步不会发生任何变化。**但在 AdamW 优化器中，零梯度依然会推进更新**：

1. **时间步前进**：优化器内部步数 $t \leftarrow t + 1$；
2. **动量衰减**：一阶动量 $m \leftarrow \beta_1 m$ 与二阶动量 $v \leftarrow \beta_2 v$ 继续按系数衰减；
3. **权重衰减（Weight Decay）生效**：即使梯度为 0，权重衰减项依然会使参数发生缩减：
   $$w \leftarrow w - \eta \cdot \lambda \cdot w$$

- **重要提醒**：如果你希望在某些步完全“冻结”或跳过特定状态的更新，**不能靠在损失方法里将梯度乘 0**。必须在调度器选择器层面（如 `rows=` 或 `update_rows=`）将该请求或状态排除在更新授权之外。

---

## 行选择器：`rows=` 与 `update_rows=`

`@pllm.train` 提供了两个可选的选择器参数：
- `rows="selector_method"`：指定一个返回布尔掩码 `[rows]` 的辅助方法；方法仍在每一行上运行，只有掩码为 `True` 的行计入损失的归约；
- `update_rows="selector_method"`：指定一个返回布尔掩码的方法，显式授权哪些行所对应的状态允许被优化器写入。

### 为什么独立请求训练会拒绝选择器？

当你尝试在独立的请求训练（无其他效果）或仅带有 `@pllm.guide_sampling` 的负载上使用 `rows=` 或 `update_rows=` 时，构建期会直接报错：

```text
rows= and update_rows= on request training require a model effect without sampler guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state, so remove rows= and update_rows=
```

- **原因**：独立的请求训练以及与 `@pllm.guide_sampling` 并存的训练走的是为 ESamp 这类"每个请求每步都学"设计的路径：每一步在采样之后给每个选中的状态做一次 AdamW 更新、发布一个新版本，并和下一步 forward 重叠。按行跳过更新需要"有条件地发布"，这条路径目前没有实现，所以构建时直接拒绝，而不是悄悄忽略你的掩码。与 `@pllm.add_logits` 或模型改写（`Adapter`、`ActivationPatch` 等）一起时，训练按有条件的更新安排，`rows=` 与 `update_rows=` 可以用。
- **替代方案（Workarounds）**：
  1. **将过滤移至采样端**：保持训练方法无条件运行，但在采样引导端挂载选择器，如 `@pllm.guide_sampling(beta=strength, rows="trained")`；
  2. **使用 `reduction="sum"` 配合零损失**：如前所述，将不需要贡献梯度的行损失乘以 0。虽然这无法阻止权重衰减，但在数学上可以使这些行贡献的梯度为零且不破坏其他行的归约分母。

---

## 旁路重叠与 `publish_lag`

### 为什么默认 `publish_lag=1` 能大幅提高吞吐？

如果要求第 $n$ 步生成的 token 在算完梯度后，其最新权重必须在第 $n+1$ 步前向的开头立刻生效（即 `publish_lag=0`），那么每一步模型前向都必须强制等待反向传播和 AdamW 算子完全执行完毕，GPU 时间线上将出现巨大的同步空隙。

pLLM 默认采用 `publish_lag=1` 机制：
1. 第 $n$ 步采样完成后，训练任务被派发到独立的旁路 CUDA stream（`side` stream）上异步执行；
2. 第 $n+1$ 步的模型前向**直接读取第 $n-1$ 步已发布的稳定版本状态**，完全无需等待第 $n$ 步训练完成；
3. 第 $n$ 步计算出的新权重在第 $n+2$ 步的前向中正式生效。

在训练方法自身内部，由于它运行在旁路流上，它读取到的是当前最新的状态版本；而在前向传播和采样器中，读取到的则是已发布的稳定版本。

在 RTX 4090（Qwen2.5-0.5B，128 并发）测试中：
- **纯在线训练（$\beta=0$）**：`publish_lag=1` 的端到端吞吐达到原生 vLLM 的 **94.3%**，而 `publish_lag=0` 仅为 **91.4%**；
- **训练 + 引导采样（$\beta=0.1$）**：`publish_lag=1` 达到原生的 **84.1%**（1,245 个旁路 kernel 中有 780 个成功与主干重叠），而 `publish_lag=0` 下跌至 **73.7%**。

### 如何设置 `publish_lag=0` 及其代价

如果你的算法在理论上严格要求每一步生成的 token 必须在紧接着的下一步立即被读取，可以通过配置显式关闭滞后：

```python
strict_config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=8,
    execution=pllm.ExecutionOptions(publish_lag=0),
)
```

- **代价**：在 `pllm.explain` 中，训练任务的通道会从 `side` 变为 `main`，执行说明将提示 `updates serialize before the next forward`，吞吐会因为串行等待而显著下降。
- **契约验证**：`pllm.testing.check` 在 CPU 上运行时会自动验证所有任务读取的状态版本号，严格核实其是否符合 `publish_lag` 的版本发布契约。

---

## 一阶梯度边：`@pllm.differentiate` 与 `pllm.result`

在许多进阶结构中，`@pllm.train` 方法并不直接消费原始激活，而是消费另一个上游辅助方法产生的结果（通过 `pllm.result("upstream_name")` 读入）。

为了让梯度能够从下游损失反向传播到上游操作，上游方法必须显式挂载 `@pllm.differentiate` 声明导数边；同时，下游训练方法也必须在输入参数中保留对应边的导数连接：

```python
@dataclass(frozen=True)
class ChainedDistiller(pllm.Workload):
    head = pllm.RequestState(
        {"w": lambda m: (m.hidden_size, 64)},
        optimizer=pllm.AdamW(1e-3),
    )

    # 上游投影：和训练在同一阶段运行；保留输入 shallow 的导数边，parameters=True 让参数 w 也接收梯度
    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.differentiate(inputs=("shallow",), parameters=True)
    def project(
        self,
        shallow: torch.Tensor = pllm.hidden(0),
        w: torch.Tensor = pllm.parameter("head", "w"),
    ) -> torch.Tensor:
        return torch.bmm(shallow.float().unsqueeze(1), w).squeeze(1)

    # 下游训练损失：通过 pllm.result 读取结果，并声明 inputs=("features",) 维持导数连接
    @pllm.train(state="head")
    @pllm.differentiate(inputs=("features",))
    def loss(
        self,
        features: torch.Tensor = pllm.result("project"),
        deep: torch.Tensor = pllm.hidden(-1),
    ) -> torch.Tensor:
        return (features - deep[:, :64].float()).square().mean(-1)


print("request isolation" in testing.check(ChainedDistiller(), CONFIG).passed)
```

### 导数边界与契约约束

- **和训练方法在同一阶段**：梯度只在一次调用之内传播，所以上游方法要用 `@pllm.on(pllm.Phase.AFTER_SAMPLE)` 放到训练所在的采样之后；否则构建时报 `the @pllm.train loss of ... must differentiate only its own state`。
- **调用内局部一阶求导**：`@pllm.differentiate` 的求导完全局限在单次引擎步的局部任务链条内。它**绝不是穿透 vLLM 底座模型权重的全模型反向传播**，也不会跨越步间边界。
- **参数冻结控制**：`parameters=False`（默认值）表示只对 `inputs` 中列出的输入张量求导，当前操作读取的参数不接收梯度；`parameters=True` 则允许参数接收梯度并更新。
- **汇聚契约**：每一个被 `@pllm.differentiate` 标记保留导数边的中间操作，其结果必须最终流向一个被 `@pllm.train` 绑定的损失方法。孤立的可微操作在构建期会被拒绝：`every @pllm.differentiate result must feed a @pllm.train loss`。

---

## 跨步状态捕获：`pllm.carry`

某些算法需要随时间步传递普通张量记忆（例如上一步的预测值、历史激活的移动统计等），这些张量**不需要一阶导数，也不需要 AdamW 动量槽位**。

可以在状态上通过 `carry=` 声明并在训练方法中使用 `pllm.carry` 进行跨步捕获：

```python
@dataclass(frozen=True)
class CarriedDistiller(pllm.Workload):
    head = pllm.RequestState(
        {"w": lambda m: (m.hidden_size,)},
        optimizer=pllm.AdamW(1e-3),
        # 声明 carry 字段：无动量槽，不被 AdamW 触碰
        carry={"prev_shallow": lambda m: (m.hidden_size,)},
    )

    @pllm.train(state="head")
    def loss(
        self,
        # 用 pllm.carry 包装输入：发布时将该输入当前请求最后一行的值写入 carry 字段
        shallow: torch.Tensor = pllm.carry(pllm.hidden(0), state="head", field="prev_shallow"),
        deep: torch.Tensor = pllm.hidden(-1),
        w: torch.Tensor = pllm.parameter("head", "w"),
        # 读取上一步发布的值（首步准入时自动为零）
        prev: torch.Tensor = pllm.parameter("head", "prev_shallow"),
    ) -> torch.Tensor:
        pred = (shallow.float() + prev.float()) * 0.5 * w
        return (pred - deep.float()).square().mean(-1)


print("publish_lag" in testing.check(CarriedDistiller(), CONFIG).passed)
```

- **语义机制**：
  - 写入：训练更新发布时，`pllm.carry` 捕获当前请求选中的最后一行张量值，写入对应字段；
  - 读取：`pllm.parameter(state, field)` 读取上一步发布的值。由于初始状态是零初始化的，第一步读取到的值严格为零。

---

## 设备端计数返回：`pllm.UpdateOutput`

普通的 `@pllm.train` 方法直接返回 `torch.Tensor` 形式的逐行损失。但在某些复杂业务或带门控的训练中，你可能需要向引擎报告状态是否被修改以及实际参与训练的行数，同时绝不能触发 GPU 到 CPU 的同步阻断。

此时可以让方法返回 `pllm.UpdateOutput`：

```python
# check-docs: skip: the return statement of a training method body
return pllm.UpdateOutput(
    output=loss,                                          # [rows] 损失张量
    changed=loss.new_ones((), dtype=torch.bool),          # 0 维布尔张量：状态是否改变
    update_count=loss.new_ones((), dtype=torch.int64),    # 0 维整数：本次推进的更新步数
    trained_rows=loss.new_full((), loss.shape[0], dtype=torch.int64),  # 0 维整数：参与训练的行数
)
```

`changed`、`update_count` 和 `trained_rows` 必须是**输出设备上的 0 维张量（0-d Tensor）**，绝不能是 Python 原生标量（`float` 或 `int`），从而确保在 CUDA Graph 内部执行时零主机同步。

---

## 观测与自检：确认训练发生

当带负载的推理引擎跑起来之后，如何验证在线训练确实生效且没有退回低效的 eager 模式？

### 1. 运行时诊断报告 `llm.diagnostics()`

调用 `llm.diagnostics()` 会返回当前引擎运行状态的轻量快照，核心关注四个字段：

```python
# check-docs: skip: requires CUDA GPU engine
from vllm import SamplingParams

sampling = SamplingParams(n=4, max_tokens=32, temperature=0.8, top_p=0.95, seed=0)
with pllm.make_llm(CONFIG, workload=ShallowDistiller()) as llm:
    llm.generate(["Explain gradient descent:"], sampling)
    report = llm.diagnostics()

print(f"loss_count: {report.loss_count}")
print(f"update_count: {report.update_count}")
print(f"trained_rows: {report.trained_rows}")
print(f"eager_reason: {report.eager_reason!r}")
```

- `loss_count`：损失计算执行的总次数；
- `update_count`：优化器成功推进更新的总步数；
- `trained_rows`：实际参与梯度更新的总 token 行数；
- `eager_reason`：若为空字符串 `""`，说明模型前向与在线训练完全维持在整图 CUDA Graph 模式下运行；若非空，会明确指出导致退回 eager 模式的算子或声明。

### 2. 本地静态契约检查 `testing.check`

在无需任何真实 GPU 的 CPU 环境下，`pllm.testing.check` 能够对训练负载进行全面的契约静态验证：
- 扰动请求排列顺序（`row permutation`），检查是否存在跨行非法归约；
- 扰动相邻请求（`request isolation`），检查请求间状态是否严格隔离；
- 检验 `publish_lag` 版本契约，确保没有算子在旁路流发布前非法读取未提交版本；
- 检验 Prompt 分块与生成阶段的输入一致性。

---

## 宿主批训练（Host Batch Training）

如果你需要的不是解码期逐请求在线学习，而是在外部宿主送入的离线或半在线数据批次（如强化学习采集的整批 Rollout）上进行模型微调，应使用宿主批训练通道，详见 [宿主循环与批训练](host-loops.md)。

读取 `pllm.batch(...)` 字段（如 `WeightedRowBatch` 或 `CausalTrainBatch`）的宿主训练方法**必须自行负责归约并直接返回一个标量损失**；在这类方法上声明非默认的 `reduction=` 会在构建期被直接拒绝。

---

## 常见错误与排查

在编写在线训练负载时，常见的构建期拒绝与运行期错误如下表所示：

| 报错信息 | 触发原因与排查修复 |
| :--- | :--- |
| `training methods return one loss per row; pLLM reduces the selected rows per state` | 训练方法返回了标量（如调用了 `.mean()` 将行轴压平）。**修复**：仅在特征维归约（`dim=-1`），保留第 0 维，返回 `[rows]` 的逐行损失。 |
| `@pllm.train reads ... which trains with ... declare ... with optimizer=...` | 状态上缺少优化器声明，或声明的优化器与训练模式不匹配。**修复**：为对应的 `RequestState` 声明 `optimizer=pllm.AdamW(lr)`。 |
| `rows= and update_rows= on request training require a model effect without sampler guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state, so remove rows= and update_rows=` | 独立在线训练或配合 `@pllm.guide_sampling` 的训练使用了行选择器，该路径要求无条件更新。**修复**：移除训练方法上的 `rows=`，改在采样方法上筛选，或使用 `reduction="sum"` 配合零权重。 |
| `every @pllm.differentiate result must feed a @pllm.train loss` | 上游声明了 `@pllm.differentiate` 的中间结果没有被任何 `@pllm.train` 方法消费。**修复**：确保中间结果经由 `pllm.result` 最终输入至损失函数，或移除未使用的可微标记。 |
| `host-batch training returns one scalar and owns its reduction; remove reduction=` | 宿主批次训练方法传入了非默认的 `reduction=`。**修复**：宿主批训练自行管理归约，从 `@pllm.train` 中移除 `reduction=` 参数。 |
| `the @pllm.train loss of ... must differentiate only its own state` | 训练方法通过导数链条反向传播到了其他不属于当前声明 `state` 的外部状态。**修复**：检查上游方法的 `parameters=True` 声明，确保导数边仅汇聚到当前训练的状态。 |

---

#### 自测小问题

1. **问题**：在 decode 训练中，如果你希望某些 token 不产生梯度更新，直接在训练方法里返回 `0.0 * loss` 是否可行？为什么？
   <details><summary>答案</summary>
   不可行。有两个原因：第一，若使用默认的 <code>reduction="mean"</code>，损失为 0 的行依然会作为分母 $n_s$ 的一部分计入均值，从而稀释同一状态在当前步其他有效行的梯度；第二，更关键的是在 AdamW 优化器中，零梯度依然会推进时间步 $t$、衰减动量并施加权重衰减（weight decay），导致参数依然发生变化。真正彻底跳过更新应在具备条件更新的环境下使用 <code>rows=</code> 选择器排除，或改用 <code>reduction="sum"</code> 配合零权重以避免稀释分母。
   </details>

2. **问题**：为什么默认的 <code>publish_lag=1</code> 让第 $n$ 步的训练结果直到第 $n+2$ 步才能被模型前向读取？如果强行设为 <code>publish_lag=0</code> 会发生什么？
   <details><summary>答案</summary>
   因为在线训练被调度在旁路 CUDA stream（<code>side</code> lane）上异步执行，第 $n$ 步的参数更新与第 $n+1$ 步的模型前向在 GPU 时间线上并行重叠。为了避免第 $n+1$ 步前向前停下来串行等待反向传播与 AdamW 完成，第 $n+1$ 步前向直接读取第 $n$ 步开始前已发布的稳定版本（即第 $n-1$ 步的更新结果），因此第 $n$ 步的新版本在第 $n+2$ 步才正式生效。如果设为 <code>publish_lag=0</code>，第 $n+1$ 步前向必须串行等待第 $n$ 步训练全部算完，这会打断流水线并引入时间线气泡，在 128 并发下可能导致吞吐明显下降（例如在带引导采样下吞吐从 84.1% 降至 73.7%）。
   </details>

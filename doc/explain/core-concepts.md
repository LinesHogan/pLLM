# 核心概念

跑通一两篇教程（如 [00 快速上手](../tutorials/00_quickstart.ipynb) 或 [03 decode 时训练](../tutorials/03_decode_training.ipynb)）之后，你可能会注意到：在 pLLM 里写算法和在 Hugging Face `generate` 里挂载 PyTorch forward hook 很不相同。输入要写成参数默认值的标记，方法里不能调用 `.item()` 或对张量值写 Python `if`，归约只能沿最后一维（`dim=-1`）进行，而每步训练算出的新参数默认要隔一步（在第 $n+2$ 步）才会被读到。

这些规则并不是随意规定的语法限制，而是由现代推理引擎的两项底层物理机制直接决定的：**连续批处理（continuous batching）**与 **CUDA Graph 异步执行**。理解下面五个核心概念——为什么用声明代替 hook、什么是“行”、一步推理的时间线与流重叠、状态如何按版本发布，以及为什么构建期会直接拒绝越界组合——你就能一眼看出每条规则背后的原因，并在写代码前预判负载的行为与开销。

---

## 1. 为什么不是 hook：连续批处理与 CUDA Graph

### 服务引擎里的真实问题

在单请求实验代码里，读取或修改第 12 层的激活只需要五行 `register_forward_hook`。但把同样的逻辑放进 vLLM 服务引擎时，会立刻撞上两堵墙：

1. **CUDA Graph 回放期根本不执行 Python 代码**：在自回归解码（decode）阶段，GPU 每步只为每个并发请求生成 1 个 token，算子本身耗时极短。为了消除逐个从 Python 发射算子的 CPU 开销，vLLM 会把整个模型前向录制成一张 CUDA Graph，后续每步只向 GPU 发射一次图回放指令（关掉 CUDA Graph 退回 eager 模式后，发射开销的代价取决于一步有多长：在 RTX 4090、128 并发下，一步约 3 ms 的 0.5B 掉到图模式的 49%，一步约 22 ms 的 7B 仍有 98%，见 [性能与显存](performance.md)）。图回放期间，Python 解释器完全不参与层的执行，挂在模块上的 Python hook 根本不会被调用。
2. **打包张量没有请求身份，且 Python 分支会触发主机同步**：即便退回 eager 模式，连续批处理也会把几十个并发请求的 token 拼进同一个二维张量。普通模块 hook 只拿到这块拼好的张量，不知道哪一行属于哪个请求、哪一行是刚算完的 prompt 末尾。更糟的是，如果 hook 里写了 `if h.norm() > 1:` 或 `print(h.item())`，CPU 就必须停下来等待 GPU 把前面的算子全部算完並把数值拷回主机内存（host sync），直接打断 GPU 流水线，也让 CUDA Graph 无法录制。

### pLLM 的做法：一次声明、静态追踪、整图捕获

pLLM 用**声明 + 纯张量数学**代替运行时回调。你编写的 `pllm.Workload` 子类只声明三件事（详见 [负载的组成](../guides/workload-anatomy.md)）：

- **读什么**：写在方法参数的默认值上（如 `pllm.hidden(0)`、`pllm.residual(12)`、`pllm.parameter("distiller", "w")`）；
- **拥有什么状态**：写成类属性或带类型注解的 `@property`（如 `pllm.RequestState`、`pllm.InstanceState`）；
- **结果去哪**：写成方法装饰器或模型位置声明（如 `@pllm.train`、`@pllm.guide_sampling`、`@pllm.export`、`pllm.model.ActivationPatch`）。

在创建引擎（`pllm.make_llm`）或查看执行计划（[`pllm.explain`](../guides/explain.md)）时，pLLM 只在启动期读取一次这些声明，在 CPU 上用 PyTorch 的 `meta` 张量把每个方法追踪（trace）一遍，算出所有中间张量的形状、显存生命周期与依赖关系，然后把你的张量数学（包括反向传播和 `AdamW` / `SGD` 更新）直接编译并捕获进 CUDA Graph。

```python
from dataclasses import dataclass

import torch
from torch import nn

import pllm
from pllm import testing


@dataclass(frozen=True)
class Distill(pllm.Workload):
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


CONFIG = pllm.ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", max_num_seqs=8)
print(pllm.explain(Distill(), CONFIG))
```

`pllm.explain` 在 CPU 上完成追踪与规划，输出节选如下：

```text
forward  CUDA graphs: no effect needs eager
member  effect  anchor / lane / graph  state reads (origin, distance)
loss  train  token.commit / side / captured  distiller (latest, 0)
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
state  scope  roots  bytes/root  reserved bytes [exact]
distiller  request  34  129104  4389596  field 0: float32, field 1: float32, ...
host     0 syncs in traced workload math; exports: none
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

### 对你写代码的直接影响

因为方法是在构建期用 `meta` 张量追踪一次、并在运行期作为 CUDA Graph 在 GPU 上回放，你的方法体需要遵守四条规则：

- **不要在方法里调用 `.item()`、`.tolist()`、`.cpu()` 或 `print(tensor)`**：`meta` 张量没有实际数值，这类操作在 `explain` 和 `make_llm` 阶段就会抛出 `RuntimeError: Tensor.item() cannot be called on meta tensors`。想把中间结果拿回 CPU 观察，使用异步有界的 `@pllm.export`（见 [结果去哪](../guides/results.md)）。
- **不要对张量的值写 Python `if` / `while` 分支**：写 `if h.norm() > 1:` 同样会尝试把张量转成 Python `bool`。条件逻辑需要写成张量掩码与 `torch.where`（例如 `torch.where(mask, a, b)`）。对负载配置项的固定 Python 循环（如 `@pllm.custom_forward` 里的 `for _ in range(self.loops):`）会在追踪时展开，可以正常使用。
- **不要在 `self` 上保存可变 Python 状态**：负载类是冻结的（`@dataclass(frozen=True)`）。方法不是每步被 Python 调用的函数，写 `self.count += 1` 或往 Python 字典里塞张量不会在解码步之间生效。跨步数据只放在 `pllm.RequestState` 或 `pllm.InstanceState` 中，步数进展通过 `pllm.version(state)` 读取。
- **把方法入参视为只读借用视图**：传入方法的激活与状态张量可能正与同一步的其他方法共享，或指向已发布的状态版本。不要做原地修改（如 `h.mul_(2)` 或 `w.add_(1)`）；始终返回新张量，并通过 `@pllm.write` 或 `@pllm.train` 发布状态修改。

---

## 2. 行（rows）：动态打包的请求与独立性契约

### 服务引擎里的真实问题

在离线批处理里，batch 张量的第 `i` 行永远是第 `i` 条样本。但在连续批处理引擎里，一步（engine step）中同时混着处于不同阶段的请求：

- 正在逐 token 生成的 **decode** 请求；
- 刚到达的 **prefill** 请求（开启分块预填充时，一条长 prompt 还会被切成多个块跨多步算完）；
- 因 KV 缓存空间不足被临时**抢占（preempt）**、稍后重新计算历史的请求。

调度器在每一步把当前活跃的 token 紧缩打包进张量的第 0 维，**行的排列顺序由调度器决定，每一步都在变，你无法控制**。

### pLLM 中的“一行”代表什么

在负载方法收到的所有按行张量中，第 0 维始终是**行轴（`[rows, ...]`）**。pLLM 已经在底层替你完成了行筛选与状态对齐：当你读取 `pllm.RequestState`（通过 `pllm.parameter` 或 `pllm.module`）时，参数张量的第 `r` 行保证属于激活张量第 `r` 行所在的那个请求。

具体“一行对应哪个 token”，取决于你声明的输入来源（详见 [读取模型内部](../guides/reading.md)）：

| 输入来源或方法类型 | 第 0 维（`rows`）里每一行的含义 |
| :--- | :--- |
| `pllm.hidden`、`pllm.residual`、`pllm.final_hidden`（默认 `tokens="decode"`） | 每个处于 decode 阶段的请求各占 **1 行**（该请求最新生成的 decode token） |
| 上述标记设 `tokens="prefill"` | 本步刚完成整个 prompt 预填充的请求各占 **1 行**（该 prompt 的最后一个 token，即使 prompt 被分块也只在完成那一步出现一次） |
| `pllm.layer_logits`、`@pllm.add_logits`、`@pllm.guide_sampling` | 采样器在本步要抽样的每个 token 各占 **1 行**（刚完成预填充的 prompt 末 token，以及每个 decode token） |
| 上述激活标记设 `tokens="all"`（仅限 `@pllm.export`） | 本步新算出的**每一个 token** 各占 **1 行**（含分块 prompt 的中间 token），按请求内位置有序排列；抢占重算的 token 只导出一次 |

### 对你写代码的直接影响

- **只沿特征维（`dim=-1`）归约，绝不跨行（`dim=0`）归约**：写下 `h - h.mean(0)`、`softmax(dim=0)`、批归一化（BatchNorm）或按行排序，都会让某一个请求的输出取决于“这一步恰好和谁拼在同一个 batch 里”。所有均值、范数、归一化都应沿 `dim=-1` 计算（如 `h.float().norm(dim=-1, keepdim=True)`）。
- **逐行训练方法返回形状为 `[rows]` 的一维 loss，不要自己对整批求均值**：在请求行上的 `@pllm.train` 方法里，不同行属于不同请求（而且为了复用固定尺寸的 CUDA Graph，未满 batch 的尾部还带有指向哑槽位的填充行）。因此方法体应返回每行各自的 loss（如 `.square().mean(-1)`，形状 `[rows]`）；由 pLLM 负责屏蔽无效行，并按每份状态副本归约属于它的选中行（默认 `reduction="mean"`，可选 `"sum"`）。只有读取 `pllm.batch(...)` 的宿主批训练才由方法自己归约并返回单个标量。
- **用 `pllm.testing.check` 在 CPU 上验证行独立性**：你不靠肉眼排查是否误写了跨行耦合。[`pllm.testing.check`](../guides/checking.md) 会在 CPU 上用真实的规划器与执行器跑三个请求的多种固定调度，逐一检验这些契约：

```python
print(testing.check(Distill(), CONFIG, neutral=Distill(beta=0.0)))
```

```text
status  contract / reason
PASS    compile
PASS    row permutation
PASS    request isolation
PASS    mixed prefill/decode
PASS    chunking / tokens=
PASS    publish_lag
PASS    preempted
PASS    neutral
```

只要某一个请求的结果在打乱行顺序（`row permutation`）、改变其他请求输入（`request isolation`）、混合 prefill 与 decode（`mixed prefill/decode`）、切分 prompt（`chunking / tokens=`）或抢占重放（`preempted`）时发生变化，`testing.check` 就会抛出 `ContractError` 并指出是哪个方法混了行。

---

## 3. 时间线：锚点、阶段窗口与主/旁路流

### 服务引擎里的真实问题

在一步解码内部，不同数据在不同时刻才诞生：输入 token ID 在模型前向之前就已知；各层隐藏状态要等模型前向跑完才有；而本步抽出的新 token 要等采样器跑完才存在。同时，如果把外部查表或反向传播、优化器更新统统串行塞进模型前向所在的同一条 GPU 流（CUDA stream），每一步前向都得干等这些计算跑完，吞吐就会大幅下滑。

### 一步的时间线：锚点（anchors）与阶段窗口（`pllm.Phase`）

pLLM 把引擎的每一步划分为有序的时间轴**锚点（anchors）**。当你在 `pllm.explain` 的 `member` 列看到 `token.commit / side / captured` 时，第一项就是任务挂载的锚点：

1. `request.prepare`：一步开始；活跃请求槽位与历史输入 token（`pllm.input_tokens`）就绪。
2. `model.execute`：基座模型执行前向传播（`ActivationPatch`、`Adapter`、`@pllm.custom_residual`、`@pllm.custom_attention`、`@pllm.train_in_forward`、`@pllm.custom_forward` 在此期间运行）。
3. `state.publish`：模型前向结束；各层激活（`pllm.hidden`、`pllm.residual`、`pllm.final_hidden`）以及在前向边界发布的状态版本就绪。
4. `sampler.pre_draw`：采样前 logits 偏置（`@pllm.add_logits`）与采样引导预测（`@pllm.guide_sampling`）就绪，送入采样器。
5. `sampler.complete`：采样器完成抽样。
6. `token.commit`：本步抽出的 token（`pllm.sampled_tokens`）与未修改的原始分布（`pllm.base_logits`）提交确定。
7. `request.finish` → `next.prepare`：已结束的请求退役回收，引擎准备下一步。

每两个锚点之间的区间构成一个**阶段窗口（`pllm.Phase`）**。每种效果装饰器都有自己的默认窗口，规划器会自动把方法安放在它的输入已经就绪、且输出赶得上消费方的窗口里；只有编写不带效果装饰器的辅助方法，或需要覆盖默认时机时，才需要写 `@pllm.on(phase)`：

| `pllm.Phase` 枚举 | 对应的锚点区间 | 默认在此窗口运行的效果与计算 |
| :--- | :--- | :--- |
| `Phase.BEFORE_FORWARD` | `request.prepare` → `model.execute` | 只依赖 `pllm.input_tokens` 和只读状态的查表计算 |
| `Phase.IN_FORWARD` | `model.execute` → `state.publish` | `@pllm.train_in_forward`（以及作用在模型内部的补丁、Adapter、残差与注意力改写） |
| `Phase.AFTER_FORWARD` | `state.publish` → `sampler.pre_draw` | `@pllm.export`（默认在前向完成后导出） |
| `Phase.BEFORE_SAMPLE` | `model.execute` → `sampler.pre_draw` | `@pllm.add_logits`、`@pllm.guide_sampling` |
| `Phase.AFTER_SAMPLE` | `token.commit` → `next.prepare` | 逐请求行 `@pllm.train`、宿主批次 `@pllm.train` |
| `Phase.AUTOMATIC` | `request.prepare` → `next.prepare` | `@pllm.write` 与 `@pllm.train` 的默认声明窗口（由规划器按依赖自动定位） |

### 主流（`main`）与旁路流（`side`）的重叠

除了决定“在哪个锚点触发”，规划器还会为每个任务选择执行车道（`lane`）：

- **`main`（主流）**：运行模型前向（`model.execute`）和采样器的主 CUDA stream，以及必须赶在本步采样前出结果的任务（如第 1 节 `Distill` 中的 `predict`）。
- **`side`（旁路流）**：与主流并发执行的辅助 CUDA stream。只要任务不阻塞当前步的采样，规划器就会把它放进 `side` 流，与主流的计算重叠：
  - 只读 `pllm.input_tokens` 与只读表的查表方法在 `request.prepare / side` 启动，与**当前步**的模型前向重叠；
  - 逐请求行的在线训练（如 `Distill` 中的 `loss`）在 `token.commit / side` 启动，与**下一步**的模型前向重叠。

以第 1 节的 `Distill` 负载（再加一个只读 token 查表辅助方法）为例，连续两个 decode 步在两条流上的时间线如下：

```text
时间轴锚点:  request.prepare   model.execute    state.publish   sampler.pre_draw  token.commit     next.prepare
                 |                 |                 |                |               |                |
第 n 步   main:  |-- [元数据准备] --|-- [模型前向 n] --|-- [predict] ---|-- [采样器抽样] -|                |
          side:  |-- [token 查表] -|                 |                |               |-- [loss 反向与 AdamW (第 n 步)] -->
                                                                                              (与第 n+1 步前向重叠)
第 n+1 步 main:                                                                       |-- [准备] --|-- [模型前向 n+1] --|
```

### 对你写代码的直接影响

- **数据只能在产生它的阶段及之后被读取**：如果一个方法在某项输入诞生之前就去读它，pLLM 会在构建期报错：`a task reads a value (sampled tokens, hidden states or another task's result) that exists only after its phase ends; declare it in a later phase`。最常见的例子是 `pllm.sampled_tokens()` 和 `pllm.base_logits()`：它们在采样结束后的 `token.commit` 才存在，因此只能用于 `@pllm.train` 或带 `@pllm.on(pllm.Phase.AFTER_SAMPLE)` 的方法；如果你想在采样前的 `@pllm.add_logits` 或 `@pllm.export` 里读取模型最后一层的 logits，应使用 `pllm.layer_logits((-1,))`。
- **把可提前的计算拆成独立辅助方法以获得流重叠**：如果你用 `placement="host"` 把大查找表放在主机内存里，希望跨 PCIe 的按行读取与模型前向重叠，就需要把查表写成一个不带效果装饰器的独立方法（只读 `pllm.input_tokens` 和 `pllm.parameter`），再让 `@pllm.add_logits` 或 `@pllm.custom_residual` 用 `pllm.result("lookup")` 读取它的返回值。如果把查表和 `pllm.hidden(0)` 写在同一个方法里，该方法就必须等到前向结束后的 `state.publish` 才能启动，失去与前向重叠的机会（详见 [性能与显存](performance.md)）。

---

## 4. 状态与版本：不可变快照与 `publish_lag`

### 服务引擎里的真实问题

当上百个请求并发生成，同时在线训练或外部导入在不断修改参数时，会出现两个并发问题：

1. **谁和谁共享哪一份参数？** 有的参数全引擎共享（如静态引导向量、全局查找表），有的参数每个请求独立拥有（如每请求在线蒸馏器或 LoRA），还有的参数需要在同一个 prompt 的 $n$ 个采样分支之间共享，或者跨多次 `generate` 调用按会话路由保留。
2. **读写冲突与半途变卦**：如果旁路流上的优化器正在原地改写一块权重张量，而主流上的下一步前向同时在读这块张量，模型就会读到写了一半的脏数据；或者当一个新请求通过 `generate` 导入了新的引导向量时，如果直接覆盖全局缓冲，正在生成途中的老请求就会在一句话说到一半时突然被换掉参数。

### 作用域、不可变版本与在途隔离

在 pLLM 中，**状态（State）是唯一可变的东西**（详见 [状态](../guides/state.md)）。

首先，你通过状态类型与 `key_fields` 声明每份状态副本的归属范围：

- **`pllm.InstanceState`**：整个引擎共享一份逻辑状态。
- **`pllm.RequestState`**：按 `key_fields` 切分独立的状态副本：
  - 默认 `key_fields=("request_id",)`：每个请求独占一份副本，请求准入时初始化，生成结束时自动回收；
  - `key_fields=("prompt_index",)`：同一个 prompt 抽出的 $n$ 条采样序列共享一份副本；
  - `key_fields=("route",)`：`generate(..., routes=(...))` 传入相同路由键的请求共享一份副本，并跨多次 `generate` 调用持续保留，直到引擎关闭。

其次，**状态更新从不在读者眼皮底下原地覆盖，而是发布新版本（Version）**。无论是 `@pllm.train` 的优化器更新、`@pllm.write` 的张量写入、外部导入 `pllm.io.StateImport`，还是外部进程推送 `llm.feed`，每次更新都先写到一份新的副本里，写完后在窗口边界整体发布为一个新版本。

读取状态的方法在窗口开始时锁定一个已发布的版本，因此：

- **绝不会读到写了一半的张量**；
- **已在途的请求保留它开始时的版本**：回看 [00 快速上手](../tutorials/00_quickstart.ipynb) 中的激活引导负载 `Steering`，它声明了 `steer = pllm.InstanceState(..., init=pllm.Imported())` 与 `pllm.model.ActivationPatch("steer", layers=(12,), operator="add", decode=True)`。每次调用 `llm.generate(prompts, greedy, batch=pllm.io.StateImport((strength * vector,)))` 都会为 `steer` 发布一个新版本；这次调用新准入的请求绑定到这个新版本，而已经在生成中的请求继续读取它准入时的版本（`ActivationPatch` 默认 `refresh="admission"`；只有显式声明 `refresh="step"` 时，才会在每个 decode 步切换读取最新版本）。

### `publish_lag`：为什么默认隔一步可见？

理解了版本与旁路流，就能看懂第 1 节 `pllm.explain` 输出里的这两行：

```text
loss     train           token.commit / side / captured   distiller (latest, 0)
predict  guide_sampling  state.publish / main / captured  distiller (boundary, 1)
timing   publish_lag=1 (default): updates at step n become readable at n + 2; training may overlap later forwards
```

在 [03 decode 时训练](../tutorials/03_decode_training.ipynb) 的在线蒸馏（ESamp）中，第 $n$ 步的训练方法 `loss` 在采样确定（`token.commit`）后启动，需要在旁路流（`side`）上跑完前向、反向传播和 `AdamW` 更新。

- 如果要求第 $n+1$ 步的 `predict` 立刻读到第 $n$ 步训练出的新权重（即 `publish_lag=0`），主流上的第 $n+1$ 步就必须停下来干等第 $n$ 步的训练全部算完，训练和推理被迫串行。
- pLLM 默认设置 **`publish_lag=1`**：第 $n$ 步算出的更新在第 **$n+2$** 步才对其他读取者可见。这样，当第 $n$ 步的训练在 `side` 流上运行时，第 $n+1$ 步的主流可以直接读取第 $n-1$ 步已发布的版本（`boundary, 1`，距离最新差 1 步），完全不用等待；与此同时，训练任务自己始终基于最新版本迭代（`latest, 0`，距离为 0），不会丢失任何一步梯度（详见 [在线训练](../guides/training.md)）。
- 因为 `side` 流正在写的新版本和 `main` 流正在读的旧版本必须同时留在显存里，pLLM 为每个状态预留的份数比并发请求数多（`explain` 里 `state` 行的 `roots`：`max_num_seqs=8` 时 `Distill` 预留 34 份）。`publish_lag` 越大，同时存在的版本越多，预留的份数也越多。

### 对你写代码的直接影响

- **用 `pllm.version(state)` 判断状态是否已经更新过**：`pllm.version("distiller")` 返回 `int64 [rows]` 张量，记录每行的状态副本已经发布了多少次可读更新（首次更新前为 `0`）。在教程 03 中，通过选择器方法 `def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor: return versions > 0` 搭配 `@pllm.guide_sampling(beta=strength, rows="trained")`，就能让刚起步、还没完成第一次训练更新的请求先按原模型采样。
- **需要严格无滞后顺序时，显式设置 `publish_lag=0`**：如果你的算法在数学上要求第 $n$ 步的更新必须在第 $n+1$ 步立刻生效，可以在 `ModelConfig` 中传入 `execution=pllm.ExecutionOptions(publish_lag=0)`，主动用旁路重叠换取严格步序。

---

## 5. 为什么有边界：构建期拒绝与显式代价

### 服务引擎里的真实问题

在研究脚本里，如果两个模块组合在一起存在语义歧义，或者某个张量超出了预留显存，很多工具会选择“悄悄退回慢路径”（例如关掉图捕获、悄悄把张量挪到 CPU 内存，或者按某种隐式顺序覆盖）。在服务引擎里，这种静默回退是灾难性的：代码看似跑通了，但要么吞吐掉了一大半却没人察觉，要么梯度永远是零、算法根本没有生效。

### pLLM 的原则：构建期验证，贵的决定由作者显式写出来

pLLM 把负载编译当成一次静态校验（详见 [组合规则](../reference/composition-rules.md) 与 [报错目录](../reference/errors.md)）：

1. **在触碰 GPU 之前拒绝非法或歧义组合**：`pllm.explain` 和 `pllm.make_llm` 使用同一套准入规则。只要声明之间存在真实冲突或语义未定义，构建期立即抛出异常并说明原因与改法，不会悄悄换一条路径执行。唯一的自动退让是 CUDA graph：默认 `cuda_graphs="auto"` 下，个别暂时不能进 graph 的写法会让模型以 eager 运行，但原因写在 `explain` 的 `forward` 行和 `llm.diagnostics().eager_reason` 里；设 `ModelConfig(cuda_graphs=True)` 则直接报错。
2. **默认值只替你做廉价的局部决定**：例如状态字段默认 `dtype="auto"` 会根据字段角色自动选择精度——被优化器更新的参数或“读出旧值再写回”的累加器使用 `torch.float32`（防止微小梯度下溢或长序列累加上漂），只读字段则跟随模型精度 `bfloat16`/`float16`（省下一半显存容量与带宽）。
3. **昂贵的取舍必须由你在声明里写明，`explain` 把账单列出来**：
   - 把只读大查找表放在主机内存（`pllm.Pretrained(path, placement="host")`）还是显存（默认 `"device"`，绝不因显存不足自动溢写到主机）；
   - 改变 `@pllm.custom_attention` 的 KV 物理保留策略（`keep="window"`、`sink=True` 或跨层共享 `keep="none", source=k`）；
   - 关闭跨步重叠（`publish_lag=0`）或要求前向内跨块立即读取最新权重（`@pllm.train_in_forward(visibility="exact")`）；
   - 显存预算旋钮（`ExecutionOptions` 的 `state_page_budget_bytes`、`execution_peak_budget_bytes`、`cuda_graph_pool_budget_bytes`）：当 `execution_peak_budget_bytes` 低于旁路重叠峰值但高于内联下限时，`explain` 的 `memory` 行会明确标出阈值并将旁路任务优雅降级到主流内联执行；低于内联下限或图池预算不足时直接报错并给出所需字节数，绝不静默关掉 CUDA Graph。

### 四个典型边界及其背后的原因

下面四个例子展示了 pLLM 为什么在构建期拦下某些写法，以及每个限制背后的数学或原因（更多关于“某类算法为什么不支持”的讨论见 [设计取舍](design-tradeoffs.md)，写法指南见 [改模型](../guides/model-edits.md) 与 [改 forward 结构](../guides/forward-structure.md)）：

| 被拒绝的写法 | 构建期报错信息（节选） | 为什么有这条边界 / 该怎么写 |
| :--- | :--- | :--- |
| 同一个负载（含子组件）声明了两个 `@pllm.guide_sampling` | `... and ... both declare @pllm.guide_sampling; keep at most one guidance method` | 多个 `@pllm.add_logits` 只是在过滤前把词表偏置相加，先加哪个结果都一样，因此可以共存。`@pllm.guide_sampling` 则要把一个 hidden 预测投影到词表，在 top-k / top-p 过滤后的分布上按 `(1 + beta) * logits - beta * pred_logits` 重新加权，采样器每步只做这一次。两个预测该相加、取平均还是先后应用，是算法本身的选择，pLLM 不替你猜：在一个方法里把它们合成一个预测返回，其余词表修正改用 `@pllm.add_logits`。 |
| 使用 `@pllm.custom_forward` 的同时，又有方法读取 `pllm.hidden` 或 `pllm.final_hidden` | `... reads block or model outputs, which a @pllm.custom_forward does not produce; use pllm.residual or pllm.layer_logits to read each layer's last visited stream` | 标准前向里每层恰好跑一次，`pllm.hidden(L)` 唯一对应第 `L` 层的那次 MLP 输出；而在 `@pllm.custom_forward` 中，同一层可能在循环深度里跑多遍，或在早退时被跳过，“第 `L` 层的唯一输出”不再有定义。改用 `pllm.residual(layer)` 或 `pllm.layer_logits(...)` 读取每层最后一次访问后的残差流。 |
| 在 `@pllm.custom_forward(tokens="all")` 中对 `layers(...)` 传入早退掩码 `active=` | `Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. ...` | 当某一行在第 $L$ 层早退跳过后续层时，pLLM 会为被跳过的层补齐近似的 Q/K/V。如果在 prefill 阶段对 prompt token 早退，整段 prompt 在高层都会留下近似的 KV，后续每一个 decode token 做注意力时都会读到这些近似值。因此早退 `active=` 仅用于 `tokens="decode"`，prompt 默认跑完整模型。 |
| 可训练的低秩 `pllm.model.Adapter(operator="low_rank")` 使用默认的全零初始化 | `...: add init=pllm.Computed(...) (e.g. random A, zero B); with both low_rank factors at zero every gradient is zero and the adapter never learns` | 低秩适配器计算的增量是 $\text{scale} \cdot B A x$。若 $A$ 和 $B$ 初值同时为 0，反向传播时 $A$ 的梯度正比于 $B^\top=0$，$B$ 的梯度正比于 $A^\top=0$，梯度永远是 0，参数一步也不会更新。使用 `init=pllm.Computed(...)` 将其中一个因子（如 $A$）初始化为随机小值、另一个因子（$B$）初始化为 0。 |

---

## 自测

1. **为什么在 `@pllm.export` 方法里写 `return h - h.mean(dim=0, keepdim=True)` 能通过 `pllm.explain`，却会在 `pllm.testing.check` 时失败？**
   <details><summary>答案</summary>
   <code>h.mean(dim=0, keepdim=True)</code> 仍然是纯张量运算，不包含 GPU 到 CPU 的同步，形状也合法，因此在做静态形状与显存规划的 <code>pllm.explain</code> 中能够通过。但第 0 维（<code>dim=0</code>）存放的是当前步动态打包的不同请求行；沿 <code>dim=0</code> 求均值会让一个请求的输出取决于同批其他请求是谁。<code>pllm.testing.check</code> 在改变同批其他请求输入和打乱行顺序时会捕获这一差异，抛出 <code>ContractError: method ... mixes rows</code>。应改为沿特征维归约：<code>h - h.mean(dim=-1, keepdim=True)</code>。
   </details>

2. **在 `Distill` 负载中，`@pllm.train` 方法和 `@pllm.guide_sampling` 方法都读取了同一个状态 `"distiller"`，为什么 `pllm.explain` 显示前者读的是 `(latest, 0)`，后者读的却是 `(boundary, 1)`？**
   <details><summary>答案</summary>
   默认 <code>publish_lag=1</code> 时，第 $n$ 步在 <code>token.commit</code> 启动的旁路训练任务与第 $n+1$ 步的主流前向并发执行。主流上的采样引导方法 <code>predict</code> 读取在前向边界已经发布好的版本（<code>boundary, 1</code>，即第 $n-1$ 步算出的版本），从而不必停下来等待旁路流完成第 $n$ 步的反向传播与优化器更新；而训练任务 <code>loss</code> 自身始终在最新参数（<code>latest, 0</code>）上继续迭代，保证每一步梯度更新都累加在最新状态上。
   </details>

3. **如果你想在 `@pllm.add_logits` 方法里根据模型原本的词表分布做过滤或缩放，为什么不能把参数写成 `logits: torch.Tensor = pllm.base_logits()`？应该用什么代替？**
   <details><summary>答案</summary>
   按一步的时间线，<code>@pllm.add_logits</code> 运行在采样器抽样之前的 <code>Phase.BEFORE_SAMPLE</code>（<code>model.execute</code> 到 <code>sampler.pre_draw</code> 之间），而 <code>pllm.base_logits()</code> 是在采样器完成抽样后的 <code>token.commit</code> 才提交给采样后任务（如 <code>@pllm.train</code>）读取的反馈输入。要在采样前读取模型最后一层的 logits，应将参数默认值写为 <code>logits: torch.Tensor = pllm.layer_logits((-1,))</code>（取 <code>logits[:, 0]</code>）。
   </details>

---

## 下一步去哪

- 按具体任务查阅写法：[状态](../guides/state.md)、[读取模型内部](../guides/reading.md)、[结果去哪](../guides/results.md)、[在线训练](../guides/training.md)、[改模型](../guides/model-edits.md)、[改 forward 结构](../guides/forward-structure.md)
- 在 CPU 上读懂执行计划与显存账单：[看成本：`pllm.explain`](../guides/explain.md)
- 在 CPU 与 GPU 上验证正确性契约：[检查正确性](../guides/checking.md)
- 了解各项机制的实测吞吐与显存数字：[性能与显存](performance.md)
- 了解为什么某些算法结构不在支持范围内：[设计取舍](design-tradeoffs.md)
- 了解这些机制是怎么被真实约束逼出来的、有过哪些弯路：[技术杂谈](highlights/00-overview.md)
- 查阅所有术语的精确定义：[术语表](../reference/glossary.md)

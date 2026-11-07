# 训练与观测

> **先决条件**：[状态与记忆](state-and-memory.md)；教程
> [01](../tutorials/01_observe_probe.ipynb)（观测）、[03](../tutorials/03_decode_training.ipynb)
> （decode 时训练）、[05](../tutorials/05_host_feedback.ipynb)（宿主批次训练）。
>
> **本文回答**：怎么声明一个训练任务？loss 按什么约定写？怎么把 loss/梯度/
> hidden 拿出来看？

读完教程之后，真正做实验时你会遇到三件具体的事：写一个能被框架更新的训练
任务、把它接到上游计算的可微边上、以及在训练中把中间值安全地拿出来看。本文
按这三件事的顺序展开。

## 训练任务的最小形

你想给某个状态（比如一个 head）做在线学习。在 PLLM 里，这意味着写一个普通
的 Tensor 方法，再用两个装饰器告诉框架它何时运行、更新谁：

```python
import torch
import pllm

@pllm.on(pllm.Phase.AFTER_SAMPLE)
@pllm.train(state="head")
def learn(
    self,
    h: torch.Tensor = pllm.hidden(0),
    target: torch.Tensor = pllm.hidden(-1),
    w: torch.Tensor = pllm.parameter("head", "w"),
) -> torch.Tensor:
    prediction = torch.bmm(h.float().unsqueeze(1), w).squeeze(1)
    return (prediction - target.float()).square().mean(1)   # 每行一个 loss
```

`@pllm.on(pllm.Phase.AFTER_SAMPLE)` 把这个任务放进采样后的相位窗口，
`@pllm.train(state="head", ...)` 声明它训练名为 `"head"` 的状态。函数体只写
前向数学——这是第一个值得记住的约定。

优化器写在状态上（这里是 `pllm.AdamW`），装饰器只说训练哪个状态。训练什么
由方法的输入决定：像上面这样读运行中的行（hidden、参数），就是逐请求训练；
读 `pllm.batch(...)` 字段，就是在宿主送来的批次上训练（`WeightedRowBatch`、
`CausalTrainBatch` 用 AdamW，`WeightedCEBatch` 用 `pllm.SGD` 加 `reset=`）。
优化器与输入对不上时，声明期就会告诉你该换成哪种优化器。

为什么只写前向就够了？绑定器从你返回的标量出发自动微分到 `Parameter` 输入，
`@train(state=...)` 本身就是对该状态的更新授权，Adam 的步数、动量与衰减语义
都随状态走。你不需要写 `backward()`，也不需要自己管优化器。

第二个约定在返回值上：逐请求训练的方法**每行返回一个 loss**，不要自己把行
压成一个数，归约交给框架。默认 `reduction="mean"` 对本步每个状态选中的行取均值；
`reduction="sum"` 对这些行求和。各状态的 loss **相加**后微分，不会把不同状态的行一起平均。
每个状态每个引擎步更新一次，prompt 行每个 prefill 分块训练一次。在宿主批次
（`WeightedRowBatch`、`CausalTrainBatch`、`WeightedCEBatch`）上训练的方法自己归约；
非默认的 `reduction=` 会在声明期报错。

第三个约定容易踩坑：**零梯度不等于不更新**。被授权的状态即使梯度为零，也会
推进优化器的步数、动量与衰减。如果你真的想跳过某些行的更新，不要靠让梯度
变零，而要在选择器层面把行排除——`@train(state=..., rows=...)` 或
`@train(state=..., update_rows=...)` 的选择器就是干这个的。注意：独立的请求训练和与
`@pllm.guide_sampling` 并存的训练会更新每个选中的状态，这两种情况下声明 `rows=`/`update_rows=` 会被拒绝。

最后，如果你想让训练影响采样本身：加一个 `BEFORE_SAMPLE` 相位的任务并声明
`@pllm.guide_sampling(beta=...)`，它返回的 hidden 预测会经原始采样器链接
参与引导。这里有一条硬性前提：训练负载必须有一个声明的后端效果，否则没有
挂载点。

## 一阶梯度边：`differentiate`

上一个例子的训练任务直接从 `pllm.hidden` 读输入。但更常见的情况是：`@train`
任务消费的是上游另一个操作的结果（`pllm.result`）——比如先有一个操作产出特征，训练任务
拿这个特征算 loss。这时上游操作必须显式声明保留哪些导数边，否则梯度到不了
它：

```python
@pllm.differentiate(inputs=("features",), parameters=False)   # 输入可微、参数冻结
def encode(self, features: torch.Tensor = pllm.result("extract")) -> torch.Tensor:
    return torch.tanh(features)
```

这段声明的意思是：这个操作保留 `features` 输入的导数边，参数导数冻结。有几
个边界值得提前说清楚，免得日后误判它的能力：

- 这是**单次调用内的一阶 autograd**：一个优化器状态、一个训练终点。它不是
  穿透 vLLM 或跨越 replay/发布边界的全模型反向。
- `parameters=False` 只冻结这个操作的参数导数，并不永久冻结优化器；同一个
  共享状态在另一个操作里照样可以可微。
- 如果你的外部浮动输入需要梯度，从
  `result.input_gradients[(operation, argument)]` 读回。

## 训练的归约：mean 还是 sum

用 `@pllm.train(state=..., reduction=...)` 定义在线请求训练时，方法针对当前批次的请求行返回一个形状为 `[rows]` 的逐行损失向量。pllm 的步进模型是：**每个状态在每个引擎步（engine step）执行一次优化器更新**，更新前先将该状态在本步中被选中的行归约为一个标量损失。

如果你写的是最常见的、按默认 `request_id` 隔离的 decode 阶段训练，每个请求在每个引擎步里只有 1 行（该状态本步选中行数 $n_s = 1$）。此时对 1 个数求均值和求和完全一样，`reduction="mean"` 与 `reduction="sum"` 没有任何区别，你不需要为此操心。

只有当**同一个状态在同一个引擎步内对应多行**时，`reduction` 才会真正决定同一状态内各行的相对权重，以及跨步之间的梯度幅度：

- 包含多个 token 的 **prompt 行**（prefill 阶段）；
- 多个请求通过共享 `key_fields` 映射到同一个状态；
- 所有请求共享同一个 `InstanceState`；
- 其它在单步内为同一状态产生多行的场景。

### 机制与怎么选

`@pllm.train` 的 `reduction` 只在每个状态内部归约本步选中的行，**两个选项都保持不同状态之间相互独立**：

- `reduction="mean"`（默认）：对每个状态 $s$ 在本步选中的 $n_s$ 行取均值，即 $\frac{1}{n_s}\sum_{r \in s} \ell_r$。
- `reduction="sum"`：对每个状态 $s$ 在本步选中的行直接求和，即 $\sum_{r \in s} \ell_r$。

选择哪一个，取决于你使用的优化器以及多行之间的关系：

1. **使用 `pllm.AdamW` 时，默认选 `"mean"`**
   AdamW 对全局常数缩放不敏感，但会受**不同步之间梯度幅度跳变**的影响。在 prefill 步中，一个请求可能一次进入 $P$ 个 prompt 行；如果用 `"sum"`，prefill 步的梯度幅度会比单行 decode 步放大约 $P$ 倍，导致 AdamW 的二阶矩估计 $v$ 被放大约 $P^2$ 倍，进而压制后续大量 decode 步的有效步长（这是按 Adam 更新公式推出的结论，尚未专门实测）。`"mean"` 消除了单步内行数 $n_s$ 的影响，让 prefill 步与 decode 步的梯度尺度保持一致。
2. **使用 `pllm.SGD` 且需要每 token 学习率恒定时，选 `"sum"`**
   SGD 没有二阶矩历史；如果你希望每个 token 对参数产生的更新量只取决于该 token 自身的梯度和学习率，而不因本步恰好有多少行同批到达而被 $1/n_s$ 摊薄，应选 `"sum"`。
3. **共享 `key_fields` 的在线组训练，选 `"sum"`**
   当一组兄弟采样请求共享同一个状态时，不同样本往往在不同步提前结束。如果使用 `"mean"`，随着部分样本结束退出，分母 $n_s$ 变小，剩余存活样本中每个 token 的权重会被动放大；使用 `"sum"` 则保证每个存活 token 始终贡献恒定的一份梯度。

### 常用归一化配方

围绕 `@pllm.train` 的返回值与参数，可以直接组合出常见的训练变体：

- **逐行加权**（如优势值、置信度、门控权重 $w_r$）：直接在训练方法里返回 `weight * loss`（即 $w_r \cdot \ell_r$），配合 `reduction="mean"` 或 `"sum"` 即可。
- **Dr.GRPO 式的常数长度归一化**：希望除以固定常数 $L_{\max}$ 而不是当前步的动态行数 $n_s$ 时，使用 `reduction="sum"`，并在返回时直接将损失乘以常数 $1 / L_{\max}$。
- **排除特定行：优先用 `rows=`，不要用零权重配 mean**：在 `reduction="mean"` 下，返回 `0.0 * loss` 的行**依然计入分母 $n_s$**，会平白稀释其它有效行的梯度；`rows=` 会把这一行同时从分子和分母里去掉。注意 `rows=` 只在请求训练与模型效果或 `@pllm.add_logits` 组合时可用；独立的请求训练、以及与 `@pllm.guide_sampling` 并存的训练会训练每个选中状态的所有行。这两种场景下想排除某些行，可以用 `reduction="sum"` 加零权重：求和时零权重行不贡献，也不改变其它行的权重。
- **时间衰减与动量**：跨步的指数移动平均（EMA）或衰减属于优化器职责（如 AdamW 的 $\beta_1$、`weight_decay`）或显式正则项，不应试图通过步内归约来实现。

### Host batch 训练：自己返回标量并完成归一化

当方法接收 `WeightedRowBatch`、`CausalTrainBatch`、`WeightedCEBatch` 字段或作用于因果模型时，它走的是 host batch 训练（例如用 `pllm.AdamW` 走一步，或配合 `reset` 用 `pllm.SGD` 走 `steps` 步）。这类方法**直接返回一个标量**并自行负责归一化，不能传入非默认的 `reduction=`。

原因在于许多序列级强化学习算法需要完整序列长度 $|o_i|$ 才能归一化，在线逐 token 步进时不满足因果性，只能在凑齐 host batch 后按公式归约。四种常见归约的区别在于如何对待组大小 $G$ 与序列长度 $|o_i|$：

| 算法 | 标量损失归约公式 | 归一化特点 |
| :--- | :--- | :--- |
| **GRPO** | $\frac{1}{G}\sum_{i=1}^{G} A_i \frac{1}{\|o_i\|} \sum_{t} \ell_{i,t}$ | 先在每条序列内按自身长度 $\|o_i\|$ 取均值，再对组内 $G$ 条序列取均值；依赖最终长度，只能用 host batch |
| **DAPO** | $\frac{\sum_{i=1}^{G} \sum_{t} A_i \ell_{i,t}}{\sum_{i=1}^{G} \|o_i\|}$ | 按整个批次的总 token 数求全局均值，长序列贡献更多 token |
| **RLOO / REINFORCE** | $\frac{1}{G}\sum_{i=1}^{G} A_i \sum_{t} \ell_{i,t}$ | 序列内对 token 求和，组间按 $G$ 取均值 |
| **Dr.GRPO** | $\frac{1}{G}\sum_{i=1}^{G} A_i \frac{1}{L_{\max}} \sum_{t} \ell_{i,t}$ | 用固定常数 $L_{\max}$ 代替序列长度 $\|o_i\|$，消除长度偏置 |

### 两个容易混淆的边界

- **Prompt 行训练的切块依赖 vs. `train_in_forward(chunk=)`**：由于 `@pllm.train` 是每个引擎步走一步优化器，prompt 行在开启 chunked prefill 时会“每个 prefill 分块训练一次”，因此更新轨迹取决于调度器如何切块。如果你的算法（如测试时训练中的小批量更新）要求严格按序列位置对齐的固定块均值、不受引擎调度切块影响，应改用 `train_in_forward(chunk=...)`。
- **跨请求耦合靠状态声明，不靠 `reduction`**：`reduction` 只决定单个状态内部在本步的多行如何合并，永远不会把不同请求的状态混在一起。如果你确实需要让多个请求共享一次参数更新，应当通过将状态声明为 `InstanceState`，或在请求状态上声明共享的 `key_fields` 来表达。

## 观测：把值拿出来

训练跑起来之后，下一个问题自然是"怎么看到里面的值"。PLLM 提供两条通道，
它们解决不同的问题，按用途选。

### `@pllm.export`：声明式导出

如果你已经知道某个方法的输出值得长期盯——比如每两步看一眼特征——在创作期
直接声明导出效果：

```python
@pllm.export(name="features", every=2, max_rows=4, max_cols=896)
def features(self, h: torch.Tensor = pllm.hidden(0)) -> torch.Tensor:
    return h
```

导出值送到负载的 `on_export(event)` 方法，按顺序、在执行进程里调用；在这里打印、
落盘或累积（见 [SAE 导出示例](../../examples/sae_export.py)）。

这条通道是方法自带的输出效果：有界、按步长门控、给出 CPU 自持副本。三个
数字约束需要记住：

- `max_rows`/`max_cols` 会同时截断输入帧与输出帧（先按行选，再按列截）；
- 每个导出事件的 `dropped_rows` 精确记录因 `max_rows` 被截去的行数；`export_dropped` 仍只计队列丢弃的事件数。
- 每事件字节上限 32 KiB，即 `max_rows × max_cols × 8 ≤ 32768`；
- 导出事件进容量 256 的全局有界队列，溢出时丢最旧的事件并计入 `dropped`。

### trace 选择器：运行时观测

如果你不想改负载，只想在运行时临时挑某个任务来看，用 trace 选择器：

```python
# check-docs: skip: requires the running CUDA engine llm, prompts and params
from pllm.io import TraceSelection

llm.select_trace(TraceSelection(names=("loss",), tensors=True, max_tensor_bytes=32768))
llm.generate(prompts, params)
snapshot = llm.trace_snapshot()          # 含 tensors=False 时的 loss/梯度元数据
llm.select_trace(TraceSelection())            # 关掉
```

选择器开启时，训练任务会按名记录它的 Hidden 输入、`loss`（按 lineage）与
`parameters.grad`（扁平梯度）。`llm.select_trace(TraceSelection())` 用空选择器
关闭观测。

有一个用法细节必须注意：trace 是**跨 generate 全局**的队列，不是每次
generate 重置。所以要拿"这一次"的新鲜事件，用游标切片：

```python
# check-docs: skip: continues the running CUDA engine trace example
before = llm.trace_snapshot()
cursor = before.dropped + len(before.events)          # 绝对事件索引
llm.generate(prompts, params)
after = llm.trace_snapshot()
fresh = after.events[max(0, cursor - after.dropped):]
```

`dropped` 补偿了队列溢出丢弃的最旧事件，让游标始终是绝对索引。

`trace_snapshot()` 只在两次 `generate` 之间调用，不要在 `generate` 运行时从
另一个线程调用。要持续消费导出值，用上面的 `on_export`。

## 诊断与自检

最后一块拼图是运行时自检。`llm.diagnostics()` 给出 `loss_count`、
`update_count`、`trained_rows` 等计数：训练跑了几步、更新了几次、多少行参与，
一眼就能看出训练有没有真的发生。

另一个便宜的自检是零状态确定性：两个全新 route 的首步 loss 应当完全相等。
这是验证记忆通道最省事的检查，教程 05 有完整示范。

精确语义（update 掩码授权、empty 选择、detach 规则、导数掩码、行域）见
[创作契约](../reference/authoring-contract.md#训练与梯度)。

### Capture an ordinary tensor step

CUDA backends automatically capture ordinary row-wise prediction, AdamW training,
computed field writes, and imports. ESamp uses this framework path; there is no task
capture switch. The model's `cuda_graphs` setting remains independent. A training graph contains parameter gathering,
the Python method's tensor operations, loss reduction, autograd, memory writes, and the
optimizer. The CPU still allocates private versions and publishes them through PTTS.
AdamW reads source slots and writes destination slots, so the five optimizer fields need
no preceding COW copy. Other written fields are copied inside the graph.

Captured methods must compute independently along their first (row) axis. They see the
maximum operation row count, including padding. Return **one loss per row** for
both `reduction="mean"` and `reduction="sum"`; do not reduce the row axis in Python
or mix rows with batch normalization, attention, or another cross-row operation. The
framework gives padding zero weight and a sink parameter row that is never optimized.
Shared instance parameters use one live state; request parameters use the admitted state
bound. Eager execution uses the same step body with the invocation's actual dimensions.
`rows=` and `update_rows=` selectors stay boolean masks on the device: the method runs on every
row, so it must accept unselected rows, and their losses get zero weight. A state without a
selected row still publishes, with its fields byte-for-byte unchanged, so selection never
reads the mask back to the host.
Captured training requires one loss per row.

One pinned int64 table carries row indices, masks, counts, last-row memory indices and
source/destination page slots. Hidden reads address the fixed backend tap buffers inside
the graph. Sampler operands use addresses and strides in that same table, so changing
sampler allocations do not force recapture or a host readback. Prediction results and
loss/gradient outputs are copied out before a later replay can overwrite them.

The first invocation warms up on a side stream and captures into a private graph pool.
By default, each tensor task's capture reservation is derived at build from its maximum-shape
allocation trace, including index tables and owned staging. It conservatively assumes no allocator
reuse; actual graph residency is usually smaller. An explicit `cuda_graph_pool_budget_bytes` sets a per-task
limit and fails at build if it cannot cover that reservation, naming the required value.
Import-only store graphs allocate no tensors during replay and reserve their exact fixed storage.
Model overlay and custom-forward graphs retain a 64 MiB default pool reservation when the option is unset; exact-decode overlays add shape-derived workspace allowances to the model pool.
Overlay tiles keep a fixed per-task capture budget; a tile whose conservative reservation does not fit executes eagerly.
Replacing hidden buffer storage starts a new
cold capture. Subsequent invocations with fixed tap buffers perform no host-device wait.

Each resource has its own source slots and row selection in the table. Only completed
prompt rows publish prefill-produced fields. All tuple outputs are retained, and changing
import allocations are copied into fixed staging before replay. External imports are
checked for finite values at intake. Computed writes check finiteness inside the graph;
a nonfinite CUDA result fails the worker, rather than raising a recoverable Python
validation error. Private stores are disabled during warmup and capture.

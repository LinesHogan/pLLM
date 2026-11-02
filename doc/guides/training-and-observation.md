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
@pllm.on(pllm.Phase.AFTER_SAMPLE)
@pllm.train(state="head", reduction="state_mean")
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
压成一个数，归约交给框架。`reduction="mean"` 对所有行取均值；
`reduction="state_mean"` 先对每个状态的行取均值，再把各状态的均值**相加**后
微分——每个请求的状态各自按自己的行数归一，不被别的请求稀释。在宿主批次
（`WeightedRowBatch`、`CausalTrainBatch`）上训练的方法则返回一个标量 loss。

第三个约定容易踩坑：**零梯度不等于不更新**。被授权的状态即使梯度为零，也会
推进优化器的步数、动量与衰减。如果你真的想跳过某些行的更新，不要靠让梯度
变零，而要在选择器层面把行排除——`@train(state=..., rows=...)` 或
`@train(state=..., update=...)` 的选择器就是干这个的。

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
- 每事件字节上限 32 KiB，即 `max_rows × max_cols × 8 ≤ 32768`；
- 导出事件进容量 256 的全局有界队列，溢出时丢最旧的事件并计入 `dropped`。

### trace 选择器：运行时观测

如果你不想改负载，只想在运行时临时挑某个任务来看，用 trace 选择器：

```python
from pllm.execution.trace import TraceSelection

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
both `reduction="mean"` and `reduction="state_mean"`; do not reduce the row axis in Python
or mix rows with batch normalization, attention, or another cross-row operation. The
framework gives padding zero weight and a sink parameter row that is never optimized.
Shared instance parameters use one live state; request parameters use the admitted state
bound. Eager execution uses the same step body with the invocation's actual dimensions.
`rows=` and `update=` selectors stay boolean masks on the device: the method runs on every
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
`capture_memory_bytes` reserves resident memory **per task**; the captured pool and index
table and owned import/write staging buffers are checked against that reservation.
Import-only store graphs allocate no tensors during replay and reserve their exact
fixed storage, up to this limit. Replacing hidden buffer storage starts a new
cold capture. Subsequent invocations with fixed tap buffers perform no host-device wait.

Each resource has its own source slots and row selection in the table. Only completed
prompt rows publish prefill-produced fields. All tuple outputs are retained, and changing
import allocations are copied into fixed staging before replay. External imports are
checked for finite values at intake. Computed writes check finiteness inside the graph;
a nonfinite CUDA result fails the worker, rather than raising a recoverable Python
validation error. Private stores are disabled during warmup and capture.

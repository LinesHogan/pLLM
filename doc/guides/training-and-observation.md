# 训练与观测

> **先决条件**：[状态与记忆](state-and-memory.md)；教程
> [01](../tutorials/01_observe_sae.ipynb)（观测）、[02](../tutorials/02_custom_head.ipynb)
> （训练）、[05](../tutorials/05_decode_training.ipynb)（decode 在线训练）。
>
> **本文回答**：怎么声明一个训练任务？loss 按什么约定写？怎么把 loss/梯度/
> hidden 拿出来看？

## 训练任务的最小形

```python
@pllm.on(pllm.Phase.AFTER_SAMPLE)
@pllm.train(state="head", reduction="state_mean")
def learn(self, h: Annotated[torch.Tensor, pllm.Hidden(0)],
          target: Annotated[torch.Tensor, pllm.Hidden(-1)],
          w: Annotated[torch.Tensor, pllm.Parameter("head", "w")],
          groups: Annotated[torch.Tensor, pllm.StateIndex("head")]) -> torch.Tensor:
    losses = ((h.float() @ w) - target.float()).square().mean(1)   # 每行一个 loss
    counts = torch.bincount(groups).to(losses)
    return losses.new_zeros(counts.shape).index_add(0, groups, losses) / counts
```

约定三条：

1. **返回每个状态一个标量**，不是把所有行压成一个数。`StateIndex` 给出每行
   属于哪个状态，`index_add + bincount` 是标准的按状态均值写法（"state_mean"）。
   绑定器微分的是这些标量的**和**，不是跨状态均值。
2. **只写前向**。绑定器从返回的标量出发自动微分到 `Parameter` 输入；
   `@train(state=...)` 授权该状态被更新（含 Adam 步数/动量/衰减语义）。
3. **零梯度≠不更新**。被授权的状态即使梯度为零也推进优化器；
   要跳过更新，在选择器层面把行排除（`rows=`/`update=`）。

想让训练影响采样，加一个 `BEFORE_SAMPLE` 任务并声明 `@pllm.guide_sampling(beta=...)`：
返回的 hidden 预测经原始采样器链接参与引导。训练负载必须有一个声明的后端效果，
否则没有挂载点。

## 一阶梯度边：`differentiate`

`@train` 消费上游操作的 `Result` 时，用 `@differentiate` 在上游声明要保留哪些
导数边：

```python
@pllm.differentiate(inputs=("features",), parameters=False)   # 输入可微、参数冻结
```

这是单次调用内的一阶 autograd：一个优化器状态、一个训练终点；不是穿透 vLLM
或跨越 replay/发布边界的全模型反向。`parameters=False` 只冻结这个操作的参数
导数，不永久冻结优化器。外部浮动输入的梯度从
`result.input_gradients[(operation, argument)]` 读回。

## 观测：把值拿出来

两条通道，用途不同：

**`@pllm.export`（声明式导出）** —— 方法自带的输出效果，有界、按步长门控、
CPU 自持副本：

```python
@pllm.export(name="features", every=2, max_rows=4, max_cols=896)
```

- `max_rows`/`max_cols` 会同时截断输入帧与输出帧（先按行选，再按列截）；
- 每事件字节上限 32 KiB：`max_rows × max_cols × 8 ≤ 32768`；
- 事件进容量 256 的全局有界队列，溢出丢最旧并计 `dropped`。

**trace 选择器（运行时观测）** —— 不改负载，运行时挑任务看：

```python
from pllm.execution.trace import TraceSelection

llm.observe(TraceSelection(tasks=(task_id,), tensors=True, max_tensor_bytes=32768))
... generate ...
snapshot = llm.trace_snapshot()          # 含 tensors=False 时的 loss/梯度元数据
llm.observe(TraceSelection())            # 关掉
```

训练任务在选择器开启时按名记录 Hidden 输入、`loss`（按 lineage）与
`parameters.grad`（扁平梯度）。trace 是**跨 generate 全局**的队列 —— 用游标
切片拿本次的新鲜事件：

```python
before = llm.trace_snapshot()
cursor = before.dropped + len(before.events)          # 绝对事件索引
... generate ...
fresh = llm.trace_snapshot().events[max(0, cursor - snapshot.dropped):]
```

消费 trace 用后台线程 drain（教程 01 的模式），不要在采样路径上同步等。

## 诊断与自检

`llm.worker_state().diagnostics` 给出 `loss_count`/`update_count`/`fallback_count`
等计数。**`fallback_count` 必须始终为零** —— 任何非零都说明执行逃出了签名
原语。零状态确定性（两个全新 route 的首步 loss 完全相等）是验证记忆通道
最便宜的检查，教程 05 有完整示范。

精确语义（update 掩码授权、empty 选择、detach 规则、导数掩码、行域）见
[创作契约](../reference/authoring-contract.md#训练与梯度)。

# 编写 Consumer

Consumer 是一个用 tLLM 声明词汇标注方法的 `torch.nn.Module`。你声明**什么**依赖**什么**；编译器决定**在哪里**、**如何**执行。

## 声明词汇

```python
from tllm.workgraph.frontend import extern, flow, hidden, local, result, sink
```

| 声明 | 含义 |
|---|---|
| `@flow(inputs, outputs, state=...)` | 一个读取引擎内部张量和/或状态、产出输出的 stage。 |
| `hidden(layer, site, phase)` | 从残差流读输入（如 `layer=0` / `layer=-1`、`site="block_output"`、`phase="decode"`）。 |
| `sink(name)` | 发布到命名 sink 的输出（其他 consumer 或 sampler bridge 订阅它）。 |
| `local()` | 仅在程序内部消费的输出（如喂给 extern）。 |
| `@extern(inputs, after, state, deadline)` | 训练步：改写命名状态，在依赖满足后调度，受 timing deadline 约束。 |
| `result(flow_name, output_name)` | 携带另一个 flow 输出**及其 autograd 图**的边 —— 编译器由此知道需要 grad 模式。 |

状态名（`state="distiller"`）指 **state bank**：runtime 持有的带版本、按 scope（shared 或 per-request）划分的参数 + optimizer 状态存储。你的模块永远不管理 buffer、stream 或 event。

## 完整示例

见 `tllm/consumers/esamp/program.py` —— 约 80 行的 ESamp 算法。固定结构：

1. `__init__` 构建模型和一个普通 `torch.optim` optimizer（SGD / Adam / AdamW，单参数组）。
2. 一个或多个 `@flow` 方法在 hidden-state 输入上计算。
3. 一个 `@extern` 方法消费 `result(...)` 边并执行 optimizer step。它的 Python 函数体是**参考 oracle**：编译器把 optimizer 降级为函数式 transition，并在编译期校验两者一致（dtype 容差内）。

## 编译器推导什么

以下全部从声明的 DAG 推出 —— 不需要为每个 consumer 手写：

- **切片**：每个 flow 体经 `torch.export` 导出；不支持的结构（数据相关控制流、非 Tensor leaf）编译期拒绝。
- **梯度模式**：若到期 extern 有指向某 flow 输出的 `result()` 边，该 flow 以 grad 模式执行一次，输出同时服务 sink（detached）和 `autograd.grad`。无双倍前向。
- **lane 放置**：写状态的 extern 走专用训练 lane；只读调用（`run_flow`，sampler bridge 用）走低优先级 reader lane；decode 主流只等它实际消费的输出事件。
- **CUDA Graph 捕获**：当一步是对持久状态 buffer 的固定 DAG（同 flow、同 batch 形状），runtime 捕获一次后 replay —— 约 3–5 次 kernel launch 替代约 70–90 次。捕获时自校验正确性（`PG146`）。
- **函数式 optimizer transition**：optimizer 变成版本化 slot buffer 上的纯张量更新，因此可以被 fence、被 replay。

## 错误码

编译与运行期拒绝使用 `WorkGraphError` 上的稳定错误码：

| 代码 | 条件 |
|---|---|
| PG100 | `result()` 引用未知 flow/输出 |
| PG110 | 依赖成环 |
| PG120 | 重复 producer / binding / state 名 |
| PG130 | export 图含不支持结构 |
| PG140 | 跨 stage 保存 autograd 图 |
| PG145 | oracle 分歧：extern Python 体与函数式 transition 不一致 |
| PG146 | 捕获的 CUDA Graph replay 与 eager transition 分歧 |
| PG150 | 未声明 mutation / 未标注 RNG |
| PG160 | 不支持的 optimizer 或其组合 |
| PG170 | 无合法 hook 可 materialize 声明的 timing 窗口 |
| PG180 | request-scoped state 的不规则 row mapping |
| PG190 | 任何静默 fallback 企图（一律 raise；不存在该路径） |

契约是：**编译期大声失败，绝不静默降级**。

## Sampler 引导

如果 consumer 还要引导采样，实现一个 `SamplerModifierProvider`（见 `tllm/runtime/sampler_bridge/`）。它在 reader lane 上调用 `ProgramConsumer.run_flow(...)`，返回候选级 logit 调整；runtime 在 top-k/top-p/min-p 过滤之后应用。ESamp 的 distiller intervention 是参考实现。

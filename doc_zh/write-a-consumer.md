# 编写 Consumer

## Native workload 优先

新代码应使用 `tllm.workload`：以 `@workload`、`@task`、typed value、effect、timing window
及可选 affine domain/access 声明。Rust 随后生成并签名 `ExecutionRegion`；Python 无权拆分它、
选择顺序、推断 capture 行为，或增加逐 task stream placement。`RegionCaptureMode` 是 Rust
签名的闭合 enum，只有显式跨 Region 依赖才会生成 fence/event。完整契约见
[附加载荷编译器](workload-compiler.md)。

本页余下部分介绍为 legacy adapter 保留的 Program/`BaseConsumer` API。当 workload 尚未完成
native takeover 时它仍然有用，但其中的 lane、stream、event 与 capture 行为不是当前优化模型。

Consumer 是一个用 tLLM 声明词汇标注方法的 `torch.nn.Module`。你声明**什么**依赖**什么**；编译器决定**在哪里**、**如何**执行。

## Legacy Program 声明词汇

```python
from tllm.workgraph.frontend import extern, flow, hidden, local, provided, result, sink
```

| 声明 | 含义 |
|---|---|
| `@flow(inputs, outputs, state=...)` | 一个读取引擎内部张量和/或状态、产出输出的 stage。 |
| `hidden(layer, site, phase)` | 从残差流读输入（如 `layer=0` / `layer=-1`、`site="block_output"`、`phase="decode"`）。 |
| `provided()` | 调用方以普通张量直接供给的输入（不经引擎 tap）。consumer 用 `provided_bundle(...)` 自构 bundle；lease entry 键即声明的输入名。 |
| `sink(name)` | 发布到命名 sink 的输出（其他 consumer 或 sampler bridge 订阅它）。 |
| `local()` | 仅在程序内部消费的输出（如喂给 extern）。 |
| `@extern(inputs, after, state, deadline)` | 训练步：改写命名状态，在依赖满足后调度，受 timing deadline 约束。 |
| `result(flow_name, output_name)` | 携带另一个 flow 输出**及其 autograd 图**的边 —— 编译器由此知道需要 grad 模式。 |

状态名（`state="distiller"`）指 **state bank**：runtime 持有的带版本、按 scope（shared 或 per-request）划分的参数 + optimizer 状态存储。你的模块永远不管理 buffer、stream 或 event。

## 薄 consumer 外壳

编译出的 `ProgramConsumer` 由一个小封装驱动：声明引擎该喂什么（`flows()`）、拿到真实形状后懒编译、然后委托。所有内置 consumer 都用这个模式：

```python
class MyConsumer(BaseConsumer[MyConfig]):
    def _ensure_compiled(self, hidden: torch.Tensor) -> None:
        if self._compiled is not None:
            return
        program = MyProgram(hidden_size=int(hidden.shape[1]), ...)
        rows = 2  # export 要求示例行数严格大于动态维下界
        self._compiled = compile_program(
            program,
            states={"adapter": StateSpec(scope=StateScope.REPLICA, optimizer=torch.optim.Adam)},
            examples={"forward": (torch.zeros(rows, hidden.shape[1]), ...)},
            dynamic_dims={
                # 每个行动态输入一个 DynamicDim，各自为独立对象
                "forward": tuple(
                    DynamicDim(value_path=(i,), dimension=0, lower_bound=1, upper_bound=65536)
                    for i in range(num_inputs)
                ),
            },
        )

    def train(self, hidden: torch.Tensor, ...) -> None:
        self._ensure_compiled(hidden)
        bundle = provided_bundle(consumer_id=self.consumer_id, step_id=self._step_id,
                                 entries={"hidden": hidden, ...})
        self._compiled.consume_flow(self._compiled.flow_ordinal("forward"), bundle, None)

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        self._ensure_compiled(hidden)
        return self._compiled.run_flow("predict", hidden)["score"]
```

经验法则：

- **张量数学进 Program；控制逻辑保持纯 Python。** 评判、投票、队列、生命周期状态机、行分组——全在 consumer 上，不在 flow 里。
- **标量/开关类输入做成逐行张量**（`head_idx: (rows,)`、`scale: (rows, 1)`）。bundle entry 必须共享行数；逐行选择（`index_select` + `bmm`）可以干净地 export，而 Python 索引 `ModuleList` 不行。
- **每个 `DynamicDim` 对象只用于一个输入**——同一个实例用两次会在 capture 时报 "declared more than once"。
- **统计从编译出的 consumer 读**（`update_count`、`loss_count`、`loss_sum` 增量）；不要自己重造计数器。
- 程序在首个真实输入处懒编译并**fail closed**：不支持的配置在编译期显式报错，绝不静默 fallback。

参考实现（从小到大）：`tllm/consumers/ttrl/`（reward 模型）、`tllm/consumers/tlm/`（logits 适配，online/offline 缓冲）、`tllm/consumers/fttt/`（带纯 Python 生命周期的反馈编辑）、`tllm/consumers/override/`（sampler bridge 驱动的多头 reweighting）、`tllm/consumers/esamp/`（引擎 tap 的 `hidden()` 输入、sink 发布、per-request state scope）。

## 完整示例

见 `tllm/consumers/esamp/program.py` —— 约 80 行的 ESamp 算法。固定结构：

1. `__init__` 构建模型和一个普通 `torch.optim` optimizer（SGD / Adam / AdamW，单参数组）。
2. 一个或多个 `@flow` 方法在 hidden-state 输入上计算。
3. 一个 `@extern` 方法消费 `result(...)` 边并执行 optimizer step。它的 Python 函数体是**参考 oracle**：编译器把 optimizer 降级为函数式 transition，并在编译期校验两者一致（dtype 容差内）。

## Legacy runtime 推导什么

以下全部从声明的 DAG 推出 —— 不需要为每个 consumer 手写：

- **切片**：每个 flow 体经 `torch.export` 导出；不支持的结构（数据相关控制流、非 Tensor leaf）编译期拒绝。
- **梯度模式**：若到期 extern 有指向某 flow 输出的 `result()` 边，该 flow 以 grad 模式执行一次，输出同时服务 sink（detached）和 `autograd.grad`。无双倍前向。
- **lane 放置**：写状态的 extern 走专用训练 lane；只读调用（`run_flow`，sampler bridge 用）走低优先级 reader lane；decode 主流只等它实际消费的输出事件。这是兼容 runtime 行为。
- **CUDA Graph 捕获**：符合条件的 legacy 固定 DAG 可以 capture/replay。native capture 则遵从已签名的 `RegionCaptureMode` 和 Region graph safety。
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

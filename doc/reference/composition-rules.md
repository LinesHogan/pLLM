# 组合规则

一个负载里的效果各自声明自己在一步中的位置和读写的状态，pLLM 按位置排序，只拒绝真正冲突的组合，并在构建时
（`pllm.explain` 或 `pllm.make_llm`）报错、说明改法。本页列出哪些组合可以并存、哪些被拒绝以及原因。组合的写法
见[组合负载](../guides/composition.md)。

## 可以并存

| 组合 | 条件与说明 |
| --- | --- |
| 多个 `@pllm.train`，各训各的状态 | 同一种训练方式（都在请求行上，或都在宿主批次上） |
| `@pllm.export` 与任何效果 | 导出只读 |
| 多个 `@pllm.add_logits`（包括来自不同组件的） | 先在过滤前相加，再应用采样引导；加法与顺序无关 |
| `@pllm.add_logits` / `@pllm.guide_sampling` 与 `Adapter`、`ActivationPatch`、`@pllm.train_in_forward` | 采样效果在 forward 之后，模型改写在 forward 之中 |
| 一个 `@pllm.custom_forward` 与一个训练 overlay、一个 `ActivationPatch` | custom forward 运行的每一层都经过它们；`Adapter` 的增量在每次投影调用时施加 |
| `@pllm.custom_residual` 与 `@pllm.custom_forward` | custom forward 运行的每一层也经过定制残差（带状态的子层除外，见下表） |
| `pllm.residual` / `pllm.layer_logits` 与 `@pllm.custom_forward` | 读每层最后一次算出的残差流 |
| 同一状态的生产者与消费者 | 例如 `@pllm.write` 或 `@pllm.train` 写、`Adapter` 读；一个状态每步只能有一个写入方法 |

## 被拒绝的组合

| 组合 | 为什么 | 怎么改 |
| --- | --- | --- |
| 一个方法上叠两个同类装饰器（如 `@pllm.export` 和 `@pllm.train`） | 后一个会覆盖前一个，其中一个效果悄悄失效；不同效果的默认时机也不同 | 每个效果一个方法，用 `pllm.result(name)` 共享计算 |
| 两个 `@pllm.guide_sampling`（包括来自不同组件的） | 采样器每步只按一个预测重新加权；两个预测怎样合成是算法的选择 | 在一个方法里合成一个预测；其余修正用 `@pllm.add_logits` |
| 请求行训练与宿主批训练在同一个负载（包括组件） | 两种训练的更新时机和显存预留不同，一个负载只有一种训练方式 | 分到两个引擎；只给宿主代码读的状态可以用 `pllm.host.make_host` 单独训练 |
| 单独的请求行训练、或与 `@pllm.guide_sampling` 并存的训练，加 `rows=` / `update_rows=` | 这条路径每步给每个状态做一次更新并发布新版本，不支持按行跳过 | 在 `@pllm.guide_sampling(rows=...)` 上选行，或用 `reduction="sum"` 并把要排除的行的 loss 乘 0（见[在线训练](../guides/training.md)） |
| `@pllm.custom_forward` 与读 `pllm.hidden` / `pllm.final_hidden` 的方法 | custom forward 里一层可能跑多遍或不跑，"这一层的输出"不唯一 | 读 `pllm.residual(layer)` 或 `pllm.layer_logits(...)` |
| `active=` 用在 `tokens="decode"` 之外 | 跳层的 prompt token 只留下近似的 K/V，之后每个 token 都会读到 | `@pllm.custom_forward(tokens="decode")` |
| 带状态的 `@pllm.custom_residual`（`state=`、`chunk=`）与 `@pllm.custom_forward` | 记忆每个 token 前进一次，而 custom forward 里一层的访问次数不定 | 二选一 |
| 同一子层上的两个 `@pllm.custom_residual`；同一层上的两个 `@pllm.custom_attention` | 一个子层的残差更新、一层的 attention 读法只能有一个所有者 | 合并成一个方法 |
| 静态 KV 布局（`keep="window"` / `"none"`）与 `pllm.PositionState` 或 `@pllm.custom_forward` | 布局让 vLLM 把 KV 分成多组，这两者要求只有一组 | 二选一 |
| `@pllm.custom_attention` 在 `@pllm.custom_forward` 里配 `kv="peek"` | 只读不写的层调用还不支持定制读法 | 用 `kv="commit"` |
| 两个用 `pllm.Fed` 的状态；fed 状态被方法直接读 | 外部进程送来的数据只经一个 `@pllm.write` 进入 `ActivationPatch` 读的状态 | 每个负载一个 fed 状态，用一个 `@pllm.write` 转写 |
| 两个 `init=pllm.Imported()` 的状态 | 一次 `generate(batch=StateImport(...))` 只导入一个状态 | 把字段合并进一个状态 |
| 组件读另一个组件的状态（状态名里带 `.`） | 组件必须和单独运行时行为一致 | 需要共享状态的方法写进同一个负载 |
| 一个类继承两个负载 | 两个父类的同名声明会互相覆盖 | 把它们声明成字段（组件） |
| 一步之内两个方法互读对方的结果 | 一步内的结果只能单向流动 | 去掉其中一个读取，或把它移到下一步读状态 |
| 组件要求的 `publish_lag` 与引擎不同 | 整个引擎只有一个 `publish_lag` | 所有组件用同一个值 |
| `Adapter`、`ActivationPatch`、`@pllm.custom_forward`、`@pllm.custom_residual` 交给 `pllm.host.make_host` | 它们作用在模型的 forward 里，宿主训练没有模型 | 用 `pllm.make_llm` |

## 与引擎设置的组合

| 写法 | 需要 | 原因 |
| --- | --- | --- |
| `@pllm.custom_residual`、`@pllm.custom_attention`、overlay、`ActivationPatch` | `enable_prefix_caching=False` | 缓存的 KV 是在别的改写下算出的 |
| `@pllm.custom_attention` | `tensor_parallel_size=1`、FlashAttention 2、未量化 KV | 每张卡只看到自己的头；定制读法直接读 KV 页 |
| 给 `@pllm.custom_residual` 喂 `pllm.input_tokens` 的方法 | `tensor_parallel_size=1` | 定制残差还不支持多卡 |
| `cuda_graphs=True` | 没有需要 eager 的写法 | 见[支持矩阵](support-matrix.md#cuda-graph-与-eager) |

更多按模型和并行方式的限制见[支持矩阵](support-matrix.md)。

# 报错目录

pLLM 在构建负载时（`pllm.explain`、`pllm.make_llm`、`pllm.testing.check`）拒绝写错或不支持的声明，报错尽量说明
改法。本页按主题收录常见报错的原文、原因和改法；`...` 代表随你的代码变化的部分。排查的步骤见
[排查报错](../howto/troubleshooting.md)。

## 声明与输入的报错

| 报错（原文） | 为什么 | 改法 |
| --- | --- | --- |
| `...: give every input a source as its default, e.g. ...` | pLLM 只从默认值知道参数读什么，不按参数名猜 | `h: torch.Tensor = pllm.hidden(0)` |
| `the workload has no effect on the model: add @pllm.train, ...` | 没有去处的计算不会被安排 | 给方法加效果装饰器，或声明 `ActivationPatch` / `Adapter` |
| `a method takes one pLLM decorator of each kind; split stacked effects into one method per effect` | 后一个装饰器会覆盖前一个 | 一个效果一个方法，用 `pllm.result` 共享计算 |
| `property ... needs a return annotation; annotate state declarations with -> pllm.RequestState or -> pllm.InstanceState, and other properties with their return type` | pLLM 不执行 getter 也要认出状态 | 给 property 写返回类型 |
| `... is not a declared state of ...` | 方法读的状态名不存在 | 检查拼写；形状依赖设置的状态写成带类型的 property |
| `pllm.parameter(..., ...) names no declared field; declared: ...` | 字段名不存在 | `shape=` 声明的状态只有字段 `"value"`；`FromModule` 状态用 `pllm.module` 读 |
| `...: pllm.module needs a state declared with init=pllm.FromModule(...)` | 只有模块状态能当模块调用 | 用 `pllm.parameter` 读字段，或改用 `init=pllm.FromModule(...)` |
| `... names no method of ...` | `rows=`、`pllm.result` 或设置引用的方法不存在 | 检查方法名 |
| `...; results can only flow one way within a step, remove one of the reads` | 两个方法在同一步互读对方的结果，没有先后 | 去掉一个读取 |
| `... and ... both declare ...; compose them as fields instead of inheriting from both` | 多重继承时同名声明会互相覆盖 | 把两个负载声明成字段 |
| `...: components are independent; to share state, put both methods in one workload` | 组件必须和单独运行时一样 | 共享状态的方法写进同一个负载 |
| `key_fields=...: use names from ...` | 只有固定几种键 | `"request_id"`、`"prompt_index"`、`"sample_index"`、`"route"` |

## 方法体的报错

| 报错（原文） | 为什么 | 改法 |
| --- | --- | --- |
| PyTorch 的 "Tensor.item() cannot be called on meta tensors"，附注 `while shape tracing method ...; methods run inside CUDA graphs: use tensor operations such as torch.where, not .item() or Python branches on tensor values` | 方法在构建时只按形状追踪，运行时在 CUDA graph 里重放 | 去掉 `.item()`、`print(tensor)`、按张量值的 `if`；用掩码和 `torch.where`，看数值用 `@pllm.export` |
| `methods run inside CUDA graphs, so outputs need shapes fixed by the inputs; return a fixed-size tensor and a tensor mask instead of ...` | 自定义算子的输出形状随数据变化（如按条件删行）无法录进 graph | 返回固定大小的张量加掩码 |
| `methods run inside CUDA graphs, so Python cannot branch on tensor values (if, while or break on a tensor); ...` | 按张量值的 `if`、`while`、`break`（多见于 `@pllm.custom_forward` 的循环）让每步算的东西随数据变化，无法录进 graph | 循环固定次数，用 `torch.where` 选行；`custom_forward` 里用 `layers(..., active=mask)` 让已停下的行保持不变（需 `tokens="decode"`） |
| `... cannot compile as one full graph: ...` | 方法里有不能整图编译的操作 | 用可追踪的张量操作；自写算子给出 fake 实现；或 `ExecutionOptions(allow_numerical_transforms=False)` |
| `@pllm.train reads ..., which trains with ...; declare ... with optimizer=...` | 训练方式由方法读的输入决定，优化器写在状态上 | 给状态加报错里说的优化器 |
| `training methods return one loss per row; pLLM reduces the selected rows per state` | 只有 pLLM 知道哪些行属于哪份状态 | 返回 `[rows]` 的 loss，不要 `.mean()` |

## 效果与组合的报错

| 报错（原文） | 为什么 | 改法 |
| --- | --- | --- |
| `... both declare @pllm.guide_sampling; keep at most one guidance method` | 采样器每步只按一个预测加权 | 合成一个预测；其余用 `@pllm.add_logits` |
| `... trains on ... but ... on ...; pLLM funds one training mode per workload, so move one of them to a separate workload` | 一个负载只有一种训练方式 | 分到两个引擎，或宿主状态用 `make_host` |
| `rows= and update_rows= on request training require a model effect without sampler guidance, or @pllm.add_logits; standalone training and @pllm.guide_sampling update every selected state, so remove rows= and update_rows=` | 这条训练路径每步更新每个状态 | 在采样方法上选行，或 `reduction="sum"` 加零权重 |
| `@pllm.write needs distinct field names and no rows=; it writes every row` | 写入发布整份新版本 | 用 `torch.where` 保留不想改的行 |
| `... and ... both write state ... from the same version; let one method write it` | 同一版本只能有一个写入者 | 合并成一个方法 |
| `StateImport supplies one imported resource per workload; ... are imported` | 一次导入只写一个状态 | 合并成一个状态 |
| `... requires publish_lag=...; the engine uses publish_lag=...; all components must use the engine-wide lag` | 引擎只有一个 `publish_lag` | 统一设置 |
| `a task reads a value (sampled tokens, hidden states or another task's result) that exists only after its phase ends; declare it in a later phase` | 值在产生它的阶段之前不存在 | 采样前读 logits 用 `pllm.layer_logits((-1,))`；读采样结果的方法放到 `@pllm.train` 或 `@pllm.on(pllm.Phase.AFTER_SAMPLE)` |

## 训练的报错

| 报错（原文） | 为什么 | 改法 |
| --- | --- | --- |
| `host-batch training returns one scalar and owns its reduction; remove reduction=` | 宿主批次的归约只有你知道 | 在方法里自己归约 |
| `training on WeightedRowBatch or CausalTrainBatch fields takes one AdamW step over the whole batch: remove rows=, update_rows=, steps=, reset=, @pllm.differentiate and pllm.carry` | 这两种批次每批只走一步 | 去掉这些参数；跳过样本用权重 0 |
| `steps= and reset= apply to training on WeightedCEBatch fields (pllm.batch); remove them to train on request rows` | 多步 SGD 只用于加权交叉熵批次 | 去掉 |
| `training on WeightedCEBatch fields restarts each state before its SGD steps: give @pllm.train(reset=...) naming a method that sets the starting values` | 每批从同一起点做多步 SGD | 写一个 reset 方法 |
| `pllm.batch fields ... must all come from one batch: WeightedRowBatch ..., CausalTrainBatch ... or WeightedCEBatch ...` | 一个方法只读一种批次 | 只读同一种批次的字段 |
| `...: add init=pllm.Computed(...) (e.g. random A, zero B); with both low_rank factors at zero every gradient is zero and the adapter never learns` | 两个因子都为零时梯度恒为零 | A 随机、B 为零 |
| `carry fields ride on the training transition; give the state an optimizer` | carry 字段随训练更新一起发布 | 给状态加优化器 |
| `pllm.carry writes into the training method's own state` | carry 只写进被训练的状态 | 换成训练状态的 carry 字段 |
| `training on CausalTrainBatch/WeightedRowBatch inputs cannot be funded under ExecutionOptions.execution_peak_budget_bytes yet; leave execution_peak_budget_bytes unset` | 宿主批训练的峰值还不能纳入这个预算 | 不设 `execution_peak_budget_bytes` |
| `causal training needs a BF16 model whose embedding and LM head are not quantized` | 整模型反向按 BF16 写，且要对 embedding 与 LM head 求导 | 换 BF16、embedding 未量化的底座 |

## 导出的报错

| 报错（原文） | 为什么 | 改法 |
| --- | --- | --- |
| `tensor export needs positive every, max_rows and max_cols, with max_cols * 8 <= ... bytes (32 KiB), so one row fits an export event` | 一行必须装进一个 32 KiB 的导出事件 | `max_cols` 不超过 4096；要导更宽的向量，先在方法里压缩（如 top-k 编号） |
| `export overflow must be 'drop' or 'block'` | 只有两种队列满时的行为 | `"drop"`（默认）或 `"block"` |

## 模型改写与 forward 结构的报错

| 报错（原文） | 为什么 | 改法 |
| --- | --- | --- |
| `... reads block or model outputs, which a @pllm.custom_forward does not produce; use pllm.residual or pllm.layer_logits to read each layer's last visited stream` | custom forward 里一层可能跑多遍或不跑 | 读 `pllm.residual` 或 `pllm.layer_logits` |
| `Use @pllm.custom_forward(tokens="decode") for active=: skipped prompt tokens would leave approximate K/V that every later token reads. ...` | prompt 的近似 K/V 会被后续所有 token 读到 | `tokens="decode"` |
| `@pllm.custom_forward methods take (self, layers, marker inputs...) and no other pLLM decorator; use layers.save instead of @write` | custom forward 自己就是一个效果 | 位置状态用 `layers.save` |
| `...: use pllm.parameter or pllm.module after layers; read plain settings from self` | `layers` 之后只能读状态 | 普通设置从 `self` 读 |
| `PositionState is read and written by layers.load/save inside a @pllm.custom_forward method` | 位置状态只在 custom forward 里有意义 | 加一个 custom forward，或去掉它 |
| `layer ... routes tokens to experts; MLP overlays need a dense MLP, so overlay the attention query or value projection instead` | MoE 层没有统一的稠密 MLP 投影 | 改 attention 的投影 |
| `@pllm.custom_residual rewrites every row of its sites: remove rows= and @pllm.differentiate` | 残差改写作用于这一层的每一行 | 在方法里用掩码 |
| `@pllm.custom_residual(state=...) needs one attention or MLP site of one layer, chunk=, and a pllm.RequestState with one field, its carry, and no optimizer` | 带状态的子层每个请求一份携带量、按块更新 | 按报错补齐声明 |
| `...: take the site tensors ... first, without defaults, then markers` | 位置张量由 pLLM 按顺序传入 | 先写 `x, y`（或 `v`），再写带默认值的输入 |
| `...: take q and kv first, without defaults, then markers` | 同上 | 先写 `q, kv` |
| `keep="none" needs explicit layers and source >= 1 pointing to an earlier layer; use @pllm.custom_forward with kv="peek" for top-layer sources` | 更深的层在当前 token 还没算到 | 源选更早的层，或用 custom forward |
| `kv.attend topk needs integer positions [rows, k] or [rows, heads, k], padded with -1` | top-K 是逻辑位置 | 按形状给整数位置，不足的用 -1 补 |
| `...: return attention with q's shape and model dtype` | 输出要接回原生的输出投影 | 返回与 `q` 同形同 dtype 的张量 |
| `@pllm.custom_attention needs tensor_parallel_size=1: each rank sees only its heads` | 定制读法要看到完整的头 | 单卡 |
| `disable prefix caching (enable_prefix_caching=False) with pllm.model.Adapter, pllm.model.ActivationPatch, @pllm.train_in_forward, @pllm.custom_residual or @pllm.custom_attention: ...` | 缓存的 KV 是在别的改写下算出的 | 关闭前缀缓存 |
| `refresh= and on_missing= apply to decode tokens; set decode=True` | 这两个设置只管 decode token | 加 `decode=True` 或去掉它们 |
| `pllm.model.Adapter, pllm.model.ActivationPatch and @pllm.custom_forward act inside a model forward; compile this workload with make_llm, not make_host` | 宿主训练没有模型 | 用 `pllm.make_llm` |

## 引擎、并行与多进程的报错

| 报错（原文） | 为什么 | 改法 |
| --- | --- | --- |
| `pLLM runs on vLLM 0.27.1 or 0.32.0; install one of them instead of vLLM ...` | pLLM 只对这两个版本做过对齐 | 安装 0.27.1 或 0.32.0 |
| `pLLM runs vLLM's own model classes, not subclasses of them` | 子类可能改变投影布局或 logits | 用列出的模型类，见[支持矩阵](support-matrix.md) |
| `completions carry each sample's final hidden row, which only a workload collects; ...` | `generate_completions` 和 `generate_feedback` 要把每个样本最终位置的 hidden 交给宿主，只有装了负载的引擎才会采集 | 只做重试或打分、不训练时，给 `make_llm` 传 `workload=pllm.workloads.DummyWorkload()` |
| `pllm.input_tokens needs runner="mrv2"` | token 历史来自 MRv2 runner | `runner="mrv2"` |
| `...; set ModelConfig(cuda_graphs="auto") or cuda_graphs=False` | `cuda_graphs=True` 时负载里有需要 eager 的写法 | 去掉那个写法，或允许 eager |
| `worker processes cannot send ... defined in __main__ or a notebook; move it into an importable .py module, ...` | 子进程按导入路径重建负载 | 把类放进 `.py` 模块，或 `multiprocessing=False` |
| `TP Adapter fields start from pllm.Zeros or pllm.Computed and take shard-local @pllm.train_in_forward updates; optimizers and other init= sources need tensor_parallel_size=1` | 多卡时增量按分片存放 | 单卡，或按报错改初值与训练方式 |

## testing.check 的 ContractError

`pllm.testing.check` 发现问题时抛 `pllm.testing.ContractError`，`failures` 里每一行以契约名开头。

| 报错（原文） | 为什么 | 改法 |
| --- | --- | --- |
| `method ... mixes rows: equal inputs for this request produced different outputs. Dimension 0 holds request rows; reduce features (dim=-1), not requests (dim=0).` | 方法跨行计算，或在 `self` 上保存了可变状态 | 只沿 `dim=-1` 计算 |
| `method ... mutates borrowed input ... in place; inputs and parameters are read-only views. ...` | 输入与参数和别的方法、别的版本共享 | 返回新张量；改状态用 `@pllm.write` 或 `@pllm.train` |
| `guide_sampling still declares a sampler stage; set its beta to zero.` | `neutral=` 的负载仍在引导采样，常见原因是装饰器写了 `beta=beta` | 用设置方法 `beta=strength` |
| `add_logits must return exactly zero` | 强度为零时加法必须为零 | 让强度乘在整个结果上 |
| `pLLM misrouted rows ...` | 先排除自己的代码：原地写参数，或混用不同状态的行；排除后这是框架的 bug | 按报错把信息报告给我们 |

# 能力地图：哪些推理时算法今天能用 pLLM 写

推理时的 LLM 算法正在变复杂：查表记忆（如 DeepSeek 的 Engram，按 n-gram 查一张大表）、稀疏 attention 的索引器以及把索引结果跨层复用（如 DeepSeek 的 DSA、IndexCache）、压缩或共享的 KV、循环与可变深度、推理时训练（如 TTT、In-Place TTT）。
这一页逐个列出这些算法族：今天能不能用 pLLM 写、用什么 API 写、对应哪篇教程；对于写不了的算法，也直接把原因说清楚。文中的数字都来自教程和示例库的实测（RTX 4090），设置写在各自的 notebook 里。

下表中的“状态”列使用三种标记：

- **能写**：在已支持的模型上，用现有的公开 API 可以写，教程里有可运行的例子（或标明“无示例”）。
- **部分**：能写其中一部分，另一部分写不了，表里写明哪部分。
- **不支持 / 不在范围**：写不了，或者不属于 pLLM 要做的事（例如需要预训练一个新结构），表里写明原因。

## 在 forward 的某个位置读或加一点计算

如果你想在模型前向传播的过程中读取内部激活、修改残差流，或者插入一段自己的张量计算与自定义算子，可以先看这一节。

| 算法（例子） | 今天怎么写 / 为什么不行 | 状态 | 教程 |
|---|---|---|---|
| 激活引导（ActAdd、CAA）、ReFT / LoReFT | `pllm.model.ActivationPatch`；现成的 `pllm.workloads.ReFTWorkload`。激活补丁在模型的 CUDA graph 里运行 | 能写 | [00](../tutorials/00_quickstart.ipynb) |
| 线性探针、SAE 监控 | `pllm.hidden` / `pllm.residual` + `@pllm.export`，模型留在 CUDA graph 里 | 能写 | [01](../tutorials/01_observe_probe.ipynb) |
| logit lens、DoLa | `pllm.layer_logits`；现成的 DoLa 负载 | 能写 | [06](../examples/06_reading_inside.ipynb) |
| value residual、超连接（hyper-connections / mHC 的推理结构）、LAuReL | `@pllm.custom_residual(site="value" / streams=n / site="both")`，在 CUDA graph 里运行；这些结构的参数需要按该结构训练过的检查点 | 能写 | [07](../examples/07_custom_residual.ipynb) |
| 需要自写 kernel 的步骤（哈希、打分、扫描） | 用 `torch.library.custom_op` 包装后在方法里直接调用；explain 列出 FLOP/字节，check_on_gpu 比较 kernel 与 CPU 参考 | 能写 | [09](../examples/09_custom_kernels.ipynb) |

## attention 与 KV

当上下文变长或需要观察注意力分布时，常见需求包括统计注意力分数、只挑选一部分 KV 块读取，或者用滑动窗口和跨层共享来降低 KV 显存占用。

| 算法（例子） | 今天怎么写 / 为什么不行 | 状态 | 教程 |
|---|---|---|---|
| 注意力统计 / 监控（落在最近位置上的注意力占比） | `@pllm.custom_attention` 返回 aux，`pllm.attention(layer, "aux")` 导出；输出与原生 vLLM 逐位相同 | 能写 | [10](../examples/10_custom_attention.ipynb) |
| Lookback Lens 式（落在 prompt 上的注意力占比） | 需要知道每行的 prompt 长度；`PagedKV` 只在 `sink=True` 的布局下提供它（`kv.sink_lens`） | 部分 | — |
| 训练无关的稀疏读取（Quest 按页挑选、retrieval heads 一类的固定模式） | `kv.attend(q, topk=...)`，每层、每个头可以有不同预算；省读不省显存；用 PyTorch 打分时吞吐很低（示例 10 在 7B 上实测为原生的 13%），提速要自定义 kernel | 能写 | [10](../examples/10_custom_attention.ipynb)、[09](../examples/09_custom_kernels.ipynb) |
| 按分数驱逐 KV（H2O、SnapKV、PyramidKV） | 只读选择可以（按分数挑位置读，每层/每头预算）；按分数真正释放显存不支持，原因见[设计取舍](../explain/design-tradeoffs.md) | 部分 | [10](../examples/10_custom_attention.ipynb) |
| 滑动窗口、局部/全局层交替 | `keep="window", window=N`，旧块由 vLLM 的滑动窗口机制释放；示例 10 中偶数层窗口 256 让 7B 每个 2048-token 请求的 KV 上限从 112.0 MiB 降到 70.4 MiB | 能写 | [10](../examples/10_custom_attention.ipynb) |
| 保留 prompt 加最近窗口（R-SWA） | `sink=True`：保留整个 prompt 加最近 N 个生成的 token；按最坏情况算的上限不变，省的是长生成时的部分 | 能写 | [10](../examples/10_custom_attention.ipynb) |
| StreamingLLM（开头几个 sink token + 最近窗口） | 读法可以用 `topk` 写成“只读前几个位置和最近窗口”；但显存只能按 R-SWA（整个 prompt + 窗口）释放，不能只留开头几个 token | 部分 | — |
| 跨层共享 KV（CLA、YOCO 一类，以更低的层为源） | `keep="none", source=k`：这一层不分配 KV，读下面第 k 层的；示例 10 实测奇数层共享时 KV 上限减半；这些模型需要按共享结构训练过 | 能写（执行） | [10](../examples/10_custom_attention.ipynb) |
| 以更深的层为源的 KV 共享（LCKV 一类） | 源层在当前 token 还没算到；报错信息建议改用 `@pllm.custom_forward` 的 `kv="peek"` | 部分 | [04](../tutorials/04_custom_forward.ipynb) |
| 压缩的远端记忆（Activation Beacon、DeepSeek V4 的压缩 KV 一类、LastKV 一类） | 暂不支持，原因见[设计取舍](../explain/design-tradeoffs.md) | 不支持 | — |
| 训练出来的稀疏 attention（DSA、NSA、MoBA） | 是模型结构本身；vLLM 原生支持 DSA，但需要 sm90 及以上的 GPU（H100 一类） | 不在范围 | — |
| 跨层复用稀疏 attention 的选择（IndexCache、DeepSeek V4.1 的索引源层） | 需要 DSA 模型，而 vLLM 的 DSA 路径需要 sm90 及以上；pLLM 的 `pllm.attention(layer, "topk")` 留到那时，目前会报错说明尚未支持 | 不支持（需要硬件） | — |
| KV 量化（KIVI 一类） | 属于 vLLM 的 `kv_cache_dtype`；`@pllm.custom_attention` 要求未量化的 KV | 不在范围 | — |

## 查表记忆

如果你想让模型在推理时查询外部参数表（比如根据最近几个 token 的 n-gram 查表并加进残差流或 logits），可以通过状态声明和残差或 logits 接口组合出来。

| 算法（例子） | 今天怎么写 / 为什么不行 | 状态 | 教程 |
|---|---|---|---|
| Engram 式 n-gram 记忆（加在现有模型上） | `pllm.input_tokens(history=n)` 取 token 历史，`InstanceState(init=pllm.Pretrained(...))` 读表，`placement="device"/"host"` 由你决定，`@pllm.custom_residual` 把查到的向量加进残差流；表要在别处训练好，pLLM 不训练、不更新表 | 能写 | [11](../examples/11_lookup_memory.ipynb) |
| n-gram 语言模型先验（按上下文给 logits 加分） | 同样的查表接 `@pllm.add_logits` | 能写 | [11](../examples/11_lookup_memory.ipynb) |
| memory layers / product-key memory | 键由 hidden 算出再取 top-k、值从大表里按行取，可以在 `custom_residual` 里写成张量数学（大表可以放主机内存）；本仓库没有示例，未验证 | 应能写（未验证） | — |
| 模型自带的 Engram（DeepSeek V4.1 等） | 由 vLLM 的模型代码实现；在 pLLM 里跑这些模型属于“模型准入”，且需要 sm90 及以上 | 不在范围 | — |

## 深度与循环

当算法需要改变层的执行顺序，比如把同一组层循环跑多遍，或者根据当前 token 的状态动态跳过后面的层，可以用自定义前向与按行激活掩码来表达。

| 算法（例子） | 今天怎么写 / 为什么不行 | 状态 | 教程 |
|---|---|---|---|
| 循环深度（Ouro、Huginn 一类） | `@pllm.custom_forward` 里循环调用 `layers(...)`；每一遍都写 KV（Ouro 的“最后一遍复用”）；prompt 每一遍各留一份 KV 目前写不了 | 能写 | [04](../tutorials/04_custom_forward.ipynb) |
| early exit、按 token 跳层（CALM 式饱和判据、MoD 的因果路由版、MoR 的 token-choice 路由） | `layers(..., active=mask)`；跳过的行由 pLLM 补 KV，决定留在 GPU 上，每步没有阻塞同步；只用于 decode token | 能写 | [12](../examples/12_adaptive_depth.ipynb) |
| MoR 的 expert-choice（在整个 batch 里选 top-k）、原版 MoD（跳过的 token 不进 attention） | 前者让一个 token 的深度依赖同 batch 的其他请求，pLLM 故意不支持；后者要求 KV 中间留洞，和按分数驱逐同样的限制 | 不支持 | — |
| prompt token 的提前退出 | `active=` 目前只用于 `tokens="decode"` | 不支持 | — |
| 连续思维 / token 级循环（Coconut 一类） | `custom_forward` + `pllm.PositionState` 把上一位置的状态带到下一位置；要有效果需要训练过的模型 | 部分 | [04](../tutorials/04_custom_forward.ipynb) |

## 状态与推理时学习

如果算法需要在每个请求或整个实例的生命周期内维护可变状态，甚至在生成过程中做梯度更新、快权重更新或宿主侧反馈循环，可以看下面这些接口。

| 算法（例子） | 今天怎么写 / 为什么不行 | 状态 | 教程 |
|---|---|---|---|
| 每请求 LoRA（Text-to-LoRA 式超网络）、decode 时在线训练（ESamp）、测试时学习（TLM） | `pllm.RequestState` / `pllm.InstanceState` + `@pllm.train`；反向和优化器更新在 CUDA graph 里 | 能写 | [02](../tutorials/02_request_adapters.ipynb)、[03](../tutorials/03_decode_training.ipynb) |
| TTT、Titans、快权重记忆、In-Place TTT | `@pllm.train_in_forward`、带 `state=` 的 `@pllm.custom_residual` | 能写 | [07](../examples/07_custom_residual.ipynb)、[08](../examples/08_inplace_ttt.ipynb) |
| 线性 attention / Gated DeltaNet 混合模型 | Qwen3.5 已准入（功能范围见[支持矩阵](../reference/support-matrix.md)）；在现有模型上加一个按块递归的子层，用带 `state=`、`chunk=` 的 `custom_residual` | 部分（取决于模型） | [07](../examples/07_custom_residual.ipynb) |
| 测试时强化学习（TTRL） | 宿主循环：生成、投票、训练在同一个引擎里 | 能写 | [05](../tutorials/05_host_feedback.ipynb) |

## 解码控制、架构与训练、agent

最后这一节整理了解码循环控制、底层预训练结构改造，以及在外层用 agent 搜索算法代码的边界。

| 算法（例子） | 今天怎么写 / 为什么不行 | 状态 | 教程 |
|---|---|---|---|
| MTP、投机解码、可回滚的解码 | 需要控制解码循环本身，是另一项路线图上的工作 | 不支持 | — |
| 从头预训练、改 MoE 路由、全新的 mixer | 需要新结构和预训练 | 不在范围 | — |
| agent 驱动的算法搜索（AlphaEvolve 一类的循环） | pLLM 提供 explain、check 和契约，循环本身是 `examples/algorithm_search` 这个例子；这次演示的分数差别在噪声范围内 | 示例 | [13](../examples/13_algorithm_search.ipynb) |

## 现成的负载

`pllm.workloads` 里的实现和你写的负载走完全相同的路径，可以直接用，也可以当作写法范例：

| 负载 | 算法 | 用到的写法 |
|---|---|---|
| `ESampWorkload` | ESamp：每个请求在线训练浅层到深层的预测器，引导采样 | `RequestState` + `@pllm.train` + `@pllm.guide_sampling` |
| `OverRIDEWorkload` | OverRIDE：实例共享的头，按真实采样反馈有条件地更新 | `InstanceState` + `@pllm.add_logits` + `@pllm.train` |
| `OutputLoRAWorkload` | 输出层 LoRA，外部加权交叉熵批次、多步 SGD | `pllm.host.WeightedCEBatch` + `steps=` / `reset=` |
| `ReFTWorkload` | ReFT / LoReFT 激活干预 | `pllm.model.ActivationPatch` + `pllm.Imported()` |
| `DoLaWorkload` | DoLa 对比解码 | `pllm.layer_logits` + `@pllm.add_logits` |
| `TLMWorkload` | 测试时学习（TLM），全层 q/v LoRA | `pllm.host.CausalTrainBatch` + `pllm.causal_model` + `Adapter` |
| `InPlaceTTTWorkload`、`PrefillAdaptersWorkload` | In-Place TTT 快权重、prompt 上训练的适配器 | `@pllm.train_in_forward` |
| `PromptFactorsWorkload`、`PromptMixtureWorkload`、`DecodeFactorsWorkload` | 超网络写 LoRA、多个 LoRA 的混合、decode 时训练的因子 | 同一个 `Adapter` 读不同生产者写的状态 |
| `RecirculationWorkload`、`JacobiRecirculationWorkload`、`StepLoopWorkload` | 层间再循环、Jacobi 式 prompt 迭代、把上一位置的 hidden 带到下一位置 | `@pllm.custom_forward`、`pllm.PositionState` |
| `RewardHeadWorkload`、`RewardHeadFeedback`、`majority_vote` | 多数投票奖励头 | `pllm.host.make_host` + `llm.generate_feedback` |
| `FTTTWorkload` | 宿主反馈驱动的低秩编辑 | `pllm.host.WeightedRowBatch` + `HostTrainer` |
| `DummyWorkload` | 最小的观测负载 | `pllm.hidden` + `@pllm.export` |

## 为什么有些写不了

按分数释放 KV 显存、压缩的远端记忆、DSA 的 top-k 复用等为什么今天写不了、有什么替代做法，见[设计取舍](../explain/design-tradeoffs.md)。

## 共同的限制

在使用上述功能时，还有几条共同的边界与设计原则：

- 只有读取 `RequestState`、可变/外部输入 `InstanceState`、`@pllm.train_in_forward`、`last_tokens > 0` 的 `ActivationPatch` 或 `@pllm.custom_attention` 时必须关闭前缀缓存（`enable_prefix_caching=False`）；无状态 `@pllm.custom_residual` 与只读不可变 `InstanceState` 的静态改写可保留开启。
- `@pllm.custom_attention` 需要未量化 KV（在稠密与 KV-MoE 上使用 FlashAttention 2，在 DeepSeek V2/V3 MLA 上直接吸收 `W_UK_T` / `W_UV` 读取 `kv_lora_rank` + `qk_rope_head_dim` 潜变量缓存且仅支持 `keep="all"`）；若读取 `pllm.attention(...)` 事实则需 `tensor_parallel_size=1`。
- 静态 KV 布局不能和 `PositionState` 或 `@pllm.custom_forward` 一起用。
- 张量并行在 vLLM 0.32.0 及以上可用，哪些功能可以多卡见[支持矩阵](../reference/support-matrix.md)。
- 一个原则：贵的选择由你显式写出来，pLLM 不替你做——大表放显存还是主机内存、KV 怎么存、训练结果几步后可见（`publish_lag`），都写在你的声明里，`pllm.explain` 把代价列出来。默认值只替你省掉小的决定。
- 这些能力各自省的是哪种资源（显存容量、显存带宽、PCIe、卡间通信、时间线气泡、SM、host CPU）、实测效果和边界，见[性能与显存](../explain/performance.md)。

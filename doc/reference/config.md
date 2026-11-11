# 配置项

本页列出 `pllm.ModelConfig`、`pllm.ExecutionOptions`、`pllm.KVOffload`、`pllm.host.HostConfig` 和
`pllm.benchmark.BenchmarkOptions` 的每个字段。默认值与 [生成的签名](../agent/reference.md#all-signatures) 一致；
字节数的单位都是字节。

```python
import pllm
from pllm.workloads import DummyWorkload

config = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_num_seqs=64,
    max_num_batched_tokens=4096,
    gpu_memory_utilization=0.5,
    execution=pllm.ExecutionOptions(publish_lag=1, state_page_budget_bytes=256 << 20),
)

print(pllm.explain(DummyWorkload(), config).eager_reason == "")
```

## ModelConfig：模型与引擎

大部分字段原样交给 vLLM，含义与 vLLM 的同名参数相同；标了“pLLM”的字段由 pLLM 解释。

| 字段 | 默认值 | 含义 | 什么时候改 |
| --- | --- | --- | --- |
| `model` | 必填 | Hugging Face 模型名或本地路径 | — |
| `dtype` | `"bfloat16"` | 模型权重和激活的精度：`"bfloat16"`、`"float16"`、`"float32"` | 一般不改；`"auto"` 状态精度按它选择 |
| `max_model_len` | `512` | 每个请求 prompt 加生成的最大 token 数 | 长 prompt 或长生成时调大；KV 显存随之增加 |
| `max_num_seqs` | `8` | 同时运行的请求数上限 | 吞吐测量常用 128；pLLM 为请求状态预留的份数随它增长 |
| `max_num_batched_tokens` | `512` | 一步最多计算的 token 数（含 prompt 分块） | 长 prompt 多时调大；`tokens="all"` 导出的行数上限也是它 |
| `gpu_memory_utilization` | `0.5` | vLLM 可用的显存比例（权重、KV、pLLM 预留都在其中） | 显存不够或要和别的引擎共享 GPU 时调整 |
| `seed` | `0` | vLLM 的全局随机种子，也是 pLLM 状态初值（`pllm.Computed`、模块初始化）的种子；单个请求的采样种子写在 `SamplingParams(seed=...)` | 复现实验时固定 |
| `trust_remote_code` | `False` | 允许加载模型仓库里的代码 | 只有需要远程 tokenizer 代码的模型（如 Kimi 切片） |
| `enable_prefix_caching` | `False` | vLLM 的前缀缓存 | 只读、不改 forward 的负载可以打开；改模型或 forward 的负载（overlay、激活补丁、`custom_residual`、`custom_attention`）要求关闭，因为缓存里的 KV 是在别的状态下算出来的 |
| `enable_chunked_prefill` | `False` | vLLM 的分块 prefill | 长 prompt 与 decode 混跑时打开；prompt 行的训练按分块进行 |
| `generated_kv` | `"isolate"` | pLLM：打开前缀缓存时，生成 token 的 KV 能否被后来的请求当作 prompt 前缀复用。负载区分 prompt 与生成 token 时（例如激活补丁只改生成的 token），同样的 token 在两种身份下算出的 KV 不同，`"isolate"` 禁止这种复用 | `"reuse"` 只在你确认生成部分的 KV 与它是不是 prompt 无关时使用 |
| `revision` | `None` | 模型仓库的版本 | 固定检查点版本 |
| `multiprocessing` | `False` | pLLM：把负载送进 vLLM 的 EngineCore 子进程 | 上生产时打开，要求见 [上生产](../howto/production.md) |
| `execution` | `ExecutionOptions()` | pLLM 的执行策略，见下一节 | — |
| `cuda_graphs` | `"auto"` | pLLM：`"auto"` 能进 CUDA graph 就进，否则 eager 并把原因写进 `eager_reason`；`True` 要求进图，不能进就报错；`False` 一律 eager | 测吞吐时设 `True`，防止悄悄退回 eager |
| `kv_offload` | `None` | 把完整的 KV 页卸载到 CPU 内存，见 `KVOffload` | KV 显存不够、又要保留长上下文时 |
| `tensor_parallel_size` | `1` | 张量并行的卡数 | 需要 vLLM 0.32.0；哪些功能可以多卡见[支持矩阵](support-matrix.md) |
| `pipeline_parallel_size`、`data_parallel_size` | `1` | 流水线并行、数据并行 | 目前必须为 1，pLLM 只支持张量并行 |

## ExecutionOptions：pLLM 的执行策略

| 字段 | 默认值 | 含义 | 什么时候改、代价 |
| --- | --- | --- | --- |
| `publish_lag` | `1` | 训练或写入在第 n 步算出的新版本，从第 n + 1 + `publish_lag` 步起被读到 | 设 `0` 得到严格顺序（第 n 步的更新第 n + 1 步可见），代价是训练不再和下一步 forward 重叠；原因见[核心概念](../explain/core-concepts.md) |
| `max_host_training_rows` | `1` | 宿主批训练每步每份状态最多接受的行（序列）数 | 宿主批次更大时调大；pLLM 按它预留训练显存。不限制 decode 时的请求行 |
| `state_page_budget_bytes` | `0` | 状态的显存页预算；`0` 用后端观测到的空闲容量 | 想让计划可复现，或限制状态占用时给定值 |
| `execution_peak_budget_bytes` | `None` | pLLM 执行峰值的上限：状态页、任务工作区与存活结果、CUDA graph 池和锚点预留；不含模型权重和 KV。`None` 不限 | 低于 `explain` 打印的重叠阈值时，旁路任务改成主 stream 内联（不再重叠）；低于失败阈值时规划报错 |
| `cuda_graph_pool_budget_bytes` | `None` | 每个任务和模型的 CUDA graph 私有内存池预算（不是总和）；`None` 时任务按追踪结果预留，模型图由 vLLM 自动管理 | 报错说“需要多少字节”时按报错调大，常见于带状态的子层和 `custom_forward` |
| `guidance_capacity` | `4096` | `@pllm.guide_sampling` 每行只引导 top-k / top-p 之后 logits 最高的这么多 token，尾部不引导；必须是 32 到 4096 之间的 2 的幂，`None` 表示对整个词表精确引导 | 采样集合通常比它大时调大；越小越慢，因为超出容量的行要走 top-K 选择 |
| `allow_numerical_transforms` | `True` | 允许 pLLM 为了速度改写数值路径（融合、并行扫描），浮点舍入可能变化，发生时有一次警告 | 需要与参考实现逐位对齐时设 `False` |
| `tuning` | `"heuristic"` | `"heuristic"` 用确定的规则选实现；`"measured"` 在启动时实测并选更快的实现 | 想用实测选择换启动时间时 |

## KVOffload：KV 卸载到 CPU

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `connector` | `"OffloadingConnector"` | vLLM 的卸载连接器：`"OffloadingConnector"` 或 `"SimpleCPUOffloadConnector"` |
| `cpu_bytes` | `536870912`（512 MiB） | 用于卸载的 CPU 内存 |
| `prompt_only` | `False` | 只卸载 prompt 的 KV；需要 `"OffloadingConnector"` |

## HostConfig：只在宿主上训练的负载

`pllm.host.make_host(HostConfig(...), workload)` 不加载模型，用这些事实编译负载（见[宿主循环与批训练](../guides/host-loops.md)）。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `device` | 必填 | 训练所在的设备，如 `torch.device("cuda")` 或 `torch.device("cpu")` |
| `hidden_size` | `0` | 批次里 hidden 行的宽度 |
| `target_shape` | `()` | 每行目标的形状；`()` 表示每行一个数 |
| `seed` | `0` | 状态初值的随机种子 |
| `state_page_budget_bytes` | `67108864`（64 MiB） | 状态页预算 |
| `max_input_rows` | `None` | 一个批次的最大行数；超过的批次在 `train` 前被拒绝 |
| `execution_peak_budget_bytes` | `None` | 每步执行峰值的上限 |

## BenchmarkOptions：对齐的吞吐对比

`pllm.benchmark.compare` 和 `tools/benchmark_*.py` 用它（见[公平地测吞吐](../howto/measure-throughput.md)）。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `model` | `"Qwen/Qwen2.5-0.5B-Instruct"` | 模型 |
| `batch` | `8` | prompt 数 |
| `tokens` | `256` | 每个样本生成的 token 数（忽略 EOS） |
| `rounds`、`warmup` | `3`、`1` | 计时轮数（取中位数）与不计时的预热轮数 |
| `sampling_n` | `16` | 每个 prompt 的样本数 |
| `gpu_memory_utilization` | `0.25` | 每个引擎的显存比例 |
| `cuda_graphs` | `True` | 两边都用 CUDA graph |
| `cuda_graph_pool_budget_bytes`、`state_page_budget_bytes` | 捕获池 None，状态页 0 | 默认自动定容；需要限制资源时显式设置上限 |
| `execution_peak_budget_bytes` | `None` | 执行峰值上限 |
| `guidance_capacity` | `4096` | 同 `ExecutionOptions.guidance_capacity` |
| `profile` | `None` | 给一个目录时，计时结束后另跑一遍做性能剖析 |

# 基准解读

维护中的基准场景调用 `pllm.benchmark.compare`，它构造公共模型句柄并使用其
正常的生成、诊断与关闭操作。场景提供算法与数据，不提供 arena、调度器、捕获、
worker 或性能分析器的实现。

```bash
.venv/bin/python -m tools.benchmark_esamp --batch 8 --tokens 256 --rounds 3
.venv/bin/python -m tools.benchmark_lora --batch 8 --tokens 256 --rounds 3
```

共享 CLI 接受 `--model`、`--batch`、`--tokens`、`--rounds`、`--warmup`、
`--gpu-memory-utilization`、`--capture-memory-mib`、`--state-memory-mib`、可选的
`--execution-memory-mib` 与 `--profile DIRECTORY`。默认 Qwen2.5-0.5B、批次 8、
256 token、三轮、一轮预热与 0.25 GPU 显存利用率。输入用种子 7、温度 0.8、
top-p 0.95、忽略 EOS、禁用前缀缓存与 `n=1`。捕获默认每任务 256 MiB 驻留上限，
原样传给公共 `ExecutionOptions`。更大形状可能需要更大的显式上限；资金不足的
捕获会失败，而不是改变形状、禁用捕获或选择回退。基准状态容量默认固定
256 MiB，经 `ExecutionOptions.state_memory_bytes` 进入原始签名计划。它必须
放得下后端观测到的可用内存；Rust 拒绝不足的状态容量。这是 arena 预算，
不是模型/KV/scratch 总内存。正常执行默认取观测容量（`state_memory_bytes=0`），
其变化合法地改变计划签名；固定基准容量让精确计时/性能分析重建可复现。
`--execution-memory-mib` 把有限的 `ExecutionOptions.memory_budget_bytes` 传给
原始 AER timeline；省略即不加限制，零即零容量。它包含所有声明的状态/驻留/
预留/工作区/存活结果成本，但不约束未声明的模型/KV、分配器或库内存。缺失
任务/输出成本或容量不足会拒绝规划，不减少工作量、不禁用捕获。报告含
`planned_peak_memory_bytes`，与实测 CUDA 峰值分开列出；保守规划界与真实
分配器测量不可互换。

公共模型负载入口不要求训练或捕获：

```python
from pllm.benchmark import BenchmarkOptions, compare

results = compare(my_workload, BenchmarkOptions(batch=4, tokens=64, rounds=1))
```

这测量普通实现。`capture=True` 加一个捕获臂；`explicit_math=True` 加一个显式
张量臂。不支持的请求在正常绑定边界失败。`require_training=True` 要求正的
loss、update 与训练行计数；观测或仅激活算法不要开。所有运行仍检查 token 数、
清理、回退与等效工作量。这个入口测模型生成；带类型宿主服务计时用
[验证指南](validation.md) 里的共享执行 trace，不是合成 GPU 循环。

内部 ESamp 与 LoRA 命令显式开启两个优化臂与训练要求。ESamp 用普通张量、
显式张量、张量捕获与分页 eager/捕获实现比较同一个请求内算法。Output LoRA
对共享头与独立头分别比较同样的五种执行选择。两个场景默认 rank 4；LoRA 默认
每状态两个训练行、两次 SGD 迭代。LoRA 还接受 `--rank`、`--training-rows`、
`--training-fraction`、`--scope instance|request|both` 与 `--state-groups N`
（默认每请求一个请求状态）。活跃行数向上取整并报告；剩余权重为零。请求
作用域下，组共享 prompt/样本键与相同 prompt 文本；训练输入按状态组一批，
不是按请求。这建模的是同步准入、一起完成的队列（忽略 EOS），不是共享键的
独立计时加入或部分退役。实例作用域仍是单独算法。合成训练张量在模型初始化
后用公共模型维度创建。同一输入经正常带类型辅助输入边界提交。生成、页选择、
发布与采样引导都在被测路径内。共享与独立算法有不同归约，不作为等效负载
宣传。

冷模型/编译/捕获准备单独报告。预热被排除。JSON 报告软件版本、基准设置、
签名计划标识、实测轮次时间、真实输出 token、loss/update 计数、训练行与峰值
分配内存。`round_process_cpu_seconds` 用 `process_time` 测每轮完整生成与最终
设备同步前后的进程用户+系统 CPU 时间。它包含该进程内的 Python、原生提交与
其他线程，排除睡眠/被调度下时间与子进程，可能超过墙钟。它不是 CUDA API
时长，也不声称全部 CPU 工作都是启动提交。性能分析重放报告自己的
`profile_round_process_cpu_seconds`；这些插桩时间不替代稳态测量。所有内部
训练对比臂必须产出请求的 token 数、做非零训练、结束时无待收请求
pin/释放或回退，且等效工作量一致。峰值内存包含持久状态与捕获池。
`baseline_allocated_bytes` 在预热后测量；`peak_extra_allocated_bytes` 是该基线
之上的峰值，包括任何驻留增长，不是"只有临时"的证明。LoRA 在每次对比前打印
其附加算法设置。结果还暴露生成后的原始 `state_usage` 快照：arena 预留、
逻辑/物理载荷、填充、页所有权与待收 receipt。这是轮末状态，不是采样峰值，
也不是独立分配台账。

`--profile` 先完成**所有**臂的吞吐计时，然后才重建每个臂、重复同样预热、
对完全相同的测量轮做性能分析。不用额外生成外推代替更早轮次；轮间更新
合法变化。重建的计划签名与聚合有效工作量必须与原始测量一致。
`profile_rounds` 记录插桩轮数；分析成本与工作量是这些轮的合计。CUPTI 插桩
在其上下文退出后可能影响后续 CUDA API 延迟，所以不要在同一进程里再启动
计时对比。LoRA 的 `--scope both --profile ...` 每个作用域用一个全新进程。
它按臂导出 Kineto Chrome trace（含内存事件、无张量快照），并报告设备内核数、
内核微秒合计、CUDA runtime/driver API 微秒合计与真实 GPU memcpy 字节。
API 时间含同步，不是纯 CPU 提交时间。重叠内核是求和的，不作为 GPU 墙钟报告。
`cuda_launch_api_count` 按线程数外层宿主启动区间，合并嵌套 runtime/driver
调用；`cuda_graph_launch_api_count` 是其图启动子集。两者都不数每次 replay
内部的设备内核。memcpy 字节不含内核内访存。这些是完整生成成本，包括 vLLM，
不是孤立的负载任务测量。性能分析从不贡献稳态计数器或计时，且必须通过同样的
等效工作量/清理检查。已有 trace 文件不被覆盖。例如：

```bash
.venv/bin/python -m tools.benchmark_lora --batch 2 --tokens 4 --rounds 1 \
  --training-rows 4 --training-fraction 0.25 --scope request --profile /tmp/lora-profile
```

独立输入用 `pllm.WeightedCEBatch.for_prompts(hidden, logits, labels, weights,
prompt_indices)`。它构造原始辅助对象及其 prompt/样本键，不拷贝张量。
`prompt_indices` 必须与传给 `generate` 的一致；共享输入仍用无键的
`pllm.WeightedCEBatch(...)`。场景不导入执行帧，不构造请求身份对象。例如
`--batch 8 --scope request --state-groups 2` 在八个请求间共享两个状态。
`compare(..., prompt_indices=...)` 把这些键传给每次公开生成，包括预热与分析；
设置会记录它们以便复现。

用 `alternatives=(("label", workload, captured), ...)` 给同一对比加显式实现。
分页臂现在用设备侧分组内核，无矩阵打包、无宿主偏移同步。ESamp 的捕获 loss
直接读原始池；其捕获 Adam 现在两次启动直接更新全部五个原始字段，每状态两个
标量偏置因子作为 scratch。LoRA 的分页捕获直接 reset 并更新原始私有字段，
保留其带种子 reset 与按适配器归一化。捕获字段现在共享逻辑分组，矩阵内核直接
消费原始行索引，无单独操作数/输出 gather。被完全覆盖的捕获输入不再收到多余
填充；较小输入仍填充签名填充区。

2026-09-09 验收运行，默认批次 8 / 256 token / 3 轮，每臂 6144 输出 token：

| 算法 | 张量 | 显式张量 | 张量捕获 | 分页 | 分页捕获 | Loss/Update/行 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ESamp | 865 | 994 | 1046 | 937 | 1040 | 6120/6120/6120 |
| 共享 LoRA | 1463 | 1450 | 1445 | 1385 | 1394 | 6/3/6 |
| 请求 LoRA | 1353 | 1358 | 1359 | 1325 | 1323 | 48/24/48 |

数值为输出 tok/s；只在行内比较。LoRA 每次 generate 调用训练一次，不是每个
token。所有臂通过等效工作量、清理与零回退检查。日志在
`outputs/acceptance-5Uinna/benchmark-{45,46}.log`。这次运行在选中状态梯度
替换之后；未用性能分析器，不足以为分页默认值背书。GPU 同时持有另一个用户
的分配。

当前源码的八配置研究在 `outputs/study-current-AV1WUw/RESULTS.md`：40 个计时臂
加 40 个独立重建的分析臂，各三轮。请求数、训练行/比例、状态共享、rank、
实例/请求与更小内存余量都通过等效工作量、签名与清理检查。提高公开 vLLM KV
预留使预热分配上涨约 3.5 GiB；那不是 OOM/耗尽测试。没有确立稳定的分页吞吐
优势。捕获降低了额外分配峰值，但增加了驻留存储与 memcpy；不要推断总内存更低。
GPU 保持共享。报告把入队/等待 API 区间与 GPU 活跃区间分开；设备内核执行数
不是宿主启动调用数，入队时间不是完整 Python 提交时间。这些限制保持显式，
不从计数器推断。随后的 CPU/启动插桩检查与既有 trace 归因在
`outputs/cpu-launch-euODct/RESULTS.md`。它不重跑也不取代八案例吞吐矩阵：
只有报告变了，负载数学与执行没变。

更早的七配置有界矩阵保留在 `outputs/gpu-matrix-uP6yrj/RESULTS.md`，含其设置
与限制。它早于那次梯度替换，不是当前完整性能研究。更早的交错计时/分析运行
被排除在实现排名之外：CUPTI 可能影响后续计时。其原始日志保留；过时吞吐表
不与当前运行并存维护。

LoRA 负载证据里保留一份独立数值参考。基准不改写该参考，不把它当替代运行时
调用，也不从计数器一致声称任意数值等价。

历史对齐 ESamp 实验用批次 8、`sampling_n=16`、至少 256 token、bfloat16、
温度 0.8、top-p 0.95 与最大模型长度 ≥512。当前公开附着支持 `n=1`；其绝对
吞吐与那些历史 4k+ tok/s 数字不可比。小验证运行是比例/行为检查，不是吞吐
回退或提升的证据。

使用仓库虚拟环境，比较完全相同的负载。异常终止后先检查 GPU 进程归属再重试；
永不终止无关 worker。不要把已退役运行时恢复为永久基线。

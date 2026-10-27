# 基准解读

这份指南写给要报告性能数字的人。核心问题不是「怎么跑」，而是「跑出来的
数字能说什么、不能说什么」。维护中的基准场景调用 `pllm.benchmark.compare`：
它构造公共模型句柄，走正常的生成、诊断与关闭操作。场景只提供算法与数据，
不提供 arena、调度器、捕获、worker 或性能分析器的实现——也就是说，基准测
的是公开路径，不是为跑分开后门。

## 我要跑一轮标准基准

两个内置场景，ESamp 与 LoRA，各一条命令：

```bash
.venv/bin/python -m tools.benchmark_esamp --batch 8 --tokens 256 --rounds 3
.venv/bin/python -m tools.benchmark_lora --batch 8 --tokens 256 --rounds 3
```

两者都显式开启两个优化臂与训练要求。ESamp 用普通张量、显式张量、张量捕获
与分页 eager/捕获实现，比较同一个请求内算法；Output LoRA 对共享头与独立头
分别比较同样的五种执行选择。两个场景默认 rank 4；LoRA 默认每状态两个训练
行、两次 SGD 迭代。

## 我要调整形状与预算

共享 CLI 接受 `--model`、`--batch`、`--tokens`、`--rounds`、`--warmup`、
`--gpu-memory-utilization`、`--capture-memory-mib`、`--state-memory-mib`、
可选的 `--execution-memory-mib` 与 `--profile DIRECTORY`。默认
Qwen2.5-0.5B、批次 8、256 token、三轮、一轮预热与 0.25 GPU 显存利用率。
输入固定为种子 7、温度 0.8、top-p 0.95、忽略 EOS、禁用前缀缓存、`n=1`——
改动这些就是另一个基准，报告时不能说成同一负载。

三个内存参数容易被误读，逐个说清：

- `--capture-memory-mib`（默认 256）：每任务捕获的驻留上限，原样传给公共
  `ExecutionOptions`。更大的形状可能需要更大的显式上限；资金不足的捕获会
  失败，而不是改变形状、禁用捕获或选择回退。失败是设计行为，别当成 bug 修。
- `--state-memory-mib`（默认 256）：固定的基准状态容量，经
  `ExecutionOptions.state_memory_bytes` 进入原始签名计划。它必须放得下后端
  观测到的可用内存；Rust 拒绝不足的状态容量。注意这是 arena 预算，不是
  模型/KV/scratch 总内存。正常执行默认取观测容量（`state_memory_bytes=0`），
  其变化合法地改变计划签名；固定基准容量的意义在于让精确计时与性能分析的
  重建可复现。
- `--execution-memory-mib`（可选）：把有限的
  `ExecutionOptions.memory_budget_bytes` 传给原始 AER timeline。省略即不加
  限制，零即零容量。它包含所有声明的状态/驻留/预留/工作区/存活结果成本，
  但不约束未声明的模型/KV、分配器或库内存。缺失任务/输出成本或容量不足会
  拒绝规划——同样不减少工作量、不禁用捕获。

报告里会含 `planned_peak_memory_bytes`，与实测 CUDA 峰值分开列出。保守
规划界与真实分配器测量不可互换，报告时不要把一个当成另一个。

## 我要测自己的负载

公共模型负载入口不要求训练或捕获：

```python
from pllm.benchmark import BenchmarkOptions, compare

results = compare(my_workload, BenchmarkOptions(batch=4, tokens=64, rounds=1))
```

这测量普通实现；CUDA 张量任务自动捕获。`explicit_math=True` 加一个
显式张量臂。不支持的请求在正常绑定边界失败。`require_training=True` 要求
正的 loss、update 与训练行计数——观测或仅激活算法不要开，否则会误报失败。
无论开不开，所有运行都检查 token 数、清理、回退与等效工作量。

这个入口测的是模型生成。要计带类型宿主服务的时间，用
[验证指南](validation.md) 里的共享执行 trace，不是合成 GPU 循环。

LoRA 场景还有自己的一组参数：`--rank`、`--training-rows`、
`--training-fraction`、`--scope instance|request|both` 与
`--state-groups N`（默认每请求一个请求状态）。活跃行数向上取整并报告，
剩余权重为零。请求作用域下，组共享 prompt/样本键与相同 prompt 文本；训练
输入按状态组一批，不是按请求。这建模的是同步准入、一起完成的队列（忽略
EOS），不是共享键的独立计时加入或部分退役——解读数字时按这个语义来。
实例作用域仍是单独算法。合成训练张量在模型初始化后用公共模型维度创建，
同一输入经正常带类型辅助输入边界提交；生成、页选择、发布与采样引导都在
被测路径内。共享与独立算法有不同归约，不作为等效负载宣传。

## 报告里的数字各是什么意思

先讲计时口径。冷模型、编译、捕获准备单独报告；预热被排除。JSON 报告软件
版本、基准设置、签名计划标识、实测轮次时间、真实输出 token、loss/update
计数、训练行与峰值分配内存。

`round_process_cpu_seconds` 用 `process_time` 测每轮完整生成与最终设备
同步前后的进程用户+系统 CPU 时间。它包含该进程内的 Python、原生提交与
其他线程，排除睡眠/被调度下时间与子进程，所以可能超过墙钟。它不是 CUDA
API 时长，也不声称全部 CPU 工作都是启动提交。性能分析重放则报告自己的
`profile_round_process_cpu_seconds`；这些插桩时间不替代稳态测量。

再讲内存口径。峰值内存包含持久状态与捕获池。`baseline_allocated_bytes`
在预热后测量；`peak_extra_allocated_bytes` 是该基线之上的峰值，包括任何
驻留增长——它不是「只有临时」的证明。结果还暴露生成后的原始 `state_usage`
快照：arena 预留、逻辑/物理载荷、填充、页所有权与待收 receipt。这是轮末
状态，不是采样峰值，也不是独立分配台账。

所有内部训练对比臂必须产出请求的 token 数、做非零训练、结束时无待收请求
pin/释放或回退，且等效工作量一致。过不了这些检查的运行没有数字可报。

## 我要做性能分析（--profile）

`--profile` 的语义要先理解再解读：它先完成**所有**臂的吞吐计时，然后才
重建每个臂、重复同样预热、对完全相同的测量轮做性能分析。不用额外生成外推
代替更早轮次；轮间更新合法变化。重建的计划签名与聚合有效工作量必须与原始
测量一致，否则分析无效。`profile_rounds` 记录插桩轮数；分析成本与工作量
是这些轮的合计。

一个硬约束：CUPTI 插桩在其上下文退出后可能影响后续 CUDA API 延迟，所以
不要在同一进程里再启动计时对比。LoRA 的 `--scope both --profile ...` 因此
每个作用域用一个全新进程。

trace 按臂导出为 Kineto Chrome trace（含内存事件、无张量快照），并报告
设备内核数、内核微秒合计、CUDA runtime/driver API 微秒合计与真实 GPU
memcpy 字节。解读时注意四条边界：API 时间含同步，不是纯 CPU 提交时间；
重叠内核是求和的，不作为 GPU 墙钟报告；`cuda_launch_api_count` 按线程数
外层宿主启动区间、合并嵌套 runtime/driver 调用，`cuda_graph_launch_api_count`
是其图启动子集，两者都不数每次 replay 内部的设备内核；memcpy 字节不含内核
内访存。这些都是完整生成成本（包括 vLLM），不是孤立的负载任务测量。

性能分析从不贡献稳态计数器或计时，且必须通过同样的等效工作量/清理检查。
已有 trace 文件不被覆盖。示例：

```bash
.venv/bin/python -m tools.benchmark_lora --batch 2 --tokens 4 --rounds 1 \
  --training-rows 4 --training-fraction 0.25 --scope request --profile /tmp/lora-profile
```

## 我要用独立训练输入或自定义实现臂

独立输入用 `pllm.host.WeightedCEBatch.for_prompts(hidden, logits, labels,
weights, prompt_indices)`。它构造原始辅助对象及其 prompt/样本键，不拷贝
张量。`prompt_indices` 必须与传给 `generate` 的一致；共享输入仍用无键的
`pllm.host.WeightedCEBatch(...)`。场景不导入执行帧，不构造请求身份对象。例如
`--batch 8 --scope request --state-groups 2` 在八个请求间共享两个状态。
`compare(..., prompt_indices=...)` 把这些键传给每次公开生成，包括预热与
分析；设置会记录它们以便复现。

用 `alternatives=(("label", workload), ...)` 给同一对比加显式
实现。分页臂现在用设备侧分组内核，无矩阵打包、无宿主偏移同步。ESamp 的
捕获 loss 直接读原始池；其捕获 Adam 现在两次启动直接更新全部五个原始字段，
每状态两个标量偏置因子作为 scratch。LoRA 的分页捕获直接 reset 并更新原始
私有字段，保留其带种子 reset 与按适配器归一化。捕获字段现在共享逻辑分组，
矩阵内核直接消费原始行索引，无单独操作数/输出 gather。被完全覆盖的捕获
输入不再收到多余填充；较小输入仍填充签名填充区。

## 现有验收数字怎么看

2026-09-09 验收运行，默认批次 8 / 256 token / 3 轮，每臂 6144 输出 token：

| 算法 | 张量 | 显式张量 | 张量捕获 | 分页 | 分页捕获 | Loss/Update/行 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ESamp | 865 | 994 | 1046 | 937 | 1040 | 6120/6120/6120 |
| 共享 LoRA | 1463 | 1450 | 1445 | 1385 | 1394 | 6/3/6 |
| 请求 LoRA | 1353 | 1358 | 1359 | 1325 | 1323 | 48/24/48 |

数值为输出 tok/s，只在行内比较——跨行比的是不同算法，不是同一负载的
不同实现。LoRA 每次 generate 调用训练一次，不是每个 token。所有臂通过
等效工作量、清理与零回退检查。日志在
`outputs/acceptance-5Uinna/benchmark-{45,46}.log`。

引用这组数字时必须连同三条限制一起引用：这次运行在选中状态梯度替换之后；
未用性能分析器，不足以为分页默认值背书；GPU 同时持有另一个用户的分配。

## 历史研究在哪、哪些不能再用

当前源码的八配置研究在 `outputs/study-current-AV1WUw/RESULTS.md`：40 个
计时臂加 40 个独立重建的分析臂，各三轮。请求数、训练行/比例、状态共享、
rank、实例/请求与更小内存余量都通过等效工作量、签名与清理检查。读它时
记住四点：提高公开 vLLM KV 预留使预热分配上涨约 3.5 GiB，那不是 OOM/耗尽
测试；没有确立稳定的分页吞吐优势；捕获降低了额外分配峰值，但增加了驻留
存储与 memcpy，不要推断总内存更低；GPU 保持共享。报告把入队/等待 API
区间与 GPU 活跃区间分开；设备内核执行数不是宿主启动调用数，入队时间不是
完整 Python 提交时间——这些限制保持显式，不从计数器推断。随后的 CPU/启动
插桩检查与既有 trace 归因在 `outputs/cpu-launch-euODct/RESULTS.md`；它不
重跑也不取代八案例吞吐矩阵，只有报告变了，负载数学与执行没变。

更早的七配置有界矩阵保留在 `outputs/gpu-matrix-uP6yrj/RESULTS.md`，含其
设置与限制。它早于那次梯度替换，不是当前完整性能研究。更早的交错计时/
分析运行被排除在实现排名之外——CUPTI 可能影响后续计时；其原始日志保留，
但过时吞吐表不与当前运行并存维护。

LoRA 负载证据里保留一份独立数值参考：基准不改写该参考，不把它当替代运行
时调用，也不从计数器一致声称任意数值等价。

历史对齐 ESamp 实验用批次 8、`sampling_n=16`、至少 256 token、bfloat16、
温度 0.8、top-p 0.95 与最大模型长度 ≥512。当前公开附着支持 `n=1`，其绝对
吞吐与那些历史 4k+ tok/s 数字不可比。小验证运行是比例/行为检查，不是吞吐
回退或提升的证据。

## 共享 GPU 礼仪与最后两条规矩

使用仓库虚拟环境，比较完全相同的负载。异常终止后先检查 GPU 进程归属再
重试；永不终止无关 worker。不要把已退役运行时恢复为永久基线。

# tLLM 架构合并、删除与精简计划

## 文档地位

当前职责与接口见 [architecture](architecture.md)、[编写契约](guides/authoring.md)
和根 README；唯一可执行证据清单是 `test/evidence_manifest.toml`。
原交接和逐提交 ledger 已退役，不再与当前文档平行维护。
下文保留历史基线、估算和验收要求；历史目录名不是当前目录设计。
行数下降、旧目录删除或单条公共路径通过都不代表整个计划完成。
纳入可编程推理契约的决定见第 10 节；当前证据编号及执行次数以下述现状和 manifest 为准。

本文是从 2026-09-01 起执行的减法计划。它不把 10k 当作倒填配额，而是先根据当前
代码的可达性、重复职责和待完成功能估算可以删除的范围，再持续向 10k 靠近。

优先原则：

1. 不再以增加兼容层换取迁移进度；
2. 每个迁移 tranche 必须净删除生产代码；
3. workload 切换公共入口时，在同一 tranche 删除旧 authority；
4. 功能完整指公共语义和真实运行边界完整，不指保留旧内部结构；
5. 测试按真实 evidence item 计数，整个仓库不得超过 50 项；
6. 10k 是长期压缩方向，不是牺牲类型、功能或重新制造巨型文件的硬指标。

[`doc/research-aer-paged-state.md`](research-aer-paged-state.md) 仍是 PTTS/AER 的唯一
设计权威。该未跟踪研究文档不得因本计划被修改、加入或提交。

## 当前状态与交接（2026-09-09）

本节只维护现状、有效证据和剩余工作，不再累加逐轮执行日志。下文的历史基线、
终态架构及验收标准仍有效；历史目录名称不是当前目录设计。

### 已完成的架构替换

- 公共入口/import 为 `pllm`，Rust crate 为 `pllm_core`。旧 Python/Rust authority、
  canonical 包及兼容转发已经删除；持久化 `canonical.*` 身份不随目录名改写。
- 全部 11 个内置 workload 共用原 SemanticGraph、MathContract、Rust 决策、
  signed PhysicalPlan、PTTS 根、exact pin 和 issued receipt。没有镜像图、
  第二版本库或 workload 专属 scheduler/session。
- `usage.md` 的适用入口全部迁移：7 个普通模型入口为 Dummy、OverRIDE、
  In-Place-TTT、ReFT、Recirculation、ESamp、Output LoRA；TLM 使用 causal
  adapter，TTRL 使用 completion loss，FTTT/GTTA 使用原 host transaction 的普通函数。
  用户不再手写 build、端口字典、prepare/output 或 implementation ID。
- 加载期反射和依赖冻结限于 `pllm/usage.py`，传输类型擦除限于 codec；执行层 strict。
  方法、配置、来源和 monkey patch 进入原签名，已绑定数学不读取后来替换的配置。
  自定义普通数学与显式 kernel 不匹配时拒绝绑定，不静默替换算法。
- 普通 eager 分别表达 forward/dx/dW/update 行域，冻结参数不截断上游 dx；
  Parameter(per_row=True) 显式表达共享参数逐行梯度。更新资格独立于零梯度，
  未授权状态不推进 Adam step/moments/decay。仍限制一个 optimizer state/训练 sink，
  不支持跨 replay 反传；state_mean 不支持独立 forward/update selector。
- ESamp/LoRA 的显式 tensor/paged/capture 绑定替换同一签定图的计算。
  TensorRows/PagedRows 共用矩阵、反向、归约及私有写入接口；原语按布局/方向/容量桶
  特化，不按 request/version 生成新实现。普通独立行域不自动 lowering 到分页/capture。
- FTTT/GTTA 的 FIFO、abort/requeue 和 host settlement，TLM 的 online/offline、
  admission pin 与 immutable publication，TTRL reward/control，以及参数/激活 overlay、
  条件/tokenwise replay、MRv1/MRv2 和 closed EngineCore 边界保留。支持边界见
  [编写契约](guides/authoring.md)，不以接口统一声称任意组合受支持。

### 已实施的资源收敛

- 临时结果只有 invocation 的原字典一份。Rust 签定最后使用点及梯度依赖的存活区间，
  执行按原 value ID 释放；loss/gradient 导出值保留到完成。普通唯一训练 sink 成功后
  清除导数叶子并 detach 保留结果，包含无训练分支；结果存储、梯度和 staged update 保留。
  完成/取消清空 context；settlement 失败保留原 receipt 的重试数据。
- 连续 hidden 行借用原 snapshot；不规则行在消费任务内打包。BF16→FP32 为一次 owned
  copy，输入登记实际消费 stream，防止 readiness 已满足但读取未完成时 allocator 复用。
  WeightedCE 全有效普通输入直接借用，无效行和 inference-mode 输入保留必要复制。
  普通请求参数在选中行全部引用同一状态时使用广播视图，混合状态才打包；导数叶子
  借用原 pin 的参数存储，不重复计作参数分配。连续采样反馈行也借用原 owned buffer。
  普通 forward 连续行及梯度 mask 复用切片，删除布尔索引复制；全行输出不再 scatter，
  部分行只分配一份补零目标。不规则行仍打包，不改变原行域与反传归属。
- eager/capture 多字段共用逻辑行分组，矩阵直接解引用原行/状态/页槽；删除物理页排序、
  inverse、矩阵前后行 gather 和每字段按行展开的页号。vector bias/norm 仍需 gather。
  eager 多字段页槽合并为一次传输，全私有读写槽复用原存储，host 已知计数不再做 GPU cumsum。
  显式行选择复用原排序做前缀扫描；16384 行内一次 launch，更大容量固定分块。
- ESamp 显式 LayerNorm 反向直接调用原 PyTorch native 算子，不建 autograd tape；
  raw/mean/rstd 仍为必要 scratch。分页 eager/capture 共用原页池上的两个融合 Adam 计算 kernel，
  删除逐字段梯度复制和 optimizer gather/update/scatter；step/bias correction 每状态只计算一次。
  eager 仍构建 launch 页槽表，其已知工作区改按 96 bytes/state 声明，不再按整份参数复制计费。
- LoRA 只有一份 reset/加权 CE/多步 SGD 数学；复用 log-softmax/导数存储，
  固定行权重归一化移出循环。写入 kernel 直接读取 selected/state/slot，覆盖和加法写入
  都应用 scale。完整覆盖模式删除 reset 的无效 COW；增量更新仍保留 COW。
  native 绑定共用一个返回出口，不重复查找资源或维护捕获/非捕获绑定包装。
  共享分页梯度归约直接按原 logical-state 选择写出紧凑结果，删除 LoRA 的整批梯度后置
  gather；tensor 训练直接保留 BMM 梯度，不再复制完整 identity selection。选中状态顺序、
  独立归约、结果跨 replay 存活及 pin/receipt 不变；成本声明仍保留原保守上限。
- 普通/显式 AdamW 共用字段级复制集合：保留 parameter/step/moments 的 COW，
  跳过随后完整覆盖的旧 gradient，包含 TLM 的每个参数组。删除独立 replace 执行模式，
  完整覆盖统一使用空复制集合；不减少私有页容量，不改变原 pin/receipt 和取消语义。
- KernelCapture 使用最终 abort 的 admission 页，删除其冷字段备份；调用者已有私有
  状态的直接 capture 仍恢复字段。ESamp/LoRA 输入改为延迟工厂，先对全部 capture
  做无存储 meta 预算检查，再分配 CUDA 输入；真实输入/字段 scratch 在 cold admission 前核对，
  graph-pool reservation 在 capture 后核算，拒绝时原 arena 回收。
- vLLM hidden snapshot、采样投影/反馈及 overlay/replay 的已知缓冲进入原 timeline；
  安装时从原 signed graph/attachment 核对预算下限，MP 保留完整 timeline。
  全行采样删除 gather/copy-back；参数/激活 patch 直接修改新生成的投影输出；
  连续低秩/replay 行借用视图，tokenwise 各层共用 sequence 张量，normal 输出不再 clone。
  输入、residual、原状态页与两种 replay 的 KV 备份时机不变。
- 输入绑定直接借用原 backend probe，按整批 packed token 行数检查签定容量，
  不以筛选后的 decode 行数或请求数代替；普通/显式数学和 causal 入口在选择状态前
  拒绝超限，vLLM 在创建 step/capture 前拒绝。无状态观察也检查签定 hidden 宽度。
  辅助输入共用原 plan 与 TensorFieldSpec 做提交/feedback 搬运/worker STAGE 前置检查，
  不先导出、映射或复制到 GPU 再发现超限。LoRA 的准备任务签定 hidden/logits 两个形状，
  checkpoint 检查原字段布局和 state scope；排队编辑行数不冒充同时驻留状态数，实际
  cohort 仍由原 arena 准入。causal 显式成本须带 sequences/tokens 上限。
  dtype 取原布局（token/label 为 int64）；内容与实际状态选择仍由消费端检查。
- TLM 离线选择在 adapted forward 前归约并释放完整词表 logits，只保留 detached 的
  每句 loss；数学 oracle 和真实 tiny HF/PEFT CPU/CUDA 检查保留。专属 objective 已从
  公共 kernels 删除，唯一实现在 TLM workload。完整 HF backward scratch 仍未声明，
  不把其无界模式或输入形状检查冒充有限显存保证。
- 冻结 causal 模型的参数/buffer 存储进入原 backend requirements 和 timeline 各 anchor，
  HF 按唯一 storage 计费，含 meta 声明；共享权重和借用的 factor 模板不重复计费。
  HF 调用关闭不需要的 KV/hidden-state/attention 输出，不以驻留量代替 backward 成本。
  handle 成功关闭后清除 engine/runner/execution 引用；shutdown 失败则关闭新工作入口，
  保留 owner 及未确认的共享输入，待 `close()` 重试成功才回收。
- 公共 `ExecutionOptions.memory_budget_bytes` 直接约束原 AER timeline，原始事实、
  workload 成本声明和用户限制取交集；声明不能抹掉已有上限。`None` 不另加限制，
  零为零容量；有限预算缺少任一 task/output 成本仍由 Rust 拒绝。benchmark 共用
  `--execution-memory-mib`，同时报告 signed planned peak 与真实 CUDA peak，不把
  状态容量、捕获容量或未声明的模型/KV/library 内存混作一个保证。
- 已核对的存储负责方、释放边界和容量证据汇总于 `doc/architecture.md` 的资源表。
  审计删除了 trace 截断非连续张量前的整源 flatten 复制，改为有界逻辑索引读取；
  原有事件大小、快照次序及 source shape 不变。CPU/CUDA 的广播、转置、offset 和四种
  dtype 进入原 Dummy 证据；不新增测试 ID，也不把有限快照预算误当整源打包预算。
- Dummy、OverRIDE、普通 In-Place-TTT 和 ReFT 在请求有限预算或已有 task 成本时，
  补齐原 timeline 的 task/output 声明，不改变无界普通使用。worker 从原事实重算下限，
  拒绝重新签名但少报的 workspace/output。分别覆盖 hidden 打包与导出、head/CE/Adam、
  chunk/tail/dense 更新及字段有限值检查；dense operand 按目标累计检查 packed-row 上限，
  ReFT 内容检查移入原 update 任务且在任何字段复制前执行，失败沿原 abort 回收。
  内置数学修改不能静默继承公式；条件模型 replay、完整 HF backward 和库 scratch
  仍不在内置有限成本保证内。
  普通绑定统一核对其公式，删除 Dummy/In-Place-TTT/ReFT 的重复 operations；
  OverRIDE 只因其额外行域准备边界保留绑定。GTTA 查询删除候选/历史匹配的中间
  tuple 及 ordinal 包装，稳定排序保持并列顺序，限量结果生成后释放完整排序列表；
  原历史与 rollback snapshot 没有删除或改换 owner。
- TTRL 有限 tensor 成本接入原 host timeline：`HostRuntimeFacts` 提供总声明预算与
  输入行上限；原 completion hidden、head/MSE/Adam 工作区及结果均进入原 task/value
  事实。prepare 前拒绝超限，CUDA 输入登记消费 stream；修改数学/控制准备不能继承
  内置公式。host binding 保留原事实，删除 compile 时再传另一套 facts 的入口。
  五项受影响 host/数学证据通过，另定点复核 source/cost 拒绝；strict/import/authority
  通过。公共 vLLM 原证据把三条 completion 交给 1 MiB 预算 TTRL，更新一次、声明
  峰值 183,512 bytes、fallback=0；日志 `outputs/completion-costs-r4ejCZ/public.log`。
  未增加 evidence ID。文本、Python metadata/history、库 scratch 不属于该 tensor 预算；
  不把 state capacity 冒充 host 事务总成本。
- FTTT 的参数/moments、回滚快照、当前 hidden、feature/train/query/result tensor
  成本也接入原 host timeline。通用 executor 在 Rust 准入后才调用 state_factory，
  删除声明时先分配状态的入口；输入数量与事务元数据对齐，prepare 在快照前检查行上限。
  原 FTTT/GTTA/host 三项证据及 1 MiB 预算 CUDA smoke 通过，loss/update/version=1、
  fallback=0；包含精确预算、少一字节且未构造状态、超限拒绝及原 abort 语义。
  strict/import/authority 通过，日志 `outputs/host-costs-3E8FP9/`；不新增 evidence ID。
  队列、文本、judge、Python 历史和库 scratch 不在这个单次执行 tensor 预算内。
- 公共 diagnostics 的 `state_usage` 从原 arena 直接派生逻辑存活字节、COW 去重后的
  payload、预留 padding、空闲/当前/私有/退役 field pages、版本/pin/receipt 和准入容量
  拒绝次数；benchmark 复用这份只读快照。读取不等待、不收集或选择版本，未建镜像
  资源账本；计数是所有权分类，不把 private/retiring 冒充 GPU 正在运行/已经完成。
  原 CPU lifecycle、CUDA fence 和 Dummy/codec 三项证据通过，包含快照不触发回收、
  共享字段只计一次、旧 pin、两类容量拒绝及清理归零。公共普通 ESamp 运行也报告
  loss/update/rows=6/6/6、30 个空闲字段页和零待释放 owner。

这些是已知 tensor/owner 的收敛，不是任意 autograd、上游库 scratch 或模型/KV 总显存的
完整容量证明。AER 正确计算已声明预算，不等于成本声明已经完备。
ESamp/LoRA 的普通数学修改也不再静默继承内置成本公式：specialization 和 worker 绑定
均核对原数学，修改后须显式提供 resource_facts；自定义成本不绕过 native kernel 的匹配检查。

### 当前证据与测量边界

所有 Python 验证使用 `/workspace/pllm/.venv/bin/python`。环境为
PyTorch 2.13.0+cu129、vLLM 0.27.1+cu129、Triton 3.7.1、RTX 4090/SM89。
唯一清单 `test/evidence_manifest.toml` 为 **47 个 ID / 50 次选定执行**，不另建测试体系。

- 本轮集中验收覆盖 26 项 pytest、8 项 Rust、13 条 runtime 和 3 组标准 benchmark，
  strict 0/0，import/authority 零违规。全量 pytest 后，完整 EngineCore 揭示 checkpoint
  预检混淆排队行数与同时准入容量；删除该限制后定点复核 ReFT、完整 EngineCore 和
  release wheel，其余 runtime 及三个 benchmark 均通过。没有提高测试中的并发容量。
  这完成现有机械门，不替代下述资源完备性与性能研究退出项。日志及执行顺序见
  `outputs/acceptance-5Uinna/RESULTS.md`；失败记录保留，没有增加 evidence ID。
  随后的报告字段/arena 可观察性改动使用 `.venv` 定点检查：3 项 pytest（含 CUDA）
  通过，strict 0/0，短程五臂 LoRA 和普通 ESamp 通过。
- 随后的四条内置成本改造使用同一 `.venv` 定点复核：四项原 pytest 修正后通过，
  strict 0/0；原 OverRIDE、In-Place-TTT、ReFT GPU smoke 均在当前环境离线运行，
  使用 1 GiB 声明预算，分别观察到 4 次更新、6 次更新、1 次发布，fallback 均为零。
  原测试覆盖精确峰值可准入、少一字节拒绝以及重新签名的少报成本拒绝；没有新增
  evidence ID 或重跑全矩阵。日志为 `outputs/builtin-costs-Oq8drf/*.log`。
- Evidence 39 现从 sdist 离线构建优化的 release wheel（`--release`），仅安装到测试
  临时目录，检查完整 Python 源码集合、native 来源、许可证、11 个 workload 导入及
  Rust codec。包含本批全部源码的最新候选产物在 `outputs/final-candidate-ZxqFZw/`，
  同次全量 26 项 pytest（无 skip）、8 项 Rust、strict/import/authority 通过；未重跑
  GPU runtime/benchmark 矩阵。源码及产物 SHA256 见该目录 `RESULTS.md`，旧候选不再
  标作当前源码。wheel 隔离检查不初始化 CUDA，真实 GPU 证据使用仓库源码环境；
  仓库虚拟环境未修改。候选构建不表示已发布或整个计划已经完成。
- 已有真实边界证据含非零 loss/update、原版本隔离/receipt 和 cleanup/fallback=0：
  ESamp capture 4 次更新、MRv2、MRv1、EngineCore、dense/activation/低秩 patch、
  条件及 tokenwise replay。TLM online/offline 各 48 次独立 QKV 对照；
  OverRIDE update=4、sample checks=6、末轮无更新。修改未涉及的边界不重复铺矩阵。
  TLM 的 HF/vLLM 必须使用相同来源：离线运行显式给两侧同一缓存 snapshot 路径，
  不混用仓库 ID 与 vLLM 改写后的路径，也不放宽来源检查。最新 EngineCore TLM 定点
  检查通过，冻结模型驻留 988,336,000 bytes、loss/update=1、父进程 CUDA 未初始化，
  关闭后子进程退出且 completion 仍有效；日志 `outputs/resident-close-9aNBjw/run.log`。
- 冻结 Huber reference：`test/reference_huber.py`，SHA256
  `e853832ffb6904c5dc8b2868e39af896ffc72b4518ebc3744049315f699b6281`。
  独立 gpt-6 适配没有修改核心/reference 或放宽容差；CPU/CUDA、重排、不等长更新、
  全部 Adam 字段、旧 pin、取消与清理通过。仅证明普通 eager，不声称分页/capture/vLLM。
- 定点优化证据：五字段 Adam capture 的 CUDA 事件由 76/94/118 降到 2；
  双字段逻辑映射原替换为 23/23/27 事件；本批仅映射构建由 18 降到 17，
  2/64、8/512、16/2048 states/rows 的保留分配从 3072/17408/68096 降到
  2048/9216/35328 bytes。这些不是端到端吞吐或完整临时峰值结论。
- 公共 benchmark 支持 rank、训练行数/比例、shared/request、同步 cohort 的 state groups、
  显式 capture 容量、独立 profiler 和原输入准备接口。共享键不支持独立到达/部分退休。
  普通 model workload 默认只测普通实现，不再强制 capture 或非零训练；内部训练对比
  显式开启两项要求。profiler 已改为新实例重放相同预热及全部测量轮次，核对原计划签名
  与累计工作量，不再用额外一轮推算多轮。benchmark 显式签定 256 MiB 状态预算，
  超出可用容量或不足以容纳状态均拒绝，不放宽重建签名检查。
  当前源码八组 LoRA 比较已完成 **40 个计时臂 / 40 个独立 profiler 臂**，覆盖请求数、
  训练行数/比例、状态共享、rank、instance/request 及较小显存余量。每臂三轮计时，
  等工作量、精确签名、非零训练、零 fallback 和 cleanup 全部通过；子进程均已退出。
  报告 `outputs/study-current-AV1WUw/RESULTS.md` 包含逐臂冷启动、轮次时间、声明峰值、
  真实分配、API enqueue/wait、GPU 活跃时间、kernel 次数和 memcpy。没有稳定分页
  吞吐优势，保持普通默认；capture 增量峰值较低不等于总显存更低。压力组通过公开
  vLLM KV 预留增加约 3.5 GiB 真实分配，不是耗尽测试。GPU 与其他作业共享，短测
  不构成统计显著性或发布 SLO；全栈 CPU 提交与 host launch 次数仍不能由 kernel 次数代替。
  后续直接重读 baseline/rows/pressure 的 15 份 trace，旧指标逐项一致；新增 host launch
  计数合并同线程嵌套 runtime/driver，graph launch 单列。三组均约 14.8 万次 host launch，
  capture 仅减少约百次；普通基线约 62.5% kernel 时间属于模型 BF16 GEMM，不能据此
  继续堆 LoRA 融合。公共 benchmark 已记录每轮进程 CPU 时间，与 API/GPU 时间分开；
  旧矩阵没有该时钟，不能倒推补数。详情 `outputs/cpu-launch-euODct/RESULTS.md`。
  最新公共有限预算检查在 1 GiB 声明上限下完成五臂等工作量执行；普通路径声明峰值
  214,105,000 bytes，213,909,504-byte 上限在生成前拒绝。日志在
  `outputs/finite-aer-ClI0fC/{run,rejected}.log`；这验证声明容量准入，不是总显存压力研究。
  32 行 LoRA 暴露了 256 MiB 硬编码上限，已改为公开 `--capture-memory-mib`；
  原 270542404-byte 需求拒绝记录保留，未 fallback 或减少训练量。
- 较早测量的原始日志保留，但不与当前源码结果平行维护性能结论。逐臂插入 profiler
  的旧吞吐排除；旧七组矩阵早于 selected-gradient 替换。来源与限制见
  [benchmarking](guides/benchmarking.md)，不以历史较快数字覆盖当前测量。

### 终态核对与剩余证据

| 要求 | 当前判定与依据 |
| --- | --- |
| 第 7.1、9 节：唯一执行路径、旧 authority 删除 | 已实施。当前 owner 表、零 import/authority 违规、原签名重建与完整 workload 证据相互印证；没有兼容栈或第二调度/版本库。 |
| 第 7.2 节：PTTS/AER | 已实施支持范围内的准入、原子发布、exact pin、ABA、CUDA receipt 与取消/结算；当前 CPU/CUDA 及 8 项 Rust oracle 通过。分布式本地执行仍拒绝。 |
| 第 7.3 节及 usage 接口 | 11 个内置 workload 经同一公共工厂，普通方法、causal/completion 和 host 事务使用各自适用的共享边界；原 13 条真实 runtime 与近期受影响有限预算路径有证据，不声称任意组合。 |
| 第 7.4 节及第二轮压缩 | 当前 20,032 行，相对原 133,210 行净减 113,178 行；47 ID / 50 次选定执行，主 worktree 唯一，无活动空壳目录。10k 仍是方向，不以隐藏源码或删语义达到。 |
| 第 10 节：学习/发布/条件 replay、来源与独立 reference | 原 40/41 组合证据及有界 trace 保留，使用原图、pin、receipt 和 KV/RNG 边界；不是全模型 capture 或跨 replay 反传的声明。 |
| 下一阶段第 8.1、8.2 节：契约、名称、共享原语替换 | 已实施，普通/分页/capture 复用共享数学，旧展开与搬运实现原子删除。2026-09-09 公开 PyPI `pllm` JSON 查询为 404；仅说明未发现公开项目，不代表名称已注册或账户有发布权限。 |
| 下一阶段第 8.4 节：性能与候选包 | 八组五臂比较、冷/热及容量观察、主要 kernel/launch 归因和当前源码候选包均有证据。结果不支持分页默认，也不支持继续无依据堆融合；保留普通默认。共享 GPU、短测、旧矩阵无进程 CPU 时钟及未耗尽显存的限制不变，不声称发布 SLO。 |
| 下一阶段第 8.3 节：独立适配流程 | 数学及生命周期结果已验证，核心未因 Huber 增加专属路径；但旧独立适配过程未保留可核实的重写次数、人类澄清及失败记录。不能从最终源码反推“一次成功”或“零澄清”，这是尚未闭合的过程证据。 |

**资源成本判定：** 已审查原 owner 的保存值、参数打包、行映射、冷捕获和完成/取消/
失败结算。ESamp、LoRA、Dummy、OverRIDE、普通 In-Place-TTT、ReFT、TTRL、FTTT
提供已声明 tensor 成本；有限请求缺少任一 task/output 成本时拒绝。`CausalModel` 的
logits/factor 接口不提供模型内部 forward/backward 成本，模型 replay 和 GTTA/Python
历史也不能套用上述公式。这些是明确不提供内置有限保证的范围，不是零字节资源。
保留其无界合法语义，不增设全自动 PyTorch/任意 Python heap 分析器作为完成前置项；
不把已声明 AER 峰值、用户自定义成本或当前绿色测试当作总进程/总显存证明。

**下一步仅补齐独立适配的可审计过程记录，然后作最终完成声明。** 若旧过程记录无法
恢复，需要未参与接口设计的人或 agent 重新完成一次现有 reference 的独立接入，记录
澄清、重写和失败；不修改冻结 reference、不增加核心专属分支、不新增证据 ID。
当前缺的是这项流程证据，不是再跑同一整套测试、重建候选包或重跑吞吐矩阵。
不提交、打 tag 或远端发布。

最近一次源码统计口径为 `pllm tools examples rust/pllm_core/src` 下全部实际 .py/.pyi/.rs
（含未跟踪文件）：**20,032 行**。CPU/host-launch 报告与原 arena 可观察性增加 74 行，
随后四条内置成本及相关输入边界增加 134 行，不将功能补齐记成净删除。
随后公共成本绑定去重与 GTTA 查询替换净删 48 行；三个受影响成本路径及 GTTA/host
原证据定点通过，strict/import/authority 通过，不重复 GPU 矩阵。
TTRL 成本及原 host 事实绑定增加 64 行，不将这部分功能补齐记成净删除。
FTTT 成本与准入后初始化增加 35 行。
普通接口、独立行域、共享分页及资源边界完善相对
17,922 的增长尚未抵消；低于历史 30k 上限不能代替功能、资源与性能验收。

仓库仅主 worktree，旧生产目录无空壳；失效内部文档/import baseline 已删除。
本次现状段整理前的可恢复副本在
`/tmp/pllm-write-audit.Gw2MwA/plan-before-consolidation.md`；
它不作为另一份维护文档。PTTS/AER 研究原稿未修改、未加入或提交。

## 1. 当前基线

基线为提交 `12551b8`，统计采用受 Git 跟踪文件的物理行数。

| 范围 | 文件数 | 物理行数/测试数 |
| --- | ---: | ---: |
| 第一方 `.py`/`.rs`，排除 `test/` 与 `typings/` | 427 | 132,361 行 |
| 第一方 `tllm/_core.pyi` | 1 | 849 行 |
| **生产及支撑源码合计** | — | **133,210 行** |
| `test/` Python | 206 | 65,073 行 |
| Python `test_*` 函数 | 167 个文件 | 1,281 个函数 |
| Rust `#[test]`/`#[tokio::test]` | — | 54 项 |

参数化 collection 和独立 GPU smoke 尚未计入 1,281，因此当前真实测试项多于 1,335。

上述基线已于 2026-09-01 按第 3 节口径实测复核：pytest 实际 collect 1,349 项（加
Rust 54 项，合计 1,403）；`schedule.rs` 恰含 38 个 test；新栈
（canonical/contracts/canonical_runtime/canonical_vllm）对旧栈 import 为零，
clean-room 隔离成立，第 2.1 节的整块删除在机械上可执行。

当前 clean-room 相关代码已经达到 27,742 行：

| 范围 | 当前行数 |
| --- | ---: |
| `tllm/canonical/` | 16,580 |
| `tllm/contracts/` | 4,303 |
| `tllm/canonical_runtime/` | 2,253 |
| `tllm/canonical_vllm/` | 1,376 |
| `rust/tllm_core/src/canonical/` | 3,230 |

其中 canonical workload 已经出现明显重复：

| workload | 当前行数/文件 |
| --- | ---: |
| FTTT | 2,634 / 10 |
| GTTA | 1,180 / 9 |
| In-Place-TTT | 2,623 / 9 |
| TLM foundation | 231 / 2 |
| flat Dummy/ESamp | 2,868 / 9 |

GTTA 和 In-Place-TTT 已删除旧 consumer，仍然显著净增长。因此这不是普通双栈过渡，
而是 CPU/CUDA 状态机、plan DTO、application/session/execution 和 backend attachment 被
重复实现。

## 2. 能删多少：基于当前仓库的估算

### 2.1 可整块删除的旧 authority

以下是当前目录的实测值，不是目标倒推：

| 最终退役集合 | 当前行数 | 删除依据 |
| --- | ---: | --- |
| `runtime/workgraph/schedule/workload/program/consumers/kit/ports` | 73,578 | 最终全部由 canonical graph/execution/typed services 取代 |
| Rust 非 canonical 部分 | 18,715 | 主要是 `schedule.rs` 9,000、`workgraph.rs` 5,147、`recurrence.rs` 2,321 及旧 placement/plan |
| **旧栈 footprint** | **92,293** | `lib.rs` 等极小入口会改写，数量级不受影响 |

这批删除完成后，133,210 行会机械下降到约 40,900 行。旧 consumer 中真正独立的 tensor
数学、reward、environment 或 loader 会被抽到新 owner，因此 92,293 是 gross deletion，
不是最终 net deletion。

### 2.2 删除旧栈前后仍需新增的真实能力

当前 canonical 尚未包含完整的：

- AER regionization 和其 correctness oracle；
- closed typed MP transport；
- workload-independent profiling；
- TLM 完整 online/offline training/publication；
- ReFT、Recirculation、OverRIDE、TTRL 的最终 compact declaration/math；
- ESamp 完整 request PTTS/sampler/capture 终态。

从现有旧 consumer 中可识别的独立数学和 host implementation，加上上述尚缺核心，预计
还需 5,000–7,500 行。3,500 的下限要求独立数学全部复用且 AER/MP/profiling 极度紧凑，
按现有证据不现实，不再列入估算。也就是说，单纯删旧目录后，实际待精简的新栈规模会在
约 45,900–48,400 行，而不是 40,900 行。

### 2.3 当前可以点名的重复代码

下面是可以通过现有文件直接定位的重复，不依赖未来猜测：

| 重复集合 | 当前 footprint | 合并判断 |
| --- | ---: | --- |
| FTTT application/session/execution/tasks/workload/init | 1,580 | 数学、judge 和 host payload 之外应由 generic executor/host transaction 接管 |
| GTTA application/session/execution/tasks/workload/init | 765 | state/environment 保留，编排删除 |
| In-Place CPU/CUDA execution/tasks + application/workload/init | 2,164 | CPU/CUDA 只保留 provider kernel，不保留两套 transition protocol |
| ESamp request CPU/CUDA execution/tasks/control | 2,181 | 合并成同一 signed transition 和 provider implementation |
| `canonical_runtime` deployment/session/runner/worker/llm | 2,253 | 两条 workload 硬编码链（session/deployment/runtime/runner import ESamp，`parameter_overlay_*` import InPlaceTTT），是「不新增 workload-specific runtime」的现行违例；整条删除，由一个真 generic executor/backend seam 取代 |
| **可见重复编排合计** | **8,943** | generic replacement 仍需一份，因此不是全部净删除 |

这 8,943 行合并后仍需约 1,500–3,000 行 generic execution/attachment 代码，可信净删除为
约 6,000–7,500 行。

另外还有三组可压缩对象：

- Python/Rust backend、parameter-overlay、sampler attachment 的 envelope、digest、codec 和
  validation 重复，预计净删 1,000–2,000 行；其中实测 Python 侧有 15 个 `validate_*`
  函数（state_representation/backend_attachment/parameter_overlay_attachment/
  tensor_layout/scheduler/dense_page_validation）复制 Rust signed validation，删除由
  Tranche B item 5 执行，worker tamper 检测改调 Rust codec 验证；
- `tllm/backends` 与 `canonical_vllm` 当前合计 7,500 行，保留 MRv1/MRv2 与 best-effort
  旧版本后，预计能合并到 4,000–6,000 行，净删 1,500–3,500 行；
- common/producers、workflows、tools、examples 与顶层入口当前约 6,000 行（仓库中不
  存在独立的 facade 目录，原计划中的 facade 一项并入此组），旧调用者退役后预计保留
  2,000–3,500 行，净删 2,500–4,000 行；其中
  `workflows/benchmarks/per_request_esamp_benchmark.py` 一个文件 1,274 行，超过所有
  生产模块，点名瘦身或删除。

### 2.4 可信终态范围

按必须保留的职责自底向上估算：

| 终态职责 | 当前证据 | 预计终态 |
| --- | --- | ---: |
| Rust graph/interner/AER/scheduler/representation/FFI | canonical Rust 已有 3,230 行，但 AER 未完成；attachment 可去重 | 3,500–5,000 |
| Python typed schema/authoring/closed codec | contracts + canonical top-level 当前约 7,000 行，存在重复 validation/DTO | 2,500–4,000 |
| PTTS/generic execution/host/MP/profiling | PTTS + canonical runtime 当前约 6,300 行，MP/profiling 未完成 | 4,500–7,000 |
| 十个 workload 的声明、独立数学、loader | 当前六个 canonical workload 约 9,500 行，四个仍需抽取 | 4,000–6,500 |
| vLLM/backend adapters | old + canonical adapter 当前约 7,500 行 | 4,000–6,000 |
| 工具、workflows、必要 benchmark/release support | 当前约 6,000 行 | 2,000–3,500 |
| **总计外沿** | — | **20,500–32,000** |

考虑各项共享后的相关性，当前最可信的工程区间是 **22,000–30,000 行**，中心估计约
**26,000 行**。对应生产及支撑源码净删除约 **103,000–111,000 行（77%–83%）**。

因此：

- 30k 是第一轮完整迁移的失败线：超过它说明上述重复尚未真正合并；
- 22k–30k 是当前有代码证据支持的验收区间，不是固定配额；其中 22k 外沿依赖第 2.2
  节已判定不现实的新增下限，现实落点是 26k–30k；
- 达到该区间后继续逐 owner 做第二轮 reachability/duplication 审计；
- 10k 仍是长期方向，但当前没有证据证明在保留十个 workload、AER/PTTS、typed MP、
  profiling 和旧 vLLM best-effort adapter 的同时能达到 10k；
- 不得为了命中 10k 删除功能、弱化类型、把代码藏到测试/工具，或把职责压回巨型文件。

## 3. 不可规避的计量口径

`SOURCE_LOC` 统计所有第一方、受 Git 跟踪、手写的 `.py`、`.pyi` 和 `.rs`，排除：

- `test/`；
- 只描述第三方 API 的 `typings/`；
- 构建产物、缓存、文档和冻结 golden。

`tllm/`、Rust、tools、example、benchmark、backend adapter 和第一方 stub 全部计入。
不得通过移动目录、字符串模板、smoke 或 typing stub 规避统计。透明命令为：

```bash
git ls-files \
  | rg '\.(py|pyi|rs)$' \
  | rg -v '^(test|typings)/' \
  | xargs wc -l
```

每个 tranche 同时记录：新增、删除、净变化、移动而未改变的行数、抽取保留的独立数学，
以及与第 2 节估算的偏差。不能只报告最终总数。

## 4. 终态架构

```text
typed authoring descriptors
        |
canonical SemanticGraph
        |
Rust dense IDs / AER / scheduler / representation selection
        |
signed PhysicalPlan（机械投影）
        |
one generic typed Python execution layer
   |              |               |
PTTS provider   host services   closed MP wire
CPU/CUDA            |               |
   +----------------+---------------+
                    |
             thin backend adapter
```

硬约束：

- workload 只声明数学、数据流、effect、state 和 timing constraint；
- workload 不声明 lane、priority、provider、fusion 或手写执行顺序；
- Rust 是调度、regionization、placement 和 representation 的唯一 authority；
- PhysicalPlan 只投影 Rust 已签名决策，worker 不重选、不猜测、不 fallback；
- 新旧共存只在最外层 routing boundary，最终 routing boundary 也删除旧分支；
- Rust 拥有 graph/dense IDs/signed validation；Python 动态解析只在 Python–Rust codec；
- Python 其他目录不出现 `Any`、宽 `object`、`cast`、反射发现或字符串 type path；
- 运行时类型守卫只存在于 FFI decode 和公共入口边界；typed-internal 层禁止防御性
  isinstance/`type(x) is not` 守卫——strict Pyright 已保证调用方类型，实测约 80 处
  此类守卫（集中在 session/host_service/deployment/hidden_capture）随所属模块重构删除；
- contract DTO 只携带 typed 数据与 effect 声明，不携带 Callable/callback，也不定义
  「存参数 + 转发一次」的浅 wrapper（现行违例：`contracts/typed_replay.py` 的
  `replay: Callable`、`LayerReplayCall`、`FinalNormCall`，随 Recirculation 切换为
  typed row-selection + effect 声明后删除）；
- CPU/CUDA 共享 task/transition/state contract，仅 provider kernel 不同；
- request/global overlay 复用一个 versioned state capability；
- host service 复用 typed transaction/receipt，不按 workload 建 session；
- 最终不存在 Program、BaseConsumer、旧 WorkGraph 或 residual global workload authority。

大文件按 authority 拆分，而不是按行数机械切片。`schedule.rs` 等旧巨型文件冻结后删除，
不在原文件内大爆炸重写。任何新模块必须拥有不可合并的独立职责；只转发一次调用、复制
参数或重导出符号的浅 wrapper 不得存在。

## 5. 测试清理

### 5.1 计数规则

`TEST_COUNT` 是以下总和：

- pytest 实际 collected items，参数化 case 分别计数；
- `cargo test -- --list` 列出的 Rust tests；
- pytest 外 correctness smoke；
- benchmark 中兼任 correctness gate 的模式。

skipped、xfail、改名为脚本或藏进 helper 仍计数。静态 typing/import/LOC gate 不计为行为
测试。最终硬约束是 **`TEST_COUNT <= 50`**；当前明确选择 39 项终态证据，为尚未发现的
真实 runtime boundary 留 11 项余量。余量不能预先用装饰性测试填满。

### 5.2 预计删除量

当前已有至少 1,335 个声明测试和 65,073 行 Python tests。压到最多 50 项意味着删除至少
96% 的行为测试项。按保留完整 lifecycle smoke 的实际长度估计，测试/oracle 最终约
3,000–7,000 行，可净删约 58,000–62,000 行。行数是 forecast，50 项才是硬门。

### 5.3 明确保留的 39 项证据

这里列的是终态 evidence item，不是原样保留 39 个现有文件。一个现有文件中的数十个
case 会被合并为一个观察稳定边界的 lifecycle/oracle；路径会随旧栈删除而变化。

#### Rust canonical core：8 项

1. `layout_and_dense_ids_are_canonical_and_tamper_evident`：TensorLayout、LayoutId、dense
   interner 的确定性及篡改拒绝。
2. `semantic_graph_closed_codec_rejects_schema_drift`：SemanticGraph closed codec、签名、
   unknown/missing/duplicate fields。
3. `scheduler_matches_exhaustive_small_dag_oracle`：通用 scheduler 与 exhaustive 小 DAG
   oracle 一致。
4. `aer_matches_bruteforce_partition_oracle`：AER regionization 与 brute-force partition
   oracle 一致。
5. `representation_selection_and_capacity_are_signed_once`：representation/provider 只在
   Rust 选择一次，capacity facts 进入 signed decision。
6. `physical_plan_is_mechanical_and_reconstruction_is_exact`：PhysicalPlan 机械投影，
   reconstruction 拒绝篡改。
7. `versioned_state_hazards_and_timing_match_oracle`：RAW/WAR/WAW、state version 和 timing
   constraints 的小图 oracle。
8. `distributed_placement_transfer_matches_small_topology_oracle`：只保留一个最小
   distributed placement/transfer topology oracle。

现有 `rust/tllm_core/src/canonical/{tensor_layout,scheduler,state_representation}.rs` 的三个
测试分别作为第 1、3、5 项基础。旧 `schedule.rs` 的 38 个测试不保留；其中仍有效的
算法性质只允许并入上述 oracle。

#### PTTS：6 项

9. `cpu_request_ptts_admit_cow_publish_pin_retire`：CPU request state 的 admission、COW、
   atomic publication、exact pin、retire 完整生命周期。
10. `multi_lineage_admission_and_publication_roll_back_atomically`：capacity/factory/transition
    失败不得产生部分 admission 或 publication。
11. `global_multifactor_adamw_matches_oracle_and_publishes_one_version`：TLM A/B factor、
    gradient、step、moments 的数值 oracle 和 multi-root publication。
12. `generation_receipt_blocks_aba_and_double_settlement`：generational identity、issued
    receipt、slot reuse 和 one-shot settlement。
13. `cuda_ptts_waits_for_exact_fence_before_reclaim`：CUDA COW、pin release、completion fence
    和 page reclaim。
14. `cuda_graph_pagebundle_keeps_stable_addresses_across_versions`：capture/replay 期间
    PageBundle 地址稳定而 logical version 正确变化。

第 9 项以 `test/core/test_canonical_dense_state_producer_unit.py` 为基础；第 13 项以
`test/gpu/test_canonical_cuda_dense_page_smoke.py` 为基础。旧 request-slot、paged-state、
parameter-overlay lifecycle 测试只抽取未覆盖反例，随后删除。

#### Python–Rust/runtime boundary：5 项

15. `signed_attachment_binds_exact_tasks_fields_stages_and_rejects_tamper`：合并 backend、
    sampler 和 parameter-overlay attachment 的 exact binding/tamper proof。
16. `closed_mp_codec_round_trips_registered_payload_and_shared_tensors`：closed typed payload
    与共享 tensor 往返，明确证明无 type-path import/reflection。
17. `mp_transaction_commit_abort_and_quorum_failure_are_atomic`：prepare/commit/abort、worker
    quorum 和 fail-stop 在一个生命周期中证明。
18. `worker_reconstructs_exact_signed_plan_and_rejects_local_drift`：worker 只重建及验证，
    不进行 local selection/fallback。
19. `typed_host_transaction_abort_requeue_and_postpublish_settlement`：generic host service 的
    pre-commit abort/requeue 和 irreversible publication 后 settlement。

`test_canonical_backend_attachment_unit.py` 与
`test_parameter_overlay_attachment_unit.py` 合并为第 15 项，不再各自维护一套数百行 DTO
fixture。旧 MP control/port transport/worker reconstruction 测试只抽取第 16–18 项需要的
边界反例。

#### 每个 workload 一个语义 oracle：10 项

20. `dummy_runs_graph_sign_plan_execute`：最小 SemanticGraph 到 execution 路径。
21. `fttt_matches_edit_oracle_and_settles_judge_retry`：edit 数值、judge、retry、receipt。
22. `gtta_updates_environment_and_rolls_back_fifo`：environment FIFO、update/query、abort
    requeue。
23. `inplace_ttt_matches_multitarget_oracle_and_cleans_versions`：exact toy、多 target、
    publication、cleanup。
24. `tlm_matches_objective_online_offline_graph_and_qv_publication`：官方 objective、q/v cold
    facts、online/offline timing 和 LoRA publication。
25. `esamp_predicts_before_training_and_isolates_request_versions`：固定 seed、request
    isolation、sampler guidance、版本和 cleanup。
26. `reft_matches_loreft_math_and_publish_skip_semantics`：LoReFT 数学、publish 和无更新分支。
27. `recirculation_selects_rows_and_replays_typed_effect`：row selection、layer replay 和版本
    lifecycle。
28. `override_matches_reweight_adam_and_final_noop`：reweight、Adam update 和 final no-op。
29. `ttrl_matches_reward_update_retry_and_control_quorum`：reward selection、conditional update、
    retry/control output。

现有六个 canonical workload unit 测试（其中 TLM 为
`test/core/test_canonical_tlm_foundation_unit.py`）是第 20–25 项的基础。ReFT、
Recirculation、OverRIDE、TTRL 只从旧 consumer tests 冻结数学 oracle/golden；不保留旧
runtime/class hierarchy assertions。

#### 真实程序–运行时边界：10 项

30. `public_mrv2_esamp_trains_pins_sampler_and_cleans_request`：公共入口上的真实 hidden
    capture、sampler intervention、PTTS、非零训练和 cleanup。
31. `mrv1_adapter_captures_exact_rows_and_sampler_links`：只验证 MRv1 与 MRv2 真正不同的
    adapter path，不复制完整 workload matrix。
32. `public_vllm_dense_overlay_pins_two_request_versions`：In-Place-TTT DENSE_ADD、两个请求、
    exact version pin/release。
33. `public_vllm_low_rank_overlay_pins_and_releases_global_version`：TLM LOW_RANK_ADD、global
    publication、admission pin 和 cleanup。
34. `public_vllm_reft_applies_activation_patch`：真实 activation patch adapter。
35. `public_vllm_recirculation_replays_signed_layer_rows`：真实 typed layer replay adapter。
36. `enginecore_mp_executes_closed_plan_and_returns_quorum`：真实 child worker、closed wire、
    exact plan 和 quorum。
37. `cuda_graph_replay_preserves_training_and_zero_fallback`：capture/replay 地址与非零
    loss/update，fallback 为零。
38. `fttt_cuda_host_receipt_crosses_stream_boundary`：typed host receipt 与 CUDA stream/fence
    边界。
39. `isolated_wheel_imports_and_executes_rust_codec`：fresh wheel、隔离目录 import 和 Rust
    codec execution。

当前 `vllm_esamp_smoke.py`、`vllm_inplace_ttt_smoke.py` 和
`fttt_cuda_smoke.py` 分别是第 30、32、38 项基础。旧 MP/TLM/ReFT/
Recirculation smoke 只能抽取场景，必须改为新公共入口后才可进入 survivor list。

GTTA 是 CPU host-service workload，Dummy 不拥有 backend，二者不增加 GPU smoke。
OverRIDE/TTRL 若最终只消费已经被上述证据覆盖的 generic native task/typed port，也不各自
复制 vLLM smoke；只有发现新的、不可由通用边界覆盖的 adapter mechanism 时，才能使用
11 项余量之一。

### 5.4 明确删除范围

在对应 oracle/golden 抽取完成、生产入口切换后，按 workload 切换提交逐个删除
（见 Tranche A/C），而不是一次性删除：

- `test/runtime/`；
- `test/workgraph/`；
- `test/program/`；
- `test/workload/`；
- `test/consumers/`；
- 不在第 30–39 项中的旧 GPU smoke；
- import-layer/region-authority gate 的 meta-tests，CI 直接执行静态 gate；
- Python 重复 Rust validator 的 tests；
- typed-internal 层的防御性运行时类型守卫及其测试（strict Pyright 之外的装饰性
  防御，含 `reportUnnecessaryIsInstance` 抑制配 "untyped callers" 注释的模式）；
- dataclass、enum、`repr`、getter、re-export 和简单 mapping tests；
- 只断言 mock 调用次数、兼容 wrapper、registry 或旧 class hierarchy 的 tests；
- MRv1/MRv2 × in-process/MP × CPU/CUDA × capture 的机械矩阵；
- 没有 numerical/state oracle 或 loss/update/version/cleanup 证据的 smoke。

第 39 项完成后，survivor manifest 必须列出每一项的稳定 observable contract、运行环境、
是否需要 GPU/MP，以及它替代了哪些旧测试文件。训练证据必须有
`loss_count > 0`、`update_count > 0`；所有运行证据必须有 `fallback_count == 0`。

## 6. 实施 tranches 与实际减量

### Tranche A：测试清单、oracle 抽取与死路径测试删除

1. 从 public API、CLI、entry point、动态 import、文档和 benchmark 建 capability/reachability
   清单。
2. 为十个 workload 只列 observable behavior，不列旧内部类和调用顺序。
3. 先产出旧 consumer 的逐文件 keep/kill manifest，把独立数学、loader、judge、
   environment 的唯一新 owner 定下来；第 2.2 节的抽取预算以这份 manifest 记账。
4. 从旧测试提取最小 oracle/golden，形成第 5.3 节明确列出的 39 项 survivor。
5. 删除生产路径已死亡的测试（已退役 workload 的旧测试、meta-test、装饰性测试）、
   fixture、mock helper 和只为旧测试存在的生产 seam。
6. 仍在服役的旧 workload（ESamp、FTTT、ReFT、Recirculation、OverRIDE、TTRL、TLM）
   的旧测试保留到其 canonical 切换提交，随旧实现原子删除；不得先删测试让活路径
   裸奔。
7. 增加一个统一 test/LOC ledger，同时跟踪 TEST_COUNT 与待删保留项；不为 workload
   建 wrapper。

验收：keep/kill manifest 完整，survivor 每项均有高价值证据覆盖；死路径测试删除
完毕；没有被改名隐藏的 smoke；本 tranche 生产代码不净增长。`TEST_COUNT <= 50`
自 Tranche E（旧栈删除）完成起强制，此前只要求单调下降且 ledger 可解释每一项的
去留。

### Tranche B：压缩已经切换的 canonical verticals

1. 合并 In-Place-TTT 和 ESamp 的 CPU/CUDA execution/tasks，差异只留在 provider kernel。
2. 删除 FTTT、GTTA、In-Place-TTT 的 application/session/execution 编排，改由 generic
   executor/host transaction 消费 typed workload。
3. 合并 parameter-overlay 的 deployment/session/runner/worker/llm 转发链。
4. 合并 backend/parameter/sampler attachment 的 envelope/digest/codec/validation plumbing，
   但不建立万能 callback bag。
5. Rust 只验证一次；Python 不复制 semantic validation；PhysicalPlan 不重新选择。
6. PTTS CPU/CUDA publication 共用 lifecycle/ledger。

基于当前保留数学/state/host 内容，预计而非强制倒填的结果：

| workload | 当前 | 合并后可信范围 | 预计净删 |
| --- | ---: | ---: | ---: |
| FTTT | 2,634 | 400–700 | 1,900–2,200 |
| GTTA | 1,180 | 300–500 | 680–880 |
| In-Place-TTT | 2,623 | 400–700 | 1,900–2,200 |
| TLM foundation | 231 | 完整实现 600–900 | 新增 370–670，但删除旧 TLM consumer 抵消 |

加上 `canonical_runtime` 合并，本 tranche 预期净删约 5,000–7,000 行。偏离范围时必须说明
是哪一项真实语义或上游 backend seam 造成，而不是通过补 wrapper 达标。

### Tranche C：剩余 workload 切换即删除

1. TLM 只增加 loader/objective/graph，复用 global LOW_RANK_ADD、version pin 和 generic
   executor；online/offline 由 timing edges 表达。
2. ESamp 复用 request PTTS/sampler/generic execution，不迁移旧 init，使用随机初始化。
3. ReFT/Recirculation 只保留 activation/replay 数学与 typed effect。
4. OverRIDE/TTRL 只保留数学、state transition 和 typed output，不迁移 Program lowering。
5. Dummy 只证明最小 graph/sign/plan/execute 路径。

全部十个 workload 的可信终态是 4,000–6,500 行，其中 ESamp/TLM/TTRL 明显比 Dummy
复杂，不能再用平均 200 行的倒填预算。每个 workload 必须在同一提交完成：最小证据、
公共入口切换、旧实现删除、旧测试删除和动态引用清理。净减少按整个 tranche 验收
（第 8 节规则 1），不按单次切换提交验收：实测旧 consumer 中 TLM 仅 491 行、ReFT
321 行、OverRIDE 235 行，其新声明加独立数学可能接近甚至超过旧 consumer 行数，差额
由本 tranche 内其他 workload 的删除抵消；不允许为凑单提交净减少而压缩功能或拆假
提交。

### Tranche D：vLLM adapter 解耦重写（旧栈删除的阻塞项）

实测 `tllm/backends` 20 个文件中 16 个（5,528/6,124 行）import 旧
runtime/workgraph/program。这批 adapter 不脱离旧 authority，第 2.1 节的整块删除就
无法执行。本 tranche 把 adapter 重写为 typed backend quarantine：

1. adapter 不拥有 workload/调度/版本选择/fallback authority；
2. MRv1/MRv2 与旧版本结构共享 probe 和 translation，不为每个版本复制
   runner/session；
3. adapter 只消费 canonical typed seam，对旧栈 import 归零；
4. adapter 实际保留规模根据 probe 后记账，预计 4,000–6,000 行，不预设 1,500 行这种
   无依据上限；这是重写而非保留，工作量按 seam 记账，不允许把旧 runtime 代码搬进
   adapter 充数。

验收：`tllm/backends` 对旧栈 import 为零；MRv1/MRv2 公共入口 smoke 通过。本
tranche 允许因解耦 seam 出现的小幅净增长，但必须点名归因到具体 seam。

### Tranche E：整体删除旧技术栈

最后一个调用者归零（含 Tranche D 的 adapter 解耦）后，直接删除第 2.1 节的 73,578
行 Python 旧 authority 和约 18,700 行 Rust 旧 authority，不在原巨型文件中重构。

独立 tensor 数学、tokenizer、environment、judge、checkpoint 和 numerical oracle 先移动到
唯一新 owner；移动后原文件立即删除，不留 compat import。

验收：生产树中 `Program`、`BaseConsumer`、`DispatchPlan`、`ExecutionPlan`、
`consumer_compat`、旧 WorkGraph 和 residual workload authority 引用为零；
`TEST_COUNT <= 50` 自本 tranche 完成起强制。

### Tranche F：只在精简核心上补齐终态设计

- AER 替换 provisional scheduler，不与旧 WorkGraph 并存；
- typed authoring descriptors 替换 string selector，不保留双 facade；
- closed typed MP codec 替换 type-path/reflection transport；
- profiling surface 替换散落 counters/hooks，不建立 workload callback；
- PageBundle/generational arena 替换 provisional dense_page 家族（13 文件 / 4,062 行，
  按 device×concern 机械切片，含 types-only 浅模块）；替换前 dense_page 冻结禁投，
  不做中间态拆分或精致化重构；
- representation/provider facts 只在 specialization/physical facts，不进入 SemanticGraph。

新增能力必须在同一 tranche 删除其替代对象。若 tranche 净增长，说明 replacement boundary
尚未选对，应重新合并而不是先堆代码。

### Tranche G：最终审计和第二轮压缩

1. 从 public API、wheel、CLI、entry point、dynamic string 和 benchmark 做 reachability 审计。
2. 删除旧内部文档和失效 module path；Git 历史承担归档。
3. 全量运行不超过 50 项的证据集。
4. 生产源码必须进入 22k–30k 的当前可信区间；超过 30k 表示第一轮合并未完成。
5. 对剩余每个 owner 记录必要职责和替代成本，再决定能否从约 26k 继续压向 20k、15k、
   最终接近 10k；没有代码证据时不提前承诺下一档。
6. benchmark 只在 workload 切换门和最终发布门运行。

## 7. 功能与架构验收

### 7.1 唯一执行路径

- 所有 workload 通过同一 typed public factory；
- 每个请求只有 `SemanticGraph -> Rust decision -> PhysicalPlan -> execution`；
- workload 中不存在 lane/priority/fusion/provider/手写顺序；
- active core 不导入任何退役包；
- strict Pyright 0 error/0 warning，动态类型只在 Python–Rust codec；
- 无运行时反射、保护性 fallback、callback bag 或第二调度 authority。

### 7.2 PTTS/AER

- `StateRef(LayoutId, LineageId, VersionId)` 基于 frozen typed layout namespace；
- capacity 在 admission-time entitlement，耗尽 fail closed；
- multi-root publication 原子，inference view immutable；
- request/global version exact pin/release/cleanup，generational identity 阻断 ABA；
- CUDA completion 由 issued receipt/fence 证明；
- AER/scheduler 与小图 exhaustive oracle 一致；
- PhysicalPlan 是 signed decision 的机械投影，worker reconstruction 检测 tamper。

### 7.3 workload observable semantics

| 能力组 | 必须观察到的证据 |
| --- | --- |
| Dummy/simple | graph/sign/plan/execute 完整路径 |
| ESamp | request PTTS、非零 loss/update、sampler pin、cleanup、零 fallback |
| FTTT/GTTA | typed host transaction、receipt、abort/retry 或 queue settlement |
| In-Place-TTT | request dense overlay、版本隔离、两请求 cleanup |
| TLM | global LOW_RANK_ADD、online/offline timing、admission pin、immutable publication |
| ReFT/Recirculation | activation patch/replay typed effect 和生命周期 |
| OverRIDE/TTRL | 数学更新、state version、reward/control、closed-loop settlement |

GPU/MP 只为真正不同的程序–运行时边界保留代表性 smoke，不以未测试组合声称支持。

### 7.4 数量验收

- 生产及支撑源码从 133,210 行降到 22,000–30,000 行；超过 30,000 不通过；
- 实际净删约 103,000–111,000 行，最终数字以 reachability 和功能 manifest 为准；
- `TEST_COUNT <= 50`，且没有脚本化规避；
- 旧 Python/Rust authority 整块删除；
- import-layer/reachability 0 violation；
- strict Pyright 0/0、`cargo check`、wheel/import probe 和代表性 backend smoke 通过；
- 训练证据有非零 loss/update，所有执行证据 fallback 为零。

所有 pytest 使用 `/workspace/pllm/.venv/bin/python` 并报告环境。纯文档 tranche 不运行
行为测试。

## 8. 每个提交的硬规则

1. 除纯文档和 Tranche D 已点名归因的解耦 seam 外，每个 tranche 的生产代码必须净减少；
2. 不允许先加新实现、下次再删旧实现；route switch 与删除原子完成；
3. 不为 `StateStorage.FIXED_BATCHED`、Program、BaseConsumer 或旧 runtime 建 bridge；
4. 不新增 workload-specific scheduler/session/runtime；
5. 不先写 `Any`/`object`/`cast` 再等待审查返工；
6. review 同时检查 authority/import 和装饰性测试；
7. 估算偏差必须归因到可点名的功能、owner 或 backend seam；
8. 不用浅 wrapper、代码搬家或巨型文件压缩数字；
9. benchmark 只在切换门和发布门运行。

## 9. 完成定义

第一轮重构完成必须同时满足：

1. 旧生产 authority 从仓库删除；
2. 十个 workload 的公共功能由唯一 native path 提供；
3. Rust 自动调度、PTTS、typed host service、closed MP 和 backend 边界完整；
4. 生产源码落入当前有证据支持的 22k–30k 区间；
5. 全仓最多 50 项高价值测试；
6. 不靠 compatibility wrapper、猜测、fallback 或重复 DTO 维持功能。

完成第一轮后继续以实际 owner/reachability 证据向 10k 压缩。10k 是方向，不是允许拍脑袋
删除功能或倒填层级预算的理由。

## 10. 可编程推理补充的实施决定（2026-09-06）

纳入 `programmable-inference-next-design.md` 的契约与最小组合路径；完整性能研究、
机制消融和独立变体接入实验留到后续。原 PTTS/AER 研究原稿继续作为其唯一权威，保持未跟踪。

**共用边界结构，而非各子系统各造一套结构后互相搬运：** SemanticGraph 是唯一语义来源；
AER 的派生索引/解引用不复制图的事实；PhysicalPlan 只投影已签定的决策；PTTS 拥有唯一
版本根。执行、replay、MP 与 trace 共用稳定身份和原始 receipt，不维护自己的版本选择、
镜像 graph 或生命周期账本。传输编码只在进程/FFI 边界存在，不能变成另一份可修改 authority。

- 沿现有语义图增加有界条件区域、predicate/carry、梯度截断及取消边界；不新建控制流 runtime。
- KV/RNG、私有写入、版本发布和 host 外部动作显式声明；未支持的 effect 在边界拒绝。
- 首条组合：request fast-weight 更新/发布后，固定 pin 新版本进行最后一层 replay，最多两轮、
  相邻结果相对 L2 阈值 `1e-5`，零更新跳过；当前正常 logits 保留，只有当前 token/选定层 KV
  被覆盖，历史 KV 和 sampler RNG 不变。不跨 replay 反传，不读未来 token。
- 第 40 项为独立 PyTorch 组合 oracle；第 41 项为 Qwen2/MRv2/eager/单 GPU 公共边界。
  不声称该组合支持 prefix caching、speculative decoding 或 CUDA Graph。总目标 41、硬上限 50。
- 同一份有界执行元数据关联 source/operation/region/request/iteration/version/completion；
  默认不保存 prompt/tensor，详细快照显式开启，缓冲溢出丢最旧事件并计数。
- 先完成整个第一轮，再做剩余 owner 的第二轮压缩。最终代码回到主目录；旧改动先完整保存
  可恢复快照，只迁入仍有价值的数学和行为，不恢复旧 runtime。不得以阶段提交冒充全部完成。

### 新采样边界的证据决定（2026-09-06）

OverRIDE 引入既有 ESamp post-filter 证据没有覆盖的 pre-filter addition 与实际 sampled-token
反馈。两者共用原 sampler、packed rows 和 invocation，因此只占一个余量项 42，验证真实
norm hidden、stock draw 与原身份关联；总目标改为 42，硬上限仍为 50。TTRL 复用现有
host/arena 边界，第 29 项覆盖其 CPU/CUDA 数学及完整控制结果，不另建 workload GPU smoke。

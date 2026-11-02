# 当前架构

活跃路径：

```text
带类型的负载声明
    -> SemanticGraph（语义图）
    -> Rust 调度与表示决策
    -> 签名的 PhysicalPlan（物理计划）
    -> Python 任务执行与 PTTS
    -> 带类型的宿主服务或 vLLM 附着
```

## 所有者

| 所有者 | 职责 |
| --- | --- |
| `pllm/contracts` | 带类型的图、状态、布局与附着数据 |
| `rust/pllm_core/src/planning` | 封闭编解码、稠密 ID、签名的调度与表示决策 |
| `pllm/execution/execution.py` | 签名任务/锚点顺序的执行 |
| `pllm/execution/state_execution.py` | 共享张量状态调用、采样借用、取消与结算 |
| `pllm/authoring.py` | 原始输入边界、声明、共享编译/绑定入口；部署引用既有执行与签名附着 |
| `pllm/usage.py` | 冷方法/状态发现与签名绑定进原始图；唯一的创作反射边界 |
| `pllm/execution/host_runtime.py` | 带类型宿主事务结算 |
| `pllm/execution/ptts` | 唯一的 PageBundle arena：版本、COW、pin 与完成；目录选择 lineage 并保留原始实例准入 pin |
| `pllm/workloads` | 负载声明、张量数学、初始化与数学输入/输出 |
| `pllm/kernels` | 显式数学原语；无版本或执行权限 |
| `pllm/backends/vllm` | 共享 MRv1/MRv2 附着、捕获与采样器；参数/激活补丁与 MRv2 当前槽 replay |

`pllm.make_llm` 是公共工厂。不带负载时构造纯 vLLM；带受支持的负载时安装
实例自有的附着。不安装任何残留运行时或进程级引导插件。

负载返回原始声明与 `MathBinding`；不构造执行器、arena 或签名附着。宿主绑定
共享既有宿主与 PTTS 所有者。`ModelConfig` 提供用户容量限制；特化在签名前
推导受支持的内核形状/资源事实。`KernelCapture` 集中管理冷态分配、捕获、
额度检查与清理。内部基准是这个公共工厂与句柄的普通客户端。

全部十一个内置负载使用构造同一张原始图的共享绑定。普通模型方法用
`pllm.Workload`，TLM 也通过这个接口声明因果批输入与模型端口；TTRL 也用
`pllm.Workload`（宿主路径，`pllm.host.make_host` 编译，奖励由 `ttrl_vote` 先算好）；
FTTT 的编辑同样是宿主路径的 `pllm.Workload`，由 `FTTTFeedback` 这个反馈回调 `step`；GTTA 是普通宿主类，其 `run` 作为 `generate_feedback` 的反馈回调，用 Python 字典维护 FP64 EMA，不经框架执行。Output LoRA 用普通加权批量 SGD。旧的 `authoring.Workload` 和
按负载的采样器 `build` 方法已移除。ESamp 用普通具名参数、预测与按状态 loss；
其显式分页/捕获绑定在同一签名图上替换计算，原始 loss/Adam 对与五字段状态
不变。LoRA 用原始加权批次、裸具名因子、reset 与多步 SGD；`SamplerWorkload`
与 `LoRAData` 已移除。加权输入归一化/reset 是一个签名准备任务，其原始结果边
喂给训练；其张量留在同一调用帧，消费后释放。ESamp 原生输入在消费任务内打包，
不跨调用缓存。导出的 loss/梯度在 AER 中存活到完成。临时张量结果只活在原始
StateTaskContext 里，按原始 `(task, output)` 作键。TensorBatch 借用该字典；
TensorFrame 不再缓存结果或并行的最后值。Rust 在 PhysicalTask 中签名每个值的
最后使用释放，包括下游声明的梯度消费者。执行只应用这些释放；导出的值存续到
结果构造与结算。Builder 引用是内部图构造 API；`FunctionMath` 与
`TensorOperation` 把普通数学函数绑定到同一 StateTaskContext。没有额外的任务
游标、参数银行或发布所有者。AdamW 字段与私有更新由共享绑定生成，直接迁移与
发布/replay 保持为单独声明的效果。原始 SemanticTask 的可选 `MathContract`
签名微分需求、行域与归约，Rust 校验其原始输入/状态引用。它不是任意张量数学
或自动 autograd/融合的证明。普通宿主函数绑定到原始事务执行器，保留
快照/恢复与签发的 receipt。负载任务类与手工图已移除；完整映射与受支持的
组合记录在[创作契约](../reference/authoring-contract.md)。

`Result` 输入直接蕴含依赖；显式 `SemanticEdge` 值添加数据flow 尚未表达的
排序约束。Rust 从这张图推导一个前驱索引、时机窗口与执行顺序。精确版本效果
把读取绑定到调用输入或原始生产者任务。Rust 推导 RAW/WAR/WAW 约束并联合
选择实现、区域、锚点、执行域与状态表示。签名传输与资源预算约束放置。歧义
状态访问失败关闭；worker 校验检查固定见证而不重跑优化。这些计划的分布式
执行不被本地执行器与 arena 支持。

Python 在 FFI 边界序列化同样的不可变描述符，只加封闭的输入变体标签。没有
第二套线上 DTO 层级或负载维护的依赖图副本。EngineCore 检查本地锚点、能力与
模式，然后保留完整原始 SchedulerTimeline —— 包括值尺寸、任务驻留、预留与
预算 —— 不重建部分 timeline。

## 运行时边界

资源所有权与进程总内存保证是两回事：

| 存储 | 既有所有者与释放边界 | 容量证据 / 限制 |
| --- | --- | --- |
| 参数、优化器字段、私有版本 | PageBundleArena；精确 pin 与签发完成 receipt 门控复用 | 签名布局/提供者额度；字段级 COW |
| 普通保存张量与打包参数 | 原始调用与唯一梯度终点；导数 detach，训练后清叶，结算/取消清上下文 | 内置 ESamp/LoRA/OverRIDE 信封；自定义数学必须声明自己的成本 |
| 有界导出、稠密块输出与激活替换检查 | 原始 Dummy/迁移/更新任务及其既有结果边 | 有限预算内置公式；按 target 操作数行界；替换内容在任何字段拷贝前检查 |
| Hidden、采样器与 replay 缓冲 | 后端步/采样器/replay 所有者；消费流排序读取；replay 保留所需 KV 回滚副本 | 原始 timeline 预留与签名行/形状界 |
| 行/页映射 | Arena 派生的调用/捕获操作数；无单独版本选择 | 共享映射工作区或固定捕获存储 |
| 捕获输入、scratch 与图池 | KernelCapture；一次性冷准入被中止，replay 结果只在必须存续时拷贝 | Meta 预检、真实输入/scratch 检查与图池核算 |
| 共享输入/completion 输出 | 原始 SharedTensor 作用域与接收完成确认 | 原始 TensorFieldSpec；关闭失败保留未确认输入所有权 |
| 跨进程只读特征页 | 摄取 arena 的原 pin；worker 仅持映射，全部字段 ACK 后回收 | 同一签署计划、CUDA 共享能力与页容量；无第二份 features pool/lineage |
| 冻结因果模型 | HF 模型/句柄；唯一张量存储计数，成功 close 丢弃引擎所有权 | 驻留字节不约束 HF backward 或库 scratch |
| Trace | 有界事件队列；快照编码自持 CPU 字节，不是活张量/autograd 引用 | 每事件至多 32 KiB 负载；截断的跨步张量用有界 `take`，不是整源扁平拷贝 |

宿主事务保留其原始可逆快照直到 commit/abort；队列与算法历史存储属于宿主
服务，不在 GPU arena 预算内。宿主绑定保留原始不可变事实，不在编译期接受
另一套。宿主 `Workload` 的成本由框架按最大形状测量；有界 TTRL 提供
head/MSE/Adam 成本与 hidden 行预留，带准备前行准入与消费流寿命。文本与
Python 元数据在该张量预算之外。FTTT 的编辑同理按最大形状测量（行数与
`target_shape`），包括交给结果的更新后参数拷贝。执行器在调用状态构造函数前
准入该计划。GTTA 不断增长的 Python 历史与外部服务分配仍没有完整堆成本声明。
`handle.diagnostics().state_usage` 是从原始 arena 根、字段槽、私有写与 receipt
直接推导的冻结快照；它不保留这些所有者。逻辑字节计入每个保留/私有版本，
分配的载荷字节对 COW 共享字段页只计一次。预留填充包含所有容量对齐/余量。
空闲、最新根、仅私有与仅退役字段页是互斥的所有权类别，不是 GPU 活动的度量；
最新根类别优先于私有，再优先于退役引用。被 pin 的版本包括内部写/发布持有与
待释放。待收 receipt 包括未收集的已结算发布，不只是未完成 GPU 工作。容量
失败计数覆盖被拒的准入，不是迁移失败或非法输入。`arena.usage()` 本身不收集、
不同步、不选版本；既有执行诊断可能先做其常规的非阻塞收集。无界自定义
autograd、HF backward、外部库 scratch、vLLM 模型/KV 与分配器预留不被声明的
AER 峰值证明。这些是受支持契约的显式容量限制，不是推断出的零字节资源。

ESamp 与 In-Place 在 CPU/CUDA 上使用同一个 StateExecution。其调用保留原始
物理计划、pin、写与发布；结算不重放训练。In-Place 数学消费原始捕获的操作数
记录。vLLM 工厂与会话消费 BackendWorkload 而不导入任何负载。稠密应用在原始
调用 pin 下借用 arena 页，使用原始签名字段绑定与打包行，没有第二个 overlay
计划或拷贝的张量银行。借用方不得修改已发布视图，不得在 pin 释放后使用；
自持外部导出仍用快照。签名附着把 target、算子、字段 ID 与缩放绑在一起。
重复的七段 overlay 描述符已删除；事件推进消费原始物理计划。不支持的组合
hidden/采样器/overlay 程序失败关闭。已验证的更新/发布/replay 组合使用同一
签名图与原始状态 pin，不是第二个运行时。

PageBundle/代次状态包含捕获稳定的间接字段选择。冷图池与固定 scratch 是驻留
TaskResourceFact 字节，区别于瞬态工作区。读/私有写用原始 PagePin/PageWrite；
切换流时先排序前一个 scratch 用户再重绑。图不发布、不选版本、不回收页。
ESamp 在原始任务中绑定 Rust 选出的定形 loss/AdamW 图，包括公开 eager 模型
vLLM 执行；这不是全模型捕获或自动区域融合。TLM 的训练与公开 vLLM overlay
对 q/v 因子与 AdamW 字段使用同一 StateExecution 与实例 arena，在线/离线时机
在原始图中。

因果训练借用已加载的 vLLM 基座权重与 RoPE 缓存，训练和推理共享同一组 q/v
因子。支持的架构及精确类准入规则见[支持的模型与限制](../../README.md#支持的模型与限制)。

原始请求准入 pin 在
全局发布后存活；推理与训练从不共享可变的 latest 缓存。`CausalTrainBatch` 被
独立与公开输入共享，不转换成另一种负载批次。公开输入每次 generate 消费一次；
在线在 token 提交后发布，离线在新准入前发布。同 GPU IPC 的原始页授予仅支持
精确版本的只读借用，由原 pin 等待全部远程读完成后回收；写入与原子发布仍在
原 arena 内进行。普通输入/独立输出使用 SharedTensor 与既有完成确认。CPU
页授权与远程私有写授权已移除。分布式计划执行/提交仍在当前本地执行器
之外。后端局部 overlay 存储与旧的后端/回调 replay 契约已删除。条件 replay 是
原始签名 SemanticTask；其新版本 pin 与释放属于同一 StateTaskContext 与
arena。后端只快照选中的解码器输入/当前 KV 槽并执行原生层。

E1 视频会话由摄取进程和一个 spawn 的 in-process vLLM 子进程组成，使用同一
完整 PlanBundle，不裁剪或另签 worker 计划。arena 只物化自身拥有的资源；
MathBinding 显式验证本地与 borrowed 集合互斥且覆盖所需资源，再为本地集合建立
目录。features 完全属于摄取方，inject 完全属于 worker；接收方的合法性来自
原 receive_field 校验及本次 grant 对应关系，不来自本地 features admission。

普通 Parameter reader 按资源名将 features 解析到 BorrowedStateRoot 关联的
ReadonlyTensorView；原映射不进入本地 roots、pin.refs 或写选择。完整借用输入
绑定到一次步，encode 的独立 hidden 经原 TaskResult/field_writes/publish 写入
本地 inject，再由原 Activation ADD 消费。接收 stream 完成、别名清除后才发
ACK，原 owner 的 release receipt 决定旧槽位回收；输出不得以别名携带特征页
逃逸。无 borrowed resources 的负载继续使用原生命周期。

示例以签署容量约束积压，一 chunk 一新请求、一次 prefill；帧特征提取属于输入
生产者，不冒充 worker 任务图中的签署计算。只读授予是同一可信程序内的协议，
不是身份认证、动态血缘认证或 OS 写保护。缺失完成证明会保留资源并停止实例，
不回退本地空页、不在线恢复。

宿主与状态执行共享一个任务游标与有界执行 trace。`execution.trace.snapshot()`
或公开模型句柄的 `trace_snapshot()` 返回至多 256 个元数据事件，带显式溢出
计数与原始调度签名。任务 ID 通过原始图与物理区域边界解析；请求行、
StateRef 与签发的操作/pin/释放 ID 串联 replay 与结算。元数据事件不保留
prompt、张量或活 receipt。`snapshot.host_task_times()` 只报告成功完成的调用与
宿主耗时纳秒。它不加 CUDA 同步；已结算调用不证明其延迟设备释放已完成。
原始 trace 或模型句柄上的 `profile_tasks(task_ids)` 让选中任务进入 CUDA 事件
计时；空元组关闭后续计时。快照暴露待完成计时不等待；
`device_task_times()` 报告按任务/设备的已完成流区间，包括流等待与争用，不是
孤立内核成本。计时事件共享元数据队列的上界与最旧丢弃策略。性能分析增加事件
启动，未插桩的性能数字必须关闭它。Schema 5 保留原始打包行，不复制请求 ID 或
decode 序号。Dummy 显式启用的快照在同一有界队列中存至多 32 KiB 原始张量
字节；它们可能同步，并通过 `event.tensor()` 重建独立 CPU 值。
`execution.trace.configure()` 或 `handle.select_trace()` 上的 `TraceSelection` 选择
原始 TaskId、请求身份、迭代与名字。张量与梯度捕获分别独立开启。数学用真实
值或算出的梯度调用 `trace.observe`；trace 收集不做签名检查或自动梯度发现。
ESamp 暴露源/目标/loss/参数梯度，Output LoRA 暴露最终 SGD 迭代的
loss/A/B 梯度，两种 replay 都暴露其进位。过滤器不会通过共享值泄露未选中的
请求。大值保留扁平前缀，`source_shape` 记录原始维度；上限每事件 32 KiB，
队列相同。`kernel_calls` 用 Kineto 对至多 16 个选中任务调用做性能分析，
保留真实 CUDA 内核名、设备与时长作为 KERNEL 事件。它选整个任务，不是按请求
的内核成本，且可能同步或扰动计时。Kineto 临时缓冲跨一个任务；只有持久结果
受 trace 事件上界约束。CUDA 捕获内的快照与性能分析启动被拒绝；捕获任务
replay 之后的显式快照受支持。这些观测不蕴含确定性的全模型 replay 或未插桩
性能声称。

维护中的支持矩阵在 [README](../../README.md)。

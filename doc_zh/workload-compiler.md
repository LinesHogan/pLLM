# 附加载荷编译器

tLLM 将 vLLM 视为外部所有、不可移动的主负载。backend 只导出真实可观察的
生命周期 anchor、固定 region、资源 reservation 和可见 value；它不会导出一份
可以重排的完整模型图。

## 用户前端

`tllm.workload` 提供 `@workload`、唯一节点装饰器 `@task`、`State`、typed value
引用、state effect、timing window，以及可选的 affine domain/access。不可捕获的
CPU、I/O 或 backend 实现通过同一个 `@task` 的 `ExternImplementation` 候选表达，
不再形成第二套程序语义。

`State` 显式声明 scope、request identity、参数所有权、optimizer 和版本转换。
REQUEST state 必须声明 `request_key`。没有 optimizer 或显式版本转换的
transition 会被拒绝。recurrence 默认为 exact；Jacobi 必须显式选择
`ApproximationPolicy.JACOBI`，并会改变 workload semantic key。

`declare_workload(instance)` 返回 `WorkloadDeclaration`。其中 `ir` 只含纯数据，
具有确定性的 JSON 和 SHA-256 semantic key；runtime callable 保留在独立 executor
表中，不参与序列化。

## 中端契约

- `MainWorkloadTimeline`：pinned backend region、可观察 anchor、graph boundary、
  capability、reservation 和启动期静态 headroom。
- `ScheduleGraph`：task、typed input/output port、value/version、依赖、effect、
  timing、实现候选和静态内存。
- `ScheduleDecision` schema v2：细粒度执行顺序和 schedule tree，加上权威的
  `ExecutionRegion`、跨 Region materialization 及 lifetime、MemoryPlan、跨 Region
  fence、成本分解、解释、artifact key 和签名。

Rust 从细粒度 task port 推导每个 Region 的 typed input/output descriptor 和有序
task invocation。每个 invocation 参数和返回值都引用 opaque、稳定的 value-binding ID；
Region 边界 binding 覆盖 provided value、typed hidden locator、内建 step ID、sink 和
跨 Region materialization。全部 descriptor 都进入 artifact key 和 decision signature，
并由 Rust validator 独立重算。input/output/effect/implementation/state/layer 与 Region
binding kind 在 Rust 内均为封闭 serde enum；未知变体在 JSON decode 时直接拒绝，planner
使用穷尽 match。`step_id()` 是独立 DSL/IR 变体；`provided("step_id")` 仍只是普通的同名
provider，不能根据拼写获得内建语义。数值 hidden locator 只在 Rust lowering 时解析一次，
并以签名后的 `layer_index` 数值交给 runtime。

`RegionCaptureMode` 同样是已签名 Region record 中的 Rust 闭合 enum。它是 lowering 能看到的
唯一 capture 契约：`eager_only` 及所有 capture/replay-capable variant 都由 Rust 选择并做
capability 检查。Python 不得从 task class、stream 名或逐 task capture partition 推断 graph policy。

Python 执行路径不读取 `TaskIR`，不从名称推断 port kind，不解析 locator 字符串，也不重建
Region 内或 materialized result edge；它只机械分派 Rust 签名的 Region enum 与 opaque
binding ID。剩余 Python 检查只验证 API/wire 对象、callable 返回 ABI，以及运行时 Tensor
shape/dtype/device 资源，不选择调度语义。graph replay 接口在 `RuntimeGraph` 资源注册时绑定
一次，不在每次 replay 时动态推断。

`compile_schedule(graph, timeline, options)` 与 `validate_schedule(...)` 只通过 Rust
FFI 传递 primitive JSON。所有新 decision 必须使用 schema v2 并包含 Region；schema
v1 只为读取旧 golden 保留，执行校验会拒绝它。内存预算是硬约束，取非零用户预算与
backend 静态 headroom 的较小值。capability 缺失、deadline 非法、依赖成环、Region
覆盖不完整、artifact 不匹配或内存超限都会 fail closed。

Granularity Planner 对细粒度 task 图做分析，但为执行产出粗粒度 Region。它从拓扑序上的最大
确定性 Region 开始，只在 anchor window 无交集、
effect/version 或 typed extern boundary 必须可观察、graph safety 不一致，或硬内存约束
要求显式 serial fallback 时拆分。Region 内消费的 value 保持为图内边；跨 Region value
才产生显式 materialization 和 lifetime，依赖 fence 也只存在于 Region 之间。结构成本
记录 materialization bytes、capture boundary、peak live memory、launch group 和跨 Region
fence。中端不预测 kernel 精确时序、不分配 SM，也不查询运行时空闲显存。

Rust 构建固定 `isl-rs==0.3.1`，使用其随包构建并静态链接的 isl 0.27；schedule
和 flow API 均有 Rust 测试。整个连通 region 的 domain/access 都能表示为
Presburger relation 时，现在会 lower 成 isl 的 validity、coincidence 和 proximity
约束；生成的 schedule 与 dependence relation 保存在 typed affine region 中，并由
独立 validator 再次检查。其他节点继续由 DAG scheduler 处理。

## Region lowering

`PhysicalFlowPlan` 现在每个 `ExecutionRegion` 对应一个物理节点，而不是每个逻辑 task
对应一个节点。Python lowering 只能绑定签名过的 Region artifact、executor key、anchor
和跨 Region fence，不能再次拆分 Region 或选择执行顺序。Region artifact identity 纳入
Region IR、shape specialization、包含 approximation 的 workload key、backend timeline、
compile/capture policy，以及静态 Python/torch/CUDA/device fingerprint。

进程内 artifact materializer 会把普通 PyTorch task 组合成一个 Region callable，并只把
这个完整 callable 交给一次 `torch.compile`。Region 内消费的 result 保持为局部 Python/FX
value，不作为 materialized output 返回。不可捕获的 typed extern 是显式 singleton Region，
并绕过 `torch.compile`；CUDA Graph capture 和 replay 同样以 Region 为单位。若 runtime
无法物化选定 artifact 或 capability，必须拒绝计划，不能静默退回逐 task eager。

eager runtime 可以在各自签名过的 anchor 提交不同 Region，只把 typed 跨 Region 输出
保留到声明的 consumer 执行，并且只为物理跨 Region fence 创建 runtime event。
进程本地对象通过显式 provider/sink registry 绑定；缺 provider、sink、artifact、
effect/state binding 或非 eager capture artifact 时，cold install 会直接拒绝计划。

state 执行要求每个 Rust typed state resource 都有显式的进程本地
`StateVersionBinding`，且其 descriptor 必须与签名过的 `ScheduleState` 完全相同。
state-read Region 必须保持所有 active slot 的 observed version 不变；state
write/transition/version-publish 必须按签名次数推进版本。缺失、多余、非单调或 descriptor
不匹配都会 fail closed。

## 迁移状态

旧 Program 声明已经有只读 adapter，可投影到新的 dependence/effect IR。现有
`BaseConsumer` 和 Program runtime 仍负责尚未进入 physical takeover 的 workload。
canonical 启动参数为 `make_llm(..., workloads=(...))`；迁移期可以直接传入旧
consumer。新的 workload declaration 在完整 native executor set 可用前必须提供
`legacy_consumer` adapter；没有 adapter 的纯声明只有在显式绑定全部进程本地 provider、
sink 和 state version 后，才能使用进程内 eager Region runtime。

MRv1/MRv2 worker 会上报纯数据 timeline 和一次初始化完成后的静态显存快照。实例作用域
的父进程 coordinator 负责编译并签名 decision（含 typed dependency fence 和主负载
reservation），再通过 string-method RPC 广播；worker 独立验证签名，并把 immutable
Region decision 机械 lower 成 `PhysicalFlowPlan` descriptor。

native MP 执行要求显式提供 `WorkloadWorkerSpec`。控制面只传 importable factory 名称和
纯 JSON config；factory 在 worker 内重建 declaration 和进程本地 callback。worker 完成
校验与物化后会上报实际、有序的 `(region_id, artifact_key)` 集合；父进程只有在它与签名
decision 完全相等时才接受 epoch。Python callable 始终不会被序列化。

该控制面设计已在 EngineCore MP 的 MRv1/MRv2 native ESamp exact eager Region 上完成正确性
验证，覆盖 worker reconstruction、signature/artifact 一致性、训练、state-version 推进和 sampler
intervention；这只是正确性结果，不是性能测量。

`NativeWorkloadResources` 是 native declaration 的显式进程本地资源契约。它拥有
declaration，以及命名 input provider、output sink、state-version snapshot、cold
specialization callback、sampler provider、统计与 enablement callback、residual-path
binding、specialization key 和可选的 worker reconstruction recipe。这些对象刻意只存在于
本进程：跨 MP 控制面只有 `WorkloadWorkerSpec` 的 importable factory 与纯 JSON config。
worker reconstruction 会按父进程签名的 runtime specialization envelope 做 cold
specialize，重新声明并比较完整 schedule graph，然后物化并上报有序的 Region artifact 集合。
specialization、graph、binding 或 artifact 任一不匹配都会拒绝 epoch。

`ESampConsumerConfig(implementation="region")` 是独立的 native、已由 CPU 测试覆盖的
ESamp declaration，而不是 legacy consumer 的包装。它拥有 replica state 及其 optimizer，
对 distiller 做 cold specialize，并有两个签名 Region：`predict_and_loss` 读取签名 hidden
input 并发布 guidance；显式的 `update` extern Region 消费其唯一的跨 Region
materialization（可微 loss）并推进 state version。sampler bridge 只消费同一步、metadata
匹配的 guidance，不会重算 Region output。source/target binding 使用签名后的数值 layer
index，其中 target layer 可为 `-1`，并通过显式 residual-path 契约解析。该模式拒绝
per-request/model-bank state、compact capture lane、row selection 和非
`post_filter_exact` sampler backend 等 legacy-only feature。native ESamp Region 不能与
legacy ESamp authority 共存。当前迁移期该隔离是 runtime-wide 的：只要 runtime 含有任一
native Region declaration，所有 legacy dispatch consumer（即使表面上是无关的
GTTA/TTRL sidecar）都会被拒绝。legacy consumer 尚未声明并签名其 anchor、residual
path、state resource 与 sampler hook 的 ownership，runtime 无法证明两套 execution
authority 不相交。只有 native 与 legacy contract 都引入显式且 decision-signed 的
authority/ownership key 后，才能将此限制缩小。

Region CUDA Graph 的 materialization/replay 基础已实现，适用于 address-stable CUDA
Tensor Region 与静态 specialization。它会显式拒绝带 state effect 的 Region，因此有状态的
ESamp `predict_and_loss` / `update` 序列当前明确为 exact eager、non-capture。特别是
model-backed native piecewise CUDA Graph 路径仍未验证且未完成。这些是 capability boundary，
不是 throughput 结论。

Recirculation 另有 native Region declaration，但只适用于显式发布的、有限的 CPU
`RecirculationExecutionRequest`。它拥有 exact serial 或已证明的 ordered-row wavefront
recurrence，并将结果写入显式 sink；由于该 request 完全由 Region 拥有，所以没有 legacy
adapter。当前 backend timeline 没有 prefill row/model-Region binding；若把此 declaration
当成隐式的真实 vLLM prefill feature 使用，会在缺少 request provider 的边界 fail closed。
不得将它描述为已接管 vLLM prefill。

另外，保留的 eager vLLM model-region adapter 已在 MRv1/MRv2 上完成只验证正确性的 serial
oracle 对照：两个 runner 选择相同 token，BF16 logit 最大差为 `0.15625`，最小 cosine similarity
为 `0.99995947`。这证明 adapter 的 wavefront 数值路径，但不表示它已经成为上文所述的 native
prefill 接管。

除 native ESamp、显式有限 request Recirculation 和 Dummy declaration 以外，其余
workload 仍保留完整的 legacy consumer 或 sidecar adapter。不存在部分权威的 native
execution：只有在完整 resource、binding、specialization 和 artifact 契约都可用时，workload
才进入 physical takeover；否则现有 adapter 继续保持 authority。

独立 Dummy workload 现在是完整的 `NativeWorkload`。它的 typed singleton extern
Region 使用通用 Region executor，并通过显式 `NativeWorkloadResources` 绑定 layer-0
residual path、CPU-export worker、synchronization callback、specialization identity 和 worker
reconstruction。它不再注册或查询 legacy `DummyConsumer`，旧的 Dummy 专用物化分支已删除。
缺少 binding、`eager_only` capability 或已解析 residual resource 都会 fail closed。
`feedback_interval > 0` 仍是 legacy-only 的逐步 drain 特性，native declaration 会拒绝；
非默认 legacy `capture_policy` 也会被拒绝，而不是被静默忽略。

## 当前验证边界

在 `torch==2.13.0+cu129` 与 `vllm==0.27.1` 上，短 GPU 正确性 smoke 已覆盖通用 native Dummy
与 native ESamp 的 exact eager in-process MRv1/MRv2。最新 ESamp：`loss_count=6`、
`update_count=6`、`state_version=6`、`fallback_count=0`。native ESamp exact eager Region 还通过
两种 runner 的 EngineCore MP 正确性：loss、update、state-version、intervention 均非零，
`fallback_count=0`、`replay_count=0`。这些只证明 execution 与 state/sampler activity，不是性能
结论。model-backed native piecewise CUDA Graph、针对真实 vLLM prefill 的 native Recirculation
接管，以及 shape-miss specialization epoch switching 仍未验证或未完成。本文不作吞吐宣称。

通用 multi-task/value eager Region 执行、签名过的内部 task invocation、typed
hidden/provided/sink binding、state-version 审计，以及带精确 artifact 回报的显式 worker
重建均已实现，并有 CPU 测试覆盖。完整 GPU 正确性矩阵和 ESamp paired throughput gate
仍未宣称完成。

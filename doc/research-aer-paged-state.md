# AER 与 Paged Test-Time State：tLLM 的数据结构与算法研究方向

## 文档状态

本文是一份研究与架构提案，不是已经验证的性能结论，也不代表实现接口已经冻结。

本文讨论两个相互独立但可以组合的方向：

1. **Anchor-Window Effect-Aware Regionization（AER）**：自动决定 workload task 如何组成 Region，以及 Region 应放置在哪个 inference anchor。
2. **Paged Test-Time State（PTTS）**：虚拟化、版本化并批处理 request-scoped mutable tensor state。

二者都必须遵守现有 Region authority 原则：workload 只声明数据流、effect、state 与 timing constraint；通用 Rust scheduler 负责调度求解；Python runtime 只机械执行 signed physical contract。任何设计都不得加入 workload 名称、专属 lane、priority、fusion group 或手写执行顺序。

## 1. 设计出发点

tLLM 当前需要同时表达：

- tensor dataflow；
- mutable state 与 optimizer transition；
- request/version lifecycle；
- host service；
- inference timeline 中的合法执行窗口；
- CUDA stream、capture、fence 与地址稳定性；
- 多 workload composition；
- Python–Rust 与 backend 边界上的可验证契约。

这些问题不适合由一种万能数据结构解决。本文采用以下组合哲学：

- **Parse, do not repeatedly validate**：开放字符串和外部动态对象只在边界解析一次。
- **Make illegal states unrepresentable**：进入 canonical core 后，非法状态应尽可能无法构造。
- **Single authority per fact**：每个语义事实只有一个 owner；序列化结构只是 projection。
- **Progressive refinement**：调度和物理 lowering 只能增加信息，不能重新解释 workload 语义。
- **Proof-carrying schedule**：scheduler 产出结果和必要 witness，runtime 不重新推断 scheduler 意图。
- **Operation-driven data structures**：先确定合法状态、操作集合与操作模式，再选择高级数据结构。

## 2. 更简单的 canonical core

### 2.1 一份 semantic authority

不建议复制出多份相互独立的 `DeclaredGraph`、`ScheduleGraph`、`DecisionGraph` 和 `PhysicalGraph`。更简单的形态是：

```text
SemanticGraph        唯一语义 authority
├── GraphIndex       可重建的派生索引
├── ScheduleSolution 调度求解结果与 witness
└── PhysicalPlan     可执行物理投影
```

示意结构：

```rust
struct SemanticGraph {
    tasks: Vec<Task>,
    values: Vec<Value>,
    states: Vec<State>,
    anchors: Vec<Anchor>,
    edges: Vec<Edge>,
}

struct ScheduleSolution {
    execution_order: Vec<TaskId>,
    regions: Vec<RegionSolution>,
    placements: Vec<Placement>,
    fences: Vec<Fence>,
    witnesses: Vec<ScheduleWitness>,
}

struct ScheduledProgram {
    semantic: Arc<SemanticGraph>,
    solution: ScheduleSolution,
}
```

`ScheduleSolution` 不复制 task、state 或 effect 语义，只通过强类型 ID 引用它们。

### 2.2 强类型稠密 ID

Python 名称只在 Python–Rust 边界进入 interner：

```text
"distiller"    -> StateId(3)
"loss"         -> ValueId(9)
"model.execute" -> AnchorId(2)
```

core 内部使用：

```rust
struct TaskId(u32);
struct ValueId(u32);
struct StateId(u32);
struct AnchorId(u16);
struct SelectionId(u32);
struct BindingId(u32);
```

immutable compiler graph 应优先使用 `Vec + typed index`，而不是 generational slot map。Generational ID 只用于会回收和复用的在线资源。

### 2.3 派生索引不是 authority

```rust
struct GraphIndex {
    successors: CsrGraph<TaskId>,
    predecessors: CsrGraph<TaskId>,
    value_users: CsrGraph<ValueId, TaskId>,
    effect_reachability: Vec<BitSet>,
}
```

`GraphIndex` 可以从 `SemanticGraph` 重建，不参与 semantic hash。CSR、dense vector 和 bitset 可以替换大量 `BTreeSet<String>` 与重复字符串 lookup。

### 2.4 RegionSummary

候选 Region 不应在每次 merge 时重新扫描全部 task。应维护闭合、可增量组合的 summary：

```rust
struct RegionSummary {
    tasks: TaskSet,
    window: AnchorInterval,
    implementation: ImplementationClass,
    effects: EffectSummary,
    boundary_values: ValueCutSummary,
    capture: CaptureFacts,
    approximation: ApproximationPolicy,
}
```

合并接口：

```rust
fn try_merge(
    left: &RegionSummary,
    right: &RegionSummary,
    graph: &SemanticGraph,
    index: &GraphIndex,
) -> Option<RegionSummary>;
```

合并只执行：

- timing window intersection；
- implementation class unify；
- typed effect legality join；
- boundary hyperedge cut 增量更新；
- capture/storage evidence join；
- approximation policy consistency。

新增 state/effect capability 时，应扩展闭合的 `EffectSummary` 代数，而不是让多个 planner 函数识别 workload 或 task 名称。

### 2.5 mutable runtime state

在线 mutable state 使用独立的状态机：

```text
Free(generation)
-> Admitted(owner, aliases)
-> InFlight(leases, receipts, fences)
-> Retiring(cleanup receipt)
-> Free(generation + 1)
```

对应结构：

```text
GenerationalArena
SlotRef(slot_id, generation)
Lease
IssuedReceipt
CompletionFence
```

compiler ID 与 runtime slot ID 不应使用同一数据结构或共享生命周期。

## 3. AER：Anchor-Window Effect-Aware Regionization

### 3.1 当前问题

当前 Rust scheduler 从 singleton Region 出发，枚举合法的 pair merge，通过 visited partition 去重，并设置了 100,000 个 partition state 的硬上限。每个候选 partition 会重新计算：

- Region 拓扑顺序；
- anchor slot；
- materialization bytes；
- fence 与 structural critical path；
- peak memory。

该搜索在最坏情况下接近 set-partition Bell number。当前的 “exact” 是在既定 latest-slot placement 规则下对 Region partition 做 exact search，并不是 partition 与 placement 的联合全局最优。

latest placement 在存在随 anchor 变化的 backend reservation 时，不一定满足 memory feasibility 的完备性。例如：

```text
Region legal slots: {0, 1}
Region memory: 50
backend reservation at slot 0: 0
backend reservation at slot 1: 100
memory budget: 100
```

latest placement 选择 slot 1 会超预算，而 slot 0 实际可行。

### 3.2 问题定义

输入：

- task DAG `G = (V, E)`；
- 每个 task 的合法 anchor interval `[release(v), deadline(v))`；
- value hyperedge、producer、consumer 与 tensor size；
- typed state effect 与 effect order；
- implementation、capture、storage capability；
- backend 随 anchor 变化的 memory/resource envelope。

输出：

- task 到 Region 的 partition；
- 每个 Region 的 placement anchor；
- cross-Region materialization；
- fence、capture、stream 与 host-submission witness。

硬约束：

- Region 成员具有非空公共 timing window；
- state effect 可以安全 internalize；
- Region DAG 和 placement 满足依赖；
- 每个 anchor 的 memory/resource 不超过 backend budget；
- request/version/capture legality 保持 fail closed。

目标可以保留 lexicographic score，也可以在研究阶段分析 weighted objective：

```text
materialization bytes
launch/capture boundaries
peak memory
Region count
fence count
critical-path extension
```

### 3.3 AER 与 ISL/多面体优化的关系

ISL 处理的是 affine iteration domain、affine access relation、instance-level dependence 与 loop schedule。AER 处理的是 task/Region 级的离散 partition 与 inference anchor placement。

```text
Affine micro-schedule
  object: loop/statement instances
  authority: ISL

Workload macro-schedule
  object: tasks and Regions
  authority: AER

Physical execution
  object: streams, buffers, fences, artifacts
  authority: physical planner/backend
```

以下 AER 事实不能自然地由当前 ISL integration 直接解决：

- opaque PyTorch、extern、optimizer 与 host-service task；
- state read/transition/publication；
- request selection 和 version pin；
- Region capture/address-stability evidence；
- cross-Region hyperedge materialization cost；
- backend event 与随时间变化的 reservation；
- Region membership 这一离散 partition 决策。

未来可以让 ISL 为 affine component 提供合法 schedule/cost summary 或 implementation candidate，再由 AER 选择；两层不能同时拥有同一物理顺序 authority。

AER 的研究 novelty 不能表述为一般的 task fusion。必须与 polyhedral fusion、operator fusion、memory-constrained graph partition 等已有工作比较。可能的新组合是：

```text
anchored intervals
+ typed mutable effects
+ background resource envelope
+ proof-carrying Region schedule
```

### 3.4 interval-window path decomposition

每个 task timing window 是一维 interval。定义 anchor `a` 上的 active bag：

```text
B[a] = {v | release(v) <= a < deadline(v)}
```

一维 interval 具有 Helly 性质：一组 task 具有公共合法 anchor，当且仅当它们的 intervals 具有公共交点。因此任何合法 Region 必然完整出现在至少一个 active bag 中。

这为 timeline sweep/frontier DP 提供了自然的 path decomposition。

DP state 可包含：

- 当前 active tasks 的 canonical partition；
- 每个 open Region 的 `RegionSummary`；
- 穿过 frontier 的 live value hyperedge；
- resource envelope；
- Pareto cost vector。

当一个 task 离开 active bag，且其 Region 不再可能加入未来 task 时，该 Region 可以 finalize。

目标复杂度形式：

```text
O(A * Bell(w) * poly(w, E))
```

其中：

- `A` 是 anchor 数；
- `w` 是最大 active-window/frontier width。

这是一个待严格证明的 FPT 目标，不是本文已经完成的结果。effect、hyperedge 和 placement state 可能引入额外 frontier 参数。

### 3.5 候选算法数据结构

- canonical set-partition encoding：消除 partition label 同构；
- bitset reachability：effect/dependency legality；
- rollback DSU：搜索分支中的 Region merge 与 aggregate rollback；
- incremental hyperedge cut counter：维护 materialization cost；
- Pareto frontier：删除 cost/resource 均被支配的 state；
- `SmallVec<[u64; N]>` memory envelope：anchor 很少时优先；
- persistent range-add/max tree：anchor 多且 placement 分支很多时使用。

高级数据结构应隐藏在 `RegionPartitionState` 与 `ResourceEnvelope` 后面，不得泄漏到 workload declaration 或 signed public contract。

### 3.6 correctness oracle

在优化算法前应先建立独立 brute-force oracle：

- 小规模 task graph；
- 枚举所有合法 partition；
- 枚举每个 Region 的所有合法 anchor placement；
- 精确计算 hyperedge cut、fence、critical path 与 peak memory；
- 与当前 search、greedy、ILP/CP-SAT 和 frontier DP 比较。

该 oracle 还可用于自动寻找 latest-placement 不完备或局部 merge 次优的 counterexample。

### 3.7 论文可行性门槛

当前单个 native workload 大约只有 1–6 个显式 task。AER 只有在以下真实场景出现时才值得成为独立算法贡献：

- TLM、model-bank 或 distributed lowering 产生几十个以上 task；
- 多 workload composition 形成显著 frontier width；
- 当前 100,000-state search 在真实图上触顶或编译延迟不可接受；
- 联合 placement 找到当前 latest-placement 无法找到的可行/更优方案；
- 更优 Regionization 转化为实际 materialization、launch、memory 或 latency 收益。

若真实 graph 长期小于约十几个 task，frontier DP 可能是过度设计，应保留 brute-force/branch-and-bound 并将研究重点转向 state paging。

## 4. Paged Test-Time State

### 4.1 动机：test-time state 与普通训练 state 不同

tLLM 的核心场景不是只执行一次无状态 side computation。许多 test-time workload 会为正在生成的 request 维护持续演化的 tensor state：

```text
request admission
-> predict with version V
-> observe token/feedback
-> train/update private state
-> publish version V+1
-> later decode pins an exact version
-> request finish
-> wait for GPU work and reclaim state
```

这种 state 同时具有三个时间尺度：

- **cold layout lifetime**：模型结构、rank、layer set、dtype 和 optimizer layout 在 specialize/epoch 时冻结；
- **step/version lifetime**：每次在线 update 产生新版本或推进 mutable state；
- **request lifetime**：state、alias、sampler pin 和异步 fence 随 request admission/finish 创建和回收。

普通 offline training 通常只有少量长生命周期 module/optimizer；test-time workload 则可能同时拥有大量短生命周期、request-isolated、频繁更新的 state。直接为每个 request 创建完整 `torch.nn.Module + Optimizer` 会产生结构性问题。

#### 4.1.1 isolation 与 batching 冲突

逻辑语义要求不同 request 拥有独立 state：

```text
request A -> state A
request B -> state B
```

但独立 Python Module/optimizer 会让每个 request 的 parameter、gradient 和 optimizer moments 位于不同 allocation/address。即使数学结构完全相同，runtime 也很难把它们组成一个规则的 batched kernel：

```text
N request
-> N Python objects
-> N optimizer instances
-> N sets of small allocations
-> N launch sequences
```

共享一个 global model 可以恢复 batching，却破坏 request isolation 和 version pinning；workload-specific model bank 可以恢复部分效率，却把 state authority 和 slot policy 固定在某个 workload 内。

PTTS 的动机之一是同时保留：

```text
logical isolation
+ physical packing
+ generic batching
```

#### 4.1.2 mutable training 与 immutable inference view 冲突

在线训练会修改 parameter/optimizer state，但正在 decode 或 sampling 的 request 必须看到一个一致的版本。若直接原地更新 Module，可能出现：

- prediction 读取一半旧、一半新 parameter；
- sampler 在一个 request 内跨 token 漂移版本；
- async optimizer 尚未完成，下一步已经读取 state；
- request finish 提前回收仍被 stream 使用的 tensor；
- slot 被新 generation 复用后，旧 completion/receipt 错误作用到新 request。

因此 test-time state 需要的不只是 allocator，还包括：

```text
version root
immutable publication
request pin
generation
lease
issued receipt
CUDA completion fence
```

#### 4.1.3 dynamic lifetime 与 CUDA Graph 地址稳定冲突

request 持续进入和离开，天然要求动态 admission/reclaim；CUDA Graph replay 又要求 parameter、optimizer state、gradient 和 input buffer 的地址保持稳定。

若每个 request 动态创建或销毁 tensor：

- allocation 地址不稳定；
- optimizer moments 可能在第一次 step 才惰性创建；
- capture artifact 无法跨 request/step 复用；
- allocator fragmentation 和峰值 reserved memory 随 workload 历史变化；
- capture 失败后容易产生错误的 eager/legacy runtime fallback。

固定物理 page pool 与逻辑 page table 可以把动态 request lifetime 和稳定 GPU 地址分离：request 改变的是 page ownership/version mapping，而不是 capture 看到的物理 bank 地址。

#### 4.1.4 memory fragmentation 与容量不可证明

Paging 不会凭空压缩所有彼此独立的 live parameter。若 `N` 个 request 都拥有完全不同且同时存活的 state，其逻辑 tensor bytes 仍然必须存在。

PTTS 希望减少的是：

- 小 tensor 与 optimizer allocation 的 external/internal fragmentation；
- Python/optimizer object 与 allocator metadata；
- 未使用 capacity 的整 slot 预留；
- 版本间未修改 page 的重复副本；
- base/shared page 可通过 COW 复用时的冗余；
- 为保持 capture 地址稳定而过度预分配的完整 Module bank。

同时，固定 page capacity 可以让 admission failure 成为明确、可签名的资源结果，而不是运行到一半才 OOM、静默 eviction 或切换 legacy path。

### 4.2 目的

Paged Test-Time State 的主要目的如下。

#### 4.2.1 解耦 logical state 与 physical storage

workload、request 和 sampler 使用：

```text
StateRef(LayoutId, LineageId, VersionId)
```

backend 使用：

```text
PageTableRoot -> PhysicalPageBundle
```

logical state 不暴露物理 slot/page，也不要求连续物理 allocation。

#### 4.2.2 让 per-request training 成为可批处理的一等路径

相同 `LayoutId` 的 active requests 可以通过 typed slot/page tensor 进入同一 native forward/backward/optimizer implementation。batching 依据 layout、effect 和 signed selection 自动形成，不由 ESamp、TLM 或 TTT runtime 手写分组策略。

#### 4.2.3 保持 exact version semantics

系统必须能表达并验证：

- online predict-then-train；
- offline train-then-predict；
- request 在多个 token 间 pin 一个 immutable version；
- update 完成后原子 publish；
- old version 在最后一个 lease/fence 释放后回收；
- epoch replacement 不影响仍存活的旧 lineage。

#### 4.2.4 提供 capture-stable mutable state

physical page pools、page-table buffers、gradient pages 和 optimizer PageBundle 在 cold phase 创建并保持地址稳定，使通用 Rust scheduler 能根据 layout/storage/fence evidence 判断 eager、capture 和 replay 资格。

Paging 只提供事实，不携带 `lane`、`priority`、`preferred_stream` 或 workload-specific capture policy。

#### 4.2.5 建立跨 workload 复用的 state capability

同一 capability 应服务：

- low-rank adapter；
- dense fast weight；
- side model；
- prefix/recurrent state；
- optimizer moments；
- immutable version overlay。

workload 之间只允许 layout 和数学 task 不同；admission、allocation、pin、publication、fence、cleanup 与 MP transport 必须复用。

#### 4.2.6 提高可观察、可控制的容量

PTTS 应提供可以直接验证的容量指标：

```text
logical live bytes
reserved arena bytes
page utilization
internal/external fragmentation
free/admitted/in-flight/retiring pages
pinned versions
pending receipts/fences
admission failures
```

capacity exhaustion 必须在 admission 时显式失败，不采用无提示 LRU、eviction 或 legacy fallback。

### 4.3 非目标

PTTS 不是：

- tLLM 的统一 workload 语义；
- 把所有 state 强制转换成 tensor page；
- KV cache/PagedAttention 的直接替代；
- 对 main-model weight paging 的笼统重实现；
- arbitrary Python object serializer；
- environment、judge、checkpoint 等 host service 的容器；
- runtime workload scheduler；
- 仅仅为了减少 malloc 次数的 allocator patch。

它是 **device-resident、tensorizable、mutable、versioned test-time state 的可选物理表示与执行 capability**。

成功标准不是“所有 workload 都使用 paging”，而是：适合分页的 workload 共享同一套正确且高效的机制，不适合分页的 workload 仍通过 typed resident/sharded/host/external representation 工作。

### 4.4 paging 是 representation，不是 workload 语义

不能在公共 workload API 中引入：

```text
paged=True
model_bank=True
preferred_slot_count
ESampRuntime
TTTRuntime
```

workload 只声明逻辑 state 事实：

- scope；
- typed tensor layout family；
- read/transition/publication semantics；
- consistency/version semantics；
- mutation granularity；
- cold specialization facts。

Rust/backend 在编译阶段选择物理表示：

```rust
enum StateRepresentation {
    ResidentModule,
    DensePageArena,
    SparsePageArena,
    VersionedOverlay,
    ShardedState,
    HostService,
    ExternalHandle,
}
```

该选择进入 signed plan。运行时不得在 paged、resident 或 legacy implementation 之间静默 fallback。

### 4.5 泛化范围

Paged Test-Time State 适合满足以下条件的 state：

1. cold specialization 可以得到闭合 tensor layout；
2. hot path 不改变 tensor topology；
3. state 由明确 tensor blocks 构成；
4. transition 可以描述修改 granularity；
5. request/version lifetime 可以由 lease 与 fence 表达；
6. task 通过 logical state reference 访问，而不是永久持有 Python Parameter 地址。

潜在覆盖：

- ESamp per-request distiller 与 packed model bank；
- TLM q/v LoRA 与 optimizer moments；
- In-Place-TTT fast weights；
- prefix/prompt tuning；
- adapter、IA3、low-rank 或 block-sparse update；
- request-specific normalization state；
- recurrent learned memory；
- side reward/control head；
- 部分 test-time full/partial fine-tuning state。

不应强制分页：

- GTTA environment simulator；
- FTTT judge/verifier I/O；
- checkpoint/file handle；
- tokenizer 或 dataset iterator；
- arbitrary Python object；
- hot-path dynamic module topology；
- opaque external service state。

这些继续作为 typed host/external service。Paging 只优化 device-resident mutable tensor state，不缩小 tLLM 的逻辑 workload 范围。

### 4.6 typed tensor layout

core 不得出现 LoRA/TTT/ESamp 专属字段：

```text
q_proj
v_proj
lora_A
lora_B
esamp_slot
ttt_weight
```

使用闭合 layout descriptor：

```rust
struct TensorLayout {
    layout_id: LayoutId,
    fields: Vec<TensorField>,
}

struct TensorField {
    field_id: FieldId,
    shape: ShapeExpr,
    dtype: DType,
    alignment: Alignment,
    mutation_granularity: PageGranularity,
}
```

HF/PEFT 模型装载属于 cold typed service：

```text
cold loader
-> resolve and freeze LayoutId
-> create physical arena
-> hot native tasks use LayoutId + StateRef
```

rank、layer set 或 topology 改变时创建新 epoch，而不是 hot path 反射推断。

### 4.7 page table 与 PageBundle

```text
Logical StateRef
  |- LayoutId
  |- LineageId
  `- VersionId
        |
        v
Version Root / Page Table
        |
        v
Physical PageBundle
  |- parameter page
  |- gradient page
  |- exp_avg page
  `- exp_avg_sq page
```

parameter、gradient 与 optimizer moments 共享 virtual page index，但可以位于不同物理 page pools。

更新流程：

```text
pin version V
-> acquire writable page set
-> COW touched pages
-> native forward/backward/optimizer
-> record completion fence
-> atomically publish V+1 root
-> request pin exact V or V+1
-> reclaim old pages after last lease/fence/receipt
```

与 PagedAttention 相同，logical continuity 与 physical storage 解耦；不同之处是 mutable state 还需要 optimizer、COW、immutable version publication 与 request pinning。

### 4.8 泛化验收

同一个 page arena/core 至少应承载四种结构明显不同的 workload：

1. ESamp low-rank side model；
2. TLM q/v LoRA；
3. In-Place-TTT dense fast-weight overlay；
4. prefix tuning 或 recurrent learned memory。

差异只允许存在于：

- typed layout；
- workload 数学 task；
- cold initializer。

以下机制必须完整复用，且不得出现 workload 名称：

- admission；
- page allocation；
- version root；
- lease/pin；
- optimizer PageBundle；
- publication；
- completion fence；
- cleanup；
- MP wire。

GTTA/FTTT host service 应作为明确的非分页反例，证明 tLLM 没有把所有 workload 强制收敛到 paged tensor state。

## 5. AER 与 Paged Test-Time State 的组合

```text
SemanticGraph + RegionSummary
              |
              v
             AER
              |
              v
ScheduleSolution + state representation choice
              |
              +-> ResidentModule
              +-> Paged Test-Time State
              +-> VersionedOverlay
              +-> HostService
              `-> ExternalHandle
```

AER 决定何时、如何组合并放置计算；PTTS 决定一类 mutable tensor state 如何高效存储、版本化和批处理。二者不能互相成为对方的前置限制。

## 6. 预期工程收益

基于当前相关文件的粗略只读统计，active native core 约有 20,000 行生产代码。canonical graph、derived index、RegionSummary 与 dense runtime layout 的长期目标可能减少约 5,000–7,000 行重复 schema、validator、字符串 binding 与执行胶水。

叠加旧 Program/WorkGraph shadow retirement 后，生产代码净减少的合理区间约为 10,000–14,000 行。该估计不是实施承诺；迁移期会暂时增加 codec、oracle 和双路径测试。

更重要的目标是降低修改扩散：

- 新 effect/resource 从修改十几个文件下降到少数 canonical variant、codec 与 service；
- runtime 不再从 `dict[str, object]` 推断 signed binding；
- scheduler optimization 不依赖 workload-specific policy；
- illegal state 由类型和 ownership structure 阻止。

## 7. Publication 验证框架

### 7.1 AER 算法实验

- brute-force oracle；
- 当前 merge enumeration；
- greedy contraction；
- ILP/CP-SAT optimal oracle；
- anchor-frontier DP；
- compile latency、visited states、optimality gap、peak memory、materialization、Region/fence count。

### 7.2 AER 系统实验

- TTFT、ITL、out tok/s；
- GPU memory peak；
- host launch 与 queue delay；
- materialization bytes；
- CUDA Graph capture/replay；
- loss/update/state-version evidence；
- 单 workload 与多 workload composition；
- MRv1/MRv2、in-process/MP、eager/CUDA Graph。

### 7.3 PTTS 实验

- concurrent request capacity；
- parameter/optimizer memory utilization；
- external/internal fragmentation；
- state admission/release latency；
- batched forward/backward/optimizer throughput；
- version publication 与 sampler pin correctness；
- eager/capture/replay parity；
- 与独立 Module/optimizer、resident model bank 及现有 multi-adapter systems 比较。

任何 throughput 结果都必须同时证明 `loss_count/update_count > 0`、state version 正确推进且 fallback 为零。

## 8. 建议的研究顺序

1. 建立 AER 小图 brute-force partition + placement oracle。
2. 自动生成 counterexample，验证 latest-placement 与当前 partition search 的边界。
3. 收集迁移后真实 workload graph 的 task count、active-window width 与搜索状态数。
4. 根据真实 width 决定使用 branch-and-bound、rollback search 或 frontier DP。
5. 独立实现 PTTS 两 request、两 version、两 page 的最小 COW/pin/reclaim oracle。
6. 用 ESamp、TLM、In-Place-TTT 与第四种非 LoRA state 检验 paging 泛化。
7. 分别评估 AER 与 PTTS 的数量级收益，再选择 publication 主线。

## 9. Open questions

- AER 一般问题的准确复杂度和 NP-hard reduction 是什么？
- interval Helly/path decomposition 在加入 arbitrary data hyperedge 与 effect order 后保留什么 FPT 参数？
- Region placement 是否存在比联合 DP 更简单的 min-cost flow 特例？
- structural cost 与真实 GPU interference cost 应如何分层，而不让 runtime profiling 成为新的调度 authority？
- page size 应按 byte、tensor block、parameter group 还是 update granularity 定义？
- variable-rank/variable-layer state 应使用多 size-class arena、buddy allocation 还是 typed page family？
- optimizer COW 应以 page、parameter group 还是完整 version 为单位？
- distributed TP/PP/DP 下 version root、page ownership 与 cleanup quorum 如何定义？
- 如何与 S-LoRA、Punica、multi-LoRA training、paged optimizer、polyhedral fusion 和 memory-constrained graph partition 明确区分贡献？

这些问题解决前，不应把 AER 的最优性或 PTTS 的泛化性写成既定结论。

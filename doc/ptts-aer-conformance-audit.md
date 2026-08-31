# PTTS / AER 实现一致性审计

状态：2026-08-31。

本文以 [`research-aer-paged-state.md`](research-aer-paged-state.md) 为
PTTS 与 AER 术语、数据结构和研究边界的权威来源。该研究文档仍是提案而非已验证
性能结论；但工程实现不得把与其冲突的临时结构描述为 PTTS/AER 终态。

2026-09-01 起，本审计同时冻结实施边界：PTTS canonical kernel 与 signed
boundary 收口后，不再向旧 frontend/runtime/WorkGraph 回填。新 Python 核心位于
`tllm/canonical/`，Rust 核心位于 `rust/tllm_core/src/canonical/`；新旧共存
只允许发生在最外层 routing boundary。

第一个纵向切片 Dummy 已经通过该边界：public `tllm.consumers.dummy`
只在 package routing boundary 转发到 clean-room implementation，实际执行链为
`SemanticGraph → Rust schedule → Rust PhysicalPlan → strict single-task execution`。旧
Dummy workload/stream/transfer/worker 与只验证它们的 runtime/MP/GPU 测试已删除。
当前 Dummy 证据只是 CPU public-entry vertical oracle，不宣称 GPU/MP 已迁移。

## 总结

当前代码具有一个有效的最小 immutable-root prototype：固定地址 tensor pools、
page-id root、source-to-distinct-target COW、exact root pin、readiness/completion
fence、old-root reclaim，以及 tensor-only stable projection 都有 CPU/GPU 证据。

它尚不是研究文档定义的完整 PTTS canonical core。arena-local logical
identity、frozen typed TensorLayout interning、provider-local capacity proof、atomic publication
和 Rust-signed representation boundary 已落地；LineageId 的 epoch/MP 作用域、完整
PTTS runtime/transport 和跨 workload 泛化仍未闭合。AER 的
`SemanticGraph`、dense typed IDs、derived `GraphIndex`、incremental
`RegionSummary` 与 partition+placement oracle 尚未实现。

在以下 P0 差异关闭前，不继续启用 ESamp compact capture，也不宣称 PTTS、AER、
MP 或 legacy deletion gate 已完成。

## PTTS 对照

### 已符合的最小语义

- `PageBundle` 持有 cold-allocated global tensor pools；root 只持 page map，consumer
  通过 global operands 与 page IDs 间接读取。
- fixed adapter 的 parameter、buffer、AdamW `step/exp_avg/exp_avg_sq` 已进入同一
  physical page family；optimizer 从 immutable source page 读取并写 distinct
  target page。
- root publication、exact root pin、writer readiness、consumer completion 与
  reclaim 已形成闭合状态机；old parameter 和 optimizer moments 均有不变性 oracle。
- `StablePackedStateView` 将 packed fields、row/lease IDs、masks、first-row、
  page IDs 与 state operands 展开为唯一 canonical tensor tuple，不携带 module、
  selection 或 representation tag。
- fixed provider 以 `StateCapacityEntitlement(max_admitted_lineages=N,
  max_live_versions_per_lineage=3)` cold-allocate `3 × logical_capacity` tensor pools；
  该数值没有进入 workload config、lane、priority 或 scheduler policy。

### P0：arena-local StateRef 已落地，全局作用域未闭合

研究定义要求：

```text
StateRef(LayoutId, LineageId, VersionId)
```

当前 root arena 已以 `StateRef` 为键；request lease/effect/receipt/pin 携带独立
`LineageId` 和 `VersionId`，`StateSlotRef` 只保留 allocator/ABA 证明。同一 slot
reuse 会获得新 lineage，exact V0 pin 在 publish V1 后仍可验证并阻止 retire。

这仍是 **arena-local identity prototype**，但 canonical LayoutId 已不再依赖 graph
state ordinal：

- `tllm/canonical/tensor_layout.py` 与 Rust `canonical/tensor_layout.rs` 使用 closed
  dtype/mutation enums、namespace、dense field IDs、shape、alignment 和 mutation
  granularity 形成 frozen descriptor。Rust 按 canonical descriptor JSON 排序签发 dense
  `LayoutId`；相同 descriptor 跨 resource 共享 ID，resource/graph 顺序不参与
  layout identity。
- dense `LayoutId` 只在 decision 携带的 `layout_namespace` catalog digest 内解释。
  catalog 增删 descriptor 可改变 ordinal，但必须同时改变 namespace；因此完整
  cold identity 是 `(layout_namespace, LayoutId, layout_digest)`。`StateRef` 只允许在
  一个 namespace-owned arena/physical plan 内解释，不得跨 namespace 裸比较 ID。
- 旧 runtime 的 graph-state-order interner 只冻结为待删旧栈接线，不再代表
  canonical PTTS LayoutId，也不再扩展。
- coordinator 与 worker 现在都从已验证的同一 graph 派生相同 binding，但 PTTS
  root/page/pin/receipt 尚未进入 signed MP wire；`LineageId` 仍由每个
  `RequestSlotArena` 本地签发。
- 在 cross-workload shared arena、epoch overlap 或 StateRef 跨进程传输前，必须冻结
  LayoutId namespace 和 coordinator-owned/deterministically witnessed lineage
  issuance；不得恢复 Python ordinal fallback。

### P0：physical representation 泄漏进 workload 语义

当前 public frontend/IR 仍包含：

- `FixedBatchedRequestTrainableState`；
- `request_batched_trainable()`；
- `StateStorage.FIXED_BATCHED`；
- ESamp model-bank 物理路径选择。

这些违反“paging 是 representation，不是 workload 语义”。终态 workload 只声明
request scope、typed layout family、read/transition/publication、consistency、
mutation granularity 与 cold facts。fixed/page/overlay/resident 只能由 Rust/backend
作为 signed physical solution 选择。`FixedBatchedSlotAdapter` 可保留为 physical
provider，但不能继续拥有 public semantic variant。

新 canonical boundary 已经绕开这些旧类型：`StateRepresentationRequest` 将
representation-neutral layout requirement 与 physical provider candidates 作为独立 cold
facts 交给 Rust；Rust 只按 capability coverage、`reserved_bytes` 和稳定 provider ID
选择并签名 `StateRepresentationDecision`。`StatePhysicalPlan` 只保留原 decision
tuple 和 signature；projection 前必须再通过 Rust exact recomputation，不重新选择。
provider 构造同时验证 request digest、decision digest、完整 plan tuple 和 reserved
bytes。worker reconstruction 必须证明 local catalog 与
parent-signed request catalog 完全相同，不得 fallback。该新 boundary 不导入旧
runtime/workload/workgraph，provider facts 也没有进入旧 `ScheduleGraph`。
此处“signed”表示 Rust 重算并校验的 proof-carrying SHA-256 decision digest，不是
带密钥的身份认证；parent/worker 来源和 quorum 仍由后续 closed MP control plane 负责。

旧公共 API 现在冻结不动；不再就地 internalize 或建立新桥接，而是在对应
workload 切换到 clean-room 新核心时直接删除。

最小真实 producer 证据使用 clean-room CPU dense-page provider 的 weight、bias、
AdamW `step/exp_avg/exp_avg_sq` 和 gradient 六个 typed pools。Rust tensor interner
签发 layout namespace/ID，representation planner 签名 dense provider，经 Rust 验证的
physical projection 驱动真实 admission、AdamW update、pin、atomic publication 和 reclaim。
provider 明确只证明 CPU，不伪装 GPU/capture/fence 支持。这是 kernel/signed-boundary
证据，不是 ESamp 公共入口已切换的声明。

### P0：provider-local capacity entitlement 已落地，signed authority 未闭合

当前 `PagedStateVersionArena` 在 cold construction 时验证：

```text
bundle pages
>= max_admitted_lineages
 * max_live_versions_per_lineage
 * root_page_count
```

admission 一次性占用 lineage entitlement，transition reserve 逐 lineage 计数 live
roots；只要生命周期不超过已声明的 version bound，就不会再遇到随机
global page exhaustion。fixed provider 的 `3` 表示 current + one retained historical
root + one target credit；保留两个不同 old versions 后再 transition 会以明确
entitlement violation 拒绝，释放一个 pin 后可继续。

尚未闭合的是 authority：该 entitlement 当前是 fixed physical provider 的 cold
capability，还未进入 Rust-signed representation solution。在此之前不宣称一般
PTTS capacity 已完成，也不得加入 eviction、LRU 或 legacy fallback。

### P0：physical atomic multi-root publication 已落地

admission 与 transition 现在都是 batch-only API：先验证全部 refs/entitlement，
一次 reserve 全部 target pages，compute 完成后记录同一 completion fence，再在
一个 root-arena lock 内同时更新 roots、latest map、page ownership 和 writer
pins。compute 提交前 abort 回收整个 staged batch，不发布半个 selection；已提交
CUDA 失败继续 fail-stop，不伪造回滚。single-root wrapper 已删除。physical
commit 后 logical commit 若失败，仍双向 poison，不从 physical latest 反推逻辑状态。

### P1：typed layout cold core 已落地，dynamic family 仍未完成

canonical Rust interner 已签名 namespace、dense field IDs、frozen shape、closed dtype、
alignment 与 mutation/page granularity。仍缺：

- `ShapeExpr`/variable layout family；
- page family/size class；
- variable-rank/layer epoch replacement contract。

这些字段必须是 cold typed facts，不得在 hot Python 路径从 tensor/module 反射推断。

### P1：gradient、epoch、metrics 与 MP

- gradient 目前是 stable global transient buffer，不属于 typed PageBundle/root。只读
  projection与 eager optimizer 可暂时使用；capture backward/optimizer 前必须把
  gradient scratch/page ownership 纳入 closed layout。
- 当前 epoch replacement 要求无 live request；研究目标允许新 epoch 与仍 pinned 的
  旧 lineage 共存。需要 epoch-owned arenas 与 cleanup quorum。
- 只有 `free_pages`，尚缺 logical/reserved bytes、utilization、fragmentation、
  admitted/in-flight/retiring pages、pinned roots、pending receipts/fences 与 admission
  failures。
- worker cold specialization 已从验证后的 signed graph 派生同一 declaration-layout
  binding；仍无 PTTS root/page/pin/receipt 的 EngineCore-MP typed reconstruction wire。

### P1：跨 workload 泛化未完成

generic ABI 与 fragmented fake producer 证明了 consumer 不依赖 fixed mapping；但真实
producer 只有 ESamp fixed side state。必须用同一 core 验收 TLM q/v LoRA、
In-Place-TTT dense fast weight，以及 prefix/recurrent state。GTTA/FTTT typed host
services 保留为明确非分页反例。

## AER 对照

当前 AER 状态为“未实现”，不能把现有 Region merge search 称为 AER：

- 仍有 string-heavy `ScheduleGraph`/Decision/physical records 与重复 validator；
- 没有单一 `SemanticGraph` authority；
- 没有 dense `TaskId/ValueId/StateId/AnchorId/BindingId` interner；
- 没有可重建 CSR/bitset `GraphIndex`；
- 没有 closed、增量 `RegionSummary` merge algebra；
- 没有 joint partition+placement brute-force oracle 或 latest-placement counterexample
  corpus。

后续新增 projection manifest、binding 或 physical witness 时，必须明确它们是
`SemanticGraph` 的 specialization/physical projection，不得成为另一份 task/state/
effect authority。AER 与 PTTS 相互独立：PTTS 修正不阻塞 AER oracle，AER 也不拥有
page allocation/version lifecycle。

## 修正后的实施门序

```text
1. [arena-local 已落地] typed LayoutId / LineageId / VersionId / StateRef
2. [已落地] RequestSlotArena lease 同时携带 logical lineage 与 physical SlotRef
3. [已落地] root arena 改用 StateRef，补 ABA 与 old-version pin oracle
4. [已落地] frozen TensorLayout-based LayoutId/namespace 与 closed codec
5. [provider-local 已落地] admission entitlement + atomic multi-root publication
6. [已落地] 独立 Rust-signed representation decision / physical projection / worker proof
7. [已落地] 真实 fixed request-state producer 闭合 layout → plan → PTTS lifecycle
8. 冻结旧栈 PTTS 接线，不再就地 internalize FIXED_BATCHED
9. 在 `tllm/canonical` / Rust `canonical` 建立新 SemanticGraph/scheduler/runtime
10. 按 simple → ESamp → GTTA/FTTT 纵向切换，每次立即删除旧 workload
```

AER 独立按以下门序推进：

```text
SemanticGraph boundary
-> dense typed IDs/interner
-> rebuildable GraphIndex
-> incremental RegionSummary
-> small-graph partition+placement brute-force oracle
-> latest-placement counterexamples
-> 根据真实 task count/frontier width 选择搜索算法
```

## 最小高价值验证

保留：

- two-request/two-version/two-page、alias、ABA、slot relocation oracle，以及
  pin V0 → publish V1 → release selection → pin 阻止 retire → release pin 后 retire；
- parameter 与 optimizer moments old-root immutability；
- readiness/completion/reclaim GPU smoke；
- identity/fragmented producer 共用 tensor-only executor；
- capacity entitlement 与 atomic batch-publication failure oracle；
- MP reconstruction round trip；
- AER 小图 brute-force optimality 与 latest-placement counterexample。

不增加 representation × workload × runner × capture 笛卡尔矩阵，也不测试 private
ID 自增、每条错误文案或 page-count 的机械排列。

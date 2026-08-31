# PTTS / AER 实现一致性审计

状态：2026-08-31。

本文以 [`research-aer-paged-state.md`](research-aer-paged-state.md) 为
PTTS 与 AER 术语、数据结构和研究边界的权威来源。该研究文档仍是提案而非已验证
性能结论；但工程实现不得把与其冲突的临时结构描述为 PTTS/AER 终态。

## 总结

当前代码具有一个有效的最小 immutable-root prototype：固定地址 tensor pools、
page-id root、source-to-distinct-target COW、exact root pin、readiness/completion
fence、old-root reclaim，以及 tensor-only stable projection 都有 CPU/GPU 证据。

它尚不是研究文档定义的 PTTS canonical core。arena-local logical
identity 已经不再依赖 physical slot，但 LayoutId/LineageId 的 graph、epoch 和
MP 作用域、public semantic API、容量证明、layout 泛化与 transport 仍未闭合。AER 的
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

这只是 **arena-local identity prototype**，还不能称为完整 PTTS identity：

- Rust 目前按一个 composed `ScheduleGraph` 的 state declaration 顺序签发 dense
  `LayoutId`。它保证同一 graph 内唯一，但尚未基于 frozen tensor fields、shape、
  dtype、alignment 与 mutation granularity 进行 layout interning；因此它是临时的
  declaration-layout ID，不是已经完成泛化的 TensorLayout identity。
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

### P1：typed layout 仍是最小子集

物理 `FrozenTensorLayout` 已有 dense field IDs、shape 与 dtype，但这些事实尚未进入
Rust layout interner 的 semantic/cold input；仍缺：

- `ShapeExpr`/layout family；
- alignment；
- mutation/page granularity；
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
4. 冻结 TensorLayout-based LayoutId namespace、epoch 边界和 MP lineage issuance witness
5. [provider-local 已落地] admission entitlement + atomic multi-root publication
6. internalize FIXED_BATCHED，physical representation 进入 signed solution
7. 补 layout alignment/mutation granularity 与 PTTS metrics
8. [projection ABI 已落地] 在前述 P0 门禁后再完成 Rust capture predicate
9. 再接 ESamp read-only capture；training loss/optimizer 保持 eager
10. PTTS MP wire 与第二、第三、第四个真实 producer
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

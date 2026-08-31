# PTTS / AER 实现一致性审计

状态：2026-08-31。

本文以 [`research-aer-paged-state.md`](research-aer-paged-state.md) 为
PTTS 与 AER 术语、数据结构和研究边界的权威来源。该研究文档仍是提案而非已验证
性能结论；但工程实现不得把与其冲突的临时结构描述为 PTTS/AER 终态。

## 总结

当前代码具有一个有效的最小 immutable-root prototype：固定地址 tensor pools、
page-id root、source-to-distinct-target COW、exact root pin、readiness/completion
fence、old-root reclaim，以及 tensor-only stable projection 都有 CPU/GPU 证据。

它尚不是研究文档定义的 PTTS canonical core，原因主要是 logical identity、
public semantic API、容量证明、layout 泛化和 MP transport 尚未闭合。AER 的
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
- fixed `2 × logical_capacity` headroom 位于 physical adapter 内，没有进入 workload
  config、lane、priority 或 scheduler policy。

### P0：logical StateRef 不符合

研究定义要求：

```text
StateRef(LayoutId, LineageId, VersionId)
```

当前 root key 实际为：

```text
resource + epoch + StateSlotRef(slot_id, generation) + version
```

`StateSlotRef` 是 allocator/ABA witness，不是 logical lineage。必须在 admission 时
生成独立 typed `LineageId`；logical lease 同时携带 lineage 与 slot ref，root 只以
`StateRef` 为逻辑键。slot relocation、page compaction 或同 lineage 的新 physical
mapping 不得改变 StateRef。

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

### P0：容量只做到明确失败，尚未 admission-time 可证明

`2 × logical_capacity` 能保证正常 lifecycle 中每个 current root 有一次 next-root
空间；它不能证明任意长期 old-root pin/version fanout 都不会在 transition 时耗尽。

必须选择并签名一种明确契约：

1. admission 预留 version/fanout entitlement，后续合法 transition 不再因 page
   capacity 失败；或
2. 研究文档明确放宽为 transition-time typed capacity result。

在权威文档未改变前，工程按方案 1 实现，不把当前 2× headroom 宣称为完整容量
证明。不得 eviction、LRU 或 legacy fallback。

### P0：multi-root publication 还不是一个物理原子事务

一个 selection 的多个 slot 当前逐 root publish，再一次性 commit logical transition。
中途失败会 fail-stop，因此不会继续错误执行，但尚未达到同一 receipt 对应的 atomic
multi-root publication。需要 batch root transaction：先 reserve/compute 全部 target，
record fence，再在一个 root-arena commit 中发布全部 refs；logical commit 失败仍
双向 poison。

### P1：typed layout 仍是最小子集

已有 dense field IDs、shape 与 dtype；仍缺：

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
- 尚无 PTTS root/page/pin/receipt 的 EngineCore-MP typed reconstruction wire。

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
1. typed LayoutId / LineageId / VersionId / StateRef
2. RequestSlotArena lease 同时携带 logical lineage 与 physical SlotRef
3. root arena 改用 StateRef，补 slot-relocation/ABA oracle
4. admission capacity entitlement + atomic multi-root publication
5. internalize FIXED_BATCHED，physical representation 进入 signed solution
6. 补 layout alignment/mutation granularity 与 PTTS metrics
7. 再完成 stable tensor projection / Rust capture predicate
8. 再接 ESamp read-only capture；training loss/optimizer 保持 eager
9. MP wire 与第二、第三、第四个真实 producer
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

- two-request/two-version/two-page、alias、ABA、slot relocation oracle；
- parameter 与 optimizer moments old-root immutability；
- readiness/completion/reclaim GPU smoke；
- identity/fragmented producer 共用 tensor-only executor；
- capacity entitlement 与 atomic batch-publication failure oracle；
- MP reconstruction round trip；
- AER 小图 brute-force optimality 与 latest-placement counterexample。

不增加 representation × workload × runner × capture 笛卡尔矩阵，也不测试 private
ID 自增、每条错误文案或 page-count 的机械排列。

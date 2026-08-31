# PTTS-0.5 Stable Packed State View 设计证明

状态：2026-08-31，当前 compact-capture 原型 **NO-GO**。

本文冻结 PTTS-0.5 的永久 capture ABI，并说明为什么当前 fixed-batched
实现尚不能满足该 ABI。本文只决定 representation 与 lifetime 边界，不为
ESamp 或任何其他 workload 指定 lane、priority、fusion、order、stream 或
capture 策略。

## 1. 结论

以下映射可以成立：

```text
PageBundle + pinned immutable version root + fence
-> StablePackedStateView
```

但成立的前提是 fixed producer 与 paged producer 都从 immutable version
root 产生 view，并遵守同一套 root pin、readiness、completion 与 reclaim
lifecycle。当前 `FixedBatchedSlotAdapter` 原地更新参数，arena 的 version 只是
logical metadata；因此已经 pin 的版本仍可能观察到下一次更新。给现有 packed
buffer 或 module input 改名不能补出 immutable root，当前 compact-capture
原型不得提交。

按 PTTS-0.5 门禁，compact capture 暂停。下一 tranche 先实现最小
`PageBundle`、immutable version root、pin/publication 与 fence core；不提前
实现 page allocator、COW 策略、eviction、paging、fragmentation 或跨 workload
page reuse。

## 2. 唯一 closed ABI

`StablePackedStateView` 只有以下顶层字段类别：

```python
@dataclass(frozen=True, slots=True)
class StablePackedStateView:
    packed_tensors: StablePackedTensorSet
    row_slot_ids: torch.Tensor
    lease_slot_ids: torch.Tensor
    valid_mask: torch.Tensor
    first_row_ids: torch.Tensor
    stability_epoch: StateStabilityEpoch
    version_epoch: PinnedStateVersionEpoch
    completion_fence: StateViewCompletionFence
```

这些名字描述契约，不是当前 Python 实现草案。正式实现必须使用 closed typed
records；不得使用 role 字符串、callback bag、`Any`、宽 `object`、反射或
workload 名称恢复字段含义。

- `StablePackedTensorSet` 的 tensor arity、顺序、shape、dtype 与 device 由 cold
  frozen layout 的 dense field IDs 唯一决定。它包含本次 view 的 packed hidden
  tensors，地址在整个 stability epoch 内保持不变。
- `row_slot_ids` 对齐 packed rows；`lease_slot_ids` 对齐 unique leases；二者都包含
  固定 capacity 的 device tensor，而不是 capture 时读取的 Python metadata。
- `valid_mask` 关闭 padding；`first_row_ids` 给出每个 unique lease 的 canonical
  first row。二者与 ID tensors 同样拥有固定地址与固定签名。
- `StateStabilityEpoch` 绑定 resource、frozen layout、producer epoch 与地址签名。
  epoch 变化意味着旧 capture artifact 不再有效，而不是运行时猜测新类型。
- `PinnedStateVersionEpoch` 绑定每个 lease 的 slot、generation、logical version，
  以及一个 pinned immutable root。root 含 consumer 所需的 fixed-address state
  operand/page-map tensors 与其 readiness fence。
- `StateViewCompletionFence` 属于本次 view lease。consumer 在实际执行 stream 上
  记录完成后，producer 才能释放 view buffer 与 root pin。

CUDA Graph callable 只接收上述契约投影出的固定签名 tensors。selection、rows、
module、root owner 与 fence owner 都是 effect/lifetime binding，不进入 callable，
因而 replay 不会固化第一次调用的 Python control object。

## 3. Authority 与生命周期

`RequestSlotArena` 继续唯一拥有 request key、alias、slot generation、selection、
logical version、receipt、pin 与 retire。representation producer 只拥有 physical
bundle、immutable root、view buffers 与 GPU fences；它不得另建 logical slot
mapping。

创建 read view 的事务顺序固定为：

```text
validate selection + (slot, generation, version)
-> pin exact immutable root
-> order root readiness fence before the consumer stream
-> acquire a reusable stable view-buffer lease
-> fill packed tensors and device metadata
-> expose StablePackedStateView
```

消费与释放顺序固定为：

```text
run eager/capture/replay on the signed consumer stream
-> record view completion fence on that stream
-> after event completion release view-buffer lease
-> release immutable-root pin
-> release selection / permit reclaim and ABA reuse
```

transition 的顺序固定为：

```text
pin source root V
-> acquire a distinct writable destination bundle
-> run forward/backward/optimizer
-> record destination readiness fence
-> atomically publish immutable root V+1
-> retain V until its last view/root pin, receipt and fence complete
```

readiness fence 证明 producer 的写已经完成；completion fence 证明 consumer 的读
已经完成。二者方向相反，不能用 Region 结束后临时创建的单个 slot pin 互相替代。

## 4. Fixed 与 paged producer 的同构映射

### Fixed producer

最小 fixed producer 使用预分配、地址稳定的 version bundles。root 的 page map 是
identity mapping；transition 写入与当前 root 不同的 bundle，完成后原子发布新
root。view 中的 slot IDs 直接索引该 root 的 fixed operands。当前原地更新的
fixed bank 只有在改为此 immutable-root backing 后才能成为 producer。

### Paged producer

Paged producer pin exact root 后，将 immutable page-id map 和 selection metadata
写入相同 shape/dtype/address signature 的 view buffers。consumer 通过 root 的
fixed-address page-map tensors 间接访问全局 page pools；它不永久持有某一版本
页面的动态地址。

两种 producer 的差异止于：

```text
logical root -> fixed-address state operands/page map
```

它们输出完全相同的 `StablePackedStateView`。因此加入真正的
`PagedStateAdapter` 时，只新增 producer；workload declaration、
`RequestSlotArena`、Rust capture consumer 与 view ABI 均不修改。Rust 只检查：

- signed read effect 与 view resource/version 对齐；
- frozen tensor signature 与 stability epoch 对齐；
- root 已 pin 且 immutable；
- readiness/completion witness 完整；
- callable inputs 全部是签名中的 tensors。

Rust 不读取 resident/fixed/paged、workload 名称或实现名称来决定 capture。

## 5. 当前原型失败点

1. fixed-batched 参数和 AdamW moments 原地更新，没有 immutable root 或旧版本
   backing；logical version 不能证明 physical snapshot isolation。
2. packed buffer 只有 resource/selection/epoch、tensors、slot IDs、mask 与 first
   rows，缺 generation、pinned root、readiness 和 completion ownership。
3. module、selection 与 rows 作为 Python control object 传入首次 CUDA Graph
   capture；replay 不会重新绑定下一 step 的 metadata。
4. captured loss 在 graph 外执行 backward/optimizer，尚未证明 replay 会产生可供
   每次 transition 独立消费的 autograd state。因此训练 loss 保持 eager；只有
   forward/backward/optimizer 形成一个完整 signed transition graph 并通过独立
   证明后，才可讨论 capture。
5. fixed-batched 专属 Rust/Python predicate 依据 physical representation 放行，
   违反 scheduler 只使用结构事实的约束。

## 6. 最小 PTTS core 的提交门

下一 tranche 只实现证明所必需的公共机制：

1. dense typed layout/field IDs 与 frozen tensor signature；
2. 预分配 `PageBundle` 和 immutable version root；
3. exact root pin、atomic publication、readiness/completion 与 reclaim；
4. capacity exhaustion admission-time fail closed，无 eviction 或 legacy fallback；
5. fixed producer 与一个最小 paged-producer double 的同一 view conformance。

只保留能覆盖真实错误空间的证据：two-slot/duplicate-row/two-version/ABA 的 CPU
oracle；一次 old-root pin 与 completion fence 阻止 reclaim 的 GPU smoke；一个
Rust structural-fact tamper test。不得扩展成 runner、mode 或负向条件的笛卡尔积。


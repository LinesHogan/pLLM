# Glossary

One meaning per term; the signing and execution boundaries use exactly these words.

## Authoring

- **Workload** — a frozen class of typed tensor methods and declared effects (`pllm.Workload`,
  `CompletionWorkload`, `CausalAdapterWorkload`). Compiled once; never a runtime hook.
- **Phase / anchor window** — the signed interval between two timeline anchors
  (`request.prepare`, `model.execute`, `state.publish`, `sampler.pre_draw`, `token.commit`,
  `next.prepare`) in which an operation runs. `Phase` is the DSL spelling of a `When` window.
- **Source marker** — the `Annotated[Tensor, ...]` argument naming where an input comes from:
  `Hidden`, `Samples`, `Batch`, `Parameter`, `StateIndex`, `StateVersion`, `Result`,
  `BaseLogits`, `SampledTokens`.
- **Capture marker** — third annotation argument `Capture(state, field)` on a training input:
  at publication, each lineage's memory field receives the input's last selected row value.
- **Fingerprint** — the cold structural signature of authored mathematics (marshaled bytecode
  plus transitive global/free-variable closure), computed without evaluating annotations or
  executing code. Changing mathematics re-signs; loading never re-executes.
- **Freeze** — the detached deep copy of an authored workload whose function values are
  recursively frozen, so later edits to the authoring environment cannot drift signed behavior.

## State

- **State** — the only mutable thing: named, versioned tensor fields under a declared layout.
  Produced three ways with identical consumption: optimizer updates, hypernetwork/external
  import, checkpoint loading.
- **Lineage** — one logical state's version chain. Instance scope has one lineage; request
  scope has one per admitted key. Lineages are never merged or split by execution.
- **Memory field** — a plain per-lineage field that rides the training transition without
  optimizer expansion (no gradient/moment slots). Read as the input version: step N+1 sees
  what step N published. Pages are zero-initialized at admission.
- **Version** — an immutable root of a lineage. Reads pin a version; writes create a private
  candidate; publication makes it visible atomically.
- **Input version** — the version pinned at step start; every input-version read in the step
  sees exactly this value, which is what makes cross-step memory semantics free.

## PTTS (paged tensor state)

- **Page** — the allocation unit of a field's storage; slots carry generations so stale
  references fail loudly instead of aliasing.
- **Pin** — an exact hold on published versions. Reclaim is exact: a page returns only when
  every pin and publication referencing it has released.
- **Publication** — the atomic act that makes a private transition visible, producing
  `after_refs` paired to `before_refs`. Failed publication preserves roots and permits retry.
- **Transition** — the private write scope of an invocation: one write set, one publication.
- **Arena** — the single version/generation/completion owner (`PageBundleArena`). No second
  store of state truth exists anywhere else.

## Execution

- **Semantic graph** — the signed task/state/effect declaration; the planner's only input.
- **Physical plan** — the signed lowering of the graph onto anchors, pages and buffers.
- **Fallback** — execution outside the signed primitives. The count is always asserted zero.
- **Overlay** — a signed parameter effect (dense or low-rank) applied at model execution from
  state fields, independent of how those fields were produced.
- **Replay** — signed re-execution of a layer range (`ConditionalLayerReplay`) or tokenwise
  recurrence, backed by per-layer KV backup/restore inside the same publication discipline.
- **Route lineage** — a dormant-persisting lineage shared by requests whose user-computed
  route key matches; retired explicitly at session close.
- **Evidence** — a numbered, manifest-recorded observable contract with a command that proves
  it. Tests extend existing evidence; new IDs require a genuinely new observable boundary.

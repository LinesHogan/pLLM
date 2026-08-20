# WorkGraph Phase 5

Phase 5 adds bounded online admission above the frozen Phase 3 placement and
Move plan. The scheduler may defer, use an authoritative fallback, or shed
explicitly optional work, but it cannot change semantics, create a Move, or
select an implementation that the compiler did not approve.

## Frozen adaptive plan

`tllm.workgraph.phase5` defines stable typed keys and frozen descriptors for
integer EWMA profiles, workload and shape buckets, interference budgets,
candidate Move sets, candidate actions, and work items. Floating-point profile
state and string policy dictionaries are not part of the contract.

Every `Phase5ControlPlan` embeds the exact `PlacementPlan` that owns its epoch.
Its scheduled Moves must equal the Phase 3 plan, and each candidate Move set
must be a dependency-closed subset of those Moves. The placement structural
signature is frozen in the plan. `compile_phase5_plan()` derives all of these
fields from the typed placement plan rather than accepting a caller-provided
Move universe.

Execute actions require a semantic-preserving authoritative fallback. Required
work cannot select a shedding action. This includes explicitly required work,
work in a `required_by_default` workload bucket, and work whose descriptor sets
`allow_shedding=False`, or an approved action marked `required=True`. Optional
work may be dropped only when its resolved candidate set contains an explicit
`SHED` action; rejection without compiler-approved shedding uses the
authoritative fallback. Unknown workload, shape, or semantic identity fails
closed unless the single unknown-bucket fallback preserves that exact semantic
identity. Candidate and Move-set counts are bounded by the compiler's
densification limit.

The greedy scheduler and the small-graph exhaustive oracle share one resource
feasibility predicate and lexicographic objective. Both enforce deadlines,
buffer credits, interference budgets, Move closure, and the densification
bound. The exhaustive search is intentionally capped and is a test oracle, not
a production hot-path solver.

`structural_signature()` supports every Phase 5 frozen descriptor. Rust reads
the same descriptors through typed `#[pyo3(attribute)]` fields and
`validate_adaptive_flow_plan()`. The Rust validator independently checks the
nested placement plan, exact Move equality, key/reference integrity, canonical
order, fallback semantics, and all resource bounds.

## Python runtime ownership

`tllm.runtime.scheduling.phase5_adaptive` owns mutable integer EWMA estimates,
per-step reservations, and typed admission decisions. Tensors, callbacks,
queues, observations, and mutable profile state never enter Rust or structural
signatures. Preparing a plan seeds interference EWMAs from its frozen profile
descriptors and applies their integer alpha and bounds to later observations.
Static mode ignores EWMA state and always returns the deterministic
compiler-approved schedule for the same resource context.

`AdaptiveRuntimeBinding` is a cold-path input. It compiles against the exact
Phase 3 plan and maps only explicitly migrated legacy flow ids to typed
`WorkItemDesc` values. `PlanResources` publishes the physical, placement, and
adaptive plans and the Python scheduler atomically at the existing prepare safe
point. A step keeps the epoch snapshot it acquired, and reservations reset only
when that prepared step begins.

Every selectable non-empty Move set also requires an `AdaptiveTransferBinding`
that names its typed producer, consumer, value, and source/destination bundle
entries. The cold validator requires exact coverage and an exact match to one
frozen Phase 3 route. The current production executor supports one route per
Move set, using the Phase 3 executor's direct one-hop copy or pinned-host
two-hop D2H/H2D chain. It does not infer payload entries or synthesize a route.

The deferred queue accepts typed `RUN`, `KEEP`, and `DROP` decisions. An
admitted execute action and an authoritative fallback both invoke the existing
consumer callback, preserving the legacy semantic oracle. A rejected optional
item stays queued while its approved deadline remains reachable and is shed
only when the deadline forces resolution. Required work falls back and is
never dropped. Flows without an `AdaptiveFlowBinding` remain an explicit legacy
boundary and retain the previous queue behavior.

For an admitted action with a Move set, the exact dependency-closed frozen
subset is submitted before the consumer callback. The queue returns `KEEP`
while its copy fences are pending and runs the callback only after completion.
An old plan epoch remains retained across safe-point publication until both the
copy and its queued callback handoff resolve; a pending transfer at its hard
deadline fails fast instead of allowing borrowed source storage to be reused.
Explicit `synchronize_esamp()` uses the same typed selector instead of bypassing
it with an unconditional queue flush. It forces admission to choose an approved
run, fallback, or shed action, then polls that action's individual transfer
fences until the callback handoff completes; it does not call a global CUDA
synchronization primitive.

## Support boundary

Phase 5 admission can only use compiler-approved actions, deadlines, buffer
credits, interference budgets, and Phase 3 Moves. It does not synthesize a
transfer, rewrite a training algorithm, alter a consumer callback, or migrate
an unbound flow. Runtime owners provide integer latency/interference
observations through the scheduler API; no hidden profiler or cross-process
policy discovery occurs on the token hot path.

The CPU exhaustive oracle validates small candidate graphs. GPU workload and
algorithm parity remain properties of the migrated Phase 2/3 execution region
and its workload-specific gates; Phase 5 does not claim throughput improvement
by itself.

## Validation

Run the repository environment explicitly:

```bash
/workspace/pllm/.venv/bin/python -m pytest -q
/workspace/pllm/.venv/bin/pyright -p pyrightconfig.strict.json
cargo test --manifest-path rust/tllm_core/Cargo.toml
cargo fmt --manifest-path rust/tllm_core/Cargo.toml -- --check
git diff --check
```

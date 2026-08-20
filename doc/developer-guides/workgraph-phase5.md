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
`allow_shedding=False`. Unknown workload or shape buckets use the single
approved unknown-bucket fallback, and explicit runtime flow bindings must share
that fallback's semantic identity. Candidate and Move-set counts are bounded by
the compiler's densification limit.

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

The deferred queue accepts typed `RUN`, `KEEP`, and `DROP` decisions. An
admitted execute action and an authoritative fallback both invoke the existing
consumer callback, preserving the legacy semantic oracle. A rejected optional
item stays queued while its approved deadline remains reachable and is shed
only when the deadline forces resolution. Required work falls back and is
never dropped. Flows without an `AdaptiveFlowBinding` remain an explicit legacy
boundary and retain the previous queue behavior.

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

# Auxiliary Workload Compiler

tLLM models vLLM as an externally owned, pinned main workload. A backend
exports only observable lifecycle anchors, fixed regions, resource
reservations, and visible values. It does not export a reorderable copy of the
full model graph.

## Public frontend

`tllm.workload` provides `@workload`, `@task`, `State`, typed value references,
state effects, timing windows, and optional affine domains/accesses. `@task` is
the only node declaration. An opaque CPU, I/O, or backend implementation is a
typed `ExternImplementation` candidate on the same decorator, not a second
program language.

`State` declares scope, request identity, parameter ownership, optimizer, and
version transition. REQUEST state must have an explicit `request_key`. A
transition without an optimizer or explicit version conversion is rejected.
Exact recurrence is the default. Jacobi must be selected explicitly with
`ApproximationPolicy.JACOBI` and changes the workload semantic key.

`declare_workload(instance)` returns a `WorkloadDeclaration`. Its `ir` is pure
metadata and has deterministic JSON and a SHA-256 semantic key. Runtime
callables stay in the declaration's executor table and are never serialized.

## Middle-end contracts

- `MainWorkloadTimeline`: pinned backend regions, observable anchors, graph
  boundaries, capabilities, reservations, and startup-time static headroom.
- `ScheduleGraph`: auxiliary tasks, typed input/output ports, values, versions,
  dependencies, effects, timing, implementation candidates, and static memory
  requirements.
- `ScheduleDecision` schema v2: the fine-grained execution order and schedule
  tree plus authoritative `ExecutionRegion` records, cross-Region
  materializations and lifetimes, memory plan, cross-Region fences, cost
  breakdown, explanations, artifact keys, and signature.

Rust derives each Region's typed input/output descriptors and ordered task
invocations from the fine-grained task ports. Every invocation parameter and
return value refers to an opaque stable value-binding ID; Region boundary
bindings cover provided values, typed hidden locators, built-in step IDs,
sinks, and cross-Region materializations. All descriptors are covered by the
artifact key and decision signature and independently recomputed by the Rust
validator. Rust represents input/output/effect/implementation/state/layer and
Region binding kinds as closed serde enums, so unknown variants fail during
JSON decoding and planner matches are exhaustive. `step_id()` is a distinct
DSL/IR variant; `provided("step_id")` remains an ordinary named provider and
cannot acquire built-in semantics from its spelling. Indexed hidden locators
are parsed once by Rust and emitted as signed numeric `layer_index` values.

`RegionCaptureMode` is a Rust closed enum in the signed Region record. It is
the sole capture contract presented to lowering: `eager_only` and any
capture/replay-capable variants are selected and capability-checked by Rust.
Python must not infer graph policy from task classes, stream names, or a
per-task capture partition.

Python does not inspect `TaskIR`, infer port kind from names, parse locator
strings, or reconstruct internal/materialized result edges on the execution
path. It mechanically dispatches Rust-signed Region enums and opaque binding
IDs. Remaining Python checks validate API/wire objects, callable return ABI,
and runtime Tensor shape/dtype/device resources; they do not select scheduling
semantics. Graph replay style is bound once when a `RuntimeGraph` resource is
registered rather than inferred on every replay.

`compile_schedule(graph, timeline, options)` and `validate_schedule(...)`
serialize only primitive JSON across the Rust FFI. New decisions must use
schema v2 and contain Regions; schema v1 remains parseable only for old golden
fixtures and is rejected by execution validation. Memory is a hard constraint
using the smaller non-zero value of user budget and backend static headroom.
Missing capabilities, invalid deadlines, dependency cycles, invalid Region
coverage, artifact mismatches, and memory overflow fail closed.

The granularity planner analyzes the fine-grained task graph but emits
coarse-grained Regions for execution. It starts with the largest deterministic
Region in topological order and splits only when anchor windows do not intersect, an
effect/version or typed-extern boundary must remain observable, graph safety
differs, or a hard memory constraint requires the explicit serial fallback.
Values consumed inside one Region stay as graph edges. A value crossing a
Region boundary gets an explicit materialization and lifetime; dependency
fences are emitted only between Regions. The structural cost records
materialization bytes, capture boundaries, peak live memory, launch groups,
and cross-Region fences. The middle-end does not predict kernel timing, assign
SMs, or query live free memory.

The Rust build pins `isl-rs==0.3.1`, which builds and statically links its
bundled isl 0.27 source. A complete connected region whose domains and accesses
are Presburger-representable is now lowered to isl validity, coincidence, and
proximity constraints. The resulting schedule and dependence relation are
stored in a typed affine region and revalidated independently. Other nodes stay
on the DAG scheduler. isl exposes legal order and parallelism; it does not
produce a per-task CUDA launch timetable.

## Region lowering

`PhysicalFlowPlan` contains one physical node per `ExecutionRegion`, not one
node per logical task. Python lowering only binds the signed Region artifact,
executor key, selected anchor, and cross-Region fence. It cannot split a Region
or choose a different order. Region artifact identities include the Region IR,
shape specialization, approximation-bearing workload key, backend timeline,
compile/capture policy, and a static Python/torch/CUDA/device fingerprint.

The process-local artifact materializer composes ordinary PyTorch tasks in one
Region callable and passes that complete callable to `torch.compile` once.
Result edges consumed within the Region remain local Python/FX values and are
not returned as materialized outputs. Uncapturable typed extern work remains
an explicit singleton Region and bypasses `torch.compile`. CUDA Graph capture
and replay are likewise Region-scoped. Runtime resources that cannot
materialize the selected artifact or capability must reject the plan rather
than silently execute a per-task eager fallback.

The eager runtime can submit different Regions at their signed anchors. It
retains only typed cross-Region outputs until their declared consumers run,
creates runtime events only for physical cross-Region fences, and invokes an
explicit provider/sink registry for process-local objects. Missing providers,
sinks, artifacts, effect/state bindings, or non-eager capture artifacts reject
the plan on the cold install path.

State execution uses an explicit process-local `StateVersionBinding` for each
Rust-typed state resource. The binding descriptor must exactly equal the
signed `ScheduleState`. A state-read Region must leave every active slot at the
observed version; state write/transition/version-publish effects must advance
each slot by the signed count. Missing, extra, non-monotonic, or mismatched
bindings fail closed.

## Migration status

Program declarations have a read-only adapter into the new dependence/effect
IR. Existing `BaseConsumer` and Program runtimes still execute production
workloads that have not entered physical takeover. The canonical startup
keyword is `make_llm(..., workloads=(...))`; an old consumer can be passed
directly as a migration workload. A new declaration must expose a
`legacy_consumer` adapter until its complete native executor set is available.
A pure declaration without an adapter can use the in-process eager Region
runtime when all process-local providers, sinks, and state versions are
explicitly bound.

MRv1/MRv2 workers report a pure-data timeline and one post-initialization
static-memory snapshot. The instance-scoped parent coordinator compiles and
signs the decision, including typed dependency fences and memory reservations,
then broadcasts it through string-method RPC. Workers independently validate
the signature and mechanically lower the immutable Region decision to a
`PhysicalFlowPlan` descriptor.

Native MP execution requires explicit `WorkloadWorkerSpec` records. Only an
importable factory name and pure JSON config cross the control plane; the
factory rebuilds declarations and process-local callbacks inside the worker.
After validation and materialization, every worker reports its actual ordered
`(region_id, artifact_key)` set. The parent requires exact equality with the
signed decision before accepting the epoch; Python callables are never
serialized.

This control-plane design is validated for the native ESamp exact eager Region
on EngineCore MP MRv1 and MRv2. That evidence covers worker reconstruction,
signature/artifact agreement, training, state-version advancement, and sampler
intervention; it is a correctness result, not a performance measurement.

`NativeWorkloadResources` is the explicit process-local resource contract for
a native declaration. It owns the declaration plus any named input providers,
output sinks, state-version snapshots, cold specialization callback, sampler
provider, statistics and enablement callbacks, residual-path bindings,
specialization key, and optional worker reconstruction recipe. These objects
are deliberately local: only `WorkloadWorkerSpec`'s importable factory and
pure JSON configuration cross an MP control plane. Worker reconstruction
cold-specializes against the parent-signed runtime specialization envelope,
re-declares and compares the complete schedule graph, then materializes and
reports its ordered Region artifact set. A specialization, graph, binding, or
artifact mismatch rejects the epoch.

`ESampConsumerConfig(implementation="region")` is a distinct native,
CPU-covered ESamp declaration, not a wrapper around the legacy consumer. It
owns replica state and its optimizer, cold-specializes the distiller, and has
two signed Regions: `predict_and_loss` reads signed hidden inputs and publishes
guidance, while the explicit `update` extern Region consumes their single
cross-Region materialized differentiable loss and advances the state version.
The sampler bridge consumes only same-step, metadata-matching guidance; it
does not recompute Region output. Source and target bindings use signed numeric
layer indices, including `-1` for the target layer, and are resolved through
the explicit residual-path contract. This mode rejects legacy-only features
such as per-request/model-bank state, compact capture lanes, row selection,
and non-`post_filter_exact` sampler backends. A native ESamp Region cannot be
co-installed with a legacy ESamp authority. During the current migration this
isolation is runtime-wide: a runtime with any native Region declaration also
rejects every legacy dispatch consumer, including an apparently unrelated
GTTA/TTRL sidecar. Legacy consumers do not yet declare signed ownership of
anchors, residual paths, state resources, and sampler hooks, so the runtime
cannot prove that two execution authorities are disjoint. The restriction may
be narrowed only after an explicit, decision-signed authority/ownership key is
implemented on both native and legacy contracts.

Region CUDA Graph materialization and replay infrastructure exists for
address-stable CUDA Tensor Regions with a static specialization. It explicitly
rejects Regions with state effects, so the stateful ESamp `predict_and_loss` /
`update` sequence is exact eager and non-capture today. In particular, the
model-backed native piecewise CUDA Graph path remains unverified and
incomplete. These are capability boundaries, not throughput claims.

Recirculation has two complete native Region declarations. The standalone
declaration consumes an explicitly published, finite CPU
`RecirculationExecutionRequest`; it owns the exact Rust-signed recurrence and
returns an explicit result sink. The experimental vLLM declaration binds a
Rust-signed typed recurrence stage at `Before("model.execute")` through
`BackendRegionBinderSlot`. Its row admission, grouping, operation order,
operands, effects, and `release_after` all come from Rust artifacts; the Python
backend adapter only binds layer-replay resources. The eager MRv1/MRv2 GPU
smokes exercise this native prefill path successfully. CUDA Graph, EngineCore
MP, decode recurrence, speculative decoding, PP/DP, and general model-family
support remain outside its verified boundary.

The older eager vLLM model-region adapter remains only as a numerical oracle and
compatibility fixture. Its MRv1/MRv2 comparison selected the same token, with
maximum BF16 logit difference `0.15625` and minimum cosine similarity
`0.99995947`; it does not own the native physical plan.

All remaining workloads other than native ESamp, standalone and vLLM
Recirculation, and Dummy retain a complete legacy consumer or sidecar adapter.
There is no partially authoritative native execution: a workload reaches
physical takeover only when its complete resource, binding, specialization,
and artifact contract is available; otherwise its existing adapter remains
the authority.

The standalone Dummy workload is now a complete `NativeWorkload`. Its typed
singleton extern Region uses the generic Region executor and explicit
`NativeWorkloadResources` for the layer-0 residual path, CPU-export worker,
synchronization callback, specialization identity, and worker reconstruction.
It neither registers nor consults a legacy `DummyConsumer`, and the old
Dummy-specific materialization branch has been removed. Missing bindings,
the `eager_only` capability, or a resolved residual resource fail closed.
`feedback_interval > 0` remains a legacy-only per-step drain feature and is
rejected by the native declaration; non-default legacy `capture_policy`
values are likewise rejected instead of being ignored.

## Current verification boundary

With `torch==2.13.0+cu129` and `vllm==0.27.1`, short GPU correctness smokes
cover the generic native Dummy, native Recirculation vLLM prefill, and native
ESamp declarations in exact eager, in-process MRv1 and MRv2. Recirculation
reported typed layer replay with zero fallback. The latest ESamp run recorded `loss_count=6`,
`update_count=6`, `state_version=6`, and `fallback_count=0`. Native ESamp
exact eager Regions also passed EngineCore MP correctness on both runners:
loss, update, state-version, and intervention counters were nonzero, with
`fallback_count=0` and `replay_count=0`. These establish execution and
state/sampler activity, not performance. The model-backed native piecewise
CUDA Graph path, Recirculation EngineCore MP/capture integration, and shape-miss
specialization epoch switching remain unverified or incomplete.
Throughput experiments are not claimed here.

Generic multi-task/value eager Region execution, signed internal task
invocations, typed hidden/provided/sink binding, state-version auditing, and
explicit worker reconstruction with exact artifact reporting are implemented
and covered by CPU tests. The full GPU correctness matrix and an ESamp paired
performance gate are not claimed.

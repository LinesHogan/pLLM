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

Packed eager inputs use nominal author descriptors rather than textual
relations:

```python
rows = packed_request_rows()
selection = packed_selection(state.selection(), rows=rows)

@task(
    inputs={
        "rows": rows,
        "source": packed_hidden(0, rows=rows),
        "target": packed_hidden(-1, rows=rows),
        "selection": selection,
    },
    effects=(read(selection),),
)
def train(rows, source, target, selection): ...
```

Each packed task declares exactly one `PackedRequestRowsInput`; packed hidden
and request-selection inputs must reference that exact object. The descriptor
may be reused by another task only when that task also declares it as its own
rows input. `packed_hidden` accepts an index or path locator at
`block_output`; it does not masquerade as an ordinary prefill/decode
`HiddenInput`. These declarations sign data provenance only and expose no
lane, priority, fusion, placement, or execution-order control.

## Middle-end contracts

- `WorkloadIR` schema v4: author declarations after deterministic lowering,
  including request-selection and packed-row provenance relations.
- `MainWorkloadTimeline`: pinned backend regions, observable anchors, graph
  boundaries, capabilities, reservations, and startup-time static headroom.
- `ScheduleGraph` schema v5: auxiliary tasks, typed input/output ports, values, versions,
  dependencies, effects, timing, implementation candidates, and static memory
  requirements.
- `ScheduleDecision` schema v7: the fine-grained execution order and schedule
  tree plus authoritative `ExecutionRegion` records, cross-Region
  materializations and lifetimes, memory plan, cross-Region fences, cost
  breakdown, explanations, artifact keys, and signature.
- `RuntimeHostCommand`: closed inline/wait/submit actions plus an optional
  Rust-derived optimizer-transition batch mode and artifact key. The mode
  cannot be authored by a workload and does not create a lane, priority, or
  execution-order policy.

Rust derives each Region's typed input/output descriptors and ordered task
invocations from the fine-grained task ports. Every invocation parameter and
return value refers to an opaque stable value-binding ID; Region boundary
bindings cover provided values, typed hidden locators, built-in step IDs,
sinks, packed request rows/hidden snapshots, and cross-Region materializations.
Packed hidden and request-state selection bindings carry the Rust-signed ID of
their same-task packed-row binding; runtime epoch/step values remain runtime
facts. All descriptors are covered by the
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
schema v7 and contain Regions; selected older schemas remain parseable only as
compatibility fixtures and are rejected by current execution validation.
Memory is a hard constraint
using the smaller non-zero value of user budget and backend static headroom.
Missing capabilities, invalid deadlines, dependency cycles, invalid Region
coverage, artifact mismatches, and memory overflow fail closed.

The granularity planner analyzes the fine-grained task graph but emits
coarse-grained Regions for execution. It starts from singleton Regions (one
task each) and runs an exact merge search: two Regions may merge only when
their anchor windows intersect, they share one non-extern implementation
kind, every effect/version and typed-extern boundary between them can be
internalized, and the merged partition remains topologically orderable.
Candidate partitions are compared by a lexicographic score —
materialization bytes, capture boundaries, peak live memory, Region count,
cross-Region fences, critical-path extension, then a deterministic
partition identity — under the hard memory budget. The search is exhaustive
up to 100,000 explored partition states and fails closed beyond that bound
or when no partition fits the budget. A signed `serial_fallback` marker
exists for explicit serial Regions and validation enforces
`allow_serial_fallback` against it, but the current exact planner never
emits it. Values consumed inside one Region stay as graph edges. A value crossing a
Region boundary gets an explicit materialization and lifetime; dependency
fences are emitted only between Regions. The structural cost records
materialization bytes, capture boundaries, peak live memory, launch groups,
and cross-Region fences. The middle-end does not predict kernel timing, assign
SMs, or query live free memory.

The Rust build pins `isl-rs==0.3.1`, which builds and statically links its
bundled isl 0.27 source. The original design report named Tadashi for the
polyhedral scheduling role; the implementation converged on isl's built-in
scheduler, which covers the required validity/coincidence/proximity
computation without an additional dependency. A complete connected region
whose domains and accesses
are Presburger-representable is now lowered to isl validity, coincidence, and
proximity constraints. The resulting schedule and dependence relation are
stored in a typed affine region and revalidated independently. Other nodes stay
on the DAG scheduler. isl exposes legal order and parallelism; it does not
produce a per-task CUDA launch timetable.

## Region lowering

`PhysicalFlowPlan` contains one physical node per `ExecutionRegion`, not one
node per logical task. Python lowering only binds the signed Region artifact,
executor key, selected anchor, and cross-Region fence. It cannot split a Region
or choose a different order. Region artifact identities include the signed
Region record, the schedule graph (which carries each workload's
approximation policy key), the backend timeline, and the complete canonical
compile options: memory budget, compile/capture policy, shape
specialization, serial/side-stream flags, and a static
Python/torch/CUDA/device compiler fingerprint.

The process-local artifact materializer composes ordinary PyTorch tasks in one
Region callable and passes that complete callable to `torch.compile` once.
Every materialization compiles under a fresh per-Region code identity, so
Dynamo recompile caches (which are per code object) never accumulate across
Regions in one process; global Dynamo lifecycle knobs are never touched,
leaving the externally owned main model's compilation state intact.
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
`PhysicalFlowPlan` descriptor. The coordinator may also persist signed
decisions under a configurable directory keyed by the unchanged compilation
cache key; a persisted entry is untrusted input and is accepted only after
the Rust wire validation re-derives its key and signature, failing closed on
any mismatch.

Native MP execution requires explicit `WorkloadWorkerSpec` records. Only an
importable factory name and pure JSON config cross the control plane; the
factory rebuilds one complete typed `WorkloadRuntimeBinding` inside the worker.
After validation and materialization, every worker reports its actual ordered
`(region_id, artifact_key)` set. The parent requires exact equality with the
signed decision before accepting the epoch; Python callables are never
serialized.

This control-plane design is validated for the native ESamp exact eager Region
on EngineCore MP MRv1 and MRv2. That evidence covers worker reconstruction,
signature/artifact agreement, training, state-version advancement, and sampler
intervention; it is a correctness result, not a performance measurement.

`WorkloadRuntimeBinding` is the closed process-local contract for a native
declaration. It contains immutable port, state, and backend registries plus
explicit service, sampler, and worker variants; absence is an empty registry
or a `No*` variant, never a nullable callback field. These objects are
deliberately local: only `WorkloadWorkerSpec`'s importable factory and
pure JSON configuration cross an MP control plane. Worker reconstruction
cold-specializes against the parent-signed runtime specialization envelope,
re-declares and compares the complete schedule graph, then materializes and
reports its ordered Region artifact set. A specialization, graph, binding, or
artifact mismatch rejects the epoch.

`ESampConsumerConfig(implementation="native")` is a distinct native
declaration, not a wrapper around the legacy consumer. Replica mode owns one
distiller and optimizer; request mode owns a generic fixed-capacity
`RequestTrainableState` keyed by stable parent request ID. Parallel child
samples therefore resolve to one signed selection and one loss/update/version
transition per parent. Fresh logical admission reinitializes fixed-address
slot parameters and optimizer state, while typed generation/slot receipts
prevent ABA reuse and retain bounded cleanup evidence. Source and target
bindings use signed numeric layer indices, including `-1` for the target
layer, and are resolved through the explicit residual-path contract.

The initial request-state mode is eager-only. Per-request sampler intervention
uses a workload-independent compact hidden payload with complete step identity,
sorted affected rows, and a full-to-compact device map. A frozen selection
before-version gates version-zero rows without moving that policy into the
sampler adapter. A later PTTS-0 tranche added the generic
`StateStorage.FIXED_BATCHED` representation: one capacity-first module, one
per-slot AdamW state, and the same arena-owned key/version/pin/retire
lifecycle. ESamp's compatibility model-bank inputs select the model-bank
algorithm and its first-row-per-key loss; that state's typed descriptor
declares the capacity-first layout without adding a scheduler policy. Rust signs
the storage fact but continues to derive packed request Regions as eager until
compact fixed-buffer provenance exists. SVD initialization, compact capture,
and non-`post_filter_exact` sampler backends remain legacy-only. The resident path passed MRv1/MRv2 ×
in-process/EngineCore-MP with four real updates, two parent slots retired at
version 2, empty live mappings, and zero fallback/replay in every cell. The
intervention matrix observed one modifier state after the first native update
and none for version-zero rows.

The corrected independent-process request-state differential is numerically
green on MRv2 eager in-process. Legacy and native use the same short text
prompts, engine/sampling seeds, and case fingerprint; tokens match exactly both
without intervention and at `beta=0.5`, loss/update counts are `4/4`, the
intervention count is `1/1`, and aggregate loss is exact after the
capture-disabled request optimizer was aligned to eager AdamW. The superseded
six-versus-four result had accidentally stringified a
`TokensPrompt` through a text-only legacy helper and did not compare the same
input.

The prefill follow-up now has a neutral CPU-executable foundation:
`PackedRowPlanInputs` lives in the contracts layer, and
`PackedPhaseRowStaging` expands the Rust packed-row plan into immutable rows
carrying plan epoch, step, raw backend ID, stable `RequestIdentity`, token
offset, and closed prefill/decode phase. Decode plan ordinals are explicitly
not sampler-row links. A single active epoch and retired-step high-watermark
prevent `(epoch, step)` ABA reuse without retaining unbounded history. Packed
input kinds and same-task row provenance are now signed by WorkloadIR v4,
ScheduleGraph v5, and Decision v7. `NativeWorkloadExecution` stages one exact
row/snapshot batch, the Region resolver binds signed packed inputs, and packed
request-state selections come only from the signed row phase of a
`PackedPhaseRowBatch`, retaining each selected row's original packed index.
MRv1/MRv2 packed hidden capture and explicit `logits_indices`-derived sampler
row links now bind that same batch, and ESamp consumes them directly. The
initial link adapter transfers one
active eager index prefix to CPU, so it is correctness evidence rather than
aligned-throughput evidence. ESamp remains decode-only; non-empty prefill rows
do not admit or update request state.

The matching eager packed-hidden foundation owns an arbitrary non-empty set of
exact resource-keyed snapshots under one canonical row batch and an explicit
cold capacity/shape/dtype/device specialization. Backend inference tensors are
cloned into ordinary service-owned storage, and reads return independent
clones; bounded finish/abort high-watermark semantics prevent step ABA without
invalidating an older active step.
The schema names this service through signed packed input bindings, and an
independent exact module-path manifest is reconstructed and compared in the
parent, EngineCore worker, and local execution gate. The CPU resolver is wired;
MRv1/MRv2 eager hooks now stage the same canonical rows and publish exact
resource-keyed snapshots before the signed model-post event. Packed
capabilities are not advertised in piecewise/full graph modes. This is not a
compile/CUDA-Graph path.

A native ESamp Region cannot be co-installed with a legacy ESamp authority.
During the current migration this
isolation is runtime-wide: a runtime with any native Region declaration also
rejects every legacy dispatch consumer, including an apparently unrelated
GTTA/TTRL sidecar. Legacy consumers do not yet declare signed ownership of
anchors, residual paths, state resources, and sampler hooks, so the runtime
cannot prove that two execution authorities are disjoint. The restriction may
be narrowed only after an explicit, decision-signed authority/ownership key is
implemented on both native and legacy contracts.

Region CUDA Graph materialization and replay infrastructure exists for
address-stable CUDA Tensor Regions with a static specialization. Request-state
read/transition Regions are explicitly signed capture-disabled; graph
ineligibility is not counted as a runtime fallback. Model-backed request-state
capture remains unverified and incomplete. These are capability boundaries,
not throughput claims.

Recirculation now has one production `RecirculationWorkload`. It binds a
Rust-signed typed recurrence stage at `Before("model.execute")` through the
generic backend-adapter seam. Rust IR constructors own schema and shape
metadata; the Rust stage family owns canonicalization, signing, wire loading,
packed-row admission, and exact artifact selection. Consumer-local Python only
declares the Recirculation operations and binds signed coordinates to layer
replay and mix resources. The eager MRv1/MRv2 GPU smokes exercise this native
prefill path in-process and under EngineCore MP, including mixed packed-row
phase selection. CUDA Graph capture, decode recurrence, speculative decoding,
TP/PP/DP, normal hook retirement, and general model-family support remain
outside its verified boundary. The old standalone and vLLM-specific workload classes,
legacy Rust planner, and model-region adapter have been removed.

FTTT, GTTA, In-Place-TTT, TLM, TTRL, ReFT, OverRIDE, Recirculation, and ESamp
have native workload definitions. Only ESamp retains explicitly named
legacy/Program benchmark baselines while its final parity and throughput gate
remains open.

The public Dummy entry now routes to a clean-room CPU vertical workload. Its
semantic graph, Rust schedule decision, physical plan, and execution are one
closed path; the old `tllm.workload`/runtime implementation and its stream,
transfer, worker, and GPU smoke entry have been deleted. Its current evidence
is the CPU vertical oracle only, not GPU or EngineCore-MP coverage.

## Current verification boundary

With `torch==2.13.0+cu129` and `vllm==0.27.1`, short GPU correctness smokes
cover native Recirculation, OverRIDE, TTRL, ESamp, and TLM. Dummy is covered
only by its clean-room CPU vertical oracle. TLM's MRv2 eager smoke
performs real HF/PEFT training and installs the published generic LoRA version;
Recirculation
passed typed layer replay in-process and under EngineCore MP on MRv1/MRv2;
each cell replayed two requests and seven prompt rows with zero fallback.
OverRIDE passed MRv1/MRv2 × in-process/EngineCore-MP with fixed per-runner token
goldens, cumulative state versions `[2, 4, 6]`, exact final-iteration no-op,
finished-request cleanup, and zero fallback. The generic mapped-generate
boundary pauses only admission until the complete typed request batch is
queued; it does not carry a workload scheduling policy.
TTRL's four eager topology cells atomically consumed hidden rows plus a typed
reward intent, trained exactly once, emitted independent reward/retry/barrier
and aggregate outputs, executed the version-pinned retry, and reported zero
fallback. Its verifier and hidden rows are deterministic smoke inputs rather
than a policy-training benchmark. The latest ESamp run recorded `loss_count=6`,
`update_count=6`, `state_version=6`, and `fallback_count=0`. Native ESamp
exact eager Regions also passed EngineCore MP correctness on both runners:
loss, update, state-version, and intervention counters were nonzero, with
`fallback_count=0` and `replay_count=0`. These establish execution and
state/sampler activity, not performance. The model-backed native piecewise
CUDA Graph path, Recirculation capture/hook-retirement integration, and
shape-miss specialization epoch switching remain unverified or incomplete.
Throughput experiments are not claimed here.

Generic multi-task/value eager Region execution, signed internal task
invocations, typed hidden/provided/sink binding, state-version auditing, and
explicit worker reconstruction with exact artifact reporting are implemented
and covered by CPU tests. The full GPU correctness matrix and an ESamp paired
performance gate are not claimed.

## Revised commitments relative to the original report

Several items promised by the original design report were deliberately
re-scoped by this architecture. They are recorded here so that readers can
distinguish "not done" from "withdrawn".

- **Backend ability classes (original §4.4) — revised.** The report asked
  backends for four free-form abilities: graph traversal, anchor node
  insertion/removal, tensor metadata queries, and buffer
  alloc/free/in-place reuse. The shipped contract replaces them with the
  `BackendDriver` protocol plus signed capability records
  (`BackendCapabilities`, `ModuleWeightOverlayCapabilities`,
  `ModelRegionCapabilities`, and the typed replay handle protocol).
  Backends export capabilities and observable anchors, not a reorderable
  model graph.
- **Profiling feedback loop (original §9.3) — partially done, loop not
  closed.** The runtime collects per-Region CPU/CUDA timings with EWMA and
  exposes a read-only cost report. That report is a diagnostic estimate: it
  excludes side-stream time and sums default-stream Regions, so it is not a
  strict main-path cost backfill. Nothing feeds measurements back into
  planner decisions.
- **Wavefront controls (original §8.3) — withdrawn.** No tiling,
  interchange, or skewing is applied to affine regions, and no
  serial/wavefront selector exists. The `Serial`/`Wavefront` labels in the
  recurrence compiler name ordered-row fusion policies, not polyhedral
  schedules. The design position is CUDA Graph capture plus hardware
  scheduling rather than extracted wavefronts.
- **Joint graph reordering — shrunk.** The main workload graph is
  externally owned and pinned; reordering applies only to auxiliary-task
  Region partitioning, never to a joint graph spanning the model.
- **Effectful Region numerical parity — withdrawn.** Regions carrying
  mutation effects run under the signed state-version audit on every
  execution; numerical comparison against a serial oracle is not
  implemented for them and is no longer promised. The audit's guarantee
  boundary is exact but narrow: it proves version-increment counts and
  slot identities match the signed effects, and it cannot detect a
  numerically wrong parameter update that still bumps versions.
  `EXTERN_IO` effects are exempt from the audit entirely.
- **Approximation signing boundary — clarified.** Only the approximation
  *policy* is signed into the schedule graph wire (and changes the
  workload semantic key). `ApproximationCriteria(rtol, atol)` is
  deliberately host-side: it stays outside the signed artifact identity
  and is consulted by the first-call parity gate at materialization time.
  The signed planner additionally refuses to merge tasks from workloads
  with different approximation policies into one Region, so a non-exact
  tolerance can never wash over an exact workload's outputs.

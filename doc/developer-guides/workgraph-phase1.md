# WorkGraph Phase 1

This guide tracks the typed WorkGraph front end that builds on the frozen
Phase 1a compatibility shadow. WorkGraph remains a control-plane artifact;
`DispatchPlan` and `ExecutionPlan` are still authoritative for execution.

## Milestone 1: Public Contracts

The public API exports `flow`, `extern`, and `BeforeNextUse` from both `tllm`
and `tllm.workgraph`.

`@flow` is transparent during normal Python calls. Decoration and normal calls
do not capture, compile, or inspect a graph. Only `.capture(...)` enters the
control plane, and it validates the original function signature without
executing the function.

The current milestone deliberately produces a conservative opaque artifact:

- `CapturedFlow.status` is `opaque_barrier`;
- `node_kind` is `barrier`;
- mobility is `pinned`;
- graph safety is `eager_only`;
- effects contain `opaque`;
- a source-aware typed diagnostic explains the fallback.

This is the required failure behavior. Milestone 2 replaces it only when
`torch.export.export(..., strict=True)` and the effect analysis both prove a
complete leaf graph. There is no `strict=False` success fallback and no
partially captured graph.

The frozen public descriptors are `CapturedFlow`, `FlowCaptureDiagnostic`,
`FlowCaptureCode`, `SourceSpan`, `ShapeGuard`, `ArtifactKey`, and
`PortBindingDesc`. `ArtifactKey` is separate from `NodeKey` and `ExecutorKey`;
it uses callable module and qualified name, never an object address, source
line, Tensor representation, runtime handle, or request id.

`PortBindingDesc` maps a structured `PortId` or `PortAddressDesc` to one typed
input/output pytree path. Raw locator strings and duplicate leaf mappings are
rejected. The closed-world structural serializer accepts these descriptors
without accepting arbitrary mappings or runtime objects.

`@extern` is also transparent at call time. Its control-plane contract is
derived from resolved input and output annotations and requires typed
`Effect`, `StateScope`, `Device`, and `Capability` values. Raw string policy
and missing annotations fail during decoration. No `@task`, `@stream`, or
`@buffer` decorator is introduced.

## Milestone 2: Strict Capture And Inference

`.capture(...)` now runs a source/closure preflight before calling PyTorch. A
supported pure Tensor program is wrapped in an internal `nn.Module` and passed
to `torch.export.export` exactly once with `strict=True`. Decoration and normal
calls still do not import or invoke the capture path.

Successful captures retain only frozen scalar descriptors:

- tensor shape, dtype, typed CPU/CUDA device, layout, and stride from
  FakeTensor metadata;
- SSA producer/input/user ordinals and logical value version;
- stable alias sets propagated from operator schema alias annotations;
- bounded static/range shape guards;
- a leaf `ArtifactKey` and ordered ATen `LeafOperatorDesc` records.

FakeTensor, Tensor, FX node, operator objects, and `ExportedProgram` never enter
`CapturedFlow` or its structural signature. The wrapper owns successful
`ExportedProgram` handles in a private control-plane specialization cache.

The preflight rejects Tensor data escape through `.item()`, `.numpy()`, or
`.tolist()`, host I/O (including referenced aliases), unregistered mutable
closure state, unsupported runtime-object inputs, and obvious Python/in-place
mutation. Postflight scans every `call_function` operator schema, including
nodes with no users. Any mutable schema or unknown schema discards the whole
leaf and produces an opaque barrier. Functionalization is not currently used
to claim mutation equivalence.

Dynamic shape declarations use frozen `DynamicDim` descriptors. The current
front end accepts bounded positional Tensor dimensions and rejects an
unbounded dimension before export with `WG206_UNBOUNDED_DYNAMIC_SHAPE`.
Unknown symbolic bounds returned by PyTorch are rejected in the same way.

## Milestone 3: Hierarchy And Compatibility Lowering

Captured flows now lower into a frozen `CaptureHierarchy` sidecar with
`schema_version=2`. Each `@flow` specialization owns one root flow region,
one or more task regions, and one exported or opaque leaf region. The existing
`SemanticGraph` and `DataCentricIR` remain the version-1 compatibility
projection used by `DispatchPlan` and `ExecutionPlan`; ATen operators are
retained only in `LeafArtifactDesc` and never become schedule nodes.

The accepted exported leaf is schema-validated ATen Tensor dataflow. Each
`LeafOperatorDesc` retains its typed namespace/name/overload, SSA ordinals,
stable module path, effect/capability facts, and typed operator constants and
attributes (`OperatorAttributeDesc`) at deterministic paths. The successful
`ArtifactKey` includes a lowercase SHA-256 structural digest over these
operator facts, captured Tensor value/use-def/alias/effect facts, and shape
guards; the digest is part of the artifact identity and structural signature.
Opaque barriers use a digest of their typed barrier status, diagnostics, and
effect facts instead of retaining a partial export.

Initial task partitions are deterministic runs of operators with identical
typed module path, operator kind, effect, and capability facts. A change in
module path, effect, or capability records the corresponding typed
`PartitionBoundary`. ATen schema tags marked `nondeterministic_seeded` infer
`Effect.RNG` and `Capability.REPLAY_SAFE`; the containing flow is marked
`REPLAY_STATEFUL`. Other accepted schema-validated ATen operators currently
infer `PURE` and `CAPTURE_SAFE`.

Module paths are retained only from stable export metadata. A Python module
object or mutable module state captured through a function closure is an
unregistered closure boundary and fails closed as an opaque capture; it is not
silently converted into a schedulable module task. The custom-op boundary is
represented in the schema, but the current capture path rejects every
non-ATen operator. Typed `@extern` contracts are not yet consumed as
`CUSTOM_OP` leaf records, so typed extern custom-op partitioning remains
unsupported until a typed registry exists. A strict-export identity graph with
no ATen calls still receives an empty exported leaf under one task region.

Runtime scalar inputs and scalar outputs are also fail-closed: without a typed
scalar SSA descriptor, capture rejects inputs before strict export and rejects
non-Tensor output leaves during lowering rather than dropping their ordinals
from the leaf dataflow.

`CaptureAttachment` connects an artifact to one existing `NodeKey` with the
exact `PortBindingDesc` records declared by its capture. A `PortId` must name a
selected output in the authoritative `ExecutionPlan`; a `PortAddressDesc`
must exactly equal a read or write address on the target consumer flow.
Unknown ports, missing or duplicate bindings, unknown nodes, and mismatched
anchors fail closed. No role, name, kind, or phase string is used to infer an
attachment.

The hierarchy and attachments participate in the structural signature, so a
sidecar-only change advances the shadow epoch and conservatively rebuilds the
owning compatibility projection. Explicit producer-port readers also produce
metadata-only logical live ranges from the producer `NodeKey` to the last
attached reader. These Phase 1 lease descriptors do not allocate physical
buffers, retain runtime storage, or introduce transfer operations: all leases
remain `materialized=False`, and both the compatibility and capture data views
require `Move=()`.

Python validates descriptor construction and plan bindings. Rust separately
reads the frozen hierarchy through `#[pyo3(attribute)]`, checks region and
artifact identity, task/leaf coverage and facts, attachment endpoints,
logical leases, and the rule that leaf operators do not enter the semantic
schedule. Neither validator receives Tensor, FakeTensor, FX nodes,
`ExportedProgram`, runtime plans, callbacks, streams, events, or handles.

`Phase1aPlan.explain()` now includes a JSON-safe `capture_hierarchy` section
with source, artifact, partition, effect, capability, attachment, and logical
lease facts. A plan without captured flows reports `capture_hierarchy: null`.

## Milestone 4: Phase 1 Closeout

The compatibility compiler now derives two frozen event traces on the cold
path. The legacy trace comes directly from the authoritative `DispatchPlan`
and `ExecutionPlan`; the shadow trace comes from `RuntimePlanBinding`. Their
ordered events cover producer hooks, consumer-flow hooks, fusion membership,
port-source bindings, and capture-resident producers. Compilation rejects an
ordinal mismatch with `WG208_TRACE_PARITY`. The equivalent traces and any
typed mismatch records are available in `Phase1aPlan.explain()` as
`event_trace`; no trace observer is installed in the token hot path.

Hierarchy explain output retains the existing display identities and adds
structured `ArtifactKey`, full source span, typed input/output port bindings,
shape/value/use-def facts, task boundaries, and the operator schema facts and
constants covered by each leaf. The existing reconciliation report continues
to state explicit Keep/Rebuild regions and reasons.

`build_lsp_snapshot()` is a read-only control-plane adapter over a frozen
`CapturedFlow` or `Phase1aPlan`. It produces frozen diagnostics, Tensor hover
facts, guards, and root/task/leaf region descriptors, with a JSON-safe outer
view for an editor client. Standalone captures use the same partition builder
as `CaptureHierarchy`, including RNG/effect boundaries. The adapter has no
command, code-action, capture, compile, execution, or runtime-recompile API and
does not retain Tensor, FX, export, callback, stream, event, or runtime-plan
objects.

The support matrix is covered for MRv1 and MRv2 in both eager and piecewise
graph modes with a real captured hierarchy and attachment. All four cases
remain shadow-only and leave the runtime plans unchanged. FULL mode for either
runner family remains the typed `WG120_UNSUPPORTED_GRAPH_MODE` legacy-only
case; it is not reported as eager or piecewise.

The Rust hierarchy validator and a closed-world Python differential oracle
agree on valid topology and systematic corruptions of schema, region and
artifact identity, operator/value facts, coverage, attachment binding,
logical lease, Move, and schedule-node constraints.

## Current Limits

- The accepted success subset is schema-validated ATen Tensor dataflow with static or
  finitely bounded positional dynamic dimensions. Unregistered custom,
  structured multi-result, and schema-less operators remain conservative
  barriers.
- Mutation is detected and retained as a typed write effect, but no mutation
  is yet proven equivalent and accepted as a captured compute leaf.
- `@extern` contracts are typed but are not yet partitioned into an opaque leaf
  region inside a captured flow.
- Hierarchy reconciliation currently rebuilds the flat owning projection; it
  does not yet retain or rebuild individual `ArtifactKey` regions.
- No WorkGraph artifact affects runtime execution or the token hot path.
- FULL graph mode remains typed unsupported in the Phase 1a compatibility
  compiler.

Phase 1 is complete at the control-plane boundary. The historical ESamp
initialization-memory regression was traced to generic vLLM hook setup calling
the consumer's private resource initializer during `load_model.post`, before
vLLM finished its KV-cache memory profile. That call is removed and a runtime
lifecycle regression test requires ESamp training resources to remain lazy
until the first complete port bundle. The current host cannot communicate with
the NVIDIA driver, so an aligned `gpu_memory_utilization=0.5` rerun is not yet
GPU-verified. Phase 2 may build and validate typed physical plans and CPU-side
lifecycle contracts, but automatic GPU execution and performance claims remain
unsupported until that aligned gate passes. The existing runtime plans remain
authoritative until then.

## Verification

Use the repository virtual environment:

```bash
/workspace/pllm/.venv/bin/python -m pytest -q test/workgraph
/workspace/pllm/.venv/bin/python -m pytest -q
/workspace/pllm/.venv/bin/pyright -p pyrightconfig.strict.json
cargo test --manifest-path rust/tllm_core/Cargo.toml
cargo fmt --manifest-path rust/tllm_core/Cargo.toml -- --check
git diff --check
```

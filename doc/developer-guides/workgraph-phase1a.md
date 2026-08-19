# WorkGraph Phase 1a

This guide documents the WorkGraph compatibility IR that is currently
implemented. Phase 1a is a startup/control-plane shadow representation. The
existing `DispatchPlan` and `ExecutionPlan` remain authoritative for runtime
execution; compiling a WorkGraph does not replace or re-schedule either plan.

## Scope

Phase 1a lowers the existing typed declarations into two immutable layers:

1. **Semantic IR**: `SemanticGraph` and `SemanticNodeDesc` describe provider,
   consumer-flow, and backend-anchor nodes, typed port accesses, timing,
   effects, executor identity, and legacy boundaries.
2. **Data-Centric IR**: `DataCentricIR` and `ExecutionNodeDesc` describe logical
   values, accesses, backend-owned residence, leases, and the correspondence
   to semantic nodes.

Both layers use `schema_version=1`. The descriptor types are frozen and
slotted. `PortAddressDesc` and `PortId` are reused as structured port
references; port identity is never reconstructed from an escaped locator
string.

The compatibility compiler consumes real `DispatchPlan` and `ExecutionPlan`
objects. It checks parity for flow order and selected hooks, fusion groups,
producer order and hooks, capture-resident producer ids, and producer port
sources. The resulting
`Phase1aPlan` exposes the immutable compiled descriptors, binding metadata,
legacy-boundary notes, and reconciliation diff.

`RuntimePlanBinding` is part of the structural signature alongside both IR
layers. A binding-only change, such as a selected hook or capture-resident
producer set, therefore rebuilds the shadow epoch even when the semantic and
data descriptors themselves are unchanged.

## Stable FFI Boundary

Stable keys are structural:

```text
consumer_flow/<consumer_id>/<ordinal>
producer/<producer_id>/0
anchor/<backend>/<runner>/<graph_mode>/<hook>
```

`NodeKey` and `ExecutorKey` do not contain object addresses, request ids,
Tensor representations, source line numbers, or runtime payloads. The
canonical signature is SHA-256 over sorted canonical JSON. The serializer is
closed-world: it accepts the declared frozen descriptors, primitive values,
Enums, `PortId`, `PortAddressDesc`, and tuples. Unknown objects, mappings, and
runtime/private backend values are rejected instead of being stringified.

`PortAddressDesc.label` is display text only. Address-backed `ValueDesc`
identities use a SHA-256 digest over the complete tagged address descriptor,
including every locator path, type, and value. Delimiters in a locator value
therefore cannot merge two compiler values.

The Rust boundary receives only CPU-visible descriptors. It does not receive
Tensor, CUDA stream/event, pointer, vLLM runner, or consumer objects.

The boundary is fail-closed. Python snapshots are restricted to the vLLM
0.27.x backend family and MRv1/MRv2 (plus the explicit unresolved startup
shadow); anchor keys use a fixed hook vocabulary. Layer-owned hooks such as
`layer.post` are recorded as legacy runtime boundaries, while an unknown
selected hook is a compile error. The Rust validator also reads the structured
`PortId`/`PortAddressDesc` reference on values and accesses and checks every
execution-node residence/lease key against the descriptors it names.

## Anchors And Provenance

Only `AnchorNode` records carry an `AnchorDesc`. A native attachment creates
an explicit `AnchorNode -> WorkNode` `anchor` edge at one stable backend hook.
This is a single attachment relation, not the future design's "between two
anchors" interval constraint; Phase 1a does not infer a second lifecycle bound
or schedule within such an interval.

A provider with more than one hook cannot be faithfully represented by one
attachment. It, and a provider or flow at a layer-owned hook, remains an
explicit `legacy_boundary` with `opaque_legacy_effect`; its complete hook set
is retained in `RuntimePlanBinding` and the lowering notes. Likewise,
kind/phase producer matching only emits a `control` edge and a
`coarse_kind_phase_binding` note. It never claims exact `data` provenance
between a producer `PortId` and a consumer address.

## Legacy And Effects

A pinned flow observing `at_declared_window` can be represented as a native
semantic node. Existing consumer declarations do not yet provide typed state
or effect contracts, so the conservative default still records
`opaque_legacy_effect` even when the timing boundary is pinned.

Deferrable, advanceable, or free flows are represented as
`legacy_boundary` unless `strict_native` is requested. An existing flow using
`latest_at_execution` or `stale_ok` is also conservative legacy work when it
reaches the WorkGraph lowering boundary. The current runtime placement layer
may reject those movable observations earlier because there is no execution-time
version resolver; that rejection is intentional rather than a silent fallback.

## Backend Matrix

The Phase 1a backend snapshot and logical anchor vocabulary cover these
metadata combinations:

| Runner family | Eager | Piecewise |
|---|---|---|
| MRv1 | supported metadata snapshot and fixed anchors | supported metadata snapshot and fixed anchors |
| MRv2 | supported metadata snapshot and fixed anchors | supported metadata snapshot and fixed anchors |

The startup shadow snapshot uses an unresolved runner and eager mode until a
concrete backend is selected. FULL CUDA-graph mode is outside Phase 1a and is
not silently treated as eager or piecewise. For FULL, the compiler preserves a
typed `WG120_UNSUPPORTED_GRAPH_MODE` diagnostic and leaves the legacy plans in
control; it deliberately skips Rust WorkGraph validation for that
legacy-only diagnostic path.

## Data And Reconciliation Limits

Phase 1a represents values as backend-owned, same-worker residence. `MoveDesc`
is not generated, and a `LeaseDesc` is metadata only with
`materialized=False`; no stream, event, storage address, or physical buffer is
held by the IR.

Structural changes are reconciled without touching runtime resources:

- the first plan starts at epoch `1`;
- an identical structural signature produces `KeepRegion` and keeps the epoch;
- a changed signature increments the epoch and produces `RebuildRegion`;
- the reconciler does not publish RCU plans, retire CUDA resources, or install
  fences.

The snapshot is control-plane metadata. Registration first records an
unresolved startup shadow; after the vLLM adapter provides a typed resolved
backend, the runtime compiles a concrete shadow from the existing authoritative
plans. When fusion placement still needs the runner's effective graph mode,
the first existing dispatch-side placement refresh completes before the
pending shadow snapshot is compiled once. Steady-state token dispatch does not
read, traverse, or compile the WorkGraph. This lifecycle does not replace or
execute `DispatchPlan` or `ExecutionPlan`.

`WorkGraphExplain` renders stable node, value, residence, lease, binding,
diagnostic, and reconciliation metadata as JSON-safe output. Its
`semantic_edges` records use `{source, target, kind}` objects. It is a compile
report, not a runtime trace.

## Explicit Non-Goals

The current implementation does not provide `@flow` capture, torch export or
FX lowering, automatic placement execution, data movement, stream/event
management, buffer allocation, FULL CUDA-graph support, TP>1, distributed
state/collectives, online RCU reconciliation, or a performance improvement.
It adds no steady-state WorkGraph decision or traversal to the hot path and
does not claim a throughput gain.

## Verification

Use the repository virtual environment for all checks:

```bash
/workspace/pllm/.venv/bin/python -m pytest -q test/workgraph
/workspace/pllm/.venv/bin/python -m pytest -q test/core/test_workgraph_validator_oracle_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/test_runtime_typing_contracts_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/test_producer_plan_unit.py test/runtime/test_producer_executor_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q
/workspace/pllm/.venv/bin/python tools/check_typing.py
cargo fmt --manifest-path rust/tllm_core/Cargo.toml -- --check
cargo test --manifest-path rust/tllm_core/Cargo.toml
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV VIRTUAL_ENV=/workspace/pllm/.venv \
  /workspace/pllm/.venv/bin/maturin develop \
  --manifest-path rust/tllm_core/Cargo.toml --release
cp rust/tllm_core/target/release/libtllm_core.so tllm/_core.abi3.so
/workspace/pllm/.venv/bin/python -m pytest -q test/core/test_workgraph_validator_oracle_unit.py
```

The black-box compatibility coverage lives in
`test/workgraph/test_phase1a_runtime_blackbox_unit.py`. It exercises real
runtime plans, binding parity, legacy boundaries, immutable backend snapshots,
reconciliation epochs, and the no-`Move` Phase 1a contract.

The Rust/Python differential validator coverage is in
`test/core/test_workgraph_validator_oracle_unit.py`.

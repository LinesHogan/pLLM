# WorkGraph Phase 2

This guide tracks the single-GPU physical execution layer. Phase 1 remains the
typed source of truth for hierarchy, effects, artifacts, and compatibility
boundaries. The existing `DispatchPlan` and `ExecutionPlan` are still
authoritative at runtime until the GPU execution gates in this guide pass.

## Milestone 1: Physical Contracts

`compile_physical_flow_plan()` lowers one verified `Phase1aPlan` into a frozen
`PhysicalFlowPlan`. The plan contains only typed control-plane facts:

- a monotonically positive plan epoch;
- stable `NodeKey`, `ExecutorKey`, and `ArtifactKey` identities;
- `StreamClass`, `CapturePolicy`, and `FallbackPolicy` enums;
- typed fence dependencies and exact buffer-bucket descriptors.

The physical plan never retains a Tensor, CUDA stream/event, CUDAGraph,
callback, request id, or backend runtime handle. Python and Rust independently
validate the descriptor graph. Rust reads frozen attributes through
`#[pyo3(attribute)]`; there is no nested magic-key dictionary protocol.

Nodes are deterministically topologically ordered before Rust validation. A
captured attachment can use `CapturePolicy.PREFER` only when the root is a
complete captured artifact, every task declares capture/replay-safe
capabilities, and the compatibility node is not a legacy executor. Opaque
artifacts remain attached for diagnostics but use disabled capture, eager
fallback, and no physical output bucket. FULL graph mode fails with the typed
`WG120_UNSUPPORTED_GRAPH_MODE` boundary.

The current bucket lowering admits one static Tensor output. Symbolic,
unknown, scalar, or multi-output layouts do not receive a bucket and therefore
remain on the eager fallback path. This is a support boundary, not a partially
materialized graph.

## Runtime Ownership

Physical runtime objects stay in Python. `Fence` owns an event and exposes only
stream-local wait, record, and non-blocking query operations. `BufferLease`
owns one Tensor slot and its ready, completion, and retire fence protocol.
Borrowed storage requires a ready fence; managed storage requires a retire
fence. A slot is reusable only after completion and retirement have both been
recorded and their events report complete.

`PlanResources` owns the runtime objects for one epoch:

- `RuntimeStream(StreamClass, stream)`;
- `RuntimeEvent(FenceKey, event)`;
- `RuntimeGraph(ArtifactKey, graph)`;
- the epoch's `BufferLease` objects and their Tensors.

These holders bind Python objects to frozen typed identities but are not
structural metadata and cannot enter a WorkGraph signature or the Rust
validator. `StreamPool`, `PipelineLane`, and `GraphCaptureLane` remain the
behavior oracles for stream topology, rotating buffers, event choreography,
and graph capture/replay. WorkGraph does not replace their execution path in
this milestone.

No physical path calls `torch.cuda.synchronize()` to satisfy a dependency.
Stream/event dependencies and retire fences are the required protocol.

## Plan Snapshots

`PlanSnapshotStore` publishes a staged higher epoch only from the next
`prepare_step()` safe point. A step that already acquired resources retains the
same `PlanResources` object until `finish_step()`, even if a later step
publishes a new plan.

Retired resources remain strongly owned while an old step is active or any
lease is not reusable. Streams, events, graphs, and buffer storage become
collectable only after the active-step and retire-fence conditions are both
satisfied. Staging an equal or older epoch is rejected.

`select_buffer_bucket()` performs exact shape, dtype, device-kind, and CUDA
device-index matching without touching the CUDA driver or synchronizing. A
miss returns `None`; the current step must use eager fallback while a future
cold path may build a specialization.

## Current Execution Boundary

This milestone builds and validates physical metadata and Python resource
lifecycle primitives. It does not install a WorkGraph executor in the token
hot path, capture a CUDA graph during a step, or select a Python implementation
dynamically. The legacy runtime remains authoritative.

The ESamp resource initializer is no longer called during `load_model.post`,
so training allocation remains lazy until the first complete bundle. The
current host cannot communicate with the NVIDIA driver. Automatic WorkGraph
GPU execution remains disabled until an aligned ESamp run at
`gpu_memory_utilization=0.5` verifies startup, `loss_count > 0`, update and
capture/replay counts, zero steady-state fallback/deadline misses, and
fence/buffer/epoch statistics.

## Verification

Use the repository virtual environment:

```bash
/workspace/pllm/.venv/bin/python -m pytest -q test/workgraph/test_physical_plan_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_physical_executor_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_physical_plan_snapshot_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/core/test_physical_plan_validator_oracle_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q
/workspace/pllm/.venv/bin/pyright -p pyrightconfig.strict.json
cargo test --manifest-path rust/tllm_core/Cargo.toml
cargo fmt --manifest-path rust/tllm_core/Cargo.toml -- --check
git diff --check
```

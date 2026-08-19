# WorkGraph Phase 2

This guide documents the single-GPU physical execution layer. Phase 1 remains
the typed source of truth for hierarchy, effects, artifacts, and compatibility
boundaries. The existing `DispatchPlan` and `ExecutionPlan` remain
authoritative for legacy regions; WorkGraph executes only explicitly attached,
fully captured regions.

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

- `RuntimeStream(StreamClass, stream, retire_fence)`;
- `RuntimeEvent(FenceKey, event)`;
- `RuntimeGraph(ArtifactKey, graph)`;
- the epoch's `BufferLease` objects and their Tensors.

These holders bind Python objects to frozen typed identities but are not
structural metadata and cannot enter a WorkGraph signature or the Rust
validator. `StreamPool`, `PipelineLane`, and `GraphCaptureLane` remain the
behavior oracles for stream topology, rotating buffers, event choreography,
and graph capture/replay. The migrated ESamp region reuses those general
runtime primitives rather than duplicating their CUDA choreography.

No physical path calls `torch.cuda.synchronize()` to satisfy a dependency.
Stream/event dependencies and retire fences are the required protocol.

## Plan Snapshots

`PlanSnapshotStore` publishes a staged higher epoch only from the next
`prepare_step()` safe point. A step that already acquired resources retains the
same `PlanResources` object until `finish_step()`, even if a later step
publishes a new plan.

Retired resources remain strongly owned while an old step is active, any
lease is not reusable, or any per-stream retire fence is incomplete. Streams,
events, graphs, and buffer storage become collectable only after the
active-step and retire-fence conditions are both satisfied. An epoch that was
never acquired has no work to retire and can be released immediately. Staging
an equal or older epoch is rejected.

`select_buffer_bucket()` performs exact shape, dtype, device-kind, and CUDA
device-index matching without touching the CUDA driver or synchronizing. A
miss returns `None`; the current step must use eager fallback while a future
cold path may build a specialization.

`PhysicalExecutor` binds handlers by typed `NodeKey` or `ExecutorKey` when a
step is prepared. It follows the frozen node order, waits and publishes typed
fences, selects exact buckets, and records eager/replay/fallback and lease
statistics. `finish_step()` waits for an in-progress execute before recording
retirement, so a concurrent lifecycle call cannot release an epoch underneath
a handler. Physical step ids are monotonic; once a step lifecycle closes, the
same id cannot reacquire resources during the retire-record window. Bucket
misses enqueue a frozen request carrying plan epoch, node,
bucket, artifact, shape, dtype, and device. The callback runs only when the
next prepare safe point explicitly drains the queue; it may stage a higher
epoch before that step acquires its snapshot. No callback, handler lookup,
graph compilation, or capture is performed by `execute_step()`.

## ESamp Residual Capture Bridge

`ESampConsumer.specialize_workgraph(source_example, target_example)` is an
explicit cold specialization entry point. It strict-captures a pure
source/target residual handoff and binds both Tensor inputs to the exact typed
`ResidualStream` addresses declared by the ESamp flow. The resulting
`CapturedFlow` and `CaptureAttachment` use stable artifact, source, and
`NodeKey.consumer_flow(...)` identities.

Specialization is never called from `consume_bundle()`, a step hook, deferred
dispatch, or the token hot path. A consumer specialized after registration can
publish its frozen metadata with
`residual_runtime.refresh_workgraph_capture_metadata()`. Publication rebuilds
only the WorkGraph shadow snapshot: it preserves the existing `DispatchPlan`
and `ExecutionPlan` objects and short-circuits when metadata is unchanged.

The MRv1/MRv2 load-model quarantine derives a typed
`RuntimeSpecializationSpec` from the resolved model layout and publishes it
before the first token. Ordinary ESamp case toggles retain that load-time
specialization; disabling WorkGraph execution or changing the consumer
identity clears it. Disabled ESamp consumers expose no active capture
metadata.

The outer ESamp consumer flow is still deferrable legacy work. Its hierarchy
can therefore explain the captured residual leaf, while physical lowering
keeps the compatibility node at `CapturePolicy.DISABLED` with eager fallback.
The attached leaf region is activated independently through the typed
`BaseConsumer.activate_workgraph_regions()` boundary. MRv1 prepares the next
bundle epoch at its prepare-inputs safe point; MRv2 prepares immediately after
allocating the new execute-model step id. Both happen before residual dispatch
and never trigger capture or compilation.

The ESamp consumer declares a frozen `PhysicalRegionExecutorDesc` whose typed
extern contract is `WRITE_STATE`, replica-scoped, CUDA, and eager-only. The
activated `PhysicalExecutor` invokes the real `ESampTrainEngine.launch_step`
residual-staging operation; the outer consumer does not launch it a second
time. Its runtime stream adapter resolves the actual `PipelineLane` forward
stream (or the caller stream in serial mode), and the plan-level retire event
is recorded on that CUDA stream. CPU unit tests use the same contract with an
immediately complete non-CUDA retirement.

Delayed backward, update, and publish remain an explicit legacy boundary after
the migrated staging region. They continue to use the existing
`ESampTrainEngine`, `PipelineLane`, and `GraphCaptureLane` choreography. This
preserves first-per-prompt filtering, training density, parameter updates, and
sampler behavior while making the migrated physical region itself perform real
work rather than an identity-only counter update.

## Execution Boundary And Gate Status

Automatic execution is enabled by default and can be disabled with
`--no-workgraph-execution`. Only attachments with an explicit typed region
executor are activated; other captures remain shadow metadata. Unmigrated
producer/consumer work remains a visible legacy boundary. FULL graph mode
requests deactivation rather than partially executing a graph; an active
snapshot is retained until its step finishes and its retire event completes.

The ESamp resource initializer is not called during `load_model.post`, so
training allocation remains lazy until the first complete bundle. Runtime
statistics include plan epoch, prepare/execute/finish, replay/eager/fallback,
fence, bucket, specialization, lease, retired-plan, and deadline counts. In
multi-process vLLM these frozen numeric snapshots are returned through the
existing worker control plane; CUDA objects never cross the process boundary.

Validation on 2026-08-20 used the repository virtual environment and a
low-memory `Qwen/Qwen2.5-0.5B-Instruct`, bfloat16, batch 1, `n=2`, 8-token,
eager workload at `gpu_memory_utilization=0.35`. The legacy-reference and
automatic runs both reported `loss_count=7` and
`loss_avg=4.092917578560965`. The automatic run reported plan epoch 2,
`execute_count=7`, `eager_count=7`, and zero fallback/deadline misses; its
single-pair throughput ratio was about 0.9969.

The formal `gpu_memory_utilization=0.5` gate is currently environment-blocked,
not waived: a co-resident process owns about 9.2 GiB, and vLLM OOMs during its
own sampler warmup before WorkGraph execution while requesting another 150
MiB. Therefore the required five aligned alternating pairs and 95% confidence
bound are not yet recorded. Do not treat the low-memory smoke as the formal
performance gate.

## Verification

Use the repository virtual environment:

```bash
/workspace/pllm/.venv/bin/python -m pytest -q test/workgraph/test_physical_plan_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_physical_executor_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_physical_flow_executor_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_physical_plan_snapshot_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_phase2_runtime_activation_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_phase2_migrated_region_activation_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_automatic_cold_specialization_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/runtime/test_phase2_gpu_gate_stats_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/core/test_physical_plan_validator_oracle_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/consumers/esamp/test_esamp_consumer_contract_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q test/workgraph/test_phase1a_runtime_refresh_unit.py
/workspace/pllm/.venv/bin/python -m pytest -q
/workspace/pllm/.venv/bin/pyright -p pyrightconfig.strict.json
cargo test --manifest-path rust/tllm_core/Cargo.toml
cargo fmt --manifest-path rust/tllm_core/Cargo.toml -- --check
git diff --check
```

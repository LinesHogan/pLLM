# WorkGraph Phase 3

Phase 3 adds local CPU/GPU placement and transfer planning on top of the
single-GPU physical execution contract. It does not relax the Phase 1 or
Phase 2 schemas: their validators still reject `Move`, while the Phase 3
planner emits a separate frozen `PlacementPlan`.

## Typed control plane

`tllm.workgraph.phase3` defines stable placement, residency, value, move,
implementation-selection, copy-engine, schedule, and lease descriptors. A
runtime implementation is registered under an `ExecutorKey`; only its frozen
`ImplementationDesc` and selected key enter a plan. Tensor, callable, stream,
event, request, and backend handles remain outside structural metadata.

When one logical executor has implementations on more than one placement,
each implementation must name the same `SemanticOracleKey`. The registry must
run that registered oracle over explicit shared input cases before those
candidates become eligible. A mismatch, absent oracle, or stale verification
fails closed. Before returning cross-placement candidates, the registry reruns
the retained verification cases against the current implementation objects, so
mutating an implementation cannot reuse an older proof. The frozen selection
records only the oracle key, never the oracle callable, cases, or runtime
outputs.

`compile_placement_plan()` consumes an already validated `PhysicalFlowPlan`, a
typed `ImplementationRegistry`, directional `DeviceTopology`, and explicit
`ValueFlowDesc` values. It performs deterministic candidate selection and a
critical-path list schedule. Missing duration or transfer measurements use a
conservative typed fallback; the planner does not invoke CP-SAT without a
credible profile.

Placement is selected jointly for the whole region. The planner enumerates the
deterministically ordered Cartesian product of eligible implementations and
simulates the same dependency, placement, and copy-engine-capacity timeline
used by the emitted schedule. It minimizes makespan, then transfer work,
compute work, and finally a stable implementation-key tie breaker. Exact search
is bounded at 100,000 assignments; a larger space fails closed and requires
placement overrides or a smaller registry instead of silently falling back to
a local greedy choice.

Exact `TransferProfileDesc` entries match move kind and directional source /
destination placement, then derive duration from byte count. Candidate scoring
and the final copy timeline use the same lookup. `TransferProfile` supplies the
conservative per-engine fallback when no exact measured route exists.

The resulting plan records:

- the implementation and `Placement` selected for every physical node;
- the frozen `DeviceTopology` inventory and directional peer-access edges used
  to validate every selected endpoint and route at runtime;
- the frozen `ValueFlowDesc` demand projection, with one producer per
  `ValueKey`, required residencies, and exact cross-placement route coverage;
- node start/end times and ordered dependencies;
- all materialized `Residency` values;
- every `ScheduledMove`, directional copy engine, interval, and dependency;
- inferred residency live ranges and pinned staging lifetimes;
- bounded H2D, D2H, and P2P copy-engine capacities.

Placement affects `structural_signature()`. A plan is also validated by Rust
through `validate_placement_plan()`, whose PyO3 boundary reads frozen typed
attributes rather than nested dictionaries.

## Transfer routing

The route matrix is explicit and directional:

| Source | Destination | Route |
| --- | --- | --- |
| same placement | same placement | no `Move` |
| CPU | CUDA | H2D |
| CUDA | CPU | D2H |
| CUDA A | CUDA B with A -> B peer access | P2P |
| CUDA A | CUDA B without A -> B peer access | D2H to pinned host, then H2D |

Peer access is not treated as symmetric. For a staged GPU transfer, the H2D
move depends on the D2H move and the pinned-host lease covers both intervals.
`cuda_devices` is a closed device inventory. When it is omitted or empty,
peer-access endpoints are the only inferred CUDA devices; with no peer edges,
the topology is CPU-only. Implementations and transfer endpoints outside that
inventory fail closed.

## Python runtime ownership

`tllm.runtime.scheduling.phase3_transfer` owns actual tensors and asynchronous
resources. `PinnedMemoryPool` is bounded and cannot reuse a slot until its
retire fence reports completion. `CopyExecutor` admits a bounded number of
transfers per typed copy engine, preserves one staging slot across a two-segment
route, and releases both its slot and copy credit only after the final fence
completes. Submission is transactional: adapter failures return an unlaunched
slot, including a failure while launching the second staged segment.

`PlacementPlanDispatcher` resolves routes by typed producer, consumer, and
`ValueKey`, then configures the executor directly from the plan's copy-engine
capacities. This is the runtime connection between the frozen moves/leases and
Python-owned tensors and fences; an undeclared route fails closed, while the
same frozen plan can dispatch the route again in later runtime steps.

Production cold paths install a typed `PlacementRuntimeBinding`. The concrete
WorkGraph compiler publishes its `PlacementPlan` and dispatcher in the same
`PlanResources` epoch as the `PhysicalFlowPlan`; `PlanSnapshotStore` makes both
visible atomically at the next prepare safe point. `dispatch_workgraph_value()`
can only submit through the dispatcher acquired for that prepared epoch. Old
resources remain non-reusable while their dispatcher has an in-flight copy,
and retirement polls its fences before collection.

`TorchTensorCopyAdapter` supplies real pinned allocation and non-blocking
Torch copies. CPU-only tests use the same protocol with fake tensors and
fences, which verifies route and lifetime semantics without probing CUDA.
No transfer path calls `torch.cuda.synchronize()`.

## Support boundary

Phase 3 covers same-GPU residency, CPU fallback, directional peer GPU, and
non-peer pinned-host fallback. The registry may contain CPU and CUDA semantic
oracles under different typed implementation keys. Distributed state,
collective order, communicator ownership, and cross-node topology remain Phase
4 work. Online profile updates and adaptive admission remain Phase 5 work.

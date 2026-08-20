# WorkGraph Phase 4

Phase 4 freezes distributed topology, state residence, communicator ownership,
and collective order above the Phase 3 placement plan. The global planner owns
these decisions; a rank-local executor may only launch the exact projection of
the frozen global sequence during its approved windows.

## Typed global plan

`tllm.workgraph.phase4` defines stable descriptors for model and helper ranks,
TP/PP/DP/global groups, state keys and shards, placement affinity, state
residence, communicator keys, collective requests, and local schedules. It
reuses the frontend `StateScope` and Phase 3 `Placement` and `ValueKey` types.
Runtime process groups, tensors, streams, events, requests, and async work
handles never enter `DistributedFlowPlan`.

`DistributedTopology` requires contiguous global rank ids and an exact
TP x PP x DP Cartesian model mesh. Helper ranks have no model coordinates.
Node id plus local rank must be unique, while a local device index may repeat
on different nodes. The topology deterministically derives every parallel
group, including cross-node groups and a separate helper group.

State requirements fail closed when their scope, shard count, or affinity is
incompatible with the topology. Request-scoped state is not legal in a global
plan. The compiler derives the complete ordered state-residence set; callers
cannot inject or omit residences. Tensor, pipeline, and data shards use the
rank's actual TP, PP, or DP coordinate as their shard index. Every selected
replica group must exactly cover that axis, so cardinality-compatible but
coordinate-incomplete affinities are rejected.

Collective dependencies are topologically sorted with stable key tie-breaking.
The resulting request tuple, communicator set, collective sequence, and every
rank-local projection are canonical. Missing ranks, unknown dependencies,
cycles, invalid roots, wrong group indexes, and conflicting communicator
backends fail before publication. `SEND_RECV` requires exactly two ranks and
NCCL requires CUDA placements; CPU-only tests use Gloo.

The plan participates in `structural_signature()`. Rust validates the same
frozen attribute descriptors through `validate_distributed_flow_plan()`.
PyO3 uses typed `#[pyo3(attribute)]` fields rather than nested dictionaries.

## Python runtime ownership

`tllm.runtime.scheduling.phase4_collective` owns process groups, payloads, and
async fences. `DistributedPlanExecutor` is bound to one rank and one immutable
plan. It rejects an unprepared step, an out-of-order key, the wrong launch
window, an undeclared communicator, and an incomplete step. An adapter launch
or fence-query failure permanently poisons that executor so later operations
fail fast.

`TorchDistributedCollectiveAdapter` submits asynchronous Torch distributed
operations. It does not call `torch.cuda.synchronize()`. Tests exercise the
same protocol with CPU-only fake process groups and fences.

`DistributedPlanSnapshotStore` publishes a staged plan only at the next
prepare safe point. A step retains the exact plan and executor it acquired.
An old epoch remains non-reusable while a local step is open or any collective
fence is in flight. Retirement polls those fences before releasing the epoch.

The production residual runtime exposes typed cold-path configuration and
publication plus prepared-step launch, polling, and finish APIs. When no
distributed binding is installed, the existing physical and legacy prepare
behavior is unchanged. WorkGraph does not discover communicators or collective
order from runtime objects. Launch does not accept a caller-provided window;
the executor window advances monotonically from the actual prepare,
execute-model pre/post, and sample completion runtime events. Passing an
unlaunched window or moving backwards fails before the next launch.

Before a multi-rank epoch is staged, every rank exchanges a frozen
`DistributedPlanAgreementDesc`. Core validation requires exact rank coverage
and matching plan epoch, full structural signature, topology signature,
communicator projection, and local collective sequence. The Torch agreement
adapter performs this exchange over the initialized default process group.
Agreement divergence fails before any collective executor becomes active.

## Support boundary

Phase 4 covers deterministic TP, PP, DP, helper, global, and cross-node
descriptor planning, rank agreement, and rank-local asynchronous launch.
Process-group creation and distributed recovery after a detected failure remain
deployment responsibilities; membership or plan mismatch and runtime failure
are fail-fast. Online profile updates, interference-aware admission, bounded
plan densification, and work shedding are Phase 5 work.

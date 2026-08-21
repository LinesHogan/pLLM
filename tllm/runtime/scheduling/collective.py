"""Python-owned collective execution for frozen distributed plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol, TypeVar, cast, runtime_checkable

import torch

from tllm.workgraph.distributed import (
    CollectiveDesc,
    CollectiveKey,
    CollectiveKind,
    CollectiveRequestDesc,
    CollectiveWindow,
    CommunicatorDesc,
    CommunicatorKey,
    DistributedFlowPlan,
    DistributedTopology,
    StateRequirementDesc,
    compile_distributed_flow_plan,
)


_WINDOW_ORDER = {
    CollectiveWindow.PREPARE: 0,
    CollectiveWindow.BEFORE_COMPUTE: 1,
    CollectiveWindow.AFTER_COMPUTE: 2,
    CollectiveWindow.STEP_END: 3,
}


@runtime_checkable
class CollectiveFenceLike(Protocol):
    def query(self) -> bool: ...


@runtime_checkable
class CollectiveAdapter(Protocol):
    def launch(
        self,
        collective: CollectiveDesc,
        communicator_desc: CommunicatorDesc,
        payload: object,
        communicator: object,
    ) -> CollectiveFenceLike: ...


@runtime_checkable
class _TorchWorkLike(Protocol):
    def is_completed(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class DistributedPlanAgreementDesc:
    """Rank projection exchanged before a distributed epoch may publish."""

    plan_epoch: int
    plan_signature: str
    topology_signature: str
    topology_size: int
    rank: int
    communicators: tuple[CommunicatorKey, ...]
    collectives: tuple[CollectiveKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_epoch",
            _integer(self.plan_epoch, "DistributedPlanAgreementDesc.plan_epoch", minimum=1),
        )
        for label, value in (
            ("plan_signature", self.plan_signature),
            ("topology_signature", self.topology_signature),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"DistributedPlanAgreementDesc.{label} must be SHA-256")
        topology_size = _integer(
            self.topology_size,
            "DistributedPlanAgreementDesc.topology_size",
            minimum=1,
        )
        rank = _integer(self.rank, "DistributedPlanAgreementDesc.rank")
        if rank >= topology_size:
            raise ValueError("distributed agreement rank is outside topology")
        if not isinstance(self.communicators, tuple) or any(
            not isinstance(item, CommunicatorKey) for item in self.communicators
        ):
            raise TypeError("distributed agreement communicators must be typed tuple")
        if not isinstance(self.collectives, tuple) or any(
            not isinstance(item, CollectiveKey) for item in self.collectives
        ):
            raise TypeError("distributed agreement collectives must be typed tuple")


@runtime_checkable
class DistributedAgreementAdapter(Protocol):
    def exchange(
        self,
        local: DistributedPlanAgreementDesc,
    ) -> tuple[DistributedPlanAgreementDesc, ...]: ...


class TorchDistributedAgreementAdapter:
    """Exchange frozen rank projections through the default process group."""

    def exchange(
        self,
        local: DistributedPlanAgreementDesc,
    ) -> tuple[DistributedPlanAgreementDesc, ...]:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized for plan agreement")
        if int(dist.get_world_size()) != local.topology_size:
            raise RuntimeError("distributed plan topology differs from process-group size")
        if int(dist.get_rank()) != local.rank:
            raise RuntimeError("distributed plan rank differs from process-group rank")
        gathered: list[object] = [None] * local.topology_size
        dist.all_gather_object(gathered, local)
        if any(not isinstance(item, DistributedPlanAgreementDesc) for item in gathered):
            raise TypeError("distributed agreement exchanged an untyped rank projection")
        return cast(tuple[DistributedPlanAgreementDesc, ...], tuple(gathered))


_T = TypeVar("_T")


def _instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def distributed_plan_agreement_desc(
    plan: DistributedFlowPlan,
    rank: int,
) -> DistributedPlanAgreementDesc:
    """Build the frozen projection that one rank contributes to agreement."""

    from tllm.workgraph.signature import structural_signature

    plan = _instance(plan, DistributedFlowPlan, "distributed agreement plan")
    rank = _integer(rank, "distributed agreement rank")
    if rank >= len(plan.topology.ranks):
        raise ValueError("distributed agreement rank is outside topology")
    communicators = tuple(
        item.key for item in plan.communicators if rank in item.ranks
    )
    return DistributedPlanAgreementDesc(
        plan_epoch=plan.plan_epoch,
        plan_signature=structural_signature(plan),
        topology_signature=structural_signature(plan.topology),
        topology_size=len(plan.topology.ranks),
        rank=rank,
        communicators=communicators,
        collectives=plan.local_schedules[rank].collectives,
    )


def validate_distributed_plan_agreement(
    plan: DistributedFlowPlan,
    agreements: tuple[DistributedPlanAgreementDesc, ...],
) -> None:
    """Require exact epoch, topology, communicator, and order agreement."""

    plan = _instance(plan, DistributedFlowPlan, "distributed agreement plan")
    if not isinstance(agreements, tuple) or any(
        not isinstance(item, DistributedPlanAgreementDesc) for item in agreements
    ):
        raise TypeError("distributed agreement must contain typed rank projections")
    by_rank = {item.rank: item for item in agreements}
    if len(by_rank) != len(agreements) or set(by_rank) != set(range(len(plan.topology.ranks))):
        raise RuntimeError("distributed agreement must exactly cover topology ranks")
    for rank in range(len(plan.topology.ranks)):
        if by_rank[rank] != distributed_plan_agreement_desc(plan, rank):
            raise RuntimeError(
                "distributed ranks disagree on plan epoch, topology, communicator, or order"
            )


@dataclass(frozen=True, slots=True)
class TorchCollectivePayload:
    input: torch.Tensor | None = None
    output: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.input is not None and not isinstance(self.input, torch.Tensor):
            raise TypeError("TorchCollectivePayload.input must be Tensor or None")
        if self.output is not None and not isinstance(self.output, torch.Tensor):
            raise TypeError("TorchCollectivePayload.output must be Tensor or None")


class _TorchWorkFence:
    __slots__ = ("_work",)

    def __init__(self, work: _TorchWorkLike) -> None:
        self._work = work

    def query(self) -> bool:
        return bool(self._work.is_completed())


class TorchDistributedCollectiveAdapter:
    """Thin async adapter over ``torch.distributed`` process groups."""

    def launch(
        self,
        collective: CollectiveDesc,
        communicator_desc: CommunicatorDesc,
        payload: object,
        communicator: object,
    ) -> CollectiveFenceLike:
        collective = _instance(
            collective,
            CollectiveDesc,
            "Torch collective descriptor",
        )
        communicator_desc = _instance(
            communicator_desc,
            CommunicatorDesc,
            "Torch communicator descriptor",
        )
        payload = _instance(
            payload,
            TorchCollectivePayload,
            "Torch collective payload",
        )
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before launch")
        group = cast(dist.ProcessGroup | None, communicator)
        input_tensor = payload.input
        output_tensor = payload.output
        kind = collective.kind
        if kind == CollectiveKind.ALL_REDUCE:
            if input_tensor is None:
                raise ValueError("all_reduce requires an input tensor")
            work = dist.all_reduce(input_tensor, group=group, async_op=True)
        elif kind == CollectiveKind.ALL_GATHER:
            if input_tensor is None or output_tensor is None:
                raise ValueError("all_gather requires input and output tensors")
            work = dist.all_gather_into_tensor(
                output_tensor,
                input_tensor,
                group=group,
                async_op=True,
            )
        elif kind == CollectiveKind.REDUCE_SCATTER:
            if input_tensor is None or output_tensor is None:
                raise ValueError("reduce_scatter requires input and output tensors")
            work = dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                group=group,
                async_op=True,
            )
        elif kind == CollectiveKind.BROADCAST:
            if input_tensor is None or collective.root_rank is None:
                raise ValueError("broadcast requires input tensor and root rank")
            work = dist.broadcast(
                input_tensor,
                src=collective.root_rank,
                group=group,
                async_op=True,
            )
        elif kind == CollectiveKind.BARRIER:
            work = dist.barrier(group=group, async_op=True)
        else:
            if collective.root_rank is None:
                raise ValueError("send_recv requires a sender rank")
            rank = int(dist.get_rank())
            peers = communicator_desc.ranks
            destination = next(item for item in peers if item != collective.root_rank)
            if rank == collective.root_rank:
                if input_tensor is None:
                    raise ValueError("send side requires an input tensor")
                work = dist.isend(
                    input_tensor,
                    dst=destination,
                    group=group,
                )
            else:
                if output_tensor is None:
                    raise ValueError("receive side requires an output tensor")
                work = dist.irecv(
                    output_tensor,
                    src=collective.root_rank,
                    group=group,
                )
        if work is None or not isinstance(cast(object, work), _TorchWorkLike):
            raise TypeError("torch.distributed work must provide is_completed()")
        return _TorchWorkFence(work)


@dataclass(slots=True)
class CollectiveHandle:
    step_id: int
    collective: CollectiveKey
    fence: CollectiveFenceLike
    complete: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.step_id = _integer(self.step_id, "CollectiveHandle.step_id")
        self.collective = _instance(
            self.collective,
            CollectiveKey,
            "CollectiveHandle.collective",
        )
        if not isinstance(cast(object, self.fence), CollectiveFenceLike):
            raise TypeError("CollectiveHandle.fence must implement query()")


@dataclass(slots=True)
class _PreparedDistributedStep:
    cursor: int = 0
    finished: bool = False
    window: CollectiveWindow = CollectiveWindow.PREPARE


class DistributedPlanExecutor:
    """Rank-local launcher constrained by one frozen global plan."""

    __slots__ = (
        "plan",
        "rank",
        "_adapter",
        "_communicators",
        "_schedule",
        "_collectives",
        "_communicator_descs",
        "_steps",
        "_in_flight",
        "_failed",
        "_highest_step",
        "_lock",
    )

    def __init__(
        self,
        *,
        plan: DistributedFlowPlan,
        rank: int,
        communicators: tuple[tuple[CommunicatorKey, object], ...],
        adapter: CollectiveAdapter,
    ) -> None:
        self.plan = _instance(plan, DistributedFlowPlan, "DistributedPlanExecutor.plan")
        self.rank = _integer(rank, "DistributedPlanExecutor.rank")
        if self.rank >= len(plan.topology.ranks):
            raise ValueError("DistributedPlanExecutor rank is outside topology")
        if not isinstance(communicators, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], CommunicatorKey)
            for item in communicators
        ):
            raise TypeError("communicators must contain CommunicatorKey/object pairs")
        communicator_map = dict(communicators)
        if len(communicator_map) != len(communicators):
            raise ValueError("runtime communicator keys must be unique")
        schedule = plan.local_schedules[self.rank].collectives
        collectives = {item.key: item for item in plan.collectives}
        communicator_descs = {item.key: item for item in plan.communicators}
        required = {
            collectives[key].communicator for key in schedule
        }
        if set(communicator_map) != required:
            raise ValueError(
                "runtime communicators must exactly cover the rank-local schedule"
            )
        if not isinstance(cast(object, adapter), CollectiveAdapter):
            raise TypeError("adapter must implement CollectiveAdapter")
        self._adapter = adapter
        self._communicators = communicator_map
        self._schedule = schedule
        self._collectives = collectives
        self._communicator_descs = communicator_descs
        self._steps: dict[int, _PreparedDistributedStep] = {}
        self._in_flight: list[CollectiveHandle] = []
        self._failed: BaseException | None = None
        self._highest_step = -1
        self._lock = Lock()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._in_flight)

    @property
    def reusable(self) -> bool:
        with self._lock:
            return not self._steps and not self._in_flight

    def prepare_step(self, step_id: int) -> None:
        step = _integer(step_id, "DistributedPlanExecutor.step_id")
        with self._lock:
            self._raise_failed()
            if step in self._steps:
                return
            if step <= self._highest_step:
                raise RuntimeError("distributed step lifecycle is already closed")
            self._steps[step] = _PreparedDistributedStep()
            self._highest_step = step

    def launch(
        self,
        step_id: int,
        *,
        collective: CollectiveKey,
        payload: object,
    ) -> CollectiveHandle:
        step = _integer(step_id, "DistributedPlanExecutor.step_id")
        collective = _instance(
            collective,
            CollectiveKey,
            "DistributedPlanExecutor.collective",
        )
        with self._lock:
            self._raise_failed()
            prepared = self._steps.get(step)
            if prepared is None or prepared.finished:
                raise RuntimeError("distributed step must be prepared before launch")
            if prepared.cursor >= len(self._schedule):
                raise RuntimeError("rank-local collective schedule is exhausted")
            expected = self._schedule[prepared.cursor]
            if collective != expected:
                raise RuntimeError("collective launch differs from frozen local order")
            descriptor = self._collectives[collective]
            if prepared.window != descriptor.window:
                raise RuntimeError("collective launch is outside its frozen window")
            communicator_desc = self._communicator_descs[descriptor.communicator]
            communicator = self._communicators[descriptor.communicator]
            try:
                fence = self._adapter.launch(
                    descriptor,
                    communicator_desc,
                    payload,
                    communicator,
                )
                if not isinstance(cast(object, fence), CollectiveFenceLike):
                    raise TypeError("collective adapter must return CollectiveFenceLike")
            except BaseException as exc:
                self._failed = exc
                raise
            handle = CollectiveHandle(step, collective, fence)
            prepared.cursor += 1
            self._in_flight.append(handle)
            return handle

    def advance_window(
        self,
        step_id: int,
        window: CollectiveWindow,
    ) -> None:
        """Advance from an actual runtime lifecycle event, never from launch."""

        step = _integer(step_id, "DistributedPlanExecutor.step_id")
        window = _instance(
            window,
            CollectiveWindow,
            "DistributedPlanExecutor.window",
        )
        with self._lock:
            self._raise_failed()
            prepared = self._steps.get(step)
            if prepared is None or prepared.finished:
                raise RuntimeError("distributed step must be prepared before window advance")
            current = _WINDOW_ORDER[prepared.window]
            target = _WINDOW_ORDER[window]
            if target < current:
                raise RuntimeError("distributed lifecycle window cannot move backwards")
            if target == current:
                return
            missed = tuple(
                key
                for key in self._schedule[prepared.cursor :]
                if _WINDOW_ORDER[self._collectives[key].window] < target
            )
            if missed:
                raise RuntimeError(
                    "distributed lifecycle cannot pass an unlaunched collective window"
                )
            prepared.window = window

    def poll(self) -> int:
        with self._lock:
            self._raise_failed()
            remaining: list[CollectiveHandle] = []
            completed = 0
            for handle in self._in_flight:
                try:
                    done = bool(handle.fence.query())
                except BaseException as exc:
                    self._failed = exc
                    raise
                if done:
                    handle.complete = True
                    completed += 1
                else:
                    remaining.append(handle)
            self._in_flight = remaining
            return completed

    def finish_step(self, step_id: int) -> None:
        step = _integer(step_id, "DistributedPlanExecutor.step_id")
        with self._lock:
            self._raise_failed()
            prepared = self._steps.get(step)
            if prepared is None:
                return
            if prepared.cursor != len(self._schedule):
                raise RuntimeError(
                    "distributed step cannot finish before all local collectives launch"
                )
            prepared.finished = True
            del self._steps[step]

    def fail(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("distributed failure must be BaseException")
        with self._lock:
            self._failed = error

    def _raise_failed(self) -> None:
        if self._failed is not None:
            raise RuntimeError("distributed executor is failed") from self._failed


@dataclass(frozen=True, slots=True)
class DistributedRuntimeBinding:
    topology: DistributedTopology
    rank: int
    communicators: tuple[tuple[CommunicatorKey, object], ...]
    adapter: CollectiveAdapter
    agreement: DistributedAgreementAdapter | None = None
    state_requirements: tuple[StateRequirementDesc, ...] = ()
    collective_requests: tuple[CollectiveRequestDesc, ...] = ()

    def __post_init__(self) -> None:
        _instance(
            self.topology,
            DistributedTopology,
            "DistributedRuntimeBinding.topology",
        )
        rank = _integer(self.rank, "DistributedRuntimeBinding.rank")
        if rank >= len(self.topology.ranks):
            raise ValueError("DistributedRuntimeBinding rank is outside topology")
        if not isinstance(self.communicators, tuple):
            raise TypeError("DistributedRuntimeBinding.communicators must be tuple")
        if not isinstance(cast(object, self.adapter), CollectiveAdapter):
            raise TypeError("DistributedRuntimeBinding.adapter must implement CollectiveAdapter")
        agreement = self.agreement
        if len(self.topology.ranks) > 1 and agreement is None:
            raise ValueError("multi-rank distributed binding requires plan agreement")
        if agreement is not None and not isinstance(
            cast(object, agreement), DistributedAgreementAdapter
        ):
            raise TypeError("agreement must implement DistributedAgreementAdapter")
        if not isinstance(self.state_requirements, tuple) or any(
            not isinstance(item, StateRequirementDesc)
            for item in self.state_requirements
        ):
            raise TypeError("state_requirements must be a typed tuple")
        if not isinstance(self.collective_requests, tuple) or any(
            not isinstance(item, CollectiveRequestDesc)
            for item in self.collective_requests
        ):
            raise TypeError("collective_requests must be a typed tuple")

    def compile(
        self,
        plan_epoch: int,
    ) -> tuple[DistributedFlowPlan, DistributedPlanExecutor]:
        plan = compile_distributed_flow_plan(
            plan_epoch=plan_epoch,
            topology=self.topology,
            state_requirements=self.state_requirements,
            collective_requests=self.collective_requests,
        )
        local_agreement = distributed_plan_agreement_desc(plan, self.rank)
        agreements = (
            (local_agreement,)
            if self.agreement is None
            else self.agreement.exchange(local_agreement)
        )
        validate_distributed_plan_agreement(plan, agreements)
        return plan, DistributedPlanExecutor(
            plan=plan,
            rank=self.rank,
            communicators=self.communicators,
            adapter=self.adapter,
        )


@dataclass(frozen=True, slots=True, eq=False)
class DistributedPlanResources:
    """One frozen distributed plan and its Python-owned rank-local executor."""

    plan: DistributedFlowPlan
    executor: DistributedPlanExecutor

    def __post_init__(self) -> None:
        plan = _instance(self.plan, DistributedFlowPlan, "DistributedPlanResources.plan")
        executor = _instance(
            self.executor,
            DistributedPlanExecutor,
            "DistributedPlanResources.executor",
        )
        if executor.plan is not plan:
            raise ValueError("distributed executor must bind its exact frozen plan")

    @property
    def reusable(self) -> bool:
        return self.executor.reusable

    def poll(self) -> int:
        return self.executor.poll()


class DistributedPlanSnapshotStore:
    """Publish distributed epochs at prepare safe points and retire by fence."""

    __slots__ = (
        "_current",
        "_pending",
        "_active",
        "_retired",
        "_acquired",
        "_lock",
    )

    def __init__(self, initial: DistributedPlanResources) -> None:
        initial = _instance(
            initial,
            DistributedPlanResources,
            "DistributedPlanSnapshotStore.initial",
        )
        self._current = initial
        self._pending: DistributedPlanResources | None = None
        self._active: dict[int, DistributedPlanResources] = {}
        self._retired: list[DistributedPlanResources] = []
        self._acquired: set[int] = set()
        self._lock = Lock()

    @property
    def current(self) -> DistributedFlowPlan:
        with self._lock:
            return self._current.plan

    @property
    def closeable(self) -> bool:
        with self._lock:
            if self._active:
                return False
            resources = (self._current, *self._retired)
            return all(
                id(item) not in self._acquired or item.reusable
                for item in resources
            )

    @property
    def retired_count(self) -> int:
        with self._lock:
            return len(self._retired)

    def stage(self, resources: DistributedPlanResources) -> None:
        resources = _instance(
            resources,
            DistributedPlanResources,
            "DistributedPlanSnapshotStore.resources",
        )
        with self._lock:
            floor = self._current.plan.plan_epoch
            if self._pending is not None:
                floor = max(floor, self._pending.plan.plan_epoch)
            if resources.plan.plan_epoch <= floor:
                raise ValueError("staged distributed plan epoch must be strictly newer")
            self._pending = resources

    def prepare_step(self, step_id: int) -> DistributedFlowPlan:
        step = _integer(step_id, "DistributedPlanSnapshotStore.step_id")
        with self._lock:
            existing = self._active.get(step)
            if existing is not None:
                return existing.plan
            pending = self._pending
            if pending is not None:
                previous = self._current
                self._current = pending
                self._pending = None
                if all(item is not previous for item in self._retired):
                    self._retired.append(previous)
            resources = self._current
            self._active[step] = resources
            self._acquired.add(id(resources))
        resources.executor.prepare_step(step)
        return resources.plan

    def active(self, step_id: int) -> DistributedFlowPlan:
        with self._lock:
            return self._active[
                _integer(step_id, "DistributedPlanSnapshotStore.step_id")
            ].plan

    def executor(self, step_id: int) -> DistributedPlanExecutor:
        with self._lock:
            return self._active[
                _integer(step_id, "DistributedPlanSnapshotStore.step_id")
            ].executor

    def finish_step(self, step_id: int) -> None:
        step = _integer(step_id, "DistributedPlanSnapshotStore.step_id")
        with self._lock:
            resources = self._active.get(step)
        if resources is None:
            return
        resources.executor.finish_step(step)
        with self._lock:
            self._active.pop(step, None)

    def poll(self) -> int:
        with self._lock:
            resources = (self._current, *self._retired, *self._active.values())
        unique = {id(item): item for item in resources}
        return sum(item.poll() for item in unique.values())

    def collect_retired(self) -> int:
        with self._lock:
            retired = tuple(self._retired)
        for resources in retired:
            resources.poll()
        with self._lock:
            active_ids = {id(resources) for resources in self._active.values()}
            kept: list[DistributedPlanResources] = []
            collected = 0
            for resources in self._retired:
                resource_id = id(resources)
                if resource_id in active_ids or (
                    resource_id in self._acquired and not resources.reusable
                ):
                    kept.append(resources)
                else:
                    self._acquired.discard(resource_id)
                    collected += 1
            self._retired = kept
            return collected


__all__ = [
    "CollectiveAdapter",
    "CollectiveFenceLike",
    "CollectiveHandle",
    "DistributedAgreementAdapter",
    "DistributedPlanAgreementDesc",
    "DistributedPlanResources",
    "DistributedPlanExecutor",
    "DistributedPlanSnapshotStore",
    "DistributedRuntimeBinding",
    "TorchDistributedAgreementAdapter",
    "TorchCollectivePayload",
    "TorchDistributedCollectiveAdapter",
    "distributed_plan_agreement_desc",
    "validate_distributed_plan_agreement",
]

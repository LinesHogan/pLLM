"""Python-owned executor for frozen single-device WorkGraph plans."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import Condition, Lock
from types import MappingProxyType
from typing import cast
from typing import Protocol

import torch

from tllm.runtime.scheduling.physical import (
    BufferLease,
    EventLike,
    GraphLike,
    StreamLike,
)
from tllm.runtime.scheduling.plan_store import (
    PlanResources,
    PlanSnapshotStore,
    RuntimeEvent,
    RuntimeGraph,
    RuntimeStream,
    select_buffer_bucket,
)
from tllm.workgraph.frontend import ArtifactKey
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.physical import (
    BufferBucketKey,
    CapturePolicy,
    FallbackPolicy,
    FenceKey,
    PhysicalNodePlan,
    StreamClass,
)


class PhysicalExecutionError(RuntimeError):
    """Raised when a frozen plan cannot be executed under its declared policy."""


class PhysicalExecutionKind(StrEnum):
    EAGER = "eager"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class PhysicalExecutionContext:
    step_id: int
    plan_epoch: int
    node: PhysicalNodePlan
    kind: PhysicalExecutionKind
    stream: object
    tensors: tuple[torch.Tensor, ...]
    leases: tuple[BufferLease, ...]


@dataclass(frozen=True, slots=True)
class PhysicalStepResult:
    step_id: int
    plan_epoch: int
    eager_nodes: tuple[str, ...]
    replayed_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpecializationRequestDesc:
    plan_epoch: int
    node_key: NodeKey
    buffer_bucket: BufferBucketKey
    artifact_key: ArtifactKey | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        if (
            not isinstance(self.plan_epoch, int)
            or isinstance(self.plan_epoch, bool)
            or self.plan_epoch < 1
        ):
            raise ValueError("specialization request plan_epoch must be positive")
        if not isinstance(cast(object, self.node_key), NodeKey):
            raise TypeError("specialization request node_key must be NodeKey")
        if not isinstance(cast(object, self.buffer_bucket), BufferBucketKey):
            raise TypeError(
                "specialization request buffer_bucket must be BufferBucketKey"
            )
        if self.artifact_key is not None and not isinstance(
            cast(object, self.artifact_key), ArtifactKey
        ):
            raise TypeError(
                "specialization request artifact_key must be ArtifactKey or None"
            )
        if not isinstance(cast(object, self.shape), tuple) or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            for dimension in self.shape
        ):
            raise TypeError(
                "specialization request shape must contain non-negative ints"
            )
        if not isinstance(cast(object, self.dtype), torch.dtype):
            raise TypeError("specialization request dtype must be torch.dtype")
        if not isinstance(cast(object, self.device), torch.device):
            raise TypeError("specialization request device must be torch.device")


@dataclass(slots=True)
class PhysicalExecutionStats:
    prepare_count: int = 0
    execute_count: int = 0
    finish_count: int = 0
    eager_count: int = 0
    replay_count: int = 0
    fallback_count: int = 0
    fence_wait_count: int = 0
    fence_record_count: int = 0
    bucket_miss_count: int = 0
    specialization_request_count: int = 0
    retired_plan_count: int = 0
    last_plan_epoch: int = 0


@dataclass(frozen=True, slots=True)
class WorkGraphRuntimeStats:
    active_plan_epoch: int = 0
    prepare_count: int = 0
    execute_count: int = 0
    finish_count: int = 0
    replay_count: int = 0
    eager_count: int = 0
    fallback_count: int = 0
    fence_wait_count: int = 0
    fence_record_count: int = 0
    bucket_miss_count: int = 0
    specialization_request_count: int = 0
    active_leases: int = 0
    retired_plans: int = 0
    deadline_flush_count: int = 0

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> "WorkGraphRuntimeStats":
        """Decode a worker-side numeric snapshot from the control plane."""

        return cls(
            active_plan_epoch=int(payload.get("active_plan_epoch", 0)),
            prepare_count=int(payload.get("prepare_count", 0)),
            execute_count=int(payload.get("execute_count", 0)),
            finish_count=int(payload.get("finish_count", 0)),
            replay_count=int(payload.get("replay_count", 0)),
            eager_count=int(payload.get("eager_count", 0)),
            fallback_count=int(payload.get("fallback_count", 0)),
            fence_wait_count=int(payload.get("fence_wait_count", 0)),
            fence_record_count=int(payload.get("fence_record_count", 0)),
            bucket_miss_count=int(payload.get("bucket_miss_count", 0)),
            specialization_request_count=int(payload.get("specialization_request_count", 0)),
            active_leases=int(payload.get("active_leases", 0)),
            retired_plans=int(payload.get("retired_plans", 0)),
            deadline_flush_count=int(payload.get("deadline_flush_count", 0)),
        )

    @classmethod
    def merge(cls, entries: tuple["WorkGraphRuntimeStats", ...]) -> "WorkGraphRuntimeStats":
        """Merge local and worker snapshots without exposing runtime objects."""

        if not entries:
            return cls()
        return cls(
            active_plan_epoch=max(entry.active_plan_epoch for entry in entries),
            prepare_count=sum(entry.prepare_count for entry in entries),
            execute_count=sum(entry.execute_count for entry in entries),
            finish_count=sum(entry.finish_count for entry in entries),
            replay_count=sum(entry.replay_count for entry in entries),
            eager_count=sum(entry.eager_count for entry in entries),
            fallback_count=sum(entry.fallback_count for entry in entries),
            fence_wait_count=sum(entry.fence_wait_count for entry in entries),
            fence_record_count=sum(entry.fence_record_count for entry in entries),
            bucket_miss_count=sum(entry.bucket_miss_count for entry in entries),
            specialization_request_count=sum(
                entry.specialization_request_count for entry in entries
            ),
            active_leases=sum(entry.active_leases for entry in entries),
            retired_plans=sum(entry.retired_plans for entry in entries),
            deadline_flush_count=sum(entry.deadline_flush_count for entry in entries),
        )


class WorkGraphRuntimeStatsReader(Protocol):
    def read_workgraph_runtime_stats(self) -> WorkGraphRuntimeStats: ...


def runtime_stats_from_executor(
    stats: PhysicalExecutionStats,
    *,
    active_leases: int = 0,
) -> WorkGraphRuntimeStats:
    return WorkGraphRuntimeStats(
        active_plan_epoch=stats.last_plan_epoch,
        prepare_count=stats.prepare_count,
        execute_count=stats.execute_count,
        finish_count=stats.finish_count,
        replay_count=stats.replay_count,
        eager_count=stats.eager_count,
        fallback_count=stats.fallback_count,
        fence_wait_count=stats.fence_wait_count,
        fence_record_count=stats.fence_record_count,
        bucket_miss_count=stats.bucket_miss_count,
        specialization_request_count=stats.specialization_request_count,
        active_leases=active_leases,
        retired_plans=stats.retired_plan_count,
    )


PhysicalNodeHandler = Callable[[PhysicalExecutionContext], None]
SpecializationRequest = Callable[[SpecializationRequestDesc], None]


@dataclass(frozen=True, slots=True)
class _PreparedStep:
    resources: PlanResources
    streams: Mapping[StreamClass, RuntimeStream]
    events: Mapping[FenceKey, RuntimeEvent]
    graphs: Mapping[ArtifactKey, RuntimeGraph]
    handlers: Mapping[NodeKey, PhysicalNodeHandler]
    outgoing_fences: Mapping[NodeKey, tuple[FenceKey, ...]]
    leases: Mapping[BufferBucketKey, tuple[BufferLease, ...]]


class PhysicalExecutor:
    """Execute immutable physical descriptors against Python-owned resources.

    Handler resolution happens only when a step is prepared.  Execution then
    follows the plan's already validated node order without implementation
    probing or graph recompilation on the hot path.
    """

    __slots__ = (
        "_snapshots",
        "_bindings",
        "_prepared",
        "_requested_specializations",
        "_pending_specializations",
        "_on_specialization_request",
        "_lock",
        "_condition",
        "_executing",
        "_highest_prepared_step",
        "stats",
    )

    def __init__(
        self,
        *,
        snapshots: PlanSnapshotStore,
        handlers: Mapping[NodeKey | ExecutorKey, PhysicalNodeHandler],
        on_specialization_request: SpecializationRequest | None = None,
    ) -> None:
        if not isinstance(cast(object, snapshots), PlanSnapshotStore):
            raise TypeError("snapshots must be PlanSnapshotStore")
        bindings = dict(handlers)
        if any(not callable(handler) for handler in bindings.values()):
            raise TypeError("physical executor handlers must be callable")
        if any(
            not isinstance(key, (NodeKey, ExecutorKey))
            for key in bindings
        ):
            raise TypeError("physical executor handler keys must be NodeKey or ExecutorKey")
        if on_specialization_request is not None and not callable(
            on_specialization_request
        ):
            raise TypeError("on_specialization_request must be callable")
        self._snapshots = snapshots
        self._bindings = MappingProxyType(bindings)
        self._prepared: dict[int, _PreparedStep] = {}
        self._requested_specializations: set[SpecializationRequestDesc] = set()
        self._pending_specializations: deque[SpecializationRequestDesc] = deque()
        self._on_specialization_request = on_specialization_request
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._executing: set[int] = set()
        self._highest_prepared_step = -1
        self.stats = PhysicalExecutionStats()

    def prepare_step(self, step_id: int) -> PlanResources:
        step = _step_id(step_id)
        with self._condition:
            existing = self._prepared.get(step)
            if existing is not None:
                return existing.resources
            if step <= self._highest_prepared_step:
                raise PhysicalExecutionError(
                    f"physical step {step} cannot be prepared after its lifecycle closed"
                )
            resources = self._snapshots.prepare_step(step)
            prepared = self._prepare(resources)
            self._prepared[step] = prepared
            self._highest_prepared_step = step
            self._requested_specializations = {
                request
                for request in self._requested_specializations
                if request.plan_epoch >= resources.plan.plan_epoch
            }
            self.stats.prepare_count += 1
            self.stats.last_plan_epoch = resources.plan.plan_epoch
            return resources

    def execute_step(
        self,
        step_id: int,
        *,
        tensors: tuple[torch.Tensor, ...] = (),
    ) -> PhysicalStepResult:
        step = _step_id(step_id)
        inputs = _tensors(tensors)
        with self._condition:
            try:
                prepared = self._prepared[step]
            except KeyError as exc:
                raise PhysicalExecutionError(
                    f"physical step {step} must be prepared before execution"
                ) from exc
            if step in self._executing:
                raise PhysicalExecutionError(
                    f"physical step {step} is already executing"
                )
            self._executing.add(step)
            self.stats.execute_count += 1

        eager_nodes: list[str] = []
        replayed_nodes: list[str] = []
        try:
            for node in prepared.resources.plan.nodes:
                stream_resource = prepared.streams[node.stream]
                stream = cast(StreamLike, stream_resource.stream)
                node_leases = (
                    ()
                    if node.buffer_bucket is None
                    else prepared.leases.get(node.buffer_bucket, ())
                )
                for lease in node_leases:
                    lease.wait_ready(stream)
                    if lease.ready_fence is not None:
                        self.stats.fence_wait_count += 1
                for dependency in node.fences:
                    event = prepared.events[dependency.fence]
                    stream.wait_event(event.event)
                    self.stats.fence_wait_count += 1

                bucket_miss = self._request_specialization_on_miss(
                    prepared.resources,
                    node,
                    inputs,
                )
                graph_resource = (
                    None
                    if node.artifact_key is None
                    else prepared.graphs.get(node.artifact_key)
                )
                if (
                    not bucket_miss
                    and node.capture_policy != CapturePolicy.DISABLED
                    and graph_resource is not None
                ):
                    cast(GraphLike, graph_resource.graph).replay()
                    replayed_nodes.append(str(node.node_key))
                    self.stats.replay_count += 1
                else:
                    if node.capture_policy != CapturePolicy.DISABLED:
                        if node.fallback_policy == FallbackPolicy.REJECT:
                            reason = "bucket specialization" if bucket_miss else "capture graph"
                            raise PhysicalExecutionError(
                                f"physical node {node.node_key} requires {reason}"
                            )
                        self.stats.fallback_count += 1
                    context = PhysicalExecutionContext(
                        step_id=step,
                        plan_epoch=prepared.resources.plan.plan_epoch,
                        node=node,
                        kind=PhysicalExecutionKind.EAGER,
                        stream=stream,
                        tensors=inputs,
                        leases=node_leases,
                    )
                    prepared.handlers[node.node_key](context)
                    eager_nodes.append(str(node.node_key))
                    self.stats.eager_count += 1

                for lease in node_leases:
                    lease.publish_completion(stream)
                    self.stats.fence_record_count += 1
                for fence_key in prepared.outgoing_fences.get(node.node_key, ()):
                    cast(EventLike, prepared.events[fence_key].event).record(stream)
                    self.stats.fence_record_count += 1
        finally:
            with self._condition:
                self._executing.discard(step)
                self._condition.notify_all()

        return PhysicalStepResult(
            step_id=step,
            plan_epoch=prepared.resources.plan.plan_epoch,
            eager_nodes=tuple(eager_nodes),
            replayed_nodes=tuple(replayed_nodes),
        )

    def finish_step(self, step_id: int) -> None:
        step = _step_id(step_id)
        with self._condition:
            while step in self._executing:
                self._condition.wait()
            prepared = self._prepared.pop(step, None)
        if prepared is None:
            self._snapshots.finish_step(step)
            return
        stream_by_bucket = {
            node.buffer_bucket: prepared.streams[node.stream]
            for node in prepared.resources.plan.nodes
            if node.buffer_bucket is not None
        }
        for bucket_key, leases in prepared.leases.items():
            stream_resource = stream_by_bucket.get(bucket_key)
            stream = (
                None
                if stream_resource is None
                else cast(StreamLike, stream_resource.stream)
            )
            for lease in leases:
                if not lease.completion_fence.recorded:
                    lease.publish_completion(stream)
                    self.stats.fence_record_count += 1
                lease.publish_retire(stream)
                if lease.retire_fence is not None:
                    self.stats.fence_record_count += 1
        for resource in prepared.resources.streams:
            resource.retire_fence.record(
                cast(StreamLike, resource.stream)
            )
            self.stats.fence_record_count += 1
        self._snapshots.finish_step(step)
        collected = self._snapshots.collect_retired()
        with self._lock:
            self.stats.finish_count += 1
            self.stats.retired_plan_count += collected

    def _prepare(self, resources: PlanResources) -> _PreparedStep:
        streams = {resource.key: resource for resource in resources.streams}
        events = {resource.key: resource for resource in resources.events}
        graphs: dict[ArtifactKey, RuntimeGraph] = {
            resource.key: resource for resource in resources.graphs
        }
        missing_streams = {
            node.stream for node in resources.plan.nodes if node.stream not in streams
        }
        if missing_streams:
            names = ", ".join(sorted(stream.value for stream in missing_streams))
            raise PhysicalExecutionError(f"physical plan is missing streams: {names}")

        outgoing: dict[NodeKey, list[FenceKey]] = {}
        missing_events: set[FenceKey] = set()
        for node in resources.plan.nodes:
            for dependency in node.fences:
                outgoing.setdefault(dependency.producer, []).append(dependency.fence)
                if dependency.fence not in events:
                    missing_events.add(dependency.fence)
        if missing_events:
            names = ", ".join(sorted(str(key) for key in missing_events))
            raise PhysicalExecutionError(f"physical plan is missing events: {names}")

        resolved_handlers: dict[NodeKey, PhysicalNodeHandler] = {}
        placement_handlers = {
            binding.node_key: binding.execute
            for binding in resources.placement_implementations
        }
        for node in resources.plan.nodes:
            handler = placement_handlers.get(node.node_key)
            if handler is None:
                handler = self._bindings.get(node.executor)
            if handler is None:
                handler = self._bindings.get(node.node_key)
            if handler is None:
                raise PhysicalExecutionError(
                    f"physical node {node.node_key} has no fixed executor binding"
                )
            resolved_handlers[node.node_key] = handler

        leases: dict[BufferBucketKey, list[BufferLease]] = {}
        for lease in resources.leases:
            leases.setdefault(lease.bucket.key, []).append(lease)
        return _PreparedStep(
            resources=resources,
            streams=MappingProxyType(streams),
            events=MappingProxyType(events),
            graphs=MappingProxyType(graphs),
            handlers=MappingProxyType(resolved_handlers),
            outgoing_fences=MappingProxyType(
                {key: tuple(values) for key, values in outgoing.items()}
            ),
            leases=MappingProxyType(
                {key: tuple(values) for key, values in leases.items()}
            ),
        )

    def _request_specialization_on_miss(
        self,
        resources: PlanResources,
        node: PhysicalNodePlan,
        tensors: tuple[torch.Tensor, ...],
    ) -> bool:
        if node.buffer_bucket is None or not tensors:
            return False
        tensor = tensors[0]
        request = SpecializationRequestDesc(
            plan_epoch=resources.plan.plan_epoch,
            node_key=node.node_key,
            buffer_bucket=node.buffer_bucket,
            artifact_key=node.artifact_key,
            shape=tuple(int(dimension) for dimension in tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        selected = select_buffer_bucket(
            resources.plan,
            request.shape,
            request.dtype,
            request.device,
        )
        if selected is not None and selected.key == node.buffer_bucket:
            return False
        with self._lock:
            if request in self._requested_specializations:
                return True
            self._requested_specializations.add(request)
            self.stats.bucket_miss_count += 1
            callback = self._on_specialization_request
            if callback is not None:
                self.stats.specialization_request_count += 1
                self._pending_specializations.append(request)
        return True

    def drain_specialization_requests(self) -> int:
        """Run queued specialization callbacks on an explicit cold path."""

        with self._lock:
            pending = tuple(self._pending_specializations)
            self._pending_specializations.clear()
            callback = self._on_specialization_request
        if callback is None:
            return 0
        for index, request in enumerate(pending):
            try:
                callback(request)
            except BaseException:
                with self._lock:
                    remaining = pending[index:]
                    self._pending_specializations.extendleft(reversed(remaining))
                raise
        return len(pending)


def _step_id(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("step_id must be int")
    if value < 0:
        raise ValueError("step_id must be non-negative")
    return value


def _tensors(value: object) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, tuple):
        raise TypeError("tensors must be tuple")
    tensors = cast(tuple[object, ...], value)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("tensors must contain only torch.Tensor")
    return cast(tuple[torch.Tensor, ...], tensors)


__all__ = [
    "PhysicalExecutionContext",
    "PhysicalExecutionError",
    "PhysicalExecutionKind",
    "PhysicalExecutionStats",
    "PhysicalExecutor",
    "PhysicalNodeHandler",
    "PhysicalStepResult",
    "SpecializationRequestDesc",
    "SpecializationRequest",
    "WorkGraphRuntimeStats",
    "WorkGraphRuntimeStatsReader",
    "runtime_stats_from_executor",
]

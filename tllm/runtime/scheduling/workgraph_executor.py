"""Python-owned executor for frozen single-device WorkGraph plans."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import contextlib
from dataclasses import dataclass
from enum import StrEnum
from threading import Condition, Lock
from types import MappingProxyType
from typing import cast
from typing import Protocol

import torch

from tllm.contracts.runtime_stats import WorkGraphRuntimeStats
from tllm.contracts.shape_specialization import RegionShapeSpecializationRequest
from tllm.runtime.torch_adapter import torch_cuda_stream

from tllm.runtime.scheduling.physical import (
    BufferLease,
    EventLike,
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
from tllm.contracts.physical_plan import (
    ArtifactKey,
    BufferBucketKey,
    CapturePolicy,
    ExecutorKey,
    FallbackPolicy,
    FenceKey,
    NodeKey,
    PhysicalNodePlan,
    StreamClass,
)


class PhysicalExecutionError(RuntimeError):
    """Raised when a frozen plan cannot be executed under its declared policy."""


class ShapeSpecializationMiss(PhysicalExecutionError):
    """Fail a step before Region output commit and request cold specialization."""

    __slots__ = ("request",)

    def __init__(self, request: RegionShapeSpecializationRequest) -> None:
        if not isinstance(request, RegionShapeSpecializationRequest):
            raise TypeError("ShapeSpecializationMiss.request must be RegionShapeSpecializationRequest")
        self.request = request
        super().__init__(
            "Region shape specialization miss: "
            f"region={request.region_id!r}, epoch={request.observed_plan_epoch}"
        )


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
class PhysicalBatchExecutionContext:
    step_id: int
    plan_epoch: int
    nodes: tuple[PhysicalNodePlan, ...]
    stream: torch.cuda.Stream


class PhysicalNodeBatchGraph(Protocol):
    def preflight_context(self, context: PhysicalBatchExecutionContext) -> None: ...

    def replay_context(
        self, context: PhysicalBatchExecutionContext
    ) -> PhysicalExecutionKind: ...


@dataclass(frozen=True, slots=True)
class PhysicalStepResult:
    step_id: int
    plan_epoch: int
    eager_nodes: tuple[str, ...]
    replayed_nodes: tuple[str, ...]


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
SpecializationRequest = Callable[[RegionShapeSpecializationRequest], None]
SpecializationRequestFactory = Callable[
    [PhysicalNodePlan, int, tuple[torch.Tensor, ...]],
    RegionShapeSpecializationRequest,
]


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
        "_completed",
        "_requested_specializations",
        "_pending_specializations",
        "_on_specialization_request",
        "_specialization_request_factory",
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
        specialization_request_factory: SpecializationRequestFactory | None = None,
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
        if specialization_request_factory is not None and not callable(
            specialization_request_factory
        ):
            raise TypeError("specialization_request_factory must be callable")
        self._snapshots = snapshots
        self._bindings = MappingProxyType(bindings)
        self._prepared: dict[int, _PreparedStep] = {}
        self._completed: dict[int, set[NodeKey]] = {}
        self._requested_specializations: set[str] = set()
        self._pending_specializations: deque[RegionShapeSpecializationRequest] = deque()
        self._on_specialization_request = on_specialization_request
        self._specialization_request_factory = specialization_request_factory
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
            self._completed[step] = set()
            self._highest_prepared_step = step
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
        with self._condition:
            try:
                prepared = self._prepared[step]
            except KeyError as exc:
                raise PhysicalExecutionError(
                    f"physical step {step} must be prepared before execution"
                ) from exc
            nodes = tuple(node.node_key for node in prepared.resources.plan.nodes)
        return self.execute_nodes(step, nodes, tensors=tensors)

    def execute_nodes(
        self,
        step_id: int,
        node_keys: tuple[NodeKey, ...],
        *,
        tensors: tuple[torch.Tensor, ...] = (),
        main_snapshot: torch.cuda.Event | None = None,
        batch_graph: PhysicalNodeBatchGraph | None = None,
    ) -> PhysicalStepResult:
        """Execute an ordered Region subset from one prepared physical plan.

        This is the anchor-facing entry point.  It never derives a new order:
        callers identify nodes, while execution retains their immutable plan
        order and rejects missing predecessors or duplicate submission.
        """

        step = _step_id(step_id)
        inputs = _tensors(tensors)
        if not isinstance(node_keys, tuple) or not node_keys:
            raise TypeError("node_keys must be a non-empty tuple[NodeKey, ...]")
        if any(not isinstance(node_key, NodeKey) for node_key in node_keys):
            raise TypeError("node_keys must contain only NodeKey")
        if len(set(node_keys)) != len(node_keys):
            raise PhysicalExecutionError("physical node subset contains duplicates")
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
            plan_nodes = prepared.resources.plan.nodes
            node_by_key = {node.node_key: node for node in plan_nodes}
            unknown = tuple(node_key for node_key in node_keys if node_key not in node_by_key)
            if unknown:
                names = ", ".join(str(node_key) for node_key in unknown)
                raise PhysicalExecutionError(
                    f"physical node subset references unknown nodes: {names}"
                )
            completed = self._completed[step]
            repeated = tuple(node_key for node_key in node_keys if node_key in completed)
            if repeated:
                names = ", ".join(str(node_key) for node_key in repeated)
                raise PhysicalExecutionError(
                    f"physical nodes already executed for step {step}: {names}"
                )
            selected = set(node_keys)
            ordered_nodes = tuple(
                node for node in plan_nodes if node.node_key in selected
            )
            if batch_graph is not None and tuple(
                node.node_key for node in ordered_nodes
            ) != node_keys:
                raise PhysicalExecutionError(
                    "captured physical batch order differs from its signed command"
                )
            available = set(completed)
            for node in ordered_nodes:
                missing = tuple(
                    dependency.producer
                    for dependency in node.fences
                    if dependency.producer not in available
                )
                if missing:
                    names = ", ".join(str(node_key) for node_key in missing)
                    raise PhysicalExecutionError(
                        f"physical node {node.node_key} has unexecuted predecessors: {names}"
                    )
                available.add(node.node_key)
            self._executing.add(step)
            self.stats.execute_count += 1

        eager_nodes: list[str] = []
        replayed_nodes: list[str] = []
        try:
            # A miss must abort an entire backend event group before a prior
            # Region can mutate state, publish a sink, or record a fence.
            if batch_graph is None:
                self._preflight_nodes(step, prepared, ordered_nodes, inputs)
            else:
                batch_streams = {node.stream for node in ordered_nodes}
                if len(batch_streams) != 1:
                    raise PhysicalExecutionError(
                        "captured physical batch requires one signed stream"
                    )
                if any(node.buffer_bucket is not None for node in ordered_nodes):
                    raise PhysicalExecutionError(
                        "captured physical batch cannot infer buffer-bucket ownership"
                    )
                batch_stream_class = next(iter(batch_streams))
                batch_stream = cast(
                    StreamLike, prepared.streams[batch_stream_class].stream
                )
                batch_cuda_stream = torch_cuda_stream(batch_stream)
                batch_context = PhysicalBatchExecutionContext(
                    step_id=step,
                    plan_epoch=prepared.resources.plan.plan_epoch,
                    nodes=ordered_nodes,
                    stream=batch_cuda_stream,
                )
                batch_graph.preflight_context(batch_context)
                if batch_stream_class is not StreamClass.DEFAULT:
                    if main_snapshot is None:
                        main_snapshot = torch.cuda.Event()
                        main_snapshot.record()
                    batch_cuda_stream.wait_event(main_snapshot)
                selected_batch_nodes = {
                    node.node_key for node in ordered_nodes
                }
                for node in ordered_nodes:
                    for dependency in node.fences:
                        if dependency.producer in selected_batch_nodes:
                            continue
                        event = prepared.events[dependency.fence]
                        batch_stream.wait_event(event.event)
                        self.stats.fence_wait_count += 1
                with torch.cuda.stream(batch_cuda_stream):
                    batch_kind = batch_graph.replay_context(batch_context)
                for node in ordered_nodes:
                    for fence_key in prepared.outgoing_fences.get(
                        node.node_key, ()
                    ):
                        cast(EventLike, prepared.events[fence_key].event).record(
                            batch_cuda_stream
                        )
                        self.stats.fence_record_count += 1
                    with self._condition:
                        self._completed[step].add(node.node_key)
                if batch_kind is PhysicalExecutionKind.EAGER:
                    eager_nodes.extend(
                        str(node.node_key) for node in ordered_nodes
                    )
                    self.stats.eager_count += 1
                else:
                    replayed_nodes.extend(
                        str(node.node_key) for node in ordered_nodes
                    )
                    self.stats.replay_count += 1
                return PhysicalStepResult(
                    step_id=step,
                    plan_epoch=prepared.resources.plan.plan_epoch,
                    eager_nodes=tuple(eager_nodes),
                    replayed_nodes=tuple(replayed_nodes),
                )
            for node in ordered_nodes:
                stream_resource = prepared.streams[node.stream]
                stream = cast(StreamLike, stream_resource.stream)
                if node.stream is not StreamClass.DEFAULT and isinstance(
                    stream, torch.cuda.Stream
                ):
                    # Side-stream nodes may read main-path outputs (hidden
                    # activations, provided values) produced before this
                    # advance.  Host-side hook ordering does not imply GPU
                    # completion, so the side stream first waits a snapshot
                    # of everything the main stream has enqueued so far.
                    if main_snapshot is None:
                        main_snapshot = torch.cuda.Event()
                        main_snapshot.record()
                    stream.wait_event(main_snapshot)
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

                graph_resource = (
                    None
                    if node.artifact_key is None
                    else prepared.graphs.get(node.artifact_key)
                )
                # GPU work for a side-stream node must launch on its signed
                # stream, not on whatever stream the host happens to be on.
                execution_stream = (
                    torch.cuda.stream(stream)
                    if isinstance(stream, torch.cuda.Stream)
                    else contextlib.nullcontext()
                )
                with execution_stream:
                    if (
                        node.capture_policy != CapturePolicy.DISABLED
                        and graph_resource is not None
                    ):
                        replay_context = PhysicalExecutionContext(
                            step_id=step,
                            plan_epoch=prepared.resources.plan.plan_epoch,
                            node=node,
                            kind=PhysicalExecutionKind.REPLAY,
                            stream=stream,
                            tensors=inputs,
                            leases=node_leases,
                        )
                        graph_resource.replay(replay_context)
                        replayed_nodes.append(str(node.node_key))
                        self.stats.replay_count += 1
                    else:
                        if node.capture_policy != CapturePolicy.DISABLED:
                            if node.fallback_policy == FallbackPolicy.REJECT:
                                reason = "capture graph"
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
                with self._condition:
                    self._completed[step].add(node.node_key)
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

    def _preflight_nodes(
        self,
        step: int,
        prepared: _PreparedStep,
        nodes: tuple[PhysicalNodePlan, ...],
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Reject all known shape/capture failures before submitting a node.

        A RuntimeGraph may resolve native Region inputs here, but its preflight
        contract must not replay a graph, invoke a handler, write a sink, or
        record a fence. Native setup rejects capture inputs that require a
        materialization from another node in this same invocation.
        """

        for node in nodes:
            specialization_request = self._request_specialization_on_miss(
                prepared.resources,
                node,
                tensors,
            )
            if specialization_request is not None:
                raise ShapeSpecializationMiss(specialization_request)
            graph_resource = (
                None
                if node.artifact_key is None
                else prepared.graphs.get(node.artifact_key)
            )
            if node.capture_policy == CapturePolicy.DISABLED:
                continue
            if graph_resource is None:
                if node.fallback_policy == FallbackPolicy.REJECT:
                    raise PhysicalExecutionError(
                        f"physical node {node.node_key} requires capture graph"
                    )
                continue
            node_leases = (
                ()
                if node.buffer_bucket is None
                else prepared.leases.get(node.buffer_bucket, ())
            )
            stream = cast(StreamLike, prepared.streams[node.stream].stream)
            graph_resource.preflight(
                PhysicalExecutionContext(
                    step_id=step,
                    plan_epoch=prepared.resources.plan.plan_epoch,
                    node=node,
                    kind=PhysicalExecutionKind.REPLAY,
                    stream=stream,
                    tensors=tensors,
                    leases=node_leases,
                )
            )

    def finish_step(self, step_id: int) -> None:
        step = _step_id(step_id)
        with self._condition:
            while step in self._executing:
                self._condition.wait()
            prepared = self._prepared.pop(step, None)
            self._completed.pop(step, None)
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
    ) -> RegionShapeSpecializationRequest | None:
        if node.buffer_bucket is None or not tensors:
            return None
        tensor = tensors[0]
        selected = select_buffer_bucket(
            resources.plan,
            tuple(int(dimension) for dimension in tensor.shape),
            tensor.dtype,
            tensor.device,
        )
        if selected is not None and selected.key == node.buffer_bucket:
            return None
        factory = self._specialization_request_factory
        if factory is None:
            raise PhysicalExecutionError(
                f"physical node {node.node_key} has a shape miss without a Region specialization factory"
            )
        request = factory(node, resources.plan.plan_epoch, tensors)
        if not isinstance(request, RegionShapeSpecializationRequest):
            raise TypeError(
                "specialization_request_factory must return RegionShapeSpecializationRequest"
            )
        with self._lock:
            if request.dedup_key not in self._requested_specializations:
                self._requested_specializations.add(request.dedup_key)
                self.stats.bucket_miss_count += 1
                callback = self._on_specialization_request
                if callback is not None:
                    self.stats.specialization_request_count += 1
                    self._pending_specializations.append(request)
        return request

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
    "PhysicalBatchExecutionContext",
    "PhysicalExecutionError",
    "PhysicalExecutionKind",
    "PhysicalExecutionStats",
    "PhysicalExecutor",
    "PhysicalNodeHandler",
    "PhysicalNodeBatchGraph",
    "PhysicalStepResult",
    "SpecializationRequest",
    "SpecializationRequestFactory",
    "ShapeSpecializationMiss",
    "WorkGraphRuntimeStats",
    "WorkGraphRuntimeStatsReader",
    "runtime_stats_from_executor",
]

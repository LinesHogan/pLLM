from __future__ import annotations

import threading
import unittest

import torch

from tllm.runtime.scheduling.physical import Fence
from tllm.runtime.scheduling.plan_store import (
    PlanResources,
    PlanSnapshotStore,
    RuntimeEvent,
    RuntimeGraph,
    RuntimeStream,
)
from tllm.runtime.scheduling.workgraph_executor import (
    PhysicalExecutionError,
    PhysicalExecutionKind,
    PhysicalExecutor,
    SpecializationRequestDesc,
)
from tllm.workgraph.frontend import ArtifactKey, Device, TensorDTypeDesc, TensorDeviceDesc
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.physical import (
    BufferBucketDesc,
    BufferBucketKey,
    BufferOwnershipKind,
    CapturePolicy,
    FallbackPolicy,
    FenceDependencyDesc,
    FenceKey,
    FenceKind,
    PhysicalFlowPlan,
    PhysicalNodePlan,
    StreamClass,
)


class _FakeEvent:
    def __init__(self, name: str, trace: list[str]) -> None:
        self.name = name
        self.trace = trace
        self.done = True

    def record(self, stream: object) -> None:
        del stream
        self.trace.append(f"record:{self.name}")

    def query(self) -> bool:
        self.trace.append(f"query:{self.name}")
        return self.done


class _FakeStream:
    def __init__(self, name: str, trace: list[str]) -> None:
        self.name = name
        self.trace = trace

    def wait_event(self, event: _FakeEvent) -> None:
        self.trace.append(f"wait:{self.name}:{event.name}")


class _FakeGraph:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def replay(self) -> None:
        self.trace.append("graph:replay")


def _artifact(name: str = "capture") -> ArtifactKey:
    return ArtifactKey("physical", name, digest="a" * 64)


def _bucket() -> BufferBucketDesc:
    return BufferBucketDesc(
        key=BufferBucketKey("physical", "hidden"),
        shape=(2, 3),
        dtype=TensorDTypeDesc(namespace="torch", name="float32"),
        device=TensorDeviceDesc(Device.CPU),
        slots=1,
        ownership=BufferOwnershipKind.MANAGED,
    )


def _plan(
    *,
    epoch: int = 1,
    capture_policy: CapturePolicy = CapturePolicy.DISABLED,
    fallback_policy: FallbackPolicy = FallbackPolicy.EAGER,
    with_dependency: bool = False,
    with_bucket: bool = False,
) -> tuple[PhysicalFlowPlan, NodeKey, NodeKey, FenceKey | None, ArtifactKey | None, BufferBucketDesc | None]:
    producer = NodeKey.producer("physical.source")
    consumer = NodeKey.consumer_flow("physical.consumer", 0)
    dependency = FenceKey("physical", "dependency") if with_dependency else None
    artifact = _artifact() if capture_policy != CapturePolicy.DISABLED else None
    bucket = _bucket() if with_bucket else None
    consumer_fences = ()
    if dependency is not None:
        consumer_fences = (
            FenceDependencyDesc(
                fence=dependency,
                producer=producer,
                consumer=consumer,
                kind=FenceKind.READY,
            ),
        )
    producer_node = PhysicalNodePlan(
        node_key=producer,
        executor=ExecutorKey("producer", "physical.source"),
        stream=StreamClass.DEFAULT,
    )
    consumer_node = PhysicalNodePlan(
        node_key=consumer,
        executor=ExecutorKey("consumer", "physical.consumer"),
        stream=StreamClass.SIDE_TRAIN,
        fences=consumer_fences,
        artifact_key=artifact,
        buffer_bucket=None if bucket is None else bucket.key,
        capture_policy=capture_policy,
        fallback_policy=fallback_policy,
    )
    return (
        PhysicalFlowPlan(
            plan_epoch=epoch,
            nodes=(producer_node, consumer_node),
            buffer_buckets=() if bucket is None else (bucket,),
        ),
        producer,
        consumer,
        dependency,
        artifact,
        bucket,
    )


def _resources(
    plan: PhysicalFlowPlan,
    *,
    trace: list[str],
    dependency: FenceKey | None = None,
    graph: _FakeGraph | None = None,
    artifact: ArtifactKey | None = None,
) -> PlanResources:
    events = ()
    if dependency is not None:
        events = (RuntimeEvent(dependency, _FakeEvent("dependency", trace)),)
    graphs = ()
    if graph is not None:
        assert artifact is not None
        graphs = (RuntimeGraph(artifact, graph),)
    streams = tuple(
        RuntimeStream(
            stream_class,
            _FakeStream(stream_class.value, trace),
            Fence(
                FenceKey("physical", f"retire-{stream_class.value}"),
                _FakeEvent(f"retire-{stream_class.value}", trace),
            ),
        )
        for stream_class in (StreamClass.DEFAULT, StreamClass.SIDE_TRAIN)
        if any(node.stream == stream_class for node in plan.nodes)
    )
    return PlanResources(
        plan=plan,
        streams=streams,
        events=events,
        graphs=graphs,
    )


class PhysicalFlowExecutorUnitTest(unittest.TestCase):
    def test_finish_rejects_same_step_reactivation_during_and_after_retirement(self) -> None:
        """A finishing step must remain closed until and after retirement completes."""

        trace: list[str] = []
        plan, producer, consumer, _dependency, _artifact, _bucket_desc = _plan()
        retire_record_started = threading.Event()
        allow_retire_record = threading.Event()

        class BlockingRetireEvent(_FakeEvent):
            def record(self, stream: object) -> None:
                super().record(stream)
                retire_record_started.set()
                allow_retire_record.wait(timeout=2.0)

        resources = PlanResources(
            plan=plan,
            streams=(
                RuntimeStream(
                    StreamClass.DEFAULT,
                    _FakeStream(StreamClass.DEFAULT.value, trace),
                    Fence(
                        FenceKey("physical", "retire-default"),
                        BlockingRetireEvent("retire-default", trace),
                    ),
                ),
                RuntimeStream(
                    StreamClass.SIDE_TRAIN,
                    _FakeStream(StreamClass.SIDE_TRAIN.value, trace),
                    Fence(
                        FenceKey("physical", "retire-side-train"),
                        _FakeEvent("retire-side-train", trace),
                    ),
                ),
            ),
        )
        executor = PhysicalExecutor(
            snapshots=PlanSnapshotStore(resources),
            handlers={
                producer: lambda _context: None,
                consumer: lambda _context: None,
            },
        )
        executor.prepare_step(1)
        executor.execute_step(1)
        finish_errors: list[BaseException] = []

        def finish() -> None:
            try:
                executor.finish_step(1)
            except BaseException as exc:
                finish_errors.append(exc)

        finish_thread = threading.Thread(target=finish)
        finish_thread.start()
        try:
            self.assertTrue(retire_record_started.wait(timeout=2.0))

            during_prepare_error: BaseException | None = None
            during_execute_error: BaseException | None = None
            try:
                executor.prepare_step(1)
            except BaseException as exc:
                during_prepare_error = exc
            try:
                executor.execute_step(1)
            except BaseException as exc:
                during_execute_error = exc

            self.assertIsInstance(during_prepare_error, PhysicalExecutionError)
            self.assertIsInstance(during_execute_error, PhysicalExecutionError)
        finally:
            allow_retire_record.set()
            finish_thread.join(timeout=2.0)

        self.assertFalse(finish_thread.is_alive())
        self.assertEqual(finish_errors, [])
        with self.assertRaises(PhysicalExecutionError):
            executor.prepare_step(1)
        with self.assertRaises(PhysicalExecutionError):
            executor.execute_step(1)

    def test_finish_waits_for_an_in_progress_execute_before_recording_retire(self) -> None:
        """A concurrent finish must not retire resources while a handler runs."""

        trace: list[str] = []
        plan, producer, consumer, _dependency, _artifact, _bucket_desc = _plan()
        default_retire = _FakeEvent("retire-default", trace)
        side_train_retire = _FakeEvent("retire-side-train", trace)
        resources = PlanResources(
            plan=plan,
            streams=(
                RuntimeStream(
                    StreamClass.DEFAULT,
                    _FakeStream(StreamClass.DEFAULT.value, trace),
                    Fence(FenceKey("physical", "retire-default"), default_retire),
                ),
                RuntimeStream(
                    StreamClass.SIDE_TRAIN,
                    _FakeStream(StreamClass.SIDE_TRAIN.value, trace),
                    Fence(FenceKey("physical", "retire-side-train"), side_train_retire),
                ),
            ),
        )
        entered_handler = threading.Event()
        release_handler = threading.Event()
        execute_done = threading.Event()
        finish_done = threading.Event()

        def blocking_handler(_context: object) -> None:
            entered_handler.set()
            self.assertTrue(release_handler.wait(timeout=2.0))

        executor = PhysicalExecutor(
            snapshots=PlanSnapshotStore(resources),
            handlers={
                producer: blocking_handler,
                consumer: lambda _context: None,
            },
        )
        executor.prepare_step(1)

        def execute() -> None:
            try:
                executor.execute_step(1)
            finally:
                execute_done.set()

        def finish() -> None:
            try:
                executor.finish_step(1)
            finally:
                finish_done.set()

        execute_thread = threading.Thread(target=execute)
        finish_thread = threading.Thread(target=finish)
        execute_thread.start()
        self.assertTrue(entered_handler.wait(timeout=2.0))
        finish_thread.start()
        try:
            self.assertFalse(
                finish_done.wait(timeout=0.1),
                "finish must wait for the in-progress execute before retiring streams",
            )
            self.assertNotIn("record:retire-default", trace)
            self.assertNotIn("record:retire-side-train", trace)
        finally:
            release_handler.set()
            execute_thread.join(timeout=2.0)
            finish_thread.join(timeout=2.0)

        self.assertTrue(execute_done.is_set())
        self.assertTrue(finish_done.is_set())

    def test_finish_step_retires_plan_streams_without_buffer_leases(self) -> None:
        trace: list[str] = []
        plan, producer, consumer, _dependency, _artifact, _bucket_desc = _plan()
        default_event = _FakeEvent("retire-default", trace)
        side_train_event = _FakeEvent("retire-side-train", trace)
        default_event.done = False
        side_train_event.done = False
        resources = PlanResources(
            plan=plan,
            streams=(
                RuntimeStream(
                    StreamClass.DEFAULT,
                    _FakeStream(StreamClass.DEFAULT.value, trace),
                    retire_fence=Fence(
                        FenceKey("physical", "retire-default"),
                        default_event,
                    ),
                ),
                RuntimeStream(
                    StreamClass.SIDE_TRAIN,
                    _FakeStream(StreamClass.SIDE_TRAIN.value, trace),
                    retire_fence=Fence(
                        FenceKey("physical", "retire-side-train"),
                        side_train_event,
                    ),
                ),
            ),
        )
        replacement_plan, *_ = _plan(epoch=2)
        replacement = _resources(replacement_plan, trace=trace)
        snapshots = PlanSnapshotStore(resources)
        executor = PhysicalExecutor(
            snapshots=snapshots,
            handlers={
                producer: lambda _ctx: None,
                consumer: lambda _ctx: None,
            },
        )

        executor.prepare_step(1)
        executor.execute_step(1)
        executor.finish_step(1)
        default_event.done = True
        snapshots.stage(replacement)
        executor.prepare_step(2)
        executor.finish_step(2)

        self.assertIn("record:retire-default", trace)
        self.assertIn("record:retire-side-train", trace)
        self.assertIn("query:retire-default", trace)
        self.assertIn("query:retire-side-train", trace)
        self.assertEqual(snapshots.retired_count, 1)

        default_event.done = True
        side_train_event.done = True
        self.assertEqual(snapshots.collect_retired(), 1)
        self.assertEqual(snapshots.retired_count, 0)

    def test_node_bindings_are_fixed_and_steps_keep_their_prepared_snapshot(self) -> None:
        trace: list[str] = []
        plan, producer, consumer, _dependency, _artifact_key, _bucket_desc = _plan()
        resources = _resources(plan, trace=trace)
        replacement_plan, *_ = _plan(epoch=2)
        replacement = _resources(replacement_plan, trace=trace)
        snapshots = PlanSnapshotStore(resources)
        calls: list[str] = []

        def first(ctx: object) -> None:
            del ctx
            calls.append("first")

        def second(ctx: object) -> None:
            del ctx
            calls.append("second")

        handlers = {
            producer: first,
            consumer: lambda _ctx: None,
        }
        executor = PhysicalExecutor(snapshots=snapshots, handlers=handlers)

        prepared = executor.prepare_step(7)
        snapshots.stage(replacement)
        self.assertIs(executor.prepare_step(7), prepared)
        handlers[producer] = second
        executor.execute_step(7)
        executor.finish_step(7)

        self.assertEqual(calls, ["first"])
        with self.assertRaises(KeyError):
            snapshots.active(7)
        self.assertIs(executor.prepare_step(8), replacement)

    def test_prepare_execute_finish_and_replay_eager_statistics(self) -> None:
        trace: list[str] = []
        plan, producer, consumer, dependency, artifact, _bucket_desc = _plan(
            capture_policy=CapturePolicy.PREFER,
            with_dependency=True,
        )
        resources = _resources(
            plan,
            trace=trace,
            dependency=dependency,
            graph=_FakeGraph(trace),
            artifact=artifact,
        )
        calls: list[tuple[str, PhysicalExecutionKind]] = []
        executor = PhysicalExecutor(
            snapshots=PlanSnapshotStore(resources),
            handlers={
                producer: lambda ctx: calls.append(
                    (str(ctx.node.node_key), ctx.kind)
                ),
                consumer: lambda _ctx: None,
            },
        )

        executor.prepare_step(1)
        result = executor.execute_step(1)
        executor.finish_step(1)

        self.assertEqual(result.replayed_nodes, (str(consumer),))
        self.assertEqual(calls, [(str(producer), PhysicalExecutionKind.EAGER)])
        self.assertEqual(
            trace[:3],
            ["record:dependency", "wait:side_train:dependency", "graph:replay"],
        )
        self.assertEqual(executor.stats.prepare_count, 1)
        self.assertEqual(executor.stats.execute_count, 1)
        self.assertEqual(executor.stats.replay_count, 1)
        self.assertEqual(executor.stats.eager_count, 1)

    def test_preferred_capture_without_graph_is_eager_but_reject_policies_fail(self) -> None:
        for fallback_policy in (FallbackPolicy.EAGER, FallbackPolicy.REJECT):
            with self.subTest(fallback_policy=fallback_policy):
                trace: list[str] = []
                plan, producer, consumer, _dependency, artifact, _bucket_desc = _plan(
                    capture_policy=CapturePolicy.PREFER,
                    fallback_policy=fallback_policy,
                )
                resources = _resources(plan, trace=trace)
                calls: list[PhysicalExecutionKind] = []
                executor = PhysicalExecutor(
                    snapshots=PlanSnapshotStore(resources),
                    handlers={
                        producer: lambda _ctx: None,
                        consumer: lambda ctx: calls.append(
                            ctx.kind
                        ),
                    },
                )
                executor.prepare_step(1)
                if fallback_policy == FallbackPolicy.EAGER:
                    result = executor.execute_step(1)
                    self.assertEqual(
                        result.eager_nodes,
                        (str(producer), str(consumer)),
                    )
                    self.assertEqual(calls, [PhysicalExecutionKind.EAGER])
                    self.assertEqual(executor.stats.eager_count, 2)
                else:
                    with self.assertRaisesRegex(PhysicalExecutionError, "capture"):
                        executor.execute_step(1)
                    self.assertEqual(calls, [])
                executor.finish_step(1)

        trace = []
        plan, producer, consumer, _dependency, _artifact, _bucket_desc = _plan(
            capture_policy=CapturePolicy.REQUIRE,
            fallback_policy=FallbackPolicy.REJECT,
        )
        executor = PhysicalExecutor(
            snapshots=PlanSnapshotStore(_resources(plan, trace=trace)),
            handlers={
                producer: lambda _ctx: None,
                consumer: lambda _ctx: self.fail(
                    "required capture must not call eager handler"
                ),
            },
        )
        executor.prepare_step(1)
        with self.assertRaisesRegex(PhysicalExecutionError, "capture"):
            executor.execute_step(1)

    def test_producer_fence_is_recorded_after_node_and_bucket_miss_requests_once(self) -> None:
        trace: list[str] = []
        plan, producer, consumer, dependency, _artifact, bucket = _plan(
            with_dependency=True,
            with_bucket=True,
        )
        assert dependency is not None
        assert bucket is not None
        resources = _resources(plan, trace=trace, dependency=dependency)
        calls: list[PhysicalExecutionKind] = []
        specialization_requests: list[SpecializationRequestDesc] = []
        executor = PhysicalExecutor(
            snapshots=PlanSnapshotStore(resources),
            handlers={
                producer: lambda ctx: calls.append(
                    ctx.kind
                ),
                consumer: lambda ctx: calls.append(
                    ctx.kind
                ),
            },
            on_specialization_request=lambda request: specialization_requests.append(
                request
            ),
        )
        mismatch = torch.empty((2, 4), dtype=torch.float32)

        executor.prepare_step(1)
        executor.execute_step(1, tensors=(mismatch,))
        executor.finish_step(1)
        executor.prepare_step(2)
        executor.execute_step(2, tensors=(mismatch,))

        self.assertEqual(calls, [PhysicalExecutionKind.EAGER] * 4)
        # A bucket miss is a hot-path fallback plus a deduplicated cold request;
        # the callback itself must only run when an explicit cold-path drain is
        # requested.
        self.assertEqual(specialization_requests, [])
        self.assertEqual(executor.stats.bucket_miss_count, 1)
        self.assertEqual(executor.stats.specialization_request_count, 1)
        self.assertEqual(executor.drain_specialization_requests(), 1)
        self.assertEqual(
            specialization_requests,
            [
                SpecializationRequestDesc(
                    plan_epoch=1,
                    node_key=consumer,
                    buffer_bucket=bucket.key,
                    artifact_key=None,
                    shape=(2, 4),
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
            ],
        )
        self.assertEqual(executor.drain_specialization_requests(), 0)
        self.assertEqual(
            trace[:2],
            ["record:dependency", "wait:side_train:dependency"],
        )
        self.assertEqual(trace.count("record:dependency"), 2)
        self.assertIn("record:dependency", trace)
        self.assertEqual(str(producer), "producer/physical.source/0")

    def test_specialization_requests_are_scoped_to_epoch_node_bucket_and_artifact(self) -> None:
        """Equal Tensor shapes must not coalesce distinct physical misses."""

        first_bucket = BufferBucketDesc(
            key=BufferBucketKey("physical", "first"),
            shape=(2, 3),
            dtype=TensorDTypeDesc(namespace="torch", name="float32"),
            device=TensorDeviceDesc(Device.CPU),
        )
        second_bucket = BufferBucketDesc(
            key=BufferBucketKey("physical", "second"),
            shape=(2, 3),
            dtype=TensorDTypeDesc(namespace="torch", name="float32"),
            device=TensorDeviceDesc(Device.CPU),
        )
        first_node = NodeKey.consumer_flow("physical.first", 0)
        second_node = NodeKey.consumer_flow("physical.second", 0)
        first_artifact = ArtifactKey("physical", "first", digest="b" * 64)
        second_artifact = ArtifactKey("physical", "second", digest="c" * 64)

        def miss_plan(epoch: int) -> PhysicalFlowPlan:
            return PhysicalFlowPlan(
                plan_epoch=epoch,
                nodes=(
                    PhysicalNodePlan(
                        node_key=first_node,
                        executor=ExecutorKey("consumer", "physical.first"),
                        stream=StreamClass.SIDE_TRAIN,
                        artifact_key=first_artifact,
                        buffer_bucket=first_bucket.key,
                        capture_policy=CapturePolicy.PREFER,
                    ),
                    PhysicalNodePlan(
                        node_key=second_node,
                        executor=ExecutorKey("consumer", "physical.second"),
                        stream=StreamClass.SIDE_TRAIN,
                        artifact_key=second_artifact,
                        buffer_bucket=second_bucket.key,
                        capture_policy=CapturePolicy.PREFER,
                    ),
                ),
                buffer_buckets=(first_bucket, second_bucket),
            )

        trace: list[str] = []
        initial = _resources(miss_plan(1), trace=trace)
        replacement = _resources(miss_plan(2), trace=trace)
        observed: list[object] = []
        snapshots = PlanSnapshotStore(initial)
        executor = PhysicalExecutor(
            snapshots=snapshots,
            handlers={
                ExecutorKey("consumer", "physical.first"): lambda _context: None,
                ExecutorKey("consumer", "physical.second"): lambda _context: None,
            },
            # The cold path needs the complete request identity to construct
            # a correct per-region replacement plan.
            on_specialization_request=lambda request: observed.append(request),  # type: ignore[arg-type]
        )
        mismatch = torch.empty((2, 4), dtype=torch.float32)

        executor.prepare_step(1)
        executor.execute_step(1, tensors=(mismatch,))
        executor.finish_step(1)
        snapshots.stage(replacement)
        executor.prepare_step(2)
        executor.execute_step(2, tensors=(mismatch,))

        self.assertEqual(executor.drain_specialization_requests(), 4)
        self.assertEqual(
            [
                (
                    request.plan_epoch,
                    request.node_key,
                    request.buffer_bucket,
                    request.artifact_key,
                )
                for request in observed
            ],
            [
                (1, first_node, first_bucket.key, first_artifact),
                (1, second_node, second_bucket.key, second_artifact),
                (2, first_node, first_bucket.key, first_artifact),
                (2, second_node, second_bucket.key, second_artifact),
            ],
        )


if __name__ == "__main__":
    unittest.main()

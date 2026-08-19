from __future__ import annotations

import gc
import unittest
from unittest import mock
import weakref

import torch

from tllm.runtime.scheduling.physical import BufferLease, Fence
from tllm.runtime.scheduling.plan_store import (
    PlanResources,
    PlanSnapshotStore,
    RuntimeEvent,
    RuntimeGraph,
    RuntimeStream,
    select_buffer_bucket,
)
from tllm.workgraph.frontend import (
    ArtifactKey,
    Device,
    TensorDTypeDesc,
    TensorDeviceDesc,
)
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
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.signature import structural_signature


class _FakeEvent:
    def __init__(self) -> None:
        self.done = False

    def record(self, stream: object) -> None:
        del stream

    def query(self) -> bool:
        return self.done


class _FakeStream:
    def wait_event(self, event: object) -> None:
        del event


class _FakeGraph:
    def replay(self) -> None:
        pass


def _runtime_stream(key: StreamClass, fence_name: str) -> RuntimeStream:
    return RuntimeStream(
        key=key,
        stream=_FakeStream(),
        retire_fence=Fence(
            FenceKey("snapshot", fence_name),
            _FakeEvent(),
        ),
    )


def _bucket(
    name: str,
    *,
    shape: tuple[int, ...] = (2, 3),
    dtype: str = "float32",
    device: Device = Device.CPU,
    device_index: int | None = None,
    slots: int = 1,
) -> BufferBucketDesc:
    return BufferBucketDesc(
        key=BufferBucketKey("snapshot", name),
        shape=shape,
        dtype=TensorDTypeDesc(namespace="torch", name=dtype),
        device=TensorDeviceDesc(device, index=device_index),
        slots=slots,
        ownership=BufferOwnershipKind.MANAGED,
    )


def _plan(epoch: int, buckets: tuple[BufferBucketDesc, ...] = ()) -> PhysicalFlowPlan:
    return PhysicalFlowPlan(plan_epoch=epoch, nodes=(), buffer_buckets=buckets)


def _runtime_plan(
    epoch: int,
    bucket: BufferBucketDesc,
) -> tuple[PhysicalFlowPlan, FenceKey, ArtifactKey]:
    producer = NodeKey.producer("snapshot.source")
    consumer = NodeKey.consumer_flow("snapshot.consumer", 0)
    fence_key = FenceKey("snapshot", "ready")
    artifact_key = ArtifactKey("snapshot", "capture", digest="a" * 64)
    dependency = FenceDependencyDesc(
        fence=fence_key,
        producer=producer,
        consumer=consumer,
        kind=FenceKind.READY,
    )
    plan = PhysicalFlowPlan(
        plan_epoch=epoch,
        nodes=(
            PhysicalNodePlan(
                node_key=producer,
                executor=ExecutorKey("producer", "snapshot.source"),
                stream=StreamClass.DEFAULT,
            ),
            PhysicalNodePlan(
                node_key=consumer,
                executor=ExecutorKey("consumer", "snapshot.consumer"),
                stream=StreamClass.SIDE_TRAIN,
                fences=(dependency,),
                artifact_key=artifact_key,
                buffer_bucket=bucket.key,
                capture_policy=CapturePolicy.PREFER,
                fallback_policy=FallbackPolicy.EAGER,
            ),
        ),
        buffer_buckets=(bucket,),
    )
    return plan, fence_key, artifact_key


def _lease(
    plan_epoch: int,
    bucket: BufferBucketDesc,
    slot: int,
) -> tuple[BufferLease, _FakeEvent, _FakeEvent]:
    completion_event = _FakeEvent()
    retire_event = _FakeEvent()
    lease = BufferLease(
        bucket=bucket,
        tensor=torch.empty(bucket.shape, dtype=torch.float32),
        plan_epoch=plan_epoch,
        slot=slot,
        ready_fence=None,
        completion_fence=Fence(
            FenceKey("snapshot", f"completion-{slot}"), completion_event
        ),
        retire_fence=Fence(FenceKey("snapshot", f"retire-{slot}"), retire_event),
    )
    return lease, completion_event, retire_event


class PhysicalPlanSnapshotUnitTest(unittest.TestCase):
    def test_stream_retire_fence_key_has_one_event_identity_across_runtime_events_and_leases(self) -> None:
        bucket = _bucket("retire-identity")
        plan, ready_key, _capture_key = _runtime_plan(1, bucket)

        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                streams=(
                    RuntimeStream(
                        StreamClass.DEFAULT,
                        _FakeStream(),
                        Fence(ready_key, _FakeEvent()),
                    ),
                ),
                events=(RuntimeEvent(ready_key, _FakeEvent()),),
            )

        lease, _completion_event, _retire_event = _lease(1, bucket, 0)
        assert lease.retire_fence is not None
        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                leases=(lease,),
                streams=(
                    RuntimeStream(
                        StreamClass.DEFAULT,
                        _FakeStream(),
                        Fence(lease.retire_fence.key, _FakeEvent()),
                    ),
                ),
            )

    def test_step_snapshot_is_fixed_until_the_next_prepare_safe_point(self) -> None:
        initial = PlanResources(plan=_plan(1))
        staged = PlanResources(plan=_plan(2))
        store = PlanSnapshotStore(initial)

        self.assertIs(store.current, initial)
        self.assertIs(store.prepare_step(10), initial)
        store.stage(staged)

        self.assertIs(store.active(10), initial)
        self.assertIs(store.prepare_step(10), initial)
        self.assertIs(store.prepare_step(11), staged)
        self.assertIs(store.current, staged)
        self.assertIs(store.active(11), staged)

    def test_stage_accepts_only_a_strictly_higher_plan_epoch(self) -> None:
        store = PlanSnapshotStore(PlanResources(plan=_plan(4)))

        with self.assertRaises(ValueError):
            store.stage(PlanResources(plan=_plan(3)))
        with self.assertRaises(ValueError):
            store.stage(PlanResources(plan=_plan(4)))

        self.assertEqual(store.current.plan.plan_epoch, 4)

    def test_old_resources_wait_for_all_leases_before_collection(self) -> None:
        bucket = _bucket("old", slots=2)
        lease0, completion0, retire0 = _lease(1, bucket, 0)
        lease1, completion1, retire1 = _lease(1, bucket, 1)
        initial = PlanResources(plan=_plan(1, (bucket,)), leases=(lease0, lease1))
        replacement = PlanResources(plan=_plan(2))
        store = PlanSnapshotStore(initial)

        store.prepare_step(20)
        store.stage(replacement)
        self.assertIs(store.prepare_step(21), replacement)
        store.finish_step(20)

        self.assertEqual(store.retired_count, 1)
        self.assertFalse(lease0.reusable)
        self.assertFalse(lease1.reusable)
        self.assertEqual(store.collect_retired(), 0)
        self.assertEqual(store.retired_count, 1)

        lease0.publish_completion(None)
        lease0.publish_retire(None)
        completion0.done = True
        retire0.done = True
        self.assertTrue(lease0.reusable)
        self.assertFalse(lease1.reusable)
        self.assertEqual(store.collect_retired(), 0)

        lease1.publish_completion(None)
        lease1.publish_retire(None)
        completion1.done = True
        retire1.done = True
        self.assertTrue(lease1.reusable)
        self.assertEqual(store.collect_retired(), 1)
        self.assertEqual(store.retired_count, 0)

    def test_runtime_resources_stay_strongly_held_until_retired_collection(self) -> None:
        bucket = _bucket("old")
        lease, completion_event, retire_event = _lease(1, bucket, 0)
        runtime_plan, ready_key, capture_key = _runtime_plan(1, bucket)
        stream = _FakeStream()
        stream_retire_event = _FakeEvent()
        event = _FakeEvent()
        graph = _FakeGraph()
        stream_resource = RuntimeStream(
            key=StreamClass.DEFAULT,
            stream=stream,
            retire_fence=Fence(
                FenceKey("snapshot", "retire-strong"),
                stream_retire_event,
            ),
        )
        event_resource = RuntimeEvent(
            key=ready_key,
            event=event,
        )
        graph_resource = RuntimeGraph(
            key=capture_key,
            graph=graph,
        )
        initial = PlanResources(
            plan=runtime_plan,
            leases=(lease,),
            streams=(stream_resource,),
            events=(event_resource,),
            graphs=(graph_resource,),
        )
        replacement = PlanResources(plan=_plan(2))
        stream_ref = weakref.ref(stream)
        event_ref = weakref.ref(event)
        graph_ref = weakref.ref(graph)
        store = PlanSnapshotStore(initial)

        self.assertEqual(initial.streams[0].key, StreamClass.DEFAULT)
        self.assertIs(initial.streams[0].stream, stream)
        self.assertEqual(initial.events[0].key, ready_key)
        self.assertIs(initial.events[0].event, event)
        self.assertEqual(initial.graphs[0].key, capture_key)
        self.assertIs(initial.graphs[0].graph, graph)

        store.prepare_step(30)
        store.stage(replacement)
        store.prepare_step(31)
        store.finish_step(31)
        store.finish_step(30)
        stream_resource.retire_fence.record(stream)
        del initial, stream_resource, event_resource, graph_resource
        del stream, event, graph
        gc.collect()

        # The old epoch is still active, so its Python runtime handles remain owned.
        self.assertIsNotNone(stream_ref())
        self.assertIsNotNone(event_ref())
        self.assertIsNotNone(graph_ref())
        self.assertEqual(store.collect_retired(), 0)

        # Finishing the step is not enough while its lease still has in-flight work.
        self.assertEqual(store.collect_retired(), 0)
        gc.collect()
        self.assertIsNotNone(stream_ref())
        self.assertIsNotNone(event_ref())
        self.assertIsNotNone(graph_ref())

        lease.publish_completion(None)
        lease.publish_retire(None)
        completion_event.done = True
        retire_event.done = True
        stream_retire_event.done = True
        self.assertTrue(lease.reusable)
        self.assertEqual(store.collect_retired(), 1)
        gc.collect()

        self.assertIsNone(stream_ref())
        self.assertIsNone(event_ref())
        self.assertIsNone(graph_ref())

    def test_runtime_stream_resources_wait_for_all_retire_fences_without_leases(self) -> None:
        bucket = _bucket("stream-retire")
        runtime_plan, ready_key, capture_key = _runtime_plan(1, bucket)
        default_stream = _FakeStream()
        side_train_stream = _FakeStream()
        default_retire = _FakeEvent()
        side_train_retire = _FakeEvent()
        default_stream_resource = RuntimeStream(
            StreamClass.DEFAULT,
            default_stream,
            retire_fence=Fence(
                FenceKey("snapshot", "retire-default"),
                default_retire,
            ),
        )
        side_train_stream_resource = RuntimeStream(
            StreamClass.SIDE_TRAIN,
            side_train_stream,
            retire_fence=Fence(
                FenceKey("snapshot", "retire-side-train"),
                side_train_retire,
            ),
        )
        event = _FakeEvent()
        graph = _FakeGraph()
        initial = PlanResources(
            plan=runtime_plan,
            streams=(default_stream_resource, side_train_stream_resource),
            events=(RuntimeEvent(ready_key, event),),
            graphs=(RuntimeGraph(capture_key, graph),),
        )
        replacement = PlanResources(plan=_plan(2))
        stream_ref = weakref.ref(default_stream)
        side_train_stream_ref = weakref.ref(side_train_stream)
        graph_ref = weakref.ref(graph)
        store = PlanSnapshotStore(initial)

        store.prepare_step(30)
        store.stage(replacement)
        store.prepare_step(31)
        store.finish_step(31)
        store.finish_step(30)
        default_stream_resource.retire_fence.record(default_stream)
        side_train_stream_resource.retire_fence.record(side_train_stream)
        del initial, default_stream_resource, side_train_stream_resource
        del default_stream, side_train_stream, graph
        gc.collect()

        self.assertEqual(store.collect_retired(), 0)
        self.assertEqual(store.retired_count, 1)
        self.assertIsNotNone(stream_ref())
        self.assertIsNotNone(side_train_stream_ref())
        self.assertIsNotNone(graph_ref())

        default_retire.done = True
        side_train_retire.done = True
        self.assertEqual(store.collect_retired(), 1)
        gc.collect()
        self.assertIsNone(stream_ref())
        self.assertIsNone(side_train_stream_ref())
        self.assertIsNone(graph_ref())

    def test_plan_resources_reject_stream_key_not_used_by_the_plan(self) -> None:
        bucket = _bucket("owned")
        plan, _ready_key, _artifact_key = _runtime_plan(1, bucket)

        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                streams=(_runtime_stream(StreamClass.SIDE_FORWARD, "retire-invalid"),),
            )

    def test_plan_resources_reject_event_key_not_owned_by_plan_or_leases(self) -> None:
        bucket = _bucket("owned-event")
        plan, _ready_key, _artifact_key = _runtime_plan(1, bucket)
        unknown_event_key = FenceKey("snapshot", "not-a-plan-fence")

        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                events=(RuntimeEvent(unknown_event_key, _FakeEvent()),),
            )

    def test_plan_resources_reject_graph_key_without_capture_artifact(self) -> None:
        bucket = _bucket("owned-graph")
        plan, _ready_key, _artifact_key = _runtime_plan(1, bucket)

        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                graphs=(
                    RuntimeGraph(
                        ArtifactKey("snapshot", "not-captured", digest="b" * 64),
                        _FakeGraph(),
                    ),
                ),
            )

    def test_plan_resources_accept_lease_fence_runtime_event_keys(self) -> None:
        bucket = _bucket("lease-event")
        plan, _ready_key, _artifact_key = _runtime_plan(1, bucket)
        lease, _completion_event, _retire_event = _lease(1, bucket, 0)
        event = RuntimeEvent(
            lease.completion_fence.key,
            lease.completion_fence.event,
        )

        resources = PlanResources(plan=plan, leases=(lease,), events=(event,))

        self.assertIs(resources.events[0], event)

        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                leases=(lease,),
                events=(RuntimeEvent(lease.completion_fence.key, _FakeEvent()),),
            )

    def test_runtime_holders_reject_untyped_keys_and_primitive_handles(self) -> None:
        stream_key = StreamClass.DEFAULT
        event_key = FenceKey("snapshot", "ready")
        graph_key = ArtifactKey("snapshot", "capture", digest="a" * 64)

        with self.assertRaises(TypeError):
            RuntimeStream(
                key="default",
                stream=_FakeStream(),
                retire_fence=Fence(FenceKey("snapshot", "retire-invalid-key"), _FakeEvent()),  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            RuntimeStream(
                key=stream_key,
                stream=None,
                retire_fence=Fence(FenceKey("snapshot", "retire-invalid-stream"), _FakeEvent()),
            )
        with self.assertRaises(TypeError):
            RuntimeStream(
                key=stream_key,
                stream=0,
                retire_fence=Fence(FenceKey("snapshot", "retire-invalid-handle"), _FakeEvent()),
            )

        with self.assertRaises(TypeError):
            RuntimeEvent(key="ready", event=_FakeEvent())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RuntimeEvent(key=event_key, event=None)
        with self.assertRaises(TypeError):
            RuntimeEvent(key=event_key, event="event")

        with self.assertRaises(TypeError):
            RuntimeGraph(key="capture", graph=_FakeGraph())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RuntimeGraph(key=graph_key, graph=None)
        with self.assertRaises(TypeError):
            RuntimeGraph(key=graph_key, graph=False)

    def test_plan_resources_reject_duplicate_keys_within_each_typed_class(self) -> None:
        bucket = _bucket("duplicate-resources")
        plan, ready_key, artifact_key = _runtime_plan(1, bucket)

        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                streams=(
                    _runtime_stream(StreamClass.DEFAULT, "retire-duplicate-1"),
                    _runtime_stream(StreamClass.DEFAULT, "retire-duplicate-2"),
                ),
            )
        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                events=(
                    RuntimeEvent(ready_key, _FakeEvent()),
                    RuntimeEvent(ready_key, _FakeEvent()),
                ),
            )
        with self.assertRaises(ValueError):
            PlanResources(
                plan=plan,
                graphs=(
                    RuntimeGraph(
                        artifact_key,
                        _FakeGraph(),
                    ),
                    RuntimeGraph(
                        artifact_key,
                        _FakeGraph(),
                    ),
                ),
            )

    def test_runtime_holders_are_not_structural_workgraph_metadata(self) -> None:
        runtime_stream = _runtime_stream(StreamClass.DEFAULT, "retire-signature")

        with self.assertRaises(TypeError):
            structural_signature((runtime_stream,))

    def test_select_buffer_bucket_requires_exact_shape_dtype_and_device(self) -> None:
        float32 = _bucket("float32", dtype="float32")
        float16 = _bucket("float16", dtype="float16")
        other_shape = _bucket("other-shape", shape=(2, 4))
        cuda = _bucket("cuda", device=Device.CUDA, device_index=0)
        plan = _plan(1, (float32, float16, other_shape, cuda))

        with mock.patch(
            "torch.cuda.synchronize",
            side_effect=AssertionError("plan lookup must not synchronize CUDA"),
        ):
            self.assertIs(
                select_buffer_bucket(plan, (2, 3), torch.float32, torch.device("cpu")),
                float32,
            )
            self.assertIs(
                select_buffer_bucket(plan, (2, 3), torch.float16, torch.device("cpu")),
                float16,
            )
            self.assertIsNone(
                select_buffer_bucket(plan, (2, 5), torch.float32, torch.device("cpu"))
            )
            self.assertIsNone(
                select_buffer_bucket(plan, (2, 3), torch.float64, torch.device("cpu"))
            )
            self.assertIs(
                select_buffer_bucket(plan, (2, 3), torch.float32, torch.device("cuda", 0)),
                cuda,
            )
            self.assertIsNone(
                select_buffer_bucket(plan, (2, 3), torch.float32, torch.device("cuda", 1))
            )


if __name__ == "__main__":
    unittest.main()

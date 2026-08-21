from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
import unittest
from unittest import mock

import torch

from tllm.backends.protocol import GraphSafety
from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, PortKind, TimingContract
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.ports.residual_stream import ResidualStream
from tllm.producers.base import HookPoint
from tllm.producers.registry import ProducerRegistry, compile_execution_plan
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.workgraph.anchors import BackendSnapshot, shadow_backend_snapshot
from tllm.workgraph.shadow import compile_shadow_parity
from tllm.workgraph.frontend import (
    ArtifactKey,
    CaptureStatus,
    Device,
    PortBindingDesc,
    PortBindingDirection,
    TensorDTypeDesc,
    TensorDeviceDesc,
    flow,
)
from tllm.workgraph.hierarchy import CaptureAttachment
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.lowering import WorkGraphError
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
    compile_physical_flow_plan,
)


class _PhysicalConsumer(BaseConsumer[object]):
    config = object()

    def __init__(self, flows: tuple[ConsumerFlow, ...]) -> None:
        self._flows = flows

    @property
    def consumer_id(self) -> str:
        return "physical.consumer"

    def flows(self) -> tuple[ConsumerFlow, ...]:
        return self._flows


class _PhysicalProducer:
    producer_id = "physical.producer"
    hook_points = frozenset({HookPoint.EXECUTE_MODEL_POST})
    graph_safety = GraphSafety.CAPTURE_SAFE
    outputs = (
        PortSpec(
            id=PortId("physical", "residual.decode"),
            kind=PortKind.RESIDUAL_STREAM,
            writable=True,
            write_modes=("replace", "add"),
        ),
    )
    requires = ()


def _flow(*, movable: bool) -> ConsumerFlow:
    if movable:
        return ConsumerFlow(
            reads=(ResidualStream.read(layer=0, site="block_output", phase="decode"),),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
            timing=TimingContract(
                mobility="deferrable",
                deadline="next_prepare",
                max_defer_steps=1,
            ),
        )
    return ConsumerFlow(
        reads=(ResidualStream.read(layer=0, site="block_output", phase="decode"),),
        writes=(),
        window="same_step",
    )


def _phase1a_plan(*, movable: bool = False):
    return _phase1a_plan_with(
        movable=movable,
        backend=shadow_backend_snapshot(),
    )


def _phase1a_plan_with(
    *,
    movable: bool = False,
    backend=None,
    captures=(),
    attachments=(),
):
    consumer = _PhysicalConsumer((_flow(movable=movable),))
    dispatch_plan = DispatchPlan.build([consumer])
    registry = ProducerRegistry()
    registry.register(_PhysicalProducer())
    execution_plan = compile_execution_plan(registry=registry, consumers=[consumer])
    return compile_shadow_parity(
        consumers=[consumer],
        dispatch_plan=dispatch_plan,
        execution_plan=execution_plan,
        backend=shadow_backend_snapshot() if backend is None else backend,
        captures=captures,
        attachments=attachments,
    )


def _capture_attachment(*, opaque: bool = False):
    binding = PortBindingDesc(
        port=_PhysicalProducer.outputs[0].id,
        direction=PortBindingDirection.INPUT,
        value_path=(0,),
    )

    if opaque:

        @flow(bindings=(binding,))
        def captured_flow(value: torch.Tensor) -> torch.Tensor:
            print(value)
            return value + 1

    else:

        @flow(bindings=(binding,))
        def captured_flow(value: torch.Tensor) -> torch.Tensor:
            return value + 1

    captured = captured_flow.capture(torch.ones(2, 4))
    attachment = CaptureAttachment(
        node_key=NodeKey.consumer_flow("physical.consumer", 0),
        artifact_key=captured.artifact_key,
        bindings=(binding,),
    )
    return captured, attachment


def _metadata() -> tuple[TensorDTypeDesc, TensorDeviceDesc]:
    return TensorDTypeDesc(namespace="torch", name="float32"), TensorDeviceDesc(Device.CPU)


class PhysicalDescriptorTest(unittest.TestCase):
    def test_descriptors_are_frozen_slots_and_keys_are_stable(self) -> None:
        dtype, device = _metadata()
        key = BufferBucketKey("workgraph", "hidden", 2)
        bucket = BufferBucketDesc(
            key=key,
            shape=(2, 4),
            dtype=dtype,
            device=device,
            slots=2,
            ownership=BufferOwnershipKind.MANAGED,
        )
        producer = NodeKey.producer("physical.producer")
        consumer = NodeKey.consumer_flow("physical.consumer", 0)
        fence = FenceDependencyDesc(
            fence=FenceKey("workgraph", "ready", 0),
            producer=producer,
            consumer=consumer,
            kind=FenceKind.READY,
        )
        node = PhysicalNodePlan(
            node_key=consumer,
            executor=ExecutorKey("consumer", "physical.consumer"),
            artifact_key=ArtifactKey("capture", "physical.consumer", digest="a" * 64),
            stream=StreamClass.SIDE_TRAIN,
            fences=(fence,),
            buffer_bucket=key,
            capture_policy=CapturePolicy.PREFER,
            fallback_policy=FallbackPolicy.EAGER,
        )
        plan = PhysicalFlowPlan(
            plan_epoch=1,
            nodes=(
                PhysicalNodePlan(
                    node_key=producer,
                    executor=ExecutorKey("producer", "physical.producer"),
                    stream=StreamClass.DEFAULT,
                ),
                node,
            ),
            buffer_buckets=(bucket,),
        )
        for descriptor in (key, fence.fence, bucket, fence, node, plan):
            self.assertTrue(is_dataclass(descriptor))
            self.assertTrue(getattr(type(descriptor), "__dataclass_params__").frozen)
            self.assertTrue(hasattr(type(descriptor), "__slots__"))
        self.assertEqual(key.value, "workgraph:hidden/2")
        self.assertEqual(str(key), key.value)
        with self.assertRaises(FrozenInstanceError):
            bucket.slots = 3  # type: ignore[misc]

    def test_keys_reject_paths_addresses_and_implicit_stringification(self) -> None:
        with self.assertRaises(ValueError):
            BufferBucketKey("workgraph/runtime", "hidden")
        with self.assertRaises(ValueError):
            FenceKey("workgraph", "ready/event")
        with self.assertRaises(TypeError):
            BufferBucketKey(object(), "hidden")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            BufferBucketKey("workgraph", "<Tensor object at 0x7f1234>")
        with self.assertRaises(ValueError):
            FenceKey("workgraph", "callback@0x7f1234")
        self.assertNotIn("0x", BufferBucketKey("workgraph", "hidden").value)

    def test_descriptors_reject_runtime_objects_and_invalid_references(self) -> None:
        dtype, device = _metadata()
        bucket = BufferBucketDesc(
            key=BufferBucketKey("workgraph", "hidden"),
            shape=(2, 4),
            dtype=dtype,
            device=device,
        )
        producer = NodeKey.producer("physical.producer")
        consumer = NodeKey.consumer_flow("physical.consumer", 0)
        with self.assertRaises(TypeError):
            BufferBucketDesc(  # type: ignore[arg-type]
                key=bucket.key,
                shape=(2, 4),
                dtype=torch.empty((1,)),
                device=device,
            )
        runtime_objects = (
            torch.empty((1,)),
            torch.cuda.Event(),
            mock.Mock(spec=torch.cuda.Stream),
            mock.Mock(spec=torch.cuda.CUDAGraph),
            lambda: None,
        )
        for runtime_object in runtime_objects:
            with self.subTest(runtime_type=type(runtime_object).__name__):
                with self.assertRaises(TypeError):
                    PhysicalFlowPlan(plan_epoch=1, nodes=(runtime_object,))  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    PhysicalFlowPlan(  # type: ignore[arg-type]
                        plan_epoch=1,
                        nodes=(),
                        buffer_buckets=(runtime_object,),
                    )
        missing_producer_fence = FenceDependencyDesc(
            fence=FenceKey("workgraph", "ready"),
            producer=producer,
            consumer=consumer,
            kind=FenceKind.READY,
        )
        with self.assertRaises(ValueError):
            PhysicalFlowPlan(
                plan_epoch=1,
                nodes=(
                    PhysicalNodePlan(
                        node_key=consumer,
                        executor=ExecutorKey("consumer", "physical.consumer"),
                        stream=StreamClass.SIDE_TRAIN,
                        fences=(missing_producer_fence,),
                    ),
                ),
            )

    def test_plan_validates_epoch_schema_uniqueness_and_bucket_references(self) -> None:
        dtype, device = _metadata()
        producer = NodeKey.producer("physical.producer")
        executor = ExecutorKey("producer", "physical.producer")
        bucket = BufferBucketDesc(
            key=BufferBucketKey("workgraph", "hidden"),
            shape=(2, 4),
            dtype=dtype,
            device=device,
        )
        node = PhysicalNodePlan(
            node_key=producer,
            executor=executor,
            stream=StreamClass.DEFAULT,
            buffer_bucket=bucket.key,
        )
        with self.assertRaises(ValueError):
            PhysicalFlowPlan(plan_epoch=0, nodes=(node,), buffer_buckets=(bucket,))
        with self.assertRaises(ValueError):
            PhysicalFlowPlan(plan_epoch=1, nodes=(node,), buffer_buckets=(bucket,), schema_version=2)
        with self.assertRaises(ValueError):
            PhysicalFlowPlan(plan_epoch=1, nodes=(node, node), buffer_buckets=(bucket,))
        with self.assertRaises(ValueError):
            PhysicalFlowPlan(plan_epoch=1, nodes=(node,), buffer_buckets=(bucket, bucket))
        unknown_bucket_node = PhysicalNodePlan(
            node_key=producer,
            executor=executor,
            stream=StreamClass.DEFAULT,
            buffer_bucket=BufferBucketKey("workgraph", "unknown"),
        )
        with self.assertRaises(ValueError):
            PhysicalFlowPlan(plan_epoch=1, nodes=(unknown_bucket_node,))

    def test_compile_inherits_epoch_and_preserves_phase1_binding(self) -> None:
        phase1 = _phase1a_plan()
        binding = phase1.binding
        physical = compile_physical_flow_plan(phase1)
        self.assertEqual(physical.plan_epoch, phase1.epoch)
        self.assertEqual(phase1.binding, binding)
        self.assertEqual(physical.schema_version, 1)
        consumer_node = next(
            node for node in physical.nodes if node.node_key.namespace == "consumer_flow"
        )
        self.assertEqual(consumer_node.stream, StreamClass.SIDE_TRAIN)
        self.assertEqual(phase1.binding, binding)

    def test_legacy_eager_consumer_disables_capture_and_uses_eager_fallback(self) -> None:
        phase1 = _phase1a_plan(movable=True)
        physical = compile_physical_flow_plan(phase1)
        consumer_node = next(
            node for node in physical.nodes if node.node_key.namespace == "consumer_flow"
        )
        self.assertEqual(consumer_node.capture_policy, CapturePolicy.DISABLED)
        self.assertEqual(consumer_node.fallback_policy, FallbackPolicy.EAGER)
        self.assertEqual(consumer_node.stream, StreamClass.SIDE_TRAIN)

    def test_legacy_capture_attachment_does_not_allocate_a_physical_bucket(self) -> None:
        captured, attachment = _capture_attachment()
        phase1 = _phase1a_plan_with(
            movable=True,
            captures=(captured,),
            attachments=(attachment,),
        )

        physical = compile_physical_flow_plan(phase1)
        consumer_node = next(
            node for node in physical.nodes if node.node_key == attachment.node_key
        )
        self.assertEqual(consumer_node.artifact_key, captured.artifact_key)
        self.assertEqual(consumer_node.capture_policy, CapturePolicy.DISABLED)
        self.assertEqual(consumer_node.fallback_policy, FallbackPolicy.EAGER)
        self.assertIsNone(consumer_node.buffer_bucket)
        self.assertEqual(physical.buffer_buckets, ())

    def test_capture_attachment_lowers_artifact_and_static_output_bucket(self) -> None:
        captured, attachment = _capture_attachment()
        self.assertEqual(captured.status, CaptureStatus.CAPTURED)
        self.assertIsNotNone(captured.leaf_artifact)
        phase1 = _phase1a_plan_with(
            captures=(captured,),
            attachments=(attachment,),
        )

        physical = compile_physical_flow_plan(phase1)
        consumer_node = next(
            node for node in physical.nodes if node.node_key == attachment.node_key
        )
        self.assertEqual(consumer_node.artifact_key, captured.artifact_key)
        self.assertEqual(consumer_node.capture_policy, CapturePolicy.PREFER)
        self.assertIsNotNone(consumer_node.buffer_bucket)
        bucket = next(
            bucket
            for bucket in physical.buffer_buckets
            if bucket.key == consumer_node.buffer_bucket
        )
        self.assertEqual(bucket.shape, (2, 4))
        output_value = captured.values[-1]
        self.assertEqual(bucket.dtype, output_value.dtype)
        self.assertEqual(bucket.device, output_value.device)

    def test_full_graph_phase1_plan_is_rejected_by_physical_lowering(self) -> None:
        phase1 = _phase1a_plan_with(
            backend=BackendSnapshot(
                backend_id="vllm-physical-test",
                runner_family="mrv2",
                graph_mode="full",
                backend_version="0.27.1",
            )
        )

        with self.assertRaisesRegex(WorkGraphError, "WG120_"):
            compile_physical_flow_plan(phase1)

    def test_opaque_attachment_does_not_enable_capture_or_allocate_output_bucket(self) -> None:
        captured, attachment = _capture_attachment(opaque=True)
        self.assertEqual(captured.status, CaptureStatus.OPAQUE_BARRIER)
        phase1 = _phase1a_plan_with(
            captures=(captured,),
            attachments=(attachment,),
        )

        physical = compile_physical_flow_plan(phase1)
        consumer_node = next(
            node for node in physical.nodes if node.node_key == attachment.node_key
        )
        self.assertEqual(consumer_node.capture_policy, CapturePolicy.DISABLED)
        self.assertIsNone(consumer_node.buffer_bucket)


if __name__ == "__main__":
    unittest.main()

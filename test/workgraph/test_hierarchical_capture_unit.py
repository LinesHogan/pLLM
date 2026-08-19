from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import torch

from tllm.backends.protocol import GraphSafety
from tllm.consumers.base import BaseConsumer
from tllm.ports.address import port_address_desc
from tllm.ports.base import ConsumerFlow, PortKind, PortRead
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.producers.base import HookPoint
from tllm.producers.registry import ExecutionPlan, PlannedProducer, ProducerEntry
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.workgraph import (
    ArtifactKey,
    BackendSnapshot,
    CaptureAttachment,
    CapturedFlow,
    ExecutorKey,
    Effect,
    NodeKey,
    PartitionBoundary,
    PortBindingDesc,
    PortBindingDirection,
    RegionKind,
    build_hierarchy,
    compile_phase1a,
    flow,
)


class _Consumer(BaseConsumer[object]):
    def __init__(self, consumer_id: str) -> None:
        self._consumer_id = consumer_id
        self.config = object()

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def flows(self) -> tuple[ConsumerFlow, ...]:
        # The role intentionally does not name the producer port. The
        # hierarchy must use the explicit CaptureAttachment binding.
        return (
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.HIDDEN_TARGET, role="unrelated-role"),),
                writes=(),
                window="same_step",
            ),
        )


class _Producer:
    producer_id = "hidden-producer"
    output_port = PortId("demo", "hidden")
    outputs = (
        PortSpec(
            id=output_port,
            kind=PortKind.HIDDEN_TARGET,
            payload_type="torch.Tensor",
        ),
    )
    requires: tuple[PortId, ...] = ()
    hook_points = frozenset({HookPoint.EXECUTE_MODEL_POST})
    graph_safety = GraphSafety.CAPTURE_SAFE


def _plans() -> tuple[DispatchPlan, ExecutionPlan, PortId]:
    consumer = _Consumer("consumer")
    dispatch = DispatchPlan.build([consumer])
    producer = _Producer()
    entry = ProducerEntry(producer=producer)
    planned = PlannedProducer(
        entry=entry,
        emits_at=producer.hook_points,
        output_ports=(producer.output_port,),
        depends_on=(),
    )
    execution = ExecutionPlan(
        producers=(planned,),
        port_sources={producer.output_port: producer.producer_id},
    )
    return dispatch, execution, producer.output_port


class WorkGraphHierarchicalCaptureTest(unittest.TestCase):
    def test_successful_flow_forms_root_task_and_leaf_without_aten_task_nodes(self) -> None:
        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def add_one(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        captured = add_one.capture(torch.ones(2))
        self.assertIsInstance(captured, CapturedFlow)
        dispatch_snapshot = tuple(dispatch.flow_targets())
        execution_snapshot = (execution.producers, dict(execution.port_sources))
        attachment = CaptureAttachment(
            node_key=NodeKey.consumer_flow("consumer", 0),
            artifact_key=captured.artifact_key,
            bindings=(binding,),
        )
        hierarchy = build_hierarchy(
            captures=(captured,),
            attachments=(attachment,),
            dispatch_plan=dispatch,
            execution_plan=execution,
        )

        root = hierarchy.root
        self.assertEqual(root.kind, RegionKind.ROOT)
        self.assertEqual(len(root.children), 1)
        task = root.children[0]
        self.assertEqual(task.kind, RegionKind.TASK)
        self.assertEqual(len(task.children), 1)
        leaf = task.children[0]
        self.assertEqual(leaf.kind, RegionKind.LEAF)
        self.assertEqual(leaf.artifact_key, captured.artifact_key)
        self.assertIsNotNone(leaf.leaf_artifact)
        self.assertTrue(leaf.leaf_artifact.operators)
        self.assertEqual(getattr(root, "operators", ()), ())
        self.assertEqual(getattr(task, "operators", ()), ())

        # The three identity domains remain independent and stable.
        self.assertIsInstance(task.node_key, NodeKey)
        self.assertIsInstance(task.executor, ExecutorKey)
        self.assertIsInstance(leaf.artifact_key, ArtifactKey)
        self.assertNotEqual(task.node_key, leaf.artifact_key)
        self.assertNotEqual(task.executor, leaf.artifact_key)
        for key in (task.node_key, task.executor, leaf.artifact_key):
            self.assertNotIn("0x", str(key))

        self.assertEqual(hierarchy.attachments, (attachment,))
        serialized_port = hierarchy.to_dict()["attachments"][0]["bindings"][0]["port"]
        self.assertEqual(
            serialized_port,
            {
                "type": "port_id",
                "namespace": port.namespace,
                "name": port.name,
                "version": port.version,
            },
        )
        self.assertEqual(tuple(dispatch.flow_targets()), dispatch_snapshot)
        self.assertEqual(
            (execution.producers, dict(execution.port_sources)),
            execution_snapshot,
        )

    def test_hierarchy_explain_keeps_port_addresses_structured(self) -> None:
        dispatch, execution, _ = _plans()
        address = port_address_desc(dispatch.flow_targets()[0].flow.reads[0])
        binding = PortBindingDesc(
            port=address,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def identity(value: torch.Tensor) -> torch.Tensor:
            return value

        captured = identity.capture(torch.ones(2))
        hierarchy = build_hierarchy(
            captures=(captured,),
            attachments=(
                CaptureAttachment(
                    node_key=NodeKey.consumer_flow("consumer", 0),
                    artifact_key=captured.artifact_key,
                    bindings=(binding,),
                ),
            ),
            dispatch_plan=dispatch,
            execution_plan=execution,
        )

        serialized_port = hierarchy.to_dict()["attachments"][0]["bindings"][0]["port"]
        self.assertEqual(
            serialized_port,
            {
                "type": "port_address",
                "kind": address.kind,
                "locator_type": address.locator_type,
                "locator_fields": [],
            },
        )

    def test_opaque_capture_forms_opaque_leaf_and_discards_partial_export_graph(self) -> None:
        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def host_io(value: torch.Tensor) -> torch.Tensor:
            print(value)
            return value + 1

        captured = host_io.capture(torch.ones(2))
        attachment = CaptureAttachment(
            node_key=NodeKey.consumer_flow("consumer", 0),
            artifact_key=captured.artifact_key,
            bindings=(binding,),
        )
        hierarchy = build_hierarchy(
            captures=(captured,),
            attachments=(attachment,),
            dispatch_plan=dispatch,
            execution_plan=execution,
        )

        opaque_leaf = hierarchy.root.children[0].children[0]
        self.assertEqual(opaque_leaf.kind, RegionKind.OPAQUE)
        self.assertIsNone(opaque_leaf.leaf_artifact)
        self.assertEqual(getattr(opaque_leaf, "operators", ()), ())

    def test_attachment_is_explicit_and_does_not_guess_by_role_name_or_kind(self) -> None:
        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )
        wrong_port = PortId("demo", "different")
        wrong_binding = dataclasses.replace(binding, port=wrong_port)

        @flow(bindings=(binding,))
        def identity(value: torch.Tensor) -> torch.Tensor:
            return value

        captured = identity.capture(torch.ones(2))
        attachment = CaptureAttachment(
            node_key=NodeKey.consumer_flow("consumer", 0),
            artifact_key=captured.artifact_key,
            bindings=(binding,),
        )
        valid = build_hierarchy(
            captures=(captured,),
            attachments=(attachment,),
            dispatch_plan=dispatch,
            execution_plan=execution,
        )
        self.assertEqual(valid.attachments[0].bindings[0].port, port)

        invalid = dataclasses.replace(attachment, bindings=(wrong_binding,))
        with self.assertRaises((TypeError, ValueError, KeyError)):
            build_hierarchy(
                captures=(captured,),
                attachments=(invalid,),
                dispatch_plan=dispatch,
                execution_plan=execution,
            )

        producer_target = dataclasses.replace(
            attachment,
            node_key=NodeKey.producer("hidden-producer"),
        )
        with self.assertRaises(ValueError):
            build_hierarchy(
                captures=(captured,),
                attachments=(producer_target,),
                dispatch_plan=dispatch,
                execution_plan=execution,
            )

    def test_live_ranges_come_from_use_def_and_phase1_has_no_moves(self) -> None:
        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def add_one(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        captured = add_one.capture(torch.ones(2))
        attachment = CaptureAttachment(
            node_key=NodeKey.consumer_flow("consumer", 0),
            artifact_key=captured.artifact_key,
            bindings=(binding,),
        )
        hierarchy = build_hierarchy(
            captures=(captured,),
            attachments=(attachment,),
            dispatch_plan=dispatch,
            execution_plan=execution,
        )
        data = hierarchy.data_centric
        self.assertEqual(data.moves, ())
        self.assertTrue(data.leases)
        for lease in data.leases:
            self.assertTrue(lease.live_start)
            self.assertTrue(lease.live_end)
            self.assertFalse(lease.materialized)

        broken_value = dataclasses.replace(
            captured.values[0],
            use_def=dataclasses.replace(captured.values[0].use_def, users=(999,)),
        )
        broken_capture = dataclasses.replace(
            captured,
            values=(broken_value, *captured.values[1:]),
        )
        with self.assertRaises(ValueError):
            build_hierarchy(
                captures=(broken_capture,),
                attachments=(attachment,),
                dispatch_plan=dispatch,
                execution_plan=execution,
            )

    def test_module_effect_and_capability_facts_form_initial_task_boundaries(self) -> None:
        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def two_stage(value: torch.Tensor) -> torch.Tensor:
            return (value + 1) * 2

        captured = two_stage.capture(torch.ones(2))
        self.assertGreaterEqual(len(captured.leaf_artifact.operators), 2)
        operators = list(captured.leaf_artifact.operators)
        operators[1] = dataclasses.replace(
            operators[1],
            module_path=("block",),
            effects=(Effect.READ_STATE,),
        )
        leaf = dataclasses.replace(captured.leaf_artifact, operators=tuple(operators))
        captured = dataclasses.replace(
            captured,
            leaf_artifact=leaf,
            effects=(Effect.PURE, Effect.READ_STATE),
        )
        attachment = CaptureAttachment(
            node_key=NodeKey.consumer_flow("consumer", 0),
            artifact_key=captured.artifact_key,
            bindings=(binding,),
        )
        hierarchy = build_hierarchy(
            captures=(captured,),
            attachments=(attachment,),
            dispatch_plan=dispatch,
            execution_plan=execution,
        )

        tasks = hierarchy.root.children
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].boundary, PartitionBoundary.ROOT)
        self.assertEqual(tasks[1].boundary, PartitionBoundary.MODULE)
        self.assertEqual(tasks[1].module_path, ("block",))
        self.assertEqual(tasks[1].effects, (Effect.READ_STATE,))
        self.assertEqual(tasks[1].dependencies, (tasks[0].key,))

        with self.assertRaises(ValueError):
            dataclasses.replace(tasks[1], effects=(Effect.PURE,))

    def test_real_rng_operator_forms_effect_and_capability_partition(self) -> None:
        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def sample(value: torch.Tensor) -> torch.Tensor:
            return value + torch.rand_like(value)

        captured = sample.capture(torch.ones(2))
        hierarchy = build_hierarchy(
            captures=(captured,),
            attachments=(
                CaptureAttachment(
                    node_key=NodeKey.consumer_flow("consumer", 0),
                    artifact_key=captured.artifact_key,
                    bindings=(binding,),
                ),
            ),
            dispatch_plan=dispatch,
            execution_plan=execution,
        )

        tasks = hierarchy.root.children
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[1].boundary, PartitionBoundary.EFFECT)
        self.assertEqual(tasks[1].dependencies, (tasks[0].key,))
        self.assertIn(Effect.RNG, hierarchy.root.effects)

    def test_python_and_rust_reject_malformed_hierarchy_descriptors(self) -> None:
        from tllm import core as _core

        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def identity(value: torch.Tensor) -> torch.Tensor:
            return value

        captured = identity.capture(torch.ones(2))
        attachment = CaptureAttachment(
            node_key=NodeKey.consumer_flow("consumer", 0),
            artifact_key=captured.artifact_key,
            bindings=(binding,),
        )
        hierarchy = build_hierarchy(
            captures=(captured,),
            attachments=(attachment,),
            dispatch_plan=dispatch,
            execution_plan=execution,
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(hierarchy, schema_version=1)
        malformed_schema = SimpleNamespace(
            roots=hierarchy.roots,
            attachments=hierarchy.attachments,
            data_centric=hierarchy.data_centric,
            schema_version=1,
        )
        semantic_plan = compile_phase1a(
            consumers=(_Consumer("consumer"),),
            dispatch_plan=dispatch,
            execution_plan=execution,
            backend=BackendSnapshot(
                backend_id="vllm-test",
                runner_family="mrv2",
                graph_mode="eager",
                backend_version="0.27.1",
            ),
        )
        with self.assertRaises(ValueError):
            _core.validate_capture_hierarchy(malformed_schema, semantic_plan.semantic)

        unknown_attachment = SimpleNamespace(
            node_key=NodeKey.consumer_flow("unknown", 0),
            artifact_key=attachment.artifact_key,
            bindings=attachment.bindings,
            anchor_node_key=None,
        )
        malformed_endpoint = SimpleNamespace(
            roots=hierarchy.roots,
            attachments=(unknown_attachment,),
            data_centric=hierarchy.data_centric,
            schema_version=2,
        )
        with self.assertRaises(ValueError):
            _core.validate_capture_hierarchy(malformed_endpoint, semantic_plan.semantic)

        missing_bindings = SimpleNamespace(
            node_key=attachment.node_key,
            artifact_key=attachment.artifact_key,
            bindings=(),
            anchor_node_key=None,
        )
        malformed_bindings = SimpleNamespace(
            roots=hierarchy.roots,
            attachments=(missing_bindings,),
            data_centric=hierarchy.data_centric,
            schema_version=2,
        )
        with self.assertRaises(ValueError):
            _core.validate_capture_hierarchy(malformed_bindings, semantic_plan.semantic)

    def test_compile_attaches_hierarchy_without_changing_flat_runtime_projection(self) -> None:
        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def add_one(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        captured = add_one.capture(torch.ones(2))
        attachment = CaptureAttachment(
            node_key=NodeKey.consumer_flow("consumer", 0),
            artifact_key=captured.artifact_key,
            bindings=(binding,),
        )
        consumer = _Consumer("consumer")
        dispatch = DispatchPlan.build([consumer])
        plan = compile_phase1a(
            consumers=(consumer,),
            dispatch_plan=dispatch,
            execution_plan=execution,
            backend=BackendSnapshot(
                backend_id="vllm-test",
                runner_family="mrv2",
                graph_mode="eager",
                backend_version="0.27.1",
            ),
            captures=(captured,),
            attachments=(attachment,),
        )

        self.assertIsNotNone(plan.hierarchy)
        self.assertEqual(plan.hierarchy.attachments, (attachment,))
        self.assertFalse(
            any(node.node_key.namespace == "capture_operator" for node in plan.semantic.nodes)
        )
        self.assertEqual(plan.data.moves, ())
        producer_lease = next(lease for lease in plan.data.leases if lease.value.port == port)
        self.assertEqual(producer_lease.live_start, str(NodeKey.producer("hidden-producer")))
        self.assertEqual(producer_lease.live_end, str(NodeKey.consumer_flow("consumer", 0)))
        self.assertFalse(producer_lease.materialized)
        self.assertIn("capture_hierarchy", plan.explain())

    def test_hierarchy_changes_participate_in_signature_and_epoch(self) -> None:
        dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def add_one(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        @flow(bindings=(binding,))
        def multiply_two(value: torch.Tensor) -> torch.Tensor:
            return value * 2

        consumer = _Consumer("consumer")
        dispatch = DispatchPlan.build([consumer])
        backend = BackendSnapshot(
            backend_id="vllm-test",
            runner_family="mrv2",
            graph_mode="eager",
            backend_version="0.27.1",
        )

        def compile_capture(captured: CapturedFlow, previous: object = None):
            attachment = CaptureAttachment(
                node_key=NodeKey.consumer_flow("consumer", 0),
                artifact_key=captured.artifact_key,
                bindings=(binding,),
            )
            return compile_phase1a(
                consumers=(consumer,),
                dispatch_plan=dispatch,
                execution_plan=execution,
                backend=backend,
                previous=previous,
                captures=(captured,),
                attachments=(attachment,),
            )

        first = compile_capture(add_one.capture(torch.ones(2)))
        second = compile_capture(multiply_two.capture(torch.ones(2)), first)
        self.assertNotEqual(first.signature, second.signature)
        self.assertEqual(second.epoch, first.epoch + 1)
        self.assertTrue(second.diff.changed)

    def test_shape_specialization_facts_participate_in_signature(self) -> None:
        _dispatch, execution, port = _plans()
        binding = PortBindingDesc(
            port=port,
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def identity(value: torch.Tensor) -> torch.Tensor:
            return value

        consumer = _Consumer("consumer")
        dispatch = DispatchPlan.build([consumer])
        backend = BackendSnapshot(
            backend_id="vllm-test",
            runner_family="mrv2",
            graph_mode="eager",
            backend_version="0.27.1",
        )

        def compile_capture(captured: CapturedFlow, previous: object = None):
            return compile_phase1a(
                consumers=(consumer,),
                dispatch_plan=dispatch,
                execution_plan=execution,
                backend=backend,
                previous=previous,
                captures=(captured,),
                attachments=(
                    CaptureAttachment(
                        node_key=NodeKey.consumer_flow("consumer", 0),
                        artifact_key=captured.artifact_key,
                        bindings=(binding,),
                    ),
                ),
            )

        first = compile_capture(identity.capture(torch.ones(2)))
        second = compile_capture(identity.capture(torch.ones(3)), first)
        self.assertNotEqual(first.signature, second.signature)
        self.assertEqual(second.epoch, first.epoch + 1)


if __name__ == "__main__":
    unittest.main()

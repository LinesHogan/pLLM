from __future__ import annotations

import inspect
import unittest

import torch

from tllm.backends.protocol import GraphSafety
from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, PortKind, PortRead
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.producers.base import HookPoint
from tllm.producers.registry import ExecutionPlan, PlannedProducer, ProducerEntry
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.workgraph.anchors import (
    BackendSnapshot,
)
from tllm.workgraph.frontend import (
    PortBindingDesc,
    PortBindingDirection,
    flow,
)
from tllm.workgraph.hierarchy import (
    CaptureAttachment,
)
from tllm.workgraph.ir import (
    BoundaryKind,
    NodeKey,
    WorkGraphDiagnosticCode,
)
from tllm.workgraph.shadow import (
    compile_shadow_parity,
)


class _MatrixConsumer(BaseConsumer[object]):
    def __init__(self) -> None:
        self.config = object()
        self.flow_calls = 0

    @property
    def consumer_id(self) -> str:
        return "matrix-consumer"

    def flows(self) -> tuple[ConsumerFlow, ...]:
        self.flow_calls += 1
        return (
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.HIDDEN_TARGET, role="matrix-input"),),
                writes=(),
                window="same_step",
            ),
        )


class _MatrixProducer:
    producer_id = "matrix-producer"
    output_port = PortId("matrix", "hidden")
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


def _fixture() -> tuple[_MatrixConsumer, DispatchPlan, ExecutionPlan, object, CaptureAttachment]:
    consumer = _MatrixConsumer()
    dispatch = DispatchPlan.build([consumer])
    producer = _MatrixProducer()
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
    binding = PortBindingDesc(
        port=producer.output_port,
        direction=PortBindingDirection.INPUT,
        value_path=(0,),
    )

    @flow(bindings=(binding,))
    def residual_side_work(value: torch.Tensor) -> torch.Tensor:
        return value + 1

    captured = residual_side_work.capture(torch.ones(2))
    attachment = CaptureAttachment(
        node_key=NodeKey.consumer_flow("matrix-consumer", 0),
        artifact_key=captured.artifact_key,
        bindings=(binding,),
    )
    return consumer, dispatch, execution, captured, attachment


def _backend(runner: str, mode: str) -> BackendSnapshot:
    return BackendSnapshot(
        backend_id=f"vllm-{runner}",
        runner_family=runner,
        graph_mode=mode,
        backend_version="0.27.1",
    )


class CompatCloseoutMatrixTest(unittest.TestCase):
    def test_mrv1_mrv2_eager_piecewise_keep_capture_as_shadow_only(self) -> None:
        consumer, dispatch, execution, captured, attachment = _fixture()
        before_flow_calls = consumer.flow_calls
        before_dispatch = tuple(
            (id(target), target.flow_id, target.placement_hook)
            for target in dispatch.flow_targets()
        )
        before_execution = tuple(
            (id(item), item.entry.producer_id, tuple(item.output_ports))
            for item in execution.producers
        )
        before_sources = tuple(
            sorted((str(port), str(source)) for port, source in execution.port_sources.items())
        )

        for runner in ("mrv1", "mrv2"):
            for mode in ("eager", "piecewise"):
                with self.subTest(runner=runner, mode=mode):
                    plan = compile_shadow_parity(
                        consumers=(consumer,),
                        dispatch_plan=dispatch,
                        execution_plan=execution,
                        backend=_backend(runner, mode),
                        captures=(captured,),
                        attachments=(attachment,),
                    )
                    self.assertIsNotNone(plan.hierarchy)
                    self.assertEqual(plan.semantic.graph_mode, mode)
                    self.assertEqual(plan.semantic.backend.runner_family, runner)
                    self.assertEqual(plan.semantic.backend.graph_mode, mode)
                    self.assertEqual(plan.diagnostics, ())
                    self.assertEqual(plan.legacy_boundaries, ())
                    self.assertEqual(plan.data.moves, ())
                    self.assertEqual(plan.hierarchy.attachments, (attachment,))

                    # Compilation observes the runtime plans and does not
                    # replace, execute, or re-order them.
                    self.assertEqual(consumer.flow_calls, before_flow_calls)
                    self.assertEqual(
                        tuple(
                            (id(target), target.flow_id, target.placement_hook)
                            for target in dispatch.flow_targets()
                        ),
                        before_dispatch,
                    )
                    self.assertEqual(
                        tuple(
                            (id(item), item.entry.producer_id, tuple(item.output_ports))
                            for item in execution.producers
                        ),
                        before_execution,
                    )
                    self.assertEqual(
                        tuple(
                            sorted(
                                (str(port), str(source))
                                for port, source in execution.port_sources.items()
                            )
                        ),
                        before_sources,
                    )

    def test_full_mrv1_and_mrv2_remain_typed_legacy_only_shadow(self) -> None:
        consumer, dispatch, execution, captured, attachment = _fixture()
        for runner in ("mrv1", "mrv2"):
            with self.subTest(runner=runner):
                plan = compile_shadow_parity(
                    consumers=(consumer,),
                    dispatch_plan=dispatch,
                    execution_plan=execution,
                    backend=_backend(runner, "full"),
                    captures=(captured,),
                    attachments=(attachment,),
                )
                self.assertIsNotNone(plan.hierarchy)
                self.assertEqual(plan.semantic.graph_mode, "full")
                self.assertEqual(plan.semantic.backend.graph_mode, "full")
                self.assertIn("matrix-consumer#0", plan.legacy_boundaries)
                consumer_node = next(
                    node for node in plan.semantic.nodes if node.source == "matrix-consumer#0"
                )
                self.assertEqual(consumer_node.legacy_boundary, BoundaryKind.LEGACY)
                self.assertEqual(len(plan.diagnostics), 1)
                self.assertEqual(
                    plan.diagnostics[0].code,
                    WorkGraphDiagnosticCode.UNSUPPORTED_GRAPH_MODE,
                )
                self.assertEqual(
                    plan.explain()["typed_diagnostics"][0]["code"],
                    "WG120_UNSUPPORTED_GRAPH_MODE",
                )
                self.assertIn("WG120_LEGACY_GRAPH_MODE:full", plan.lowering_notes)
                self.assertEqual(plan.data.moves, ())

    def test_dispatch_api_does_not_traverse_capture_artifacts_in_hot_path(self) -> None:
        # These APIs are the event/flow dispatch surface. Their implementation
        # must stay independent of capture descriptors and hierarchy traversal.
        for method_name in ("select", "flow_targets", "flow_targets_for_hook"):
            source = inspect.getsource(getattr(DispatchPlan, method_name))
            with self.subTest(method=method_name):
                self.assertNotIn("CapturedFlow", source)
                self.assertNotIn("CaptureAttachment", source)
                self.assertNotIn("build_hierarchy", source)


if __name__ == "__main__":
    unittest.main()

"""CPU-only Phase 1a compatibility coverage for the real ESamp consumer."""

from __future__ import annotations

import unittest

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.producers.registry import ExecutionPlan
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.execution_plan import compile_default_plan
from tllm.workgraph.anchors import shadow_backend_snapshot
from tllm.workgraph.compiler import compile_phase1a
from tllm.workgraph.ir import BoundaryKind, NodeKind, TimingDeadlineKind, TimingMobilityKind


def _build_real_esamp_plans() -> tuple[ESampConsumer, DispatchPlan, ExecutionPlan]:
    """Build the same typed plans used by runtime startup, without CUDA."""

    consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
    dispatch_plan = DispatchPlan.build([consumer])
    execution_plan = compile_default_plan(consumers=[consumer])
    return consumer, dispatch_plan, execution_plan


def _dispatch_facts(dispatch_plan: DispatchPlan) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            target.flow_id,
            target.placement_hook,
            str(target.flow.window),
            str(target.flow.timing.mobility),
            str(target.flow.timing.deadline),
            target.flow.timing.max_defer_steps,
        )
        for target in dispatch_plan.flow_targets()
    )


def _execution_facts(execution_plan: ExecutionPlan) -> tuple[object, ...]:
    # Keep this snapshot to authoritative plan data only; no runtime object is
    # copied into or compared through the WorkGraph descriptors.
    return (
        tuple(
            (
                planned.entry.producer_id,
                tuple(sorted(hook.value for hook in planned.emits_at)),
                tuple(str(port) for port in planned.output_ports),
                tuple(str(port) for port in planned.depends_on),
            )
            for planned in execution_plan.producers
        ),
        tuple(sorted((str(port), str(source)) for port, source in execution_plan.port_sources.items())),
        tuple(execution_plan.capture_resident_producer_ids),
    )


class Phase1aEsampLegacyUnitTest(unittest.TestCase):
    def test_real_esamp_timing_is_conservatively_lowered_and_plans_remain_authoritative(self) -> None:
        consumer, dispatch_plan, execution_plan = _build_real_esamp_plans()
        flow = consumer.flows()[0]

        self.assertEqual(str(flow.timing.mobility), TimingMobilityKind.DEFERRABLE)
        self.assertEqual(str(flow.timing.deadline), TimingDeadlineKind.NEXT_PREPARE)
        self.assertEqual(flow.timing.max_defer_steps, 1)

        dispatch_before = _dispatch_facts(dispatch_plan)
        execution_before = _execution_facts(execution_plan)

        phase1a = compile_phase1a(
            consumers=[consumer],
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            backend=shadow_backend_snapshot(),
        )

        node = next(node for node in phase1a.semantic.nodes if node.source == "esamp#0")
        self.assertEqual(node.kind, NodeKind.COMPUTE)
        self.assertEqual(node.legacy_boundary, BoundaryKind.LEGACY)
        self.assertEqual(node.effects, ("opaque_legacy_effect",))
        self.assertIsNotNone(node.executor)
        self.assertEqual(node.executor.namespace, "legacy")
        self.assertEqual(node.timing.mobility, TimingMobilityKind.DEFERRABLE)
        self.assertEqual(node.timing.deadline, TimingDeadlineKind.NEXT_PREPARE)
        self.assertEqual(node.timing.max_defer_steps, 1)
        self.assertIn("esamp#0", phase1a.legacy_boundaries)
        self.assertTrue(
            any(note.startswith("esamp#0:legacy_boundary:") for note in phase1a.lowering_notes)
        )
        self.assertEqual(phase1a.data.moves, ())

        self.assertEqual(dispatch_before, _dispatch_facts(dispatch_plan))
        self.assertEqual(execution_before, _execution_facts(execution_plan))


if __name__ == "__main__":
    unittest.main()

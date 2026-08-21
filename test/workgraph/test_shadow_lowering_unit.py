from __future__ import annotations

import json
import unittest

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, TimingContract
from tllm.ports.request_meta import RequestMeta
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.producers.registry import ExecutionPlan
from tllm.workgraph.shadow import compile_shadow_parity
from tllm.workgraph.ir import BackendSnapshot, EdgeKind, NodeKind
from tllm.workgraph.lowering import WorkGraphError


class _FlowConsumer(BaseConsumer[object]):
    def __init__(self, consumer_id: str, timing: object = "pinned") -> None:
        self._consumer_id = consumer_id
        self.config = object()
        self._timing = timing

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def flows(self) -> tuple[ConsumerFlow, ...]:
        return (
            ConsumerFlow(
                reads=(RequestMeta.read(),),
                writes=(),
                window="same_step",
                bundle_key=("engine_step_id", "phase")
                if getattr(self._timing, "mobility", "pinned") != "pinned"
                else (),
                timing=self._timing,
            ),
        )


def _compile(consumer: _FlowConsumer, *, strict_native: bool = False, graph_mode: str = "eager"):
    dispatch = DispatchPlan.build([consumer])
    return compile_shadow_parity(
        consumers=[consumer],
        dispatch_plan=dispatch,
        execution_plan=ExecutionPlan(producers=(), port_sources={}),
        backend=BackendSnapshot(
            backend_id="vllm-test",
            runner_family="mrv2",
            graph_mode=graph_mode,
            backend_version="0.27.1",
        ),
        strict_native=strict_native,
    )


class ShadowLoweringTest(unittest.TestCase):
    def test_pinned_flow_is_native_but_effect_is_conservative(self) -> None:
        plan = _compile(_FlowConsumer("native"), strict_native=True)
        node = plan.semantic.nodes[0]
        self.assertEqual(node.kind, NodeKind.COMPUTE)
        self.assertIsNone(node.legacy_boundary)
        self.assertEqual(node.effects, ("opaque_legacy_effect",))
        self.assertEqual(plan.legacy_boundaries, ())
        self.assertEqual(plan.data.moves, ())

    def test_backend_anchor_is_an_explicit_semantic_and_execution_node(self) -> None:
        plan = _compile(_FlowConsumer("anchored"), strict_native=True)
        compute = next(node for node in plan.semantic.nodes if node.kind == NodeKind.COMPUTE)
        anchors = tuple(node for node in plan.semantic.nodes if node.kind == NodeKind.ANCHOR)
        self.assertEqual(len(anchors), 1)
        self.assertIsNone(compute.anchor)
        anchor = anchors[0]
        self.assertIsNotNone(anchor.anchor)
        self.assertEqual(anchor.node_key, anchor.anchor.node_key)
        self.assertIn((anchor.node_key, compute.node_key, EdgeKind.ANCHOR), plan.semantic.edges)
        self.assertEqual(
            {node.node_key for node in plan.semantic.nodes},
            {node.node_key for node in plan.data.nodes},
        )
        data_anchor = next(node for node in plan.data.nodes if node.node_key == anchor.node_key)
        self.assertIsNone(data_anchor.executor)
        self.assertEqual(data_anchor.anchor, anchor.anchor)
        self.assertIn(
            {
                "source": str(anchor.node_key),
                "target": str(compute.node_key),
                "kind": "anchor",
            },
            plan.explain()["semantic_edges"],
        )

    def test_non_pinned_flow_is_a_legacy_boundary(self) -> None:
        timing = TimingContract(mobility="deferrable", deadline="next_prepare", max_defer_steps=1)
        plan = _compile(_FlowConsumer("legacy", timing), strict_native=False)
        node = plan.semantic.nodes[0]
        self.assertIsNotNone(node.legacy_boundary)
        self.assertEqual(plan.legacy_boundaries, ("legacy#0",))
        self.assertIn("opaque_legacy_effect", node.effects)
        with self.assertRaisesRegex(WorkGraphError, "WG111_UNSUPPORTED_MOBILITY"):
            _compile(_FlowConsumer("legacy-strict", timing), strict_native=True)

    def test_latest_observation_is_legacy_unless_native_is_required(self) -> None:
        declared_timing = TimingContract(
            mobility="deferrable",
            deadline="next_prepare",
            max_defer_steps=1,
        )
        consumer = _FlowConsumer("versioned", declared_timing)
        dispatch = DispatchPlan.build([consumer])
        # DispatchPlan is the authoritative runtime contract and rejects a
        # version mode it cannot execute. Exercise lowering's conservative
        # compatibility boundary with a pre-established plan snapshot.
        latest_timing = TimingContract(
            mobility="deferrable",
            deadline="next_prepare",
            max_defer_steps=1,
            observes_version="latest_at_execution",
        )
        object.__setattr__(dispatch.flow_targets()[0].flow, "timing", latest_timing)
        kwargs = dict(
            consumers=[consumer],
            dispatch_plan=dispatch,
            execution_plan=ExecutionPlan(producers=(), port_sources={}),
            backend=BackendSnapshot(
                backend_id="vllm-test",
                runner_family="mrv2",
                graph_mode="eager",
                backend_version="0.27.1",
            ),
        )
        plan = compile_shadow_parity(**kwargs, strict_native=False)
        self.assertIn("non_declared_version_observation", plan.lowering_notes[0])
        with self.assertRaisesRegex(WorkGraphError, "WG110_UNSUPPORTED_VERSION"):
            compile_shadow_parity(**kwargs, strict_native=True)

    def test_full_mode_is_legacy_only(self) -> None:
        plan = _compile(_FlowConsumer("full"), graph_mode="full")
        self.assertEqual(plan.legacy_boundaries, ("full#0",))
        self.assertTrue(any("WG120_LEGACY_GRAPH_MODE" in note for note in plan.lowering_notes))
        with self.assertRaisesRegex(WorkGraphError, "WG120_UNSUPPORTED_GRAPH_MODE"):
            _compile(_FlowConsumer("full-strict"), strict_native=True, graph_mode="full")

    def test_unknown_selected_hook_fails_closed(self) -> None:
        consumer = _FlowConsumer("unknown-hook")
        dispatch = DispatchPlan.build([consumer])
        object.__setattr__(dispatch._flow_targets[0], "placement_hook", "future.hook")
        with self.assertRaisesRegex(WorkGraphError, "WG121_UNSUPPORTED_ANCHOR"):
            compile_shadow_parity(
                consumers=[consumer],
                dispatch_plan=dispatch,
                execution_plan=ExecutionPlan(producers=(), port_sources={}),
                backend=BackendSnapshot(
                    backend_id="vllm-test",
                    runner_family="mrv2",
                    graph_mode="eager",
                    backend_version="0.27.1",
                ),
            )

    def test_explain_includes_plan_binding_and_is_json_safe(self) -> None:
        plan = _compile(_FlowConsumer("explain"))
        report = plan.explain()
        self.assertEqual(report["binding"]["dispatch_flow_ids"], ["explain#0"])
        self.assertEqual(report["binding"]["execution_port_sources"], [])
        self.assertEqual(json.loads(json.dumps(report))["epoch"], plan.epoch)


if __name__ == "__main__":
    unittest.main()

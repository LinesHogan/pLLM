from __future__ import annotations

import dataclasses
import json
import unittest
from typing import cast

from tllm.backends.protocol import GraphSafety
from tllm.consumers.base import BaseConsumer
from tllm.ports.base import BufferOwnership, ConsumerFlow, FlowDelivery, FlowFusion, PortKind
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.ports.residual_stream import ResidualStream
from tllm.producers.base import HookPoint
from tllm.producers.registry import ExecutionPlan, PlannedProducer, ProducerEntry
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.graph_policy import GraphMode
from tllm.workgraph.compiler import compile_phase1a
from tllm.workgraph.ir import BackendSnapshot, RuntimePlanBinding
from tllm.workgraph.trace import (
    TraceEventKind,
    compare_event_traces,
    legacy_event_trace,
    shadow_event_trace,
)


class _Consumer(BaseConsumer[object]):
    config = object()

    def __init__(self, flows: tuple[ConsumerFlow, ...]) -> None:
        self._flows = flows

    @property
    def consumer_id(self) -> str:
        return "trace"

    def flows(self) -> tuple[ConsumerFlow, ...]:
        return self._flows


class _Producer:
    producer_id = "trace.producer"
    hook_points = frozenset({HookPoint.EXECUTE_MODEL_POST, HookPoint.PREPARE_INPUTS_PRE})
    graph_safety = GraphSafety.CAPTURE_SAFE

    def __init__(self) -> None:
        self.outputs = (
            PortSpec(
                id=PortId("trace", "residual"),
                kind=PortKind.RESIDUAL_STREAM,
                writable=True,
                write_modes=("replace",),
            ),
        )
        self.requires = ()


def _flow(*, fusion: bool = False, layer: int = 0) -> ConsumerFlow:
    write = ResidualStream.write(
        layer=layer,
        site="block_output",
        phase="decode",
        mode="replace",
        fusion_candidate=fusion,
    )
    return ConsumerFlow(
        reads=(),
        writes=(write,),
        window="same_step",
        bundle_key=("engine_step_id", "phase") if fusion else (),
        delivery=FlowDelivery.DEVICE_LEASE if fusion else FlowDelivery.BUNDLE,
        ownership=BufferOwnership.RUNTIME_LEASE if fusion else BufferOwnership.BORROWED,
        max_bundle_rows=8 if fusion else 0,
        fusion=FlowFusion.SIDE_STREAM_CUDAGRAPH if fusion else FlowFusion.NONE,
    )


def _plans() -> tuple[DispatchPlan, ExecutionPlan]:
    dispatch = DispatchPlan.build([_Consumer((_flow(fusion=True), _flow(layer=1)))])
    dispatch.ensure_graph_mode(GraphMode.PIECEWISE)
    producer = _Producer()
    entry = ProducerEntry(producer=producer)
    planned = PlannedProducer(
        entry=entry,
        emits_at=producer.hook_points,
        output_ports=(PortId("trace", "residual"),),
        depends_on=(),
    )
    execution = ExecutionPlan(
        producers=(planned,),
        port_sources={PortId("trace", "residual"): producer.producer_id},
        capture_resident_producer_ids=(producer.producer_id,),
    )
    return dispatch, execution


def _binding(dispatch: DispatchPlan, execution: ExecutionPlan) -> RuntimePlanBinding:
    return RuntimePlanBinding(
        dispatch_flow_ids=tuple(target.flow_id for target in dispatch.flow_targets()),
        dispatch_hooks=tuple(
            (target.flow_id, target.placement_hook) for target in dispatch.flow_targets()
        ),
        execution_producer_ids=tuple(
            planned.entry.producer_id for planned in execution.producers
        ),
        producer_hooks=tuple(
            (
                planned.entry.producer_id,
                tuple(sorted(hook.value for hook in planned.emits_at)),
            )
            for planned in execution.producers
        ),
        dispatch_fusion_groups=dispatch.fusion_groups,
        execution_port_sources=tuple(
            sorted(
                (str(port), str(producer_id))
                for port, producer_id in execution.port_sources.items()
            )
        ),
        capture_resident_producer_ids=execution.capture_resident_producer_ids,
    )


class TraceEquivalenceTest(unittest.TestCase):
    def test_trace_oracle_is_exported_from_workgraph_control_plane(self) -> None:
        import tllm.workgraph as workgraph

        self.assertIs(workgraph.compare_event_traces, compare_event_traces)

    def test_compiler_records_equivalent_event_trace_in_explain(self) -> None:
        dispatch, execution = _plans()
        consumer = cast(_Consumer, dispatch.flow_targets()[0].consumer)

        plan = compile_phase1a(
            consumers=(consumer,),
            dispatch_plan=dispatch,
            execution_plan=execution,
            backend=BackendSnapshot(
                backend_id="vllm-test",
                runner_family="mrv2",
                graph_mode="piecewise",
                backend_version="0.27.1",
            ),
        )

        self.assertIsNotNone(plan.trace)
        assert plan.trace is not None
        self.assertTrue(plan.trace.equivalent)
        trace_report = cast(dict[str, object], plan.explain()["event_trace"])
        self.assertEqual(trace_report["equivalent"], True)
        self.assertEqual(trace_report["mismatches"], [])
        self.assertEqual(
            cast(dict[str, object], trace_report["legacy"])["events"],
            cast(dict[str, object], trace_report["shadow"])["events"],
        )

    def test_authoritative_and_shadow_traces_are_event_by_event_equivalent(self) -> None:
        dispatch, execution = _plans()
        legacy = legacy_event_trace(dispatch, execution)
        shadow = shadow_event_trace(_binding(dispatch, execution))

        comparison = compare_event_traces(legacy, shadow)

        self.assertTrue(comparison.equivalent)
        self.assertEqual(comparison.mismatches, ())
        self.assertEqual(
            tuple(event.kind for event in legacy.events),
            (
                TraceEventKind.PRODUCER_HOOK,
                TraceEventKind.PRODUCER_HOOK,
                TraceEventKind.CONSUMER_FLOW_HOOK,
                TraceEventKind.CONSUMER_FLOW_HOOK,
                TraceEventKind.FUSION_MEMBERSHIP,
                TraceEventKind.PORT_SOURCE_BINDING,
                TraceEventKind.CAPTURE_RESIDENCY,
            ),
        )

    def test_legacy_trace_uses_producer_then_hook_and_flow_declaration_order(self) -> None:
        dispatch, execution = _plans()
        trace = legacy_event_trace(dispatch, execution)

        producer_events = [
            event for event in trace.events if event.kind == TraceEventKind.PRODUCER_HOOK
        ]
        flow_events = [
            event for event in trace.events if event.kind == TraceEventKind.CONSUMER_FLOW_HOOK
        ]
        self.assertEqual(
            tuple(event.details for event in producer_events),
            (("execute_model.post",), ("prepare_inputs.pre",)),
        )
        self.assertEqual(
            tuple(event.subject for event in flow_events),
            tuple(target.flow_id for target in dispatch.flow_targets()),
        )

    def test_mismatch_reports_ordinal_and_both_sides(self) -> None:
        dispatch, execution = _plans()
        legacy = legacy_event_trace(dispatch, execution)
        binding = _binding(dispatch, execution)
        shadow = shadow_event_trace(
            dataclasses.replace(
                binding,
                execution_port_sources=(("trace:residual@1", "other.producer"),),
            )
        )

        comparison = compare_event_traces(legacy, shadow)

        self.assertFalse(comparison.equivalent)
        self.assertEqual(len(comparison.mismatches), 1)
        mismatch = comparison.mismatches[0]
        legacy_event = mismatch.legacy
        shadow_event = mismatch.shadow
        self.assertIsNotNone(legacy_event)
        self.assertIsNotNone(shadow_event)
        assert legacy_event is not None
        assert shadow_event is not None
        self.assertEqual(mismatch.ordinal, 5)
        self.assertEqual(legacy_event.subject, "trace:residual@1")
        self.assertEqual(shadow_event.details, ("other.producer",))
        self.assertIn("ordinal=5", mismatch.message)
        self.assertIn("legacy=", mismatch.message)
        self.assertIn("shadow=", mismatch.message)

    def test_comparison_has_json_safe_explain_payload(self) -> None:
        dispatch, execution = _plans()
        legacy = legacy_event_trace(dispatch, execution)
        shadow = shadow_event_trace(_binding(dispatch, execution))

        payload = compare_event_traces(legacy, shadow).to_dict()
        legacy_payload = cast(dict[str, object], payload["legacy"])

        self.assertEqual(legacy_payload["origin"], "legacy")
        self.assertEqual(cast(dict[str, object], payload["shadow"])["origin"], "shadow")
        self.assertIn("events", legacy_payload)
        self.assertEqual(payload["equivalent"], True)
        self.assertEqual(payload["mismatches"], [])
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()

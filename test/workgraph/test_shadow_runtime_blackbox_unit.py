from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError
import unittest
from unittest import mock

from tllm.backends.protocol import GraphSafety
from tllm.consumers.base import BaseConsumer
from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.ports.base import ConsumerFlow, FlowFusion, PortKind, TimingContract
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.ports.residual_stream import ResidualStream
from tllm.producers.base import HookPoint
from tllm.producers.registry import ExecutionPlan, ProducerRegistry, compile_execution_plan
from tllm.producers.residual import default_residual_producers
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.graph_policy import GraphMode
from tllm.workgraph.anchors import shadow_backend_snapshot
from tllm.workgraph.shadow import ShadowParityPlan, compile_shadow_parity
from tllm.workgraph.ir import BackendSnapshot, BoundaryKind, EdgeKind, GraphSafetyKind, NodeKind
from tllm.workgraph.lowering import WorkGraphError
from tllm.workgraph.signature import structural_signature


class _BlackBoxConsumer(BaseConsumer[object]):
    config = object()

    def __init__(self, flows: tuple[ConsumerFlow, ...], consumer_id: str = "blackbox") -> None:
        self._flows = flows
        self._consumer_id = consumer_id

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def flows(self) -> tuple[ConsumerFlow, ...]:
        return self._flows


class _BlackBoxProducer:
    producer_id = "blackbox.producer"
    hook_points = frozenset({HookPoint.EXECUTE_MODEL_POST})
    graph_safety = GraphSafety.CAPTURE_SAFE

    def __init__(self) -> None:
        self.outputs = (
            PortSpec(
                id=PortId("blackbox", "residual.decode"),
                kind=PortKind.RESIDUAL_STREAM,
                writable=True,
                write_modes=("replace", "add"),
            ),
        )
        self.requires = ()


def _pinned_flow(*, window: str = "same_step") -> ConsumerFlow:
    return ConsumerFlow(
        reads=(ResidualStream.read(layer=0, site="block_output", phase="decode"),),
        writes=(),
        window=window,
        timing=TimingContract(
            mobility="pinned",
            observes_version="at_declared_window",
        ),
    )


def _deferrable_flow(*, observes_version: str = "at_declared_window") -> ConsumerFlow:
    return ConsumerFlow(
        reads=(ResidualStream.read(layer=0, site="block_output", phase="decode"),),
        writes=(),
        window="background",
        bundle_key=("engine_step_id", "phase"),
        timing=TimingContract(
            mobility="deferrable",
            deadline="next_prepare",
            max_defer_steps=1,
            observes_version=observes_version,
        ),
    )


def _prefill_flow() -> ConsumerFlow:
    return ConsumerFlow(
        reads=(ResidualStream.read(layer=0, site="block_output", phase="prefill"),),
        writes=(),
        window="same_step",
    )


def _fusion_flow() -> ConsumerFlow:
    return ConsumerFlow(
        reads=(),
        writes=(
            ResidualStream.write(
                layer=0,
                site="block_output",
                phase="decode",
                mode="add",
                fusion_candidate=True,
            ),
        ),
        window="out_of_band",
        bundle_key=("engine_step_id", "phase"),
        delivery="device_lease",
        ownership="runtime_lease",
        max_bundle_rows=8,
        fusion=FlowFusion.SIDE_STREAM_CUDAGRAPH,
    )


def _real_plans(
    flows: tuple[ConsumerFlow, ...],
    *,
    consumer_id: str = "blackbox",
) -> tuple[_BlackBoxConsumer, DispatchPlan, object]:
    consumer = _BlackBoxConsumer(flows, consumer_id=consumer_id)
    dispatch_plan = DispatchPlan.build([consumer])
    registry = ProducerRegistry()
    registry.register(_BlackBoxProducer())
    execution_plan = compile_execution_plan(registry=registry, consumers=[consumer])
    return consumer, dispatch_plan, execution_plan


def _compile(
    flows: tuple[ConsumerFlow, ...],
    *,
    backend: BackendSnapshot | None = None,
    previous: ShadowParityPlan | None = None,
    strict_native: bool = False,
    consumer_id: str = "blackbox",
) -> tuple[ShadowParityPlan, DispatchPlan, object]:
    consumer, dispatch_plan, execution_plan = _real_plans(flows, consumer_id=consumer_id)
    plan = compile_shadow_parity(
        consumers=[consumer],
        dispatch_plan=dispatch_plan,
        execution_plan=execution_plan,
        backend=backend or shadow_backend_snapshot(),
        previous=previous,
        strict_native=strict_native,
    )
    return plan, dispatch_plan, execution_plan


class ShadowRuntimeBlackBoxTest(unittest.TestCase):
    def test_real_runtime_plans_remain_authoritative_and_ir_is_observation_only(self) -> None:
        consumer, dispatch_plan, execution_plan = _real_plans((_pinned_flow(),))
        before_flows = tuple(
            (target.flow_id, target.placement_hook)
            for target in dispatch_plan.flow_targets()
        )
        before_producers = tuple(
            planned.entry.producer_id for planned in execution_plan.producers
        )
        before_port_sources = tuple(
            sorted((str(port), str(source)) for port, source in execution_plan.port_sources.items())
        )
        backend = shadow_backend_snapshot()

        plan = compile_shadow_parity(
            consumers=[consumer],
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            backend=backend,
        )

        self.assertEqual(
            before_flows,
            tuple((target.flow_id, target.placement_hook) for target in dispatch_plan.flow_targets()),
        )
        self.assertEqual(before_producers, tuple(item.entry.producer_id for item in execution_plan.producers))
        self.assertEqual(
            before_port_sources,
            tuple(sorted((str(port), str(source)) for port, source in execution_plan.port_sources.items())),
        )
        with self.assertRaises(FrozenInstanceError):
            backend.graph_mode = "piecewise"  # type: ignore[misc]
        self.assertEqual(plan.data.moves, ())

    def test_binding_parity_covers_flow_hooks_fusion_producers_and_port_sources(self) -> None:
        flows = (_fusion_flow(), _pinned_flow())
        consumer, dispatch_plan, execution_plan = _real_plans(flows)
        self.assertTrue(dispatch_plan.ensure_graph_mode(GraphMode.PIECEWISE))
        backend = BackendSnapshot(
            backend_id="vllm-blackbox",
            runner_family="mrv2",
            graph_mode="piecewise",
            backend_version="0.27.1",
            capabilities=("test",),
        )
        plan = compile_shadow_parity(
            consumers=[consumer],
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            backend=backend,
        )
        self.assertEqual(
            plan.binding.dispatch_flow_ids,
            tuple(target.flow_id for target in dispatch_plan.flow_targets()),
        )
        self.assertEqual(
            plan.binding.dispatch_hooks,
            tuple((target.flow_id, target.placement_hook) for target in dispatch_plan.flow_targets()),
        )
        self.assertEqual(plan.binding.dispatch_fusion_groups, dispatch_plan.fusion_groups)
        self.assertEqual(
            plan.binding.execution_producer_ids,
            tuple(item.entry.producer_id for item in execution_plan.producers),
        )
        self.assertEqual(
            plan.binding.producer_hooks,
            tuple(
                (item.entry.producer_id, tuple(sorted(hook.value for hook in item.emits_at)))
                for item in execution_plan.producers
            ),
        )
        self.assertEqual(
            plan.binding.execution_port_sources,
            tuple(sorted((str(port), str(source)) for port, source in execution_plan.port_sources.items())),
        )

    def test_pinned_declared_window_is_native_but_untyped_effect_is_conservative(self) -> None:
        plan, _dispatch, _execution = _compile((_pinned_flow(),))
        node = next(node for node in plan.semantic.nodes if node.source == "blackbox#0")
        self.assertEqual(node.kind, NodeKind.COMPUTE)
        self.assertIsNone(node.legacy_boundary)
        self.assertEqual(node.graph_safety, GraphSafetyKind.EAGER_ONLY)
        self.assertEqual(node.effects, ("opaque_legacy_effect",))

    def test_deferrable_flow_is_explicit_legacy_boundary(self) -> None:
        plan, _dispatch, _execution = _compile((_deferrable_flow(),))
        node = next(node for node in plan.semantic.nodes if node.source == "blackbox#0")
        self.assertEqual(node.legacy_boundary, BoundaryKind.LEGACY)
        self.assertEqual(node.effects, ("opaque_legacy_effect",))
        self.assertEqual(plan.legacy_boundaries, ("blackbox#0",))
        self.assertTrue(
            any(note.startswith("blackbox#0:legacy_boundary:") for note in plan.lowering_notes)
        )

    def test_actual_esamp_deferrable_declaration_stays_a_legacy_boundary(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=mock.Mock())
        dispatch_plan = DispatchPlan.build([consumer])
        plan = compile_shadow_parity(
            consumers=[consumer],
            dispatch_plan=dispatch_plan,
            execution_plan=ExecutionPlan(producers=(), port_sources={}),
            backend=shadow_backend_snapshot(),
        )
        node = next(node for node in plan.semantic.nodes if node.source == "esamp#0")
        self.assertEqual(node.legacy_boundary, BoundaryKind.LEGACY)
        self.assertEqual(node.timing.mobility, "deferrable")
        self.assertIn("opaque_legacy_effect", node.effects)
        self.assertIn("esamp#0", plan.legacy_boundaries)

    def test_coarse_kind_phase_binding_is_control_not_data_provenance(self) -> None:
        plan, _dispatch, _execution = _compile((_pinned_flow(),))
        producer = next(node for node in plan.semantic.nodes if node.source == "blackbox.producer")
        flow = next(node for node in plan.semantic.nodes if node.source == "blackbox#0")
        self.assertIn((producer.node_key, flow.node_key, EdgeKind.CONTROL), plan.semantic.edges)
        self.assertNotIn((producer.node_key, flow.node_key, EdgeKind.DATA), plan.semantic.edges)
        self.assertTrue(
            any(
                note == "blackbox#0:coarse_kind_phase_binding:blackbox.producer"
                for note in plan.lowering_notes
            )
        )

    def test_default_multi_hook_residual_producers_stay_legacy_without_false_anchor_ordering(self) -> None:
        registry = ProducerRegistry()
        for producer in default_residual_producers():
            registry.register(producer)
        consumer = _BlackBoxConsumer((_pinned_flow(), _prefill_flow()), consumer_id="residual-default")
        execution_plan = compile_execution_plan(registry=registry, consumers=[consumer])
        dispatch_plan = DispatchPlan.build([consumer])
        plan = compile_shadow_parity(
            consumers=[consumer],
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            backend=shadow_backend_snapshot(),
        )
        provider_nodes = tuple(
            node for node in plan.semantic.nodes if node.kind == NodeKind.PROVIDER
        )
        self.assertEqual(
            {node.source for node in provider_nodes},
            {"tllm.residual.decode", "tllm.residual.prefill"},
        )
        self.assertTrue(all(node.legacy_boundary == BoundaryKind.LEGACY for node in provider_nodes))
        self.assertTrue(all("opaque_legacy_effect" in node.effects for node in provider_nodes))
        provider_keys = {node.node_key for node in provider_nodes}
        self.assertFalse(
            any(
                kind == EdgeKind.ANCHOR and target in provider_keys
                for _source, target, kind in plan.semantic.edges
            )
        )
        self.assertEqual(
            dict(plan.binding.producer_hooks),
            {
                "tllm.residual.decode": (
                    "execute_model.post",
                    "layer.post",
                    "prepare_inputs.post",
                ),
                "tllm.residual.prefill": ("execute_model.post", "prepare_inputs.post"),
            },
        )
        self.assertTrue(
            any(
                note.startswith("tllm.residual.decode:legacy_multi_hook:")
                for note in plan.lowering_notes
            )
        )

    def test_latest_and_stale_observation_are_conservative_legacy_unless_strict(self) -> None:
        for observation in ("latest_at_execution", "stale_ok"):
            with self.subTest(observation=observation):
                flow = _deferrable_flow(observes_version=observation)
                with self.assertRaisesRegex(RuntimeError, "no execution-time port-version resolver"):
                    _real_plans((flow,))
                # Build the real runtime plans while the declaration is
                # executable, then mutate only the existing flow's version
                # observation to model a plan supplied by an older boundary.
                # Shadow must retain this flow conservatively instead of
                # requiring DispatchPlan to understand the new resolver.
                declared_flow = _deferrable_flow()
                consumer, dispatch_plan, execution_plan = _real_plans((declared_flow,))
                object.__setattr__(declared_flow, "timing", flow.timing)
                plan = compile_shadow_parity(
                    consumers=[consumer],
                    dispatch_plan=dispatch_plan,
                    execution_plan=execution_plan,
                    backend=shadow_backend_snapshot(),
                )
                node = next(item for item in plan.semantic.nodes if item.source == "blackbox#0")
                self.assertEqual(node.legacy_boundary, BoundaryKind.LEGACY)
                self.assertIn("non_declared_version_observation", plan.lowering_notes[0] if plan.lowering_notes else "")
                with self.assertRaisesRegex(WorkGraphError, "WG110_UNSUPPORTED_VERSION"):
                    compile_shadow_parity(
                        consumers=[_BlackBoxConsumer((flow,))],
                        dispatch_plan=dispatch_plan,
                        execution_plan=execution_plan,
                        backend=shadow_backend_snapshot(),
                        strict_native=True,
                    )

    def test_reconcile_keeps_epoch_for_same_plan_and_rebuilds_on_structure_change(self) -> None:
        first, _dispatch, _execution = _compile((_pinned_flow(),))
        second, _dispatch, _execution = _compile((_pinned_flow(),), previous=first)
        self.assertEqual(second.epoch, first.epoch)
        self.assertEqual(second.diff.rebuild_regions, ())
        self.assertTrue(second.diff.keep_regions)

        changed, _dispatch, _execution = _compile(
            (_pinned_flow(window="background"),),
            previous=first,
        )
        self.assertEqual(changed.epoch, first.epoch + 1)
        self.assertTrue(changed.diff.rebuild_regions)

    def test_binding_only_hook_change_rebuilds_the_shadow_epoch(self) -> None:
        consumer, dispatch_plan, execution_plan = _real_plans((_pinned_flow(),))
        target = dispatch_plan.flow_targets()[0]
        object.__setattr__(target, "placement_hook", "layer.pre")
        first = compile_shadow_parity(
            consumers=[consumer],
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            backend=shadow_backend_snapshot(),
        )
        object.__setattr__(target, "placement_hook", "layer.post")
        changed = compile_shadow_parity(
            consumers=[consumer],
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            backend=shadow_backend_snapshot(),
            previous=first,
        )
        self.assertEqual(
            structural_signature((first.semantic, first.data)),
            structural_signature((changed.semantic, changed.data)),
        )
        self.assertNotEqual(first.signature, changed.signature)
        self.assertEqual(changed.epoch, first.epoch + 1)

    def test_capture_resident_binding_change_rebuilds_the_shadow_epoch(self) -> None:
        consumer, dispatch_plan, execution_plan = _real_plans((_pinned_flow(),))
        first = compile_shadow_parity(
            consumers=(consumer,),
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            backend=shadow_backend_snapshot(),
        )
        changed_execution_plan = dataclasses.replace(
            execution_plan,
            capture_resident_producer_ids=("blackbox.producer",),
        )
        changed = compile_shadow_parity(
            consumers=(consumer,),
            dispatch_plan=dispatch_plan,
            execution_plan=changed_execution_plan,
            backend=shadow_backend_snapshot(),
            previous=first,
        )
        self.assertNotEqual(first.signature, changed.signature)
        self.assertEqual(changed.epoch, first.epoch + 1)
        self.assertEqual(changed.binding.capture_resident_producer_ids, ("blackbox.producer",))
        self.assertEqual(
            changed.explain()["binding"]["capture_resident_producer_ids"],
            ["blackbox.producer"],
        )
        self.assertTrue(changed.diff.rebuild_regions)


if __name__ == "__main__":
    unittest.main()

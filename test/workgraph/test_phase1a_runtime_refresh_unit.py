from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety, SamplerStage
from tllm.backends.vllm.workgraph_runtime import record_resolved_backend
from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, FlowFusion, TimingContract
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream
from tllm.producers.registry import ExecutionPlan
from tllm.runtime import residual_runtime
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.graph_policy import GraphMode
from tllm.runtime.ports.residual_bundle_dispatch import dispatch_decode_port_bundles
from tllm.runtime.state import ResidualRuntimeState
from tllm.workgraph.anchors import BackendResolution
from tllm.workgraph.compiler import compile_shadow_snapshot
from tllm.workgraph.ir import WorkGraphDiagnosticCode


class _FlowConsumer(BaseConsumer[object]):
    config = object()

    def __init__(self, consumer_id: str, *, fused: bool = False) -> None:
        self._consumer_id = consumer_id
        self._fused = fused

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def flows(self) -> tuple[ConsumerFlow, ...]:
        if self._fused:
            return (
                ConsumerFlow(
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
                    timing=TimingContract(
                        mobility="pinned",
                        observes_version="at_declared_window",
                    ),
                ),
            )
        return (
            ConsumerFlow(
                reads=(RequestMeta.read(),),
                writes=(),
                window="same_step",
                timing=TimingContract(
                    mobility="pinned",
                    observes_version="at_declared_window",
                ),
            ),
        )


def _capabilities(
    family: str,
    *,
    tuple_sampler_return: bool = True,
) -> BackendCapabilities:
    return BackendCapabilities(
        backend_id=f"vllm-0.27-{family}",
        runner_family=family,
        sampler_stages=frozenset({SamplerStage.BEFORE_DRAW}),
        native_min_p=True,
        tuple_sampler_return=tuple_sampler_return,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.CAPTURE_SAFE,
    )


def _runtime_state(*, fused: bool = False) -> ResidualRuntimeState:
    consumer = _FlowConsumer("workgraph.refresh", fused=fused)
    dispatch_plan = DispatchPlan.build([consumer])
    execution_plan = ExecutionPlan(producers=(), port_sources={})
    state = ResidualRuntimeState()
    state.dispatch_plan = dispatch_plan
    state.execution_plan = execution_plan
    state.workgraph_snapshot = compile_shadow_snapshot(
        consumers=(consumer,),
        dispatch_plan=dispatch_plan,
        execution_plan=execution_plan,
    )
    state.workgraph_refresh_pending = True
    return state


class Phase1aRuntimeRefreshTest(unittest.TestCase):
    def test_resolution_before_consumer_registration_refreshes_the_new_plan(self) -> None:
        state = ResidualRuntimeState()
        resolution = BackendResolution(
            capabilities=_capabilities("mrv2"),
            graph_mode=GraphMode.PIECEWISE,
        )
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.record_workgraph_backend_resolution(resolution=resolution)
            self.assertIs(state.workgraph_backend_resolution, resolution)
            self.assertIsNone(state.workgraph_snapshot)

            residual_runtime.register_dispatch_consumer(_FlowConsumer("late.registration"))

        snapshot = state.workgraph_snapshot
        self.assertIsNotNone(snapshot)
        self.assertFalse(state.workgraph_refresh_pending)
        self.assertEqual(snapshot.backend.runner_family, "mrv2")
        self.assertEqual(snapshot.backend.graph_mode, "piecewise")

    def test_mrv1_mrv2_eager_and_piecewise_refresh_once_without_replacing_runtime_plans(self) -> None:
        for family in ("mrv1", "mrv2"):
            for graph_mode in (GraphMode.EAGER, GraphMode.PIECEWISE):
                with self.subTest(family=family, graph_mode=graph_mode):
                    state = _runtime_state()
                    dispatch_plan = state.dispatch_plan
                    execution_plan = state.execution_plan
                    with mock.patch.object(residual_runtime, "RUNTIME", state):
                        residual_runtime.record_workgraph_backend_resolution(
                            resolution=BackendResolution(
                                capabilities=_capabilities(family),
                                graph_mode=graph_mode,
                            )
                        )
                        snapshot = state.workgraph_snapshot
                        self.assertIsNotNone(snapshot)
                        self.assertFalse(state.workgraph_refresh_pending)
                        self.assertIs(state.dispatch_plan, dispatch_plan)
                        self.assertIs(state.execution_plan, execution_plan)
                        self.assertEqual(snapshot.backend.runner_family, family)
                        self.assertEqual(snapshot.backend.graph_mode, graph_mode.name.lower())

                        residual_runtime.record_workgraph_backend_resolution(
                            resolution=BackendResolution(
                                capabilities=_capabilities(family),
                                graph_mode=graph_mode,
                            )
                        )
                        self.assertIs(state.workgraph_snapshot, snapshot)
                        self.assertEqual(state.workgraph_snapshot.epoch, snapshot.epoch)

    def test_changed_typed_resolution_rebuilds_the_concrete_snapshot_once(self) -> None:
        state = _runtime_state()
        first_resolution = BackendResolution(
            capabilities=_capabilities("mrv1"),
            graph_mode=GraphMode.EAGER,
        )
        changed_capabilities = BackendCapabilities(
            backend_id="vllm-0.27-mrv2",
            runner_family="mrv2",
            sampler_stages=frozenset({SamplerStage.BEFORE_DRAW, SamplerStage.AFTER_DRAW}),
            native_min_p=True,
            tuple_sampler_return=True,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.CAPTURE_SAFE,
        )
        changed_resolution = BackendResolution(
            capabilities=changed_capabilities,
            graph_mode=GraphMode.PIECEWISE,
        )
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.record_workgraph_backend_resolution(resolution=first_resolution)
            first = state.workgraph_snapshot
            self.assertIsNotNone(first)

            residual_runtime.record_workgraph_backend_resolution(resolution=changed_resolution)

        snapshot = state.workgraph_snapshot
        self.assertIsNotNone(snapshot)
        self.assertIsNot(snapshot, first)
        self.assertEqual(snapshot.backend.backend_id, "vllm-0.27-mrv2")
        self.assertEqual(snapshot.backend.runner_family, "mrv2")
        self.assertEqual(snapshot.backend.graph_mode, "piecewise")
        self.assertIn("sampler_stage:after_draw", snapshot.backend.capabilities)
        self.assertEqual(snapshot.epoch, first.epoch + 1)
        self.assertTrue(snapshot.diff.changed)
        self.assertTrue(snapshot.diff.rebuild_regions)
        self.assertFalse(state.workgraph_refresh_pending)

    def test_tuple_sampler_return_change_rebuilds_the_concrete_snapshot(self) -> None:
        state = _runtime_state()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.record_workgraph_backend_resolution(
                resolution=BackendResolution(
                    capabilities=_capabilities("mrv1", tuple_sampler_return=True),
                    graph_mode=GraphMode.EAGER,
                )
            )
            first = state.workgraph_snapshot
            self.assertIsNotNone(first)

            residual_runtime.record_workgraph_backend_resolution(
                resolution=BackendResolution(
                    capabilities=_capabilities("mrv1", tuple_sampler_return=False),
                    graph_mode=GraphMode.EAGER,
                )
            )

        snapshot = state.workgraph_snapshot
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.epoch, first.epoch + 1)
        self.assertIn("tuple_sampler_return:False", snapshot.backend.capabilities)

    def test_fusion_rebuild_waits_only_when_placement_needs_refresh(self) -> None:
        state = _runtime_state(fused=True)
        state.event_step_id = 1
        runner = type(
            "Runner",
            (),
            {
                "device": torch.device("cpu"),
                "model": None,
                "enforce_eager": False,
                "compilation_config": type("CompilationConfig", (), {"level": 1, "cudagraph_mode": None})(),
            },
        )()
        initial = BackendResolution(
            capabilities=_capabilities("mrv1"),
            graph_mode=GraphMode.PIECEWISE,
        )
        changed = BackendResolution(
            capabilities=BackendCapabilities(
                backend_id="vllm-0.27-mrv1-extended",
                runner_family="mrv1",
                sampler_stages=frozenset({SamplerStage.BEFORE_DRAW, SamplerStage.AFTER_DRAW}),
                native_min_p=True,
                tuple_sampler_return=True,
                split_execute_and_sample=True,
                graph_safety=GraphSafety.CAPTURE_SAFE,
            ),
            graph_mode=GraphMode.PIECEWISE,
        )
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.record_workgraph_backend_resolution(resolution=initial)
            self.assertTrue(state.workgraph_refresh_pending)
            with mock.patch(
                "tllm.runtime.ports.residual_bundle_dispatch._flow_due_for_step",
                return_value=False,
            ):
                dispatch_decode_port_bundles(core=residual_runtime, runner=runner)
            first = state.workgraph_snapshot
            self.assertFalse(state.workgraph_refresh_pending)

            residual_runtime.record_workgraph_backend_resolution(resolution=changed)

        snapshot = state.workgraph_snapshot
        self.assertIsNot(snapshot, first)
        self.assertEqual(snapshot.backend.backend_id, "vllm-0.27-mrv1-extended")
        self.assertEqual(snapshot.epoch, first.epoch + 1)
        self.assertFalse(state.workgraph_refresh_pending)

    def test_backend_handoff_keeps_fake_cores_outside_global_runtime_state(self) -> None:
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            record_resolved_backend(
                core=object(),
                capabilities=_capabilities("mrv2"),
                graph_mode=GraphMode.PIECEWISE,
            )
        self.assertIsNone(state.workgraph_backend_resolution)
        self.assertIsNone(state.workgraph_snapshot)

    def test_resolved_backend_plan_rebuilds_reconcile_from_the_concrete_snapshot(self) -> None:
        state = ResidualRuntimeState()
        resolution = BackendResolution(
            capabilities=_capabilities("mrv2"),
            graph_mode=GraphMode.PIECEWISE,
        )
        first_consumer = _FlowConsumer("workgraph.first")
        second_consumer = _FlowConsumer("workgraph.second")
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.record_workgraph_backend_resolution(resolution=resolution)
            residual_runtime.replace_dispatch_consumers((first_consumer,))
            first = state.workgraph_snapshot
            self.assertIsNotNone(first)

            residual_runtime.replace_dispatch_consumers((first_consumer,))
            unchanged = state.workgraph_snapshot
            self.assertEqual(unchanged.epoch, first.epoch)
            self.assertFalse(unchanged.diff.changed)

            residual_runtime.replace_dispatch_consumers((first_consumer, second_consumer))

        rebuilt = state.workgraph_snapshot
        self.assertEqual(rebuilt.epoch, first.epoch + 1)
        self.assertTrue(rebuilt.diff.changed)
        self.assertTrue(rebuilt.diff.rebuild_regions)

    def test_full_mode_records_typed_legacy_diagnostic_without_changing_dispatch(self) -> None:
        state = _runtime_state()
        dispatch_plan = state.dispatch_plan
        execution_plan = state.execution_plan
        expected_targets = tuple(
            (target.flow_id, target.placement_hook) for target in dispatch_plan.flow_targets()
        )
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.record_workgraph_backend_resolution(
                resolution=BackendResolution(
                    capabilities=_capabilities("mrv2"),
                    graph_mode=GraphMode.FULL,
                )
            )

        snapshot = state.workgraph_snapshot
        self.assertIsNotNone(snapshot)
        self.assertFalse(state.workgraph_refresh_pending)
        self.assertEqual(snapshot.backend.graph_mode, "full")
        self.assertEqual(
            snapshot.diagnostics[0].code,
            WorkGraphDiagnosticCode.UNSUPPORTED_GRAPH_MODE,
        )
        self.assertIn("legacy DispatchPlan", snapshot.diagnostics[0].detail)
        self.assertEqual(snapshot.explain()["typed_diagnostics"][0]["code"], "WG120_UNSUPPORTED_GRAPH_MODE")
        self.assertIs(state.dispatch_plan, dispatch_plan)
        self.assertIs(state.execution_plan, execution_plan)
        self.assertEqual(
            tuple((target.flow_id, target.placement_hook) for target in dispatch_plan.flow_targets()),
            expected_targets,
        )

    def test_fusion_snapshot_waits_for_existing_placement_refresh_and_only_compiles_once(self) -> None:
        state = _runtime_state(fused=True)
        self.assertTrue(state.dispatch_plan.requires_graph_mode_resolution())
        state.event_step_id = 1
        runner = type(
            "Runner",
            (),
            {
                "device": torch.device("cpu"),
                "model": None,
                "enforce_eager": False,
                "compilation_config": type("CompilationConfig", (), {"level": 1, "cudagraph_mode": None})(),
            },
        )()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.record_workgraph_backend_resolution(
                resolution=BackendResolution(
                    capabilities=_capabilities("mrv1"),
                    graph_mode=GraphMode.PIECEWISE,
                )
            )
            self.assertTrue(state.workgraph_refresh_pending)
            with mock.patch(
                "tllm.runtime.ports.residual_bundle_dispatch._flow_due_for_step",
                return_value=False,
            ):
                dispatch_decode_port_bundles(core=residual_runtime, runner=runner)
            snapshot = state.workgraph_snapshot
            self.assertFalse(state.workgraph_refresh_pending)
            self.assertEqual(snapshot.backend.graph_mode, "piecewise")
            self.assertGreaterEqual(state.placement_stats.fusion_eligible_count, 0)

            with mock.patch(
                "tllm.runtime.ports.residual_bundle_dispatch._flow_due_for_step",
                return_value=False,
            ):
                dispatch_decode_port_bundles(core=residual_runtime, runner=runner)
            self.assertIs(state.workgraph_snapshot, snapshot)


if __name__ == "__main__":
    unittest.main()

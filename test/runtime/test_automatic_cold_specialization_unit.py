"""Red contracts for automatic, load-time WorkGraph specialization."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety, SamplerStage
from tllm.backends.vllm import runner_adapter
from tllm.backends.vllm import v027_mrv2
from tllm.consumers.base import BaseConsumer
from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RuntimeContext
from tllm.runtime import residual_runtime
from tllm.runtime.graph_policy import GraphMode
from tllm.runtime.scheduling.plan_store import PlanSnapshotStore
from tllm.runtime.scheduling.workgraph_executor import PhysicalExecutor
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.workgraph.anchors import BackendResolution


def _resolution(mode: GraphMode = GraphMode.PIECEWISE) -> BackendResolution:
    return BackendResolution(
        capabilities=BackendCapabilities(
            backend_id="vllm-auto-specialization",
            runner_family="mrv2",
            sampler_stages=frozenset({SamplerStage.BEFORE_DRAW}),
            native_min_p=True,
            tuple_sampler_return=True,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.CAPTURE_SAFE,
        ),
        graph_mode=mode,
    )


def _spec() -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=4,
        hidden_size=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _bundle() -> PortBundle:
    return PortBundle(
        key=BundleKey(engine_step_id=1, phase="decode", request_id="req-a", sample_idx=0),
        entries={
            "device_lease": DeviceTensorLease(
                entries={
                    "source": torch.ones((4, 8)),
                    "target": torch.zeros((4, 8)),
                },
                active_rows=4,
            ),
            "request_meta": RowBatchMeta(
                request_ids=("req-a", "req-b", "req-c", "req-d"),
                prompt_idxs=(0, 1, 2, 3),
                sample_idxs=(0, 0, 0, 0),
                phase="decode",
                engine_step_id=1,
            ),
        },
    )


def _ctx() -> RuntimeContext:
    return RuntimeContext(
        runner=SimpleNamespace(max_num_reqs=4),
        model=None,
        device=torch.device("cpu"),
        main_stream=None,
        is_compiling=False,
        uses_cudagraph=False,
        event_name="flow:decode",
    )


class _HookConsumer(BaseConsumer[object]):
    config = object()

    def __init__(self) -> None:
        self.seen: object | None = None

    @property
    def consumer_id(self) -> str:
        return "cold-specialization-hook"

    def cold_specialize_workgraph(self, spec: RuntimeSpecializationSpec) -> None:
        self.seen = spec


class AutomaticColdSpecializationTest(unittest.TestCase):
    def tearDown(self) -> None:
        residual_runtime.clear_dispatch_consumers()

    def test_runtime_specialization_spec_is_typed(self) -> None:
        spec = _spec()

        self.assertIsInstance(spec, RuntimeSpecializationSpec)
        self.assertEqual((spec.rows, spec.hidden_size), (4, 8))
        self.assertIs(spec.dtype, torch.float32)
        self.assertEqual(spec.device, torch.device("cpu"))

    def test_base_consumer_exposes_polymorphic_cold_hook(self) -> None:
        consumer = _HookConsumer()
        spec = _spec()

        self.assertTrue(callable(getattr(BaseConsumer, "cold_specialize_workgraph", None)))
        consumer.cold_specialize_workgraph(spec)

        self.assertIs(consumer.seen, spec)

    def test_vllm_cold_specialization_rejects_missing_prefill_capacity(self) -> None:
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=torch.float16,
                get_hidden_size=lambda: 8,
            ),
            max_num_reqs=6,
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(RuntimeError, "max_num_tokens"):
            runner_adapter.runner_runtime_specialization(runner)

    def test_enabled_esamp_is_cold_specialized_and_physically_staged_before_first_token(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=mock.Mock())
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
            residual_runtime.cold_specialize_workgraph(_spec())

            self.assertEqual(len(consumer.workgraph_captures()), 1)
            self.assertEqual(len(consumer.workgraph_attachments()), 1)
            self.assertIsInstance(state.physical_plan_store, PlanSnapshotStore)
            self.assertIsNotNone(consumer.workgraph_execution_stats())
            self.assertEqual(state.physical_prepared_step_id, -1)

    def test_disabled_esamp_cold_specialization_is_a_no_op(self) -> None:
        consumer = ESampConsumer(
            ESampConsumerConfig(enable_esamp_training=False),
            engine=mock.Mock(),
        )
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.cold_specialize_workgraph(_spec())

            self.assertEqual(consumer.workgraph_captures(), ())
            self.assertEqual(consumer.workgraph_attachments(), ())
            self.assertIsNone(state.physical_plan_store)

    def test_repeating_the_same_spec_does_not_advance_physical_epoch(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=mock.Mock())
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
            residual_runtime.cold_specialize_workgraph(_spec())
            store = state.physical_plan_store
            self.assertIsInstance(store, PlanSnapshotStore)
            assert isinstance(store, PlanSnapshotStore)
            first_epoch = store.current.plan.plan_epoch
            first_snapshot = state.workgraph_snapshot

            residual_runtime.cold_specialize_workgraph(_spec())

            self.assertIs(state.workgraph_snapshot, first_snapshot)
            self.assertEqual(store.current.plan.plan_epoch, first_epoch)

    def test_runtime_config_toggle_preserves_load_time_specialization(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=mock.Mock())
        consumer.cold_specialize_workgraph(_spec())
        captured = consumer.workgraph_captures()

        consumer.update_config(
            ESampConsumerConfig(
                graph_scratch_rows=4,
                enable_esamp_training=False,
            )
        )
        self.assertEqual(consumer.workgraph_captures(), ())

        consumer.update_config(ESampConsumerConfig(graph_scratch_rows=4))
        self.assertEqual(consumer.workgraph_captures(), captured)

    def test_full_mode_retains_cold_metadata_without_physical_activation(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=mock.Mock())
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(GraphMode.FULL),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
            residual_runtime.cold_specialize_workgraph(_spec())

            self.assertEqual(state.workgraph_captures, consumer.workgraph_captures())
            self.assertEqual(state.workgraph_attachments, consumer.workgraph_attachments())
            self.assertIsNone(state.physical_plan_store)
            self.assertIsNone(consumer.workgraph_execution_stats())

    def test_prepare_and_consume_do_not_capture(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
            residual_runtime.cold_specialize_workgraph(_spec())

            with mock.patch(
                "tllm.workgraph.capture.capture_flow",
                side_effect=AssertionError("capture is cold-path only"),
            ) as capture_flow, mock.patch.object(
                consumer,
                "_ensure_runtime_resources",
                return_value=None,
            ):
                consumer.prepare_workgraph_step(1)
                consumer.consume_bundle(_bundle(), _ctx())

        capture_flow.assert_not_called()
        engine.launch_step.assert_called_once()

    def test_cold_specialized_executor_performs_training_without_consume_wrapper(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(graph_scratch_rows=4),
            engine=engine,
        )
        state = ResidualRuntimeState()
        source = torch.randn(4, 8)
        target = torch.randn(4, 8)

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
            residual_runtime.cold_specialize_workgraph(_spec())
            consumer.prepare_workgraph_step(1)
            executor = getattr(consumer, "_workgraph_executor", None)
            self.assertIsInstance(executor, PhysicalExecutor)
            assert isinstance(executor, PhysicalExecutor)
            executor.execute_step(1, tensors=(source, target))
            consumer.finish_workgraph_step(1)

        engine.launch_step.assert_called_once_with(source, target)

    def test_mrv2_prepare_safe_point_uses_new_event_step_id(self) -> None:
        state = ResidualRuntimeState()
        state.event_step_id = 4
        runner = SimpleNamespace(model=None)
        orig = mock.Mock(return_value="executed")

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(v027_mrv2.active_targets, "runtime_has_active_targets", return_value=True),
            mock.patch.object(v027_mrv2._common_hooks, "dispatch_runtime_event"),
            mock.patch.object(v027_mrv2._hooks, "setup_runtime_hooks_if_active"),
            mock.patch.object(residual_runtime, "prepare_physical_workgraph_step") as prepare,
        ):
            result = v027_mrv2._wrapped_execute_model(
                core=residual_runtime,
                runner=runner,
                args=(),
                kwargs={},
                orig=orig,
            )

        self.assertEqual(result, "executed")
        prepare.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()

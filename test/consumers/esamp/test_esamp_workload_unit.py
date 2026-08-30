from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import torch

from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.initializers.svd import (
    SVDModelBankInitializerConfig,
)
from tllm.consumers.esamp.legacy_workload import build_esamp_workload
from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_plugins import reconstruct_worker_bindings
from tllm.runtime.workload_bindings import (
    LocalRuntimeServices,
    SamplerRuntimeBinding,
    WorkerRuntimeBinding,
)
from tllm.runtime.mp.worker_extension import (
    TllmWorkerExtension,
    _decision_requires_hidden_taps,
)
from tllm.ports.residual_stream import ResidualLocator
from tllm.workload import declare_workload


def _config(**overrides: object) -> ESampConfig:
    values: dict[str, object] = {
        "intervention": True,
        "distiller_beta": 0.25,
        "distiller_hidden_dim": 3,
    }
    values.update(overrides)
    return ESampConfig(**values)  # type: ignore[arg-type]


def _decode_state(step_rows: int = 2) -> ResidualRuntimeState:
    state = ResidualRuntimeState()
    state.decode_count = step_rows
    state.decode_request_ids = [f"request-{row}" for row in range(step_rows)]
    state.decode_prompt_idxs = [10 + row for row in range(step_rows)]
    state.decode_sample_idxs = [row for row in range(step_rows)]
    return state


class _Model:
    def __init__(self) -> None:
        self.lm_head = torch.nn.Linear(4, 7, bias=False)


def _view(step_id: int, rows: int = 2, *, request_ids: tuple[str, ...] | None = None) -> SamplerStepView:
    return SamplerStepView(
        engine_step_id=step_id,
        phase="decode",
        logits=torch.empty(rows, 7),
        sampling_metadata=object(),
        decode_count=rows,
        request_ids=request_ids or tuple(f"request-{row}" for row in range(rows)),
        prompt_idxs=tuple(10 + row for row in range(rows)),
        sample_idxs=tuple(row for row in range(rows)),
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=torch.empty(rows, 4),
        device=torch.device("cpu"),
        model=_Model(),
        runner=object(),
    )


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="vllm-0.27-mrv2",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.EAGER_ONLY,
        hidden_locators=frozenset({HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}),
    )


def _timeline() -> object:
    return export_main_workload_timeline(
        _capabilities(), graph_mode="eager", static_headroom_bytes=1 << 20
    )


def _specialized_workload(**overrides: object) -> ESampWorkload:
    workload = ESampWorkload(_config(**overrides))
    workload.cold_specialize(
        RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
    )
    return workload


def _publish_guidance(workload: ESampWorkload, step_id: int) -> torch.Tensor:
    """Drive the real signed sink path: task output -> auto-bound store."""
    prediction, _materialized, _target, _sample_indices = workload.predict(
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.zeros(2, dtype=torch.long),
    )
    with mock.patch.object(residual_runtime, "RUNTIME", _decode_state(2)):
        workload.native_workload_resources().ports.outputs["guidance"].publish(
            prediction, step_id
        )
    return prediction


class ESampRegionUnitTest(unittest.TestCase):
    def test_piecewise_capture_is_limited_to_read_only_prediction(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(
                2, 4, torch.float32, torch.device("cpu")
            )
        )
        for graph_mode in ("piecewise", "full"):
            with self.subTest(graph_mode=graph_mode):
                compilation = runtime.compile_workloads(
                    replace(_timeline(), graph_mode=graph_mode)
                )
                regions = {
                    task_name: region
                    for region in compilation.decision.execution_regions
                    for task_name in region.tasks
                }

                self.assertTrue(regions["predict"].capture_safe)
                self.assertEqual(
                    regions["predict"].capture_mode.value, "require"
                )
                for task_name in ("loss", "loss__optimize_distiller"):
                    self.assertFalse(regions[task_name].capture_safe)
                    self.assertEqual(
                        regions[task_name].capture_mode.value, "disabled"
                    )

    def test_native_dispatch_rejects_every_legacy_only_option(self) -> None:
        cases: tuple[tuple[dict[str, object], str], ...] = (
            ({"consumer_id": "custom"}, "consumer_id"),
            ({"graph_scratch_rows": 8}, "graph_scratch_rows"),
            ({"enable_workgraph_execution": False}, "enable_workgraph_execution"),
            ({"per_request_models": True}, "per_request_models"),
            (
                {
                    "per_request_models": True,
                    "per_request_model_bank": True,
                },
                "per_request_model_bank",
            ),
            ({"model_bank_slots": 2}, "model_bank_slots"),
            ({"model_bank_flush_interval": 2}, "model_bank_flush_interval"),
            ({"model_bank_rank": 32}, "model_bank_rank"),
            (
                {"model_bank_initializer": SVDModelBankInitializerConfig()},
                "model_bank_initializer",
            ),
            ({"model_bank_train_cudagraph": True}, "model_bank_train_cudagraph"),
            ({"model_bank_forward_backend": "triton"}, "model_bank_forward_backend"),
            ({"adaptation_pipeline_slots": 2}, "adaptation_pipeline_slots"),
            ({"adaptation_stream_mode": "single"}, "adaptation_stream_mode"),
            ({"adaptation_stream_priority": 1}, "adaptation_stream_priority"),
            ({"trace_per_request_losses": True}, "trace_per_request_losses"),
            ({"trace_interval": 2}, "trace_interval"),
            ({"trace_max_points": 2}, "trace_max_points"),
            ({"compact_capture_lane": True}, "compact_capture_lane"),
            ({"distiller_sampler_backend": "pre_filter_dense"}, "distiller_sampler_backend"),
        )
        for overrides, option in cases:
            with self.subTest(option=option):
                config = ESampConsumerConfig(
                    implementation="native", **overrides
                )
                with self.assertRaisesRegex(ValueError, option):
                    build_esamp_workload(config)

    def test_native_dispatch_preserves_explicit_training_row_selection(self) -> None:
        workload = build_esamp_workload(
            ESampConsumerConfig(
                implementation="native",
                training_row_selection="first_per_prompt",
            )
        )
        self.assertIsInstance(workload, ESampWorkload)
        assert isinstance(workload, ESampWorkload)
        self.assertEqual(
            workload.config.training_row_selection,
            "first_per_prompt",
        )

    def test_first_per_prompt_selection_uses_typed_sample_identity(self) -> None:
        workload = _specialized_workload(
            training_row_selection="first_per_prompt"
        )
        source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        target = -source

        prediction, training_source, loss_target, sample_index_tensor = workload.predict(
            source,
            target,
            torch.tensor((0, 1, 0, 2)),
        )
        loss = workload.loss(
            training_source, loss_target, sample_index_tensor
        )

        expected_prediction = workload.distiller(source)
        torch.testing.assert_close(
            prediction, expected_prediction
        )
        assert loss is not None
        torch.testing.assert_close(
            loss,
            torch.nn.functional.mse_loss(
                expected_prediction.index_select(0, torch.tensor((0, 2))).float(),
                target.index_select(0, torch.tensor((0, 2))).float(),
            ),
        )

    def test_native_region_rejects_a_coinstalled_legacy_consumer(self) -> None:
        native = ESampWorkload(_config())
        legacy = ESampConsumer(ESampConsumerConfig(implementation="legacy"))

        with self.assertRaisesRegex(RuntimeError, "native Region authority"):
            TllmRuntime(workloads=(native,), consumers=(legacy,))

    def test_disabled_workload_exports_a_disabled_immutable_worker_spec(self) -> None:
        workload = ESampWorkload(_config(enabled=True))

        workload.set_enabled(False)

        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec
        self.assertFalse(worker_spec.config["enabled"])
        reconstructed = ESampWorkload.reconstruct(worker_spec.config)
        assert isinstance(reconstructed.services, LocalRuntimeServices)
        self.assertFalse(reconstructed.services.is_enabled())

    def test_new_worker_epoch_refreshes_the_current_enable_snapshot(self) -> None:
        workload = ESampWorkload(_config(enabled=True))
        runtime = TllmRuntime(workloads=(workload,))

        workload.set_enabled(False)
        runtime._refresh_native_worker_specs()

        worker_spec = runtime._workload_worker_specs[0]
        self.assertFalse(worker_spec.config["enabled"])
        self.assertIn(
            ("esamp/enabled", "False"),
            runtime.workload_compile_options.shape_specialization,
        )

    def test_in_process_enable_transition_publishes_a_new_immutable_epoch(self) -> None:
        workload = ESampWorkload(_config(enabled=True))
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        initial = runtime.compile_workloads(_timeline())
        initial_model = workload.distiller
        initial_optimizer = workload.optimizer
        initial_state_version = workload.state_version
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                initial, bindings=runtime.workload_execution_bindings
            )
            old_execution = state.workload_execution
            old_store = state.workload_execution_store
            assert old_execution is not None
            assert old_store is not None
            assert old_execution._value_store is not None
            self.assertTrue(old_execution._value_store.is_enabled())
            with self.assertRaisesRegex(RuntimeError, "immutable"):
                old_execution.set_enabled(False)

            self.assertTrue(runtime.transition_native_workload_enabled(False))

            replacement = runtime.workload_compilation
            assert replacement is not None
            self.assertNotEqual(replacement.cache_key, initial.cache_key)
            self.assertNotEqual(
                replacement.decision.signature, initial.decision.signature
            )
            self.assertNotEqual(
                replacement.decision.execution_regions[0].artifact_key,
                initial.decision.execution_regions[0].artifact_key,
            )
            self.assertIs(old_store.current, old_execution)
            from tllm.runtime.workload_execution import StagedExecution

            self.assertIsInstance(old_store.staged, StagedExecution)
            self.assertTrue(old_execution._value_store.is_enabled())
            self.assertIs(workload.distiller, initial_model)
            self.assertIs(workload.optimizer, initial_optimizer)
            self.assertEqual(workload.state_version, initial_state_version)
            worker = workload.native_workload_resources().worker
            assert isinstance(worker, WorkerRuntimeBinding)
            worker_spec = worker.spec
            self.assertFalse(worker_spec.config["enabled"])

            residual_runtime.prepare_physical_workgraph_step(1)
            current = state.workload_execution
            assert current is not None
            assert current._value_store is not None
            self.assertIsNot(current, old_execution)
            self.assertFalse(current._value_store.is_enabled())

    def test_mp_install_cold_binds_hidden_taps_before_first_forward(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        compilation = runtime.compile_workloads(_timeline())
        self.assertTrue(_decision_requires_hidden_taps(compilation.decision))
        extension = TllmWorkerExtension()
        extension.model_runner = mock.Mock(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=16,
            model_config=mock.Mock(
                dtype=torch.float32,
                get_hidden_size=mock.Mock(return_value=4),
                get_num_layers=None,
            ),
            model=torch.nn.Module(),
        )
        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec
        state = residual_runtime.RUNTIME
        saved = (
            state.native_schedule_graph,
            state.native_workloads_pending,
            dict(state.residual_raw_paths),
            state.workload_compilation,
            state.workload_execution,
            state.physical_executor,
        )
        try:
            with mock.patch(
                "tllm.runtime.vllm_patch.port_runtime_hooks."
                "setup_runtime_hooks_if_active"
            ) as setup:
                report = extension.tllm_install_workload_decision(
                    compilation.cache_key,
                    compilation.graph.to_json(),
                    compilation.timeline.to_json(),
                    compilation.options.to_json(),
                    compilation.decision.to_json(),
                    compilation.physical.plan_epoch,
                    [worker_spec.to_wire()],
                )
            setup.assert_called_once_with(core=residual_runtime, runner=extension.model_runner)
            self.assertTrue(report["native"])
            self.assertFalse(state.native_workloads_pending)
            self.assertEqual(state.native_schedule_graph, compilation.graph)
            self.assertEqual(len(state.residual_raw_paths), 2)
        finally:
            current = state.workload_execution
            if current is not None and current is not saved[4]:
                current.close()
            (
                state.native_schedule_graph,
                state.native_workloads_pending,
                residual_paths,
                state.workload_compilation,
                state.workload_execution,
                state.physical_executor,
            ) = saved
            state.residual_raw_paths = residual_paths

    def test_worker_spec_reconstructs_exact_signed_graph_and_bindings(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        spec = RuntimeSpecializationSpec(
            2, 4, torch.float32, torch.device("cpu")
        )
        runtime.cold_specialize_workloads(spec)
        compilation = runtime.compile_workloads(_timeline())
        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec

        bindings = reconstruct_worker_bindings(
            compilation, (worker_spec,), specialization=spec
        )

        self.assertEqual(
            bindings.declarations[0].ir.semantic_key,
            declare_workload(workload).ir.semantic_key,
        )
        self.assertEqual(set(bindings.ports.outputs), {"guidance"})
        self.assertEqual(set(bindings.state_versions), {"distiller"})
        self.assertIs(type(bindings.sampler), SamplerRuntimeBinding)

    def test_worker_reconstruction_rejects_a_signed_specialization_mismatch(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        expected = RuntimeSpecializationSpec(
            2, 4, torch.float32, torch.device("cpu")
        )
        runtime.cold_specialize_workloads(expected)
        compilation = runtime.compile_workloads(_timeline())
        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec

        with self.assertRaisesRegex(RuntimeError, "specialization does not match"):
            reconstruct_worker_bindings(
                compilation,
                (worker_spec,),
                specialization=RuntimeSpecializationSpec(
                    2, 4, torch.float64, torch.device("cpu")
                ),
            )

    def test_signed_physical_regions_drive_one_complete_training_step(self) -> None:
        workload = ESampWorkload(_config())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        compilation = runtime.compile_workloads(_timeline())
        signed_tasks = tuple(
            task
            for region in compilation.decision.execution_regions
            for task in region.tasks
        )
        self.assertEqual(
            set(signed_tasks),
            {
                "predict",
                "loss",
                "loss__optimize_distiller",
            },
        )
        self.assertEqual(len(signed_tasks), 3)

        state = _decode_state(2)
        source_locator = ResidualLocator(0, "block_output", "decode")
        target_locator = ResidualLocator(-1, "block_output", "decode")
        state.residual_bindings = {
            "source": ResidualPathBinding(source_locator, "source"),
            "target": ResidualPathBinding(target_locator, "target"),
        }
        state.tap_decode_hidden = {
            "source": torch.randn(2, 4),
            "target": torch.randn(2, 4),
        }
        before = tuple(
            parameter.detach().clone() for parameter in workload.distiller.parameters()
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=runtime.workload_execution_bindings,
            )
            residual_runtime.prepare_physical_workgraph_step(7)
            sampler_binding = workload.native_workload_resources().sampler
            assert isinstance(sampler_binding, SamplerRuntimeBinding)
            sampler_binding.stage_identity(
                7,
                tuple(state.decode_request_ids),
                tuple(state.decode_sample_idxs),
                torch.tensor(state.decode_sample_idxs),
            )
            for region in compilation.decision.execution_regions:
                residual_runtime.advance_workload_anchor(region.placement_anchor)

            execution = state.workload_execution
            assert execution is not None
            self.assertEqual(
                execution.state_versions.observed_version("distiller", None), 1
            )
            stats = residual_runtime.read_and_reset_esamp_stats()

        after = tuple(parameter.detach() for parameter in workload.distiller.parameters())
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, after, strict=True)
            )
        )
        self.assertEqual(workload.state_version, 1)
        self.assertEqual(stats.loss_count, 1)
        services = workload.native_workload_resources().services
        assert isinstance(services, LocalRuntimeServices)
        self.assertEqual(services.diagnostics()["guidance_rows"], 2)

    def test_declaration_has_replica_state_and_no_legacy_adapter(self) -> None:
        workload = ESampWorkload(_config())
        declaration = declare_workload(workload)

        self.assertFalse(hasattr(workload, "legacy_consumer"))
        self.assertEqual(declaration.ir.states[0].scope.value, "replica")
        self.assertEqual(
            tuple(task.name for task in declaration.ir.tasks),
            (
                "predict",
                "loss",
                "loss__optimize_distiller",
            ),
        )
        self.assertEqual(declaration.ir.tasks[0].outputs[0].target, "guidance")
        self.assertEqual(declaration.ir.tasks[0].outputs[1].kind.value, "temporary")

    def test_cold_specialize_forward_update_and_stats(self) -> None:
        workload = ESampWorkload(_config())
        resources = workload.native_workload_resources()
        self.assertEqual(resources.states.snapshots["distiller"](0), {None: 0})
        assert isinstance(resources.services, LocalRuntimeServices)
        diagnostics = resources.services.diagnostics
        self.assertEqual(
            diagnostics(),
            {"guidance_rows": 0, "loss_count": 0, "update_count": 0},
        )
        with self.assertRaises(RuntimeError):
            workload.distiller  # parameters do not exist before cold_specialize

        with torch.inference_mode():
            workload.cold_specialize(
                RuntimeSpecializationSpec(
                    rows=2,
                    hidden_size=4,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
            )
        self.assertIsNotNone(workload.optimizer)
        self.assertFalse(any(parameter.is_inference() for parameter in workload.distiller.parameters()))
        before = tuple(parameter.detach().clone() for parameter in workload.distiller.parameters())
        _publish_guidance(workload, 7)
        self.assertEqual(diagnostics()["guidance_rows"], 2)

        # The signed update path (extern task body is one line of logic).
        loss = torch.nn.functional.mse_loss(
            workload.distiller(torch.randn(2, 4)), torch.randn(2, 4)
        )
        executors = dict(workload.declaration().executors)
        executors["loss__optimize_distiller"](loss)

        after = tuple(parameter.detach() for parameter in workload.distiller.parameters())
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, after, strict=True)))
        self.assertEqual(resources.states.snapshots["distiller"](0), {None: 1})
        self.assertEqual(
            diagnostics(),
            {"guidance_rows": 2, "loss_count": 1, "update_count": 1},
        )

    def test_cold_specialization_rejects_any_changed_static_fact(self) -> None:
        workload = ESampWorkload(_config())
        workload.cold_specialize(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )

        for mismatch in (
            RuntimeSpecializationSpec(3, 4, torch.float32, torch.device("cpu")),
            RuntimeSpecializationSpec(2, 4, torch.float64, torch.device("cpu")),
        ):
            with self.subTest(mismatch=mismatch):
                with self.assertRaisesRegex(RuntimeError, "specialized"):
                    workload.cold_specialize(mismatch)

    def test_sampler_semantic_config_changes_the_signed_cache_key(self) -> None:
        timeline = _timeline()

        def compile_for(beta: float, intervention: bool) -> object:
            runtime = TllmRuntime(
                workloads=(
                    ESampWorkload(
                        _config(
                            distiller_beta=beta,
                            intervention=intervention,
                        )
                    ),
                ),
                workload_memory_budget_bytes=1 << 20,
            )
            runtime.cold_specialize_workloads(
                RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
            )
            return runtime.compile_workloads(timeline)

        baseline = compile_for(0.25, True)
        changed_beta = compile_for(0.75, True)
        disabled_intervention = compile_for(0.25, False)
        self.assertNotEqual(baseline.cache_key, changed_beta.cache_key)
        self.assertNotEqual(baseline.cache_key, disabled_intervention.cache_key)

    def test_sampler_reads_only_same_step_guidance(self) -> None:
        workload = _specialized_workload()
        _publish_guidance(workload, 11)

        sampler = workload.native_workload_resources().sampler
        assert isinstance(sampler, SamplerRuntimeBinding)
        with mock.patch.object(
            workload.distiller,
            "forward",
            side_effect=AssertionError("sampler must not re-run the Region model"),
        ) as forward:
            state = sampler.provider.prepare_step(_view(11))
        forward.assert_not_called()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.affected_row_ids.tolist(), [0, 1])
        self.assertEqual(tuple(state.pred_hidden.shape), (2, 4))
        self.assertIsNone(sampler.provider.prepare_step(_view(12)))
        with self.assertRaisesRegex(RuntimeError, "active step"):
            sampler.provider.prepare_step(
                _view(11, request_ids=("different", "request-1"))
            )


if __name__ == "__main__":
    unittest.main()

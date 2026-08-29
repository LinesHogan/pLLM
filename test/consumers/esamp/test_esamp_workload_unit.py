from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.consumers.esamp.config import ESampConsumerConfig
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
    prediction, _loss = workload.predict_and_loss(torch.randn(2, 4), torch.randn(2, 4))
    with mock.patch.object(residual_runtime, "RUNTIME", _decode_state(2)):
        workload.native_workload_resources().output_sinks["guidance"](prediction, step_id)
    return prediction


class ESampRegionUnitTest(unittest.TestCase):
    def test_native_region_rejects_a_coinstalled_legacy_consumer(self) -> None:
        native = ESampWorkload(_config())
        legacy = ESampConsumer(ESampConsumerConfig(implementation="legacy"))

        with self.assertRaisesRegex(RuntimeError, "native Region authority"):
            TllmRuntime(workloads=(native,), consumers=(legacy,))

    def test_disabled_workload_exports_a_disabled_immutable_worker_spec(self) -> None:
        workload = ESampWorkload(_config(enabled=True))

        workload.set_enabled(False)

        worker_spec = workload.native_workload_resources().worker_spec
        assert worker_spec is not None
        self.assertFalse(worker_spec.config["enabled"])
        reconstructed = ESampWorkload.reconstruct(worker_spec.config)
        assert reconstructed.enabled_predicate is not None
        self.assertFalse(reconstructed.enabled_predicate())

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
            self.assertIsNotNone(old_store.staged)
            self.assertTrue(old_execution._value_store.is_enabled())
            self.assertIs(workload.distiller, initial_model)
            self.assertIs(workload.optimizer, initial_optimizer)
            self.assertEqual(workload.state_version, initial_state_version)
            worker_spec = workload.native_workload_resources().worker_spec
            assert worker_spec is not None
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
        worker_spec = workload.native_workload_resources().worker_spec
        assert worker_spec is not None
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
        worker_spec = workload.native_workload_resources().worker_spec
        assert worker_spec is not None

        bindings = reconstruct_worker_bindings(
            compilation, (worker_spec,), specialization=spec
        )

        self.assertEqual(
            bindings.declarations[0].ir.semantic_key,
            declare_workload(workload).ir.semantic_key,
        )
        self.assertEqual(set(bindings.output_sinks), {"guidance"})
        self.assertEqual(set(bindings.state_versions), {"distiller"})
        self.assertIsNotNone(bindings.sampler_provider)

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
        worker_spec = workload.native_workload_resources().worker_spec
        assert worker_spec is not None

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
        self.assertEqual(
            tuple(region.tasks for region in compilation.decision.execution_regions),
            (("predict_and_loss",), ("predict_and_loss__optimize_distiller",)),
        )
        self.assertEqual(
            tuple(
                region.implementation_kind
                for region in compilation.decision.execution_regions
            ),
            ("compiled", "extern"),
        )
        self.assertEqual(len(compilation.decision.materializations), 1)

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
        diagnostics = workload.native_workload_resources().diagnostics_reader
        assert diagnostics is not None
        self.assertEqual(diagnostics()["guidance_rows"], 2)

    def test_declaration_has_replica_state_and_no_legacy_adapter(self) -> None:
        workload = ESampWorkload(_config())
        declaration = declare_workload(workload)

        self.assertFalse(hasattr(workload, "legacy_consumer"))
        self.assertEqual(declaration.ir.states[0].scope.value, "replica")
        self.assertEqual(
            tuple(task.name for task in declaration.ir.tasks),
            ("predict_and_loss", "predict_and_loss__optimize_distiller"),
        )
        self.assertEqual(declaration.ir.tasks[0].outputs[0].target, "guidance")
        self.assertEqual(declaration.ir.tasks[0].outputs[1].kind.value, "temporary")

    def test_cold_specialize_forward_update_and_stats(self) -> None:
        workload = ESampWorkload(_config())
        resources = workload.native_workload_resources()
        self.assertEqual(resources.state_snapshots["distiller"](0), {None: 0})
        diagnostics = resources.diagnostics_reader
        assert diagnostics is not None
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
        executors["predict_and_loss__optimize_distiller"](loss)

        after = tuple(parameter.detach() for parameter in workload.distiller.parameters())
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, after, strict=True)))
        self.assertEqual(resources.state_snapshots["distiller"](0), {None: 1})
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

        provider = workload.native_workload_resources().sampler_provider
        assert provider is not None
        with mock.patch.object(
            workload.distiller,
            "forward",
            side_effect=AssertionError("sampler must not re-run the Region model"),
        ) as forward:
            state = provider.prepare_step(_view(11))
        forward.assert_not_called()
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.affected_row_ids.tolist(), [0, 1])
        self.assertEqual(tuple(state.pred_hidden.shape), (2, 4))
        self.assertIsNone(provider.prepare_step(_view(12)))
        with self.assertRaisesRegex(RuntimeError, "active step"):
            provider.prepare_step(_view(11, request_ids=("different", "request-1")))


if __name__ == "__main__":
    unittest.main()

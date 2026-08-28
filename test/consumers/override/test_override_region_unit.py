from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.consumers.override.config import OverRIDEConsumerConfig
from tllm.consumers.override.consumer import OverRIDEConsumer
from tllm.consumers.override.region_workload import (
    OverRIDERegionWorkload,
    iteration_from_request_id,
)
from tllm.consumers.override.register import build as build_legacy
from tllm.consumers.override.workload import build_override_workload
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.workload import declare_workload


def _config(**overrides: object) -> OverRIDEConsumerConfig:
    values: dict[str, object] = {
        "implementation": "region",
        "hidden_size": 4,
        "vocab_size": 7,
        "rank": 2,
        "num_iterations": 2,
        "lambd": 0.8,
        "learning_rate": 1e-3,
    }
    values.update(overrides)
    return OverRIDEConsumerConfig(**values)


def _spec(rows: int = 2) -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=rows,
        hidden_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _decode_state(request_ids: tuple[str, ...]) -> ResidualRuntimeState:
    rows = len(request_ids)
    state = ResidualRuntimeState()
    state.decode_count = rows
    state.decode_request_ids = list(request_ids)
    state.decode_prompt_idxs = list(range(rows))
    state.decode_sample_idxs = [0] * rows
    return state


def _view(
    step_id: int,
    request_ids: tuple[str, ...],
    *,
    logits: torch.Tensor,
    source_hidden: torch.Tensor,
) -> SamplerStepView:
    rows = len(request_ids)
    return SamplerStepView(
        engine_step_id=step_id,
        phase="decode",
        logits=logits,
        sampling_metadata=object(),
        decode_count=rows,
        request_ids=request_ids,
        prompt_idxs=tuple(range(rows)),
        sample_idxs=tuple(0 for _ in range(rows)),
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=source_hidden,
        device=logits.device,
        model=object(),
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


def _publish_reweight(
    workload: OverRIDERegionWorkload,
    residual: torch.Tensor,
    step_id: int,
    request_ids: tuple[str, ...],
) -> None:
    """Drive the real signed sink path: task output -> auto-bound store."""
    with mock.patch.object(
        residual_runtime, "RUNTIME", _decode_state(request_ids)
    ):
        workload.native_workload_resources().output_sinks["reweight"](
            residual, step_id
        )


class OverRIDERegionDeclarationUnitTest(unittest.TestCase):
    def test_declaration_signs_replica_state_with_adam(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        declaration = declare_workload(workload)

        self.assertFalse(hasattr(workload, "legacy_consumer"))
        self.assertEqual(len(declaration.ir.states), 1)
        state = declaration.ir.states[0]
        self.assertEqual(state.name, "heads")
        self.assertEqual(state.scope.value, "replica")
        assert state.optimizer is not None
        self.assertEqual(state.optimizer.kind, "adam")
        self.assertEqual(
            tuple(task.name for task in declaration.ir.tasks),
            ("reweight", "train", "update"),
        )
        reweight, train, update = declaration.ir.tasks
        self.assertEqual(reweight.outputs[0].target, "reweight")
        self.assertEqual(reweight.timing.after, "model.execute")
        self.assertEqual(reweight.timing.before, "sampler.pre_draw")
        self.assertEqual(train.timing.after, "token.commit")
        self.assertEqual(train.timing.before, "step.end")
        self.assertEqual(update.timing.before, "step.end")
        # Zero-kept-row steps skip the optimizer update without a version
        # bump, so the update transition is signed as conditional.
        self.assertEqual(
            [effect.kind.value for effect in update.effects], ["transition_maybe"]
        )

    def test_iteration_parse_matches_legacy_rules(self) -> None:
        self.assertEqual(iteration_from_request_id("plain"), 1)
        self.assertEqual(iteration_from_request_id("0_req"), 1)
        self.assertEqual(iteration_from_request_id("3_req"), 4)
        self.assertEqual(iteration_from_request_id("x_req"), 1)
        legacy = OverRIDEConsumer(OverRIDEConsumerConfig())
        for request_id in ("plain", "0_req", "3_req", "x_req"):
            self.assertEqual(
                iteration_from_request_id(request_id),
                legacy.iteration_from_request_id(request_id),
            )

    def test_unsupported_region_configs_fail_closed(self) -> None:
        for overrides, needle in (
            ({"implementation": "consumer"}, "implementation='region'"),
            ({"num_iterations": 1}, "num_iterations"),
            ({"rank": 0}, "rank"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"vocab_size": 0}, "vocab_size"),
            ({"source_layer_path": ""}, "source_layer_path"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, needle):
                    OverRIDERegionWorkload(_config(**overrides))

    def test_dispatcher_and_legacy_register_are_switchable(self) -> None:
        native = build_override_workload(_config())
        self.assertIsInstance(native, OverRIDERegionWorkload)
        with self.assertRaisesRegex(ValueError, "legacy Program consumer"):
            build_override_workload(_config(implementation="consumer"))
        with self.assertRaisesRegex(ValueError, "unknown OverRIDE implementation"):
            build_override_workload(_config(implementation="bogus"))
        with self.assertRaisesRegex(ValueError, "native @workload"):
            build_legacy(_config())
        legacy = build_legacy(_config(implementation="consumer"))
        self.assertIsInstance(legacy, OverRIDEConsumer)


class OverRIDERegionParityUnitTest(unittest.TestCase):
    def _build_parity_pair(
        self,
    ) -> tuple[OverRIDEConsumer, OverRIDERegionWorkload]:
        config = _config(implementation="consumer", num_iterations=3, learning_rate=0.05)
        legacy = OverRIDEConsumer(config)
        hidden = torch.zeros((2, 4), dtype=torch.float32)
        logits = torch.zeros((2, 7), dtype=torch.float32)
        legacy.reweight_logits_batch(hidden, logits, iterations=(1, 1))
        compiled = legacy.compiled
        assert compiled is not None
        params = compiled.bank("heads").slot_for(None).params

        native = OverRIDERegionWorkload(
            _config(num_iterations=3, learning_rate=0.05)
        )
        native.cold_specialize(_spec())
        with torch.no_grad():
            native.param("down").copy_(params["down"])
            native.param("up").copy_(params["up"])
        return legacy, native

    def test_reweight_residual_and_training_match_the_legacy_program(self) -> None:
        legacy, native = self._build_parity_pair()
        torch.manual_seed(7)
        hidden = torch.randn(3, 4)
        logits = torch.randn(3, 7)
        tokens = torch.tensor([0, 3, 5], dtype=torch.long)
        # iterations (1, 2, 3): row 0 trains head 0, row 1 reweights with
        # head 0 and trains head 1, row 2 reweights with head 1 only.
        request_ids = ("0_a", "1_b", "2_c")

        reweighted = legacy.reweight_logits_batch(
            hidden, logits, request_ids=request_ids
        )
        legacy_residual = logits - reweighted

        head_idx = torch.tensor([0, 0, 1], dtype=torch.long)
        scale = torch.tensor([[0.0], [0.8], [0.8]], dtype=torch.float32)
        residual, _kept = native.reweight(hidden, head_idx, scale)
        _publish_reweight(native, residual, 3, request_ids)
        self.assertTrue(torch.allclose(residual, legacy_residual, atol=1e-6))

        legacy.update_from_sampled_tokens(tokens)
        legacy_loss = legacy.read_stats().last_loss

        native_loss = native.train(
            hidden,
            torch.tensor([0, 1], dtype=torch.long),
            logits[:2],
            torch.tensor([0, 1], dtype=torch.long),
            tokens[:2],
        )
        native.update(native_loss)
        self.assertAlmostEqual(float(native_loss.detach()), legacy_loss, places=6)
        assert legacy.compiled is not None
        params = legacy.compiled.bank("heads").slot_for(None).params
        self.assertTrue(torch.allclose(native.param("down"), params["down"], atol=1e-6))
        self.assertTrue(torch.allclose(native.param("up"), params["up"], atol=1e-6))
        self.assertEqual(native.state_version, 1)
        self.assertEqual(
            native.resources.stats(),
            {"reweight_rows": 3, "loss_count": 1, "update_count": 1},
        )

    def test_zero_trainable_rows_skip_the_update(self) -> None:
        _legacy, native = self._build_parity_pair()
        torch.manual_seed(11)
        hidden = torch.randn(2, 4)
        # num_iterations=3: iteration-3 rows have no current head to train.
        loss = native.train(
            hidden,
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 7)),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )
        native.update(loss)
        self.assertFalse(bool(loss.requires_grad))
        self.assertEqual(native.state_version, 0)
        self.assertEqual(
            native.resources.stats(),
            {"reweight_rows": 0, "loss_count": 0, "update_count": 0},
        )


class OverRIDERegionSignedStepUnitTest(unittest.TestCase):
    def _compile(
        self, workload: OverRIDERegionWorkload
    ) -> tuple[TllmRuntime, object]:
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(_spec())
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                _capabilities(), graph_mode="eager", static_headroom_bytes=1 << 20
            )
        )
        return runtime, compilation

    def test_signed_regions_drive_one_complete_training_step(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        runtime, compilation = self._compile(workload)
        regions = compilation.decision.execution_regions
        self.assertEqual(
            tuple(region.tasks for region in regions),
            (("reweight",), ("train",), ("update",)),
        )
        self.assertEqual(
            tuple(region.implementation_kind for region in regions),
            ("compiled", "compiled", "extern"),
        )

        request_ids = ("0_a", "1_b")
        torch.manual_seed(13)
        hidden = torch.randn(2, 4)
        logits = torch.randn(2, 7)
        state = _decode_state(request_ids)
        locator = ResidualLocator(0, "block_output", "decode")
        state.residual_bindings = {"source": ResidualPathBinding(locator, "source")}
        state.tap_decode_hidden = {"source": hidden}
        up_before = workload.param("up").detach().clone()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=runtime.workload_execution_bindings,
            )
            residual_runtime.prepare_physical_workgraph_step(7)
            residual_runtime.advance_workload_anchor(regions[0].placement_anchor)

            record = workload.resources.sinks.for_step("reweight", 7)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.identity.engine_step_id, 7)
            self.assertEqual(record.identity.request_ids, request_ids)

            provider = workload.sampler_provider
            view = _view(7, request_ids, logits=logits, source_hidden=hidden)
            modifier = provider.prepare_step(view)
            self.assertIsNotNone(modifier)
            assert modifier is not None and modifier.precomputed_dense_logits is not None
            expected_dense = logits + record.value
            self.assertTrue(
                torch.allclose(
                    modifier.precomputed_dense_logits, expected_dense, atol=1e-6
                )
            )
            tokens = torch.tensor([2, 4], dtype=torch.long)
            provider.after_sample(view, tokens)

            for region in regions[1:]:
                residual_runtime.advance_workload_anchor(region.placement_anchor)

        # Only row 0 (iteration 1) trains head 0; row 1 (iteration 2) has no
        # current head under num_iterations=2.
        self.assertEqual(workload.state_version, 1)
        self.assertFalse(torch.equal(workload.param("up").detach(), up_before))
        stats = workload.resources.read_and_reset_stats()
        self.assertEqual(stats.loss_count, 1)
        self.assertGreater(stats.loss_avg, 0.0)

    def test_signed_regions_accept_a_zero_trainable_row_step(self) -> None:
        # Both requests are past num_iterations: no row trains, the update
        # region executes its signed conditional transition as a skip (+0),
        # and the runtime version audit must accept it.
        workload = OverRIDERegionWorkload(_config())
        runtime, compilation = self._compile(workload)
        regions = compilation.decision.execution_regions

        request_ids = ("2_a", "3_b")
        torch.manual_seed(17)
        hidden = torch.randn(2, 4)
        logits = torch.randn(2, 7)
        state = _decode_state(request_ids)
        locator = ResidualLocator(0, "block_output", "decode")
        state.residual_bindings = {"source": ResidualPathBinding(locator, "source")}
        state.tap_decode_hidden = {"source": hidden}
        up_before = workload.param("up").detach().clone()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=runtime.workload_execution_bindings,
            )
            residual_runtime.prepare_physical_workgraph_step(9)
            for region in regions:
                residual_runtime.advance_workload_anchor(region.placement_anchor)
                if region.tasks == ("reweight",):
                    record = workload.resources.sinks.for_step("reweight", 9)
                    self.assertIsNotNone(record)
                    provider = workload.sampler_provider
                    view = _view(9, request_ids, logits=logits, source_hidden=hidden)
                    provider.prepare_step(view)
                    provider.after_sample(view, torch.tensor([2, 4], dtype=torch.long))

        self.assertEqual(workload.state_version, 0)
        self.assertTrue(torch.equal(workload.param("up").detach(), up_before))
        stats = workload.resources.read_and_reset_stats()
        self.assertEqual(stats.loss_count, 0)

    def test_missing_reweight_publication_fails_closed(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        workload.cold_specialize(_spec())
        view = _view(
            5,
            ("0_a", "1_b"),
            logits=torch.zeros(2, 7),
            source_hidden=torch.zeros(2, 4),
        )
        with self.assertRaisesRegex(RuntimeError, "no signed reweight publication"):
            workload.sampler_provider.prepare_step(view)

    def test_provider_rejects_request_id_mismatch(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        workload.cold_specialize(_spec())
        _publish_reweight(workload, torch.zeros(2, 7), 5, ("0_a", "1_b"))
        view = _view(
            5,
            ("0_a", "9_z"),
            logits=torch.zeros(2, 7),
            source_hidden=torch.zeros(2, 4),
        )
        with self.assertRaisesRegex(RuntimeError, "do not match the sampler step"):
            workload.sampler_provider.prepare_step(view)

    def test_after_sample_requires_a_prepared_step(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        view = _view(
            5,
            ("0_a", "1_b"),
            logits=torch.zeros(2, 7),
            source_hidden=torch.zeros(2, 4),
        )
        with self.assertRaisesRegex(RuntimeError, "unprepared sampler step"):
            workload.sampler_provider.after_sample(view, torch.tensor([0, 1]))


class OverRIDERegionResourceUnitTest(unittest.TestCase):
    def test_worker_spec_roundtrip_preserves_the_signed_identity(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        resources = workload.native_workload_resources()
        worker_spec = resources.worker_spec
        assert worker_spec is not None
        reconstructed = OverRIDERegionWorkload.reconstruct(worker_spec.config)
        self.assertEqual(
            reconstructed.specialization_key, resources.specialization_key
        )
        self.assertEqual(
            set(reconstructed.provided_inputs), set(resources.provided_inputs)
        )
        self.assertEqual(set(reconstructed.output_sinks), set(resources.output_sinks))

    def test_set_enabled_publishes_a_new_epoch_snapshot(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        predicate = workload.native_workload_resources().enabled_predicate
        assert predicate is not None
        self.assertTrue(predicate())

        workload.set_enabled(False)

        resources = workload.native_workload_resources()
        predicate = resources.enabled_predicate
        assert predicate is not None
        self.assertFalse(predicate())
        self.assertEqual(resources.specialization_key["enabled"], "False")
        worker_spec = resources.worker_spec
        assert worker_spec is not None
        self.assertFalse(worker_spec.config["enabled"])
        self.assertFalse(workload.sampler_provider.is_active())

    def test_cold_specialization_rejects_changed_static_facts(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        workload.cold_specialize(_spec())
        with self.assertRaisesRegex(RuntimeError, "already specialized"):
            workload.cold_specialize(_spec(rows=3))
        workload.cold_specialize(_spec())  # identical spec is a no-op

    def test_tasks_require_cold_specialization(self) -> None:
        workload = OverRIDERegionWorkload(_config())
        with self.assertRaisesRegex(RuntimeError, "cold_specialize"):
            workload.reweight(
                torch.zeros(2, 4),
                torch.zeros(2, dtype=torch.long),
                torch.zeros(2, 1),
            )


if __name__ == "__main__":
    unittest.main()

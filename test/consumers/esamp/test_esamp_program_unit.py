#!/usr/bin/env python3
"""Program-backed ESamp consumer tests (plan Phase 4).

Covers: config validation (unsupported combinations fail at startup), the
generated consumer's training semantics (REPLICA and REQUEST scopes, varying
batch shapes via dynamic dims, per-prompt dedup), parameter-trajectory parity
with a hand-rolled AdamW reference, sampler-intervention provider behavior,
and the no-ESampTrainEngine-import rule for the program path.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.initializers.svd import SVDModelBankInitializerConfig
from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.consumers.esamp.program_consumer import (
    ESampProgramConsumer,
    _GuidanceSink,
    _ProgramSamplerModifierProvider,
    validate_program_config,
)
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.runtime.sampler_bridge.types import SamplerStepView

_HIDDEN = 16


def _bundle(
    source: torch.Tensor,
    target: torch.Tensor,
    prompt_idxs: tuple[int, ...],
    *,
    step_id: int,
) -> PortBundle:
    rows = int(source.shape[0])
    return PortBundle(
        key=BundleKey(
            engine_step_id=step_id, phase="decode", request_id="req", sample_idx=0
        ),
        entries={
            "device_lease": DeviceTensorLease(
                entries={"source": source, "target": target}, active_rows=rows
            ),
            "request_meta": RowBatchMeta(
                request_ids=tuple(f"req-{i}" for i in range(rows)),
                prompt_idxs=prompt_idxs,
                sample_idxs=tuple(0 for _ in range(rows)),
                phase="decode",
                engine_step_id=step_id,
            ),
        },
    )


def _consumer(**overrides: object) -> ESampProgramConsumer:
    kwargs = {"distiller_hidden_dim": 8, "distiller_lr": 1e-2}
    kwargs.update(overrides)
    return ESampProgramConsumer(ESampConsumerConfig(**kwargs))


class ProgramConfigValidationTest(unittest.TestCase):
    def test_default_implementation_is_legacy(self) -> None:
        self.assertEqual(ESampConsumerConfig().implementation, "legacy")

    def test_unknown_implementation_fails(self) -> None:
        with self.assertRaises(ValueError):
            ESampConsumerConfig(implementation="eager")

    def test_native_selector_cannot_fall_through_to_legacy_consumer(self) -> None:
        from tllm.consumers.esamp.register import build

        with self.assertRaisesRegex(ValueError, "native @workload"):
            build(ESampConsumerConfig(implementation="native"))

    def test_supported_program_config_passes(self) -> None:
        validate_program_config(ESampConsumerConfig(implementation="program"))
        validate_program_config(
            ESampConsumerConfig(implementation="program", per_request_models=True)
        )

    def test_unsupported_combinations_fail_at_startup(self) -> None:
        cases = [
            {"per_request_models": True, "per_request_model_bank": True},
            {"model_bank_train_cudagraph": True},
            {"model_bank_initializer": SVDModelBankInitializerConfig()},
            {"trace_per_request_losses": True},
            {"enable_distiller_intervention": True, "distiller_beta": 0.5,
             "per_request_models": True},
            {"enable_distiller_intervention": True, "distiller_beta": 0.5,
             "distiller_sampler_backend": "pre_filter_dense"},
        ]
        for overrides in cases:
            with self.subTest(**overrides):
                with self.assertRaises(ValueError):
                    ESampProgramConsumer(ESampConsumerConfig(**overrides))

    def test_program_path_never_imports_engine(self) -> None:
        code = (
            "import sys\n"
            "import tllm.consumers.esamp.program\n"
            "import tllm.consumers.esamp.program_consumer\n"
            "assert 'tllm.consumers.esamp.engine' not in sys.modules, "
            "'program path must not import ESampTrainEngine'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class ReplicaTrainingTest(unittest.TestCase):
    def test_trains_over_varying_batch_shapes(self) -> None:
        torch.manual_seed(0)
        consumer = _consumer()
        for step in (1, 2, 3):
            rows = step + 1
            consumer.consume_flow(
                0,
                _bundle(
                    torch.randn(rows, _HIDDEN),
                    torch.randn(rows, _HIDDEN),
                    tuple(0 for _ in range(rows)),
                    step_id=step,
                ),
                None,
            )
        stats = consumer.read_and_reset_stats()
        self.assertEqual(stats.loss_count, 3)
        self.assertGreater(stats.loss_avg, 0.0)
        # Reading resets the counters.
        self.assertEqual(consumer.read_and_reset_stats().loss_count, 0)

    def test_trains_under_inference_mode(self) -> None:
        """vLLM executes consumers under inference_mode; training must still run."""
        torch.manual_seed(0)
        consumer = _consumer()
        with torch.inference_mode():
            for step in (1, 2):
                consumer.consume_flow(
                    0,
                    _bundle(
                        torch.randn(3, _HIDDEN),
                        torch.randn(3, _HIDDEN),
                        (0, 0, 0),
                        step_id=step,
                    ),
                    None,
                )
        self.assertEqual(consumer.read_and_reset_stats().loss_count, 2)

    def test_matches_handrolled_adamw_reference(self) -> None:
        hidden, rank, lr = _HIDDEN, 8, 1e-2
        consumer = _consumer()
        batches = [
            (torch.randn(3, hidden), torch.randn(3, hidden)),
            (torch.randn(5, hidden), torch.randn(5, hidden)),
        ]

        # The program model is constructed lazily at the first consume; seed
        # immediately before it so the init is reproducible, then replay the
        # same seed to build the reference model with identical parameters.
        torch.manual_seed(0)
        source0, target0 = batches[0]
        consumer.consume_flow(
            0, _bundle(source0, target0, (0,) * source0.shape[0], step_id=1), None
        )
        torch.manual_seed(0)
        reference = LowRankGatedResidualModel(
            hidden_size=hidden,
            rank=rank,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=lr, foreach=False
        )
        pred = reference(source0)
        loss = F.mse_loss(pred.float(), target0.float())
        reference_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        reference_optimizer.step()

        source1, target1 = batches[1]
        consumer.consume_flow(
            0, _bundle(source1, target1, (0,) * source1.shape[0], step_id=2), None
        )
        pred = reference(source1)
        loss = F.mse_loss(pred.float(), target1.float())
        reference_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        reference_optimizer.step()

        compiled = consumer._compiled
        assert compiled is not None
        slot = compiled.bank("distiller").slot_for(None)
        state = reference.state_dict()
        for name, param in slot.params.items():
            reference_param = state[name.removeprefix("model.")]
            self.assertTrue(
                torch.allclose(param, reference_param, rtol=1e-5, atol=1e-6),
                f"parameter {name} diverged from the AdamW reference",
            )

    def test_prediction_sink_receives_model_output(self) -> None:
        torch.manual_seed(0)
        consumer = _consumer()
        source = torch.randn(4, _HIDDEN)
        consumer.consume_flow(
            0, _bundle(source, torch.randn(4, _HIDDEN), (0, 0, 0, 0), step_id=1), None
        )
        latest = consumer._sink.latest
        assert latest is not None
        step_id, prediction, prompt_idxs = latest
        self.assertEqual(step_id, 1)
        self.assertEqual(tuple(prediction.shape), (4, _HIDDEN))
        self.assertEqual(prompt_idxs, (0, 0, 0, 0))

    def test_disabled_consumer_skips_bundles(self) -> None:
        consumer = _consumer(enable_esamp_training=False)
        self.assertEqual(consumer.flows(), ())
        consumer.consume_flow(
            0, _bundle(torch.randn(2, _HIDDEN), torch.randn(2, _HIDDEN), (0, 0), step_id=1), None
        )
        self.assertIsNone(consumer._compiled)
        self.assertEqual(consumer._disabled_skip_count, 1)


class RequestTrainingTest(unittest.TestCase):
    def test_duplicate_prompt_updates_once_per_batch(self) -> None:
        torch.manual_seed(0)
        consumer = _consumer(per_request_models=True)
        consumer.consume_flow(
            0,
            _bundle(torch.randn(4, _HIDDEN), torch.randn(4, _HIDDEN), (7, 7, 9, 7), step_id=1),
            None,
        )
        compiled = consumer._compiled
        assert compiled is not None
        self.assertEqual(compiled.update_count, 2)
        bank = compiled.bank("distiller")
        self.assertEqual(bank.slot_for(7).step, 1)
        self.assertEqual(bank.slot_for(9).step, 1)
        per = consumer.read_and_reset_per_request_stats()
        self.assertEqual(set(per), {7, 9})
        self.assertTrue(all(stats.loss_count == 1 for stats in per.values()))
        self.assertEqual(consumer.read_and_reset_per_request_stats(), {})

    def test_prompts_do_not_contaminate_each_other(self) -> None:
        torch.manual_seed(0)
        consumer = _consumer(per_request_models=True)
        source = torch.randn(2, _HIDDEN)
        target_a = torch.randn(2, _HIDDEN)
        target_b = -target_a
        consumer.consume_flow(
            0,
            _bundle(
                torch.cat([source, source], dim=0),
                torch.cat([target_a, target_b], dim=0),
                (1, 1, 2, 2),
                step_id=1,
            ),
            None,
        )
        compiled = consumer._compiled
        assert compiled is not None
        bank = compiled.bank("distiller")
        self.assertFalse(
            torch.allclose(
                bank.slot_for(1).params["model.a"],
                bank.slot_for(2).params["model.a"],
            )
        )


class SamplerProviderTest(unittest.TestCase):
    def _view(self, *, step_id: int, rows: int) -> SamplerStepView:
        class _Model:
            def __init__(self) -> None:
                self.lm_head = torch.nn.Linear(_HIDDEN, 32, bias=False)

        return SamplerStepView(
            engine_step_id=step_id,
            phase="decode",
            logits=torch.zeros(rows, 32),
            sampling_metadata=None,
            decode_count=rows,
            request_ids=tuple(f"req-{i}" for i in range(rows)),
            prompt_idxs=tuple(range(rows)),
            sample_idxs=tuple(0 for _ in range(rows)),
            prompt_idx_tensor=None,
            sample_idx_tensor=None,
            source_hidden=torch.randn(rows, _HIDDEN),
            device=torch.device("cpu"),
            model=_Model(),
            runner=None,
        )

    def _active_provider(self) -> tuple[object, ESampProgramConsumer]:
        consumer = _consumer(enable_distiller_intervention=True, distiller_beta=0.5)
        provider = consumer.sampler_modifier_provider()
        self.assertTrue(provider.is_active())
        return provider, consumer

    def test_inactive_without_intervention_or_beta(self) -> None:
        self.assertFalse(_consumer().sampler_modifier_provider().is_active())
        self.assertFalse(
            _consumer(enable_distiller_intervention=True, distiller_beta=0.0)
            .sampler_modifier_provider()
            .is_active()
        )

    def test_prepare_step_none_before_specialization(self) -> None:
        provider, _consumer_instance = self._active_provider()
        self.assertIsNone(provider.prepare_step(self._view(step_id=1, rows=2)))

    def test_prepare_step_predicts_current_source_rows(self) -> None:
        provider, consumer = self._active_provider()
        consumer.consume_flow(
            0, _bundle(torch.randn(2, _HIDDEN), torch.randn(2, _HIDDEN), (0, 0), step_id=1), None
        )
        view = self._view(step_id=2, rows=3)
        state = provider.prepare_step(view)
        assert state is not None
        self.assertEqual(tuple(state.affected_row_ids.tolist()), (0, 1, 2))
        self.assertEqual(tuple(state.pred_hidden.shape), (3, _HIDDEN))
        self.assertAlmostEqual(state.beta, 0.5)
        self.assertEqual(tuple(state.lm_head_weight.shape), (32, _HIDDEN))
        self.assertIsNone(state.lm_head_bias)
        self.assertEqual(provider.published_state_count, 1)

        # The prediction equals the current bank state applied to the view's
        # source hidden (on-demand prediction, not a stale flow output).
        compiled = consumer.compiled
        assert compiled is not None
        slot = compiled.bank("distiller").slot_for(None)
        model = LowRankGatedResidualModel(
            hidden_size=_HIDDEN,
            rank=8,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        model.load_state_dict(
            {name.removeprefix("model."): param for name, param in slot.params.items()}
        )
        with torch.no_grad():
            expected = model(view.source_hidden[:3])
        self.assertTrue(torch.allclose(state.pred_hidden, expected))

    def test_prepare_step_tracks_training_updates(self) -> None:
        provider, consumer = self._active_provider()
        view = self._view(step_id=1, rows=2)
        consumer.consume_flow(
            0,
            _bundle(view.source_hidden.clone(), torch.randn(2, _HIDDEN), (0, 0), step_id=1),
            None,
        )
        first = provider.prepare_step(view)
        consumer.consume_flow(
            0,
            _bundle(view.source_hidden.clone(), torch.randn(2, _HIDDEN), (0, 0), step_id=2),
            None,
        )
        second = provider.prepare_step(view)
        assert first is not None and second is not None
        self.assertFalse(torch.allclose(first.pred_hidden, second.pred_hidden))
        self.assertEqual(provider.published_state_count, 2)


if __name__ == "__main__":
    unittest.main()

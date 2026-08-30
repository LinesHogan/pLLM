#!/usr/bin/env python3
"""MRv1 raw-logit ordering contract for pre-filter sampler modifiers."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.backends.vllm import sampler_mrv1
from tllm.runtime.residual_runtime import SamplerPrecomputeState
from tllm.runtime.sampler_bridge.provider import SamplerInterventionStage
from tllm.runtime.sampler_bridge.types import CandidateModifierState


class _Processor:
    def __init__(self) -> None:
        self.input: torch.Tensor | None = None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        self.input = logits.clone()
        return logits


class _Sampler:
    def __init__(
        self,
        temperature: float,
        *,
        runtime: object,
        runner: object,
    ) -> None:
        self._tllm_runtime = runtime
        self._tllm_runner = runner
        self.logprobs_mode = "raw_logprobs"
        self.temperature = float(temperature)
        self.temperature_input: torch.Tensor | None = None
        self.sample_input: torch.Tensor | None = None

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def apply_temperature(
        self,
        logits: torch.Tensor,
        _temperature: torch.Tensor,
        _all_random: bool,
    ) -> torch.Tensor:
        self.temperature_input = logits.clone()
        return logits / self.temperature

    def topk_topp_sampler(
        self,
        logits: torch.Tensor,
        _generators: dict[int, torch.Generator],
        _top_k: None,
        _top_p: None,
    ) -> tuple[torch.Tensor, None]:
        self.sample_input = logits.clone()
        return logits.argmax(dim=-1), None


def _state(raw: torch.Tensor, reweighted: torch.Tensor) -> CandidateModifierState:
    rows = int(raw.shape[0])
    return CandidateModifierState(
        beta=1.0,
        backend="pre_filter_dense",
        affected_row_ids=torch.arange(rows, dtype=torch.long),
        pred_hidden=torch.ones((rows, 4)),
        lm_head_weight=torch.empty((0, 0)),
        lm_head_bias=None,
        precomputed_dense_logits=(2.0 * raw) - reweighted,
    )


class _Provider:
    def __init__(self, raw: torch.Tensor, reweighted: torch.Tensor) -> None:
        self._raw = raw
        self._reweighted = reweighted

    def is_active(self) -> bool:
        return True

    def intervention_stage(self) -> SamplerInterventionStage:
        return SamplerInterventionStage.PRE_FILTER

    def prepare_step(self, _view: object) -> CandidateModifierState:
        return _state(self._raw, self._reweighted)

    def after_sample(self, _view: object, _tokens: torch.Tensor) -> None:
        return None


class Mrv1SamplerPipelineOrderTest(unittest.TestCase):
    def test_dense_intervention_precedes_temperature_and_processors(self) -> None:
        raw = torch.tensor(
            ((1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0))
        )
        reweighted = torch.tensor(
            ((0.0, 0.0, 10.0, 0.0), (0.0, 12.0, 0.0, 0.0))
        )
        raw_snapshot = raw.clone()
        processor = _Processor()
        metadata = SimpleNamespace(
            max_num_logprobs=None,
            logprob_token_ids=(),
            all_random=True,
            all_greedy=False,
            temperature=torch.full((2,), 2.0),
            logitsprocs=SimpleNamespace(argmax_invariant=(processor,)),
            generators={},
            top_k=None,
            top_p=None,
        )
        provider = _Provider(raw, reweighted)
        runtime = SimpleNamespace(
            sampler_precompute=SamplerPrecomputeState(),
        )
        runner = SimpleNamespace(model=object())
        sampler = _Sampler(
            temperature=2.0,
            runtime=runtime,
            runner=runner,
        )

        def _build_view(**kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(logits=kwargs["logits"], runner=runner)

        with (
            mock.patch.object(sampler_mrv1, "_resolve_provider", return_value=provider),
            mock.patch.object(
                sampler_mrv1,
                "dispatch_sampler_pre_draw",
            ),
            mock.patch(
                "tllm.runtime.sampler_bridge.runtime_view.build_sampler_step_view",
                side_effect=_build_view,
            ),
        ):
            sampled, _processed = sampler_mrv1.wrapped_sampler_sample_v027(
                sampler=sampler,
                logits=raw,
                sampling_metadata=metadata,
                orig_sample=mock.Mock(),
            )

        assert sampler.temperature_input is not None
        assert processor.input is not None
        assert sampler.sample_input is not None
        torch.testing.assert_close(sampler.temperature_input, reweighted)
        torch.testing.assert_close(processor.input, reweighted / 2.0)
        torch.testing.assert_close(sampler.sample_input, reweighted / 2.0)
        torch.testing.assert_close(sampled, torch.tensor((2, 1)))
        torch.testing.assert_close(raw, raw_snapshot)
        self.assertEqual(runtime.sampler_precompute.intervention_state_count, 1)


if __name__ == "__main__":
    unittest.main()

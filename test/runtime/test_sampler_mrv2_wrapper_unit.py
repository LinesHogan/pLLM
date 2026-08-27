#!/usr/bin/env python3
"""Unit tests for the MRv2 sampler wrapper ordering contracts.

Two regressions are pinned here:

* the pre-draw fence (``dispatch_sampler_pre_draw``) must run *before*
  ``provider.prepare_step`` consumes Region outputs (side-stream join
  precedes consumption);
* ``pre_filter_dense`` providers (OverRIDE) intervene on the raw logits
  *before* ``apply_sampling_params`` — softmax((p - λh)/T), not
  softmax(p/T - λh) — and the runner's raw logits tensor is never mutated.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import unittest
from unittest import mock

import torch

from tllm.backends.vllm import sampler_mrv2
from tllm.runtime.residual_runtime import SamplerPrecomputeState
from tllm.runtime.sampler_bridge.types import CandidateModifierState


def _runtime(**kwargs: Any) -> Any:
    kwargs.setdefault("decode_count", 2)
    kwargs.setdefault("source_resolved_path", "layers.0")
    kwargs.setdefault("tap_decode_hidden", {"layers.0": torch.ones((2, 4))})
    kwargs.setdefault("decode_prompt_idx_tensor", torch.tensor([7, 8]))
    kwargs.setdefault("decode_sample_idx_tensor", torch.tensor([0, 1]))
    kwargs.setdefault("decode_prompt_idxs", [7, 8])
    kwargs.setdefault("decode_sample_idxs", [0, 1])
    kwargs.setdefault("decode_request_ids", ["req-a", "req-b"])
    kwargs.setdefault("event_step_id", 11)
    kwargs.setdefault("sampler_precompute", SamplerPrecomputeState())
    return SimpleNamespace(**kwargs)


class _FakeSamplerStates:
    def __init__(self, *, temperature: torch.Tensor) -> None:
        self.temperature = SimpleNamespace(gpu=temperature)
        self.seeds = SimpleNamespace(gpu=torch.zeros(2, dtype=torch.long))

    def get_top_k_top_p(self, expanded_idx_mapping: Any, idx_mapping_np: Any) -> tuple[None, None]:
        return None, None


class _FakeSampler:
    """Minimal MRv2 sampler double: temperature divide + optional min-p mask."""

    def __init__(self, *, temperature: float = 1.0, min_p: float = 0.0) -> None:
        self.logprobs_mode = "raw_logprobs"
        self.use_fp64_gumbel = False
        self.sampling_states = _FakeSamplerStates(
            temperature=torch.full((2,), float(temperature), dtype=torch.float32)
        )
        self._min_p = float(min_p)
        self.captured: dict[str, torch.Tensor] = {}

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: Any,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        *,
        skip_top_k_top_p: bool,
    ) -> torch.Tensor:
        self.captured["input"] = logits.clone()
        out = logits / self.sampling_states.temperature.gpu.unsqueeze(1)
        if self._min_p > 0.0:
            probs = torch.softmax(out, dim=-1)
            threshold = probs.max(dim=-1, keepdim=True).values * self._min_p
            out = out.masked_fill(probs < threshold, float("-inf"))
        return out


def _pre_filter_state(*, raw: torch.Tensor, reweighted: torch.Tensor) -> CandidateModifierState:
    rows = int(raw.shape[0])
    return CandidateModifierState(
        beta=1.0,
        backend="pre_filter_dense",
        affected_row_ids=torch.arange(rows, dtype=torch.long),
        pred_hidden=torch.ones((rows, 4)),
        lm_head_weight=torch.empty((0, 0)),
        lm_head_bias=None,
        # beta=1.0 substitution: (1+beta)*raw - beta*dense == reweighted.
        precomputed_dense_logits=(2.0 * raw) - reweighted,
    )


class Mrv2SamplerWrapperOrderTest(unittest.TestCase):
    def _wrapper(self, *, runtime: Any, provider: Any, sampler: Any) -> Any:
        runtime.consumer = SimpleNamespace(sampler_modifier_provider=lambda: provider)
        runner = SimpleNamespace(model=object())
        orig_sample = mock.Mock(return_value=("orig-sampled", "orig-processed"))
        return (
            sampler_mrv2.make_mrv2_sample_wrapper(
                sampler=sampler,
                runtime=runtime,
                runner=runner,
                orig_sample=orig_sample,
            ),
            orig_sample,
        )

    def _call(self, wrapped: Any, logits: torch.Tensor) -> Any:
        return wrapped(
            logits,
            torch.arange(2, dtype=torch.long),  # expanded_idx_mapping
            None,  # idx_mapping_np
            torch.arange(2, dtype=torch.long),  # pos
            torch.zeros(2, dtype=torch.long),  # input_ids
            torch.arange(2, dtype=torch.long),  # expanded_local_pos
        )

    def test_pre_draw_dispatch_precedes_prepare_step(self) -> None:
        order: list[str] = []
        provider = SimpleNamespace(
            is_active=lambda: True,
            prepare_step=lambda view: order.append("provider") or None,
        )
        runtime = _runtime()
        sampler = _FakeSampler()
        wrapped, orig_sample = self._wrapper(runtime=runtime, provider=provider, sampler=sampler)

        with mock.patch.object(
            sampler_mrv2,
            "dispatch_sampler_pre_draw",
            side_effect=lambda **_: order.append("dispatch"),
        ):
            sampled, _ = wrapped(*self._args())

        self.assertEqual(order, ["dispatch", "provider"])
        orig_sample.assert_called_once()
        self.assertEqual(sampled, "orig-sampled")

    def _args(self) -> tuple[Any, ...]:
        return (
            torch.ones((2, 8), dtype=torch.float32),
            torch.arange(2, dtype=torch.long),
            None,
            torch.arange(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.arange(2, dtype=torch.long),
        )

    def test_inactive_provider_neither_dispatches_nor_prepares(self) -> None:
        order: list[str] = []
        provider = SimpleNamespace(
            is_active=lambda: False,
            prepare_step=lambda view: order.append("provider") or None,
        )
        runtime = _runtime()
        sampler = _FakeSampler()
        wrapped, orig_sample = self._wrapper(runtime=runtime, provider=provider, sampler=sampler)

        with mock.patch.object(
            sampler_mrv2,
            "dispatch_sampler_pre_draw",
            side_effect=lambda **_: order.append("dispatch"),
        ):
            self._call(wrapped, torch.ones((2, 8), dtype=torch.float32))

        self.assertEqual(order, [])
        orig_sample.assert_called_once()


class Mrv2SamplerPipelineReorderTest(unittest.TestCase):
    def _run(self, *, raw: torch.Tensor, reweighted: torch.Tensor, temperature: float, min_p: float) -> dict[str, torch.Tensor]:
        provider = SimpleNamespace(
            is_active=lambda: True,
            prepare_step=lambda view: _pre_filter_state(raw=raw, reweighted=reweighted),
            after_sample=lambda view, sampled: None,
        )
        runtime = _runtime()
        runtime.consumer = SimpleNamespace(sampler_modifier_provider=lambda: provider)
        sampler = _FakeSampler(temperature=temperature, min_p=min_p)
        runner = SimpleNamespace(model=object())
        wrapped = sampler_mrv2.make_mrv2_sample_wrapper(
            sampler=sampler,
            runtime=runtime,
            runner=runner,
            orig_sample=mock.Mock(),
        )
        captured: dict[str, torch.Tensor] = {}

        def _fake_gumbel(filtered: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            captured["filtered"] = filtered.clone()
            return filtered.argmax(dim=-1)

        with (
            mock.patch.object(sampler_mrv2, "dispatch_sampler_pre_draw", lambda **_: None),
            mock.patch(
                "vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p",
                side_effect=lambda logits, top_k, top_p: logits,
            ),
            mock.patch(
                "vllm.v1.worker.gpu.sample.gumbel.gumbel_sample",
                side_effect=_fake_gumbel,
            ),
        ):
            wrapped(
                raw,
                torch.arange(2, dtype=torch.long),
                None,
                torch.arange(2, dtype=torch.long),
                torch.zeros(2, dtype=torch.long),
                torch.arange(2, dtype=torch.long),
            )
        captured["pipeline_input"] = sampler.captured["input"]
        return captured

    def test_intervention_precedes_temperature(self) -> None:
        raw = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        reweighted = torch.tensor([[0.0, 0.0, 10.0, 0.0], [0.0, 12.0, 0.0, 0.0]])
        raw_snapshot = raw.clone()

        captured = self._run(raw=raw, reweighted=reweighted, temperature=2.0, min_p=0.0)

        # apply_sampling_params receives the intervened logits, not the raw
        # ones: softmax((p - λh)/T), not softmax(p/T - λh).
        self.assertTrue(torch.allclose(captured["pipeline_input"], reweighted))
        self.assertTrue(torch.allclose(captured["filtered"], reweighted / 2.0))
        # The runner's raw logits tensor is never mutated in place.
        self.assertTrue(torch.equal(raw, raw_snapshot))

    def test_min_p_masks_the_intervened_distribution(self) -> None:
        raw = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        reweighted = torch.tensor([[5.0, 5.5, 6.0, 0.0], [0.0, 12.0, 0.0, 0.0]])
        temperature = 0.6
        min_p = 0.05

        captured = self._run(raw=raw, reweighted=reweighted, temperature=temperature, min_p=min_p)

        scaled = reweighted / temperature
        probs = torch.softmax(scaled, dim=-1)
        threshold = probs.max(dim=-1, keepdim=True).values * min_p
        expected = scaled.masked_fill(probs < threshold, float("-inf"))
        self.assertTrue(torch.allclose(captured["pipeline_input"], reweighted))
        self.assertTrue(torch.equal(captured["filtered"], expected))


if __name__ == "__main__":
    unittest.main()

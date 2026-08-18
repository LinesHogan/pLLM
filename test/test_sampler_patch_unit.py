#!/usr/bin/env python3
"""Unit tests for the sampler patch wrapper."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.runtime.residual_runtime import SamplerPrecomputeState
from tllm.runtime.vllm_patch import sampler_patch


class _FakeTopKTopPSampler:
    def __call__(self, logits, generators, k, p):
        _ = (generators, k, p)
        return logits.argmax(dim=-1)

    def forward_native(self, logits, generators, k, p):
        return self(logits, generators, k, p)


class _FakeSampler:
    def __init__(self) -> None:
        self.topk_topp_sampler = _FakeTopKTopPSampler()

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def apply_temperature(self, logits: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        return logits.div(temp.unsqueeze(dim=1))


class SamplerPatchUnitTest(unittest.TestCase):
    def tearDown(self) -> None:
        sampler_patch._ORIG_VLLM_SAMPLER_SAMPLE = None

    def test_wrapped_sampler_sample_uses_bridge_when_provider_is_active(self) -> None:
        sampler = _FakeSampler()
        runtime = SimpleNamespace(
            consumer=SimpleNamespace(
                sampler_modifier_provider=lambda: SimpleNamespace(is_active=lambda: True)
            ),
            event_step_id=1,
            decode_count=1,
            decode_request_ids=["reqA"],
            decode_prompt_idxs=[3],
            decode_sample_idxs=[0],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.ones((1, 2), dtype=torch.float32)},
            sampler_precompute=SamplerPrecomputeState(),
        )
        runner = SimpleNamespace(sampler=sampler, model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)))
        sampler_patch.bind_runner_sampler(runtime=runtime, runner=runner)
        sampler._tllm_runtime = runtime
        sampler._tllm_runner = runner
        sampling_metadata = SimpleNamespace(
            all_random=True,
            all_greedy=False,
            temperature=torch.tensor([1.0], dtype=torch.float32),
            logitsprocs=SimpleNamespace(argmax_invariant=[]),
            top_k=torch.tensor([1], dtype=torch.int64),
            top_p=None,
            generators={},
        )

        with mock.patch.object(
            sampler_patch,
            "sample_with_optional_modifier",
            return_value=torch.tensor([2], dtype=torch.long),
        ) as p_bridge:
            sampled = sampler_patch.wrapped_sampler_sample(
                sampler=sampler,
                logits=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
                sampling_metadata=sampling_metadata,
            )

        self.assertTrue(torch.equal(sampled, torch.tensor([2], dtype=torch.long)))
        p_bridge.assert_called_once()

    def test_wrapped_sampler_sample_falls_back_to_original_when_provider_is_inactive(self) -> None:
        sampler = _FakeSampler()
        runtime = SimpleNamespace(
            consumer=SimpleNamespace(
                sampler_modifier_provider=lambda: SimpleNamespace(is_active=lambda: False)
            ),
        )
        runner = SimpleNamespace(model=object())
        sampler_patch.bind_runner_sampler(runtime=runtime, runner=runner)
        sampler._tllm_runtime = runtime
        sampler._tllm_runner = runner
        sampling_metadata = SimpleNamespace(
            all_random=True,
            all_greedy=False,
            temperature=torch.tensor([1.0], dtype=torch.float32),
            logitsprocs=SimpleNamespace(argmax_invariant=[]),
            top_k=torch.tensor([1], dtype=torch.int64),
            top_p=None,
            generators={},
        )
        sampler_patch._ORIG_VLLM_SAMPLER_SAMPLE = mock.Mock(return_value=torch.tensor([1], dtype=torch.long))

        sampled = sampler_patch.wrapped_sampler_sample(
            sampler=sampler,
            logits=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            sampling_metadata=sampling_metadata,
        )

        self.assertTrue(torch.equal(sampled, torch.tensor([1], dtype=torch.long)))
        sampler_patch._ORIG_VLLM_SAMPLER_SAMPLE.assert_called_once()


if __name__ == "__main__":
    unittest.main()

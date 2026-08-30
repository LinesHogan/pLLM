from __future__ import annotations

import unittest

import torch

from tllm.runtime.sampler_bridge.context import SamplerStepContext
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.workload.runtime_values import SKIPPED


def _view(step_id: int) -> SamplerStepView:
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return SamplerStepView(
        engine_step_id=step_id,
        phase="decode",
        logits=torch.arange(10, dtype=torch.float32).reshape(2, 5),
        sampling_metadata=object(),
        decode_count=2,
        request_ids=("request-0", "request-1"),
        prompt_idxs=(4, 5),
        sample_idxs=(6, 7),
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=hidden,
        device=torch.device("cpu"),
        model=torch.nn.Module(),
        runner=object(),
    )


class SamplerStepContextUnitTest(unittest.TestCase):
    def test_ports_only_expose_the_exact_prepared_step(self) -> None:
        context = SamplerStepContext()
        view = _view(7)

        self.assertIs(context.request_ids(7), SKIPPED)
        self.assertIs(context.sample_indices(7), SKIPPED)
        context.prepare_identity(7, view.request_ids, view.sample_idxs)
        context.prepare(view)

        self.assertIs(context.logits(6), SKIPPED)
        self.assertIs(context.tokens(6), SKIPPED)
        self.assertIs(context.request_ids(6), SKIPPED)
        self.assertIs(context.sample_indices(6), SKIPPED)
        self.assertIs(context.logits(7), view.logits)
        self.assertIs(context.tokens(7), SKIPPED)
        self.assertEqual(context.request_ids(7), view.request_ids)
        self.assertEqual(context.sample_indices(7), view.sample_idxs)

    def test_sampled_tokens_do_not_leak_into_the_next_step(self) -> None:
        context = SamplerStepContext()
        view = _view(7)
        tokens = torch.tensor((8, 9))

        context.sampled(view, tokens)
        sampled = context.tokens(7)
        self.assertIsInstance(sampled, torch.Tensor)
        assert isinstance(sampled, torch.Tensor)
        torch.testing.assert_close(sampled, tokens)

        next_view = _view(8)
        context.prepare_identity(
            8, next_view.request_ids, next_view.sample_idxs
        )
        context.prepare(next_view)

        self.assertIs(context.logits(7), SKIPPED)
        self.assertIs(context.tokens(7), SKIPPED)
        self.assertIs(context.request_ids(7), SKIPPED)
        self.assertIs(context.sample_indices(7), SKIPPED)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for the OverRIDE sampler-provider runtime adapter."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from tllm.runtime.sampler_bridge.dense_backend import modify_rows_dense
from tllm.runtime.sampler_bridge.types import SamplerStepView


class OverRIDESamplerProviderUnitTest(unittest.TestCase):
    def _view(
        self,
        *,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        request_ids: tuple[str, ...],
    ) -> SamplerStepView:
        row_count = int(logits.shape[0])
        return SamplerStepView(
            engine_step_id=7,
            phase="decode",
            logits=logits,
            sampling_metadata=SimpleNamespace(),
            decode_count=row_count,
            request_ids=request_ids,
            prompt_idxs=tuple(range(row_count)),
            sample_idxs=tuple(range(row_count)),
            prompt_idx_tensor=None,
            sample_idx_tensor=None,
            source_hidden=hidden,
            device=logits.device,
            model=SimpleNamespace(),
            runner=SimpleNamespace(),
        )

    def test_prepare_step_reweights_logits_from_view_and_post_sample_updates_heads(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=3,
                lambd=0.5,
                learning_rate=0.1,
            )
        )
        with torch.no_grad():
            previous, current = consumer.engine.heads
            previous.down.weight.copy_(torch.eye(2))
            previous.up.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
            current.down.weight.copy_(torch.eye(2))
            current.up.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.25], [0.5, 0.5]]))
        logits = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
        hidden = torch.tensor([[2.0, 3.0], [1.0, 4.0]])
        view = self._view(logits=logits, hidden=hidden, request_ids=("1_parent", "0_parent"))

        provider = consumer.sampler_modifier_provider()
        state = provider.prepare_step(view)

        self.assertIsNotNone(state)
        assert state is not None
        reweighted = modify_rows_dense(logits=logits, state=state, row_ids=state.affected_row_ids)
        expected_row0 = logits[0] - 0.5 * torch.tensor([2.0, 6.0, 5.0])
        self.assertTrue(torch.allclose(reweighted[0], expected_row0))
        self.assertTrue(torch.allclose(reweighted[1], logits[1]))

        before = consumer.engine.heads[1].up.weight.detach().clone()
        provider.after_sample(view, torch.tensor([2, 1]))

        stats = consumer.read_stats()
        self.assertFalse(torch.allclose(before, consumer.engine.heads[1].up.weight.detach()))
        self.assertGreater(stats.loss_count, 0)
        self.assertGreater(stats.update_count, 0)

    def test_post_sample_update_uses_prepare_step_row_request_mapping_when_rows_are_reordered(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=3,
                learning_rate=0.1,
            )
        )
        provider = consumer.sampler_modifier_provider()
        logits = torch.randn(3, 3)
        hidden = torch.randn(3, 2)
        view = self._view(logits=logits, hidden=hidden, request_ids=("0_req-a", "1_req-b", "0_req-c"))

        provider.prepare_step(view)
        provider.after_sample(view, torch.tensor([0, 1, 2]))
        first_updates = consumer.read_stats().update_count

        reordered_view = self._view(
            logits=logits.index_select(0, torch.tensor([2, 0, 1])),
            hidden=hidden.index_select(0, torch.tensor([2, 0, 1])),
            request_ids=("0_req-c", "0_req-a", "1_req-b"),
        )
        provider.prepare_step(reordered_view)
        provider.after_sample(reordered_view, torch.tensor([2, 0, 1]))

        self.assertEqual(first_updates, 1)
        self.assertEqual(consumer.read_stats().update_count, 2)

    def test_after_sample_rejects_cross_step_update(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=3,
                learning_rate=0.1,
            )
        )
        provider = consumer.sampler_modifier_provider()
        view = self._view(
            logits=torch.randn(2, 3),
            hidden=torch.randn(2, 2),
            request_ids=("0_req-a", "1_req-b"),
        )
        provider.prepare_step(view)
        next_step_view = self._view(
            logits=torch.randn(2, 3),
            hidden=torch.randn(2, 2),
            request_ids=("0_req-a", "1_req-b"),
        )
        next_step_view = SamplerStepView(
            engine_step_id=view.engine_step_id + 1,
            phase=next_step_view.phase,
            logits=next_step_view.logits,
            sampling_metadata=next_step_view.sampling_metadata,
            decode_count=next_step_view.decode_count,
            request_ids=next_step_view.request_ids,
            prompt_idxs=next_step_view.prompt_idxs,
            sample_idxs=next_step_view.sample_idxs,
            prompt_idx_tensor=next_step_view.prompt_idx_tensor,
            sample_idx_tensor=next_step_view.sample_idx_tensor,
            source_hidden=next_step_view.source_hidden,
            device=next_step_view.device,
            model=next_step_view.model,
            runner=next_step_view.runner,
        )

        with self.assertRaisesRegex(RuntimeError, "prepared sampler step"):
            provider.after_sample(next_step_view, torch.tensor([0, 1]))

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_after_sample_rejects_sampled_token_count_mismatch(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=3,
                learning_rate=0.1,
            )
        )
        provider = consumer.sampler_modifier_provider()
        view = self._view(
            logits=torch.randn(2, 3),
            hidden=torch.randn(2, 2),
            request_ids=("0_req-a", "1_req-b"),
        )
        provider.prepare_step(view)

        with self.assertRaisesRegex(RuntimeError, "one sampled token per prepared decode row"):
            provider.after_sample(view, torch.tensor([0]))

        self.assertEqual(consumer.read_stats().update_count, 0)


if __name__ == "__main__":
    unittest.main()

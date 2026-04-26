#!/usr/bin/env python3
"""Unit tests for the OverRIDE-inspired devlist consumer."""

from __future__ import annotations

import unittest

import torch


class OverRIDEConsumerUnitTest(unittest.TestCase):
    def test_package_exports_override_consumer(self) -> None:
        from tllm.consumers.devlist import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig())

        self.assertEqual(consumer.consumer_id, "override")

    def test_sampler_only_consumer_declares_no_flows(self) -> None:
        from tllm.consumers.devlist import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig())

        self.assertEqual(tuple(consumer.flows()), ())

    def test_disabled_config_returns_original_logits(self) -> None:
        from tllm.consumers.devlist import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig(enabled=False, hidden_size=3, vocab_size=5, rank=2))
        hidden = torch.randn(4, 3)
        logits = torch.randn(4, 5)

        result = consumer.reweight_logits(hidden, logits, iteration=1)

        self.assertIs(result.q_logits_next, None)
        self.assertTrue(torch.equal(result.logits, logits))

    def test_enabled_reweighting_uses_previous_iteration_residual(self) -> None:
        from tllm.consumers.devlist import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=3,
                lambd=0.5,
            )
        )
        with torch.no_grad():
            previous = consumer.engine.heads[0]
            current = consumer.engine.heads[1]
            previous.down.weight.copy_(torch.eye(2))
            previous.up.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
            current.down.weight.copy_(torch.eye(2))
            current.up.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.25], [0.5, 0.5]]))
        hidden = torch.tensor([[2.0, 3.0]])
        logits = torch.tensor([[10.0, 20.0, 30.0]])

        result = consumer.reweight_logits(hidden, logits, iteration=2)

        previous_residual = torch.tensor([[2.0, 6.0, 5.0]])
        current_residual = torch.tensor([[1.0, 0.75, 2.5]])
        self.assertTrue(torch.allclose(result.logits, logits - 0.5 * previous_residual))
        self.assertIsNotNone(result.q_logits_next)
        assert result.q_logits_next is not None
        self.assertTrue(torch.allclose(result.q_logits_next, logits + current_residual))

    def test_sampled_token_update_changes_head_and_records_loss(self) -> None:
        from tllm.consumers.devlist import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=2,
                learning_rate=0.1,
            )
        )
        hidden = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
        logits = torch.randn(2, 3)
        result = consumer.reweight_logits(hidden, logits, iteration=1)
        assert result.q_logits_next is not None
        before = consumer.engine.heads[0].up.weight.detach().clone()

        consumer.update_from_sampled_tokens(result.q_logits_next, torch.tensor([0, 2]))
        after = consumer.engine.heads[0].up.weight.detach()
        stats = consumer.read_stats()

        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(stats.loss_count, 1)
        self.assertEqual(stats.update_count, 1)

    def test_update_accepts_hidden_and_logits_from_inference_mode(self) -> None:
        from tllm.consumers.devlist import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=2,
                learning_rate=0.1,
            )
        )
        with torch.inference_mode():
            hidden = torch.randn(2, 2)
            logits = torch.randn(2, 3)

        result = consumer.reweight_logits(hidden, logits, iteration=1)
        consumer.update_from_sampled_tokens(result.q_logits_next, torch.tensor([0, 2]))

        self.assertEqual(consumer.read_stats().update_count, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_configured_heads_move_to_hidden_device(self) -> None:
        from tllm.consumers.devlist import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=2,
            )
        )
        hidden = torch.randn(2, 2, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(2, 3, device="cuda", dtype=torch.bfloat16)

        result = consumer.reweight_logits(hidden, logits, iteration=1)

        self.assertEqual(consumer.engine.heads[0].down.weight.device.type, "cuda")
        assert result.q_logits_next is not None
        self.assertEqual(result.q_logits_next.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()

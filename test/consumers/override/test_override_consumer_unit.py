#!/usr/bin/env python3
"""Unit tests for the OverRIDE-inspired devlist consumer."""

from __future__ import annotations

import unittest

import torch


class OverRIDEConsumerUnitTest(unittest.TestCase):
    def test_package_exports_override_consumer(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig())

        self.assertEqual(consumer.consumer_id, "override")

    def test_sampler_only_consumer_declares_no_flows(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig())

        self.assertEqual(tuple(consumer.flows()), ())

    def test_disabled_config_returns_original_logits(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig(enabled=False, hidden_size=3, vocab_size=5, rank=2))
        hidden = torch.randn(4, 3)
        logits = torch.randn(4, 5)

        result = consumer.reweight_logits(hidden, logits, iteration=1)

        self.assertIs(result.q_logits_next, None)
        self.assertTrue(torch.equal(result.logits, logits))

    def test_enabled_reweighting_uses_previous_iteration_residual(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

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

    def test_runtime_bridge_mixed_iterations_use_row_specific_heads(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=4,
                lambd=0.5,
            )
        )
        with torch.no_grad():
            head0, head1, head2 = consumer.engine.heads
            for head in (head0, head1, head2):
                head.down.weight.copy_(torch.eye(2))
            head0.up.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
            head1.up.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.25], [0.5, 0.5]]))
            head2.up.weight.copy_(torch.tensor([[0.25, 0.0], [0.0, 0.5], [1.0, -1.0]]))
        hidden = torch.tensor([[2.0, 3.0], [1.0, 4.0], [3.0, -2.0]])
        logits = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])

        result = consumer.reweight_logits_batch(hidden, logits, request_ids=("0_a", "1_b", "2_c"))

        expected = torch.stack(
            [
                logits[0],
                logits[1] - 0.5 * torch.tensor([1.0, 8.0, 5.0]),
                logits[2] - 0.5 * torch.tensor([1.5, -0.5, 0.5]),
            ]
        )
        self.assertTrue(torch.allclose(result.logits, expected))
        self.assertEqual(tuple(tuple(idx.tolist()) for idx in result.q_indices), ((0,), (1,), (2,)))

    def test_runtime_bridge_parses_request_ids_like_upstream_and_accepts_explicit_iterations(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig(hidden_size=2, vocab_size=3, rank=2, num_iterations=3))
        hidden = torch.randn(4, 2)
        logits = torch.randn(4, 3)

        self.assertEqual(consumer.iterations_from_request_ids(("0_parent", "2_parent", "abc_parent", "plain")), (1, 3, 1, 1))

        from_request_ids = consumer.reweight_logits_batch(
            hidden,
            logits,
            request_ids=("0_parent", "1_parent", "abc_parent", "plain"),
        )
        from_explicit_iterations = consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2, 1, 1))

        self.assertTrue(torch.equal(from_request_ids.logits, from_explicit_iterations.logits))
        self.assertEqual(tuple(tuple(idx.tolist()) for idx in from_request_ids.q_indices), ((0, 2, 3), (1,)))

    def test_runtime_bridge_updates_cached_q_logits_after_sampling_and_records_stats(self) -> None:
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
        hidden = torch.tensor([[1.0, -1.0], [0.5, 2.0], [2.0, 1.0]])
        logits = torch.randn(3, 3)

        consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2, 3))
        before0 = consumer.engine.heads[0].up.weight.detach().clone()
        before1 = consumer.engine.heads[1].up.weight.detach().clone()

        consumer.update_from_sampled_tokens_batch(torch.tensor([0, 1, 2]))

        stats = consumer.read_stats()
        self.assertFalse(torch.allclose(before0, consumer.engine.heads[0].up.weight.detach()))
        self.assertFalse(torch.allclose(before1, consumer.engine.heads[1].up.weight.detach()))
        self.assertGreater(stats.loss_count, 0)
        self.assertGreater(stats.update_count, 0)
        self.assertGreater(stats.last_loss, 0.0)

    def test_runtime_bridge_normalizes_tensor_row_ids_for_cached_update(self) -> None:
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
        hidden = torch.tensor([[1.0, -1.0], [0.5, 2.0], [2.0, 1.0]])
        logits = torch.randn(3, 3)

        consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2, 3), row_ids=torch.tensor([10, 11, 12]))
        before0 = consumer.engine.heads[0].up.weight.detach().clone()

        consumer.update_from_sampled_tokens_batch(torch.tensor([0, 1, 2]), row_ids=torch.tensor([10, 11, 12]))

        self.assertFalse(torch.allclose(before0, consumer.engine.heads[0].up.weight.detach()))
        self.assertGreater(consumer.read_stats().update_count, 0)

    def test_runtime_bridge_rejects_non_integral_row_ids(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig(hidden_size=2, vocab_size=3, rank=2, num_iterations=3))
        hidden = torch.randn(3, 2)
        logits = torch.randn(3, 3)

        with self.assertRaisesRegex(RuntimeError, "row ids must be integers"):
            consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2, 3), row_ids=("10", "11", "12"))

        with self.assertRaisesRegex(RuntimeError, "row id tensors must use an integer dtype"):
            consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2, 3), row_ids=torch.tensor([10.0, 11.0, 12.0]))

    def test_runtime_bridge_rejects_non_canonical_python_row_id_containers(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig(hidden_size=2, vocab_size=3, rank=2, num_iterations=3))
        hidden = torch.randn(3, 2)
        logits = torch.randn(3, 3)

        for row_ids in ([10, 11, 12], range(10, 13)):
            with self.subTest(row_ids=type(row_ids).__name__):
                with self.assertRaisesRegex(RuntimeError, "row ids must be a tuple of integers"):
                    consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2, 3), row_ids=row_ids)

    def test_runtime_bridge_rejects_duplicate_tuple_row_ids(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig(hidden_size=2, vocab_size=3, rank=2, num_iterations=3))
        hidden = torch.randn(2, 2)
        logits = torch.randn(2, 3)

        with self.assertRaisesRegex(RuntimeError, "row ids must be unique"):
            consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2), row_ids=(7, 7))

    def test_runtime_bridge_rejects_duplicate_tensor_row_ids(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(OverRIDEConsumerConfig(hidden_size=2, vocab_size=3, rank=2, num_iterations=3))
        hidden = torch.randn(2, 2)
        logits = torch.randn(2, 3)

        with self.assertRaisesRegex(RuntimeError, "row ids must be unique"):
            consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2), row_ids=torch.tensor([7, 7]))

    def test_sampled_token_update_changes_head_and_records_loss(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

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
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

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
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

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

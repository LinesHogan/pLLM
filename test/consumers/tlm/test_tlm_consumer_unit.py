#!/usr/bin/env python3
"""Unit tests for the Program-backed TLM consumer."""

from __future__ import annotations

import unittest

import torch


class TLMConsumerUnitTest(unittest.TestCase):
    def test_package_exports_tlm_consumer(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig())

        self.assertEqual(consumer.consumer_id, "tlm")

    def test_online_logits_update_matches_reference_weighted_loss(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.consumers._common.alignment import (
            sentence_cross_entropy,
            weighted_one_hot_kl_loss,
        )

        hidden = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 4.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 1])
        config = TLMConsumerConfig(rank=2, learning_rate=0.0, threshold=1.0, lamb=0.5)
        consumer = TLMConsumer(config)

        consumer.update_online(hidden, logits=logits, labels=labels)

        stats = consumer.read_stats()
        # The adapter's up-projection is zero-initialized, so adapted logits
        # equal the base logits at step 0 and the masked-mean loss equals the
        # reference selected-row weighted loss (exactly one row clears the
        # threshold here).
        expected = weighted_one_hot_kl_loss(
            logits,
            labels,
            sentence_ce=sentence_cross_entropy(logits, labels),
            threshold=1.0,
            lamb=0.5,
        )
        self.assertAlmostEqual(stats.last_loss, float(expected), places=5)
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.loss_count, 1)

    def test_training_actually_updates_bank_params(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        torch.manual_seed(0)
        consumer = TLMConsumer(TLMConsumerConfig(rank=2, learning_rate=0.2, threshold=0.1))
        hidden = torch.randn(4, 4)
        logits = torch.randn(4, 5)
        labels = torch.tensor([0, 1, 2, 3])
        consumer.update_online(hidden, logits=logits, labels=labels)
        compiled = consumer.compiled
        assert compiled is not None
        slot = compiled.bank("adapters").slot_for(None)
        up = slot.params["adapter.up.weight"]
        self.assertGreater(float(up.abs().max()), 0.0)

    def test_closed_loop_online_predicts_before_training(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        calls: list[str] = []
        consumer = TLMConsumer(TLMConsumerConfig(mode="online", rank=2, learning_rate=0.2, threshold=1.0))
        hidden = torch.eye(3, 4)
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 4.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 1])

        def _predict() -> str:
            calls.append("predict")
            consumer.predict(hidden, logits)
            return "prediction"

        result = consumer.run_ordered_step(
            predict=_predict,
            hidden=hidden,
            logits=logits,
            labels=labels,
        )

        self.assertEqual(result, "prediction")
        self.assertEqual(calls, ["predict"])
        self.assertEqual(consumer.read_stats().update_count, 1)

    def test_offline_buffers_then_flushes(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(mode="offline", rank=2, threshold=0.1))
        hidden = torch.eye(3, 4)
        logits = torch.randn(3, 5)
        labels = torch.tensor([0, 1, 2])

        consumer.update_offline(hidden, logits=logits, labels=labels)
        consumer.update_offline(hidden, logits=logits, labels=labels)
        self.assertEqual(consumer.read_stats().buffered_count, 6)
        self.assertEqual(consumer.read_stats().update_count, 0)

        consumer.flush_offline()
        stats = consumer.read_stats()
        self.assertEqual(stats.buffered_count, 0)
        self.assertEqual(stats.update_count, 1)


if __name__ == "__main__":
    unittest.main()

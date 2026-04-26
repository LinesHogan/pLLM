#!/usr/bin/env python3
"""Unit tests for the TLM-inspired devlist consumer."""

from __future__ import annotations

import sys
import unittest

import torch


class TLMConsumerUnitTest(unittest.TestCase):
    def test_package_exports_tlm_consumer(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig())

        self.assertEqual(consumer.consumer_id, "tlm")

    def test_config_rejects_unknown_mode(self) -> None:
        from tllm.consumers.tlm import TLMConsumerConfig

        with self.assertRaisesRegex(ValueError, "mode"):
            TLMConsumerConfig(mode="streaming")

    def test_select_samples_keeps_raw_ce_above_threshold(self) -> None:
        from tllm.consumers.tlm import select_ttl_samples

        sentence_ce = torch.tensor([0.2, 0.8, 0.4])

        selected = select_ttl_samples(sentence_ce, threshold=0.5)

        self.assertEqual(selected, [1])

    def test_online_logits_update_matches_shared_weighted_loss(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.consumers._common.alignment import sentence_cross_entropy, weighted_one_hot_kl_loss

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
        config = TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.0, threshold=1.0, lamb=0.5)
        consumer = TLMConsumer(config)

        consumer.update_online(hidden, logits=logits, labels=labels)

        stats = consumer.read_stats()
        expected = weighted_one_hot_kl_loss(
            logits,
            labels,
            sentence_ce=sentence_cross_entropy(logits, labels),
            threshold=1.0,
            lamb=0.5,
        )
        self.assertAlmostEqual(stats.last_loss, float(expected), places=6)
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.loss_count, 1)
        self.assertEqual(stats.selected_count, 1)

    def test_online_logits_update_trains_hidden_to_vocab_adapter(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.2, threshold=1.0, lamb=0.5))
        hidden = torch.randn(3, 4)
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 4.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 1])

        consumer.update_online(hidden, logits=logits, labels=labels)

        adapter = consumer.engine.adapters[0]
        self.assertEqual(adapter.out_dim, 3)
        self.assertGreater(consumer.read_stats().selected_count, 0)
        self.assertFalse(torch.allclose(adapter.up.weight.detach(), torch.zeros_like(adapter.up.weight)))

    def test_offline_uses_base_logits_for_selection_then_trains_adapter_logits(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.consumers._common.alignment import sentence_cross_entropy, weighted_one_hot_kl_loss

        consumer = TLMConsumer(
            TLMConsumerConfig(mode="offline", hidden_size=4, rank=2, learning_rate=0.0, threshold=1.0, lamb=0.25)
        )
        hidden = torch.eye(2, 4)
        train_logits = torch.tensor(
            [
                [0.0, 4.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        base_logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 0])

        consumer.update_offline(hidden, logits=train_logits, labels=labels, base_logits=base_logits)
        self.assertEqual(consumer.read_stats().update_count, 0)
        consumer.flush_offline()

        stats = consumer.read_stats()
        expected = weighted_one_hot_kl_loss(
            train_logits,
            labels,
            sentence_ce=sentence_cross_entropy(base_logits, labels),
            threshold=1.0,
            lamb=0.25,
        )
        wrong_online_selection = weighted_one_hot_kl_loss(
            train_logits,
            labels,
            sentence_ce=sentence_cross_entropy(train_logits, labels),
            threshold=1.0,
            lamb=0.25,
        )
        self.assertEqual(stats.update_count, 1)
        self.assertAlmostEqual(stats.last_loss, float(expected), places=6)
        self.assertNotAlmostEqual(stats.last_loss, float(wrong_online_selection), places=4)
        self.assertEqual(stats.selected_count, 1)

    def test_ordering_helper_runs_online_predict_then_train_and_offline_train_then_predict(self) -> None:
        from tllm.consumers.tlm import run_tlm_ordered_step

        calls: list[str] = []

        def predict() -> str:
            calls.append("predict")
            return "ok"

        def train() -> None:
            calls.append("train")

        self.assertEqual(run_tlm_ordered_step("online", predict=predict, train=train), "ok")
        self.assertEqual(calls, ["predict", "train"])

        calls.clear()
        self.assertEqual(run_tlm_ordered_step("offline", predict=predict, train=train), "ok")
        self.assertEqual(calls, ["train", "predict"])

    def test_scalar_score_fallback_online_changes_adapter_and_records_stats(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1, threshold=0.5))
        hidden = torch.randn(3, 4)
        scores = torch.tensor([0.2, 0.8, 0.1])
        before = consumer.engine.adapters[0].up.weight.detach().clone()

        consumer.update_online(hidden, scores, allow_scalar_fallback=True)

        after = consumer.engine.adapters[0].up.weight.detach()
        stats = consumer.read_stats()
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.selected_count, 2)

    def test_legacy_scalar_score_online_call_remains_supported(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1, threshold=0.5))

        consumer.update_online(torch.randn(2, 4), torch.tensor([0.2, 0.8]))

        self.assertEqual(consumer.read_stats().update_count, 1)

    def test_scalar_score_fallback_offline_accumulates_until_flush(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(mode="offline", hidden_size=4, rank=2, learning_rate=0.1, threshold=0.5))

        consumer.update_offline(torch.randn(2, 4), torch.tensor([0.2, 0.8]), allow_scalar_fallback=True)
        self.assertEqual(consumer.read_stats().update_count, 0)
        consumer.flush_offline()

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.selected_count, 1)

    def test_disabled_legacy_scalar_score_call_is_noop(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(enabled=False, hidden_size=4))

        consumer.update_online(torch.randn(2, 4), torch.tensor([0.1, 0.2]))

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_disabled_consumer_does_not_update(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(enabled=False, hidden_size=4))

        consumer.update_online(
            torch.randn(2, 4),
            logits=torch.randn(2, 3),
            labels=torch.tensor([0, 1]),
        )

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_tlm_module_does_not_import_llamafactory(self) -> None:
        before = set(sys.modules)

        __import__("tllm.consumers.tlm")

        self.assertNotIn("llamafactory", set(sys.modules) - before)


if __name__ == "__main__":
    unittest.main()

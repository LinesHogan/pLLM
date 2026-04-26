#!/usr/bin/env python3
"""Unit tests for the TLM-inspired devlist consumer."""

from __future__ import annotations

import sys
import unittest

import torch


class TLMConsumerUnitTest(unittest.TestCase):
    def test_package_exports_tlm_consumer(self) -> None:
        from tllm.consumers.devlist import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig())

        self.assertEqual(consumer.consumer_id, "tlm")

    def test_config_rejects_unknown_mode(self) -> None:
        from tllm.consumers.devlist import TLMConsumerConfig

        with self.assertRaisesRegex(ValueError, "mode"):
            TLMConsumerConfig(mode="streaming")

    def test_select_samples_keeps_scores_below_threshold(self) -> None:
        from tllm.consumers.devlist.tlm import select_ttl_samples

        scores = torch.tensor([0.2, 0.8, 0.4])

        selected = select_ttl_samples(scores, threshold=0.5)

        self.assertEqual(selected, [0, 2])

    def test_online_update_changes_adapter_and_records_stats(self) -> None:
        from tllm.consumers.devlist import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1, threshold=0.5))
        hidden = torch.randn(3, 4)
        scores = torch.tensor([0.2, 0.8, 0.1])
        before = consumer.engine.adapters[0].up.weight.detach().clone()

        consumer.update_online(hidden, scores)

        after = consumer.engine.adapters[0].up.weight.detach()
        stats = consumer.read_stats()
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.selected_count, 2)

    def test_offline_update_accumulates_until_flush(self) -> None:
        from tllm.consumers.devlist import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(mode="offline", hidden_size=4, rank=2, learning_rate=0.1, threshold=0.5))

        consumer.update_offline(torch.randn(2, 4), torch.tensor([0.2, 0.8]))
        self.assertEqual(consumer.read_stats().update_count, 0)
        consumer.flush_offline()

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.selected_count, 1)

    def test_disabled_consumer_does_not_update(self) -> None:
        from tllm.consumers.devlist import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(enabled=False, hidden_size=4))

        consumer.update_online(torch.randn(2, 4), torch.tensor([0.1, 0.2]))

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_tlm_module_does_not_import_llamafactory(self) -> None:
        before = set(sys.modules)

        __import__("tllm.consumers.devlist.tlm")

        self.assertNotIn("llamafactory", set(sys.modules) - before)


if __name__ == "__main__":
    unittest.main()

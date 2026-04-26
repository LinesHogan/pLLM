#!/usr/bin/env python3
"""Unit tests for the FTTT-inspired devlist consumer."""

from __future__ import annotations

import sys
import unittest

import torch


class FTTTConsumerUnitTest(unittest.TestCase):
    def test_package_exports_fttt_consumer(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig

        consumer = FTTTConsumer(FTTTConsumerConfig())

        self.assertEqual(consumer.consumer_id, "fttt")

    def test_enqueue_feedback_records_bounded_queue(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.devlist.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(max_queue_size=1))

        accepted = consumer.enqueue_feedback(FeedbackRecord(prompt_idx=0, feedback="incorrect", reward=0.0))
        dropped = consumer.enqueue_feedback(FeedbackRecord(prompt_idx=1, feedback="incorrect", reward=0.0))

        stats = consumer.read_stats()
        self.assertTrue(accepted)
        self.assertFalse(dropped)
        self.assertEqual(stats.queued_feedback, 1)
        self.assertEqual(stats.dropped_feedback, 1)

    def test_apply_feedback_update_changes_adapter_and_records_stats(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.devlist.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        hidden = torch.randn(2, 4)
        before = consumer.engine.adapter.up.weight.detach().clone()

        consumer.apply_feedback_update(
            hidden,
            [
                FeedbackRecord(prompt_idx=0, feedback="incorrect", reward=0.0),
                FeedbackRecord(prompt_idx=1, feedback="correct", reward=1.0),
            ],
        )

        after = consumer.engine.adapter.up.weight.detach()
        stats = consumer.read_stats()
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.feedback_count, 2)

    def test_inference_mode_hidden_is_accepted(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.devlist.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        with torch.inference_mode():
            hidden = torch.randn(1, 4)

        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=0, feedback="incorrect", reward=0.0)])

        self.assertEqual(consumer.read_stats().update_count, 1)

    def test_disabled_consumer_does_not_update(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.devlist.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(enabled=False, hidden_size=4))

        consumer.apply_feedback_update(torch.randn(1, 4), [FeedbackRecord(prompt_idx=0, feedback="incorrect", reward=0.0)])

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_fttt_module_does_not_import_external_feedback_apis(self) -> None:
        before = set(sys.modules)

        __import__("tllm.consumers.devlist.fttt")

        imported = set(sys.modules) - before
        self.assertNotIn("requests", imported)
        self.assertNotIn("fastapi", imported)


if __name__ == "__main__":
    unittest.main()

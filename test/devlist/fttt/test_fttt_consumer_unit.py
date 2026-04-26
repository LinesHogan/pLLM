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
        self.assertEqual(stats.feedback_count, 1)
        self.assertEqual(stats.skipped_feedback, 1)

    def test_same_reward_with_different_feedback_text_produces_different_edit(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.devlist.fttt import FeedbackRecord

        torch.manual_seed(7)
        config = FTTTConsumerConfig(hidden_size=4, rank=3, learning_rate=0.2)
        hidden = torch.tensor([[0.25, -0.5, 0.75, 1.0]])

        arithmetic = FTTTConsumer(config)
        arithmetic_feedback = [FeedbackRecord(prompt_idx=0, feedback="wrong arithmetic carry", reward=0.0)]
        arithmetic.apply_feedback_update(hidden, arithmetic_feedback)
        arithmetic_delta = arithmetic.edit_delta(hidden, arithmetic_feedback)

        torch.manual_seed(7)
        notation = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=3, learning_rate=0.2))
        notation_feedback = [FeedbackRecord(prompt_idx=0, feedback="wrong final notation", reward=0.0)]
        notation.apply_feedback_update(hidden, notation_feedback)
        notation_delta = notation.edit_delta(hidden, notation_feedback)

        self.assertFalse(torch.allclose(arithmetic_delta, notation_delta))

    def test_positive_feedback_stops_later_adaptation_for_instance(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.devlist.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        hidden = torch.randn(1, 4)

        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=3, feedback="correct", reward=1.0)])
        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=3, feedback="now fix arithmetic", reward=0.0)])

        state = consumer.instance_state(3)
        stats = consumer.read_stats()
        self.assertTrue(state.stopped)
        self.assertEqual(state.retries, 0)
        self.assertEqual(stats.update_count, 0)
        self.assertEqual(stats.skipped_feedback, 2)

    def test_per_instance_reset_allows_adaptation_after_stop(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.devlist.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        hidden = torch.randn(1, 4)

        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=3, feedback="correct", reward=1.0)])
        consumer.reset_instance(3)
        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=3, feedback="fix the sign", reward=0.0)])

        state = consumer.instance_state(3)
        stats = consumer.read_stats()
        self.assertFalse(state.stopped)
        self.assertEqual(state.retries, 1)
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.feedback_count, 1)

    def test_edit_api_changes_hidden_and_logits(self) -> None:
        from tllm.consumers.devlist import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.devlist.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.2))
        hidden = torch.randn(1, 4)
        feedback = [FeedbackRecord(prompt_idx=0, feedback="revise the last operation", reward=0.0)]
        consumer.apply_feedback_update(hidden, feedback)

        edited_hidden = consumer.apply_edit_to_hidden(hidden, feedback)
        logits = torch.zeros(1, 4)
        edited_logits = consumer.apply_edit_to_logits(logits, hidden, feedback)

        self.assertFalse(torch.allclose(edited_hidden, hidden))
        self.assertFalse(torch.allclose(edited_logits, logits))

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

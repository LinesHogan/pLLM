#!/usr/bin/env python3
"""Unit tests for the GTTA-inspired devlist consumer."""

from __future__ import annotations

import sys
import unittest


class GTTAConsumerUnitTest(unittest.TestCase):
    def test_package_exports_gtta_consumer(self) -> None:
        from tllm.consumers.devlist import GTTAConsumer, GTTAConsumerConfig

        consumer = GTTAConsumer(GTTAConsumerConfig())

        self.assertEqual(consumer.consumer_id, "gtta")

    def test_trajectory_record_requires_action(self) -> None:
        from tllm.consumers.devlist.gtta import TrajectoryFeedback

        with self.assertRaisesRegex(ValueError, "action"):
            TrajectoryFeedback(task_id="task", action="", observation="obs", reward=1.0)

    def test_parse_reward_accepts_numeric_and_dict_values(self) -> None:
        from tllm.consumers.devlist.gtta import parse_reward

        self.assertEqual(parse_reward(2), 2.0)
        self.assertEqual(parse_reward({"reward": "3.5"}), 3.5)

    def test_enqueue_trajectory_records_bounded_queue(self) -> None:
        from tllm.consumers.devlist import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.devlist.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig(max_queue_size=1))

        accepted = consumer.enqueue_trajectory(TrajectoryFeedback("task", "click", "obs", 1.0))
        dropped = consumer.enqueue_trajectory(TrajectoryFeedback("task", "type", "obs", 0.0))

        stats = consumer.read_stats()
        self.assertTrue(accepted)
        self.assertFalse(dropped)
        self.assertEqual(stats.queued_trajectories, 1)
        self.assertEqual(stats.dropped_trajectories, 1)

    def test_process_trajectories_updates_action_scores(self) -> None:
        from tllm.consumers.devlist import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.devlist.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(TrajectoryFeedback("task", "click", "obs", 1.0))
        consumer.enqueue_trajectory(TrajectoryFeedback("task", "type", "obs", 0.0))

        consumer.process_queued_trajectories()

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 2)
        self.assertEqual(stats.queued_trajectories, 0)
        self.assertGreater(consumer.score_action("click"), consumer.score_action("type"))

    def test_disabled_consumer_does_not_update_scores(self) -> None:
        from tllm.consumers.devlist import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.devlist.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig(enabled=False))
        consumer.enqueue_trajectory(TrajectoryFeedback("task", "click", "obs", 1.0))

        consumer.process_queued_trajectories()

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_gtta_module_does_not_import_environment_frameworks(self) -> None:
        before = set(sys.modules)

        __import__("tllm.consumers.devlist.gtta")

        imported = set(sys.modules) - before
        self.assertNotIn("browsergym", imported)
        self.assertNotIn("bfcl", imported)


if __name__ == "__main__":
    unittest.main()

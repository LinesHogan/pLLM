#!/usr/bin/env python3
"""Unit tests for the GTTA-inspired consumer."""

from __future__ import annotations

import unittest


class GTTAConsumerUnitTest(unittest.TestCase):
    def test_package_exports_gtta_consumer(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig

        consumer = GTTAConsumer(GTTAConsumerConfig())

        self.assertEqual(consumer.consumer_id, "gtta")

    def test_rank_candidate_actions_prefers_matching_state_task_reward(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="find_route",
                action="open directions",
                previous_observation="map homepage with search results",
                next_observation="directions panel opened",
                reward=0.9,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="find_route",
                action="clear search",
                previous_observation="map homepage with search results",
                next_observation="empty search page",
                reward=-0.2,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="different_task",
                action="clear search",
                previous_observation="map homepage with search results",
                next_observation="desired outcome elsewhere",
                reward=1.0,
            )
        )

        consumer.process_queued_trajectories()

        ranked = consumer.rank_candidate_actions(
            task_id="find_route",
            observation="map homepage with search results",
            candidate_actions=["clear search", "unknown action", "open directions"],
        )

        self.assertEqual([action for action, _score in ranked], ["open directions", "unknown action", "clear search"])
        self.assertGreater(ranked[0][1], ranked[-1][1])


if __name__ == "__main__":
    unittest.main()

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

    def test_same_action_scores_are_isolated_by_task_and_state(self) -> None:
        from tllm.consumers.devlist import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.devlist.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="buy_item",
                action="click submit",
                previous_observation="checkout form with valid payment details",
                next_observation="order confirmation page",
                reward=1.0,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="buy_item",
                action="click submit",
                previous_observation="empty cart page",
                next_observation="validation error page",
                reward=-1.0,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="cancel_order",
                action="click submit",
                previous_observation="checkout form with valid payment details",
                next_observation="wrong task outcome",
                reward=-0.5,
            )
        )

        consumer.process_queued_trajectories()

        self.assertGreater(
            consumer.score_action(
                "click submit",
                task_id="buy_item",
                observation="checkout form with valid payment details",
            ),
            0.0,
        )
        self.assertLess(
            consumer.score_action(
                "click submit",
                task_id="buy_item",
                observation="empty cart page",
            ),
            0.0,
        )
        self.assertLess(
            consumer.score_action(
                "click submit",
                task_id="cancel_order",
                observation="checkout form with valid payment details",
            ),
            0.0,
        )

    def test_rank_candidate_actions_prefers_matching_state_task_reward(self) -> None:
        from tllm.consumers.devlist import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.devlist.gtta import TrajectoryFeedback

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

    def test_dynamics_records_preserve_observation_and_dynamics_text(self) -> None:
        from tllm.consumers.devlist import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.devlist.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="wishlist",
                action="click add to wishlist",
                previous_observation="product page with wishlist button",
                next_observation="wishlist page with confirmation alert",
                reward={"reward": "0.75"},
                environmental_dynamics="A confirmation alert appeared and the wishlist count increased.",
            )
        )

        consumer.process_queued_trajectories()

        records = consumer.export_dynamics_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["task_id"], "wishlist")
        self.assertEqual(records[0]["previous_observation"], "product page with wishlist button")
        self.assertEqual(records[0]["next_observation"], "wishlist page with confirmation alert")
        self.assertEqual(
            records[0]["environmental_dynamics"],
            "A confirmation alert appeared and the wishlist count increased.",
        )
        self.assertEqual(records[0]["reward"], 0.75)

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

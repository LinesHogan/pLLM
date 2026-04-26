#!/usr/bin/env python3
"""Toy coverage for GTTA consumer-local dynamics accumulation."""

from __future__ import annotations

import unittest


class GTTADynamicsToyUnitTest(unittest.TestCase):
    def test_query_dynamics_records_filters_task_state_and_action(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="reserve_table",
                action="click confirm",
                previous_observation="booking form with a 7 pm time selected",
                next_observation="reservation confirmation page",
                environmental_dynamics="The restaurant accepted the selected time.",
                reward=1.0,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="reserve_table",
                action="click confirm",
                previous_observation="booking form with no time selected",
                next_observation="validation error asking for a time",
                environmental_dynamics="Confirmation failed because the time field was empty.",
                reward=-0.5,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="cancel_table",
                action="click confirm",
                previous_observation="booking form with a 7 pm time selected",
                next_observation="wrong workflow page",
                environmental_dynamics="The same button is not useful for the cancel task.",
                reward=-1.0,
            )
        )

        consumer.process_queued_trajectories()

        matching_records = consumer.query_dynamics_records(
            task_id="reserve_table",
            observation="booking form with a 7 pm time selected",
            action="click confirm",
        )
        ranked = consumer.rank_candidate_actions(
            task_id="reserve_table",
            observation="booking form with a 7 pm time selected",
            candidate_actions=["click confirm", "go back"],
        )

        self.assertEqual(len(matching_records), 1)
        self.assertEqual(
            matching_records[0]["environmental_dynamics"],
            "The restaurant accepted the selected time.",
        )
        self.assertEqual(matching_records[0]["reward"], 1.0)
        self.assertEqual([action for action, _score in ranked], ["click confirm", "go back"])

    def test_query_dynamics_records_returns_copy(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="wishlist",
                action="click heart",
                previous_observation="product page",
                next_observation="wishlist counter incremented",
                environmental_dynamics="The item was added to the wishlist.",
                reward=0.8,
            )
        )
        consumer.process_queued_trajectories()

        queried = consumer.query_dynamics_records(task_id="wishlist")
        queried[0]["environmental_dynamics"] = "mutated by caller"

        self.assertEqual(
            consumer.query_dynamics_records(task_id="wishlist")[0]["environmental_dynamics"],
            "The item was added to the wishlist.",
        )


if __name__ == "__main__":
    unittest.main()

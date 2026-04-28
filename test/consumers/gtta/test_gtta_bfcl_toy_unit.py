#!/usr/bin/env python3
"""Tests for dependency-light GTTA BFCL toy fixtures."""

from __future__ import annotations

from pathlib import Path
import unittest


_FIXTURE = Path(__file__).with_name("fixtures") / "bfcl_toy_tasks.jsonl"


class GTTABFCLToyUnitTest(unittest.TestCase):
    def test_load_bfcl_toy_tasks_preserves_function_call_fields(self) -> None:
        from tllm.consumers.gtta.bfcl_toy import load_bfcl_toy_tasks

        tasks = load_bfcl_toy_tasks(_FIXTURE)

        self.assertEqual([task.id for task in tasks], ["toy_file_positive", "toy_file_negative"])
        self.assertEqual(tasks[0].task_id, "toy_file_create")
        self.assertEqual(tasks[0].environment, "ToyFileSystem")
        self.assertEqual(tasks[0].candidate_actions, ("touch(path='note.txt')", "ls()"))
        self.assertEqual(tasks[0].gold_action, "touch(path='note.txt')")
        self.assertEqual(tasks[0].initial_observation, "files=[]")
        self.assertEqual(tasks[0].next_observation, "files=['note.txt']")
        self.assertEqual(tasks[0].reward, 1.0)
        self.assertEqual(tasks[0].functions[0]["name"], "touch")

    def test_execute_bfcl_toy_action_returns_observation_reward_and_feedback(self) -> None:
        from tllm.consumers.gtta.bfcl_toy import execute_bfcl_toy_action, load_bfcl_toy_tasks

        task = load_bfcl_toy_tasks(_FIXTURE)[0]

        result = execute_bfcl_toy_action(task, "touch(path='note.txt')")

        self.assertEqual(result.action, "touch(path='note.txt')")
        self.assertEqual(result.tool_result, "created note.txt")
        self.assertEqual(result.next_observation, "files=['note.txt']")
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.feedback.task_id, "toy_file_create")
        self.assertEqual(result.feedback.previous_observation, "files=[]")
        self.assertEqual(result.feedback.action, "touch(path='note.txt')")
        self.assertEqual(result.feedback.environmental_dynamics, "Creating note.txt adds it to the file list.")

    def test_toy_loop_updates_ranking_and_prompt_context(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.gtta.bfcl_toy import execute_bfcl_toy_action, load_bfcl_toy_tasks

        consumer = GTTAConsumer(GTTAConsumerConfig())
        positive, negative = load_bfcl_toy_tasks(_FIXTURE)
        consumer.enqueue_trajectory(execute_bfcl_toy_action(negative, "ls()").feedback)
        consumer.enqueue_trajectory(execute_bfcl_toy_action(positive, "touch(path='note.txt')").feedback)

        consumer.process_queued_trajectories()

        ranked = consumer.rank_candidate_actions(
            task_id=positive.task_id,
            observation=positive.initial_observation,
            candidate_actions=positive.candidate_actions,
        )
        context = consumer.build_environment_dynamics_prompt_context_entry(
            prompt_idx=0,
            task_id=positive.task_id,
            observation=positive.initial_observation,
            candidate_actions=positive.candidate_actions,
        )

        self.assertEqual([action for action, _score in ranked], ["touch(path='note.txt')", "ls()"])
        self.assertIn("Creating note.txt adds it to the file list.", context.text)
        self.assertIn("Listing files before creating note.txt leaves the file list unchanged.", context.text)
        self.assertEqual(consumer.read_stats().update_count, 2)

    def test_execute_bfcl_toy_action_rejects_unknown_action(self) -> None:
        from tllm.consumers.gtta.bfcl_toy import execute_bfcl_toy_action, load_bfcl_toy_tasks

        task = load_bfcl_toy_tasks(_FIXTURE)[0]

        with self.assertRaisesRegex(ValueError, "candidate action"):
            execute_bfcl_toy_action(task, "rm(path='note.txt')")


if __name__ == "__main__":
    unittest.main()

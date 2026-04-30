#!/usr/bin/env python3
"""Deterministic BFCL-shaped minimal alignment tests for GTTA."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


_FIXTURE_DIR = Path(__file__).with_name("fixtures") / "bfcl_minimal"
_TASKS = _FIXTURE_DIR / "tasks.jsonl"
_EXPECTED_ACTIONS = _FIXTURE_DIR / "expected_actions.jsonl"
_SEED_DYNAMICS = _FIXTURE_DIR / "seed_dynamics.json"


def _load_expected_actions() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with _EXPECTED_ACTIONS.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            records[str(payload["task_id"])] = payload
    return records


def _load_seed_dynamics() -> list[dict[str, object]]:
    with _SEED_DYNAMICS.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["records"])


class GTTABFCLMinimalAlignmentUnitTest(unittest.TestCase):
    def test_loads_bfcl_shaped_minimal_tasks_and_expected_records(self) -> None:
        from tllm.consumers.gtta.bfcl_toy import load_bfcl_toy_tasks

        tasks = load_bfcl_toy_tasks(_TASKS)
        expected = _load_expected_actions()
        dynamics = _load_seed_dynamics()

        self.assertEqual(len(tasks), 3)
        self.assertEqual({task.task_id for task in tasks}, set(expected))
        self.assertEqual([function["name"] for function in tasks[0].functions], ["touch", "ls", "read"])
        self.assertEqual(tasks[0].question[0][0]["role"], "user")
        self.assertEqual(len(dynamics), 6)
        self.assertEqual(
            {str(record["action"]).split("(", maxsplit=1)[0] for record in dynamics},
            {"touch", "ls", "read"},
        )

    def test_baseline_ranking_uses_deterministic_candidate_order(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.gtta.bfcl_toy import load_bfcl_toy_tasks

        consumer = GTTAConsumer(GTTAConsumerConfig())
        expected = _load_expected_actions()

        for task in load_bfcl_toy_tasks(_TASKS):
            ranked = consumer.rank_candidate_actions(
                task_id=task.task_id,
                observation=task.initial_observation,
                candidate_actions=task.candidate_actions,
            )

            self.assertEqual([score for _action, score in ranked], [0.0] * len(task.candidate_actions))
            self.assertEqual(ranked[0][0], expected[task.task_id]["baseline_selected"])

    def test_seed_dynamics_changes_selected_action_toward_expected_action(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig, TrajectoryFeedback
        from tllm.consumers.gtta.bfcl_toy import load_bfcl_toy_tasks

        consumer = GTTAConsumer(GTTAConsumerConfig())
        for record in _load_seed_dynamics():
            consumer.enqueue_trajectory(TrajectoryFeedback(**record))
        consumer.process_queued_trajectories()

        expected = _load_expected_actions()
        for task in load_bfcl_toy_tasks(_TASKS):
            ranked = consumer.rank_candidate_actions(
                task_id=task.task_id,
                observation=task.initial_observation,
                candidate_actions=task.candidate_actions,
            )
            selected_action, selected_score = ranked[0]
            baseline_action = str(expected[task.task_id]["baseline_selected"])

            self.assertEqual(selected_action, expected[task.task_id]["expected_action"])
            self.assertGreater(selected_score, consumer.score_action(baseline_action, task_id=task.task_id, observation=task.initial_observation))
        self.assertEqual(consumer.read_stats().update_count, 6)

    def test_tool_execution_observation_and_reward_match_expected_actions(self) -> None:
        from tllm.consumers.gtta.bfcl_toy import execute_bfcl_toy_action, load_bfcl_toy_tasks

        expected = _load_expected_actions()

        for task in load_bfcl_toy_tasks(_TASKS):
            record = expected[task.task_id]
            result = execute_bfcl_toy_action(task, str(record["expected_action"]))

            self.assertEqual(result.tool_result, record["tool_result"])
            self.assertEqual(result.next_observation, record["next_observation"])
            self.assertEqual(result.reward, record["reward"])
            self.assertEqual(result.feedback.previous_observation, task.initial_observation)
            self.assertEqual(result.feedback.environmental_dynamics, task.environmental_dynamics)

    def test_unknown_minimal_action_still_fails(self) -> None:
        from tllm.consumers.gtta.bfcl_toy import execute_bfcl_toy_action, load_bfcl_toy_tasks

        task = load_bfcl_toy_tasks(_TASKS)[0]

        with self.assertRaisesRegex(ValueError, "candidate action"):
            execute_bfcl_toy_action(task, "delete(path='readme.md')")


if __name__ == "__main__":
    unittest.main()

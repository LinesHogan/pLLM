#!/usr/bin/env python3
"""Unit tests for reward outcome runtime storage."""

from __future__ import annotations

import unittest


class RewardOutcomeRuntimeUnitTest(unittest.TestCase):
    def test_reward_outcome_entry_rejects_implicit_identity_coercion(self) -> None:
        from tllm.runtime.reward_outcome_runtime import RewardOutcomeEntry

        with self.assertRaisesRegex(TypeError, "request_id must be str"):
            RewardOutcomeEntry(request_id=7, prompt_idx=0, sample_idx=0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "request_id must not contain surrounding whitespace"):
            RewardOutcomeEntry(request_id=" req-a ", prompt_idx=0, sample_idx=0)
        with self.assertRaisesRegex(TypeError, "prompt_idx must be int"):
            RewardOutcomeEntry(request_id="req-a", prompt_idx=True, sample_idx=0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "reward must be real"):
            RewardOutcomeEntry(request_id="req-a", prompt_idx=0, sample_idx=0, reward=True)  # type: ignore[arg-type]

    def test_reward_outcome_store_appends_reads_and_drains_batches(self) -> None:
        from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry, RewardOutcomeStore

        store = RewardOutcomeStore()
        first = RewardOutcomeEntry(
            request_id="req-a",
            prompt_idx=4,
            sample_idx=2,
            output_index=1,
            group_id="vote-1",
            reward=1.0,
            judge="boxed",
            label="4",
            feedback_text="correct",
            is_correct=True,
        )
        second = RewardOutcomeEntry(
            request_id="req-b",
            prompt_idx=5,
            sample_idx=0,
            reward=0.0,
            judge="boxed",
            skip_reason="missing-boxed-answer",
            is_correct=False,
        )

        self.assertEqual(store.append(RewardOutcomeBatch(entries=(first, second))), 2)
        snapshot = store.read()
        self.assertEqual(snapshot.entries, (first, second))
        self.assertEqual(snapshot.entries[0].reward, 1.0)
        self.assertEqual(snapshot.entries[1].skip_reason, "missing-boxed-answer")

        drained = store.drain()
        self.assertEqual(drained.entries, (first, second))
        self.assertEqual(store.read().entries, ())

    def test_runtime_helpers_store_last_reward_outcomes(self) -> None:
        from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry, read_reward_outcomes, write_reward_outcomes

        runtime = type("Runtime", (), {})()
        batch = RewardOutcomeBatch(
            entries=(
                RewardOutcomeEntry(request_id="req-a", prompt_idx=1, sample_idx=0, reward=0.5, group_id="round-a"),
            )
        )

        self.assertEqual(write_reward_outcomes(runtime, batch), 1)
        self.assertIs(runtime.last_reward_outcomes, batch)
        self.assertEqual(read_reward_outcomes(runtime).entries, batch.entries)
        self.assertEqual(read_reward_outcomes(runtime, drain=True).entries, batch.entries)
        self.assertEqual(read_reward_outcomes(runtime).entries, ())

    def test_runtime_helpers_reject_non_bool_drain_flag(self) -> None:
        from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry, read_reward_outcomes, write_reward_outcomes

        runtime = type("Runtime", (), {})()
        batch = RewardOutcomeBatch(entries=(RewardOutcomeEntry(request_id="req-a", prompt_idx=1, sample_idx=0),))
        write_reward_outcomes(runtime, batch)

        with self.assertRaisesRegex(TypeError, "drain must be bool"):
            read_reward_outcomes(runtime, drain="False")  # type: ignore[arg-type]
        self.assertEqual(read_reward_outcomes(runtime).entries, batch.entries)

    def test_algorithm_owned_strings_preserve_whitespace(self) -> None:
        from tllm.runtime.reward_outcome_runtime import RewardOutcomeEntry

        entry = RewardOutcomeEntry(
            request_id="req-a",
            prompt_idx=1,
            sample_idx=0,
            judge=" local judge ",
            label=" answer label ",
            skip_reason=" parser skipped ",
        )

        self.assertEqual(entry.judge, " local judge ")
        self.assertEqual(entry.label, " answer label ")
        self.assertEqual(entry.skip_reason, " parser skipped ")


if __name__ == "__main__":
    unittest.main()

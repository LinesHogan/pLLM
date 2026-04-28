#!/usr/bin/env python3
"""Tests for TTRL lightweight verifier and toy closed-loop helpers."""

from __future__ import annotations

import sys
import unittest

import torch


def _trace_445():
    from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

    return GenerationTraceBatch(
        entries=(
            CompletionTraceEntry(
                request_id="req-a",
                prompt_idx=0,
                sample_idx=0,
                output_index=0,
                token_ids=(),
                text=r"First route. \boxed{4}",
                finish_reason="stop",
                group_id="round-0",
            ),
            CompletionTraceEntry(
                request_id="req-b",
                prompt_idx=0,
                sample_idx=1,
                output_index=0,
                token_ids=(),
                text=r"Second route. \boxed{4}",
                finish_reason="stop",
                group_id="round-0",
            ),
            CompletionTraceEntry(
                request_id="req-c",
                prompt_idx=0,
                sample_idx=2,
                output_index=0,
                token_ids=(),
                text=r"Third route. \boxed{5}",
                finish_reason="stop",
                group_id="round-0",
            ),
        )
    )


class TTRLLightweightClosedLoopUnitTest(unittest.TestCase):
    def test_default_boxed_verifier_is_replaceable_and_does_not_import_math_grader(self) -> None:
        from tllm.consumers.ttrl import BoxedNormalizedVerifier, TTRLSample, build_ttrl_reward_outcome_batch, build_ttrl_reward_result

        class PrefixVerifier:
            judge_name = "prefix_exact"
            missing_reason = "missing-prefix-answer"

            def label_for_completion(self, completion: str) -> str:
                if completion.startswith("OK:"):
                    return completion.split(":", 1)[1].strip()
                return ""

        before = set(sys.modules)
        default_result = build_ttrl_reward_result(
            [
                TTRLSample(prompt_idx=0, sample_idx=0, completion=r"\boxed{ 4 }"),
                TTRLSample(prompt_idx=0, sample_idx=1, completion=r"\boxed{4}"),
                TTRLSample(prompt_idx=0, sample_idx=2, completion=r"\boxed{5}"),
            ],
            verifier=BoxedNormalizedVerifier(),
        )
        custom_result = build_ttrl_reward_result(
            [
                TTRLSample(prompt_idx=0, sample_idx=0, completion="OK: A"),
                TTRLSample(prompt_idx=0, sample_idx=1, completion="OK: A"),
                TTRLSample(prompt_idx=0, sample_idx=2, completion=r"\boxed{A}"),
            ],
            verifier=PrefixVerifier(),
        )
        after = set(sys.modules)

        self.assertEqual(default_result.rewards, [1.0, 1.0, 0.0])
        self.assertEqual(default_result.verifier_name, "boxed_normalized_exact")
        self.assertEqual(custom_result.rewards, [1.0, 1.0, 0.0])
        self.assertEqual(custom_result.verifier_name, "prefix_exact")
        self.assertEqual(custom_result.sample_results[2].skip_reason, "missing-prefix-answer")
        self.assertEqual(
            build_ttrl_reward_outcome_batch(_trace_445(), verifier=PrefixVerifier()).entries[0].judge,
            "ttrl_majority:prefix_exact",
        )
        self.assertNotIn("math_verify", after - before)
        self.assertNotIn("latex2sympy2_extended", after - before)

    def test_toy_no_model_closed_loop_updates_from_rewards_and_retries_minority(self) -> None:
        from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig, run_ttrl_toy_closed_loop

        consumer = TTRLConsumer(TTRLConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        hidden_rows = torch.randn(3, 4)

        intent = run_ttrl_toy_closed_loop(consumer, _trace_445(), hidden_rows)

        self.assertEqual([entry.reward for entry in intent.reward_outcomes.entries], [1.0, 1.0, 0.0])
        self.assertEqual([entry.action for entry in intent.retry_controls.entries], ["accept", "accept", "retry"])
        self.assertTrue(intent.retry_controls.entries[2].retry_after_update)
        self.assertFalse(hasattr(intent, "update_count"))
        self.assertEqual(intent.training_row_count, 3)
        self.assertEqual(intent.selected_reward_count, 3)
        self.assertEqual(consumer.read_stats().update_count, 1)
        self.assertEqual(consumer.read_stats().reward_count, 3)

    def test_retry_schedule_carries_updated_adapter_version_for_second_round(self) -> None:
        from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig, run_ttrl_toy_closed_loop

        class FakeScheduler:
            def __init__(self) -> None:
                self.scheduled: list[tuple[str, str]] = []

            def schedule(self, intent) -> None:
                for retry in intent.retry_schedule:
                    self.scheduled.append((retry.request_id, retry.adapter_version_id))

        consumer = TTRLConsumer(TTRLConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        scheduler = FakeScheduler()

        first_intent = run_ttrl_toy_closed_loop(consumer, _trace_445(), torch.randn(3, 4))
        scheduler.schedule(first_intent)

        self.assertEqual(first_intent.adapter_version_id, "ttrl:update:1")
        self.assertEqual(scheduler.scheduled, [("req-c", "ttrl:update:1")])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for the TTRL-inspired devlist consumer."""

from __future__ import annotations

import sys
import unittest

import torch


class TTRLConsumerUnitTest(unittest.TestCase):
    def test_package_exports_ttrl_consumer(self) -> None:
        from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig

        consumer = TTRLConsumer(TTRLConsumerConfig())

        self.assertEqual(consumer.consumer_id, "ttrl")

    def test_consumer_declares_trace_to_reward_outcome_flow(self) -> None:
        from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig
        from tllm.ports.base import PortKind

        consumer = TTRLConsumer(TTRLConsumerConfig())

        flows = consumer.flows()

        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0].window, "out_of_band")
        self.assertEqual(flows[0].reads[0].kind, PortKind.GENERATION_TRACE)
        self.assertEqual(flows[0].writes[0].kind, PortKind.REWARD_OUTCOME)

    def test_majority_rewards_ignore_unboxed_completions(self) -> None:
        from tllm.consumers.ttrl import TTRLSample, build_ttrl_reward_result

        samples = [
            TTRLSample(prompt_idx=0, sample_idx=0, completion="Answer: 4"),
            TTRLSample(prompt_idx=0, sample_idx=1, completion=r"Therefore \boxed{5}"),
            TTRLSample(prompt_idx=0, sample_idx=2, completion="Answer: 4"),
        ]

        result = build_ttrl_reward_result(samples)

        self.assertEqual(result.rewards, [0.0, 1.0, 0.0])
        self.assertEqual(result.majority_by_prompt[0].label, "5")
        self.assertEqual(result.majority_by_prompt[0].count, 1)
        self.assertEqual(result.sample_results[0].skip_reason, "missing-boxed-answer")
        self.assertEqual(result.sample_results[2].skip_reason, "missing-boxed-answer")

    def test_majority_rewards_vote_on_normalized_extracted_answers(self) -> None:
        from tllm.consumers.ttrl import TTRLSample, majority_vote_rewards

        samples = [
            TTRLSample(prompt_idx=0, sample_idx=0, completion=r"First derivation. Therefore \boxed{4}"),
            TTRLSample(prompt_idx=0, sample_idx=1, completion=r"Second derivation. Final: \boxed{4}"),
            TTRLSample(prompt_idx=0, sample_idx=2, completion=r"Wrong path. Therefore \boxed{5}"),
        ]

        rewards = majority_vote_rewards(samples)

        self.assertEqual(rewards, [1.0, 1.0, 0.0])

    def test_majority_rewards_break_ties_by_first_seen_completion(self) -> None:
        from tllm.consumers.ttrl import TTRLSample, majority_vote_rewards

        samples = [
            TTRLSample(prompt_idx=0, sample_idx=0, completion=r"\boxed{A}"),
            TTRLSample(prompt_idx=0, sample_idx=1, completion=r"\boxed{B}"),
        ]

        rewards = majority_vote_rewards(samples)

        self.assertEqual(rewards, [1.0, 0.0])

    def test_vote_sample_selection_matches_upstream_block_indices(self) -> None:
        from tllm.consumers.ttrl import select_ttrl_training_indices

        selected = select_ttrl_training_indices(
            num_prompts=2,
            n_votes_per_prompt=4,
            n_samples_per_prompt=2,
        )

        self.assertEqual(selected, [0, 1, 4, 5])

    def test_structured_vote_result_exposes_majority_label_ratio_and_rewards(self) -> None:
        from tllm.consumers.ttrl import TTRLSample, build_ttrl_reward_result

        samples = [
            TTRLSample(prompt_idx=0, sample_idx=0, completion=r"Reasoning A. \boxed{4}"),
            TTRLSample(prompt_idx=0, sample_idx=1, completion=r"Reasoning B. \boxed{4}"),
            TTRLSample(prompt_idx=0, sample_idx=2, completion=r"Reasoning C. \boxed{5}"),
        ]

        result = build_ttrl_reward_result(samples)

        self.assertEqual(result.majority_by_prompt[0].label, "4")
        self.assertEqual(result.majority_by_prompt[0].ratio, 2 / 3)
        self.assertEqual(result.sample_results[0].label, "4")
        self.assertEqual([item.reward for item in result.sample_results], [1.0, 1.0, 0.0])

    def test_trace_to_reward_outcome_batch_uses_boxed_majority_and_preserves_identity(self) -> None:
        from tllm.consumers.ttrl import build_ttrl_reward_outcome_batch
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-a",
                    prompt_idx=7,
                    sample_idx=0,
                    output_index=2,
                    token_ids=(1, 2),
                    text=r"Path A. \boxed{4}",
                    finish_reason="stop",
                    group_id="vote-a",
                ),
                CompletionTraceEntry(
                    request_id="req-b",
                    prompt_idx=7,
                    sample_idx=1,
                    output_index=0,
                    token_ids=(3,),
                    text=r"Path B. \boxed{4}",
                    finish_reason="stop",
                    group_id="vote-a",
                ),
                CompletionTraceEntry(
                    request_id="req-c",
                    prompt_idx=7,
                    sample_idx=2,
                    output_index=0,
                    token_ids=(4,),
                    text=r"Path C. \boxed{5}",
                    finish_reason="length",
                    group_id="vote-a",
                ),
            )
        )

        batch = build_ttrl_reward_outcome_batch(trace)

        self.assertEqual(len(batch.entries), 3)
        self.assertEqual(batch.entries[0].request_id, "req-a")
        self.assertEqual(batch.entries[0].prompt_idx, 7)
        self.assertEqual(batch.entries[0].sample_idx, 0)
        self.assertEqual(batch.entries[0].output_index, 2)
        self.assertEqual(batch.entries[0].group_id, "vote-a")
        self.assertEqual(batch.entries[0].judge, "ttrl_boxed_majority")
        self.assertEqual(batch.entries[0].label, "4")
        self.assertEqual(batch.entries[0].reward, 1.0)
        self.assertIs(batch.entries[0].is_correct, True)
        self.assertEqual(batch.entries[2].label, "5")
        self.assertEqual(batch.entries[2].reward, 0.0)
        self.assertIs(batch.entries[2].is_correct, False)

    def test_trace_to_reward_outcome_batch_marks_unboxed_as_skip(self) -> None:
        from tllm.consumers.ttrl import build_ttrl_reward_outcome_batch
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-a",
                    prompt_idx=0,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(),
                    text="Answer: 4",
                    finish_reason="stop",
                    group_id="vote-a",
                ),
                CompletionTraceEntry(
                    request_id="req-b",
                    prompt_idx=0,
                    sample_idx=1,
                    output_index=0,
                    token_ids=(),
                    text=r"Therefore \boxed{5}",
                    finish_reason="stop",
                    group_id="vote-a",
                ),
            )
        )

        batch = build_ttrl_reward_outcome_batch(trace)

        self.assertIsNone(batch.entries[0].reward)
        self.assertIsNone(batch.entries[0].is_correct)
        self.assertEqual(batch.entries[0].label, "")
        self.assertEqual(batch.entries[0].skip_reason, "missing-boxed-answer")
        self.assertEqual(batch.entries[1].reward, 1.0)
        self.assertEqual(batch.entries[1].label, "5")

    def test_ttrl_module_does_not_import_verl(self) -> None:
        before = set(sys.modules)

        __import__("tllm.consumers.ttrl")

        after = set(sys.modules)
        self.assertNotIn("verl", after - before)

    def test_update_from_rewards_changes_adapter_and_records_stats(self) -> None:
        from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig

        consumer = TTRLConsumer(TTRLConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        hidden = torch.randn(3, 4)
        rewards = torch.tensor([1.0, 0.0, 1.0])
        before = consumer.engine.adapter.up.weight.detach().clone()

        consumer.update_from_rewards(hidden, rewards)

        after = consumer.engine.adapter.up.weight.detach()
        stats = consumer.read_stats()
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.reward_count, 3)

    def test_disabled_consumer_does_not_update(self) -> None:
        from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig

        consumer = TTRLConsumer(TTRLConsumerConfig(enabled=False, hidden_size=4))

        consumer.update_from_rewards(torch.randn(2, 4), torch.tensor([1.0, 0.0]))

        self.assertEqual(consumer.read_stats().update_count, 0)


if __name__ == "__main__":
    unittest.main()

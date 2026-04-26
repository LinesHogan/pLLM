#!/usr/bin/env python3
"""Unit tests for the TTRL-inspired devlist consumer."""

from __future__ import annotations

import sys
import unittest

import torch


class TTRLConsumerUnitTest(unittest.TestCase):
    def test_package_exports_ttrl_consumer(self) -> None:
        from tllm.consumers.devlist import TTRLConsumer, TTRLConsumerConfig

        consumer = TTRLConsumer(TTRLConsumerConfig())

        self.assertEqual(consumer.consumer_id, "ttrl")

    def test_majority_rewards_assign_one_to_winning_completion(self) -> None:
        from tllm.consumers.devlist.ttrl import TTRLSample, majority_vote_rewards

        samples = [
            TTRLSample(prompt_idx=0, sample_idx=0, completion="Answer: 4"),
            TTRLSample(prompt_idx=0, sample_idx=1, completion="answer: 4 "),
            TTRLSample(prompt_idx=0, sample_idx=2, completion="Answer: 5"),
        ]

        rewards = majority_vote_rewards(samples)

        self.assertEqual(rewards, [1.0, 1.0, 0.0])

    def test_majority_rewards_vote_on_normalized_extracted_answers(self) -> None:
        from tllm.consumers.devlist.ttrl import TTRLSample, majority_vote_rewards

        samples = [
            TTRLSample(prompt_idx=0, sample_idx=0, completion=r"First derivation. Therefore \boxed{4}"),
            TTRLSample(prompt_idx=0, sample_idx=1, completion=r"Second derivation. Final: \boxed{4}"),
            TTRLSample(prompt_idx=0, sample_idx=2, completion=r"Wrong path. Therefore \boxed{5}"),
        ]

        rewards = majority_vote_rewards(samples)

        self.assertEqual(rewards, [1.0, 1.0, 0.0])

    def test_majority_rewards_break_ties_by_first_seen_completion(self) -> None:
        from tllm.consumers.devlist.ttrl import TTRLSample, majority_vote_rewards

        samples = [
            TTRLSample(prompt_idx=0, sample_idx=0, completion="A"),
            TTRLSample(prompt_idx=0, sample_idx=1, completion="B"),
        ]

        rewards = majority_vote_rewards(samples)

        self.assertEqual(rewards, [1.0, 0.0])

    def test_vote_sample_selection_matches_upstream_block_indices(self) -> None:
        from tllm.consumers.devlist.ttrl import select_ttrl_training_indices

        selected = select_ttrl_training_indices(
            num_prompts=2,
            n_votes_per_prompt=4,
            n_samples_per_prompt=2,
        )

        self.assertEqual(selected, [0, 1, 4, 5])

    def test_structured_vote_result_exposes_majority_label_ratio_and_rewards(self) -> None:
        from tllm.consumers.devlist.ttrl import TTRLSample, build_ttrl_reward_result

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

    def test_ttrl_module_does_not_import_verl(self) -> None:
        before = set(sys.modules)

        __import__("tllm.consumers.devlist.ttrl")

        after = set(sys.modules)
        self.assertNotIn("verl", after - before)

    def test_update_from_rewards_changes_adapter_and_records_stats(self) -> None:
        from tllm.consumers.devlist import TTRLConsumer, TTRLConsumerConfig

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
        from tllm.consumers.devlist import TTRLConsumer, TTRLConsumerConfig

        consumer = TTRLConsumer(TTRLConsumerConfig(enabled=False, hidden_size=4))

        consumer.update_from_rewards(torch.randn(2, 4), torch.tensor([1.0, 0.0]))

        self.assertEqual(consumer.read_stats().update_count, 0)


if __name__ == "__main__":
    unittest.main()

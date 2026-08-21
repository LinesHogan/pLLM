#!/usr/bin/env python3
"""Unit tests for the TTRL-inspired consumer."""

from __future__ import annotations

import unittest


class TTRLConsumerUnitTest(unittest.TestCase):
    def test_consumer_declares_trace_to_reward_outcome_flow(self) -> None:
        from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig
        from tllm.ports.base import PortKind

        consumer = TTRLConsumer(TTRLConsumerConfig())

        flows = consumer.flows()

        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0].window, "out_of_band")
        self.assertEqual(flows[0].reads[0].kind, PortKind.GENERATION_TRACE)
        self.assertEqual(flows[0].writes[0].kind, PortKind.REWARD_OUTCOME)

    def test_majority_rewards_vote_on_normalized_extracted_answers(self) -> None:
        from tllm.consumers.ttrl import TTRLSample, majority_vote_rewards

        samples = [
            TTRLSample(prompt_idx=0, sample_idx=0, completion=r"First derivation. Therefore \boxed{4}"),
            TTRLSample(prompt_idx=0, sample_idx=1, completion=r"Second derivation. Final: \boxed{4}"),
            TTRLSample(prompt_idx=0, sample_idx=2, completion=r"Wrong path. Therefore \boxed{5}"),
        ]

        rewards = majority_vote_rewards(samples)

        self.assertEqual(rewards, [1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()

class TTRLProgramTrainingTest(unittest.TestCase):
    def test_update_from_rewards_trains_bank_params(self) -> None:
        import torch

        from tllm.consumers.ttrl import TTRLConsumer, TTRLConsumerConfig

        torch.manual_seed(0)
        consumer = TTRLConsumer(TTRLConsumerConfig(rank=2, learning_rate=0.2))
        hidden = torch.randn(4, 8)
        rewards = [1.0, 1.0, 0.0, 1.0]
        consumer.update_from_rewards(hidden, rewards)

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.reward_count, 4)
        compiled = consumer.compiled
        assert compiled is not None
        slot = compiled.bank("reward_model").slot_for(None)
        self.assertGreater(float(slot.params["adapter.up.weight"].abs().max()), 0.0)
        self.assertEqual(consumer.adapter_version_id(), "ttrl:update:1")

        # zero-init up-proj means step-0 predictions are exactly zero
        scores = consumer.predict_scores(hidden)
        self.assertEqual(tuple(scores.shape), (4, 1))

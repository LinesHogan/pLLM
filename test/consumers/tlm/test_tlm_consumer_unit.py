#!/usr/bin/env python3
"""Unit tests for the TLM-inspired consumer."""

from __future__ import annotations

import unittest

import torch


class TLMConsumerUnitTest(unittest.TestCase):
    def test_package_exports_tlm_consumer(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig())

        self.assertEqual(consumer.consumer_id, "tlm")

    def test_online_logits_update_matches_shared_weighted_loss(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.consumers._common.alignment import sentence_cross_entropy, weighted_one_hot_kl_loss

        hidden = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 4.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 1])
        config = TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.0, threshold=1.0, lamb=0.5)
        consumer = TLMConsumer(config)

        consumer.update_online(hidden, logits=logits, labels=labels)

        stats = consumer.read_stats()
        expected = weighted_one_hot_kl_loss(
            logits,
            labels,
            sentence_ce=sentence_cross_entropy(logits, labels),
            threshold=1.0,
            lamb=0.5,
        )
        self.assertAlmostEqual(stats.last_loss, float(expected), places=6)
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.loss_count, 1)
        self.assertEqual(stats.selected_count, 1)

    def test_closed_loop_online_predicts_before_training_and_stages_request_lora(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate

        class _StageOnlyService:
            def __init__(self) -> None:
                self.updates: list[TestTimeLoRAUpdate] = []

            def stage_update(self, update: TestTimeLoRAUpdate) -> None:
                calls.append("stage")
                self.updates.append(update)

            def commit_staged_updates(self) -> int:
                calls.append("commit")
                return len(self.updates)

        calls: list[str] = []
        consumer = TLMConsumer(TLMConsumerConfig(mode="online", hidden_size=4, rank=2, learning_rate=0.2, threshold=1.0))
        hidden = torch.eye(3, 4)
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 4.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 1])
        service = _StageOnlyService()

        result = consumer.run_lora_learning_step(
            predict=lambda: calls.append("predict") or "prediction",
            hidden=hidden,
            logits=logits,
            labels=labels,
            lora_service=service,
            adapter_key="tlm-online-req-1",
            lora_int_id=41,
            target_module_name="model.layers.0.self_attn.q_proj",
            request_ids=("req-1",),
            step_id=7,
        )

        self.assertEqual(result, "prediction")
        self.assertEqual(calls, ["predict", "stage", "commit"])
        self.assertEqual(consumer.read_stats().update_count, 1)
        self.assertEqual(len(service.updates), 1)
        self.assertEqual(service.updates[0].request_ids, ("req-1",))
        self.assertEqual(service.updates[0].step_id, 7)


if __name__ == "__main__":
    unittest.main()

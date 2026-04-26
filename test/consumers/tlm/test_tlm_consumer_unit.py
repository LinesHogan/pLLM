#!/usr/bin/env python3
"""Unit tests for the TLM-inspired devlist consumer."""

from __future__ import annotations

import sys
import unittest

import torch


def _upstream_causal_lm_sentence_ce(raw_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    from tllm.consumers._common.alignment import sentence_cross_entropy

    return sentence_cross_entropy(raw_logits[:, :-1, :], labels[:, 1:])


class TLMConsumerUnitTest(unittest.TestCase):
    def test_package_exports_tlm_consumer(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig())

        self.assertEqual(consumer.consumer_id, "tlm")

    def test_config_rejects_unknown_mode(self) -> None:
        from tllm.consumers.tlm import TLMConsumerConfig

        with self.assertRaisesRegex(ValueError, "mode"):
            TLMConsumerConfig(mode="streaming")

    def test_select_samples_keeps_raw_ce_above_threshold(self) -> None:
        from tllm.consumers.tlm import select_ttl_samples

        sentence_ce = torch.tensor([0.2, 0.8, 0.4])

        selected = select_ttl_samples(sentence_ce, threshold=0.5)

        self.assertEqual(selected, [1])

    def test_raw_causal_lm_logits_must_be_shifted_before_ttl_selection(self) -> None:
        from tllm.consumers.tlm import select_ttl_samples
        from tllm.consumers._common.alignment import sentence_cross_entropy

        labels = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        raw_logits = torch.full((2, 3, 3), -4.0)
        raw_logits[0, 0, 0] = 4.0
        raw_logits[0, 1, 1] = 4.0
        raw_logits[0, 2, 2] = 4.0
        raw_logits[1, 0, 1] = 4.0
        raw_logits[1, 1, 2] = 4.0
        raw_logits[1, 2, 0] = 4.0
        threshold = 2.0

        wrong_direct_ce = sentence_cross_entropy(raw_logits, labels)
        upstream_ce = _upstream_causal_lm_sentence_ce(raw_logits, labels)

        self.assertEqual(select_ttl_samples(wrong_direct_ce, threshold=threshold), [1])
        self.assertEqual(select_ttl_samples(upstream_ce, threshold=threshold), [0])
        self.assertNotEqual(
            select_ttl_samples(wrong_direct_ce, threshold=threshold),
            select_ttl_samples(upstream_ce, threshold=threshold),
        )
        self.assertTrue(torch.allclose(upstream_ce, sentence_cross_entropy(raw_logits[:, :-1, :], labels[:, 1:])))

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

    def test_online_logits_update_trains_hidden_to_vocab_adapter(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.2, threshold=1.0, lamb=0.5))
        hidden = torch.randn(3, 4)
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 4.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 1])

        consumer.update_online(hidden, logits=logits, labels=labels)

        adapter = consumer.engine.adapters[0]
        self.assertEqual(adapter.out_dim, 3)
        self.assertGreater(consumer.read_stats().selected_count, 0)
        self.assertFalse(torch.allclose(adapter.up.weight.detach(), torch.zeros_like(adapter.up.weight)))

    def test_offline_uses_base_logits_for_selection_then_trains_adapter_logits(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.consumers._common.alignment import sentence_cross_entropy, weighted_one_hot_kl_loss

        consumer = TLMConsumer(
            TLMConsumerConfig(mode="offline", hidden_size=4, rank=2, learning_rate=0.0, threshold=1.0, lamb=0.25)
        )
        hidden = torch.eye(2, 4)
        train_logits = torch.tensor(
            [
                [0.0, 4.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        base_logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 0])

        consumer.update_offline(hidden, logits=train_logits, labels=labels, base_logits=base_logits)
        self.assertEqual(consumer.read_stats().update_count, 0)
        consumer.flush_offline()

        stats = consumer.read_stats()
        expected = weighted_one_hot_kl_loss(
            train_logits,
            labels,
            sentence_ce=sentence_cross_entropy(base_logits, labels),
            threshold=1.0,
            lamb=0.25,
        )
        wrong_online_selection = weighted_one_hot_kl_loss(
            train_logits,
            labels,
            sentence_ce=sentence_cross_entropy(train_logits, labels),
            threshold=1.0,
            lamb=0.25,
        )
        self.assertEqual(stats.update_count, 1)
        self.assertAlmostEqual(stats.last_loss, float(expected), places=6)
        self.assertNotAlmostEqual(stats.last_loss, float(wrong_online_selection), places=4)
        self.assertEqual(stats.selected_count, 1)

    def test_offline_base_logits_contract_expects_caller_aligned_causal_lm_logits(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.consumers._common.alignment import sentence_cross_entropy, weighted_one_hot_kl_loss

        labels = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
        raw_train_logits = torch.full((2, 3, 3), -4.0)
        raw_base_logits = torch.full((2, 3, 3), -4.0)
        raw_train_logits[0, 0, 1] = 4.0
        raw_train_logits[0, 1, 2] = 4.0
        raw_train_logits[0, 2, 0] = 4.0
        raw_train_logits[1, 0, 0] = 4.0
        raw_train_logits[1, 1, 1] = 4.0
        raw_train_logits[1, 2, 2] = 4.0
        raw_base_logits[0, 0, 0] = 4.0
        raw_base_logits[0, 1, 1] = 4.0
        raw_base_logits[0, 2, 2] = 4.0
        raw_base_logits[1, 0, 1] = 4.0
        raw_base_logits[1, 1, 2] = 4.0
        raw_base_logits[1, 2, 0] = 4.0
        aligned_labels = labels[:, 1:]
        aligned_train_logits = raw_train_logits[:, :-1, :]
        aligned_base_logits = raw_base_logits[:, :-1, :]
        consumer = TLMConsumer(
            TLMConsumerConfig(mode="offline", hidden_size=4, rank=2, learning_rate=0.0, threshold=2.0, lamb=0.25)
        )

        consumer.update_offline(
            torch.eye(2, 4),
            logits=aligned_train_logits,
            labels=aligned_labels,
            base_logits=aligned_base_logits,
        )
        consumer.flush_offline()

        expected = weighted_one_hot_kl_loss(
            aligned_train_logits,
            aligned_labels,
            sentence_ce=sentence_cross_entropy(aligned_base_logits, aligned_labels),
            threshold=2.0,
            lamb=0.25,
        )
        wrong_unshifted_base_selection = weighted_one_hot_kl_loss(
            aligned_train_logits,
            aligned_labels,
            sentence_ce=sentence_cross_entropy(raw_base_logits, labels),
            threshold=2.0,
            lamb=0.25,
        )
        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.selected_count, 1)
        self.assertAlmostEqual(stats.last_loss, float(expected), places=6)
        self.assertNotAlmostEqual(stats.last_loss, float(wrong_unshifted_base_selection), places=4)

    def test_ordering_helper_runs_online_predict_then_train_and_offline_train_then_predict(self) -> None:
        from tllm.consumers.tlm import run_tlm_ordered_step

        calls: list[str] = []

        def predict() -> str:
            calls.append("predict")
            return "ok"

        def train() -> None:
            calls.append("train")

        self.assertEqual(run_tlm_ordered_step("online", predict=predict, train=train), "ok")
        self.assertEqual(calls, ["predict", "train"])

        calls.clear()
        self.assertEqual(run_tlm_ordered_step("offline", predict=predict, train=train), "ok")
        self.assertEqual(calls, ["train", "predict"])

    def test_scalar_score_fallback_online_changes_adapter_and_records_stats(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1, threshold=0.5))
        hidden = torch.randn(3, 4)
        scores = torch.tensor([0.2, 0.8, 0.1])
        before = consumer.engine.adapters[0].up.weight.detach().clone()

        consumer.update_online(hidden, scores, allow_scalar_fallback=True)

        after = consumer.engine.adapters[0].up.weight.detach()
        stats = consumer.read_stats()
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.selected_count, 2)

    def test_legacy_scalar_score_online_call_remains_supported(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1, threshold=0.5))

        consumer.update_online(torch.randn(2, 4), torch.tensor([0.2, 0.8]))

        self.assertEqual(consumer.read_stats().update_count, 1)

    def test_scalar_score_fallback_offline_accumulates_until_flush(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(mode="offline", hidden_size=4, rank=2, learning_rate=0.1, threshold=0.5))

        consumer.update_offline(torch.randn(2, 4), torch.tensor([0.2, 0.8]), allow_scalar_fallback=True)
        self.assertEqual(consumer.read_stats().update_count, 0)
        consumer.flush_offline()

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.selected_count, 1)

    def test_disabled_legacy_scalar_score_call_is_noop(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(enabled=False, hidden_size=4))

        consumer.update_online(torch.randn(2, 4), torch.tensor([0.1, 0.2]))

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_disabled_consumer_does_not_update(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(enabled=False, hidden_size=4))

        consumer.update_online(
            torch.randn(2, 4),
            logits=torch.randn(2, 3),
            labels=torch.tensor([0, 1]),
        )

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_exports_local_adapter_as_strict_test_time_lora_update(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time_lora import TestTimeLoRAService

        class _FakeLoRAModule:
            def __init__(self) -> None:
                self.calls: list[tuple[int, torch.Tensor, torch.Tensor, float]] = []

            def set_lora(
                self,
                adapter_id: int,
                lora_a: torch.Tensor,
                lora_b: torch.Tensor,
                *,
                bias: torch.Tensor | None,
                scaling: float,
            ) -> None:
                self.calls.append((adapter_id, lora_a.detach().clone(), lora_b.detach().clone(), float(scaling)))

        class _FakeManager:
            def __init__(self, module: _FakeLoRAModule) -> None:
                self.modules = {"model.layers.0.self_attn.q_proj": module}

        torch.manual_seed(11)
        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, vocab_size=3, rank=2, learning_rate=0.2, threshold=1.0))
        consumer.update_online(
            torch.eye(3, 4),
            logits=torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0],
                    [0.0, 4.0, 0.0],
                ]
            ),
            labels=torch.tensor([0, 1, 1]),
        )

        update = consumer.export_test_time_lora_update(
            adapter_key="tlm-req-1",
            lora_int_id=17,
            target_module_name="model.layers.0.self_attn.q_proj",
            rank=2,
            scope="request",
            request_ids=("req-1",),
            step_id=5,
            alpha=4.0,
        )

        self.assertIsInstance(update, TestTimeLoRAUpdate)
        self.assertEqual(update.adapter_key, "tlm-req-1")
        self.assertEqual(update.lora_int_id, 17)
        self.assertEqual(update.rank, 2)
        self.assertEqual(update.scope, "request")
        self.assertEqual(update.request_ids, ("req-1",))
        self.assertEqual(update.step_id, 5)
        self.assertEqual(tuple(update.target_modules), ("model.layers.0.self_attn.q_proj",))
        weights = update.target_modules["model.layers.0.self_attn.q_proj"]
        self.assertIsInstance(weights, TestTimeLoRAWeights)
        self.assertEqual(tuple(weights.lora_a.shape), (4, 2))
        self.assertEqual(tuple(weights.lora_b.shape), (2, 3))
        self.assertEqual(weights.alpha, 4.0)
        self.assertGreater(float(weights.lora_b.abs().sum()), 0.0)

        module = _FakeLoRAModule()
        service = TestTimeLoRAService(manager=_FakeManager(module), max_rank=2)
        service.stage_update(update)

        self.assertEqual(service.commit_staged_updates(), 1)
        self.assertEqual(service.adapter_for_request("req-1"), 17)
        self.assertEqual(len(module.calls), 1)
        adapter_id, committed_lora_a, committed_lora_b, scaling = module.calls[0]
        self.assertEqual(adapter_id, 17)
        self.assertTrue(torch.equal(committed_lora_a, weights.lora_a))
        self.assertTrue(torch.equal(committed_lora_b, weights.lora_b))
        self.assertEqual(scaling, 2.0)

    def test_lora_export_rejects_rank_target_and_request_boundary_mistakes(self) -> None:
        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig

        consumer = TLMConsumer(TLMConsumerConfig(hidden_size=4, vocab_size=3, rank=2))

        with self.assertRaisesRegex(ValueError, "adapter rank"):
            consumer.export_test_time_lora_update(
                adapter_key="tlm-rank",
                lora_int_id=1,
                target_module_name="model.layers.0.self_attn.q_proj",
                rank=3,
            )

        with self.assertRaisesRegex(TypeError, "request_ids must be tuple"):
            consumer.export_test_time_lora_update(
                adapter_key="tlm-ids",
                lora_int_id=2,
                target_module_name="model.layers.0.self_attn.q_proj",
                rank=2,
                scope="request",
                request_ids=["req-1"],  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(TypeError, "request_ids must contain str"):
            consumer.export_test_time_lora_update(
                adapter_key="tlm-id-member",
                lora_int_id=2,
                target_module_name="model.layers.0.self_attn.q_proj",
                rank=2,
                scope="request",
                request_ids=(7,),  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(TypeError, "rank must be int"):
            consumer.export_test_time_lora_update(
                adapter_key="tlm-bool-rank",
                lora_int_id=3,
                target_module_name="model.layers.0.self_attn.q_proj",
                rank=True,  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(ValueError, "target module name"):
            consumer.export_test_time_lora_update(
                adapter_key="tlm-target",
                lora_int_id=4,
                target_module_name=" model.layers.0.self_attn.q_proj ",
                rank=2,
            )

    def test_tlm_module_does_not_import_llamafactory(self) -> None:
        before = set(sys.modules)

        __import__("tllm.consumers.tlm")

        self.assertNotIn("llamafactory", set(sys.modules) - before)


if __name__ == "__main__":
    unittest.main()

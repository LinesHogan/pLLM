#!/usr/bin/env python3
"""Guarded cached-model smoke tests for the TLM consumer."""

from __future__ import annotations

import os
import unittest

import torch


@unittest.skipUnless(
    os.environ.get("TLLM_TLM_RUN_CACHED_MODEL_SMOKE") == "1",
    "set TLLM_TLM_RUN_CACHED_MODEL_SMOKE=1 to run cached-model TLM smoke",
)
class TLMCachedModelSmokeTest(unittest.TestCase):
    def test_cached_tiny_gpt2_closed_loop_stages_lora_update(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate

        class _StageOnlyService:
            def __init__(self) -> None:
                self.updates: list[TestTimeLoRAUpdate] = []

            def stage_update(self, update: TestTimeLoRAUpdate) -> None:
                self.updates.append(update)

        model_name = os.environ.get("TLLM_TLM_CACHED_SMOKE_MODEL", "sshleifer/tiny-gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        model.eval()

        batch = tokenizer(
            ["The capital of France is Paris", "The capital of Italy is Milan"],
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**batch, output_hidden_states=True)

        aligned_logits = output.logits[:, :-1, :].detach()
        aligned_labels = batch["input_ids"][:, 1:].detach()
        hidden = output.hidden_states[-1][:, -2, :].detach()
        consumer = TLMConsumer(
            TLMConsumerConfig(
                mode="online",
                hidden_size=int(hidden.shape[-1]),
                vocab_size=int(aligned_logits.shape[-1]),
                rank=2,
                learning_rate=0.1,
                threshold=0.0,
            )
        )
        service = _StageOnlyService()

        result = consumer.run_lora_learning_step(
            predict=lambda: "generated",
            hidden=hidden,
            logits=aligned_logits,
            labels=aligned_labels,
            lora_service=service,
            adapter_key="tlm-cached-smoke",
            lora_int_id=71,
            target_module_name="lm_head",
            request_ids=("req-cached",),
            step_id=1,
            alpha=2.0,
        )

        self.assertEqual(result, "generated")
        self.assertEqual(consumer.read_stats().update_count, 1)
        self.assertGreater(consumer.read_stats().selected_count, 0)
        self.assertEqual(len(service.updates), 1)
        update = service.updates[0]
        self.assertEqual(update.adapter_key, "tlm-cached-smoke")
        self.assertEqual(update.request_ids, ("req-cached",))
        self.assertEqual(tuple(update.target_modules), ("lm_head",))


if __name__ == "__main__":
    unittest.main()

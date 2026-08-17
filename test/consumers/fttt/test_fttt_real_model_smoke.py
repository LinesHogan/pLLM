#!/usr/bin/env python3
"""Opt-in cached-model smoke for the FTTT feedback-LM trainer handoff."""

from __future__ import annotations

import os
import unittest

import torch


@unittest.skipUnless(
    os.environ.get("TLLM_FTTT_RUN_CACHED_MODEL_SMOKE") == "1",
    "set TLLM_FTTT_RUN_CACHED_MODEL_SMOKE=1 to run cached-model FTTT smoke",
)
class FTTTRealModelSmokeTest(unittest.TestCase):
    def test_cached_tiny_model_feedback_batch_stages_lora_update_and_retry_intent(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.contracts.test_time_training import TestTimeTrainerRequest, TestTimeTrainerResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
        from tllm.runtime.test_time.lora import TestTimeLoRAService

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
                self.modules = {"lm_head": module}

        class _FeedbackLMTrainer:
            def __init__(self, hidden_size: int, vocab_size: int) -> None:
                self.hidden_size = int(hidden_size)
                self.vocab_size = int(vocab_size)
                self.requests: list[TestTimeTrainerRequest] = []

            def train(self, request: TestTimeTrainerRequest) -> TestTimeTrainerResult:
                self.requests.append(request)
                self.assert_feedback_lm_request(request)
                token_total = sum(sum(example.token_ids) for example in request.examples.examples)
                torch.manual_seed(int(token_total % 997))
                update = TestTimeLoRAUpdate(
                    adapter_key=request.adapter_key,
                    lora_int_id=41,
                    rank=request.rank,
                    scope=request.scope,
                    request_ids=request.request_ids,
                    target_modules={
                        "lm_head": TestTimeLoRAWeights(
                            lora_a=torch.randn(self.hidden_size, request.rank) * 0.01,
                            lora_b=torch.randn(request.rank, self.vocab_size) * 0.01,
                            alpha=float(request.rank),
                        )
                    },
                    step_id=11,
                )
                return TestTimeTrainerResult(updates=(update,), metrics={"feedback_lm_examples": float(len(request.examples.examples))})

            @staticmethod
            def assert_feedback_lm_request(request: TestTimeTrainerRequest) -> None:
                for example in request.examples.examples:
                    if example.objective_kind != "feedback_lm":
                        raise AssertionError("FTTT smoke trainer expects feedback_lm examples")
                    if example.labels is None or not any(label >= 0 for label in example.labels):
                        raise AssertionError("feedback_lm examples must expose feedback token labels")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_name = os.environ.get("TLLM_FTTT_CACHED_SMOKE_MODEL", "sshleifer/tiny-gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        model.eval()

        encoded = tokenizer("Question: 2+2=5\nFeedback: fix the arithmetic.", return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
        hidden_size = int(output.hidden_states[-1].shape[-1])
        vocab_size = int(output.logits.shape[-1])

        feedback_ids = tuple(int(x) for x in tokenizer.encode("fix the arithmetic", add_special_tokens=False)[:4])
        consumer = FTTTConsumer(
            FTTTConsumerConfig(max_retries=2),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                reward=0.0,
                feedback_text="Your answer is incorrect. Fix the arithmetic.",
                feedback_token_ids=feedback_ids,
            ),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-fttt-smoke",
                    prompt_idx=0,
                    sample_idx=0,
                    output_index=0,
                    token_ids=tuple(int(x) for x in encoded["input_ids"][0, -3:].tolist()),
                    text="5",
                    finish_reason="stop",
                    group_id="round-fttt-smoke",
                ),
            )
        )

        module = _FakeLoRAModule()
        service = TestTimeLoRAService(manager=_FakeManager(module), max_rank=1)
        trainer = _FeedbackLMTrainer(hidden_size=hidden_size, vocab_size=vocab_size)

        result = consumer.run_trainer_bridge_closed_loop_step(
            trace,
            trainer_bridge=trainer,
            lora_service=service,
            adapter_key_prefix="fttt",
            target_modules=("lm_head",),
            rank=1,
            step_id=11,
        )

        self.assertEqual(len(trainer.requests), 1)
        self.assertEqual(result.staged_update_count, 1)
        self.assertEqual(service.commit_staged_updates(), 1)
        self.assertEqual(service.adapter_for_request("req-fttt-smoke"), 41)
        self.assertEqual(len(module.calls), 1)
        retry = result.retry_controls.entries[0]
        self.assertTrue(retry.retry_after_update)
        self.assertIn("adapter-key=fttt:round-fttt-smoke:0:v11", retry.reason)


if __name__ == "__main__":
    unittest.main()

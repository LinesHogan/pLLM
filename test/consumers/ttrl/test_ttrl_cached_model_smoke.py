#!/usr/bin/env python3
"""Opt-in cached-model smoke for TTRL rollout -> GRPO trainer handoff."""

from __future__ import annotations

import os
import unittest

import torch


@unittest.skipUnless(
    os.environ.get("TLLM_TTRL_RUN_CACHED_MODEL_SMOKE") == "1",
    "set TLLM_TTRL_RUN_CACHED_MODEL_SMOKE=1 to run cached-model TTRL smoke",
)
class TTRLCachedModelSmokeTest(unittest.TestCase):
    def test_cached_qwen_rollout_feeds_grpo_batch_and_stages_lora_update(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tllm.consumers.ttrl import (
            BoxedNormalizedVerifier,
            TTRLConsumer,
            TTRLConsumerConfig,
            TTRLSample,
            build_ttrl_grpo_training_batch,
            build_ttrl_reward_result,
        )
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

        class _ToyGRPOTrainer:
            def __init__(self, hidden_size: int, vocab_size: int) -> None:
                self.hidden_size = int(hidden_size)
                self.vocab_size = int(vocab_size)
                self.requests: list[TestTimeTrainerRequest] = []

            def train(self, request: TestTimeTrainerRequest) -> TestTimeTrainerResult:
                self.requests.append(request)
                if request.algorithm_id != "ttrl_grpo":
                    raise AssertionError("TTRL smoke trainer expects ttrl_grpo requests")
                if not request.examples.examples:
                    raise AssertionError("TTRL smoke trainer expects non-empty GRPO examples")
                for example in request.examples.examples:
                    if example.objective_kind != "grpo":
                        raise AssertionError("TTRL smoke trainer expects grpo examples")
                    if not example.rewards or "majority_label" not in example.metadata:
                        raise AssertionError("GRPO examples must retain reward and majority metadata")
                token_total = sum(sum(example.token_ids) for example in request.examples.examples)
                torch.manual_seed(int(token_total % 997))
                update = TestTimeLoRAUpdate(
                    adapter_key=request.adapter_key,
                    lora_int_id=73,
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
                    step_id=17,
                )
                return TestTimeTrainerResult(updates=(update,), metrics={"grpo_examples": float(len(request.examples.examples))})

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_name = os.environ.get("TLLM_TTRL_CACHED_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
        model.eval()

        prompt = "Solve 2+2. End the answer exactly as \\boxed{4}."
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            forward = model(**encoded, output_hidden_states=True)
            generated = model.generate(
                **encoded,
                do_sample=True,
                num_return_sequences=4,
                max_new_tokens=16,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        hidden_size = int(forward.hidden_states[-1].shape[-1])
        vocab_size = int(forward.logits.shape[-1])
        entries: list[CompletionTraceEntry] = []
        for idx, sequence in enumerate(generated):
            token_ids = tuple(int(x) for x in sequence.tolist())
            text = tokenizer.decode(sequence, skip_special_tokens=True)
            entries.append(
                CompletionTraceEntry(
                    request_id=f"req-ttrl-smoke-{idx}",
                    prompt_idx=0,
                    sample_idx=idx,
                    output_index=0,
                    token_ids=token_ids,
                    text=text,
                    finish_reason="length",
                    group_id="round-ttrl-smoke",
                )
            )
        trace = GenerationTraceBatch(entries=tuple(entries))
        reward_result = build_ttrl_reward_result(
            [
                TTRLSample(
                    prompt_idx=entry.prompt_idx,
                    sample_idx=entry.sample_idx,
                    completion=entry.text,
                )
                for entry in trace.entries
            ],
            verifier=BoxedNormalizedVerifier(),
        )
        if not any(sample.label for sample in reward_result.sample_results):
            self.skipTest("cached model did not emit a boxed answer under this smoke prompt")

        batch = build_ttrl_grpo_training_batch(trace, verifier=BoxedNormalizedVerifier())
        module = _FakeLoRAModule()
        service = TestTimeLoRAService(manager=_FakeManager(module), max_rank=1)
        trainer = _ToyGRPOTrainer(hidden_size=hidden_size, vocab_size=vocab_size)
        consumer = TTRLConsumer(TTRLConsumerConfig(rank=1))

        result = consumer.submit_grpo_training_batch(
            batch,
            trainer_bridge=trainer,
            lora_service=service,
            adapter_key="ttrl:cached-smoke",
            target_modules=("lm_head",),
            scope="request",
            request_ids=tuple(entry.request_id for entry in trace.entries),
        )

        self.assertEqual(len(trainer.requests), 1)
        self.assertEqual(result.updates[0].lora_int_id, 73)
        self.assertEqual(service.commit_staged_updates(), 1)
        self.assertEqual(service.adapter_for_request("req-ttrl-smoke-0"), 73)
        self.assertEqual(len(module.calls), 1)
        print(
            "TTRL cached-model smoke: "
            f"generated={len(trace.entries)}, "
            f"grpo_examples={int(result.metrics['grpo_examples'])}, "
            f"committed_lora_updates=1, "
            f"staged_module_calls={len(module.calls)}"
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Opt-in cached-model decode smoke for the OverRIDE sampler path."""

from __future__ import annotations

import os
import time
import unittest

import torch


@unittest.skipUnless(
    os.environ.get("TLLM_OVERRIDE_RUN_CACHED_MODEL_SMOKE") == "1" and torch.cuda.is_available(),
    "set TLLM_OVERRIDE_RUN_CACHED_MODEL_SMOKE=1 and provide CUDA to run cached-model vLLM smoke",
)
class OverRIDERealDecodeSmokeTest(unittest.TestCase):
    def test_cached_qwen_decode_updates_sampler_head_and_records_throughput(self) -> None:
        from vllm import SamplingParams

        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig
        from tllm.runtime import residual_runtime

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        model_name = os.environ.get("TLLM_OVERRIDE_CACHED_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        source_path = os.environ.get("TLLM_OVERRIDE_SOURCE_LAYER_PATH", "model.model.layers[0].input_layernorm")
        target_path = os.environ.get("TLLM_OVERRIDE_TARGET_LAYER_PATH", "model.model.layers[-1].input_layernorm")
        prompts = (
            "Write one word for a bright color:",
            "Write one word for a cold season:",
            "Write one word for a prime number:",
            "Write one word for a metal:",
        )
        max_tokens = int(os.environ.get("TLLM_OVERRIDE_SMOKE_MAX_TOKENS", "16"))
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=0,
                vocab_size=0,
                rank=4,
                num_iterations=3,
                lambd=0.8,
                learning_rate=1e-4,
            )
        )

        residual_runtime.clear_dispatch_consumers()
        residual_runtime.set_runtime_consumer(consumer)
        residual_runtime.configure_runtime(
            graph_scratch_rows=len(prompts),
            tap_layer_paths=(source_path,),
            source_layer_path=source_path,
            target_layer_path=target_path,
            enable_esamp_training=False,
            distiller_hidden_dim=8,
            distiller_lr=1e-4,
        )
        residual_runtime.install_runner_patch()
        llm = residual_runtime.make_llm(
            model_name,
            "bfloat16",
            float(os.environ.get("TLLM_OVERRIDE_SMOKE_GPU_MEMORY_UTILIZATION", "0.45")),
            128,
            False,
            True,
            123,
        )
        provider = consumer.sampler_modifier_provider()
        provider_counts = {"prepare": 0, "after": 0}
        original_prepare_step = provider.prepare_step
        original_after_sample = provider.after_sample

        def _prepare_step_with_count(view):
            provider_counts["prepare"] += 1
            return original_prepare_step(view)

        def _after_sample_with_count(view, sampled_token_ids):
            provider_counts["after"] += 1
            return original_after_sample(view, sampled_token_ids)

        provider.prepare_step = _prepare_step_with_count  # type: ignore[method-assign]
        provider.after_sample = _after_sample_with_count  # type: ignore[method-assign]
        try:
            consumer.config.enabled = False
            baseline_start = time.perf_counter()
            baseline_outputs = llm.generate(list(prompts), params, use_tqdm=False)
            torch.cuda.synchronize()
            baseline_elapsed = time.perf_counter() - baseline_start

            consumer.config.enabled = True
            enabled_start = time.perf_counter()
            enabled_outputs = llm.generate(list(prompts), params, use_tqdm=False)
            torch.cuda.synchronize()
            enabled_elapsed = time.perf_counter() - enabled_start

            baseline_tokens = sum(len(output.outputs[0].token_ids) for output in baseline_outputs)
            enabled_tokens = sum(len(output.outputs[0].token_ids) for output in enabled_outputs)
            baseline_tok_s = baseline_tokens / max(baseline_elapsed, 1e-9)
            enabled_tok_s = enabled_tokens / max(enabled_elapsed, 1e-9)
            ratio = enabled_tok_s / max(baseline_tok_s, 1e-9)
            min_ratio = float(os.environ.get("TLLM_OVERRIDE_SMOKE_MIN_THROUGHPUT_RATIO", "0.80"))
            stats = consumer.read_stats()
            print(
                "OverRIDE cached-model smoke: "
                f"baseline_tok_s={baseline_tok_s:.3f}, "
                f"enabled_tok_s={enabled_tok_s:.3f}, "
                f"ratio={ratio:.3f}, "
                f"loss_count={stats.loss_count}, "
                f"update_count={stats.update_count}, "
                f"provider_counts={provider_counts}"
            )

            self.assertEqual(len(enabled_outputs), len(prompts))
            self.assertGreater(
                provider_counts["prepare"],
                0,
                (
                    f"provider_counts={provider_counts}, "
                    f"source_resolved_path={residual_runtime.RUNTIME.source_resolved_path!r}, "
                    f"decode_count={residual_runtime.RUNTIME.decode_count}, "
                    f"tap_keys={tuple(residual_runtime.RUNTIME.tap_decode_hidden.keys())}"
                ),
            )
            self.assertGreater(provider_counts["after"], 0)
            self.assertGreater(stats.loss_count, 0, f"provider_counts={provider_counts}")
            self.assertGreater(stats.update_count, 0, f"provider_counts={provider_counts}")
            self.assertGreater(enabled_tokens, 0)
            self.assertGreaterEqual(
                ratio,
                min_ratio,
                (
                    f"OverRIDE enabled decode throughput ratio {ratio:.3f} below {min_ratio:.3f}; "
                    f"baseline_tok_s={baseline_tok_s:.3f}, enabled_tok_s={enabled_tok_s:.3f}, "
                    f"loss_count={stats.loss_count}, update_count={stats.update_count}"
                ),
            )
        finally:
            residual_runtime.set_runtime_consumer(None)
            residual_runtime.clear_dispatch_consumers()


if __name__ == "__main__":
    unittest.main()

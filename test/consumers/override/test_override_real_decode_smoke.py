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
        prompts = (
            "Write one word for a bright color:",
            "Write one word for a cold season:",
            "Write one word for a prime number:",
            "Write one word for a metal:",
        )
        params = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)

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
            target_layer_path=source_path,
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
            stats = consumer.read_stats()

            self.assertEqual(len(enabled_outputs), len(prompts))
            self.assertGreater(stats.loss_count, 0)
            self.assertGreater(stats.update_count, 0)
            self.assertGreater(enabled_tokens, 0)
            self.assertGreater(ratio, 0.0)
        finally:
            residual_runtime.set_runtime_consumer(None)
            residual_runtime.clear_dispatch_consumers()


if __name__ == "__main__":
    unittest.main()

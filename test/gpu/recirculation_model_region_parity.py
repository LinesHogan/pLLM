#!/usr/bin/env python3
"""Compare eager compiler-derived wavefront with its serial schedule oracle."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--gpu-mem", type=float, default=0.2)
    parser.add_argument("--max-logit-diff", type=float, default=0.25)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    from vllm import SamplingParams

    from tllm.consumers.recirculation import (
        RecirculationConfig,
        RecirculationConsumer,
    )
    from tllm.runtime.runtime import TllmRuntime
    from tllm.util.tools import shutdown_llm_instance

    runtime = TllmRuntime(
        consumers=[
            RecirculationConsumer(
                RecirculationConfig(
                    source_layer=8,
                    destination_layer=4,
                    alpha=0.02,
                    ramp_tokens=2,
                )
            )
        ]
    )
    llm = runtime.make_llm(
        model_name=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        max_num_seqs=1,
        max_num_batched_tokens=16,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    binding = runner._tllm_runner_binding
    model = runner.model
    original_compute_logits = model.compute_logits
    logits_records: list[torch.Tensor] = []

    def _record_logits(hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = original_compute_logits(hidden_states)
        if isinstance(logits, torch.Tensor):
            logits_records.append(logits.detach().float().cpu())
        return logits

    model.compute_logits = _record_logits
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    prompt = [{"prompt_token_ids": [10, 11, 12, 13]}]
    try:
        wavefront_output = llm.generate(prompt, params, use_tqdm=False)[0]
        wavefront_logits = logits_records[-1].clone()
        handle = binding.model_region_handles["compiled-recurrence"]
        wavefront_fused = int(handle.stats.fused_group_count)
        if wavefront_fused <= 0:
            print("FAIL: wavefront run did not execute an ordered-row group")
            return 1

        region = binding.capabilities.model_region
        assert region is not None
        binding.capabilities = replace(
            binding.capabilities,
            model_region=replace(region, ordered_row_fusion=None),
        )
        logits_records.clear()
        serial_output = llm.generate(prompt, params, use_tqdm=False)[0]
        serial_logits = logits_records[-1].clone()
        serial_fused = int(handle.stats.fused_group_count) - wavefront_fused

        difference = (wavefront_logits - serial_logits).abs()
        max_diff = float(difference.max())
        mean_diff = float(difference.mean())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                wavefront_logits, serial_logits, dim=-1
            ).min()
        )
        wavefront_tokens = wavefront_output.outputs[0].token_ids
        serial_tokens = serial_output.outputs[0].token_ids
        if wavefront_tokens != serial_tokens:
            print(
                "FAIL: wavefront and serial selected different tokens: "
                f"{wavefront_tokens} != {serial_tokens}"
            )
            return 1
        if serial_fused != 0:
            print(f"FAIL: serial oracle unexpectedly fused {serial_fused} groups")
            return 1
        if max_diff > float(args.max_logit_diff) or cosine < float(args.min_cosine):
            print(
                "FAIL: wavefront exceeded BF16 parity tolerance: "
                f"max={max_diff} mean={mean_diff} cosine={cosine}"
            )
            return 1
        print(
            "OK: Recirculation wavefront/serial parity passed "
            f"(runner={args.runner}, token={wavefront_tokens}, max={max_diff:.6f}, "
            f"mean={mean_diff:.6f}, cosine={cosine:.8f}, "
            f"wavefront_groups={wavefront_fused})"
        )
        return 0
    finally:
        model.compute_logits = original_compute_logits
        runtime.clear_consumers()
        shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    sys.exit(main())

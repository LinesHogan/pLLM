#!/usr/bin/env python3
"""GPU smoke for eager Recirculation model-region replay on vLLM 0.27."""

from __future__ import annotations

import argparse
import os
import sys


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--gpu-mem", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16)
    args = parser.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

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
        max_num_seqs=2,
        max_num_batched_tokens=int(args.max_num_batched_tokens),
    )
    try:
        outputs = llm.generate(
            [
                {"prompt_token_ids": [10, 11, 12]},
                {"prompt_token_ids": [20, 21, 22, 23]},
            ],
            SamplingParams(
                max_tokens=int(args.max_new_tokens),
                temperature=0.0,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        token_counts = [len(output.outputs[0].token_ids) for output in outputs]
        expected = [int(args.max_new_tokens)] * 2
        if token_counts != expected:
            print(f"FAIL: generation did not complete: {token_counts}")
            return 1

        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        actual_family = (
            "mrv2" if type(runner).__module__.endswith("gpu.model_runner") else "mrv1"
        )
        if actual_family != args.runner:
            print(f"FAIL: requested {args.runner}, got {type(runner).__module__}")
            return 1
        binding = getattr(runner, "_tllm_runner_binding", None)
        handles = getattr(binding, "model_region_handles", {})
        if tuple(handles) != ("compiled-recurrence",):
            print(f"FAIL: model-region handle was not installed: {tuple(handles)}")
            return 1
        handle = handles["compiled-recurrence"]
        stats = handle.stats
        output_request_ids = {str(output.request_id) for output in outputs}
        replay_ids = set(stats.request_replays)
        if not all(
            any(
                replay_id == request_id
                or replay_id.startswith(f"{request_id}-")
                for replay_id in replay_ids
            )
            for request_id in output_request_ids
        ):
            print(
                "FAIL: packed requests did not receive independent replay records: "
                f"outputs={output_request_ids} stats={stats.request_replays}"
            )
            return 1
        if (
            stats.normal_row_count <= 0
            or stats.recurrent_row_count <= 0
            or stats.fused_group_count <= 0
        ):
            print(f"FAIL: recurrence schedule did not execute: {stats}")
            return 1

        original_forward = handle.original_model_forward
        wrapped_forward = handle.inner_model.forward
        runtime.clear_consumers()
        if handles or not handle.restored:
            print("FAIL: disabling Recirculation left the handle installed")
            return 1
        if (
            handle.inner_model.forward != original_forward
            or handle.inner_model.forward == wrapped_forward
        ):
            print("FAIL: disabling Recirculation did not restore model.forward")
            return 1

        print(
            "OK: Recirculation eager model-region smoke passed "
            f"(runner={args.runner}, tokens={token_counts}, stats={stats})"
        )
        return 0
    finally:
        runtime.clear_consumers()
        shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""GPU gate for live mp=1 control changes and worker-side stats merging."""

from __future__ import annotations

import argparse
import os
import sys


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--gpu-mem", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

    import tllm
    from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
    from tllm.runtime import residual_runtime as core
    from tllm.util.tools import shutdown_llm_instance

    core.clear_dispatch_consumers()
    core.set_runtime_consumer(
        ESampConsumer(
            ESampConsumerConfig(
                graph_scratch_rows=8,
                distiller_hidden_dim=64,
                enable_esamp_training=False,
                adaptation_stream_mode="serial",
            )
        )
    )
    core.configure_runtime(
        graph_scratch_rows=8,
        tap_layer_paths=(
            "model.model.layers[0].input_layernorm",
            "model.model.layers[-1].input_layernorm",
        ),
        source_layer_path="model.model.layers[0].input_layernorm",
        target_layer_path="model.model.layers[-1].input_layernorm",
        adaptation_pipeline_slots=4,
        adaptation_stream_mode="serial",
        adaptation_stream_priority=0,
        compact_capture_lane=False,
    )
    llm = tllm.make_llm(
        model_name=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=512,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
    )
    try:
        from vllm import SamplingParams

        from tllm.runtime.mp.control import ControlPlane
        from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

        prompts = ["Summarize runtime control planes."] * 2
        params = SamplingParams(
            temperature=0.0,
            max_tokens=int(args.max_new_tokens),
            ignore_eos=True,
        )

        def generate() -> None:
            run_generate_with_request_mapping(
                runtime=core.RUNTIME,
                llm=llm,
                prompts=prompts,
                params=params,
            )

        generate()
        disabled = core.read_and_reset_esamp_stats(sync=True)
        if disabled.loss_count != 0:
            print(f"FAIL: disabled worker trained unexpectedly: {disabled}")
            return 1

        core.set_esamp_training_enabled(True)
        generate()
        core.synchronize_esamp()
        enabled = core.read_and_reset_esamp_stats(sync=True)
        if enabled.loss_count <= 0:
            print(f"FAIL: live worker enable did not produce training stats: {enabled}")
            return 1

        replies = ControlPlane(llm).ping(timeout=60.0)
        print(
            f"OK: mp runtime controls reached worker (runner={args.runner}, "
            f"loss_count={enabled.loss_count}, replies={replies})"
        )
        return 0
    finally:
        shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    sys.exit(main())

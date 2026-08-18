#!/usr/bin/env python3
"""GPU gate for placement-driven timing mobility and byte cross-checking."""

from __future__ import annotations

import argparse
import os
import sys


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SOURCE_PATH = "model.model.layers[0].input_layernorm"
TARGET_PATH = "model.model.layers[-1].input_layernorm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--mp", choices=("0", "1"), default="0")
    parser.add_argument("--gpu-mem", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    if args.mp == "1" and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork") == "spawn":
        print("FAIL: the ad-hoc cross-check probe is intentionally fork-only under mp=1")
        return 2
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = args.mp
    os.environ["TLLM_DEBUG_TIMING_MODE"] = "cross_check"

    import tllm
    from tllm.consumers.base import BaseConsumer
    from tllm.ports.base import ConsumerFlow, TimingContract
    from tllm.ports.request_meta import RequestMeta
    from tllm.ports.residual_stream import ResidualStream
    from tllm.runtime import residual_runtime as core
    from tllm.util.tools import shutdown_llm_instance

    class CrossCheckProbe(BaseConsumer[object]):
        config = object()
        consumer_id = "test.timing_cross_check"

        def flows(self):
            return (
                ConsumerFlow(
                    reads=(
                        ResidualStream.read(
                            layer=0,
                            site="block_output",
                            phase="decode",
                            role="source",
                        ),
                        RequestMeta.read(),
                    ),
                    writes=(),
                    window="same_step",
                    bundle_key=("engine_step_id", "phase"),
                    timing=TimingContract(
                        mobility="deferrable",
                        deadline="next_prepare",
                        max_defer_steps=1,
                        observes_version="at_declared_window",
                    ),
                ),
            )

        def consume_bundle(self, bundle, ctx) -> None:
            _ = (bundle, ctx)

    core.configure_runtime(
        graph_scratch_rows=8,
        tap_layer_paths=(SOURCE_PATH, TARGET_PATH),
        source_layer_path=SOURCE_PATH,
        target_layer_path=TARGET_PATH,
    )
    core.register_dispatch_consumer(CrossCheckProbe())
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

        from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

        prompts = ["Explain list scheduling in one sentence."] * 2
        params = SamplingParams(
            temperature=0.0,
            max_tokens=int(args.max_new_tokens),
            ignore_eos=True,
        )
        outputs = run_generate_with_request_mapping(
            runtime=core.RUNTIME,
            llm=llm,
            prompts=prompts,
            params=params,
        )
        core.synchronize_esamp()
        stats = core.read_and_reset_placement_stats()
        token_counts = [len(output.outputs[0].token_ids) for output in outputs]
        print(f"tokens={token_counts} placement={stats}")
        if not all(count == int(args.max_new_tokens) for count in token_counts):
            print("FAIL: generation ended before the requested token count")
            return 1
        if stats.moved_dispatch_count <= 0 or stats.enqueued_count <= 0:
            print("FAIL: compiled movable flow did not enqueue")
            return 1
        if stats.bubble_drain_count <= 0 or stats.max_queue_depth <= 0:
            print("FAIL: deferred work did not drain at the sampler bubble")
            return 1
        print(
            f"OK: timing cross-check passed (runner={args.runner}, mp={args.mp}, "
            f"moved={stats.moved_dispatch_count}, drained={stats.bubble_drain_count})"
        )
        return 0
    finally:
        shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    sys.exit(main())

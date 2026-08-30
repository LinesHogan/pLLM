#!/usr/bin/env python3
"""GPU correctness smoke for the native vLLM Recirculation workload."""

from __future__ import annotations

import argparse
import os
import sys


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--gpu-mem", type=float, default=0.15)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    args = parser.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import SamplingParams

    from tllm.consumers.recirculation import (
        PREFILL_BINDER_IMPLEMENTATION_ID,
        RecirculationConfig,
        RecirculationWorkload,
    )
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.runtime import TllmRuntime
    from tllm.schedule import RegionImplementationKind
    from tllm.util.tools import shutdown_llm_instance

    workload = RecirculationWorkload(
        RecirculationConfig(
            source_layer=8,
            destination_layer=4,
            alpha=0.02,
            beta=0.98,
            ramp_tokens=2,
        )
    )
    runtime = TllmRuntime(
        workloads=(workload,),
        workload_memory_budget_bytes=8 << 20,
    )
    if runtime.consumers:
        raise AssertionError("native Recirculation installed a legacy consumer")

    llm = runtime.make_llm(
        model_name=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        max_num_seqs=2,
        max_num_batched_tokens=16,
    )
    try:
        compilation = runtime.workload_compilation
        if compilation is None:
            raise AssertionError("native Recirculation did not compile a Region decision")
        regions = compilation.decision.execution_regions
        if len(regions) != 1:
            raise AssertionError(f"expected one native Recirculation Region: {regions!r}")
        region = regions[0]
        if (
            region.tasks != ("bind_prefill",)
            or region.implementation_kind is not RegionImplementationKind.EXTERN
            or region.implementation_ids != (PREFILL_BINDER_IMPLEMENTATION_ID,)
            or region.placement_anchor != "prepare.post"
        ):
            raise AssertionError(f"invalid native Recirculation Region: {region!r}")
        if core.RUNTIME.dispatch_plan is not None:
            raise AssertionError("native Recirculation retained a legacy DispatchPlan")

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
        expected = [int(args.max_new_tokens)] * 2
        token_counts = [len(output.outputs[0].token_ids) for output in outputs]
        if token_counts != expected:
            raise AssertionError(f"generation did not complete: {token_counts}")

        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        actual_family = (
            "mrv2" if type(runner).__module__.endswith("gpu.model_runner") else "mrv1"
        )
        if actual_family != args.runner:
            raise AssertionError(
                f"requested {args.runner}, got runner module {type(runner).__module__}"
            )
        binding = getattr(runner, "_tllm_runner_binding", None)
        handle = getattr(binding, "typed_layer_replay_handle", None)
        if handle is None:
            raise AssertionError("native model-region handle missing")
        stats = handle.stats
        output_request_ids = {str(output.request_id) for output in outputs}
        replay_ids = set(stats.request_replays)
        if (
            stats.prepare_count <= 0
            or stats.replay_count < 2
            or stats.replay_row_count <= 0
            or not all(
                any(
                    replay_id == request_id
                    or replay_id.startswith(f"{request_id}-")
                    for replay_id in replay_ids
                )
                for request_id in output_request_ids
            )
        ):
            raise AssertionError(f"native recurrence did not replay the prefill: {stats}")
        execution = core.RUNTIME.workload_execution
        if execution is None:
            raise AssertionError("native Recirculation execution is not active")
        physical = execution.executor.stats
        # Prefill itself produces the first sampled token; each additional
        # token contributes one decode model step.
        minimum_steps = max(1, int(args.max_new_tokens))
        if (
            physical.execute_count < minimum_steps
            or physical.eager_count < minimum_steps
            or physical.fallback_count != 0
            or physical.replay_count != 0
        ):
            raise AssertionError(
                "signed binder Region did not execute once per prefill/decode step: "
                f"{physical}"
            )

        print(
            "OK: native Recirculation ExecutionRegion smoke passed "
            f"(runner={actual_family}, tokens={token_counts}, "
            f"signature={compilation.decision.signature}, "
            f"model_region_stats={stats}, physical_stats={physical})"
        )
        return 0
    finally:
        runtime.clear_consumers()
        shutdown_llm_instance(llm, cooldown_s=1.0)
        core.publish_workload_compilation(None)


if __name__ == "__main__":
    sys.exit(main())

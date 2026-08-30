#!/usr/bin/env python3
"""GPU execution-parity smoke for the native Recirculation workload."""

from __future__ import annotations

import argparse
import json
import os


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EXPECTED_TOKENS = ((220, 4726), (220, 16))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--engine-core-mp", action="store_true")
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--gpu-mem", type=float, default=0.15)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = (
        "1" if args.runner == "mrv2" else "0"
    )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
        "1" if args.engine_core_mp else "0"
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from tllm.consumers.recirculation import (
        PREFILL_BINDER_IMPLEMENTATION_ID,
        RecirculationConfig,
        RecirculationWorkload,
    )
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.runtime.vllm_adapter import generate_requests
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
        model_name=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        max_num_seqs=2,
        max_num_batched_tokens=16,
    )
    completed = False
    try:
        compilation = runtime.workload_compilation
        if compilation is None or compilation.physical is None:
            raise AssertionError("native Recirculation has no signed physical plan")
        family = compilation.timeline.runner_family
        if family != args.runner:
            raise AssertionError(f"requested {args.runner}, got {family}")
        regions = compilation.decision.execution_regions
        if len(regions) != 1:
            raise AssertionError(
                f"expected one native Recirculation Region: {regions!r}"
            )
        region = regions[0]
        if (
            region.tasks != ("bind_prefill",)
            or region.implementation_kind is not RegionImplementationKind.EXTERN
            or region.implementation_ids != (PREFILL_BINDER_IMPLEMENTATION_ID,)
            or region.placement_anchor != "prepare.post"
        ):
            raise AssertionError(f"invalid native Recirculation Region: {region!r}")
        expected_artifacts = [
            {
                "region_id": region.region_id,
                "artifact_key": region.artifact_key,
            }
        ]

        prompts = (
            TokensPrompt(prompt_token_ids=[10, 11, 12]),
            TokensPrompt(prompt_token_ids=[20, 21, 22, 23]),
        )
        outputs = generate_requests(
            llm,
            prompts,
            SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=0.0,
                ignore_eos=True,
            ),
        )
        generated_tokens = [
            tuple(output.outputs[0].token_ids) for output in outputs
        ]
        token_counts = [len(tokens) for tokens in generated_tokens]
        if token_counts != [args.max_new_tokens, args.max_new_tokens]:
            raise AssertionError(f"generation did not complete: {token_counts}")
        if (
            args.model_name == MODEL
            and args.max_new_tokens == 2
            and generated_tokens != list(EXPECTED_TOKENS)
        ):
            raise AssertionError(
                "Recirculation deterministic token golden diverged: "
                f"{generated_tokens!r}"
            )

        plane = ControlPlane(llm)
        (status,) = plane.workload_status(timeout=60.0)
        if status["signature"] != compilation.decision.signature:
            raise AssertionError("Recirculation worker rewrote the signed decision")
        if status["plan_epoch"] != compilation.physical.plan_epoch:
            raise AssertionError("Recirculation worker rewrote the signed plan epoch")
        if status["artifacts"] != expected_artifacts:
            raise AssertionError(
                "Recirculation worker artifact report diverged from the signed plan: "
                f"expected={expected_artifacts!r} actual={status['artifacts']!r}"
            )
        if args.engine_core_mp and status["pid"] == os.getpid():
            raise AssertionError("EngineCore-MP Recirculation remained in the parent")
        if not args.engine_core_mp and status["pid"] != os.getpid():
            raise AssertionError("in-process Recirculation unexpectedly used a child")

        diagnostics = status["diagnostics"].get("recirculation")
        if diagnostics is None:
            raise AssertionError(
                f"Recirculation worker diagnostics are missing: {status!r}"
            )
        staged_steps = diagnostics["staged_step_count"]
        staged_requests = diagnostics["staged_request_count"]
        replay_requests = diagnostics["replay_request_count"]
        replay_rows = diagnostics["replay_row_count"]
        selected_rows = diagnostics["selected_row_count"]
        mixed_steps = diagnostics["mixed_step_count"]
        mixed_passthrough_rows = diagnostics["mixed_passthrough_row_count"]
        if (
            staged_steps <= 0
            or staged_requests < len(prompts)
            or replay_requests != staged_requests
            or selected_rows != replay_rows
            or replay_rows < sum(len(prompt["prompt_token_ids"]) for prompt in prompts)
        ):
            raise AssertionError(
                "native Recirculation did not execute every staged prefill replay: "
                f"{diagnostics!r}"
            )
        if (
            args.runner == "mrv1"
            and args.engine_core_mp
            and (mixed_steps <= 0 or mixed_passthrough_rows <= 0)
        ):
            raise AssertionError(
                "MRv1 EngineCore-MP did not exercise mixed phase selection: "
                f"{diagnostics!r}"
            )

        (worker_stats,) = plane.pull_stats(
            "workgraph", sync=False, timeout=60.0
        )
        physical = worker_stats.workgraph
        if (
            physical.execute_count <= 0
            or physical.eager_count <= 0
            or physical.fallback_count != 0
            or physical.replay_count != 0
        ):
            raise AssertionError(
                "Recirculation left its signed eager physical path: "
                f"{physical!r}"
            )

        print(
            json.dumps(
                {
                    "runner": family,
                    "engine_core_mp": args.engine_core_mp,
                    "signature": compilation.decision.signature,
                    "plan_epoch": compilation.physical.plan_epoch,
                    "artifacts": expected_artifacts,
                    "worker_pid": status["pid"],
                    "diagnostics": diagnostics,
                    "execute_count": physical.execute_count,
                    "eager_count": physical.eager_count,
                    "fallback_count": physical.fallback_count,
                    "physical_replay_count": physical.replay_count,
                    "tokens": generated_tokens,
                },
                sort_keys=True,
            )
        )
        completed = True
        return 0
    finally:
        try:
            if completed:
                runtime.clear_consumers()
        finally:
            shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    raise SystemExit(main())

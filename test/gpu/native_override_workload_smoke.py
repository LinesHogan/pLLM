#!/usr/bin/env python3
"""Native OverRIDE train/reweight GPU evidence across runner topologies."""

from __future__ import annotations

import argparse
import json
import os


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_DEFAULT_TRAINING_TOKENS: dict[
    str, tuple[tuple[tuple[int, ...], ...], ...]
] = {
    "mrv1": (
        ((11730, 16, 3837), (24, 16, 17)),
        ((5373, 16, 16), (5373, 17, 16)),
        ((16, 59, 31716), (24, 15, 10)),
    ),
    "mrv2": (
        ((48272, 101889, 69041), (24, 198, 220)),
        ((92, 10829, 198), (271, 2, 18614)),
        ((1648, 10829, 10829), (24, 15, 198)),
    ),
}
_DEFAULT_FINAL_TOKENS: dict[str, tuple[tuple[int, ...], ...]] = {
    "mrv1": ((27866, 15, 15), (24, 15, 15)),
    "mrv2": ((99661, 9370, 82587), (198, 33, 13)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--engine-core-mp", action="store_true")
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--gpu-mem", type=float, default=0.20)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    args = parser.parse_args()
    if args.max_new_tokens <= 1:
        raise ValueError("--max-new-tokens must be greater than one")

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

    from tllm.consumers.override import OverRIDEConfig, OverRIDEWorkload
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.util.tools import shutdown_llm_instance

    workload = OverRIDEWorkload(
        OverRIDEConfig(
            rank=2,
            num_iterations=4,
            lambd=0.8,
            learning_rate=0.05,
        )
    )
    runtime = TllmRuntime(
        workloads=(workload,),
        workload_memory_budget_bytes=256 << 20,
    )
    if runtime.consumers:
        raise AssertionError("native OverRIDE installed a legacy consumer")

    llm = runtime.make_llm(
        model_name=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
    )
    completed = False
    try:
        compilation = runtime.workload_compilation
        if compilation is None or compilation.physical is None:
            raise AssertionError("native OverRIDE has no signed physical plan")
        family = compilation.timeline.runner_family
        if family != args.runner:
            raise AssertionError(f"requested {args.runner}, got {family}")
        expected_artifacts = [
            {
                "region_id": region.region_id,
                "artifact_key": region.artifact_key,
            }
            for region in compilation.decision.execution_regions
        ]
        prompts = (
            TokensPrompt(prompt_token_ids=[10, 11, 12]),
            TokensPrompt(prompt_token_ids=[20, 21, 22, 23]),
        )
        plane = ControlPlane(llm)
        iteration_tokens: list[list[tuple[int, ...]]] = []
        iteration_diagnostics: list[dict[str, int | float]] = []
        iteration_versions: list[int] = []

        for sample_idx in range(3):
            outputs = runtime.generate_requests(
                llm,
                prompts,
                SamplingParams(
                    temperature=0.8,
                    top_p=0.95,
                    top_k=-1,
                    seed=2000 + sample_idx,
                    max_tokens=args.max_new_tokens,
                    ignore_eos=True,
                ),
                prompt_indices=(0, 1),
                sample_indices=(sample_idx, sample_idx),
            )
            tokens = [
                tuple(output.outputs[0].token_ids) for output in outputs
            ]
            if [len(value) for value in tokens] != [
                args.max_new_tokens,
                args.max_new_tokens,
            ]:
                raise AssertionError(
                    f"OverRIDE iteration {sample_idx} did not finish: {tokens!r}"
                )
            iteration_tokens.append(tokens)
            (iteration_status,) = plane.workload_status(timeout=60.0)
            iteration_diagnostic = iteration_status["diagnostics"].get(
                "override"
            )
            if iteration_diagnostic is None:
                raise AssertionError(
                    "OverRIDE worker diagnostics are missing after iteration "
                    f"{sample_idx}: {iteration_status!r}"
                )
            iteration_diagnostics.append(dict(iteration_diagnostic))
            iteration_versions.append(
                next(
                    (
                        entry["version"]
                        for entry in iteration_status["state_versions"]
                        if entry["resource"] == "heads" and entry["key"] is None
                    ),
                    -1,
                )
            )

        (before_final,) = plane.workload_status(timeout=60.0)
        before_diagnostics = before_final["diagnostics"].get("override")
        if before_diagnostics is None:
            raise AssertionError(
                f"OverRIDE worker diagnostics are missing: {before_final!r}"
            )
        trained_version = next(
            (
                entry["version"]
                for entry in before_final["state_versions"]
                if entry["resource"] == "heads" and entry["key"] is None
            ),
            -1,
        )
        if (
            before_diagnostics["loss_count"] <= 0
            or before_diagnostics["update_count"] <= 0
            or trained_version != before_diagnostics["update_count"]
        ):
            raise AssertionError(
                "OverRIDE training did not advance signed state: "
                f"status={before_final!r}"
            )
        expected_versions = [
            (sample_idx + 1) * (args.max_new_tokens - 1)
            for sample_idx in range(3)
        ]
        if iteration_versions != expected_versions or any(
            diagnostic["loss_count"] != expected
            or diagnostic["update_count"] != expected
            for diagnostic, expected in zip(
                iteration_diagnostics, expected_versions, strict=True
            )
        ):
            raise AssertionError(
                "OverRIDE public batches did not share one admission boundary: "
                f"versions={iteration_versions!r} "
                f"diagnostics={iteration_diagnostics!r}"
            )

        final_outputs = runtime.generate_requests(
            llm,
            prompts,
            SamplingParams(
                temperature=0.8,
                top_p=0.95,
                top_k=-1,
                seed=2003,
                max_tokens=args.max_new_tokens,
                ignore_eos=True,
            ),
            prompt_indices=(0, 1),
            sample_indices=(3, 3),
        )
        final_tokens = [
            tuple(output.outputs[0].token_ids) for output in final_outputs
        ]
        if [len(value) for value in final_tokens] != [
            args.max_new_tokens,
            args.max_new_tokens,
        ]:
            raise AssertionError(
                f"OverRIDE final iteration did not finish: {final_tokens!r}"
            )
        if args.model_name == MODEL and args.max_new_tokens == 3:
            frozen_training = tuple(
                tuple(iteration) for iteration in iteration_tokens
            )
            frozen_final = tuple(final_tokens)
            if (
                frozen_training != _DEFAULT_TRAINING_TOKENS[family]
                or frozen_final != _DEFAULT_FINAL_TOKENS[family]
            ):
                raise AssertionError(
                    "OverRIDE fixed-RNG token golden diverged: "
                    f"training={frozen_training!r} final={frozen_final!r}"
                )

        (status,) = plane.workload_status(timeout=60.0)
        diagnostics = status["diagnostics"].get("override")
        if diagnostics is None:
            raise AssertionError(f"OverRIDE diagnostics disappeared: {status!r}")
        final_version = next(
            (
                entry["version"]
                for entry in status["state_versions"]
                if entry["resource"] == "heads" and entry["key"] is None
            ),
            -1,
        )
        if (
            final_version != trained_version
            or diagnostics["update_count"] != before_diagnostics["update_count"]
            or diagnostics["loss_count"] != before_diagnostics["loss_count"]
        ):
            raise AssertionError(
                "OverRIDE final iteration performed an optimizer transition: "
                f"before={before_final!r} after={status!r}"
            )
        if diagnostics.get("logit_delta_rows", 0) <= 0:
            raise AssertionError(
                f"OverRIDE sampler delta was not published: {diagnostics!r}"
            )
        if status["signature"] != compilation.decision.signature:
            raise AssertionError("OverRIDE worker rewrote the signed decision")
        if status["plan_epoch"] != compilation.physical.plan_epoch:
            raise AssertionError("OverRIDE worker rewrote the signed epoch")
        if status["artifacts"] != expected_artifacts:
            raise AssertionError(
                "OverRIDE worker artifacts diverged: "
                f"expected={expected_artifacts!r} actual={status['artifacts']!r}"
            )
        if args.engine_core_mp and status["pid"] == os.getpid():
            raise AssertionError("EngineCore-MP OverRIDE remained in the parent")
        if not args.engine_core_mp and status["pid"] != os.getpid():
            raise AssertionError("in-process OverRIDE unexpectedly used a child")
        mapping_sizes = [
            int(reply["mapping_size"])
            for reply in plane.ping(timeout=60.0)
        ]
        if any(mapping_sizes):
            raise AssertionError(
                "OverRIDE retained finished request mappings: "
                f"{mapping_sizes!r}"
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
                f"OverRIDE left its signed eager path: {physical!r}"
            )

        print(
            json.dumps(
                {
                    "runner": family,
                    "engine_core_mp": args.engine_core_mp,
                    "signature": compilation.decision.signature,
                    "plan_epoch": compilation.physical.plan_epoch,
                    "worker_pid": status["pid"],
                    "state_version": final_version,
                    "diagnostics": diagnostics,
                    "execute_count": physical.execute_count,
                    "eager_count": physical.eager_count,
                    "fallback_count": physical.fallback_count,
                    "mapping_sizes": mapping_sizes,
                    "training_tokens": iteration_tokens,
                    "iteration_diagnostics": iteration_diagnostics,
                    "iteration_versions": iteration_versions,
                    "final_tokens": final_tokens,
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

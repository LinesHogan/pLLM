#!/usr/bin/env python3
"""GPU correctness smoke for the native ESamp workload path.

This deliberately constructs ``ESampWorkload`` through ``TllmRuntime``.
It must never install a legacy/program consumer: the assertions below inspect
the signed Region decision and the materialized local execution artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from math import isfinite


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--gpu-mem", type=float, default=0.35)
    return parser.parse_args()


def _token_counts(outputs: Sequence[object]) -> list[int]:
    return [len(getattr(output, "outputs")[0].token_ids) for output in outputs]


def _runner_family(llm: object) -> str:
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    return "mrv2" if type(runner).__module__.endswith("gpu.model_runner") else "mrv1"


def _active_esamp_diagnostics(raw: object) -> dict[str, int | float]:
    """Validate observational metrics from the active native Region instance."""

    if not isinstance(raw, Mapping):
        raise AssertionError(f"active resource diagnostics are not a mapping: {raw!r}")
    payload = raw.get("esamp")
    if not isinstance(payload, Mapping):
        raise AssertionError(
            "active native ESamp Region did not report namespaced diagnostics: "
            f"{raw!r}"
        )
    metrics: dict[str, int | float] = {}
    for name, value in payload.items():
        if type(name) is not str or not name:
            raise AssertionError(f"invalid ESamp diagnostic name: {payload!r}")
        if type(value) not in (int, float) or (
            type(value) is float and not isfinite(value)
        ):
            raise AssertionError(f"invalid ESamp diagnostic value: {payload!r}")
        metrics[name] = value
    for name in ("guidance_rows", "loss_count", "update_count"):
        value = metrics.get(name)
        if type(value) is not int or value <= 0:
            raise AssertionError(
                "active native ESamp Region did not train: "
                f"diagnostics={metrics!r}"
            )
    return metrics


def _active_distiller_version(execution: object) -> int:
    versions = getattr(execution, "state_versions", None)
    if versions is None:
        raise AssertionError("active ESamp execution has no signed state-version registry")
    version = versions.observed_version("distiller", None)
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise AssertionError(f"active ESamp execution reported invalid state version: {version!r}")
    return version


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--batch-size and --max-new-tokens must be positive")

    # These must be set before vLLM imports its runner configuration.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    # Keep the native artifact and state-version evidence inspectable here.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.util.tools import shutdown_llm_instance
    from tllm.workgraph.physical import CapturePolicy

    workload = ESampWorkload(
        ESampConfig(
            enabled=True,
            intervention=True,
            distiller_hidden_dim=8,
            distiller_lr=1e-3,
            distiller_beta=0.5,
        )
    )
    runtime = TllmRuntime(
        workloads=(workload,),
        workload_memory_budget_bytes=64 << 20,
    )
    if runtime.consumers or hasattr(workload, "legacy_consumer"):
        raise AssertionError("native ESamp smoke must not install a legacy adapter")

    llm = runtime.make_llm(
        model_name=str(args.model_name),
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
        max_num_seqs=int(args.batch_size),
        max_num_batched_tokens=16,
    )
    try:
        actual_runner = _runner_family(llm)
        if actual_runner != args.runner:
            raise AssertionError(
                f"requested runner {args.runner!r}, got {actual_runner!r}"
            )

        compilation = runtime.workload_compilation
        execution = core.RUNTIME.workload_execution
        if compilation is None:
            raise AssertionError("native ESamp Region plan was not compiled")
        if core.RUNTIME.consumer is not None or runtime.consumers:
            raise AssertionError("native ESamp Region silently installed a legacy consumer")

        decision = compilation.decision
        regions = decision.execution_regions
        expected_artifacts = tuple(
            (region.region_id, region.artifact_key) for region in regions
        )
        signed_tasks = tuple(
            task for region in regions for task in region.tasks
        )
        if set(signed_tasks) != {
            "predict",
            "loss",
            "loss__optimize_distiller",
        } or len(signed_tasks) != 3:
            raise AssertionError(f"unexpected signed ESamp tasks: {signed_tasks!r}")
        store = core.RUNTIME.workload_execution_store
        installed = [execution]
        if store is not None:
            from tllm.runtime.workload_execution import StagedExecution

            installed = [store.current]
            match store.staged:
                case StagedExecution(execution=staged_execution):
                    installed.append(staged_execution)
                case _:
                    pass
        if not any(
            candidate.artifact_report == expected_artifacts
            for candidate in installed
        ):
            raise AssertionError(
                "no materialized ESamp artifact report matches the signed decision: "
                f"expected={expected_artifacts!r}, "
                f"installed={tuple(candidate.artifact_report for candidate in installed)!r}"
            )
        if store is not None:
            match store.staged:
                case StagedExecution(execution=staged_execution, activated=False) if staged_execution.artifact_report == expected_artifacts:
                    raise AssertionError(
                        "coordinated ESamp epoch was staged but not activated"
                    )
                case _:
                    pass
        if len(decision.signature) != 64 or not all(
            character in "0123456789abcdef" for character in decision.signature
        ):
            raise AssertionError("ESamp Region decision does not carry a SHA-256 signature")
        capture_safe_tasks = {
            task_name
            for region in regions
            if region.capture_safe
            for task_name in region.tasks
        }
        if capture_safe_tasks != {"predict"} or any(
            node.capture_policy is not CapturePolicy.DISABLED
            for node in compilation.physical.nodes
        ):
            raise AssertionError(
                "eager ESamp must disable capture while retaining only "
                f"prediction capture eligibility: {capture_safe_tasks!r}"
            )

        params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=32,
            seed=7,
            max_tokens=int(args.max_new_tokens),
            ignore_eos=True,
        )
        prompts = tuple(
            TokensPrompt(
                prompt_token_ids=[10 + index, 20 + index, 30 + index]
            )
            for index in range(int(args.batch_size))
        )
        _ = core.read_and_reset_distiller_timing_stats()
        prompt_indices = list(range(int(args.batch_size)))
        sample_indices = [0] * int(args.batch_size)
        first = runtime.generate_requests(
            llm,
            prompts,
            params,
            prompt_indices=prompt_indices,
            sample_indices=sample_indices,
        )
        execution = core.RUNTIME.workload_execution
        if execution is None or execution.artifact_report != expected_artifacts:
            raise AssertionError("first ESamp request did not activate the signed Region epoch")
        first_versions = _active_distiller_version(execution)
        second = runtime.generate_requests(
            llm,
            prompts,
            params,
            prompt_indices=prompt_indices,
            sample_indices=sample_indices,
        )
        execution = core.RUNTIME.workload_execution
        if execution is None or execution.artifact_report != expected_artifacts:
            raise AssertionError("second ESamp request replaced the signed Region epoch")
        final_versions = _active_distiller_version(execution)
        diagnostics = _active_esamp_diagnostics(
            execution.read_resource_diagnostics()
        )
        timing = core.read_and_reset_distiller_timing_stats()
        training = core.read_and_reset_esamp_stats(sync=True)

        expected_counts = [int(args.max_new_tokens)] * int(args.batch_size)
        if _token_counts(first) != expected_counts or _token_counts(second) != expected_counts:
            raise AssertionError("native ESamp generation did not complete")
        if first_versions <= 0 or final_versions <= first_versions:
            raise AssertionError(
                "ESamp state version did not monotonically advance across requests: "
                f"first={first_versions} final={final_versions}"
            )
        if training.loss_count <= 0:
            raise AssertionError(
                "native ESamp Region did not train: "
                f"loss_count={training.loss_count} diagnostics={diagnostics!r}"
            )
        observed = execution.state_versions
        if observed is None or observed.observed_version("distiller", None) != final_versions:
            raise AssertionError("signed state-version registry diverged from ESamp state")
        if diagnostics["guidance_rows"] <= 0:
            raise AssertionError(
                "native ESamp Region did not publish same-step guidance: "
                f"guidance_rows={diagnostics['guidance_rows']!r}, state_version={final_versions}, "
                f"diagnostics={diagnostics!r}, timing={timing!r}"
            )
        mapping_sizes = [
            int(reply["mapping_size"])
            for reply in ControlPlane(llm).ping(timeout=60.0)
        ]
        if any(mapping_sizes):
            raise AssertionError(
                f"native ESamp retained finished request mappings: {mapping_sizes!r}"
            )
        # The Region sampler provider returns a modifier only after requiring
        # exact equality of guidance request IDs, prompt IDs, sample IDs,
        # phase, and step ID with the current sampler view.  The matching
        # rejection path is covered in the CPU Region unit test above.
        if int(getattr(timing, "intervention_state_count", 0)) <= 0:
            raise AssertionError(
                "sampler did not consume native Region guidance: "
                f"timing={timing!r}"
            )

        stats = execution.executor.stats
        if (
            stats.execute_count <= 0
            or stats.eager_count <= 0
            or stats.fallback_count != 0
            or stats.replay_count != 0
        ):
            raise AssertionError(
                "native ESamp execution violated Region fallback/capture contract: "
                f"{stats!r}"
            )
        print(
            json.dumps(
                {
                    "runner": actual_runner,
                    "signature": decision.signature,
                    "artifacts": execution.artifact_report,
                    "loss_count": training.loss_count,
                    "update_count": diagnostics["update_count"],
                    "guidance_rows": diagnostics["guidance_rows"],
                    "state_version": final_versions,
                    "intervention_state_count": int(
                        getattr(timing, "intervention_state_count", 0)
                    ),
                    "execute_count": stats.execute_count,
                    "eager_count": stats.eager_count,
                    "fallback_count": stats.fallback_count,
                    "replay_count": stats.replay_count,
                    "mapping_sizes": mapping_sizes,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        runtime.clear_consumers()
        shutdown_llm_instance(llm, cooldown_s=1.0)
        core.clear_dispatch_consumers()


if __name__ == "__main__":
    sys.exit(main())

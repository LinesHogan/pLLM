#!/usr/bin/env python3
"""GPU correctness smoke for native ESamp workloads under mp=1.

This is intentionally a correctness-only, single-engine test.  The parent
constructs a native workload, workers reconstruct the signed Region artifacts
through the control plane, and all execution evidence is read back through
plain RPC payloads rather than child-process Python objects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, cast


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    return parser.parse_args()


def _token_counts(outputs: Sequence[object]) -> list[int]:
    return [len(getattr(output, "outputs")[0].token_ids) for output in outputs]


def _active_esamp_diagnostics(
    status: Mapping[str, object], *, require_activity: bool
) -> dict[str, int | float]:
    """Validate the closed worker diagnostic payload for the active Region."""

    raw_diagnostics = status.get("diagnostics")
    if type(raw_diagnostics) is not dict:
        raise AssertionError(f"worker diagnostics are not a JSON object: {status!r}")
    raw_metrics = raw_diagnostics.get("esamp")
    if type(raw_metrics) is not dict:
        raise AssertionError(
            "worker did not report namespaced ESamp Region diagnostics: "
            f"{status!r}"
        )
    metrics: dict[str, int | float] = {}
    for name, value in raw_metrics.items():
        if type(name) is not str or not name:
            raise AssertionError(f"worker returned an invalid diagnostic name: {status!r}")
        if type(value) not in (int, float) or (
            type(value) is float and not isfinite(value)
        ):
            raise AssertionError(f"worker returned an invalid diagnostic value: {status!r}")
        metrics[name] = value
    for name in ("guidance_rows", "loss_count", "update_count"):
        value = metrics.get(name)
        if type(value) is not int:
            raise AssertionError(
                f"worker ESamp diagnostics omitted integer {name!r}: {status!r}"
            )
        if require_activity and value <= 0:
            raise AssertionError(
                "active worker ESamp Region did not train: "
                f"diagnostics={metrics!r} status={status!r}"
            )
    return metrics


def _worker_statuses(
    statuses: object,
    *,
    parent_pid: int,
    signature: str,
    plan_epoch: int,
    artifacts: tuple[tuple[str, str], ...],
    require_state_version: bool,
) -> list[Mapping[str, object]]:
    if not isinstance(statuses, list) or not statuses:
        raise AssertionError(f"workload_status returned no worker reports: {statuses!r}")
    normalized: list[Mapping[str, object]] = []
    expected_artifacts = [
        {"region_id": region_id, "artifact_key": artifact_key}
        for region_id, artifact_key in artifacts
    ]

    def worker_pid(status: Mapping[str, object]) -> int:
        pid = status.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise AssertionError(f"worker returned an invalid PID: {status!r}")
        return pid

    def has_distiller_version(versions: list[object]) -> bool:
        for raw_entry in versions:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = cast(Mapping[str, object], raw_entry)
            version = entry.get("version")
            if (
                entry.get("resource") == "distiller"
                and entry.get("key") is None
                and isinstance(version, int)
                and not isinstance(version, bool)
                and version > 0
            ):
                return True
        return False

    for raw_status in statuses:
        if not isinstance(raw_status, Mapping):
            raise AssertionError(f"workload_status reply is not a mapping: {raw_status!r}")
        status = cast(Mapping[str, object], raw_status)
        if worker_pid(status) == parent_pid:
            raise AssertionError(f"workload_status reached the parent, not a worker: {status!r}")
        if status.get("signature") != signature or status.get("plan_epoch") != plan_epoch:
            raise AssertionError(f"worker rewrote the signed Region epoch: {status!r}")
        if status.get("artifacts") != expected_artifacts:
            raise AssertionError(
                "worker artifact report differs from the signed Region decision: "
                f"expected={expected_artifacts!r} actual={status.get('artifacts')!r}"
            )
        versions = status.get("state_versions")
        if not isinstance(versions, list) or (
            require_state_version and not has_distiller_version(versions)
        ):
            raise AssertionError(
                "worker did not report a positive observed distiller state version: "
                f"{status!r}"
            )
        _active_esamp_diagnostics(
            status,
            require_activity=require_state_version,
        )
        normalized.append(status)
    return normalized


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--batch-size and --max-new-tokens must be positive")

    # Set before importing vLLM: the model runner and EngineCore topology are
    # fixed during vLLM configuration construction.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.util.tools import shutdown_llm_instance

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
        workloads=(workload,), workload_memory_budget_bytes=64 << 20
    )
    if runtime.consumers or hasattr(workload, "legacy_consumer"):
        raise AssertionError("native MP ESamp smoke must not install a legacy adapter")

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
        compilation = runtime.workload_compilation
        if compilation is None:
            raise AssertionError("parent did not compile a native ESamp Region plan")
        physical_plan = compilation.physical
        if physical_plan is None:
            raise AssertionError("native ESamp compilation has no physical Region plan")
        expected_artifacts = tuple(
            (region.region_id, region.artifact_key)
            for region in compilation.decision.execution_regions
        )
        expected_epoch = physical_plan.plan_epoch
        signed_tasks = tuple(
            task
            for region in compilation.decision.execution_regions
            for task in region.tasks
        )
        if set(signed_tasks) != {
            "predict",
            "loss",
            "loss__optimize_distiller",
        } or len(signed_tasks) != 3:
            raise AssertionError(
                f"unexpected signed native ESamp tasks: {signed_tasks!r}"
            )

        plane = ControlPlane(llm)

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
        prompt_indices = list(range(int(args.batch_size)))
        sample_indices = [0] * int(args.batch_size)
        _ = core.read_and_reset_distiller_timing_stats()
        first = runtime.generate_requests(
            llm,
            prompts,
            params,
            prompt_indices=prompt_indices,
            sample_indices=sample_indices,
        )
        second = runtime.generate_requests(
            llm,
            prompts,
            params,
            prompt_indices=prompt_indices,
            sample_indices=sample_indices,
        )
        expected_counts = [int(args.max_new_tokens)] * int(args.batch_size)
        if _token_counts(first) != expected_counts or _token_counts(second) != expected_counts:
            raise AssertionError("native MP ESamp generation did not complete")

        mapping_sizes = [
            int(reply["mapping_size"])
            for reply in plane.ping(timeout=60.0)
        ]
        if any(mapping_sizes):
            raise AssertionError(
                "native MP ESamp retained finished request mappings: "
                f"{mapping_sizes!r}"
            )

        # State versions advance during the two generate calls, so inspect the
        # worker only after execution rather than only at cold installation.
        statuses = _worker_statuses(
            plane.workload_status(),
            parent_pid=os.getpid(),
            signature=compilation.decision.signature,
            plan_epoch=expected_epoch,
            artifacts=expected_artifacts,
            require_state_version=True,
        )
        worker_diagnostics = [
            _active_esamp_diagnostics(status, require_activity=True)
            for status in statuses
        ]
        training = core.read_and_reset_esamp_stats(sync=True)
        timing = core.read_and_reset_distiller_timing_stats(sync=True)
        physical = core.read_workgraph_runtime_stats()
        if training.loss_count <= 0:
            raise AssertionError(f"parent did not merge worker ESamp losses: {training!r}")
        intervention_count = int(getattr(timing, "intervention_state_count", 0))
        if intervention_count <= 0:
            raise AssertionError(f"worker sampler intervention did not engage: {timing!r}")
        if (
            physical.execute_count <= 0
            or physical.eager_count <= 0
            or physical.fallback_count != 0
            or physical.replay_count != 0
        ):
            raise AssertionError(
                "native MP ESamp violated the signed non-capture execution contract: "
                f"{physical!r}"
            )
        print(
            json.dumps(
                {
                    "runner": args.runner,
                    "signature": compilation.decision.signature,
                    "plan_epoch": expected_epoch,
                    "artifacts": expected_artifacts,
                    "worker_pids": [
                        cast(int, status["pid"]) for status in statuses
                    ],
                    "loss_count": training.loss_count,
                    "worker_diagnostics": worker_diagnostics,
                    "intervention_state_count": intervention_count,
                    "execute_count": physical.execute_count,
                    "eager_count": physical.eager_count,
                    "fallback_count": physical.fallback_count,
                    "replay_count": physical.replay_count,
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

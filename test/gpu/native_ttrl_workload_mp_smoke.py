#!/usr/bin/env python3
"""GPU correctness smoke for the native TTRL workload under mp=1.

The parent stages a synthetic hidden/reward batch through the typed port
transaction. The EngineCore worker must reconstruct the workload, receive the
shared safetensors payload, train on the next signed step, and report a nonzero
loss/update count and reward-model version.
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


def _worker_diagnostics(status: Mapping[str, object]) -> dict[str, int | float]:
    raw_diagnostics = status.get("diagnostics")
    if type(raw_diagnostics) is not dict:
        raise AssertionError(f"worker diagnostics are not a JSON object: {status!r}")
    raw_metrics = raw_diagnostics.get("ttrl")
    if type(raw_metrics) is not dict:
        raise AssertionError(
            f"worker did not report namespaced TTRL diagnostics: {status!r}"
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
    for name in ("loss_count", "update_count"):
        value = metrics.get(name)
        if type(value) is not int:
            raise AssertionError(
                f"worker TTRL diagnostics omitted integer {name!r}: {status!r}"
            )
        if value < 1:
            raise AssertionError(
                "worker TTRL did not train from the staged batch: "
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
) -> list[Mapping[str, object]]:
    if not isinstance(statuses, list) or not statuses:
        raise AssertionError(f"workload_status returned no worker reports: {statuses!r}")
    normalized: list[Mapping[str, object]] = []
    expected_artifacts = [
        {"region_id": region_id, "artifact_key": artifact_key}
        for region_id, artifact_key in artifacts
    ]
    for raw_status in statuses:
        if not isinstance(raw_status, Mapping):
            raise AssertionError(f"workload_status reply is not a mapping: {raw_status!r}")
        status = cast(Mapping[str, object], raw_status)
        pid = status.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise AssertionError(f"worker returned an invalid PID: {status!r}")
        if pid == parent_pid:
            raise AssertionError(f"workload_status reached the parent, not a worker: {status!r}")
        if status.get("signature") != signature or status.get("plan_epoch") != plan_epoch:
            raise AssertionError(f"worker rewrote the signed Region epoch: {status!r}")
        if status.get("artifacts") != expected_artifacts:
            raise AssertionError(
                "worker artifact report differs from the signed Region decision: "
                f"expected={expected_artifacts!r} actual={status.get('artifacts')!r}"
            )
        versions = status.get("state_versions")
        if not isinstance(versions, list):
            raise AssertionError(f"worker did not report state versions: {status!r}")
        reward_versions: list[object] = []
        for raw_entry in versions:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = cast(Mapping[str, object], raw_entry)
            if entry.get("resource") == "reward_model":
                reward_versions.append(entry.get("version"))
        if reward_versions != [1]:
            raise AssertionError(
                "worker TTRL reward-model version did not advance exactly once: "
                f"{status!r}"
            )
        _worker_diagnostics(status)
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
    import torch

    from tllm.consumers.ttrl.workload import TTRLConfig, TTRLWorkload
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.util.tools import shutdown_llm_instance

    workload = TTRLWorkload(
        TTRLConfig(
            rank=2,
            learning_rate=1e-3,
        )
    )
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=64 << 20)
    if runtime.consumers:
        raise AssertionError("native MP TTRL smoke must not install a legacy consumer")

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
            raise AssertionError("parent did not compile a native TTRL Region plan")
        physical_plan = compilation.physical
        if physical_plan is None:
            raise AssertionError("native TTRL compilation has no physical Region plan")
        expected_artifacts = tuple(
            (region.region_id, region.artifact_key)
            for region in compilation.decision.execution_regions
        )
        expected_epoch = physical_plan.plan_epoch
        if len(expected_artifacts) != 3:
            raise AssertionError(
                f"unexpected native TTRL Region partition: {expected_artifacts!r}"
            )

        plane = ControlPlane(llm)

        params = SamplingParams(
            temperature=0.0,
            max_tokens=int(args.max_new_tokens),
            ignore_eos=True,
        )
        prompts = [
            {"prompt_token_ids": [10 + index, 20 + index, 30 + index]}
            for index in range(int(args.batch_size))
        ]
        first = llm.generate(prompts, params, use_tqdm=False)
        spec = workload.specialization
        runtime.stage(
            TTRLWorkload.train_hidden,
            torch.randn(
                int(args.batch_size),
                spec.hidden_size,
                device=spec.device,
                dtype=spec.dtype,
            ),
        )
        runtime.stage(
            TTRLWorkload.train_rewards,
            torch.linspace(
                0.0,
                1.0,
                int(args.batch_size),
                device=spec.device,
                dtype=torch.float32,
            ),
        )
        second = llm.generate(prompts, params, use_tqdm=False)
        expected_counts = [int(args.max_new_tokens)] * int(args.batch_size)
        if _token_counts(first) != expected_counts or _token_counts(second) != expected_counts:
            raise AssertionError("native MP TTRL generation did not complete")

        statuses = _worker_statuses(
            plane.workload_status(),
            parent_pid=os.getpid(),
            signature=compilation.decision.signature,
            plan_epoch=expected_epoch,
            artifacts=expected_artifacts,
        )
        worker_diagnostics = [_worker_diagnostics(status) for status in statuses]

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
                    "worker_diagnostics": worker_diagnostics,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        runtime.clear_consumers()
        shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    sys.exit(main())

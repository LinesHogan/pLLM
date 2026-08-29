#!/usr/bin/env python3
"""GPU correctness smoke for the native TTRL workload path.

The closed loop uses typed staged ports and one typed host output. This smoke
drives real engine steps with plain ``llm.generate`` and asserts training,
state-version advancement, published control intent, and the executor's
non-capture contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from math import isfinite

import torch


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_HAPPY_TEXTS = (
    "reasoning ... \\boxed{4}",
    "different path ... \\boxed{4}",
    "minority ... \\boxed{5}",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--gpu-mem", type=float, default=0.3)
    return parser.parse_args()


def _ttrl_diagnostics(execution: object) -> dict[str, int | float]:
    raw = execution.read_resource_diagnostics()
    if not isinstance(raw, Mapping):
        raise AssertionError(f"resource diagnostics are not a mapping: {raw!r}")
    payload = raw.get("ttrl")
    if not isinstance(payload, Mapping):
        raise AssertionError(
            f"native TTRL did not report namespaced diagnostics: {raw!r}"
        )
    metrics: dict[str, int | float] = {}
    for name, value in payload.items():
        if type(name) is not str or not name:
            raise AssertionError(f"invalid TTRL diagnostic name: {payload!r}")
        if type(value) not in (int, float) or (
            type(value) is float and not isfinite(value)
        ):
            raise AssertionError(f"invalid TTRL diagnostic value: {payload!r}")
        metrics[name] = value
    return metrics


def _observed_version(execution: object) -> int:
    versions = getattr(execution, "state_versions", None)
    if versions is None:
        raise AssertionError("native TTRL execution has no state-version registry")
    version = versions.observed_version("reward_model", None)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise AssertionError(f"invalid reward_model state version: {version!r}")
    return version


def main() -> int:
    args = _parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    # The host output is process-local; keep the engine in-process so it can be
    # drained directly after the signed publish task.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import SamplingParams

    from tllm.consumers.ttrl.workload import (
        TTRLConfig,
        TTRLWorkload,
        prepare_ttrl_training_batch,
    )
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.generation_trace_runtime import (
        CompletionTraceEntry,
        GenerationTraceBatch,
    )
    from tllm.runtime.runtime import TllmRuntime
    from tllm.schedule import RegionImplementationKind
    from tllm.util.tools import shutdown_llm_instance

    workload = TTRLWorkload(
        TTRLConfig(
            rank=2,
            learning_rate=1e-3,
        )
    )
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=64 << 20)
    if runtime.consumers:
        raise AssertionError("native TTRL smoke must not install a legacy consumer")

    llm = runtime.make_llm(
        model_name=str(args.model_name),
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
        max_num_seqs=2,
        max_num_batched_tokens=16,
    )
    try:
        compilation = runtime.workload_compilation
        if compilation is None:
            raise AssertionError("native TTRL Region plan was not compiled")
        if core.RUNTIME.consumer is not None:
            raise AssertionError("native TTRL Region silently installed a legacy consumer")

        regions = compilation.decision.execution_regions
        if len(regions) != 3 or tuple(region.tasks for region in regions) != (
            ("train",),
            ("train__optimize_reward_model",),
            ("publish",),
        ):
            raise AssertionError(f"unexpected TTRL Region partition: {regions!r}")
        if tuple(region.implementation_kind for region in regions) != (
            RegionImplementationKind.COMPILED,
            RegionImplementationKind.EXTERN,
            RegionImplementationKind.EXTERN,
        ):
            raise AssertionError(f"unexpected TTRL Region kinds: {regions!r}")
        if any(region.capture_safe for region in regions):
            raise AssertionError("stateful/extern TTRL Regions must be non-capture")
        if len(compilation.decision.signature) != 64:
            raise AssertionError("TTRL decision does not carry a SHA-256 signature")

        params = SamplingParams(
            temperature=0.0, max_tokens=int(args.max_new_tokens), ignore_eos=True
        )
        prompts = [
            {"prompt_token_ids": [10, 11, 12]},
            {"prompt_token_ids": [20, 21, 22]},
        ]

        # Step 1: nothing staged -> zero-completion steps, no version bump.
        first = llm.generate(prompts, params, use_tqdm=False)
        execution = core.RUNTIME.workload_execution
        if execution is None:
            raise AssertionError("native TTRL decision was not bound to an executor")
        if _observed_version(execution) != 0:
            raise AssertionError(
                "TTRL bumped the reward-model version without a staged batch"
            )

        # Stage one closed-loop batch (host API) and publish target, then step.
        active_workload = runtime.local_workload(workload)
        spec = active_workload.specialization
        trace = GenerationTraceBatch(
            entries=tuple(
                CompletionTraceEntry(
                    request_id=f"req-{index}",
                    prompt_idx=0,
                    sample_idx=index,
                    output_index=0,
                    token_ids=(1, 2),
                    text=text,
                    finish_reason="stop",
                    group_id="g0",
                )
                for index, text in enumerate(_HAPPY_TEXTS)
            )
        )
        hidden_rows = torch.randn(3, spec.hidden_size, dtype=spec.dtype).to(spec.device)
        batch = prepare_ttrl_training_batch(trace, hidden_rows)
        if batch.intent.training_row_count != 3:
            raise AssertionError(f"unexpected staged TTRL intent: {batch.intent!r}")
        runtime.stage(TTRLWorkload.train_hidden, batch.hidden)
        runtime.stage(TTRLWorkload.train_rewards, batch.rewards)
        runtime.stage(TTRLWorkload.pending_intent, batch.intent)

        second = llm.generate(prompts, params, use_tqdm=False)
        execution = core.RUNTIME.workload_execution
        assert execution is not None
        version = _observed_version(execution)
        if version != 1:
            raise AssertionError(
                "TTRL reward-model version must advance exactly once per staged "
                f"batch, got {version}"
            )

        for outputs in (first, second):
            counts = [len(output.outputs[0].token_ids) for output in outputs]
            if counts != [int(args.max_new_tokens)] * 2:
                raise AssertionError(f"generation did not complete: {counts}")

        published_outputs = runtime.drain(TTRLWorkload.control)
        if len(published_outputs) != 1:
            raise AssertionError(
                f"TTRL publish task produced {len(published_outputs)} controls"
            )
        published = published_outputs[0]
        if published.adapter_version_id != "ttrl:update:1":
            raise AssertionError(f"TTRL publish stamped a stale version: {published!r}")

        diagnostics = _ttrl_diagnostics(execution)
        if diagnostics.get("update_count", 0) <= 0:
            raise AssertionError(
                f"native TTRL did not train: diagnostics={diagnostics!r}"
            )
        physical = execution.read_workgraph_runtime_stats()
        if (
            physical.execute_count <= 0
            or physical.eager_count <= 0
            or physical.fallback_count != 0
            or physical.replay_count != 0
        ):
            raise AssertionError(
                f"native TTRL violated the signed non-capture contract: {physical!r}"
            )

        print(
            json.dumps(
                {
                    "runner": args.runner,
                    "signature": compilation.decision.signature,
                    "state_version": version,
                    "adapter_version_id": published.adapter_version_id,
                    "reward_rows": len(batch.rewards),
                    "diagnostics": diagnostics,
                    "execute_count": physical.execute_count,
                    "eager_count": physical.eager_count,
                    "fallback_count": physical.fallback_count,
                    "replay_count": physical.replay_count,
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

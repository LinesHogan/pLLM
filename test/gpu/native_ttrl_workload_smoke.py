#!/usr/bin/env python3
"""Native TTRL closed-loop GPU evidence across runner topologies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os

import torch


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_DEFAULT_TOKENS = (16, 11, 15)


@dataclass(frozen=True, slots=True)
class _SmokeVerifier:
    judge_name: str = "gpu_marker_exact"
    missing_reason: str = "missing-gpu-marker"

    def label_for_completion(self, completion: str) -> str:
        return "minority" if completion.endswith("[minority]") else "majority"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--engine-core-mp", action="store_true")
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
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

    from tllm.consumers.ttrl import (
        TTRLConfig,
        TTRLWorkload,
        prepare_ttrl_training_batch,
        stage_ttrl_training,
    )
    from tllm.runtime.generation_trace_runtime import (
        CompletionTraceEntry,
        GenerationTraceBatch,
    )
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.util.tools import shutdown_llm_instance

    workload = TTRLWorkload(TTRLConfig(rank=2, learning_rate=1e-3))
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=64 << 20
    )
    if runtime.consumers:
        raise AssertionError("native TTRL installed a legacy consumer")

    llm = runtime.make_llm(
        model_name=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
        max_num_seqs=3,
        max_num_batched_tokens=16,
    )
    completed = False
    try:
        compilation = runtime.workload_compilation
        if compilation is None or compilation.physical is None:
            raise AssertionError("native TTRL has no signed physical plan")
        if compilation.timeline.runner_family != args.runner:
            raise AssertionError(
                f"requested {args.runner}, got {compilation.timeline.runner_family}"
            )
        signed_tasks = tuple(
            task
            for region in compilation.decision.execution_regions
            for task in region.tasks
        )
        if set(signed_tasks) != {
            "prepare",
            "train",
            "train__optimize_reward_model",
            "publish",
        } or len(signed_tasks) != 4:
            raise AssertionError(f"unexpected signed TTRL tasks: {signed_tasks!r}")
        expected_artifacts = [
            {
                "region_id": region.region_id,
                "artifact_key": region.artifact_key,
            }
            for region in compilation.decision.execution_regions
        ]

        prompts = (
            TokensPrompt(prompt_token_ids=[10, 11, 12]),
            TokensPrompt(prompt_token_ids=[10, 11, 12]),
            TokensPrompt(prompt_token_ids=[10, 11, 12]),
        )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
        )
        first = runtime.generate_requests(
            llm,
            prompts,
            params,
            prompt_indices=(0, 0, 0),
            sample_indices=(0, 1, 2),
        )
        if [len(output.outputs[0].token_ids) for output in first] != [
            args.max_new_tokens
        ] * 3:
            raise AssertionError("TTRL source generation did not complete")
        source_tokens = [
            tuple(output.outputs[0].token_ids) for output in first
        ]

        trace_entries: list[CompletionTraceEntry] = []
        for index, output in enumerate(first):
            candidate = output.outputs[0]
            text = candidate.text
            if index == 2:
                text = f"{text}[minority]"
            trace_entries.append(
                CompletionTraceEntry(
                    request_id=output.request_id,
                    prompt_idx=0,
                    sample_idx=index,
                    output_index=0,
                    token_ids=tuple(candidate.token_ids),
                    text=text,
                    finish_reason=candidate.finish_reason or "",
                    group_id="gpu-closed-loop",
                )
            )
        active = runtime.local_workload(workload)
        spec = active.specialization
        batch = prepare_ttrl_training_batch(
            GenerationTraceBatch(entries=tuple(trace_entries)),
            torch.randn(
                3,
                spec.hidden_size,
                device=spec.device,
                dtype=spec.dtype,
            ),
            consumer_id=active.config.consumer_id,
            verifier=_SmokeVerifier(),
        )
        transaction = stage_ttrl_training(runtime, batch)

        second = runtime.generate_requests(
            llm,
            prompts,
            params,
            prompt_indices=(0, 0, 0),
            sample_indices=(0, 1, 2),
        )
        if [len(output.outputs[0].token_ids) for output in second] != [
            args.max_new_tokens
        ] * 3:
            raise AssertionError("TTRL training generation did not complete")
        training_tokens = [
            tuple(output.outputs[0].token_ids) for output in second
        ]
        published = transaction.commit()
        if published.adapter_version_id != "ttrl:update:1":
            raise AssertionError(f"TTRL published a stale version: {published!r}")
        if tuple(
            entry.reward for entry in published.reward_outcomes.entries
        ) != (1.0, 1.0, 0.0):
            raise AssertionError(f"TTRL reward oracle diverged: {published!r}")
        if tuple(
            entry.action for entry in published.retry_controls.entries
        ) != ("accept", "accept", "retry"):
            raise AssertionError(f"TTRL retry controls diverged: {published!r}")
        if (
            len(published.update_barriers.entries) != 1
            or published.update_barriers.entries[0].state != "ready"
            or published.update_barriers.entries[0].ready_count != 3
        ):
            raise AssertionError(f"TTRL update barrier diverged: {published!r}")
        if (
            len(published.retry_schedule) != 1
            or published.retry_schedule[0].adapter_version_id
            != "ttrl:update:1"
            or published.retry_schedule[0].output_index != 0
        ):
            raise AssertionError(f"TTRL retry schedule diverged: {published!r}")

        retry = published.retry_schedule[0]
        retried = runtime.generate_requests(
            llm,
            (prompts[0],),
            params,
            prompt_indices=(retry.prompt_idx,),
            sample_indices=(retry.sample_idx,),
        )
        if len(retried[0].outputs[0].token_ids) != args.max_new_tokens:
            raise AssertionError("TTRL scheduled retry did not complete")
        retry_tokens = tuple(retried[0].outputs[0].token_ids)
        if args.model_name == MODEL and args.max_new_tokens == 3 and (
            source_tokens != [_DEFAULT_TOKENS] * 3
            or training_tokens != [_DEFAULT_TOKENS] * 3
            or retry_tokens != _DEFAULT_TOKENS
        ):
            raise AssertionError(
                "TTRL fixed-token golden diverged: "
                f"source={source_tokens!r} training={training_tokens!r} "
                f"retry={retry_tokens!r}"
            )

        plane = ControlPlane(llm)
        (status,) = plane.workload_status(timeout=60.0)
        if status["signature"] != compilation.decision.signature:
            raise AssertionError("TTRL worker rewrote the signed decision")
        if status["plan_epoch"] != compilation.physical.plan_epoch:
            raise AssertionError("TTRL worker rewrote the signed plan epoch")
        if status["artifacts"] != expected_artifacts:
            raise AssertionError("TTRL worker artifact identity diverged")
        if args.engine_core_mp and status["pid"] == os.getpid():
            raise AssertionError("EngineCore-MP TTRL remained in the parent")
        if not args.engine_core_mp and status["pid"] != os.getpid():
            raise AssertionError("in-process TTRL unexpectedly used a child")
        reward_versions = [
            entry["version"]
            for entry in status["state_versions"]
            if entry["resource"] == "reward_model" and entry["key"] is None
        ]
        if reward_versions != [1]:
            raise AssertionError(f"TTRL state version diverged: {status!r}")
        diagnostics = status["diagnostics"].get("ttrl")
        if diagnostics is None or (
            diagnostics.get("loss_count") != 1
            or diagnostics.get("update_count") != 1
        ):
            raise AssertionError(f"TTRL did not train exactly once: {status!r}")
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
                f"TTRL left its signed eager path: {physical!r}"
            )

        print(
            json.dumps(
                {
                    "runner": args.runner,
                    "engine_core_mp": args.engine_core_mp,
                    "signature": compilation.decision.signature,
                    "plan_epoch": compilation.physical.plan_epoch,
                    "worker_pid": status["pid"],
                    "state_version": reward_versions[0],
                    "adapter_version_id": published.adapter_version_id,
                    "rewards": [
                        entry.reward
                        for entry in published.reward_outcomes.entries
                    ],
                    "retry_count": len(published.retry_schedule),
                    "source_tokens": source_tokens,
                    "training_tokens": training_tokens,
                    "retry_tokens": retry_tokens,
                    "diagnostics": diagnostics,
                    "execute_count": physical.execute_count,
                    "eager_count": physical.eager_count,
                    "fallback_count": physical.fallback_count,
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

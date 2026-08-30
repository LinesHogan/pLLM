#!/usr/bin/env python3
"""Typed FTTT feedback -> signed GPU update -> closed-loop output smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), default="mrv2")
    parser.add_argument("--engine-core-mp", action="store_true")
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--gpu-mem", type=float, default=0.35)
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

    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from tllm.consumers.fttt import (
        FTTTConfig,
        FTTTFeedbackService,
        FTTTInstanceState,
        FTTTWorkload,
    )
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.generation_trace_runtime import (
        CompletionTraceEntry,
        GenerationTraceBatch,
    )
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.runtime import TllmRuntime
    from tllm.runtime.vllm_adapter import generate_requests
    from tllm.util.tools import shutdown_llm_instance

    config = FTTTConfig(
        rank=2,
        learning_rate=1e-2,
        feedback_feature_size=8,
        max_retries=2,
    )
    workload = FTTTWorkload(config)
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=64 << 20
    )
    if runtime.consumers:
        raise AssertionError("native FTTT smoke installed a legacy consumer")

    llm = runtime.make_llm(
        model_name=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
    )
    try:
        compilation = runtime.workload_compilation
        if compilation is None or compilation.physical is None:
            raise AssertionError("native FTTT workload has no signed physical plan")
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
        if not expected_artifacts:
            raise AssertionError("native FTTT signed plan has no execution Regions")

        service = FTTTFeedbackService(config)
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="feedback-0",
                    prompt_idx=7,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(17, 18),
                    text="wrong arithmetic result",
                    finish_reason="stop",
                    group_id="round-0",
                ),
            )
        )
        spec = runtime.local_workload(workload).specialization
        hidden_rows = torch.linspace(
            -1.0,
            1.0,
            steps=spec.hidden_size,
            device=spec.device,
            dtype=spec.dtype,
        ).reshape(1, spec.hidden_size)
        if not service.enqueue(trace, hidden_rows):
            raise AssertionError("negative FTTT feedback did not enqueue training")
        transaction = service.stage_next(runtime)
        if transaction is None:
            raise AssertionError("FTTT service did not stage its typed transaction")

        params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            ignore_eos=True,
        )
        prompts = (TokensPrompt(prompt_token_ids=[1, 2, 3]),)
        with transaction:
            first = generate_requests(llm, prompts, params)
            intent = transaction.commit()

        token_counts = [len(output.outputs[0].token_ids) for output in first]
        if token_counts != [args.max_new_tokens]:
            raise AssertionError(f"native FTTT generation did not complete: {token_counts}")
        if intent.adapter_version != 1:
            raise AssertionError(f"FTTT published a stale adapter version: {intent!r}")
        if service.instance_state(7, request_id="feedback-0") != FTTTInstanceState(
            retries=1,
            stopped=False,
            adapter_version=1,
        ):
            raise AssertionError("FTTT service did not acknowledge signed lifecycle state")

        (delta,) = runtime.drain(FTTTWorkload.edit_delta)
        (reward_outcomes,) = runtime.drain(FTTTWorkload.reward_outcomes)
        (retry_controls,) = runtime.drain(FTTTWorkload.retry_controls)
        (update_barriers,) = runtime.drain(FTTTWorkload.update_barriers)
        if (
            delta.shape != (1, spec.hidden_size)
            or not torch.isfinite(delta).all().item()
            or delta.abs().max().item() == 0.0
        ):
            raise AssertionError(f"FTTT signed edit prediction is empty: {delta!r}")
        if reward_outcomes != intent.reward_outcomes:
            raise AssertionError("FTTT reward output diverged from closed-loop intent")
        if retry_controls != intent.retry_controls:
            raise AssertionError("FTTT retry output diverged from closed-loop intent")
        if update_barriers != intent.update_barriers:
            raise AssertionError("FTTT barrier output diverged from closed-loop intent")
        retry = retry_controls.entries[0]
        barrier = update_barriers.entries[0]
        if (
            retry.action != "retry"
            or retry.reason != "feedback-update"
            or not retry.retry_after_update
        ):
            raise AssertionError(f"FTTT retry intent is invalid: {retry!r}")
        if (
            barrier.state != "ready"
            or barrier.ready_count != 1
            or barrier.request_ids != ("feedback-0",)
        ):
            raise AssertionError(f"FTTT update barrier is invalid: {barrier!r}")

        drained_ports = (
            runtime.drain(FTTTWorkload.edit_delta),
            runtime.drain(FTTTWorkload.reward_outcomes),
            runtime.drain(FTTTWorkload.retry_controls),
            runtime.drain(FTTTWorkload.update_barriers),
            runtime.drain(FTTTWorkload.closed_loop),
        )
        if (
            drained_ports != ((), (), (), (), ())
            or service.stage_next(runtime) is not None
        ):
            raise AssertionError("FTTT transaction ports were not drained exactly once")

        if not service.enqueue(trace, hidden_rows):
            raise AssertionError("second negative FTTT feedback did not enqueue")
        second_transaction = service.stage_next(runtime)
        if second_transaction is None:
            raise AssertionError("FTTT service did not stage its second transaction")
        with second_transaction:
            second = generate_requests(llm, prompts, params)
            second_intent = second_transaction.commit()
        second_counts = [len(output.outputs[0].token_ids) for output in second]
        if second_counts != [args.max_new_tokens]:
            raise AssertionError(
                f"second FTTT generation did not complete: {second_counts}"
            )
        if second_intent.adapter_version != 2:
            raise AssertionError(
                f"FTTT second publication is stale: {second_intent!r}"
            )
        if service.instance_state(7, request_id="feedback-0") != FTTTInstanceState(
            retries=2,
            stopped=True,
            adapter_version=2,
        ):
            raise AssertionError("FTTT retry-limit lifecycle did not settle")
        (second_delta,) = runtime.drain(FTTTWorkload.edit_delta)
        runtime.drain(FTTTWorkload.reward_outcomes)
        (second_retry_controls,) = runtime.drain(FTTTWorkload.retry_controls)
        (second_update_barriers,) = runtime.drain(FTTTWorkload.update_barriers)
        final_retry = second_retry_controls.entries[0]
        final_barrier = second_update_barriers.entries[0]
        if (
            second_delta.shape != (1, spec.hidden_size)
            or final_retry.action != "stop"
            or final_retry.reason != "retry-limit"
            or final_retry.retry_after_update
            or final_barrier.state != "ready"
        ):
            raise AssertionError(
                "FTTT second signed lifecycle is invalid: "
                f"delta={second_delta!r} retry={final_retry!r} "
                f"barrier={final_barrier!r}"
            )

        third = generate_requests(llm, prompts, params)
        third_counts = [len(output.outputs[0].token_ids) for output in third]
        if third_counts != [args.max_new_tokens]:
            raise AssertionError(
                f"post-transaction FTTT generation did not complete: {third_counts}"
            )
        post_transaction_outputs = (
            runtime.drain(FTTTWorkload.edit_delta),
            runtime.drain(FTTTWorkload.reward_outcomes),
            runtime.drain(FTTTWorkload.retry_controls),
            runtime.drain(FTTTWorkload.update_barriers),
            runtime.drain(FTTTWorkload.closed_loop),
        )
        if post_transaction_outputs != ((), (), (), (), ()):
            raise AssertionError(
                "an unstaged FTTT step republished consumed transaction outputs"
            )

        plane = ControlPlane(llm)
        (status,) = plane.workload_status(timeout=60.0)
        if status["signature"] != compilation.decision.signature:
            raise AssertionError("FTTT worker rewrote the signed decision")
        if status["plan_epoch"] != compilation.physical.plan_epoch:
            raise AssertionError("FTTT worker rewrote the signed plan epoch")
        if status["artifacts"] != expected_artifacts:
            raise AssertionError(
                "FTTT worker artifact report diverged from the signed plan: "
                f"expected={expected_artifacts!r} actual={status['artifacts']!r}"
            )
        if args.engine_core_mp and status["pid"] == os.getpid():
            raise AssertionError("EngineCore-MP FTTT execution remained in the parent")
        if not args.engine_core_mp and status["pid"] != os.getpid():
            raise AssertionError("in-process FTTT execution unexpectedly used a child")

        edit_versions = [
            entry["version"]
            for entry in status["state_versions"]
            if entry["resource"] == "edit_adapter" and entry["key"] is None
        ]
        if edit_versions != [2]:
            raise AssertionError(
                f"FTTT adapter state did not advance twice: {status!r}"
            )
        diagnostics = status["diagnostics"].get("fttt")
        if diagnostics is None:
            raise AssertionError(f"FTTT worker diagnostics are missing: {status!r}")
        if diagnostics.get("loss_count", 0) <= 0 or diagnostics.get(
            "update_count", 0
        ) <= 0:
            raise AssertionError(f"FTTT signed training did not run: {diagnostics!r}")
        if diagnostics["loss_count"] != 2 or diagnostics["update_count"] != 2:
            raise AssertionError(
                "drained FTTT input ports were executed more than once: "
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
                "FTTT workload did not stay on its signed eager execution path: "
                f"{physical!r}"
            )

        print(
            json.dumps(
                {
                    "runner": family,
                    "engine_core_mp": args.engine_core_mp,
                    "signature": compilation.decision.signature,
                    "plan_epoch": compilation.physical.plan_epoch,
                    "worker_pid": status["pid"],
                    "adapter_version": second_intent.adapter_version,
                    "delta_max": float(second_delta.abs().max().item()),
                    "retry_action": final_retry.action,
                    "barrier_state": final_barrier.state,
                    "diagnostics": diagnostics,
                    "execute_count": physical.execute_count,
                    "eager_count": physical.eager_count,
                    "fallback_count": physical.fallback_count,
                    "tokens": (token_counts, second_counts, third_counts),
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

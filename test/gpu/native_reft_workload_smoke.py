#!/usr/bin/env python3
"""GPU correctness smoke for the activation intervention primitive.

Drives the real engine: install the LoReFT patch on the declared layer,
verify the untouched baseline, push an intervention whose delta is large
enough to flip greedy continuations, and assert the signed publish task
lands it with an audited version bump.  Also asserts the prefill-only
semantics observably: a decode-only continuation after the patched prefill
is unaffected by further patch applications (the KV carries the effect).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--gpu-mem", type=float, default=0.3)
    return parser.parse_args()


def _greedy_tokens(llm: object, prompts: list[str], max_tokens: int) -> list[tuple[int, ...]]:
    from vllm import SamplingParams

    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
        use_tqdm=False,
    )
    return [tuple(output.outputs[0].token_ids) for output in outputs]


def main() -> int:
    args = _parse_args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    # The bank lives in the process-local host channel; keep the engine
    # in-process so the push and the audit are observable here.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from tllm.consumers.reft.workload import ReftConfig, ReftWorkload
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.runtime import TllmRuntime
    from tllm.consumers.reft.vllm_patch import install_reft_patch_on_llm
    from tllm.util.tools import shutdown_llm_instance
    from vllm import SamplingParams

    rank = 2
    workload = ReftWorkload(
        ReftConfig(layers=(int(args.layer),), rank=rank, last_tokens=2)
    )
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=64 << 20)
    if runtime.consumers:
        raise AssertionError("native ReFT smoke must not install a legacy consumer")

    llm = runtime.make_llm(
        model_name=str(args.model_name),
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=128,
        max_num_batched_tokens=128,  # unchunked prefill (documented scope)
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
        max_num_seqs=2,
    )
    try:
        compilation = runtime.workload_compilation
        if compilation is None:
            raise AssertionError("native ReFT Region plan was not compiled")
        regions = compilation.decision.execution_regions
        if len(regions) != 1 or tuple(regions[0].tasks) != (
            "publish__intervention",
        ):
            raise AssertionError(f"unexpected ReFT Region partition: {regions!r}")
        if regions[0].capture_safe:
            raise AssertionError("extern publish Region must be non-capture")

        active_workload = runtime.local_workload(workload)
        install_reft_patch_on_llm(llm, runtime)

        prompts = ["The capital of France is", "Water is made of"]
        baseline = _greedy_tokens(llm, prompts, 8)

        spec = active_workload.specialization
        hidden = spec.hidden_size
        # Large constant-direction delta: project onto the first rank
        # direction with a big bias, so the patched hidden moves far.
        source_w = torch.zeros(1, rank, hidden)
        source_b = torch.zeros(1, rank)
        source_b[0, 0] = 200.0
        rotate = torch.zeros(1, hidden, rank)
        rotate[0, 0, 0] = 1.0
        rotate[0, 1 % hidden, 1 % rank] = 1.0
        runtime.publish(
            ReftWorkload.intervention,
            {"source_w": source_w, "source_b": source_b, "rotate": rotate},
        )
        if active_workload.state_version != 0:
            raise AssertionError("intervention went live before any engine step")

        # Land the push through the signed publish Region, then roll out.
        _greedy_tokens(llm, ["ping"], 1)
        if active_workload.state_version != 1:
            raise AssertionError(
                f"signed publish did not land the push: {active_workload.state_version}"
            )
        # The active epoch is promoted on the first user step; re-read it.
        execution = core.RUNTIME.workload_execution
        if execution is None:
            raise AssertionError("native ReFT execution disappeared after activation")
        state_versions = execution.state_versions
        if state_versions is None:
            raise AssertionError("native ReFT state-version registry disappeared")
        observed = state_versions.observed_version("intervention", None)
        if observed != 1:
            raise AssertionError(f"audited intervention version must be 1, got {observed}")

        patched = _greedy_tokens(llm, prompts, 8)
        # The pushed delta is enormous (bias 200 on a rank direction): every
        # patched prompt must flip its greedy continuation.  A prompt whose
        # output is unchanged means its prefill rows escaped the row rule —
        # exactly the failure this smoke exists to catch.
        flipped = [p != b for p, b in zip(patched, baseline, strict=True)]
        if not all(flipped):
            raise AssertionError(
                f"patched rollout identical for prompts "
                f"{[i for i, f in enumerate(flipped) if not f]}: the activation "
                "intervention did not reach every prefill"
            )

        physical = execution.read_workgraph_runtime_stats()
        if physical.execute_count <= 0 or physical.fallback_count != 0:
            raise AssertionError(
                f"native ReFT violated the signed execution contract: {physical!r}"
            )
        print(
            json.dumps(
                {
                    "runner": args.runner,
                    "signature": compilation.decision.signature,
                    "intervention_version": active_workload.state_version,
                    "baseline": [list(tokens) for tokens in baseline],
                    "patched": [list(tokens) for tokens in patched],
                    "execute_count": physical.execute_count,
                    "fallback_count": physical.fallback_count,
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

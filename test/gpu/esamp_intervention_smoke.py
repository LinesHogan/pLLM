#!/usr/bin/env python3
"""GPU smoke: ESamp distiller intervention on the v0.27 MRv1/MRv2 runners.

Trains the distiller for one warmup round, then generates with
``enable_distiller_intervention=True`` and asserts the intervention path
actually engaged (provider published candidate state on at least one decode
step) and generation completed.
"""

from __future__ import annotations

import argparse
import json

from tllm.runtime import residual_runtime as core
from tllm.workflows import esamp_support as esamp_workflow_support
from tllm.util.tools import build_prompt_batch, read_prompts, shutdown_llm_instance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--sampler-backend", type=str, default="post_filter_exact")
    parser.add_argument("--implementation", type=str, default="legacy", choices=["legacy", "program"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    prompts = build_prompt_batch(
        read_prompts("", ["Give me a short introduction to large language models."]),
        int(args.batch_size),
    )

    esamp_workflow_support.configure_esamp_runtime(
        implementation=str(args.implementation),
        graph_scratch_rows=max(64, len(prompts)),
        enable_esamp_training=True,
        enable_distiller_intervention=True,
        distiller_beta=float(args.beta),
        distiller_sampler_backend=str(args.sampler_backend),
    )

    llm = core.make_llm(
        model_name=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
    )

    try:
        from vllm import SamplingParams

        # Warmup round: train the distiller so intervention has candidate state.
        # NOTE: greedy warmup with no top-k/top-p makes the exact post-filter
        # backend treat the whole vocab as candidates (per-row projection
        # scales with vocab) — bound the candidate set explicitly.
        warmup = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            seed=7,
            max_tokens=int(args.max_new_tokens),
            ignore_eos=True,
        )
        llm.generate(prompts, warmup)

        core.read_and_reset_distiller_timing_stats()

        params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            seed=42,
            max_tokens=int(args.max_new_tokens),
            ignore_eos=True,
        )
        outputs = llm.generate(prompts, params)
        tokens = [list(o.outputs[0].token_ids) for o in outputs]

        stats = core.read_and_reset_distiller_timing_stats()
        esamp_stats = core.read_and_reset_esamp_stats()
    finally:
        shutdown_llm_instance(llm, cooldown_s=1.0)

    attempts = int(getattr(stats, "port_publish_attempt_count", 0))
    hits = int(getattr(stats, "port_publish_hit_count", 0))
    payload = {
        "tokens": tokens,
        "port_publish_attempts": attempts,
        "port_publish_hits": hits,
        "esamp_stats": {
            "loss_avg": float(getattr(esamp_stats, "loss_avg", 0.0)),
            "loss_count": int(getattr(esamp_stats, "loss_count", 0)),
        },
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "tokens"}, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    if str(args.implementation) == "program":
        applied = int(getattr(stats, "intervention_state_count", 0))
        loss_count = int(getattr(esamp_stats, "loss_count", 0))
        if applied <= 0 or loss_count <= 0:
            raise RuntimeError(
                "program intervention never engaged: "
                f"intervention_states={applied} loss_count={loss_count} "
                f"(publish attempts={attempts} hits={hits})"
            )
        print(
            "esamp program intervention smoke OK: "
            f"intervention_states={applied} loss_count={loss_count}"
        )
        return 0
    if attempts <= 0 or hits <= 0:
        raise RuntimeError(
            f"intervention never engaged: publish attempts={attempts} hits={hits}. "
            "Check provider gating / decode localization on this runner."
        )
    print(f"esamp intervention smoke OK: attempts={attempts} hits={hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""README-ratio benchmark: ESamp min-p active path vs raw vLLM baseline.

Reproduces the README's "aligned min-p ESamp benchmark" shape on the new
architecture: CUDA graphs ON, FlashInfer sampler ON, bf16, prefix caching
disabled for fairness, batch=8, n=16, min-p sampling, 128 new tokens.

Usage (one process per arm):
    python test/gpu/esamp_ratio_benchmark.py run --mode raw   --runner mrv2 --out raw.json
    python test/gpu/esamp_ratio_benchmark.py run --mode tllm  --runner mrv2 --out tllm.json
    python test/gpu/esamp_ratio_benchmark.py ratio raw.json tllm.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

PROMPTS = [
    "Give me a short introduction to large language models.",
    "Explain the theory of relativity to a high school student.",
    "Write a short story about a robot learning to paint.",
    "Describe the process of photosynthesis step by step.",
    "What are the main causes of climate change?",
    "Explain how a neural network learns from data.",
    "Summarize the history of the Roman Empire.",
    "How does the human immune system fight viruses?",
]

BATCH = 8
N_SAMPLES = 16
MAX_NEW_TOKENS = 128


def run(args: argparse.Namespace) -> None:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    # Pin the process topology BEFORE any tllm/vllm import so compared pairs
    # differ only in the tllm consumer, not in mp mode.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = args.mp

    from vllm import SamplingParams

    # No per-request seed: a seeded SamplingParams gives every request its own
    # torch.Generator, and the exact-candidate fast path (correctly) refuses
    # to bypass generator-based RNG. The engine-level seed still keeps the
    # global RNG reproducible.
    params = SamplingParams(
        temperature=1.0,
        min_p=0.05,
        n=N_SAMPLES,
        max_tokens=MAX_NEW_TOKENS,
        ignore_eos=True,
    )
    prompts = PROMPTS[:BATCH]

    if args.mode == "tllm":
        # Optimized ESamp path: exact candidate sampling + triton min-p
        # keep-mask kernel (README's "triton kernel" arm).
        os.environ["TLLM_ENABLE_EXACT_CANDIDATE_SAMPLING"] = "1"
        os.environ["TLLM_ENABLE_TRITON_MINP_KEEP_MASK"] = "1"
        from tllm.runtime import residual_runtime as core
        from tllm.workflows import esamp_support as esamp_workflow_support

        graph_scratch_rows = max(64, BATCH * N_SAMPLES)
        esamp_workflow_support.configure_esamp_runtime(
            graph_scratch_rows=graph_scratch_rows,
            enable_esamp_training=True,
            enable_distiller_intervention=not args.no_intervention,
            distiller_beta=0.5,
            distiller_sampler_backend="post_filter_exact",
        )
        llm = core.make_llm(
            model_name=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=float(args.gpu_mem),
            max_model_len=1024,
            enable_prefix_caching=False,
            enforce_eager=False,
            seed=1234,
        )
    else:
        from vllm import LLM

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=float(args.gpu_mem),
            max_model_len=1024,
            enable_prefix_caching=False,
            enforce_eager=False,
            seed=1234,
        )

    try:
        # Warmup (also trains the distiller so the intervention path is hot).
        llm.generate(prompts, params)
        if args.mode == "tllm":
            core.read_and_reset_distiller_timing_stats()
            core.read_and_reset_esamp_stats()
        rounds = max(1, int(args.rounds))
        best = None
        outputs = None
        for _ in range(rounds):
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params)
            dt = time.perf_counter() - t0
            if best is None or dt < best:
                best = dt
        elapsed = float(best)
    finally:
        from tllm.util.tools import shutdown_llm_instance

        shutdown_llm_instance(llm, cooldown_s=1.0)

    out_tokens = sum(len(o.token_ids) for req in outputs for o in req.outputs)
    payload = {
        "mode": args.mode,
        "runner": args.runner,
        "mp": args.mp,
        "model": args.model,
        "elapsed_s": elapsed,
        "output_tokens": out_tokens,
        "out_tok_per_s": out_tokens / elapsed,
    }
    if args.mode == "tllm":
        precompute = core.RUNTIME.sampler_precompute
        payload["candidate_sample_count"] = int(precompute.candidate_sample_count)
        payload["candidate_row_count"] = int(precompute.candidate_row_count)
        payload["candidate_token_count"] = int(precompute.candidate_token_count)
        stats = core.read_and_reset_esamp_stats()
        payload["esamp_loss_count"] = int(getattr(stats, "loss_count", 0))
        timing = core.read_and_reset_distiller_timing_stats()
        payload["distiller_timing"] = {
            k: getattr(timing, k)
            for k in (
                "precompute_ms_total",
                "precompute_count",
                "wait_ms_total",
                "wait_count",
                "fallback_ms_total",
                "fallback_count",
                "port_publish_attempt_count",
                "port_publish_hit_count",
            )
        }
        hotspots = core.read_and_reset_path_hotspot_stats()
        payload["path_hotspots_ms"] = dict(hotspots.cpu_ms_total)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(json.dumps(payload))


def ratio(args: argparse.Namespace) -> None:
    with open(args.a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        b = json.load(fh)
    r = b["out_tok_per_s"] / a["out_tok_per_s"]
    print(f"baseline({a['mode']}/{a['runner']}/mp{a.get('mp', '?')}): {a['out_tok_per_s']:.3f} tok/s")
    print(f"esamp   ({b['mode']}/{b['runner']}/mp{b.get('mp', '?')}): {b['out_tok_per_s']:.3f} tok/s")
    print(f"ratio: {r:.4f}  (README old-version reference: 0.9878)")
    if r < 0.9878:
        print("REGRESSED below README ratio")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--mode", choices=["raw", "tllm"], required=True)
    p_run.add_argument("--runner", choices=["mrv1", "mrv2"], default="mrv2")
    p_run.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_run.add_argument("--gpu-mem", default="0.4")
    p_run.add_argument("--no-intervention", action="store_true")
    p_run.add_argument("--rounds", type=int, default=3)
    p_run.add_argument("--mp", choices=["0", "1"], default="0", help="VLLM_ENABLE_V1_MULTIPROCESSING")
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=run)
    p_ratio = sub.add_parser("ratio")
    p_ratio.add_argument("a")
    p_ratio.add_argument("b")
    p_ratio.set_defaults(func=ratio)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GPU parity harness: raw vLLM vs tLLM-patched vLLM.

Runs one mode per process (vLLM engines do not reliably coexist in one
process) and dumps generated token ids to JSON. ``compare`` mode checks
two result files for bit-exact equality.

Usage:
    python test/gpu/parity_harness.py run --mode raw|tllm --runner mrv1|mrv2 \
        --out /tmp/out.json [--sampling greedy|random] [--flashinfer 0|1]
    python test/gpu/parity_harness.py compare A.json B.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "Give me a short introduction to large language models.",
    "What is the capital of France? Answer in one word.",
    "Write a haiku about autumn leaves.",
    "Explain the difference between TCP and UDP in two sentences.",
    "List three prime numbers greater than 100.",
    "How do I reverse a string in Python?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "What year did the Apollo 11 mission land on the Moon?",
]


def _runner_env(runner: str) -> None:
    if runner == "mrv1":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    elif runner == "mrv2":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    else:
        raise ValueError(f"unknown runner {runner}")


def run(args: argparse.Namespace) -> None:
    _runner_env(args.runner)
    # Pin the multiprocess mode BEFORE any tllm/vllm import so both modes of
    # a compared pair run the same process topology.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = args.mp
    if args.flashinfer == "0":
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    if args.mode == "tllm":
        import tllm

        llm = tllm.make_llm(
            model_name=MODEL,
            dtype="bfloat16",
            gpu_memory_utilization=float(args.gpu_mem),
            max_model_len=1024,
            enable_prefix_caching=True,
            enforce_eager=args.eager == "1",
            seed=1234,
        )
    else:
        if args.mode == "raw_import":
            import tllm  # noqa: F401 - import side effects only, no hooks
        from vllm import LLM

        llm = LLM(
            model=MODEL,
            enforce_eager=args.eager == "1",
            gpu_memory_utilization=float(args.gpu_mem),
            max_model_len=1024,
            seed=1234,
        )

    from vllm import SamplingParams

    if args.sampling == "greedy":
        params = SamplingParams(temperature=0.0, max_tokens=32, logprobs=1)
    else:
        params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            min_p=0.05,
            seed=42,
            max_tokens=32,
            logprobs=1,
        )

    outputs = llm.generate(PROMPTS, params)
    tokens = [list(o.outputs[0].token_ids) for o in outputs]
    # Per-step logprob of the sampled token: distinguishes "logits drifted"
    # (numeric difference) from "same logits, different draw" (RNG difference).
    logprobs = []
    for o in outputs:
        row = []
        for step_lps in o.outputs[0].logprobs or []:
            chosen = next(iter(step_lps.values())) if step_lps else None
            row.append(None if chosen is None else float(chosen.logprob))
        logprobs.append(row)

    # Report which runner class actually served the engine (lives in the
    # EngineCore subprocess under mp=1, so query it via collective_rpc with a
    # STRING method name — closures need VLLM_ALLOW_INSECURE_SERIALIZATION and
    # cannot cross the process boundary. The tllm worker extension provides
    # tllm_runner_name; raw-mode engines lack it and keep the local probe.
    runner_name = "unknown"
    if args.mode == "tllm":
        try:
            runner_name = llm.llm_engine.engine_core.collective_rpc("tllm_runner_name")[0]
        except Exception as exc:  # pragma: no cover - diagnostic only
            runner_name = f"unknown ({exc})"
    else:
        try:
            runner_name = llm.llm_engine.collective_rpc(
                lambda worker: f"{type(worker.model_runner).__module__}.{type(worker.model_runner).__name__}"
            )[0]
        except Exception as exc:  # pragma: no cover - diagnostic only
            runner_name = f"unknown ({exc})"

    drivers: list[str] = []
    if args.mode == "tllm":
        from tllm.runtime import residual_runtime

        drivers = list(getattr(residual_runtime, "_TLLM_BACKEND_DRIVERS", ()) or ())

    payload = {
        "mode": args.mode,
        "runner": args.runner,
        "sampling": args.sampling,
        "eager": args.eager,
        "flashinfer": args.flashinfer,
        "mp": args.mp,
        "runner_name": runner_name,
        "drivers": drivers,
        "tokens": tokens,
        "logprobs": logprobs,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"wrote {args.out}: runner={runner_name} drivers={drivers}")


def compare(args: argparse.Namespace) -> None:
    with open(args.a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        b = json.load(fh)
    print(f"A: {a['mode']}/{a['runner']} mp={a.get('mp', '?')} -> {a['runner_name']} drivers={a.get('drivers')}")
    print(f"B: {b['mode']}/{b['runner']} mp={b.get('mp', '?')} -> {b['runner_name']} drivers={b.get('drivers')}")
    if a["tokens"] == b["tokens"]:
        print("PARITY OK: token ids bit-exact across", len(a["tokens"]), "prompts")
        return
    # Compare logprobs up to each divergence point to classify the cause.
    for i, (la, lb) in enumerate(zip(a.get("logprobs", []), b.get("logprobs", []))):
        for j, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                print(f"  prompt {i}: first logprob diff at step {j}: {x} vs {y}")
                break
    mismatches = [
        i for i, (ta, tb) in enumerate(zip(a["tokens"], b["tokens"])) if ta != tb
    ]
    print(f"PARITY FAIL: {len(mismatches)}/{len(a['tokens'])} prompts differ: {mismatches}")
    for i in mismatches[:3]:
        print(f"  prompt {i}: A={a['tokens'][i][:12]}... B={b['tokens'][i][:12]}...")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--mode", choices=["raw", "tllm", "raw_import"], required=True)
    p_run.add_argument("--runner", choices=["mrv1", "mrv2"], required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--sampling", choices=["greedy", "random"], default="greedy")
    p_run.add_argument("--eager", choices=["0", "1"], default="1")
    p_run.add_argument("--flashinfer", choices=["0", "1"], default="1")
    p_run.add_argument("--mp", choices=["0", "1"], default="0", help="VLLM_ENABLE_V1_MULTIPROCESSING")
    p_run.add_argument("--gpu-mem", default="0.4")
    p_run.set_defaults(func=run)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_cmp.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

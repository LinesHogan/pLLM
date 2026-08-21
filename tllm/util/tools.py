#!/usr/bin/env python3
"""Non-core utility functions for verification and benchmarking scripts."""

from __future__ import annotations

import gc
import time
from typing import Any, Callable, Dict, List, Optional

import torch
from vllm import LLM, SamplingParams

# Canonical home: tllm.runtime.lifecycle (engine construction is runtime
# business). Re-exported here for existing benchmark/verification scripts.
from tllm.runtime.lifecycle import make_plain_llm


def build_greedy_params(max_new_tokens: int, seed: int, ignore_eos: bool = False) -> SamplingParams:
    return SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens,
        min_tokens=max_new_tokens if ignore_eos else 0,
        ignore_eos=ignore_eos,
        seed=seed,
    )


def sum_output_tokens(outputs) -> int:
    total = 0
    for out in outputs:
        if not getattr(out, "outputs", None):
            continue
        total += len(out.outputs[0].token_ids)
    return total


def shutdown_llm_instance(llm: LLM, cooldown_s: float) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
    except Exception as e:
        print(f"[benchmark warning] explicit shutdown failed: {e}")
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                print(f"[benchmark warning] destroy_process_group failed: {e}")
        if cooldown_s > 0:
            time.sleep(cooldown_s)


def build_prompt_batch(base_prompts: List[str], batch_size: int) -> List[str]:
    if batch_size <= len(base_prompts):
        return base_prompts[:batch_size]
    prompts: List[str] = []
    for i in range(batch_size):
        base = base_prompts[i % len(base_prompts)]
        prompts.append(f"{base} [bench-{i}]")
    return prompts






def read_prompts(prompt_file: str, prompt_args: List[str]) -> List[str]:
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if prompts:
            return prompts

    if prompt_args:
        prompts = [p for p in prompt_args if p.strip()]
        if prompts:
            return prompts

    return [
        "1+2=3; 5+4=",
        "4*9=36; 7*8=",
    ]


def gpu_mem_gib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    gib = 1024 ** 3
    return {
        "allocated": float(allocated / gib),
        "reserved": float(reserved / gib),
        "free": float(free / gib),
        "total": float(total / gib),
    }


def print_gpu_mem(prefix: str) -> None:
    m = gpu_mem_gib()
    print(
        f"{prefix} gpu_mem_gib: allocated={m['allocated']:.2f} "
        f"reserved={m['reserved']:.2f} free={m['free']:.2f}/{m['total']:.2f}"
    )

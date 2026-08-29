#!/usr/bin/env python3
"""ReFT closed loop on tLLM, ~100 lines.

The shape of the loop (this is the whole demo):

    rollout:  tLLM serves the model with the signed LoReFT intervention
              (activation patch on the prompt's last tokens, prefill only)
    train:    the completions go to a plain transformers forward+train loop
              using pyreft's own LoreftIntervention module
    push:     the trained (W, b, R) are pushed back through the signed
              publish Region (audited state-version bump)
    rollout:  the same engine now serves the patched model

Usage:  .venv/bin/python examples/reft_rollout.py [--model Qwen/Qwen2.5-0.5B-Instruct]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch


def _load_pyreft_loreft():
    """Load pyreft's LoreftIntervention from the source checkout (skips the
    package __init__, which pulls the datasets dependency we don't need)."""
    path = Path.home() / "pyreft" / "pyreft" / "interventions.py"
    spec = importlib.util.spec_from_file_location("pyreft_interventions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LoreftIntervention


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--last-tokens", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tllm.consumers.reft import ReftConfig, ReftWorkload
    from tllm.consumers.reft.vllm_patch import install_reft_patch_on_llm
    from tllm.runtime.runtime import TllmRuntime

    prompts = ["The capital of France is", "Water is made of"]
    workload = ReftWorkload(ReftConfig(
        layers=(args.layer,), rank=args.rank, last_tokens=args.last_tokens))
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=64 << 20)
    llm = runtime.make_llm(model_name=args.model, dtype="bfloat16", enforce_eager=True,
        gpu_memory_utilization=0.3, max_model_len=128, max_num_batched_tokens=128,
        enable_prefix_caching=False, seed=0)
    install_reft_patch_on_llm(llm, runtime)
    greedy = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)

    def rollout(tag: str) -> list[str]:
        outs = llm.generate(prompts, greedy, use_tqdm=False)
        completions = [o.outputs[0].text for o in outs]
        for prompt, completion in zip(prompts, completions):
            print(f"[{tag}] {prompt!r} -> {completion.strip()!r}")
        return completions

    baseline = rollout("baseline")  # bank inactive: unpatched model

    # --- transformers-side training with pyreft's own intervention module ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16").cuda()
    LoreftIntervention = _load_pyreft_loreft()
    reft = LoreftIntervention(embed_dim=hf.config.hidden_size,
        low_rank_dimension=args.rank, dtype=torch.bfloat16).cuda()
    optimizer = torch.optim.Adam(reft.parameters(), lr=workload.config.learning_rate)
    hook_state = {}

    def patch_hook(module, hook_args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        end = hook_state["prompt_len"]
        patched[:, end - args.last_tokens:end] = reft(base=hidden[:, end - args.last_tokens:end])
        return (patched,) + tuple(output[1:]) if isinstance(output, tuple) else patched

    handle = hf.model.layers[args.layer].register_forward_hook(patch_hook)
    try:  # train the intervention to reproduce the rollout's own completions
        for step in range(args.steps):
            prompt, completion = prompts[step % len(prompts)], baseline[step % len(baseline)]
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            full = tokenizer(prompt + completion, return_tensors="pt").input_ids.cuda()
            hook_state["prompt_len"] = prompt_ids.shape[1]
            labels = full.clone(); labels[0, :prompt_ids.shape[1]] = -100
            loss = hf(input_ids=full, labels=labels).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    finally:
        handle.remove()
    print(f"[train] final loss {float(loss.detach()):.4f}")

    # --- push the trained intervention back through the signed Region ---
    state = reft.state_dict()
    runtime.publish(ReftWorkload.intervention, {
        "source_w": state["weight"].unsqueeze(0).float(),
        "source_b": state["bias"].unsqueeze(0).float(),
        "rotate": state["rotate_layer"].unsqueeze(0).float(),
    })
    before = workload.state_version
    # One engine step lands the push through the signed publish Region (the
    # prefill of a request in the same step would race the publication).
    llm.generate(["ping"], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True), use_tqdm=False)
    rollout("patched")
    print(f"[push] intervention version {before} -> {workload.state_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""GPU smoke for TLM immutable q/v versions through vLLM native LoRA."""

from __future__ import annotations

import argparse
import os
import sys

import torch


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _weights(*, value: float, rank: int):
    from transformers import AutoConfig

    from tllm.runtime.test_time_lora import LoRAAdapterWeights

    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden // config.num_attention_heads))
    q_out = int(config.num_attention_heads) * head_dim
    v_out = int(config.num_key_value_heads) * head_dim
    modules = {}
    for layer in range(int(config.num_hidden_layers)):
        prefix = f"model.layers.{layer}.self_attn"
        modules[f"{prefix}.q_proj"] = LoRAAdapterWeights(
            torch.full((rank, hidden), value, dtype=torch.bfloat16),
            torch.full((q_out, rank), value, dtype=torch.bfloat16),
            alpha=2 * rank,
        )
        modules[f"{prefix}.v_proj"] = LoRAAdapterWeights(
            torch.full((rank, hidden), value, dtype=torch.bfloat16),
            torch.full((v_out, rank), value, dtype=torch.bfloat16),
            alpha=2 * rank,
        )
    return modules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=("mrv1", "mrv2"), required=True)
    parser.add_argument("--gpu-mem", type=float, default=0.35)
    args = parser.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "mrv2" else "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import tllm
    from tllm.consumers.tlm import TLMConsumer, TLMConsumerConfig
    from tllm.runtime import residual_runtime as core
    from tllm.runtime.test_time_lora import LoRAVersionStore
    from tllm.util.tools import shutdown_llm_instance

    rank = 2
    store = LoRAVersionStore("tlm-smoke")
    store.publish(_weights(value=0.0, rank=rank))
    consumer = TLMConsumer(
        TLMConsumerConfig(rank=rank, lora_alpha=2 * rank),
        version_store=store,
    )
    core.register_dispatch_consumer(consumer)
    llm = tllm.make_llm(
        model_name=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=1234,
        max_num_seqs=2,
        enable_lora=True,
        max_lora_rank=8,
        max_loras=2,
        max_cpu_loras=4,
    )
    try:
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
        first = llm.generate(["Name one primary color."], params, use_tqdm=False)
        store.publish(_weights(value=1e-5, rank=rank))
        second = llm.generate(["Name one secondary color."], params, use_tqdm=False)
        if [len(first[0].outputs[0].token_ids), len(second[0].outputs[0].token_ids)] != [2, 2]:
            print("FAIL: generation did not complete")
            return 1
        publisher = core.RUNTIME.test_time_lora_publisher
        if publisher is None or publisher._published_version != 2:
            print("FAIL: TLM LoRA versions were not published")
            return 1
        history = store.assignment_history()
        if 2 not in history.values():
            print(
                "FAIL: later request did not pin version 2: "
                f"history={dict(history)}"
            )
            return 1
        print(
            f"OK: TLM LoRA smoke passed (runner={args.runner}, "
            f"published=2, assignments={dict(history)})"
        )
        return 0
    finally:
        shutdown_llm_instance(llm, cooldown_s=1.0)
        core.clear_dispatch_consumers()


if __name__ == "__main__":
    sys.exit(main())

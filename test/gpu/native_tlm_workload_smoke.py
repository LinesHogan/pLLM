#!/usr/bin/env python3
"""Real HF/PEFT train -> generic overlay -> vLLM LoRA admission smoke."""

from __future__ import annotations

import argparse
import json
from typing import cast

import torch
from vllm import SamplingParams

from tllm.consumers.tlm import TLMConfig, TLMWorkload
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.workload_bindings import LocalRuntimeServices
from tllm.util.tools import shutdown_llm_instance


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--gpu-mem", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = _args()
    workload = TLMWorkload(
        TLMConfig(
            model_id=args.model,
            mode="online",
            rank=2,
            lora_alpha=4,
            threshold=-1.0,
            local_files_only=True,
        )
    )
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=2 << 30
    )
    llm = runtime.make_llm(
        model_name=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=8,
        seed=1234,
    )
    try:
        spec = workload.specialization
        runtime.stage(
            TLMWorkload.batches,
            {
                "input_ids": torch.tensor(
                    ((1, 2, 3, 4),), device=spec.device, dtype=torch.long
                ),
                "labels": torch.tensor(
                    ((1, 2, 3, 4),), device=spec.device, dtype=torch.long
                ),
                "attention_mask": torch.ones(
                    (1, 4), device=spec.device, dtype=torch.long
                ),
            },
        )
        params = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
        prompt = ({"prompt_token_ids": [1, 2, 3]},)
        first = llm.generate(prompt, params, use_tqdm=False)
        publication = runtime.drain(TLMWorkload.publications)[0]
        second = llm.generate(prompt, params, use_tqdm=False)
        local_workload = runtime.local_workload(workload)
        services = cast(
            LocalRuntimeServices,
            local_workload.native_workload_resources().services,
        )
        stats = services.stats(True)
        diagnostics = services.diagnostics()
        if stats.loss_count <= 0 or diagnostics["update_count"] <= 0:
            raise AssertionError(
                f"TLM training did not run: stats={stats!r} diagnostics={diagnostics!r}"
            )
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        adapters = runner.lora_manager._adapter_manager.list_adapters()
        if not adapters:
            from tllm.backends.protocol import runner_binding_for

            binding = runner_binding_for(runner)
            handles = {} if binding is None else binding.parameter_overlay_handles
            controller = handles.get("tllm.versioned_parameter_overlay")
            raise AssertionError(
                "generic TLM overlay was not installed in vLLM: "
                f"lineages={local_workload.overlay_versions.store.lineages()!r} "
                f"controller_lineages={None if controller is None else {channel: store.lineages() for channel, store in controller.stores.items()}!r} "
                f"handles={handles!r}"
            )
        print(
            json.dumps(
                {
                    "version": publication.version.version,
                    "loss_count": stats.loss_count,
                    "update_count": diagnostics["update_count"],
                    "adapter_ids": sorted(adapters),
                    "first_tokens": first[0].outputs[0].token_ids,
                    "second_tokens": second[0].outputs[0].token_ids,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        runtime.clear_consumers()
        shutdown_llm_instance(llm, cooldown_s=1.0)


if __name__ == "__main__":
    raise SystemExit(main())

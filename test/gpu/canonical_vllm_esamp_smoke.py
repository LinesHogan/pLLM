"""One real MRv2 boundary proof for canonical ESamp request execution."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tllm
import torch
from vllm import LLM, SamplingParams

from tllm.canonical.workloads.esamp_request import (
    CanonicalRequestSamplerExecution,
    ESampRequestWorkload,
)
from tllm.canonical.workloads.esamp_request_types import ESampRequestConfig
from tllm.canonical_vllm.runtime import (
    CanonicalVllmConfig,
    CanonicalVllmDType,
    CanonicalVllmLlm,
)


def _require_esamp_execution(
    handle: CanonicalVllmLlm,
) -> CanonicalRequestSamplerExecution:
    execution = handle.deployment.execution
    if type(handle.deployment.workload) is not ESampRequestWorkload:
        raise RuntimeError("canonical vLLM smoke did not receive an ESamp deployment")
    if type(execution) is not CanonicalRequestSamplerExecution:
        raise RuntimeError("canonical vLLM smoke did not receive ESamp CUDA execution")
    return execution


def main() -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    plain = tllm.make_llm(
        model_name="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16",
        gpu_memory_utilization=0.25, max_model_len=64,
        enable_prefix_caching=False, enforce_eager=True, seed=7,
        max_num_seqs=1, max_num_batched_tokens=64,
    )
    if not isinstance(plain, LLM):
        raise RuntimeError("plain public route did not construct vLLM")
    try:
        outputs = plain.generate(
            ["The sky is"], SamplingParams(max_tokens=3, temperature=0.0),
            use_tqdm=False,
        )
        if not outputs[0].outputs[0].text:
            raise RuntimeError("plain public inference produced no text")
    finally:
        plain.llm_engine.engine_core.shutdown()
        del plain
        gc.collect()
        torch.cuda.empty_cache()
    workload = ESampRequestWorkload(
        ESampRequestConfig(
            rank=2,
            source_layer=0,
            target_layer=-1,
            intervention=True,
            intervention_beta=0.2,
        )
    )
    routed = tllm.make_llm(
        workloads=(workload,),
        canonical_config=CanonicalVllmConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype=CanonicalVllmDType.BFLOAT16,
            max_model_len=64,
            max_num_seqs=1,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.4,
            seed=7,
        ),
    )
    match routed:
        case CanonicalVllmLlm() as handle:
            pass
        case _:
            raise RuntimeError("public make_llm did not select canonical runtime")
    try:
        outputs = handle.generate(
            ["The sky is"],
            SamplingParams(
                max_tokens=3,
                temperature=0.0,
                n=1,
                ignore_eos=True,
            ),
        )
        execution = _require_esamp_execution(handle)
        diagnostics = execution.diagnostics()
        if (
            not outputs[0].outputs[0].text
            or diagnostics.loss_count != 2
            or diagnostics.update_count != 2
            or diagnostics.completed_sampler_pins != 2
            or diagnostics.active_lineages != 0
            or diagnostics.pending_state_releases != 0
            or execution.plan.fallback_count != 0
        ):
            raise RuntimeError("canonical MRv2 ESamp lifecycle evidence is incomplete")
        print(
            {
                "text": outputs[0].outputs[0].text,
                "loss_count": diagnostics.loss_count,
                "update_count": diagnostics.update_count,
                "active_lineages": diagnostics.active_lineages,
                "fallback_count": execution.plan.fallback_count,
                "attachment": (
                    handle.deployment.attachment.physical.bundle_signature[:12]
                ),
            }
        )
    finally:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

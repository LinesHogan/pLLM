"""One real MRv2 boundary proof for request parameter overlays, dense and low-rank."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass
from typing import Annotated

import torch

import pllm
from pllm import TensorUpdate
from vllm import SamplingParams

from pllm.workloads.inplace_ttt import (
    InPlaceTTTConfig,
    InPlaceTTTWorkload,
)
from pllm.config import ModelConfig, VllmDType
from pllm.backends.vllm.runtime import VllmLlm
from pllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterOverlayOperator,
    ParameterRole,
    ParameterTarget,
)


@dataclass(frozen=True)
class RequestLowRankTTT(pllm.Workload):
    """Per-request LoRA trained on the same captured operand pairs as the dense path."""

    @property
    def factors(self) -> pllm.RequestState:
        return pllm.RequestState({
            "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
            "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
        })

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(state="factors",
        targets=(ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT),),
        operator=ParameterOverlayOperator.LOW_RANK_ADD, factors=("a", "b"), scale=0.5)
    def transition(self, hidden: Annotated[torch.Tensor, pllm.Operand.INPUT],
                   desired: Annotated[torch.Tensor, pllm.Operand.TARGET],
                   base: Annotated[torch.Tensor, pllm.Operand.WEIGHT],
                   a: Annotated[torch.Tensor, pllm.Parameter("factors", "a")],
                   b: Annotated[torch.Tensor, pllm.Parameter("factors", "b")]) -> TensorUpdate:
        output = hidden @ (base + 0.5 * b @ a).T
        error = output - desired
        gradient_b = 2 * error.T @ (hidden @ a.T) / hidden.shape[0]
        gradient_a = 2 * (error @ b).T @ hidden / hidden.shape[0]
        a.add_(-1e-3 * gradient_a)
        b.add_(-1e-3 * gradient_b)
        return TensorUpdate(output, changed=True, update_count=1, trained_rows=hidden.shape[0])


def _check(handle: VllmLlm[object, object], label: str) -> dict[str, object]:
    outputs = handle.generate(
        ["The sky is", "Water freezes at"],
        SamplingParams(max_tokens=2, temperature=0.0, n=1, ignore_eos=True),
    )
    diagnostics = handle.deployment.execution.diagnostics()
    if (
        any(not output.outputs[0].text for output in outputs)
        or diagnostics.update_count < 1
        or diagnostics.trained_rows < 1
        or handle.runner.overlay_apply_count < 2
        or handle.runner.overlay_updated_apply_count < 1
        or diagnostics.active_lineages != 0
        or diagnostics.pending_state_releases != 0
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError(f"{label} MRv2 parameter-overlay lifecycle evidence is incomplete")
    return {
        "text": tuple(output.outputs[0].text for output in outputs),
        "update_count": diagnostics.update_count,
        "trained_rows": diagnostics.trained_rows,
        "apply_rows": handle.runner.overlay_apply_count,
        "updated_apply_rows": handle.runner.overlay_updated_apply_count,
        "active_lineages": diagnostics.active_lineages,
        "fallback_count": diagnostics.fallback_count,
        "attachment": handle.deployment.attachment.physical.bundle_signature[:12],
    }


def main() -> int:
    results: dict[str, object] = {}
    workload = InPlaceTTTWorkload(
        InPlaceTTTConfig(
            (
                ParameterTarget(
                    0,
                    ParameterModuleKind.MLP,
                    ParameterRole.DOWN_PROJECTION_WEIGHT,
                ),
            ),
            chunk_size=1,
            learning_rate=1e-4,
            update_prefill=True,
            update_decode=False,
        )
    )
    routed = pllm.make_llm(
        workload=workload,
        config=ModelConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype=VllmDType.BFLOAT16,
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.25,
            seed=7,
            execution=pllm.ExecutionOptions(memory_budget_bytes=1 << 30),
        ),
    )
    match routed:
        case VllmLlm() as handle:
            pass
        case _:
            raise RuntimeError("public make_llm did not route parameter overlay")
    try:
        results["dense"] = _check(handle, "dense")
    finally:
        handle.close()

    low_rank = pllm.make_llm(
        workload=RequestLowRankTTT(),
        config=ModelConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype=VllmDType.BFLOAT16,
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.25,
            seed=7,
        ),
    )
    match low_rank:
        case VllmLlm() as handle:
            pass
        case _:
            raise RuntimeError("public make_llm did not route request low-rank overlay")
    try:
        results["low_rank"] = _check(handle, "request low-rank")
    finally:
        handle.close()
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

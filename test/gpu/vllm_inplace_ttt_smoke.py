"""One real MRv2 boundary proof for request parameter overlays, dense and low-rank."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass
from typing import Annotated

import torch
from vllm import SamplingParams

import pllm
from pllm import TensorUpdate
from pllm.backends.vllm.runtime import VllmLlm
from pllm.config import ModelConfig, VllmDType
from pllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterOverlayOperator,
    ParameterRole,
    ParameterTarget,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.workloads.inplace_ttt import (
    InPlaceTTTConfig,
    InPlaceTTTWorkload,
)


@dataclass(frozen=True)
class RequestLowRankTTT(pllm.Workload):
    """Per-request LoRA trained on the same captured operand pairs as the dense path."""

    @property
    def factors(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
                "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
            }
        )

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(
        state="factors",
        targets=(ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT),),
        operator=ParameterOverlayOperator.LOW_RANK_ADD,
        factors=("a", "b"),
        scale=0.5,
    )
    def transition(
        self,
        hidden: Annotated[torch.Tensor, pllm.Operand.INPUT],
        desired: Annotated[torch.Tensor, pllm.Operand.TARGET],
        base: Annotated[torch.Tensor, pllm.Operand.WEIGHT],
        a: Annotated[torch.Tensor, pllm.Parameter("factors", "a")],
        b: Annotated[torch.Tensor, pllm.Parameter("factors", "b")],
    ) -> TensorUpdate:
        output = hidden @ (base + 0.5 * b @ a).T
        error = output - desired
        gradient_b = 2 * error.T @ (hidden @ a.T) / hidden.shape[0]
        gradient_a = 2 * (error @ b).T @ hidden / hidden.shape[0]
        a.add_(-1e-3 * gradient_a)
        b.add_(-1e-3 * gradient_b)
        return TensorUpdate(output, changed=True, update_count=1, trained_rows=hidden.shape[0])


@dataclass(frozen=True)
class ImportedLowRank(pllm.Workload):
    """Per-request LoRA factors produced outside the engine (e.g. by a hypernetwork)."""

    @property
    def factors(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
                "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
            }
        )

    @property
    def adapter(self) -> pllm.Overlay:
        return pllm.Overlay(
            "factors",
            (ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT),),
            operator=ParameterOverlayOperator.LOW_RANK_ADD,
            factors=("a", "b"),
            scale=0.5,
        )


@dataclass(frozen=True)
class RoutedLowRankTTT(pllm.Workload):
    """Route-keyed per-adapter LoRA; the routing policy below is ordinary host code."""

    @property
    def factors(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
                "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
            },
            keys=(pllm.RequestKeyField.ROUTE,),
        )

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(
        state="factors",
        targets=(ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT),),
        operator=ParameterOverlayOperator.LOW_RANK_ADD,
        factors=("a", "b"),
        scale=0.5,
    )
    def transition(
        self,
        hidden: Annotated[torch.Tensor, pllm.Operand.INPUT],
        desired: Annotated[torch.Tensor, pllm.Operand.TARGET],
        base: Annotated[torch.Tensor, pllm.Operand.WEIGHT],
        a: Annotated[torch.Tensor, pllm.Parameter("factors", "a")],
        b: Annotated[torch.Tensor, pllm.Parameter("factors", "b")],
    ) -> TensorUpdate:
        output = hidden @ (base + 0.5 * b @ a).T
        error = output - desired
        gradient_b = 2 * error.T @ (hidden @ a.T) / hidden.shape[0]
        gradient_a = 2 * (error @ b).T @ hidden / hidden.shape[0]
        a.add_(-1e-3 * gradient_a)
        b.add_(-1e-3 * gradient_b)
        return TensorUpdate(output, changed=True, update_count=1, trained_rows=hidden.shape[0])


def _route(prompt: str) -> str:
    """The framework signs the key declaration; users compute the value."""
    import re

    return "math" if re.search(r"\d|plus|sum", prompt) else "code"


def _check_routed(handle: VllmLlm[object, object]) -> dict[str, object]:
    prompts = ["2 plus 2 is", "The sum of 1 and 1 is", "def main():"]
    sampling = SamplingParams(max_tokens=2, temperature=0.0, n=1, ignore_eos=True)
    try:
        handle.generate(prompts, sampling)
    except ValueError as exc:
        if "route" not in str(exc):
            raise
    else:
        raise RuntimeError("route-keyed state accepted generation without routes")
    outputs = handle.generate(prompts, sampling, routes=tuple(_route(prompt) for prompt in prompts))
    diagnostics = handle.deployment.execution.diagnostics()
    # Three retired requests, two route lineages still resident: the two math
    # prompts shared one adapter, and the dormant pool outlives its requests.
    if (
        diagnostics.active_lineages != 2
        or diagnostics.update_count < 1
        or handle.runner.overlay_apply_count < 3
        or handle.runner.overlay_updated_apply_count < 1
        or diagnostics.pending_state_releases != 0
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError("routed overlay lifecycle evidence is incomplete")
    return {
        "text": tuple(output.outputs[0].text for output in outputs),
        "update_count": diagnostics.update_count,
        "apply_rows": handle.runner.overlay_apply_count,
        "updated_apply_rows": handle.runner.overlay_updated_apply_count,
        "active_lineages": diagnostics.active_lineages,
        "fallback_count": diagnostics.fallback_count,
        "attachment": handle.deployment.attachment.physical.bundle_signature[:12],
    }


def _key(prompt_index: int) -> RequestStateKey:
    return RequestStateKey.project(RequestIdentity("", prompt_index, 0), (RequestKeyField.PROMPT_IDX,))


def _check_imported(handle: VllmLlm[object, object]) -> dict[str, object]:
    prompts = ["The sky is", "Water freezes at"]
    sampling = SamplingParams(max_tokens=2, temperature=0.0, n=1, ignore_eos=True)
    (layout,) = handle.attachment.inputs.plan.tensor_layout_request.layouts
    down_shape, up_shape = (tuple(field.shape) for field in layout.layout.fields[:2])
    device = handle.device
    # No import: every row reads its zero-initialized request state, identical to the base model.
    baseline = tuple(output.outputs[0].text for output in handle.generate(prompts, sampling))
    if any(not text for text in baseline):
        raise RuntimeError("imported overlay baseline generation is incomplete")
    # One batch, one imported request: the second row must keep the exact base outputs.
    generator = torch.Generator().manual_seed(11)
    down = torch.randn(down_shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    up = torch.randn(up_shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    mixed = handle.generate(
        prompts,
        sampling,
        prompt_indices=(2, 3),
        auxiliary=pllm.StateTensorUpdate((down.unsqueeze(0), up.unsqueeze(0)), (_key(2),)),
    )
    mixed_texts = tuple(output.outputs[0].text for output in mixed)
    if mixed_texts[0] == baseline[0] or mixed_texts[1] != baseline[1]:
        raise RuntimeError("imported factors must move exactly their own request's outputs")
    # Imported zero factors are an exact no-op on fresh requests.
    zeros = (
        torch.zeros((2, *down_shape), device=device, dtype=torch.bfloat16),
        torch.zeros((2, *up_shape), device=device, dtype=torch.bfloat16),
    )
    replayed = handle.generate(
        prompts, sampling, prompt_indices=(4, 5), auxiliary=pllm.StateTensorUpdate(zeros, (_key(4), _key(5)))
    )
    replayed_texts = tuple(output.outputs[0].text for output in replayed)
    if replayed_texts != baseline:
        raise RuntimeError("imported zero factors must reproduce the base model exactly")
    diagnostics = handle.deployment.execution.diagnostics()
    if (
        diagnostics.update_count != 3
        or diagnostics.active_lineages != 0
        or diagnostics.pending_state_releases != 0
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError("imported overlay lifecycle evidence is incomplete")
    return {
        "text": mixed_texts,
        "update_count": diagnostics.update_count,
        "apply_rows": handle.runner.overlay_apply_count,
        "updated_apply_rows": handle.runner.overlay_updated_apply_count,
        "active_lineages": diagnostics.active_lineages,
        "fallback_count": diagnostics.fallback_count,
        "attachment": handle.deployment.attachment.physical.bundle_signature[:12],
    }


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

    imported = pllm.make_llm(
        workload=ImportedLowRank(),
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
    match imported:
        case VllmLlm() as handle:
            pass
        case _:
            raise RuntimeError("public make_llm did not route imported request low-rank overlay")
    try:
        results["imported"] = _check_imported(handle)
    finally:
        handle.close()

    routed = pllm.make_llm(
        workload=RoutedLowRankTTT(),
        config=ModelConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype=VllmDType.BFLOAT16,
            max_model_len=64,
            max_num_seqs=3,
            max_num_batched_tokens=96,
            gpu_memory_utilization=0.25,
            seed=7,
        ),
    )
    match routed:
        case VllmLlm() as handle:
            pass
        case _:
            raise RuntimeError("public make_llm did not route route-keyed overlay")
    try:
        results["routed"] = _check_routed(handle)
    finally:
        handle.close()
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

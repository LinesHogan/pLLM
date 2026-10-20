"""Request parameter overlays on a real vLLM MRv2 model: dense, low-rank, imported and route-keyed.

Each overlay must be applied to every row, update only its own request (or route),
and leave no lineage, pending release or fallback behind.
"""

import re
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
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload

TARGET = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
PROMPTS = ["The sky is", "Water freezes at"]
SAMPLING = SamplingParams(max_tokens=2, temperature=0.0, n=1, ignore_eos=True)


def low_rank_factors(keys=()) -> pllm.RequestState:
    shapes = {
        "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
        "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
    }
    return pllm.RequestState(shapes, keys=keys) if keys else pllm.RequestState(shapes)


@dataclass(frozen=True)
class RequestLowRankTTT(pllm.Workload):
    """Per-request LoRA trained on the same captured operand pairs as the dense path."""

    @property
    def factors(self) -> pllm.RequestState:
        return low_rank_factors()

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(
        state="factors",
        targets=(TARGET,),
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
class RoutedLowRankTTT(RequestLowRankTTT):
    """Route-keyed per-adapter LoRA; the routing policy is ordinary host code."""

    @property
    def factors(self) -> pllm.RequestState:
        return low_rank_factors((pllm.RequestKeyField.ROUTE,))


@dataclass(frozen=True)
class ImportedLowRank(pllm.Workload):
    """Per-request LoRA factors produced outside the engine (e.g. by a hypernetwork)."""

    @property
    def factors(self) -> pllm.RequestState:
        return low_rank_factors()

    @property
    def adapter(self) -> pllm.Overlay:
        return pllm.Overlay(
            "factors", (TARGET,), operator=ParameterOverlayOperator.LOW_RANK_ADD, factors=("a", "b"), scale=0.5
        )


def open_llm(workload, seqs=2, tokens=64, **execution) -> VllmLlm[object, object]:
    options = {"execution": pllm.ExecutionOptions(**execution)} if execution else {}
    config = ModelConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype=VllmDType.BFLOAT16,
        max_model_len=64,
        max_num_seqs=seqs,
        max_num_batched_tokens=tokens,
        gpu_memory_utilization=0.25,
        seed=7,
        **options,
    )
    handle = pllm.make_llm(workload=workload, config=config)
    if not isinstance(handle, VllmLlm):
        raise RuntimeError("public make_llm did not route the parameter overlay to vLLM")
    return handle


def texts(outputs) -> tuple[str, ...]:
    return tuple(output.outputs[0].text for output in outputs)


def _check(handle: VllmLlm[object, object], label: str) -> tuple[str, ...]:
    outputs = texts(handle.generate(PROMPTS, SAMPLING))
    diagnostics = handle.deployment.execution.diagnostics()
    if (
        not all(outputs)
        or diagnostics.update_count < 1
        or diagnostics.trained_rows < 1
        or handle.runner.overlay_apply_count < 2
        or handle.runner.overlay_updated_apply_count < 1
        or diagnostics.active_lineages != 0
        or diagnostics.pending_state_releases != 0
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError(f"{label} MRv2 parameter-overlay lifecycle evidence is incomplete")
    return outputs


def _key(prompt_index: int) -> RequestStateKey:
    return RequestStateKey.project(RequestIdentity("", prompt_index, 0), (RequestKeyField.PROMPT_IDX,))


def _check_imported(handle: VllmLlm[object, object]) -> tuple[str, ...]:
    (layout,) = handle.attachment.inputs.plan.tensor_layout_request.layouts
    down_shape, up_shape = (tuple(field.shape) for field in layout.layout.fields[:2])
    device = handle.device
    # No import: every row reads its zero-initialized request state, identical to the base model.
    baseline = texts(handle.generate(PROMPTS, SAMPLING))
    if not all(baseline):
        raise RuntimeError("imported overlay baseline generation is incomplete")
    # One batch, one imported request: the second row must keep the exact base outputs.
    generator = torch.Generator().manual_seed(11)
    down = torch.randn(down_shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    up = torch.randn(up_shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    update = pllm.StateTensorUpdate((down.unsqueeze(0), up.unsqueeze(0)), (_key(2),))
    mixed = texts(handle.generate(PROMPTS, SAMPLING, prompt_indices=(2, 3), auxiliary=update))
    if mixed[0] == baseline[0] or mixed[1] != baseline[1]:
        raise RuntimeError("imported factors must move exactly their own request's outputs")
    # Imported zero factors are an exact no-op on fresh requests.
    zeros = (
        torch.zeros((2, *down_shape), device=device, dtype=torch.bfloat16),
        torch.zeros((2, *up_shape), device=device, dtype=torch.bfloat16),
    )
    update = pllm.StateTensorUpdate(zeros, (_key(4), _key(5)))
    if texts(handle.generate(PROMPTS, SAMPLING, prompt_indices=(4, 5), auxiliary=update)) != baseline:
        raise RuntimeError("imported zero factors must reproduce the base model exactly")
    diagnostics = handle.deployment.execution.diagnostics()
    if (
        diagnostics.update_count != 3
        or diagnostics.active_lineages != 0
        or diagnostics.pending_state_releases != 0
        or diagnostics.fallback_count != 0
    ):
        raise RuntimeError("imported overlay lifecycle evidence is incomplete")
    return mixed


def _check_routed(handle: VllmLlm[object, object]) -> tuple[str, ...]:
    prompts = ["2 plus 2 is", "The sum of 1 and 1 is", "def main():"]
    try:
        handle.generate(prompts, SAMPLING)
    except ValueError as exc:
        if "route" not in str(exc):
            raise
    else:
        raise RuntimeError("route-keyed state accepted generation without routes")
    routes = tuple("math" if re.search(r"\d|plus|sum", prompt) else "code" for prompt in prompts)
    outputs = texts(handle.generate(prompts, SAMPLING, routes=routes))
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
    return outputs


def main() -> int:
    dense = InPlaceTTTWorkload(
        InPlaceTTTConfig((TARGET,), chunk_size=1, learning_rate=1e-4, update_prefill=True, update_decode=False)
    )
    scenarios = (
        ("dense", lambda: open_llm(dense, memory_budget_bytes=1 << 30), lambda h: _check(h, "dense")),
        ("low_rank", lambda: open_llm(RequestLowRankTTT()), lambda h: _check(h, "request low-rank")),
        ("imported", lambda: open_llm(ImportedLowRank(), memory_budget_bytes=1 << 30), _check_imported),
        ("routed", lambda: open_llm(RoutedLowRankTTT(), seqs=3, tokens=96), _check_routed),
    )
    results: dict[str, tuple[str, ...]] = {}
    for name, open_handle, check in scenarios:
        handle = open_handle()
        try:
            results[name] = check(handle)
        finally:
            handle.close()
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check schedule contracts on CPU/GPU or step workloads on CPU, for unit tests and notebooks.

`check(workload, config)` compares fixed request schedules and diagnoses row mixing or routing errors.

`backend_facts` describes an in-process model (hidden size, hidden taps, state pages) in place of the facts
the vLLM runner probes; `compile_execution` compiles a workload against them and admits its instance state.
`step_input` builds one packed step (one row per request) the way the runner would hand it to the
workload; `run_step` and `run_guided_step` execute every anchor of that step, delivering sampler feedback
when the workload reads it. `hidden_binding` and `final_hidden_binding` name the tap `pllm.hidden(layer)` and
`pllm.final_hidden()` read, exactly as the runner does, so the same bindings serve the facts and the step input.

    binding = testing.hidden_binding(0, layer_count=1)
    execution = testing.compile_execution(workload, testing.backend_facts(4, (binding,), layer_count=1))
    rows = (testing.RequestIdentity("a", 0, 0),)
    execution.admit(rows)
    testing.run_step(execution, testing.step_input(rows, {binding: torch.randn(1, 4)}))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from pllm.backends.vllm import architecture
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    ModelProbe,
    ProjectionShape,
)
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.host_batch import HostBatch
from pllm.contracts.packed_hidden import (
    HiddenRowBatch,
    HiddenRows,
    PackedHiddenResourceBinding,
)
from pllm.contracts.packed_rows import PackedSamplerRowBatch, label_rows
from pllm.contracts.parameter_overlay import Projection, ProjectionOperands
from pllm.contracts.request_identity import RequestIdentity, prompt_rows
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.state_execution import StateExecution
from pllm.execution.step_input import StepInput
from pllm.testing_contracts import (
    SIMULATED_CAPABILITIES,
    CheckResult as CheckResult,
    ContractError as ContractError,
    check as check,
    scheduled_step as scheduled_step,
)
from pllm.testing_distribution import (
    DistanceSummary as DistanceSummary,
    DistributionResult as DistributionResult,
    TokenDistance as TokenDistance,
    check_distribution as check_distribution,
)
from pllm.testing_gpu import check_on_gpu as check_on_gpu

if TYPE_CHECKING:
    from pllm.execution.tensor_declaration import TensorOutput
    from pllm.workload import Workload


def backend_facts(
    hidden_size: int,
    hidden: tuple[PackedHiddenResourceBinding, ...] = (),
    *,
    device: str = "cpu",
    capabilities: tuple[str, ...] = SIMULATED_CAPABILITIES,
    pages: int = 0,
    max_lineages: int = 2,
    memory_bytes: int = 10000,
    max_rows: int | None = None,
    dtype: torch.dtype = torch.float32,
    vocab_size: int = 0,
    layer_count: int = 0,
    targets: Mapping[Projection, tuple[int, int]] = {},
) -> BackendCompileFacts:
    """Facts for an eager in-process model, in place of what the vLLM runner probes.

    `hidden` lists the taps the workload reads (see `hidden_binding`). `device` ("cpu" or "cuda") holds the
    state pages; each state gets the versions its plan needs, or `pages` if a test holds more. `capabilities`
    are the backend features plans may use; the default is everything a CPU simulation runs (overlays, AdamW,
    custom forwards, token capture). `max_lineages` bounds
    admitted requests, `memory_bytes` the state budget, and `max_rows` the packed rows per step (twice
    `max_lineages` by default). `targets` maps overlay targets to their (input, output) widths; `vocab_size`
    and `layer_count` are needed by workloads that read logits or layers.
    """
    kind = StateDeviceKind(torch.device(device).type)
    return BackendCompileFacts(
        ModelProbe(
            "c" * 64,
            AttachmentGraphMode.EAGER,
            kind,
            TensorDType(str(dtype).removeprefix("torch.")),
            hidden_size,
            max_lineages * 2 if max_rows is None else max_rows,
            tuple(
                AttachmentHiddenResource(binding, HiddenPhase.DECODE)
                for binding in sorted(hidden, key=lambda item: item.locator_key)
            ),
            (),
            targets=tuple(ProjectionShape(target, *widths) for target, widths in targets.items()),
        ),
        model_timeline(torch.device(device), capabilities),
        (PageBundleProviderFacts(f"testing.{kind}@1", kind, pages),),
        max_lineages,
        memory_bytes,
        7,
        vocab_size,
        layer_count,
    )


def hidden_binding(layer: int, layer_count: int) -> PackedHiddenResourceBinding:
    """The tap `pllm.hidden(layer)` reads, named as the vLLM runner names it."""
    return architecture.hidden_binding(layer, HiddenSite.BLOCK_OUTPUT, layer_count)


def final_hidden_binding(layer_count: int) -> PackedHiddenResourceBinding:
    """The tap `pllm.final_hidden()` reads, named as the vLLM runner names it."""
    return architecture.hidden_binding(-1, HiddenSite.MODEL_OUTPUT, layer_count)


def compile_execution(
    workload: Workload, facts: BackendCompileFacts, stream: torch.cuda.Stream | None = None
) -> StateExecution:
    """Compile `workload` against `facts` and admit its instance state, as the vLLM runner does on install.

    Request state is admitted per request with `execution.admit(identities)`.
    """
    execution = workload.compile_backend(facts, map_host=get_accelerator_view_from_cpu_tensor).execution
    execution.admit_instances(stream)
    return execution


def step_input(
    identities: Sequence[RequestIdentity],
    hidden: Mapping[PackedHiddenResourceBinding, torch.Tensor] | None = None,
    *,
    step_id: int = 0,
    token_offset: int = 0,
    tokens: Sequence[Literal["prefill", "decode"]] | None = None,
    sampler_order: Sequence[int] | None = None,
    batch: HostBatch[torch.Tensor] | None = None,
    operands: tuple[ProjectionOperands, ...] = (),
    stream: torch.cuda.Stream | None = None,
) -> StepInput:
    """One packed step: row i is the one token of request `identities[i]` at `token_offset`; it decodes unless
    `tokens` says "prefill", which makes it the last token of the request's prompt.

    `sampler_order` lists the packed row behind each sampler row (identity order by default; `()` for a
    step without sampling). Hidden tensors are given per tap, one row per identity; CUDA tensors need the
    `stream` they were produced on.
    """
    count = len(identities)
    prompts = [
        token_offset + 1 if token == "prefill" else 0
        for token, _ in zip(tokens or ("decode",) * count, identities, strict=True)
    ]
    labeled = label_rows(tuple(identities), [1] * count, prompts, [token_offset] * count, step_id)
    rows = labeled.row_batch
    taps = tuple(
        HiddenRows(binding, tensor, np.arange(count, dtype=np.int64))
        for binding, tensor in sorted((hidden or {}).items(), key=lambda item: item[0].locator_key)
    )
    return StepInput(
        rows,
        labeled if sampler_order is None else PackedSamplerRowBatch(rows, np.asarray(sampler_order, dtype=np.int64)),
        HiddenRowBatch(rows, taps) if taps else None,
        operands=operands,
        batch=batch,
        import_rows=prompt_rows(identities),
        hidden_readiness=None if stream is None else CudaStreamReadiness(stream),
    )


def run_step(
    execution: StateExecution,
    value: StepInput,
    *,
    feedback: SamplerFeedback | None = None,
    stream: torch.cuda.Stream | None = None,
) -> TensorOutput:
    """Execute every anchor of one step, completing the sampler with `feedback` when the workload reads it."""
    return execution.step(value, stream, feedback=feedback)


def run_guided_step(
    execution: StateExecution,
    value: StepInput,
    *,
    feedback: SamplerFeedback | None = None,
    stream: torch.cuda.Stream | None = None,
) -> tuple[tuple[PackedSamplerGuidanceBatch, ...], TensorOutput]:
    """Like `run_step`, also returning additions in declaration order, followed by any guidance stage."""
    context = execution.start_step(value, stream)
    guidance: tuple[PackedSamplerGuidanceBatch, ...] = ()
    for anchor in execution.arena.plan.timeline.anchors:
        execution.execute_anchor(context, AnchorId(anchor.ordinal))
        if anchor.ordinal == execution.sampler_anchor_id:
            guidance = execution.sampler_guidance(context, stream)
            execution.complete_sampler(context, stream, feedback=feedback)
    assert guidance
    return guidance, execution.finish_step(context, stream)


__all__ = [
    "CheckResult",
    "DistanceSummary",
    "DistributionResult",
    "TokenDistance",
    "check_distribution",
    "ContractError",
    "check",
    "check_on_gpu",
    "RequestIdentity",
    "SamplerFeedback",
    "backend_facts",
    "compile_execution",
    "final_hidden_binding",
    "hidden_binding",
    "run_guided_step",
    "run_step",
    "step_input",
    "scheduled_step",
]

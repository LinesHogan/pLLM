"""Compile and step a workload without vLLM, with hand-chosen rows, for unit tests and notebooks.

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
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import torch

from pllm.backends.vllm import architecture
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    BackendPlacementProbe,
    ProjectionShape,
)
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.host_batch import HostBatch
from pllm.contracts.packed_hidden import (
    HiddenRowBatch,
    HiddenRows,
    PackedHiddenResourceBinding,
)
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.parameter_overlay import CapturedParameterOperand
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId, SchedulerTimeline
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.state_execution import StateExecution
from pllm.execution.step_input import StepInput

if TYPE_CHECKING:
    from pllm.authoring import BackendWorkload

DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")

_CPU_TIMELINE = model_timeline(torch.device("cpu"))
_CPU_PAGES = PageBundleProviderFacts("testing.cpu@1", 16)


def backend_facts(
    hidden_size: int,
    hidden: tuple[PackedHiddenResourceBinding, ...] = (),
    timeline: SchedulerTimeline = _CPU_TIMELINE,
    provider: PageBundleProviderFacts | None = _CPU_PAGES,
    *,
    max_lineages: int = 2,
    memory_bytes: int = 10000,
    max_rows: int | None = None,
    dtype: TensorDType = TensorDType.FLOAT32,
    vocab_size: int = 0,
    layer_count: int = 0,
    targets: tuple[ProjectionShape, ...] = (),
) -> BackendCompileFacts:
    """Facts for an eager in-process model, in place of what the vLLM runner probes.

    `hidden` lists the taps the workload reads (see `hidden_binding`); `provider` holds state pages, None for
    a workload without state. `max_lineages` bounds admitted requests, `memory_bytes` the state budget, and
    `max_rows` the packed rows per step (twice `max_lineages` by default). `targets` declares parameter
    shapes for overlays; `vocab_size` and `layer_count` are needed by workloads that read logits or layers.
    """
    return BackendCompileFacts(
        BackendPlacementProbe(
            "c" * 64,
            AttachmentGraphMode.EAGER,
            _CPU_PAGES.device_kind if provider is None else provider.device_kind,
            dtype,
            hidden_size,
            max_lineages * 2 if max_rows is None else max_rows,
            tuple(
                AttachmentHiddenResource(binding, HiddenPhase.DECODE)
                for binding in sorted(hidden, key=lambda item: item.locator_key)
            ),
            (),
            targets=targets,
        ),
        timeline,
        () if provider is None else (provider,),
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
    workload: BackendWorkload[DataT, ResultT], facts: BackendCompileFacts, stream: torch.cuda.Stream | None = None
) -> StateExecution[StepInput, DataT, ResultT]:
    """Compile `workload` against `facts` and admit its instance state, as the vLLM runner does on install.

    Request state is admitted per request with `execution.admit(identities)`.
    """
    execution = workload.compile_backend(facts).execution
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
    operands: tuple[CapturedParameterOperand, ...] = (),
    stream: torch.cuda.Stream | None = None,
) -> StepInput:
    """One packed step: row i belongs to `identities[i]` and decodes unless `tokens` says otherwise.

    `sampler_order` lists the packed row behind each sampler row (identity order by default; `()` for a
    step without sampling). Hidden tensors are given per tap, one row per identity; CUDA tensors need the
    `stream` they were produced on.
    """
    count = len(identities)
    row_phases = tuple(HiddenPhase(token) for token in tokens) if tokens is not None else (HiddenPhase.DECODE,) * count
    rows = PackedPhaseRowBatch(
        step_id,
        count,
        tuple(
            PackedPhaseRow(i, i, token_offset, phase, identity)
            for i, (phase, identity) in enumerate(zip(row_phases, identities, strict=True))
        ),
    )
    order = range(count) if sampler_order is None else sampler_order
    captures = tuple(
        HiddenRows(binding, tensor, np.arange(count, dtype=np.int64))
        for binding, tensor in sorted((hidden or {}).items(), key=lambda item: item[0].locator_key)
    )
    return StepInput(
        rows,
        PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, row) for i, row in enumerate(order))),
        HiddenRowBatch(rows, captures) if captures else None,
        operands=operands,
        batch=batch,
        hidden_readiness=None if stream is None else CudaStreamReadiness(stream),
    )


def run_step(
    execution: StateExecution[StepInput, DataT, ResultT],
    value: StepInput,
    *,
    feedback: SamplerFeedback | None = None,
    stream: torch.cuda.Stream | None = None,
) -> ResultT:
    """Execute every anchor of one step, completing the sampler with `feedback` when the workload reads it."""
    return execution.step(value, stream, feedback=feedback)


def run_guided_step(
    execution: StateExecution[StepInput, DataT, ResultT],
    value: StepInput,
    *,
    feedback: SamplerFeedback | None = None,
    stream: torch.cuda.Stream | None = None,
) -> tuple[PackedSamplerGuidanceBatch, ResultT]:
    """Like `run_step`, also returning the sampler guidance the workload produced before sampling."""
    context = execution.start_step(value, stream)
    guidance: PackedSamplerGuidanceBatch | None = None
    for anchor in execution.arena.plan.timeline.anchors:
        execution.execute_anchor(context, AnchorId(anchor.ordinal))
        if anchor.ordinal == execution.sampler_anchor_id:
            guidance = execution.sampler_guidance(context, stream)
            execution.complete_sampler(context, stream, feedback=feedback)
    assert guidance is not None
    return guidance, execution.finish_step(context, stream)


__all__ = [
    "RequestIdentity",
    "SamplerFeedback",
    "backend_facts",
    "compile_execution",
    "final_hidden_binding",
    "hidden_binding",
    "run_guided_step",
    "run_step",
    "step_input",
]

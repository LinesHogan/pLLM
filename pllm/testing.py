"""Drive a compiled workload step by step with hand-chosen rows, for unit tests and notebooks.

`step_input` builds one packed step (one row per request) the way the vLLM runner would hand it to the
workload; `run_step` and `run_guided_step` execute every anchor of that step, delivering sampler feedback
when the workload reads it. `hidden_binding` names a hidden tap exactly as the runner does, so the same
bindings can be signed into backend facts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

import torch

from pllm.contracts.auxiliary import AuxiliaryInput
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
    hidden_binding,
)
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.parameter_overlay import CapturedParameterOperand
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.state_execution import StateExecution
from pllm.execution.step_input import TensorStepInput

DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")


def step_input(
    identities: Sequence[RequestIdentity],
    hidden: Mapping[PackedHiddenResourceBinding, torch.Tensor] | None = None,
    *,
    step_id: int = 0,
    token_offset: int = 0,
    phases: Sequence[HiddenPhase] | None = None,
    sampler_order: Sequence[int] | None = None,
    auxiliary: AuxiliaryInput[torch.Tensor] | None = None,
    operands: tuple[CapturedParameterOperand, ...] = (),
    stream: torch.cuda.Stream | None = None,
) -> TensorStepInput:
    """One packed step: row i belongs to `identities[i]` and decodes unless `phases` says otherwise.

    `sampler_order` lists the packed row behind each sampler row (identity order by default; `()` for a
    step without sampling). Hidden tensors are given per tap, one row per identity; CUDA tensors need the
    `stream` they were produced on.
    """
    count = len(identities)
    row_phases = tuple(phases) if phases is not None else (HiddenPhase.DECODE,) * count
    rows = PackedPhaseRowBatch(
        1,
        step_id,
        count,
        tuple(
            PackedPhaseRow(i, i, token_offset, phase, identity)
            for i, (phase, identity) in enumerate(zip(row_phases, identities, strict=True))
        ),
    )
    order = range(count) if sampler_order is None else sampler_order
    snapshots = tuple(
        PackedHiddenSnapshot(binding, tensor)
        for binding, tensor in sorted((hidden or {}).items(), key=lambda item: item[0].locator_key)
    )
    return TensorStepInput(
        rows,
        PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, row) for i, row in enumerate(order))),
        PackedHiddenSnapshotBatch(rows, snapshots) if snapshots else None,
        operands=operands,
        auxiliary=auxiliary,
        hidden_readiness=None if stream is None else CudaStreamReadiness(stream),
    )


def run_step(
    execution: StateExecution[TensorStepInput, DataT, ResultT],
    value: TensorStepInput,
    *,
    feedback: SamplerFeedback | None = None,
    stream: torch.cuda.Stream | None = None,
) -> ResultT:
    """Execute every anchor of one step, completing the sampler with `feedback` when the workload reads it."""
    context = execution.start_step(value, stream)
    for anchor in execution.arena.plan.timeline.anchors:
        execution.execute_anchor(context, AnchorId(anchor.ordinal))
        if anchor.ordinal == execution.sampler_anchor_id:
            execution.complete_sampler(context, stream, feedback=feedback)
    return execution.finish_step(context)


def run_guided_step(
    execution: StateExecution[TensorStepInput, DataT, ResultT],
    value: TensorStepInput,
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
    return guidance, execution.finish_step(context)


__all__ = [
    "HiddenPhase",
    "HiddenSite",
    "RequestIdentity",
    "SamplerFeedback",
    "hidden_binding",
    "run_guided_step",
    "run_step",
    "step_input",
]

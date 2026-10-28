"""Drive in-forward overlay segments at model projections through the existing state task runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, NamedTuple, Protocol, TypeVar

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterEffect,
    ParameterOverlayOperator,
    ParameterTarget,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerRowBatch
from pllm.contracts.semantic_graph import TaskId
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.chunk_tail import join_tail, save_tail
from pllm.execution.cuda_stream import upload
from pllm.execution.scan import scan
from pllm.execution.scan_execution import Step
from pllm.execution.tensor_frame import OverlayEntry, OverlayTargetResult, TensorFrame, Tensors, TensorUpdate
from pllm.execution.tensor_program import PoolSGD, TaskOutput, TensorBatch
from pllm.kernels.head_overlay import head_correction
from pllm.kernels.page_copy import copy_pages

DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
if TYPE_CHECKING:
    from pllm.execution.state_execution import StateExecution, StateTaskContext
    from pllm.execution.step_input import TensorStepInput


def chunk_update(step: Step, chunk: int, arguments: tuple[str, ...], /, **values: torch.Tensor) -> TensorUpdate:
    """Adapt an authored chunk recurrence to the ordinary overlay update boundary."""
    carry, *columns = arguments
    rows = tuple(values[name] for name in columns)
    tails = tuple(values[f"_pending_rows_{name}"] for name in columns)
    length = values["_pending_length"]
    joined, valid = join_tail(tails, rows, length, chunk)
    positions = torch.arange(len(rows[0]), device=length.device) + length
    trained = valid // chunk * chunk
    updated, outputs = scan(step, carry=values[carry], over=joined, chunk=chunk, valid_rows=trained)
    values[carry].copy_(updated)
    remaining = save_tail(tails, joined, length, valid, trained)
    return TensorUpdate(
        torch.cat(outputs).index_select(0, positions),
        trained > 0,
        trained // chunk,
        trained,
        {"tail_rows": remaining},
    )


def overlay_apply(effect: ParameterEffect, hidden: torch.Tensor, values: Tensors) -> torch.Tensor:
    """Read the declared overlay operator for ordinary and chunk-exclusive projections."""
    applied = effect.applied_fields
    if effect.operator is ParameterOverlayOperator.HEAD_ADD:
        return head_correction(hidden, values[applied[0]])
    if effect.operator is ParameterOverlayOperator.LOW_RANK_ADD:
        return (hidden @ values[applied[0]].T) @ values[applied[1]].T
    return hidden @ values[applied[0]].T


Gradient = Callable[[Tensors, Tensors, torch.Tensor], Tensors]


def overlay_gradient(function: Callable[..., torch.Tensor]) -> Gradient:
    """Differentiate the sum of valid per-row losses, leaving chunk normalization to execution."""

    def loss(parameters: Tensors, operands: Tensors, valid: torch.Tensor) -> torch.Tensor:
        rows = function(**parameters, **operands)
        if rows.shape != valid.shape:
            raise ValueError("autograd overlays must return one loss per input row")
        return torch.where(valid, rows, 0).sum()

    eager = torch.func.grad(loss)
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        eager, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )

    def gradient(parameters: Tensors, operands: Tensors, valid: torch.Tensor) -> Tensors:
        # Planning traces the unfused allocations; real invocations warm before capture.
        return (eager if valid.is_meta else compiled)(parameters, operands, valid)

    return gradient


def gradient_update(
    gradient: Gradient,
    optimizer: PoolSGD,
    effect: ParameterEffect,
    chunk: int,
    flush: bool,
    bindings: Mapping[str, str],
    /,
    **values: torch.Tensor,
) -> TensorUpdate:
    """Accumulate arriving-row gradients in PTTS and update once per completed chunk."""
    parameters = {name: values[name] for name, source in bindings.items() if source in effect.applied_fields}
    row_names = tuple(name for name, source in bindings.items() if source in ("input", "target"))
    input_name = next(name for name, source in bindings.items() if source == "input")
    hidden = values[input_name]
    state = {source: parameters[name] for name, source in bindings.items() if name in parameters}
    sums = tuple(values[f"_pending_grad_{name}"] for name in parameters)
    length = values["_pending_length"]
    offset = length.clone()
    positions = torch.arange(chunk, device=hidden.device)
    outputs: list[torch.Tensor] = []
    updates, trained = torch.zeros_like(length), torch.zeros_like(length)
    with torch.inference_mode(False), torch.enable_grad():
        constants = {name: values[name] for name, source in bindings.items() if source == "weight"}
        for start in range(0, len(hidden) + chunk - 1, chunk):
            indices = positions + start - offset
            valid = (indices >= 0) & (indices < len(hidden))
            rows = {name: values[name].index_select(0, indices.clamp(0, len(hidden) - 1)) for name in row_names}
            outputs.append(overlay_apply(effect, rows[input_name], state))
            gradients = gradient(
                {name: value.clone() for name, value in parameters.items()}, {**constants, **rows}, valid
            )
            length.add_(valid.sum())
            complete = (length == chunk) | (
                flush & values["_prompt_end"] & (start + chunk >= offset + len(hidden)) & (length > 0)
            )
            for total, grad in zip(sums, gradients.values(), strict=True):
                total.add_(grad)
            optimizer.tensors(
                tuple(parameters.values()),
                tuple(
                    torch.where(complete, total / length.clamp_min(1), 0).to(parameter.dtype)
                    for total, parameter in zip(sums, parameters.values(), strict=True)
                ),
            )
            updates.add_(complete)
            trained.add_(torch.where(complete, length, 0))
            for total in sums:
                total.masked_fill_(complete, 0)
            length.masked_fill_(complete, 0)
    output = torch.cat(outputs).index_select(0, torch.arange(len(hidden), device=hidden.device) + offset)
    return TensorUpdate(output, updates > 0, updates, trained, {"tail_rows": length.clone()})


class ProjectionTransition(Protocol):
    """The output boundary consumed by a model projection."""

    def run(
        self, operands: tuple[CapturedParameterOperand, ...]
    ) -> Mapping[tuple[RequestIdentity, ParameterTarget], torch.Tensor]: ...


class ProjectionTask(Generic[DataT, ResultT]):
    """One invocation's projection segments; private writes remain owned by StateExecution."""

    def __init__(
        self,
        execution: StateExecution[TensorStepInput, DataT, ResultT],
        context: StateTaskContext[DataT],
        rows: PackedSamplerRowBatch,
        task: TaskId,
        targets: tuple[ParameterTarget, ...],
    ) -> None:
        self.execution, self.context, self.rows, self.task = execution, context, rows, task
        self.pending = set(targets)

    def run(
        self, operands: tuple[CapturedParameterOperand, ...]
    ) -> Mapping[tuple[RequestIdentity, ParameterTarget], torch.Tensor]:
        """Execute available target segments and hand their residuals to the model reader."""
        from pllm.execution.step_input import TensorStepInput

        targets = {operand.target for operand in operands}
        assert targets and targets <= self.pending
        self.pending.difference_update(targets)
        return self.execution.execute_part(
            self.context,
            TensorStepInput(self.rows.row_batch, self.rows, operands=operands),
            self.task,
            complete=not self.pending,
        )


@dataclass(frozen=True, slots=True)
class OverlayTransition:
    """Request/target adaptation shared by source and explicitly lowered math.

    Replacing this immutable callable during binding changes only the imported
    mathematics. The original TensorOperation still owns private-page access.
    """

    buffers: tuple[str, ...]
    targets: tuple[ParameterTarget, ...]
    phases: tuple[HiddenPhase, ...]
    apply: Callable[[torch.Tensor, Tensors], torch.Tensor]
    function: Callable[..., TensorUpdate]
    arguments: Mapping[str, str]
    visibility: VisibilitySource

    def __call__(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        operands = {
            (operand.identity, operand.target): operand
            for operand in data.step.operands
            if self.visibility.in_forward or operand.phase in self.phases
        }
        if not operands:
            return TaskOutput()
        selected = dict.fromkeys(
            (row.identity, index)
            for row, index in zip(batch.selection.rows, batch.selection.state_indices, strict=True)
        )
        indices = {identity: index for identity, index in selected}
        updates: dict[tuple[RequestIdentity, ParameterTarget], TensorUpdate] = {}
        with torch.no_grad():
            for target_index, target in enumerate(self.targets):
                for train in (False, True):
                    group = tuple(
                        operand
                        for (_, name), operand in operands.items()
                        if name == target and (operand.phase in self.phases) == train
                    )
                    if group:
                        results = self.group(group, target_index, tuple(indices[o.identity] for o in group), batch)
                        updates.update(((o.identity, target), result) for o, result in zip(group, results, strict=True))
        start = len(data.entries)
        changed: dict[int, torch.Tensor] = {}
        for identity, index in selected:
            results = tuple(
                OverlayTargetResult(target, updates[identity, target])
                for target in self.targets
                if (identity, target) in updates
            )
            if results:
                data.entries.append(
                    OverlayEntry(identity, batch.selection.refs[index], batch.after_refs[index], results)
                )
                changed[index] = torch.stack(tuple(result.update.changed for result in results)).any()
        return TaskOutput(
            residuals={
                (entry.identity, target.target): target.update.output
                for entry in data.entries[start:]
                for target in entry.targets
            },
            changed=tuple(changed),
            update_mask=torch.stack(tuple(changed.values())),
            counters=torch.stack(
                tuple(
                    torch.stack((target.update.update_count, target.update.trained_rows))
                    for entry in data.entries[start:]
                    for target in entry.targets
                )
            ).sum(0),
        )

    def group(
        self,
        operands: tuple[CapturedParameterOperand, ...],
        target: int,
        indices: tuple[int, ...],
        batch: TensorBatch,
    ) -> tuple[TensorUpdate, ...]:
        """Gather issued slots once per target/phase; settlement still belongs to the task."""
        width = len(self.buffers)
        fields = tuple(batch.field_slots(target * width + i) for i in range(width))
        pools = tuple(pool for pool, _ in fields)
        slots = upload(tuple(tuple(slots[i] for i in indices) for _, slots in fields), pools[0].device)
        values = tuple(pool.index_select(0, row) for pool, row in zip(pools, slots, strict=True))
        results = tuple(
            self.result(operand, dict(zip(self.buffers, (value[i] for value in values), strict=True)))
            for i, operand in enumerate(operands)
        )
        if operands[0].phase in self.phases:
            sources = torch.arange(len(operands), device=pools[0].device)
            for value, pool, destinations in zip(values, pools, slots, strict=True):
                copy_pages(value, pool, sources, destinations)
        return results

    def result(self, operand: CapturedParameterOperand, values: Tensors) -> TensorUpdate:
        """Validate the author's result at its tensor boundary without observing CUDA values."""
        inputs = {
            **values,
            "input": operand.hidden,
            "target": operand.desired,
            "weight": operand.base_weight,
            "prompt_end": operand.prompt_end,
        }
        result = (
            self.function(**{key: inputs[source] for key, source in self.arguments.items()})
            if operand.phase in self.phases
            else TensorUpdate.unchanged(self.apply(operand.hidden, values))
        )
        if (
            result.output.shape != operand.desired.shape
            or result.output.dtype not in (operand.hidden.dtype, torch.float32)
            or result.output.device != operand.hidden.device
        ):
            raise ValueError("overlay output differs from its target shape, dtype or device")
        if any(
            value.shape != () or value.device != operand.hidden.device or value.dtype != dtype
            for value, dtype in (
                (result.changed, torch.bool),
                (result.update_count, torch.int64),
                (result.trained_rows, torch.int64),
            )
        ):
            raise ValueError("overlay counters need a bool changed flag and int64 counts on the operand device")
        return result


class RouteOperand(NamedTuple):
    """Capture outputs without retaining the cold invocation's request or phase."""

    target: ParameterTarget
    hidden: torch.Tensor
    desired: torch.Tensor
    weight: torch.Tensor


class RouteOverlay:
    """Read and capture at projections; apply grouped operands within one owned transition."""

    def __init__(
        self,
        math: OverlayTransition,
        batch: TensorBatch,
        entry: TensorBatch,
        data: TensorFrame,
    ) -> None:
        self.math, self.batch, self.entry, self.data = math, batch, entry, data
        self.entry_fields: tuple[torch.Tensor, ...] = ()
        self.fields: tuple[torch.Tensor, ...] = ()
        self.saved: tuple[torch.Tensor, ...] = ()
        self.gate = torch.zeros((1, 1), device=batch.roots[0][0].device)
        self.prompt_end = torch.zeros((), device=batch.roots[0][0].device, dtype=torch.bool)
        self.pending: list[CapturedParameterOperand] = []
        self.ordinary: list[CapturedParameterOperand] = []
        self.results: list[OverlayTargetResult] = []
        self.row = 0
        self.routed = self.private = False

    def stage(self, row: int, routed: bool) -> None:
        """Resolve input/private roots once, outside cold execution and graph replay."""
        self.row, self.routed = row, routed
        if not self.fields:
            self.entry_fields, self.fields, self.saved = (
                tuple(torch.empty_like(field) for field in self.batch.roots[0]) for _ in range(3)
            )
        index = self.batch.selection.state_indices[row]
        for ports, fields in (
            (self.entry_fields, self.entry.roots[index]),
            (self.saved, self.batch.roots[index]),
        ):
            for target, source in zip(ports, fields, strict=True):
                target.copy_(source)

    def reset(self) -> None:
        """Restore every cold execution from this token's incoming private fields."""
        for target, source in zip(self.fields, self.saved, strict=True):
            target.copy_(source)
        self.pending, self.results = [], []
        self.private = self.math.visibility.in_forward
        self.gate.zero_()

    def values(self, target: ParameterTarget, private: bool) -> Tensors:
        width = len(self.math.buffers)
        start = self.math.targets.index(target) * width
        return dict(
            zip(
                self.math.buffers,
                (self.fields if private else self.entry_fields)[start : start + width],
                strict=True,
            )
        )

    def run(
        self, operands: tuple[CapturedParameterOperand, ...]
    ) -> Mapping[tuple[RequestIdentity, ParameterTarget], torch.Tensor]:
        """Own operands immediately because later model calls may mutate their inputs."""
        outputs: dict[tuple[RequestIdentity, ParameterTarget], torch.Tensor] = {}
        for operand in operands:
            captured = replace(
                operand, hidden=operand.hidden.detach().clone(), desired=operand.desired.detach().clone()
            )
            if not self.routed and self.math.visibility.in_forward:
                result = self.math.result(captured, self.values(operand.target, True))
                self.results.append(OverlayTargetResult(operand.target, result))
                output = result.output
            else:
                visible = {
                    name: torch.where(
                        self.gate.reshape(()).bool(),
                        value,
                        self.values(operand.target, self.math.visibility.in_forward)[name],
                    )
                    for name, value in self.values(operand.target, True).items()
                }
                output = self.math.apply(operand.hidden, visible)
                self.pending.append(captured)
            outputs[operand.identity, operand.target] = output
        return outputs

    def apply_updates(self) -> None:
        """Concatenate every encounter at a target before invoking the author's batch math."""
        for target in dict.fromkeys(operand.target for operand in self.pending):
            records = tuple(operand for operand in self.pending if operand.target == target)
            operand = replace(
                records[0],
                hidden=torch.cat(tuple(record.hidden for record in records)),
                desired=torch.cat(tuple(record.desired for record in records)),
                prompt_end=records[-1].prompt_end,
            )
            result = self.math.result(operand, self.values(target, True))
            self.results.append(OverlayTargetResult(target, result))
        if self.pending:
            self.private = True
            self.gate.fill_(1.0)
        self.pending = []

    def captures(self) -> tuple[RouteOperand, ...]:
        """Export graph outputs independently of the cold request identity."""
        return tuple(
            RouteOperand(operand.target, operand.hidden, operand.desired, operand.base_weight)
            for operand in self.pending
        )

    def collect(self, results: tuple[OverlayTargetResult, ...], pending: tuple[RouteOperand, ...]) -> None:
        """Reconstruct invocation metadata from device outputs, including cached graph replays."""
        row = self.data.step.rows.rows[self.row]
        index = self.batch.selection.state_indices[self.row]
        if self.batch.after_refs[index] != self.batch.selection.refs[index]:
            for target, source in zip(self.batch.roots[index], self.fields, strict=True):
                target.copy_(source)
        copied = tuple(
            OverlayTargetResult(
                item.target,
                replace(
                    item.update,
                    output=item.update.output.clone(),
                    changed=item.update.changed.clone(),
                    update_count=item.update.update_count.clone(),
                    trained_rows=item.update.trained_rows.clone(),
                    metrics={key: value.clone() for key, value in item.update.metrics.items()},
                ),
            )
            for item in results
        )
        if copied:
            self.data.entries.append(
                OverlayEntry(row.identity, self.batch.selection.refs[index], self.batch.after_refs[index], copied)
            )
        if not self.routed:
            self.ordinary.extend(
                CapturedParameterOperand(
                    row.identity,
                    operand.target,
                    row.phase,
                    operand.hidden.clone(),
                    operand.desired.clone(),
                    operand.weight,
                    operand.hidden.new_full(
                        (), row.token_offset + len(operand.hidden) == row.prompt_length, dtype=torch.bool
                    ),
                )
                for operand in pending
            )

    def finish(self) -> TaskOutput:
        """Apply ordinary pinned rows once as a batch; routed trailing captures are discarded."""
        grouped: list[CapturedParameterOperand] = []
        for identity, target in dict.fromkeys((operand.identity, operand.target) for operand in self.ordinary):
            records = tuple(
                operand for operand in self.ordinary if (operand.identity, operand.target) == (identity, target)
            )
            grouped.append(
                replace(
                    records[0],
                    hidden=torch.cat(tuple(record.hidden for record in records)),
                    desired=torch.cat(tuple(record.desired for record in records)),
                    prompt_end=records[-1].prompt_end,
                )
            )
        self.math(replace(self.data, step=replace(self.data.step, operands=tuple(grouped))), self.batch)
        return TaskOutput(
            counters=None
            if not self.data.entries
            else torch.stack(
                tuple(
                    torch.stack((target.update.update_count, target.update.trained_rows))
                    for entry in self.data.entries
                    for target in entry.targets
                )
            ).sum(0)
        )

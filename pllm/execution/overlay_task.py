"""Drive in-forward overlay segments at model projections through the existing state task runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, NamedTuple, Protocol, TypeVar

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import CapturedParameterOperand, ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerRowBatch
from pllm.contracts.semantic_graph import TaskId
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.tensor_frame import OverlayEntry, OverlayTargetResult, TensorFrame, Tensors, TensorUpdate
from pllm.execution.tensor_program import TaskOutput, TensorBatch

DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
if TYPE_CHECKING:
    from pllm.execution.execution import TaskGraphInvocation
    from pllm.execution.state_execution import ProjectionResiduals, StateExecution, StateTaskContext
    from pllm.execution.step_input import TensorStepInput


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
        invocation: TaskGraphInvocation[StateTaskContext[DataT], ProjectionResiduals],
        rows: PackedSamplerRowBatch,
        task: TaskId,
        targets: tuple[ParameterTarget, ...],
    ) -> None:
        self.execution, self.invocation, self.rows, self.task = execution, invocation, rows, task
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
            self.invocation,
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
    function: Callable[[Tensors], TensorUpdate]
    visibility: VisibilitySource

    def __call__(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        operands = {
            (operand.identity, operand.target): operand
            for operand in data.step.operands
            if self.visibility.in_forward or operand.phase in self.phases
        }
        if not operands:
            return TaskOutput()
        start = len(data.entries)
        changed: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for identity, index in dict.fromkeys(
                (row.identity, index)
                for row, index in zip(batch.selection.rows, batch.selection.state_indices, strict=True)
            ):
                results: list[OverlayTargetResult] = []
                for target_index, target in enumerate(self.targets):
                    if (identity, target) not in operands:
                        continue
                    operand = operands[identity, target]
                    values = dict(
                        zip(
                            self.buffers,
                            batch.roots[index][
                                len(self.buffers) * target_index : len(self.buffers) * (target_index + 1)
                            ],
                            strict=True,
                        )
                    )
                    result = self.result(operand, values)
                    results.append(OverlayTargetResult(target, result))
                    changed[index] = result.changed if index not in changed else changed[index] | result.changed
                if results:
                    data.entries.append(
                        OverlayEntry(identity, batch.selection.refs[index], batch.after_refs[index], tuple(results))
                    )
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

    def result(self, operand: CapturedParameterOperand, values: Tensors) -> TensorUpdate:
        """Validate the author's result at its tensor boundary without observing CUDA values."""
        result = (
            self.function(
                {
                    **values,
                    "input": operand.hidden,
                    "target": operand.desired,
                    "weight": operand.base_weight,
                    "prompt_end": operand.prompt_end,
                }
            )
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

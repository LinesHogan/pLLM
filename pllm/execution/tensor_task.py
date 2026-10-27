"""One declared tensor method, executed over the rows of one task invocation.

A TensorTask reads the method's inputs for the current rows, calls it, and stores the result. A
training task also differentiates its loss with respect to the selected states' parameters and
hands the per-state gradients to the optimizer. TensorBinding builds one per declared method.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.semantic_graph import (
    MathContract,
    SemanticInputSource,
    StateResource,
    StateSelectionInputSource,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.declaration import Call
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.tensor_frame import (
    LeafRoots,
    TensorFrame,
    TensorMath,
    TensorPort,
    gradient_rows,
    row_index,
    row_plan,
    select_rows,
    selected_rows,
    state_leaf,
    state_means,
)
from pllm.execution.tensor_program import TaskOutput, TensorBatch


@dataclass(frozen=True)
class FieldWriter:
    """Write a task result into its declared field; used by the write lowering."""

    source: Call | None
    fields: tuple[FieldView, ...]
    resource: StateResource
    phase: HiddenPhase | None

    def values(
        self, data: TensorFrame, batch: TensorBatch, selected: tuple[int, ...], *, validate: bool
    ) -> tuple[torch.Tensor, ...]:
        """Validate field metadata before staging caller-owned results."""
        instance = batch.selection.for_resource(self.resource).keys == (None,)
        values = (
            tuple(batch.results[self.source.symbol, name] for name in self.source.outputs)
            if self.source is not None
            else tuple(
                torch.stack(tuple(fields[i] for _, fields in data.replacements)) for i in range(len(self.fields))
            )
        )
        values = tuple(value.unsqueeze(0) for value in values) if instance and self.source is not None else values
        if validate:
            for view, value in zip(self.fields, values, strict=True):
                target = view.read(batch.roots[selected[0]][view.field])
                if (
                    value.shape != (len(selected), *view.shape)
                    or value.dtype != target.dtype
                    or value.device != target.device
                ):
                    raise ValueError(
                        f"write to {self.resource} needs finite fields with their declared shape, dtype, device and request axis"
                    )
        return values

    def select(self, data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
        """Select only this producer's eligible resource rows."""
        if self.source is None:
            return tuple(index for index, _ in data.replacements)
        group = selection.for_resource(self.resource)
        if self.phase is not None and not data.step.rows.producer_rows(self.phase):
            return ()
        if group.keys == (None,):
            return tuple(selection.refs.index(ref) for ref in group.refs)
        indices = {row.identity: index for row, index in zip(selection.rows, group.state_indices, strict=True)}
        return tuple(dict.fromkeys(indices[row.identity] for row in data.step.rows.producer_rows(self.phase)))


@dataclass(frozen=True, slots=True)
class TensorTask:
    """One row-wise method and its input and gradient policies."""

    symbol: TaskSymbol
    function: TensorMath
    ports: Mapping[str, TensorPort]
    sources: Mapping[str, SemanticInputSource]
    policy: MathContract
    outputs: tuple[str, ...]
    # Memory fields written from inputs at publication; only training tasks have them.
    captures: Mapping[str, LayoutFieldId]
    phase: HiddenPhase | None = None
    resource: StateResource = StateResource("")
    writer: FieldWriter | None = None

    def dependencies(self, tasks: Mapping[TaskSymbol, TensorTask]) -> tuple[TensorTask, ...]:
        """This task's tensor DAG in dependency order, ending with the task itself."""
        ordered: dict[TaskSymbol, TensorTask] = {}
        for source in self.sources.values():
            if isinstance(source, TaskResultInputSource):
                ordered.update((task.symbol, task) for task in tasks[source.task].dependencies(tasks))
        ordered[self.symbol] = self
        return tuple(ordered.values())

    @property
    def domains(self) -> tuple[str, ...]:
        """Inputs that are boolean row masks (row selector, gradient rows, update states)."""
        policy = self.policy
        keys = (policy.forward_rows, policy.input_gradient_rows, policy.parameter_gradient_rows, policy.update_states)
        return tuple(
            dict.fromkeys(key for key in keys if key and not isinstance(self.sources[key], StateSelectionInputSource))
        )

    @property
    def differentiable(self) -> bool:
        return bool(self.policy.input_gradients or self.policy.parameter_gradients)

    def read(
        self, data: TensorFrame, batch: TensorBatch, *, compact: bool = True
    ) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
        """The function's inputs and the operation rows they cover.

        With `compact=False` every row is evaluated and the caller masks the result, which avoids
        reading the row selector back to the host.
        """
        policy = self.policy
        if policy.parameter_gradients:
            batch = replace(batch, roots=LeafRoots(state_leaf(data, batch), batch.roots))
        domains = self.domains
        inputs = {key: self.ports[key].read(data, batch) for key in domains}
        for key in domains:
            mask = inputs[key]
            if mask.dtype != torch.bool or mask.shape != (len(data.rows),):
                raise ValueError("row domain must be a boolean mask over the operation rows")
        compacted = bool(policy.forward_rows) and compact
        rows = selected_rows(inputs[policy.forward_rows]) if compacted else tuple(range(len(data.rows)))
        selected = (
            replace(
                data,
                rows=tuple(data.rows[index] for index in rows),
                index=row_index(data, batch, inputs[policy.forward_rows].device).select(rows),
                plan=None if data.plan is None else data.plan.select(rows),
            )
            if compacted
            else data
        )
        for key in self.ports:
            inputs[key] = self._input(key, inputs, data, selected, batch, rows, compacted)
        return inputs, rows

    def _input(
        self,
        key: str,
        inputs: dict[str, torch.Tensor],
        data: TensorFrame,
        selected: TensorFrame,
        batch: TensorBatch,
        rows: tuple[int, ...],
        compacted: bool,
    ) -> torch.Tensor:
        """Read one input and attach the gradient behaviour its declaration asks for."""
        policy, port, source = self.policy, self.ports[key], self.sources[key]
        state_rows = isinstance(source, StateSelectionInputSource) and port.rowwise
        value = inputs[key] if key in self.domains else port.read(selected if state_rows else data, batch)
        differentiable = key not in policy.detached_inputs and (
            source.resource in policy.parameter_gradients
            if isinstance(source, StateSelectionInputSource)
            else key in policy.input_gradients
        )
        if not differentiable:
            value = value.detach()
        elif not isinstance(source, (StateSelectionInputSource, TaskResultInputSource)):
            if not value.is_floating_point():
                raise ValueError("input gradients require floating tensors")
            value = value.detach().requires_grad_(True)
            data.derivative_sources[str(self.symbol), key] = value
        if compacted and port.rowwise and key not in self.domains and not state_rows:
            mask = inputs[policy.forward_rows]
            if value.ndim == 0 or value.shape[0] != len(data.rows) or value.device != mask.device:
                raise ValueError("forward rows require aligned row tensors on the same device")
            value = select_rows(value, rows)
        domain = (
            policy.parameter_gradient_rows
            if isinstance(source, StateSelectionInputSource)
            else policy.input_gradient_rows
        )
        if differentiable and domain:
            mask = select_rows(inputs[domain], rows) if compacted else inputs[domain]
            if not port.rowwise or value.ndim == 0 or value.shape[0] != len(rows) or mask.device != value.device:
                raise ValueError("gradient rows require aligned row tensors on the same device")
            if value.requires_grad:
                value = gradient_rows(value, mask.reshape(len(rows), *([1] * (value.ndim - 1))))
        return value

    def evaluate(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        """Run a forward method and store its result for later tasks and the invocation output."""
        differentiable = self.differentiable
        with torch.set_grad_enabled(differentiable):
            # Pure predictions run on every row and mask the result; differentiable forwards still
            # compact so gradients see only the selected rows.
            inputs, rows = self.read(data, batch, compact=differentiable and data.weights is None)
            results = self.function(inputs)
            if type(results) is not tuple or len(results) != len(self.outputs):
                raise ValueError("write must return a tuple matching its declared fields")
            for name, result in zip(self.outputs, results, strict=True):
                for borrowed in batch.borrowed_roots:
                    borrowed.check_output(result)
                batch.results[self.symbol, name] = (
                    self.expand_selected(result, inputs, rows, data) if self.policy.forward_rows else result
                )
        return TaskOutput()

    def expand_selected(
        self, result: torch.Tensor, inputs: dict[str, torch.Tensor], rows: tuple[int, ...], data: TensorFrame
    ) -> torch.Tensor:
        """Give unselected rows a zero result, on the device."""
        if result.ndim == 0 or result.shape[0] != len(rows):
            raise ValueError("selected forward functions must preserve their row axis")
        mask = inputs[self.policy.forward_rows]
        if not self.differentiable or data.weights is not None:
            zero = torch.zeros((), dtype=result.dtype, device=result.device)
            return torch.where(mask.reshape(-1, *(1,) * (result.ndim - 1)), result, zero)
        if len(rows) == len(data.rows):
            return result
        return result.new_zeros((len(data.rows), *result.shape[1:])).index_copy_(0, mask.nonzero().flatten(), result)

    def differentiate(
        self, inputs: dict[str, torch.Tensor], rows: tuple[int, ...], data: TensorFrame, batch: TensorBatch
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Shared eager and captured forward, loss reduction and autograd step."""
        (loss,) = self.function(inputs)
        loss, objective = self._loss(loss, rows, data, batch)
        leaf = state_leaf(data, batch)
        targets = (leaf, *data.derivative_sources.values())
        derivatives = (
            torch.autograd.grad(objective, targets, allow_unused=True, materialize_grads=True)
            if objective.requires_grad
            else tuple(torch.zeros_like(target) for target in targets)
        )
        return loss, derivatives

    def _loss(
        self, loss: torch.Tensor, rows: tuple[int, ...], data: TensorFrame, batch: TensorBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The reported loss and the scalar to differentiate."""
        if self.policy.reduction == "state_mean":
            if loss.shape != (len(rows),):
                raise ValueError("state_mean training returns one loss per row; pllm averages them per state")
            if data.weights is not None:
                assert data.index is not None and data.counts is not None
                loss = loss.new_zeros(len(data.counts)).index_add(
                    0, data.index.states.device, loss * data.weights
                ) / data.counts.clamp_min(1)
            else:
                loss = state_means(loss, row_plan(data, batch).select(rows))
            return loss, loss.sum()
        if loss.ndim == 1:
            loss = (
                loss.mean() if data.weights is None else (loss * data.weights).sum() / data.weights.sum().clamp_min(1)
            )
        elif data.weights is not None:
            raise ValueError("captured training must return one loss per row; move the row mean into pllm")
        if loss.ndim != 0:
            raise ValueError("optimizer functions must return a scalar or one loss per row")
        return loss, loss


__all__ = ["TensorTask"]

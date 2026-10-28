"""One declared tensor method, executed over the rows of one task invocation.

A TensorTask reads the method's inputs for the current rows, calls it, and stores the result. A
training task also differentiates its loss with respect to the selected states' parameters and
hands the per-state gradients to the optimizer. Export and sampler settings adapt the stored result
after execution. TensorBinding builds one per declared method.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import (
    MathContract,
    SemanticInputSource,
    StateResource,
    StateSelectionInputSource,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.declaration import Call, State
from pllm.execution.module_graph import ModuleCall
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
    take_rows,
)
from pllm.execution.tensor_program import Observation, TaskOutput, TensorBatch
from pllm.execution.trace import MAX_SNAPSHOT_BYTES


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
        if validate and any(
            value.shape != (len(selected), *view.shape)
            or value.dtype != (target := view.read(batch.roots[selected[0]][view.field])).dtype
            or value.device != target.device
            for view, value in zip(self.fields, values, strict=True)
        ):
            raise ValueError(
                f"write to {self.resource} needs fields with their declared shape, dtype, device and request axis"
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
class TensorExport:
    """Bounded observation settings carried by an exported TensorTask."""

    name: str
    every: int
    max_rows: int
    max_cols: int
    enabled: bool
    overflow: Literal["drop", "block"]

    def __post_init__(self) -> None:
        if min(self.every, self.max_rows, self.max_cols) < 1 or self.max_rows * self.max_cols * 8 > MAX_SNAPSHOT_BYTES:
            raise ValueError("tensor export requires positive stride and bounded snapshot dimensions")

    def prepare(self, data: TensorFrame) -> TensorFrame | None:
        """Skip disabled steps and bound the rows before evaluating tensor math."""
        if not self.enabled or data.step.rows.step_id % self.every or not data.rows:
            return None
        return replace(data, rows=data.rows[: self.max_rows], index=None, plan=None)

    def output(self, value: torch.Tensor, data: TensorFrame) -> TaskOutput:
        """Publish the bounded result with its row identities and delivery policy."""
        return TaskOutput(
            observations=(
                Observation(
                    self.name,
                    value[: self.max_rows, : self.max_cols],
                    step_id=data.step.rows.step_id,
                    identities=tuple(row.identity for row in data.rows),
                    export=self.overflow,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class TensorTask:
    """One row-wise method with its input, gradient and output policies."""

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
    # A write returns a tuple even when it declares only one field.
    tuple_output: bool = False
    export: TensorExport | None = None
    sampler_stage: SamplerStage | None = None
    sampler_beta: float = 0.0
    training: State | None = None

    def sampler_output(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        """Adapt a computed prediction to sampler rows without executing its mathematics."""
        result = batch.results[self.symbol, self.outputs[0]]
        stage = self.sampler_stage
        assert stage is not None
        mask = self.ports[self.policy.forward_rows] if self.policy.forward_rows else None
        local = {row.packed_row_index: index for index, row in enumerate(data.rows)}
        links = tuple(link for link in data.step.sampler_rows.links if link.packed_row_index in local)
        weights = None
        if stage is SamplerStage.PACKED_POST_FILTER_EXACT:
            positions = tuple(local[link.packed_row_index] for link in links)
            result = take_rows(result, positions)
            # Unselected rows keep weight 0, so the sampler leaves their logits unchanged;
            # selecting rows on the host would need a device-to-host read.
            if mask is not None:
                weights = take_rows(mask.read(data, batch), positions).to(result.dtype)
        observations = (
            tuple(
                Observation(
                    "guidance",
                    result[index],
                    data.step.rows.rows[link.packed_row_index],
                    batch.selection.refs[0]
                    if batch.selection.keys == (None,)
                    else batch.selection.ref_for(data.step.rows.rows[link.packed_row_index].identity),
                    data.step.rows.step_id,
                )
                for index, link in enumerate(links)
            )
            if batch.observing
            else ()
        )
        return TaskOutput(
            guidance=PackedSamplerGuidanceBatch(
                data.step.sampler_rows,
                result.detach(),
                tuple(link.sampler_row_index for link in links),
                self.sampler_beta,
                stage=stage,
                row_weights=weights,
            ),
            observations=observations,
        )

    def dependencies(self, tasks: Mapping[TaskSymbol, TensorTask]) -> tuple[TensorTask, ...]:
        """This task's tensor DAG in dependency order, ending with the task itself."""
        ordered: dict[TaskSymbol, TensorTask] = {}
        for source in self.sources.values():
            match source:
                case TaskResultInputSource(task=upstream):
                    ordered.update((task.symbol, task) for task in tasks[upstream].dependencies(tasks))
                case _:
                    pass
        return (*ordered.values(), self)

    @property
    def domains(self) -> tuple[str, ...]:
        """Inputs that are boolean row masks (row selector, gradient rows, update states)."""
        policy = self.policy
        keys = (policy.forward_rows, policy.input_gradient_rows, policy.parameter_gradient_rows, policy.update_states)
        return tuple(
            key
            for key in dict.fromkeys(keys)
            if key in self.sources and type(self.sources[key]) is not StateSelectionInputSource
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
        if any(inputs[key].dtype != torch.bool or inputs[key].shape != (len(data.rows),) for key in domains):
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
        match source:
            case StateSelectionInputSource(resource=state_resource):
                state_rows = port.rowwise
                differentiable = key not in policy.detached_inputs and state_resource in policy.parameter_gradients
                leaf = False
                domain = policy.parameter_gradient_rows
            case TaskResultInputSource():
                state_rows = False
                differentiable = key not in policy.detached_inputs and key in policy.input_gradients
                leaf = False
                domain = policy.input_gradient_rows
            case _:
                state_rows = False
                differentiable = key not in policy.detached_inputs and key in policy.input_gradients
                leaf = True
                domain = policy.input_gradient_rows
        value = inputs[key] if key in self.domains else port.read(selected if state_rows else data, batch)
        if not differentiable:
            value = value.detach()
        elif leaf:
            assert value.is_floating_point()
            value = value.detach().requires_grad_(True)
            data.derivative_sources[str(self.symbol), key] = value
        if compacted and port.rowwise and key not in self.domains and not state_rows:
            assert value.ndim > 0 and value.shape[0] == len(data.rows)
            value = select_rows(value, rows)
        if differentiable and domain:
            mask = select_rows(inputs[domain], rows) if compacted else inputs[domain]
            assert port.rowwise and value.ndim > 0 and value.shape[0] == len(rows)
            if value.requires_grad:
                value = gradient_rows(value, mask.reshape(len(rows), *([1] * (value.ndim - 1))))
        return value

    def arguments(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor | ModuleCall]:
        """Adapt method arguments after tensor selection and gradient policy, omitting row selectors."""
        domains = self.domains
        return {key: port.adapt(inputs[key]) for key, port in self.ports.items() if key not in domains}

    def evaluate(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        """Run a forward method and store its result for later tasks and the invocation output."""
        differentiable = self.differentiable
        with torch.set_grad_enabled(differentiable):
            # Pure predictions run on every row and mask the result; differentiable forwards still
            # compact so gradients see only the selected rows.
            inputs, rows = self.read(data, batch, compact=differentiable and data.weights is None)
            results = self.function(**self.arguments(inputs))
            if not self.tuple_output:
                if not isinstance(results, torch.Tensor):
                    raise ValueError("tensor methods must return one tensor")
                results = (results,)
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
        loss = self.function(**self.arguments(inputs))
        if not isinstance(loss, torch.Tensor):
            raise ValueError("training methods must return a loss tensor")
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
        elif loss.ndim != 0 or data.weights is not None:
            raise ValueError("optimizer functions must return a scalar or one loss per row")
        return loss, loss


__all__ = ["TensorTask"]

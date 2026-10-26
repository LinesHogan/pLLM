"""Shared row mathematics and settlement for compact and fixed tensor operations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

from pllm.contracts.parameter_overlay import FieldView
from pllm.contracts.semantic_graph import HiddenInputSource, TaskSymbol
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.step_capture import StepCapture
from pllm.execution.tensor_frame import (
    FieldRoots,
    MaskedRows,
    ParameterGradients,
    RowIndex,
    RowPlan,
    TensorFrame,
    row_plan,
    selected_rows,
    state_leaf,
    training_observations,
)
from pllm.execution.tensor_program import PageUpdate, TaskOutput, TensorBatch
from pllm.execution.tensor_task import TensorTask
from pllm.kernels.page_copy import gather_rows

Table = TypeVar("Table", torch.Tensor, npt.NDArray[np.int64])


@dataclass(frozen=True, slots=True)
class TableLayout:
    """Segments of a step's int64 index table, uploaded to the GPU in one copy per step.

    rows          [6, R]   per row: packed hidden row, sampler row, sample index, state, version, weight
    counts        [S + 1]  rows per state; index S is the padding sink
    last          [S]      last row of each state (memory-field writes)
    sources       [F, S+1] source page slot per field and state (-1: none)
    destinations  [F, S]   destination page slot per field and state (-1: do not write)
    feedback      [K, 3]   address and strides of each sampler-feedback tensor
    """

    rows: int
    states: int
    fields: int
    feedback: int

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        rows, states, fields = self.rows, self.states, self.fields
        return ((6, rows), (states + 1,), (states,), (fields, states + 1), (fields, states), (self.feedback, 3))

    @property
    def size(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes)

    def split(self, table: Table) -> tuple[Table, Table, Table, Table, Table, Table]:
        """Named views of `table` (a torch tensor or numpy array of `size` elements)."""
        views: list[Table] = []
        start = 0
        for shape in self.shapes:
            end = start + math.prod(shape)
            views.append(table[start:end].reshape(shape))
            start = end
        rows, counts, last, sources, destinations, feedback = views
        return rows, counts, last, sources, destinations, feedback


def reads_rows(task: TensorTask, tasks: Mapping[TaskSymbol, TensorTask]) -> bool:
    """Whether the task reads per-row inputs (hidden, feedback, request state), directly or upstream.

    Only such tasks have a fixed row table to capture; the rest evaluate compact rows.
    """
    return any(port.rowwise for item in task.dependencies(tasks) for port in item.ports.values())


@dataclass
class TensorExecutor:
    """One body and one result settlement for compact (eager) or masked (fixed-table) evaluation."""

    task: TensorTask
    tasks: Mapping[TaskSymbol, TensorTask]
    optimizer: PageUpdate | None = None
    compact: bool = True  # chosen at binding: compact host-selected rows, or masked fixed tables
    capture: StepCapture | None = None
    device: torch.device = torch.device("cpu")
    table: torch.Tensor | None = None
    pools: tuple[torch.Tensor, ...] = ()
    result: torch.Tensor | None = None
    gradients: torch.Tensor | None = None
    inputs: dict[str, torch.Tensor] = dataclass_field(default_factory=lambda: dict[str, torch.Tensor]())
    evaluated_rows: tuple[int, ...] = ()
    input_gradients: dict[tuple[str, str], torch.Tensor] = dataclass_field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    addresses: tuple[int, ...] = ()
    hidden_inputs: Mapping[tuple[str, str, str], torch.Tensor] | None = None
    active_rows: int = 0
    active_states: int = 0
    updated: tuple[int, ...] = ()
    read_fields: frozenset[int] = frozenset()
    feedback_names: tuple[str, ...] = ()
    feedback_values: tuple[torch.Tensor, ...] = ()
    staging: PinnedStaging = dataclass_field(default_factory=PinnedStaging)

    def __post_init__(self) -> None:
        if self.compact:
            return
        dependencies = self.task.dependencies(self.tasks)
        self.read_fields = frozenset(
            port.field for task in dependencies for port in task.ports.values() if port.field is not None
        )
        if any(task.policy.parameter_gradients for task in dependencies):
            self.read_fields |= {0}
        self.feedback_names = tuple(
            sorted({port.feedback for task in dependencies for port in task.ports.values() if port.feedback})
        )
        if (
            self.capture is not None
            and 8 * (6 * self.rows + self.state_count + 1) > self.capture.entitlement.resident_bytes
        ):
            raise ValueError("capture index table exceeds the task resident bytes")

    @property
    def rows(self) -> int:
        return self.capture.entitlement.input_shapes[0][0] if self.capture is not None else self.active_rows

    @property
    def state_count(self) -> int:
        return self.capture.states if self.capture is not None else self.active_states

    @property
    def layout(self) -> TableLayout:
        return TableLayout(self.rows, self.state_count, len(self.pools), len(self.feedback_names))

    def _fill(self, host: npt.NDArray[np.int64], plan: RowPlan) -> None:
        """Stage a RowPlan and feedback addresses into the zeroed fixed table."""
        layout = self.layout
        rows, counts, last, sources, destinations, feedback = layout.split(host)
        count = len(plan.packed)
        if self.feedback_names and -1 in plan.sampler:
            raise ValueError("sampler feedback must cover every operation row")
        rows[3] = layout.states
        rows[:5, :count] = plan.columns
        rows[1, :count] = np.maximum(rows[1, :count], 0)
        rows[5, :count] = 1
        counts[: len(plan.sources) + 1] = plan.counts
        if len(plan.sources):
            positions = plan.last
            last[list(positions)] = tuple(positions.values())
        sources[:] = destinations[:] = -1
        sources[:, : len(plan.sources)] = plan.sources.T
        if plan.writes is not None:
            updated, slots = plan.writes
            destinations[:, list(updated)] = slots.T
        for row, value in zip(feedback, self.feedback_values, strict=True):
            row[:] = (value.data_ptr(), value.stride(0), value.stride(1) if value.ndim > 1 else 1)

    def _views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.table is not None
        rows, counts, _, sources, destinations, _ = self.layout.split(self.table)
        return rows, counts, sources, destinations

    def _feedback(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.table is not None
        locations = self.layout.split(self.table)[5]
        return {
            name: gather_rows(location, rows, value)
            for name, location, value in zip(self.feedback_names, locations, self.feedback_values, strict=True)
        }

    def _frame(self, data: TensorFrame, batch: TensorBatch) -> tuple[TensorFrame, TensorBatch]:
        columns, counts, sources, _ = self._views()
        reads = self.read_fields
        if self.optimizer is not None and (self.device.type == "cpu" or self.pools[0].dtype != torch.float32):
            reads = reads | frozenset(range(5))
        fields = tuple(
            pool.index_select(0, slots.clamp_min(0)).masked_fill(
                (slots < 0).reshape(-1, *(1 for _ in pool.shape[1:])), 0
            )
            if field in reads
            else pool.new_empty((self.state_count + 1, 0))
            for field, (pool, slots) in enumerate(zip(self.pools, sources, strict=True))
        )
        frame = replace(
            data,
            rows=(data.rows[0],) * self.rows,
            index=RowIndex(*(MaskedRows(column) for column in columns[:5])),
            fields=fields,
            hidden_inputs=self.hidden_inputs,
            feedback_inputs=self._feedback(columns[1]),
            leaf=None,
            weights=columns[5] if self.capture is not None or self.task.policy.reduction == "state_mean" else None,
            counts=counts,
            derivative_sources={},
            input_gradients={},
            parameter_gradient_fields={},
        )
        batch = replace(batch, roots=FieldRoots(fields), results={}, observing=False)
        if self.optimizer is not None or self.task.policy.parameter_gradients:
            frame.leaf = fields[0].detach().requires_grad_(True)
        return frame, batch

    def resident(self, capture_bytes: int) -> int:
        """Only the graph runner owns capture residency."""
        return capture_bytes if self.capture is not None else 0

    def _body(self, data: TensorFrame, batch: TensorBatch) -> None:
        frame, operands = (data, batch) if self.compact else self._frame(data, batch)
        self.evaluated_rows = tuple(range(len(frame.rows)))
        self.result = self.gradients = None
        self.input_gradients = {}
        if not self.compact:
            for dependency in self.task.dependencies(self.tasks)[:-1]:
                dependency.evaluate(frame, operands)
        if self.optimizer is None:
            self.task.evaluate(frame, operands)
            self.result = operands.results[self.task.symbol, self.task.outputs[0]]
            return
        with torch.enable_grad():
            self.inputs, self.evaluated_rows = self.task.read(frame, operands, compact=self.compact)
            if self.compact:
                selected = (
                    selected_rows(self.inputs[self.task.policy.update_states])
                    if self.task.policy.update_states in self.task.domains
                    else self.evaluated_rows
                )
                self.updated = tuple(sorted(row_plan(data, batch).select(selected).last))
            if self.evaluated_rows:
                self.result, derivatives = self.task.differentiate(self.inputs, self.evaluated_rows, frame, operands)
                self.result = self.result.detach()
                self.input_gradients = dict(
                    zip(frame.derivative_sources, (value.detach() for value in derivatives[1:]), strict=True)
                )
                self.gradients = derivatives[0].detach()
            else:
                self.gradients = torch.zeros_like(state_leaf(frame, operands))
        if not self.compact:
            self.gradients = self.gradients[: self.state_count].contiguous()
            assert self.table is not None
            _, _, last, sources, destinations, _ = self.layout.split(self.table)
            self.optimizer.staged(
                self.pools,
                frame.fields,
                sources,
                destinations,
                self.gradients,
                {field: self.inputs[key] for key, field in self.task.captures.items()},
                last,
                paged=self.device.type == "cuda" and self.pools[0].dtype == torch.float32,
            )
        frame.derivative_sources.clear()
        frame.leaf = None
        for key, value in operands.results.items():
            operands.results[key] = value.detach()

    def __call__(
        self, data: TensorFrame, batch: TensorBatch, context: StateTaskContext[TensorFrame] | None = None
    ) -> TaskOutput:
        """Prepare operands, run the shared body or graph, then settle invocation outputs."""
        parent, frame = batch, data
        if self.task.resource and any(group.resource == self.task.resource for group in batch.selection.groups):
            batch = batch.for_resource(self.task.resource)
        eligible = set(data.rows) if self.task.phase is None else set(data.step.rows.producer_rows(self.task.phase))
        rows = tuple(row for row in data.rows if row in eligible) if self.task.phase is not None else data.rows
        if rows != data.rows or batch.selection is not parent.selection:
            data = replace(data, rows=rows, plan=None, index=None)
        if not data.rows and self.task.phase is not None:
            return TaskOutput()
        addresses: tuple[int, ...] = ()
        if not self.compact:
            if not data.rows:
                return TaskOutput()
            addresses = self._inputs(data, batch)
            plan = row_plan(data, batch)
            self.updated = tuple(sorted(plan.last)) if self.optimizer is not None else ()
            if context is not None and self.optimizer is not None:
                context.write = context.arena.prepare_transition(
                    batch.selection.refs.take(self.updated), batch.stream, copy_fields=()
                )
                write = context.write
                plan = replace(
                    plan,
                    writes=(self.updated, context.arena.slots((write,) * len(write.after_refs), write.after_refs)[1]),
                )
            if context is None:
                self.table = torch.empty(self.layout.size, device="meta", dtype=torch.long)
            else:
                self._prepare(plan)
        if self.capture is None or context is None:
            self._body(data, batch)
        else:
            assert self.table is not None
            table = self.table
            self.capture.run(
                lambda: self._body(data, batch),
                self.table,
                self._views()[3],
                addresses,
                batch.stream,
                lambda: self.staging.copy_to(table),
            )
        output = self.settle_rows(data, batch)
        frame.parameter_gradient_fields = data.parameter_gradient_fields
        if batch.selection is not parent.selection:
            positions = tuple(parent.selection.refs.index(ref) for ref in batch.selection.refs)
            output = replace(
                output,
                gradient_indices=tuple(positions[i] for i in output.gradient_indices),
                changed=tuple(positions[i] for i in output.changed),
            )
        return output

    def _inputs(self, data: TensorFrame, batch: TensorBatch) -> tuple[int, ...]:
        self.active_rows, self.active_states = len(data.rows), len(batch.roots)
        if not self.pools and batch.roots:
            self.pools = tuple(batch.field_slots(i)[0] for i in range(len(batch.roots[0])))
        snapshots = () if data.step.hidden is None else data.step.hidden_batch(batch.stream).snapshots
        addresses = tuple(item.hidden.data_ptr() for item in snapshots)
        if self.hidden_inputs is None or self.addresses != addresses:
            self.hidden_inputs = {item.resource.locator_key: item.storage for item in snapshots}
        if self.feedback_names:
            assert batch.feedback is not None
            self.feedback_values = tuple(
                batch.feedback.base_logits if name == "base_logits" else batch.feedback.sampled_tokens
                for name in self.feedback_names
            )
            if batch.stream is not None:
                for value in self.feedback_values:
                    value.record_stream(batch.stream)
        return addresses

    def _prepare(self, plan: RowPlan) -> None:
        device = next(iter(self.hidden_inputs.values())).device if self.hidden_inputs else self.pools[0].device
        size = self.layout.size
        if self.table is None or self.table.numel() != size:
            self.table = torch.empty(size, dtype=torch.long, device=device)
        if device.type == "cuda":
            self._fill(self.staging.take(size).numpy(), plan)
            self.staging.copy_to(self.table)
        else:
            self.table.zero_()
            self._fill(self.table.numpy(), plan)

    def settle_rows(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        """Record gradients, losses and observations once for both evaluation strategies."""
        rows = self.evaluated_rows if self.compact else tuple(range(len(data.rows)))
        count = (
            len(batch.roots)
            if self.optimizer is not None and self.task.policy.reduction == "state_mean"
            else len(data.rows)
        )
        value = None if self.result is None else retain_rows(self.result, count, copy=not self.compact)
        if value is not None:
            batch.results[self.task.symbol, self.task.outputs[0]] = value
        if self.optimizer is None:
            return TaskOutput()
        assert self.gradients is not None
        gradients = retain_rows(self.gradients, len(batch.roots), copy=not self.compact)
        if value is not None:
            data.input_gradients.update(
                (key, retain_rows(value, len(data.rows), copy=not self.compact))
                for key, value in self.input_gradients.items()
            )
            data.parameter_gradient_fields = ParameterGradients(
                batch.selection.refs, gradients, data.parameter_gradient_fields
            )
        pending = tuple(gradients[index] for index in self.updated) if self.compact else ()
        positions = self.updated if self.compact else ()
        plan = row_plan(data, batch).select(rows)
        last = plan.last
        writes = tuple(
            (
                FieldView(field, 0, tuple(batch.roots[0][field].shape)),
                torch.stack(
                    tuple(
                        self.inputs[key][last[index]].detach() if index in last else batch.roots[index][field]
                        for index in self.updated
                    )
                ),
            )
            for key, field in self.task.captures.items()
            if self.compact and self.updated and rows
        )
        observed = (
            {key: value for key, value in self.inputs.items() if isinstance(self.task.sources[key], HiddenInputSource)}
            if self.compact
            else self.inputs
        )
        return TaskOutput(
            loss=value,
            gradients=pending,
            gradient_indices=positions,
            changed=self.updated,
            loss_count=(len(last) if self.task.policy.reduction == "state_mean" else 1) if value is not None else 0,
            update_count=len(self.updated),
            trained_rows=len(rows),
            field_writes=writes,
            observations=() if value is None else training_observations(data, batch, value, observed, rows),
        )


def retain_rows(value: torch.Tensor, count: int, *, copy: bool) -> torch.Tensor:
    """Own graph outputs; ordinary eager results retain their autograd edges."""
    return (value[:count] if value.ndim else value).detach().clone() if copy else value

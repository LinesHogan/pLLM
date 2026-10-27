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
from pllm.contracts.semantic_graph import HiddenInputSource, StateResource, StateSelectionInputSource, TaskSymbol
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
from pllm.kernels.page_copy import copy_pages, gather_rows

Table = TypeVar("Table", torch.Tensor, npt.NDArray[np.int64])


@dataclass(frozen=True, slots=True)
class TableLayout:
    """One resource's segments in the shared int64 table, uploaded in one copy per step.

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
    results: tuple[torch.Tensor, ...] = ()
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
    resource_pools: dict[StateResource, tuple[torch.Tensor, ...]] = dataclass_field(
        default_factory=lambda: dict[StateResource, tuple[torch.Tensor, ...]]()
    )
    layouts: dict[StateResource, TableLayout] = dataclass_field(
        default_factory=lambda: dict[StateResource, TableLayout]()
    )
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

    def _fill(self, host: npt.NDArray[np.int64], plan: RowPlan, layout: TableLayout) -> None:
        """Stage a RowPlan and feedback addresses into the zeroed fixed table."""
        rows, counts, last, sources, destinations, feedback = layout.split(host)
        count = len(plan.packed)
        if layout.feedback and -1 in plan.sampler:
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
        if self.task.writer is not None:
            last[list(self.updated)] = np.arange(len(self.updated))
        for row, value in zip(feedback, self.feedback_values[: layout.feedback], strict=True):
            row[:] = (value.data_ptr(), value.stride(0), value.stride(1) if value.ndim > 1 else 1)

    def _views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.table is not None
        rows, counts, _, sources, destinations, _ = self.layout.split(self.table[: self.layout.size])
        return rows, counts, sources, destinations

    def _feedback(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.table is not None
        locations = self.layout.split(self.table[: self.layout.size])[5]
        return {
            name: gather_rows(location, rows, value)
            for name, location, value in zip(self.feedback_names, locations, self.feedback_values, strict=True)
        }

    def _frame(self, data: TensorFrame, batch: TensorBatch) -> tuple[TensorFrame, TensorBatch]:
        frames: dict[StateResource, TensorFrame] = {}
        batches: dict[StateResource, TensorBatch] = {}
        assert self.table is not None
        offset = 0
        for resource, layout in self.layouts.items():
            selected = batch.for_resource(resource) if resource else batch
            frames[resource], batches[resource] = self._resource_frame(
                data, selected, layout, self.table[offset : offset + layout.size], self.resource_pools[resource]
            )
            offset += layout.size
        frame = frames[self.task.resource]
        frame.resource_frames = {resource: item for resource, item in frames.items() if item is not frame}
        return frame, replace(batches[self.task.resource], select_resource=batches.__getitem__)

    def _resource_frame(
        self,
        data: TensorFrame,
        batch: TensorBatch,
        layout: TableLayout,
        table: torch.Tensor,
        pools: tuple[torch.Tensor, ...],
    ) -> tuple[TensorFrame, TensorBatch]:
        columns, counts, _, sources, _, _ = layout.split(table)
        reads = self.read_fields
        if self.optimizer is not None and (self.device.type == "cpu" or self.pools[0].dtype != torch.float32):
            reads = reads | frozenset(range(5))
        instance = batch.selection.keys == (None,)
        # A single-page instance has no alternate slot; borrow its pool instead of copying the network.
        fields = tuple(
            (
                (pool if len(pool) == 1 else pool.index_select(0, slots[:1]))
                if instance
                else pool.index_select(0, slots.clamp_min(0)).masked_fill(
                    (slots < 0).reshape(-1, *(1 for _ in pool.shape[1:])), 0
                )
            )
            if field in reads
            else pool.new_empty((layout.states + 1, 0))
            for field, (pool, slots) in enumerate(zip(pools, sources, strict=True))
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
        """Reserve graph residency; import stores need only their fixed table and inputs."""
        if self.capture is None:
            return 0
        if self.task.writer is not None and self.task.writer.source is None:
            # Import graphs only store into pages: every allocation is in the table and staging.
            assert self.table is not None
            return min(
                capture_bytes, sum(value.untyped_storage().nbytes() for value in (self.table, *self.inputs.values()))
            )
        return capture_bytes

    def _body(self, data: TensorFrame, batch: TensorBatch) -> None:
        if self.task.writer is not None:
            assert self.table is not None
            _, _, last, _, destinations, _ = self.layout.split(self.table[: self.layout.size])
            for view, value in zip(self.task.writer.fields, self.inputs.values(), strict=True):
                if self.task.writer.source is not None:
                    torch._assert_async(torch.isfinite(value).all(), "computed fields must be finite")  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
                copy_pages(value, view.read(self.pools[view.field], pooled=True), last, destinations[view.field])
            return
        frame, operands = (data, batch) if self.compact else self._frame(data, batch)
        self.evaluated_rows = tuple(range(len(frame.rows)))
        self.results, self.gradients = (), None
        self.input_gradients = {}
        if not self.compact:
            for dependency in self.task.dependencies(self.tasks)[:-1]:
                dependency.evaluate(frame, operands)
        if self.optimizer is None:
            self.task.evaluate(frame, operands)
            self.results = tuple(operands.results[self.task.symbol, name] for name in self.task.outputs)
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
                result, derivatives = self.task.differentiate(self.inputs, self.evaluated_rows, frame, operands)
                self.results = (result.detach(),)
                self.input_gradients = dict(
                    zip(frame.derivative_sources, (value.detach() for value in derivatives[1:]), strict=True)
                )
                self.gradients = derivatives[0].detach()
            else:
                self.gradients = torch.zeros_like(state_leaf(frame, operands))
        if not self.compact:
            self.gradients = self.gradients[: self.state_count].contiguous()
            assert self.table is not None
            _, _, last, sources, destinations, _ = self.layout.split(self.table[: self.layout.size])
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
        writer = self.task.writer
        selected = () if writer is None else writer.select(data, batch.selection)
        if writer is not None and not selected:
            return TaskOutput()
        values = () if writer is None else writer.values(data, batch, selected, validate=context is None)
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
            if not data.rows and writer is None:
                return TaskOutput()
            addresses = self._inputs(data, batch)
            plan = row_plan(data, batch)
            self.updated = (
                tuple(batch.selection.refs.index(parent.selection.refs[i]) for i in selected)
                if writer is not None
                else tuple(sorted(plan.last))
                if self.optimizer is not None
                else ()
            )
            for index, value in enumerate(values):
                key = str(index)
                if key not in self.inputs:
                    self.inputs[key] = value.new_zeros((self.state_count, *value.shape[1:]))
                self.inputs[key][: len(value)].copy_(value.detach())
            if context is not None and (self.optimizer is not None or writer is not None):
                context.write = context.arena.prepare_transition(
                    batch.selection.refs.take(self.updated), batch.stream, copy_fields=() if writer is None else None
                )
                write = context.write
                plan = replace(
                    plan,
                    writes=(self.updated, context.arena.slots((write,) * len(write.after_refs), write.after_refs)[1]),
                )
            if context is None:
                self.table = torch.empty(
                    sum(layout.size for layout in self.layouts.values()), device="meta", dtype=torch.long
                )
            else:
                self._prepare(plan, data, batch)
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
                buffers=tuple(self.inputs.values()) if writer is not None else (),
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
        if not self.resource_pools or self.capture is None:
            self.inputs = {}
            resources = dict.fromkeys(
                (
                    self.task.resource,
                    *(
                        source.resource
                        for task in self.task.dependencies(self.tasks)
                        for source in task.sources.values()
                        if isinstance(source, StateSelectionInputSource)
                    ),
                )
            )
            for resource in resources:
                selected = batch.for_resource(resource) if resource else batch
                pools = (
                    tuple(selected.field_slots(i)[0] for i in range(len(selected.roots[0]))) if selected.roots else ()
                )
                states = (
                    self.state_count
                    if resource == self.task.resource
                    else (1 if selected.selection.keys == (None,) else self.rows)
                    if self.capture
                    else len(selected.roots)
                )
                self.resource_pools[resource] = pools
                self.layouts[resource] = TableLayout(
                    self.rows, states, len(pools), len(self.feedback_names) if resource == self.task.resource else 0
                )
            self.pools = self.resource_pools[self.task.resource]
        snapshots = () if data.step.hidden is None else data.step.hidden_batch(batch.stream).snapshots
        addresses = tuple(item.hidden.data_ptr() for item in snapshots)
        if self.hidden_inputs is None or self.addresses != addresses:
            self.hidden_inputs = {item.resource.locator_key: item.storage for item in snapshots}
        self.addresses = addresses
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

    def _prepare(self, plan: RowPlan, data: TensorFrame, batch: TensorBatch) -> None:
        device = next(iter(self.hidden_inputs.values())).device if self.hidden_inputs else self.pools[0].device
        size = sum(layout.size for layout in self.layouts.values())
        if self.table is None or self.table.numel() != size:
            self.table = torch.empty(size, dtype=torch.long, device=device)
        host = self.staging.take(size).numpy() if device.type == "cuda" else self.table.numpy()
        host.fill(0)
        offset = 0
        for resource, layout in self.layouts.items():
            selected = plan if resource == self.task.resource else RowPlan.build(data, batch.for_resource(resource))
            self._fill(host[offset : offset + layout.size], selected, layout)
            offset += layout.size
        if device.type == "cuda":
            self.staging.copy_to(self.table)

    def settle_rows(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        """Record gradients, losses and observations once for both evaluation strategies."""
        if self.task.writer is not None:
            return TaskOutput(changed=self.updated, update_count=len(self.updated))
        rows = self.evaluated_rows if self.compact else tuple(range(len(data.rows)))
        count = (
            len(batch.roots)
            if self.optimizer is not None and self.task.policy.reduction == "state_mean"
            else len(data.rows)
        )
        values = tuple(
            result
            if self.compact
            else retain_rows(result, count, copy=self.capture is not None or not self.task.differentiable)
            for result in self.results
        )
        batch.results.update(((self.task.symbol, name), value) for name, value in zip(self.task.outputs, values))
        value = values[0] if values else None
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
    value = value[:count] if value.ndim else value
    return value.detach().clone() if copy else value

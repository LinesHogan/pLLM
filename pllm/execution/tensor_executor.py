"""TensorExecutor: one tensor task's body over a fixed int64 row table, eager or captured, and its settlement.

TensorBinding.executables builds one per tensor task. Row-reading tasks gather every input through the
table and mask unselected rows on the device; tasks without row inputs evaluate directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

from pllm.contracts.semantic_graph import StateResource, StateSelectionInputSource, TaskSymbol
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.step_capture import StepCapture
from pllm.execution.tensor_frame import (
    FieldRoots,
    FixedRows,
    ParameterGradients,
    RowIndex,
    RowPlan,
    TensorFrame,
    row_plan,
    training_observations,
)
from pllm.execution.tensor_program import PageUpdate, TaskOutput, TensorBatch
from pllm.execution.tensor_task import TensorTask
from pllm.kernels.page_copy import copy_pages, gather_rows

Table = TypeVar("Table", torch.Tensor, npt.NDArray[np.int64])


@dataclass(frozen=True, slots=True)
class TableLayout:
    """One resource's segments in the shared int64 table, uploaded in one copy per step.

    rows          [5, R]   per row: sampler row, sample index, state (S: padding sink), version, active
    last          [S]      row each written state takes its field values from (field writers)
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
        return ((5, rows), (states,), (fields, states + 1), (fields, states), (self.feedback, 3))

    @property
    def size(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes)

    def split(self, table: Table) -> tuple[Table, Table, Table, Table, Table]:
        """Named views of `table` (a torch tensor or numpy array of `size` elements)."""
        views: list[Table] = []
        start = 0
        for shape in self.shapes:
            end = start + math.prod(shape)
            views.append(table[start:end].reshape(shape))
            start = end
        rows, last, sources, destinations, feedback = views
        return rows, last, sources, destinations, feedback


def reads_rows(task: TensorTask, tasks: Mapping[TaskSymbol, TensorTask]) -> bool:
    """Whether the task reads per-row inputs (hidden, feedback, request state), directly or upstream.

    Only such tasks have a fixed row table; the rest evaluate directly.
    """
    return any(port.rowwise for item in task.dependencies(tasks) for port in item.ports.values())


@dataclass
class TensorExecutor:
    """One body and one result settlement for a tensor task, evaluated over masked fixed tables."""

    task: TensorTask
    tasks: Mapping[TaskSymbol, TensorTask]
    optimizer: PageUpdate | None = None
    capture: StepCapture | None = None
    device: torch.device = torch.device("cpu")
    table: torch.Tensor | None = None
    pools: tuple[torch.Tensor, ...] = ()
    results: tuple[torch.Tensor, ...] = ()
    gradients: torch.Tensor | None = None
    # Device [applied updates, trained rows] of the last training body.
    counters: torch.Tensor | None = None
    inputs: dict[str, torch.Tensor] = dataclass_field(default_factory=lambda: dict[str, torch.Tensor]())
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
    # Fixed by the task at construction: whether it reads rows through the table, and whether its body
    # differentiates the flat parameter field.
    rowwise: bool = dataclass_field(init=False, default=False)
    differentiates: bool = dataclass_field(init=False, default=False)

    def __post_init__(self) -> None:
        self.rowwise = self.task.writer is not None or reads_rows(self.task, self.tasks)
        assert self.rowwise or self.optimizer is None
        dependencies = self.task.dependencies(self.tasks)
        self.read_fields = frozenset(
            port.field for task in dependencies for port in task.ports.values() if port.field is not None
        )
        parameters = any(task.policy.parameter_gradients for task in dependencies)
        if parameters:
            self.read_fields |= {0}
        self.differentiates = self.optimizer is not None or parameters
        self.feedback_names = tuple(
            sorted({port.feedback for task in dependencies for port in task.ports.values() if port.feedback})
        )
        assert (
            self.capture is None
            or 8 * (5 * self.rows + self.state_count + 1) <= self.capture.entitlement.resident_bytes
        )

    @property
    def rows(self) -> int:
        return self.capture.entitlement.input_shapes[0][0] if self.capture is not None else self.active_rows

    @property
    def state_count(self) -> int:
        return self.capture.states if self.capture is not None else self.active_states

    @property
    def paged(self) -> bool:
        """FP32 CUDA pages update through the paged AdamW kernel; others through Torch on gathered copies."""
        return self.device.type == "cuda" and self.pools[0].dtype == torch.float32

    @property
    def layout(self) -> TableLayout:
        return TableLayout(self.rows, self.state_count, len(self.pools), len(self.feedback_names))

    @property
    def table_size(self) -> int:
        """Resource tables followed by one hidden slot column per bound locator."""
        return sum(layout.size for layout in self.layouts.values()) + len(self.hidden_inputs or {}) * self.rows

    def _fill(self, host: npt.NDArray[np.int64], plan: RowPlan, layout: TableLayout) -> None:
        """Stage a RowPlan and feedback addresses into the zeroed fixed table."""
        rows, last, sources, destinations, feedback = layout.split(host)
        count = len(plan.packed)
        assert not layout.feedback or -1 not in plan.sampler
        rows[2] = layout.states
        rows[:4, :count] = plan.columns
        rows[0, :count] = np.maximum(rows[0, :count], 0)
        rows[4, :count] = 1
        sources[:] = destinations[:] = -1
        sources[:, : len(plan.sources)] = plan.sources.T
        if plan.writes is not None:
            updated, slots = plan.writes
            destinations[:, list(updated)] = slots.T
        if self.task.writer is not None:
            last[list(self.updated)] = np.arange(len(self.updated))
        for row, value in zip(feedback, self.feedback_values[: layout.feedback], strict=True):
            row[:] = (value.data_ptr(), value.stride(0), value.stride(1) if value.ndim > 1 else 1)

    def _views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.table is not None
        _, last, sources, destinations, _ = self.layout.split(self.table[: self.layout.size])
        return last, sources, destinations

    def _feedback(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.table is not None
        locations = self.layout.split(self.table[: self.layout.size])[4]
        return {
            name: gather_rows(location, rows, value)
            for name, location, value in zip(self.feedback_names, locations, self.feedback_values, strict=True)
        }

    def _frame(self, data: TensorFrame, batch: TensorBatch) -> tuple[TensorFrame, TensorBatch]:
        tables: dict[StateResource, FixedRows] = {}
        batches: dict[StateResource, TensorBatch] = {}
        assert self.table is not None
        offset = 0
        for resource, layout in self.layouts.items():
            selected = batch.for_resource(resource) if resource else batch
            tables[resource], batches[resource] = self._resource_rows(
                selected, layout, self.table[offset : offset + layout.size], self.resource_pools[resource]
            )
            offset += layout.size
        table = tables.pop(self.task.resource)
        frame = replace(
            data,
            rows=(data.rows[0],) * self.rows,
            fixed=replace(table, resources=tables),
            input_gradients={},
            parameter_gradient_fields={},
        )
        return frame, replace(batches[self.task.resource], select_resource=batches.__getitem__)

    def _resource_rows(
        self, batch: TensorBatch, layout: TableLayout, table: torch.Tensor, pools: tuple[torch.Tensor, ...]
    ) -> tuple[FixedRows, TensorBatch]:
        assert self.table is not None
        columns, _, sources, _, _ = layout.split(table)
        # The eager optimizer updates gathered copies of its fields in place.
        copied = frozenset(range(5)) if self.optimizer is not None and not self.paged else frozenset[int]()
        reads = self.read_fields | copied
        instance = batch.selection.keys == (None,)
        # A single-page instance has no alternate slot; borrow its pool instead of copying the network.
        fields = tuple(
            (
                (pool if len(pool) == 1 and field not in copied else pool.index_select(0, slots[:1]))
                if instance
                else pool.index_select(0, slots.clamp_min(0)).masked_fill(
                    (slots < 0).reshape(-1, *(1 for _ in pool.shape[1:])), 0
                )
            )
            if field in reads
            else pool.new_empty((layout.states + 1, 0))
            for field, (pool, slots) in enumerate(zip(pools, sources, strict=True))
        )
        rows = FixedRows(
            fields,
            layout.states,
            RowIndex(*columns[:4]),
            columns[4] > 0,
            {
                key: (pool, slots)
                for (key, pool), slots in zip(
                    (self.hidden_inputs or {}).items(),
                    self.table[sum(item.size for item in self.layouts.values()) :].view(-1, self.rows),
                    strict=True,
                )
            },
            self._feedback(columns[0]),
            fields[0].detach().requires_grad_(True) if self.differentiates else None,
        )
        return rows, replace(batch, roots=FieldRoots(fields), results={}, observing=False)

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
            last, _, destinations = self._views()
            for view, value in zip(self.task.writer.fields, self.inputs.values(), strict=True):
                copy_pages(value, view.read(self.pools[view.field], pooled=True), last, destinations[view.field])
            return
        if not self.rowwise:
            self.task.evaluate(data, batch)
            self.results = tuple(batch.results[self.task.symbol, name] for name in self.task.outputs)
            return
        frame, operands = self._frame(data, batch)
        sources: dict[tuple[str, str], torch.Tensor] = {}
        for dependency in self.task.dependencies(self.tasks)[:-1]:
            sources |= dependency.evaluate(frame, operands)
        if self.optimizer is None:
            self.task.evaluate(frame, operands)
            self.results = tuple(operands.results[self.task.symbol, name] for name in self.task.outputs)
            return
        with torch.enable_grad():
            self.inputs, own = self.task.read(frame, operands)
            sources |= own
            trained, updates, last = self._selection(frame.table)
            result, derivatives = self.task.differentiate(self.inputs, sources, frame, trained)
        self.results = (result.detach(),)
        self.input_gradients = dict(zip(sources, (value.detach() for value in derivatives[1:]), strict=True))
        self.gradients = derivatives[0].detach()[: self.state_count].contiguous()
        self.counters = torch.stack((updates.sum(), trained.sum()))
        _, slots, destinations = self._views()
        self.optimizer.staged(
            self.pools,
            frame.table.fields,
            slots,
            destinations,
            self.gradients,
            {field: self.inputs[key] for key, field in self.task.captures.items()},
            updates,
            last,
            paged=self.paged,
        )
        for key, value in operands.results.items():
            operands.results[key] = value.detach()

    def _selection(self, rows: FixedRows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Which rows train, which states update and each state's last trained row (-1: none), on the device.

        `rows=` selects the trained rows; `update=` (or else the trained rows) selects the states whose optimizer
        fields change. Every other state publishes its fields unchanged, so selectors never sync the host.
        """
        policy, states = self.task.policy, rows.index.states
        trained = rows.active & self.inputs[policy.forward_rows] if policy.forward_rows else rows.active
        chosen = (
            rows.active & self.inputs[policy.update_states] if policy.update_states in self.task.domains else trained
        )
        count = states.new_zeros(rows.states + 1).index_add_(0, states, chosen.long())
        positions = torch.where(trained, torch.arange(len(states), device=states.device), -1)
        last = torch.full_like(count, -1).scatter_reduce_(0, states, positions, "amax")
        return trained, count[: self.state_count] > 0, last[: self.state_count]

    def __call__(
        self, data: TensorFrame, batch: TensorBatch, context: StateTaskContext[TensorFrame] | None = None
    ) -> TaskOutput:
        """Prepare operands, run the shared body or graph, then settle invocation outputs."""
        prepared = data if self.task.export is None else self.task.export.prepare(data)
        if prepared is None:
            return TaskOutput()
        data = prepared
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
            data = replace(data, rows=rows, plan=None)
        if not data.rows and self.task.phase is not None:
            return TaskOutput()
        addresses: tuple[int, ...] = ()
        if self.rowwise:
            if not data.rows and writer is None:
                return TaskOutput()
            addresses = self._inputs(data, batch)
            plan = row_plan(data, batch)
            if writer is not None:
                self.updated = tuple(batch.selection.refs.index(parent.selection.refs[i]) for i in selected)
            elif self.optimizer is not None:
                self.updated = tuple(sorted(plan.last))
            else:
                self.updated = ()
            for index, value in enumerate(values):
                key = str(index)
                if key not in self.inputs:
                    self.inputs[key] = value.new_zeros((self.state_count, *value.shape[1:]))
                self.inputs[key][: len(value)].copy_(value.detach())
            if context is not None and (self.optimizer is not None or writer is not None):
                write = context.prepare_owned_write(
                    batch.selection.refs.take(self.updated), copy_fields=() if writer is None else None
                )
                plan = replace(
                    plan,
                    writes=(self.updated, context.arena.slots((write,) * len(write.after_refs), write.after_refs)[1]),
                )
            if context is None:
                self.table = torch.empty(self.table_size, device="meta", dtype=torch.long)
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
                self._views()[2],
                addresses,
                batch.stream,
                lambda: self.staging.copy_to(table),
                buffers=tuple(self.inputs.values()) if writer is not None else (),
            )
        output = self.settle_rows(data, batch)
        frame.parameter_gradient_fields = data.parameter_gradient_fields
        if batch.selection is not parent.selection:
            positions = tuple(parent.selection.refs.index(ref) for ref in batch.selection.refs)
            output = replace(output, changed=tuple(positions[i] for i in output.changed))
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
                if resource == self.task.resource:
                    states = self.state_count
                elif self.capture is None:
                    states = len(selected.roots)
                else:  # A captured reader of another resource: one instance root or one slot per row.
                    states = 1 if selected.selection.keys == (None,) else self.rows
                self.resource_pools[resource] = pools
                self.layouts[resource] = TableLayout(
                    self.rows, states, len(pools), len(self.feedback_names) if resource == self.task.resource else 0
                )
            self.pools = self.resource_pools[self.task.resource]
        captures = () if data.step.hidden is None else data.step.hidden_batch(batch.stream).captures
        addresses = tuple(item.storage.data_ptr() for item in captures)
        if self.hidden_inputs is None or self.addresses != addresses:
            self.hidden_inputs = {item.resource.locator_key: item.storage for item in captures}
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
        size = self.table_size
        if self.table is None or self.table.numel() != size:
            self.table = torch.empty(size, dtype=torch.long, device=device)
        host = self.staging.take(size).numpy() if device.type == "cuda" else self.table.numpy()
        host.fill(0)
        offset = 0
        for resource, layout in self.layouts.items():
            selected = plan if resource == self.task.resource else RowPlan.build(data, batch.for_resource(resource))
            self._fill(host[offset : offset + layout.size], selected, layout)
            offset += layout.size
        captures = () if data.step.hidden is None else data.step.hidden.captures
        for capture in captures:
            host[offset : offset + self.rows] = capture.slots[0] if capture.slots.size else 0
            host[offset : offset + len(data.rows)] = capture.slots[plan.packed]
            offset += self.rows
        if device.type == "cuda":
            self.staging.copy_to(self.table)

    def settle_rows(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        """Record results, gradients, losses and observations of the last body."""
        if self.task.writer is not None:
            return TaskOutput(changed=self.updated, update_count=len(self.updated))
        state_mean = self.task.policy.reduction == "state_mean"
        count = len(batch.roots) if self.optimizer is not None and state_mean else len(data.rows)
        # Results without row inputs have no row axis to trim.
        values = tuple(
            retain_rows(result, count, copy=self.capture is not None or not self.task.differentiable)
            if self.rowwise
            else result
            for result in self.results
        )
        batch.results.update(((self.task.symbol, name), value) for name, value in zip(self.task.outputs, values))
        if self.optimizer is None:
            if self.task.export is not None:
                return self.task.export.output(values[0], data)
            if self.task.sampler_stage is not None:
                return self.task.sampler_output(data, batch)
            return TaskOutput()
        assert self.gradients is not None and self.counters is not None
        data.input_gradients.update(
            (key, retain_rows(value, len(data.rows), copy=True)) for key, value in self.input_gradients.items()
        )
        data.parameter_gradient_fields = ParameterGradients(
            batch.selection.refs,
            retain_rows(self.gradients, len(batch.roots), copy=True),
            data.parameter_gradient_fields,
        )
        return TaskOutput(
            loss=values[0],
            changed=self.updated,
            loss_count=count if state_mean else 1,
            counters=self.counters.clone(),
            observations=training_observations(data, batch, values[0], self.inputs),
        )


def retain_rows(value: torch.Tensor, count: int, *, copy: bool) -> torch.Tensor:
    """Own graph outputs; ordinary eager results retain their autograd edges."""
    value = value[:count] if value.ndim else value
    return value.detach().clone() if copy else value

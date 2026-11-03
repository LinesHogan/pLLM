"""TensorExecutor: one tensor task's body over a fixed int64 row table, eager or captured, and how it writes its outputs.

TensorBinding.executables builds one per tensor task. Row-reading tasks gather every input through the
table and mask unselected rows on the device; tasks without row inputs evaluate directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
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
from pllm.execution.tensor_program import ADAM, PageUpdate, TaskOutput, TensorBatch
from pllm.execution.tensor_task import FieldWriter, TensorTask
from pllm.kernels.page_copy import copy_pages, gather_rows

Table = TypeVar("Table", torch.Tensor, npt.NDArray[np.int64])


def _gather_field(pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Gather one field's state pages, filling padding slots with zero."""
    return pool.index_select(0, slots.clamp_min(0)).masked_fill(
        (slots < 0).reshape(-1, *(1 for _ in pool.shape[1:])), 0
    )


_cuda_gather: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.compile(  # pyright: ignore[reportUnknownMemberType]
    _gather_field, dynamic=True, fullgraph=True
)


@dataclass(frozen=True, slots=True)
class TableLayout:
    """One resource's segments in the shared int64 table, uploaded in one copy per step.

    rows          [5, R]   per row: sampler row, sample index, state (S: padding sink), version, active
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
        return ((5, rows), (fields, states + 1), (fields, states), (self.feedback, 3))

    @property
    def size(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes)

    def split(self, table: Table) -> tuple[Table, Table, Table, Table]:
        """Named views of `table` (a torch tensor or numpy array of `size` elements)."""
        views: list[Table] = []
        start = 0
        for shape in self.shapes:
            end = start + math.prod(shape)
            views.append(table[start:end].reshape(shape))
            start = end
        rows, sources, destinations, feedback = views
        return rows, sources, destinations, feedback


def reads_rows(task: TensorTask, tasks: Mapping[TaskSymbol, TensorTask]) -> bool:
    """Whether the task reads per-row inputs (hidden, feedback, request state), directly or upstream.

    Only such tasks have a fixed row table; the rest evaluate directly.
    """
    return any(port.rowwise for item in task.dependencies(tasks) for port in item.ports.values())


@dataclass
class TensorExecutor:
    """One body and one result write for a tensor task, evaluated over masked fixed tables.

    Field writes (`@pllm.write`, imports) run in FieldWriteExecutor instead.
    """

    task: TensorTask
    tasks: Mapping[TaskSymbol, TensorTask]
    optimizer: PageUpdate | None = None
    capture: StepCapture | None = None
    device: torch.device = torch.device("cpu")
    filter_rows: bool = True
    table: torch.Tensor | None = None
    pools: tuple[torch.Tensor, ...] = ()
    results: tuple[torch.Tensor, ...] = ()
    forward_weights: torch.Tensor | None = None
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
    # Named segments bound to the fixed table; eager layouts are rebound each call.
    views: dict[StateResource, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = dataclass_field(
        default_factory=lambda: dict[StateResource, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]()
    )
    hidden_slots: torch.Tensor = dataclass_field(default_factory=lambda: torch.empty((0, 0), dtype=torch.long))
    # Fixed by the task at construction: whether it reads rows through the table, and whether its body
    # differentiates the flat parameter field.
    rowwise: bool = dataclass_field(init=False, default=False)
    differentiates: bool = dataclass_field(init=False, default=False)

    def __post_init__(self) -> None:
        assert self.task.writer is None
        self.rowwise = reads_rows(self.task, self.tasks)
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
        assert self.capture is None or 8 * (5 * self.rows + self.state_count + 1) <= self.capture.budget.resident_bytes

    @property
    def rows(self) -> int:
        return self.capture.budget.input_shapes[0][0] if self.capture is not None else self.active_rows

    @property
    def state_count(self) -> int:
        return self.capture.states if self.capture is not None else self.active_states

    @property
    def paged(self) -> bool:
        """FP32 CUDA pages update through the paged AdamW kernel; others through Torch on gathered copies."""
        return self.device.type == "cuda" and self.pools[0].dtype == torch.float32

    @property
    def table_size(self) -> int:
        """Resource tables followed by one hidden slot column per bound locator."""
        return sum(layout.size for layout in self.layouts.values()) + len(self.hidden_inputs or {}) * self.rows

    def _fill(self, host: npt.NDArray[np.int64], plan: RowPlan, layout: TableLayout) -> None:
        """Stage a RowPlan and feedback addresses into the zeroed fixed table."""
        rows, sources, destinations, feedback = layout.split(host)
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
        for row, value in zip(feedback, self.feedback_values[: layout.feedback], strict=True):
            row[:] = (value.data_ptr(), value.stride(0), value.stride(1) if value.ndim > 1 else 1)

    def _split(self) -> None:
        """Bind the table's resource views and hidden slot columns."""
        assert self.table is not None
        offset = 0
        for resource, layout in self.layouts.items():
            self.views[resource] = layout.split(self.table[offset : offset + layout.size])
            offset += layout.size
        self.hidden_slots = self.table[offset:].view(-1, self.rows)

    def _feedback(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        locations = self.views[self.task.resource][3]
        return {
            name: gather_rows(location, rows, value)
            for name, location, value in zip(self.feedback_names, locations, self.feedback_values, strict=True)
        }

    def _frame(self, data: TensorFrame, batch: TensorBatch) -> tuple[TensorFrame, TensorBatch]:
        tables: dict[StateResource, FixedRows] = {}
        batches: dict[StateResource, TensorBatch] = {}
        for resource, layout in self.layouts.items():
            selected = batch.for_resource(resource) if resource else batch
            tables[resource], batches[resource] = self._resource_rows(
                selected, layout, self.views[resource], self.resource_pools[resource]
            )
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
        self,
        batch: TensorBatch,
        layout: TableLayout,
        views: tuple[torch.Tensor, ...],
        pools: tuple[torch.Tensor, ...],
    ) -> tuple[FixedRows, TensorBatch]:
        columns, sources = views[0], views[1]
        # The eager optimizer updates gathered copies of its fields in place.
        copied = frozenset(range(ADAM)) if self.optimizer is not None and not self.paged else frozenset[int]()
        reads = self.read_fields | copied
        instance = batch.selection.keys == (None,)
        gather = _cuda_gather if pools and pools[0].is_cuda else _gather_field
        # A single-page instance has no alternate slot; borrow its pool instead of copying the network.
        fields = tuple(
            (
                (pool if len(pool) == 1 and field not in copied else pool.index_select(0, slots[:1]))
                if instance
                else gather(pool, slots)
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
                for (key, pool), slots in zip((self.hidden_inputs or {}).items(), self.hidden_slots, strict=True)
            },
            self._feedback(columns[0]),
            fields[0].detach().to(torch.promote_types(fields[0].dtype, torch.float32)).requires_grad_(True)
            if self.differentiates
            else None,
        )
        return rows, replace(batch, roots=FieldRoots(fields), results={}, observing=False)

    def resident(self, capture_bytes: int) -> int:
        """Reserve graph residency for a captured body."""
        return 0 if self.capture is None else capture_bytes

    def _body(self, data: TensorFrame, batch: TensorBatch) -> None:
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
            self.forward_weights = (
                self.task.ports[self.task.policy.forward_rows].read(frame, operands)
                if self.task.policy.forward_rows
                else None
            )
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
        self.counters = None if updates is None else torch.stack((updates.sum(), trained.sum()))
        _, slots, destinations, _ = self.views[self.task.resource]
        self.optimizer.staged(
            self.pools,
            frame.table.fields,
            slots,
            destinations,
            self.gradients,
            {field: self.inputs[key] for key, field in self.task.carries.items()},
            updates,
            last,
            paged=self.paged,
        )
        for key, value in operands.results.items():
            operands.results[key] = value.detach()

    def _selection(self, rows: FixedRows) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Which rows train, which states update and each state's last trained row (-1: none), on the device.

        `rows=` selects the trained rows; `update=` (or else the trained rows) selects the states whose optimizer
        fields change. Every other state publishes its fields unchanged, so selectors never sync the host.
        """
        policy, states = self.task.policy, rows.index.states
        trained = rows.active & self.inputs[policy.forward_rows] if policy.forward_rows else rows.active
        update_rows = self.inputs[policy.update_rows] if policy.update_rows in self.task.domains else None
        updates = None
        if policy.forward_rows or update_rows is not None:
            chosen = trained if update_rows is None else rows.active & update_rows
            count = states.new_zeros(rows.states + 1).index_add_(0, states, chosen.long())
            updates = count[: self.state_count] > 0
        last = None
        if self.task.carries:
            positions = torch.where(trained, torch.arange(len(states), device=states.device), -1)
            last = states.new_full((rows.states + 1,), -1).scatter_reduce_(0, states, positions, "amax")
            last = last[: self.state_count]
        return trained, updates, last

    def __call__(
        self, data: TensorFrame, batch: TensorBatch, context: StateTaskContext[TensorFrame] | None = None
    ) -> TaskOutput:
        """Prepare operands, run the shared body or graph, then write invocation outputs."""
        parent, frame = batch, data
        if not self.task.resource:
            batch = batch.state_free()
        elif any(group.resource == self.task.resource for group in batch.selection.groups):
            batch = batch.for_resource(self.task.resource)
        rows = data.producer_rows(self.task.phase) if self.filter_rows else data.rows
        if rows != data.rows or batch.selection is not parent.selection:
            data = replace(data, rows=rows, plan=None)
        if not data.rows and self.task.phase is not None:
            return TaskOutput()
        source_rows = len(data.rows)
        prepared = data if self.task.export is None else self.task.export.prepare(data)
        if prepared is None:
            return TaskOutput()
        data = prepared
        addresses: tuple[int, ...] = ()
        if self.capture is not None and context is not None:
            self.capture.order(batch.stream)
        if self.rowwise:
            if not data.rows:
                return TaskOutput()
            addresses = self._inputs(data, batch)
            plan = row_plan(data, batch)
            self.updated = () if self.optimizer is None else tuple(sorted(plan.last))
            if context is not None and self.optimizer is not None:
                slots = context.owned_write_slots(batch.selection.refs.take(self.updated), copy_fields=())
                plan = replace(plan, writes=(self.updated, slots))
            if context is None:
                self.table = torch.empty(self.table_size, device="meta", dtype=torch.long)
                self._split()
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
                self.views[self.task.resource][2],
                addresses,
                lambda: self.staging.copy_to(table),
            )
        output = self.write_outputs(data, batch, source_rows)
        frame.parameter_gradient_fields = data.parameter_gradient_fields
        if batch.selection is not parent.selection:
            positions = tuple(parent.selection.refs.index(ref) for ref in batch.selection.refs)
            output = replace(output, changed=tuple(positions[i] for i in output.changed))
        return output

    def _inputs(self, data: TensorFrame, batch: TensorBatch) -> tuple[int, ...]:
        self.active_rows, self.active_states = len(data.rows), len(batch.roots)
        if not self.resource_pools or self.capture is None:
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
        taps = () if data.step.hidden is None else data.step.hidden_batch(batch.stream).taps
        addresses = tuple(item.storage.data_ptr() for item in taps)
        if self.hidden_inputs is None or self.addresses != addresses:
            self.hidden_inputs = {item.resource.locator_key: item.storage for item in taps}
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
            self._split()
        elif self.capture is None:
            self._split()
        with self.staging.fill(self.table) as host:
            offset = 0
            for resource, layout in self.layouts.items():
                selected = plan if resource == self.task.resource else RowPlan.build(data, batch.for_resource(resource))
                self._fill(host[offset : offset + layout.size], selected, layout)
                offset += layout.size
            taps = () if data.step.hidden is None else data.step.hidden.taps
            for tap in taps:
                host[offset : offset + self.rows] = tap.slots[0] if tap.slots.size else 0
                host[offset : offset + len(data.rows)] = tap.slots[plan.packed]
                offset += self.rows

    def write_outputs(self, data: TensorFrame, batch: TensorBatch, source_rows: int) -> TaskOutput:
        """Record results, gradients, losses and observations of the last body."""
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
                return self.task.export.output(values[0], data, source_rows)
            if self.task.sampler_stage is not None:
                return self.task.sampler_output(data, batch, self.forward_weights)
            return TaskOutput()
        assert self.gradients is not None
        if batch.retain_gradients:
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
            update_count=len(self.updated) if self.counters is None else 0,
            trained_rows=len(data.rows) if self.counters is None else 0,
            counters=None if self.counters is None else self.counters.clone(),
            observations=training_observations(data, batch, values[0], self.inputs),
        )


@dataclass
class FieldWriteExecutor:
    """Store a write's field values into the issued private pages of the written states, captured on CUDA.

    The values are staged into persistent buffers, one row per written state; the table holds each field's
    destination slots (-1: do not write).
    """

    writer: FieldWriter
    capture: StepCapture | None = None
    buffers: tuple[torch.Tensor, ...] = ()
    targets: tuple[torch.Tensor, ...] = ()
    table: torch.Tensor | None = None
    rows: torch.Tensor | None = None
    staging: PinnedStaging = dataclass_field(default_factory=PinnedStaging)

    def resident(self, capture_bytes: int) -> int:
        """Reserve graph residency; import stores need only their table and staged values."""
        if self.capture is None:
            return 0
        if self.writer.source is None:
            # Import graphs only store into pages: every allocation is in the table and staging.
            assert self.table is not None
            return min(capture_bytes, sum(value.untyped_storage().nbytes() for value in (self.table, *self.buffers)))
        return capture_bytes

    def _store(self) -> None:
        assert self.table is not None and self.rows is not None
        for value, target, destinations in zip(self.buffers, self.targets, self.table, strict=True):
            copy_pages(value, target, self.rows, destinations)

    def __call__(
        self, data: TensorFrame, batch: TensorBatch, context: StateTaskContext[TensorFrame] | None = None
    ) -> TaskOutput:
        """Stage the selected states' values and destinations, then store them in one body or graph replay."""
        writer = self.writer
        selected = writer.select(data, batch.selection)
        if not selected:
            return TaskOutput()
        values = writer.values(data, batch, selected, validate=context is None)
        if self.capture is not None and context is not None:
            self.capture.order(batch.stream)
        parent = batch
        if any(group.resource == writer.resource for group in batch.selection.groups):
            batch = batch.for_resource(writer.resource)
        updated = tuple(batch.selection.refs.index(parent.selection.refs[i]) for i in selected)
        states = len(batch.roots) if self.capture is None else self.capture.states
        if not self.buffers or len(self.buffers[0]) != states:
            self.buffers = tuple(value.new_zeros((states, *value.shape[1:])) for value in values)
            self.table = values[0].new_full((len(writer.fields), states), -1, dtype=torch.long)
            self.rows = torch.arange(states, device=values[0].device)
        for buffer, value in zip(self.buffers, values, strict=True):
            buffer[: len(value)].copy_(value.detach())
        self.targets = tuple(view.read(batch.field_slots(view.field)[0], pooled=True) for view in writer.fields)
        table = self.table
        assert table is not None
        if context is not None:
            slots = context.owned_write_slots(batch.selection.refs.take(updated), copy_fields=None)
            with self.staging.fill(table) as destinations:
                destinations.fill(-1)
                destinations[:, : len(updated)] = slots[:, [view.field for view in writer.fields]].T
        if self.capture is None or context is None:
            self._store()
        else:
            self.capture.run(self._store, table, table, (), lambda: self.staging.copy_to(table), self.buffers)
        return TaskOutput(changed=selected, update_count=len(selected))


def retain_rows(value: torch.Tensor, count: int, *, copy: bool) -> torch.Tensor:
    """Own graph outputs; ordinary eager results retain their autograd edges."""
    value = value[:count] if value.ndim else value
    return value.detach().clone() if copy else value

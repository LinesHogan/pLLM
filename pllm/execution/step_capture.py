"""Fixed-shape tensor steps and their CUDA graphs, owned by ordinary tensor operations.

Methods must be row-wise: padding has a sink parameter row and zero reduction weight.
Training returns one loss per row. PTTS allocates and publishes versions outside the graph.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

from pllm.contracts.semantic_graph import TaskResourceFact, TaskResultInputSource, TaskSymbol
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.codec import cuda_graph_pool_bytes
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.tensor_frame import FieldRoots, RowIndex, TensorFrame, training_observations
from pllm.execution.tensor_program import PoolAdamW, TaskOutput, TensorBatch
from pllm.execution.tensor_task import TensorTask
from pllm.kernels.adamw import AdamWFields, apply_adamw, paged_adamw
from pllm.kernels.page_copy import copy_pages, gather_rows

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

    Only such tasks have a row table to capture; the rest stay on the plain TensorTask path.
    """
    return any(
        reads_rows(tasks[source.task], tasks) if isinstance(source, TaskResultInputSource) else task.ports[key].rowwise
        for key, source in task.sources.items()
    )


@dataclass
class StepCapture:
    """One task's packed index table, static outputs, and private graph pool."""

    task: TensorTask
    tasks: Mapping[TaskSymbol, TensorTask]
    entitlement: TaskResourceFact
    states: int
    optimizer: PoolAdamW | None = None
    enabled: bool = True
    graph: torch.cuda.CUDAGraph | None = None
    table: torch.Tensor | None = None
    pools: tuple[torch.Tensor, ...] = ()
    result: torch.Tensor | None = None
    gradients: torch.Tensor | None = None
    inputs: dict[str, torch.Tensor] | None = None
    input_gradients: dict[tuple[str, str], torch.Tensor] = dataclass_field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    addresses: tuple[int, ...] = ()
    hidden_inputs: Mapping[tuple[str, str, str], torch.Tensor] | None = None
    resident_bytes: int = 0
    active_rows: int = 0
    active_states: int = 0
    updated: tuple[int, ...] = ()
    read_fields: frozenset[int] = frozenset()
    feedback_names: tuple[str, ...] = ()
    feedback_values: tuple[torch.Tensor, ...] = ()
    stream: torch.cuda.Stream | None = None
    staging: PinnedStaging = dataclass_field(default_factory=PinnedStaging)

    def __post_init__(self) -> None:
        self.read_fields = self._read_fields(self.task)
        self.feedback_names = tuple(sorted(self._feedback_names(self.task)))
        if self.enabled and 8 * (6 * self.rows + self.states + 1) > self.entitlement.resident_bytes:
            raise ValueError("capture index table exceeds the task resident bytes")

    def _read_fields(self, task: TensorTask) -> frozenset[int]:
        fields = {port.field for port in task.ports.values() if port.field is not None}
        if task.policy.parameter_gradients:
            fields.add(0)
        for source in task.sources.values():
            if isinstance(source, TaskResultInputSource):
                fields.update(self._read_fields(self.tasks[source.task]))
        return frozenset(fields)

    def _feedback_names(self, task: TensorTask) -> set[str]:
        names = {port.feedback for port in task.ports.values() if port.feedback}
        for source in task.sources.values():
            if isinstance(source, TaskResultInputSource):
                names.update(self._feedback_names(self.tasks[source.task]))
        return names

    @property
    def rows(self) -> int:
        return self.entitlement.input_shapes[0][0] if self.enabled else self.active_rows

    @property
    def state_count(self) -> int:
        return self.states if self.enabled else self.active_states

    @property
    def layout(self) -> TableLayout:
        return TableLayout(self.rows, self.state_count, len(self.pools), len(self.feedback_names))

    def _fill(
        self, host: npt.NDArray[np.int64], data: TensorFrame, batch: TensorBatch, context: StateTaskContext[TensorFrame]
    ) -> None:
        """Write this step's index table into the zeroed host buffer."""
        layout = self.layout
        rows, counts, last, sources, destinations, feedback = layout.split(host)
        count = len(data.rows)
        links = {link.packed_row_index: link.sampler_row_index for link in data.step.sampler_rows.links}
        if self.feedback_names and any(row.packed_row_index not in links for row in data.rows):
            raise ValueError("sampler feedback must cover every operation row")
        states = np.fromiter(
            (batch.selection.row_states[row.identity] if batch.roots else 0 for row in data.rows), np.int64, count
        )
        versions = np.fromiter((ref.version_id for ref in batch.selection.refs), np.int64) if batch.roots else states
        rows[0, :count] = [row.packed_row_index for row in data.rows]
        rows[1, :count] = [links.get(row.packed_row_index, 0) for row in data.rows]
        rows[2, :count] = [row.identity.sample_idx for row in data.rows]
        rows[3] = layout.states
        rows[3, :count] = states
        rows[4, :count] = versions[states] if batch.roots else 0
        rows[5, :count] = 1
        counts[:] = np.bincount(states, minlength=layout.states + 1)[: layout.states + 1]
        if batch.roots:
            last[states] = np.arange(count)  # repeated states keep their last row
        sources[:] = -1
        destinations[:] = -1
        write = context.write
        if batch.roots:
            _, slots = batch.slot_matrix()
            sources[:, : len(slots)] = slots.T
        if write is not None:
            _, slots = context.arena.slots((write,) * len(write.after_refs), write.after_refs)
            destinations[:, list(self.updated)] = slots.T
        for row, value in zip(feedback, self.feedback_values, strict=True):
            row[:] = (value.data_ptr(), value.stride(0), value.stride(1) if value.ndim > 1 else 1)

    def _views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.table is not None
        rows, counts, _, sources, destinations, _ = self.layout.split(self.table)
        return rows, counts, sources, destinations

    def _dependencies(self, task: TensorTask, data: TensorFrame, batch: TensorBatch) -> None:
        for source in task.sources.values():
            if isinstance(source, TaskResultInputSource) and (source.task, source.output) not in batch.results:
                dependency = self.tasks[source.task]
                self._dependencies(dependency, data, batch)
                dependency.evaluate(data, batch)

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
        if self.optimizer is not None and self.pools[0].dtype != torch.float32:
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
            index=RowIndex((), (), (), (), (), columns[:5], fixed=True),
            fields=fields,
            hidden_inputs=self.hidden_inputs,
            feedback_inputs=self._feedback(columns[1]),
            leaf=None,
            weights=columns[5] if self.enabled or self.task.policy.reduction == "state_mean" else None,
            counts=counts,
            derivative_sources={},
            input_gradients={},
            parameter_gradient_fields={},
        )
        batch = replace(batch, roots=FieldRoots(fields), results={}, observing=False)
        if self.optimizer is not None or self.task.policy.parameter_gradients:
            frame.leaf = fields[0].detach().requires_grad_(True)
        return frame, batch

    def _body(self, data: TensorFrame, batch: TensorBatch) -> None:
        frame, batch = self._frame(data, batch)
        _, _, sources, destinations = self._views()
        self._dependencies(self.task, frame, batch)
        with torch.set_grad_enabled(self.optimizer is not None or self.task.differentiable):
            inputs, rows = self.task.read(frame, batch, compact=False)
            self.inputs = inputs
            if self.optimizer is None:
                result = self.task.function(inputs)
                if self.task.policy.forward_rows:
                    result = self.task.expand_selected(result, inputs, rows, frame)
                self.result = result.detach()
                return
            loss, gradients = self.task.differentiate(inputs, rows, frame, batch)
        self.result, self.gradients = loss.detach(), gradients[0].detach()[: self.state_count].contiguous()
        self.input_gradients = dict(
            zip(frame.derivative_sources, (value.detach() for value in gradients[1:]), strict=True)
        )
        self._update(inputs, sources, destinations, frame.fields)

    def _cpu_update(self, sources: torch.Tensor, destinations: torch.Tensor) -> None:
        assert self.optimizer is not None and self.gradients is not None
        roots: list[tuple[torch.Tensor, ...]] = []
        for row in self.updated:
            root = tuple(pool[int(destinations[field, row])] for field, pool in enumerate(self.pools))
            for field in (0, 2, 3, 4):
                root[field].copy_(self.pools[field][int(sources[field, row])])
            roots.append(root)
        apply_adamw(
            roots,
            AdamWFields(*(LayoutFieldId(i) for i in range(5))),
            self.optimizer.config,
            tuple(self.gradients[row] for row in self.updated),
        )

    def _tensor_update(self, fields: tuple[torch.Tensor, ...], destinations: torch.Tensor) -> None:
        """Other parameter dtypes use the same captured step with Torch's optimizer."""
        assert self.optimizer is not None and self.gradients is not None
        roots = tuple(tuple(value[row] for value in fields[:5]) for row in range(self.state_count))
        apply_adamw(
            roots,
            AdamWFields(*(LayoutFieldId(i) for i in range(5))),
            self.optimizer.config,
            tuple(self.gradients.unbind()),
        )
        rows = torch.arange(self.state_count, device=destinations.device)
        for field in range(5):
            copy_pages(fields[field], self.pools[field], rows, destinations[field])

    def _update(
        self,
        inputs: dict[str, torch.Tensor],
        sources: torch.Tensor,
        destinations: torch.Tensor,
        fields: tuple[torch.Tensor, ...],
    ) -> None:
        with torch.no_grad():
            assert self.optimizer is not None and self.gradients is not None
            if self.pools[0].is_cuda and self.pools[0].dtype == torch.float32:
                paged_adamw(
                    self.pools[:5],
                    destinations[:5],
                    self.gradients,
                    self.optimizer.config,
                    sources[:5, : self.state_count].contiguous(),
                )
            elif self.pools[0].is_cuda:
                self._tensor_update(fields, destinations)
            else:
                self._cpu_update(sources, destinations)
            for field in set(range(5, len(self.pools))) - set(self.task.captures.values()):
                copy_pages(
                    self.pools[field], self.pools[field], sources[field, : self.state_count], destinations[field]
                )

            assert self.table is not None
            last = self.layout.split(self.table)[2]
            for key, field in self.task.captures.items():
                value, pool = inputs[key].detach(), self.pools[field]
                if value.shape[1:] != pool.shape[1:] or value.dtype != pool.dtype:
                    raise ValueError("memory capture must match its declared field shape and dtype")
                copy_pages(value, pool, last, destinations[field])

    def _capture(self, data: TensorFrame, batch: TensorBatch) -> None:
        assert self.table is not None
        stream = torch.cuda.Stream(device=self.table.device)
        current = torch.cuda.current_stream(self.table.device)
        stream.wait_stream(current)
        # Inactive destinations make warmup and capture read-only with respect to the arena.
        with torch.cuda.stream(stream):
            self._views()[3].fill_(-1)
            for _ in range(3):
                self._body(data, batch)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
            self._body(data, batch)
        current.wait_stream(stream)
        self.resident_bytes = self.table.untyped_storage().nbytes() + cuda_graph_pool_bytes(
            self.graph.pool(), self.table.device.index or 0
        )
        if self.resident_bytes > self.entitlement.resident_bytes:
            raise ValueError(
                f"capture requires {self.resident_bytes} resident bytes, reserved {self.entitlement.resident_bytes}"
            )

    def execute(self, context: StateTaskContext[TensorFrame], batch: TensorBatch) -> TaskOutput:
        """Allocate private versions, upload one table, then execute the tensor step."""
        data = context.data()
        if not data.rows:
            return TaskOutput()
        if batch.stream is not None and self.stream is not None and batch.stream != self.stream:
            batch.stream.wait_stream(self.stream)
        self.stream = batch.stream
        self.active_rows, self.active_states = len(data.rows), len(batch.roots)
        if not self.pools and batch.roots:
            self.pools = tuple(batch.field_slots(i)[0] for i in range(len(batch.roots[0])))
        if self.optimizer is not None:
            self.updated = tuple(sorted({batch.selection.row_states[row.identity] for row in data.rows}))
            context.write = context.arena.prepare_transition(
                tuple(batch.selection.refs[i] for i in self.updated), batch.stream, copy_fields=()
            )
        self._prepare(data, batch, context)
        if self.graph is None:
            self._body(data, batch)
        else:
            self.graph.replay()
        return self._output(data, batch)

    def _prepare(self, data: TensorFrame, batch: TensorBatch, context: StateTaskContext[TensorFrame]) -> None:
        hidden = data.step.hidden_batch(batch.stream)
        addresses = tuple(item.hidden.data_ptr() for item in hidden.snapshots)
        if self.hidden_inputs is None or self.addresses != addresses:
            self.hidden_inputs = {item.resource.locator_key: item.storage for item in hidden.snapshots}
        if self.feedback_names:
            assert batch.feedback is not None
            self.feedback_values = tuple(
                batch.feedback.base_logits if name == "base_logits" else batch.feedback.sampled_tokens
                for name in self.feedback_names
            )
            if batch.stream is not None:
                for value in self.feedback_values:
                    value.record_stream(batch.stream)
        device = hidden.snapshots[0].hidden.device if hidden.snapshots else self.pools[0].device
        size = self.layout.size
        if self.table is None or self.table.numel() != size:
            self.table = torch.empty(size, dtype=torch.long, device=device)
        if device.type == "cuda":
            self._fill(self.staging.take(size).numpy(), data, batch, context)
            self.staging.copy_to(self.table)
        else:
            self.table.zero_()
            self._fill(self.table.numpy(), data, batch, context)
        if self.enabled and (self.graph is None or self.addresses != addresses):
            self._capture(data, batch)
            if device.type == "cuda":
                self.staging.copy_to(self.table)
        self.addresses = addresses

    def _output(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        assert self.result is not None
        count = (
            len(batch.roots)
            if self.optimizer is not None and self.task.policy.reduction == "state_mean"
            else len(data.rows)
        )
        result = self.result[:count].clone() if self.result.ndim else self.result.clone()
        batch.results[self.task.symbol, self.task.output] = result
        if self.optimizer is None:
            return TaskOutput()
        assert self.gradients is not None and self.inputs is not None
        gradients = self.gradients[: len(batch.roots)].clone()
        data.input_gradients.update(
            (key, value[: len(data.rows)].clone()) for key, value in self.input_gradients.items()
        )
        data.parameter_gradient_fields.update(
            (ref, (gradient,)) for ref, gradient in zip(batch.selection.refs, gradients.unbind(), strict=True)
        )
        return TaskOutput(
            loss=result,
            loss_count=len(self.updated) if self.task.policy.reduction == "state_mean" else 1,
            update_count=len(self.updated),
            trained_rows=len(data.rows),
            changed=self.updated,
            observations=training_observations(data, batch, result, self.inputs, tuple(range(len(data.rows)))),
        )

"""Fixed-shape tensor steps and their CUDA graphs, owned by ordinary tensor operations.

Methods must be row-wise: padding has a sink parameter row and zero reduction weight.
Training returns one loss per row. PTTS allocates and publishes versions outside the graph.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field

import torch

from pllm.contracts.semantic_graph import TaskResourceFact, TaskResultInputSource, TaskSymbol
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.codec import cuda_graph_pool_bytes
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.tensor_frame import FieldRoots, RowIndex, TensorFrame, training_observations
from pllm.execution.tensor_program import PoolAdamW, TaskOutput, TensorBatch
from pllm.execution.tensor_task import TensorTask
from pllm.kernels.adamw import AdamWFields, apply_adamw, paged_adamw
from pllm.kernels.page_copy import copy_pages, gather_rows


def hidden_buffer(value: torch.Tensor) -> torch.Tensor:
    """Recover the fixed tap buffer behind a snapshot's leading-row view."""
    elements = value.untyped_storage().nbytes() // value.element_size() - int(value.storage_offset())
    row_width = (value.shape[1] - 1) * value.stride(1) + 1
    rows = 1 + (elements - row_width) // value.stride(0)
    return value.as_strided((rows, value.shape[1]), value.stride())


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

    def _columns(self, data: TensorFrame, batch: TensorBatch, context: StateTaskContext[TensorFrame]) -> list[int]:
        rows, states = self.rows, self.state_count
        links = {link.packed_row_index: link.sampler_row_index for link in data.step.sampler_rows.links}
        if self.feedback_names and any(row.packed_row_index not in links for row in data.rows):
            raise ValueError("sampler feedback must cover every operation row")
        indices = (
            [batch.selection.row_states[row.identity] for row in data.rows] if batch.roots else [0] * len(data.rows)
        )
        counts = Counter(indices)
        columns = [
            [row.packed_row_index for row in data.rows],
            [links.get(row.packed_row_index, 0) for row in data.rows],
            [row.identity.sample_idx for row in data.rows],
            indices,
            [int(batch.selection.refs[i].version_id) for i in indices] if batch.roots else indices,
            [1] * len(data.rows),
        ]
        packed = [
            value
            for i, column in enumerate(columns)
            for value in (*column, *([states if i == 3 else 0] * (rows - len(column))))
        ]
        packed.extend(counts[i] for i in range(states + 1))
        last = {state: row for row, state in enumerate(indices)}
        packed.extend(last.get(i, 0) for i in range(states))
        for field in range(len(self.pools)):
            _, slots = batch.field_slots(field)
            packed.extend((*slots, *([-1] * (states + 1 - len(slots)))))
        write = context.write
        roots = None if write is None else context.arena.lazy_roots((write,) * len(write.after_refs), write.after_refs)
        for field in range(len(self.pools)):
            slots = {} if roots is None else dict(zip(self.updated, roots.field_slots(field)[1], strict=True))
            packed.extend(slots.get(state, -1) for state in range(states))
        for value in self.feedback_values:
            packed.extend((value.data_ptr(), value.stride(0), value.stride(1) if value.ndim > 1 else 1))
        return packed

    def _views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.table is not None
        rows, states, fields = self.rows, self.state_count, len(self.pools)
        row_end = 6 * rows
        count_end = row_end + states + 1
        slot_start = count_end + states
        source_end = slot_start + fields * (states + 1)
        return (
            self.table[:row_end].view(6, rows),
            self.table[row_end:count_end],
            self.table[slot_start:source_end].view(fields, states + 1),
            self.table[source_end : source_end + fields * states].view(fields, states),
        )

    def _dependencies(self, task: TensorTask, data: TensorFrame, batch: TensorBatch) -> None:
        for source in task.sources.values():
            if isinstance(source, TaskResultInputSource) and (source.task, source.output) not in batch.results:
                dependency = self.tasks[source.task]
                self._dependencies(dependency, data, batch)
                dependency.evaluate(data, batch)

    def _feedback(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.table is not None
        start = self.table.numel() - 3 * len(self.feedback_names)
        return {
            name: gather_rows(self.table[start + 3 * i : start + 3 * i + 3], rows, value)
            for i, (name, value) in enumerate(zip(self.feedback_names, self.feedback_values, strict=True))
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
            last = self.table[6 * self.rows + self.state_count + 1 : 6 * self.rows + 2 * self.state_count + 1]
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
            self.hidden_inputs = {item.resource.locator_key: hidden_buffer(item.hidden) for item in hidden.snapshots}
        if self.feedback_names:
            assert batch.feedback is not None
            self.feedback_values = tuple(
                batch.feedback.base_logits if name == "base_logits" else batch.feedback.sampled_tokens
                for name in self.feedback_names
            )
            if batch.stream is not None:
                for value in self.feedback_values:
                    value.record_stream(batch.stream)
        values = self._columns(data, batch, context)
        device = hidden.snapshots[0].hidden.device if hidden.snapshots else self.pools[0].device
        host = torch.tensor(values, dtype=torch.long, pin_memory=device.type == "cuda")
        if self.table is None or not self.enabled:
            self.table = torch.empty(host.shape, dtype=torch.long, device=device)
        self.table.copy_(host, non_blocking=True)
        if self.enabled and (self.graph is None or self.addresses != addresses):
            self._capture(data, batch)
            self.table.copy_(host, non_blocking=True)
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

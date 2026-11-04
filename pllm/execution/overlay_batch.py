"""Bounded overlay tiles over PTTS-issued slots; the task retains all page ownership."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.parameter_overlay import ParameterOverlayOperator, ProjectionOperands
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.semantic_graph import TaskResourceFact
from pllm.contracts.tensor_policy import TensorPolicy
from pllm.execution.cuda_graph import device_branch
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.overlay_task import ChunkUpdate, GradientUpdate, save_tail
from pllm.execution.scan import execute_scan
from pllm.execution.scan_analysis import chunk_step, row_readout
from pllm.execution.scan_execution import PreparedScan
from pllm.execution.step_capture import StepCapture
from pllm.execution.tensor_frame import Tensors, UpdateOutput
from pllm.execution.tensor_program import TensorBatch
from pllm.execution.torch_private import measure
from pllm.kernels.page_copy import copy_pages

if TYPE_CHECKING:
    from pllm.execution.overlay_task import OverlayTransition


class OverlayBatch:
    """At most eight captured shapes per target: four row capacities and two phases."""

    states = 8

    def __init__(self, math: OverlayTransition, budget: TaskResourceFact | None = None) -> None:
        self.math, self.budget = math, budget
        if budget is not None:
            self.states = min(self.states, budget.input_shapes[0][0])
        self.tiles: dict[tuple[int, bool, int], OverlayTile] = {}
        self.batchable = False

    def resident(self, capture_bytes: int) -> int:
        """Reserve the planner's capture budget for batchable overlays."""
        return capture_bytes if self.budget is not None and self.batchable else 0

    def capacities(self) -> tuple[int, ...]:
        """Pad within the native power-of-two chunk-count bucket used by scan selection."""
        assert self.math.chunk is not None
        chunk = self.math.chunk.chunk
        return tuple(dict.fromkeys((1, chunk + 1, 3 * chunk + 1, 7 * chunk + 1)))

    def run(
        self,
        operands: tuple[ProjectionOperands, ...],
        target: int,
        indices: tuple[int, ...],
        batch: TensorBatch,
    ) -> tuple[tuple[UpdateOutput, ...], torch.Tensor]:
        """Bucket actual rows, then tile states; oversized rows stay eager at native size."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i, operand in enumerate(operands):
            rows = len(operand.hidden)
            groups[next((size for size in self.capacities() if rows <= size), rows)].append(i)
        train = operands[0].phase in self.math.phases
        keys = tuple((target, train, rows) for rows in groups)
        tile = self.tiles.get(keys[0]) if len(keys) == 1 else None
        if tile is not None and not tile.batchable:
            return self.math.serial(operands, target, indices, batch, plan=tile.plan)
        results: dict[int, UpdateOutput] = {}
        counters: list[torch.Tensor] = []
        width = len(self.math.buffers)
        fields = tuple(batch.field_slots(target * width + i) for i in range(width))
        pools = tuple(pool for pool, _ in fields)
        slots = np.array(tuple(slots for _, slots in fields))
        invariants = self.math.invariant_values(target, batch)
        for rows, members in groups.items():
            for start in range(0, len(members), self.states):
                selected = members[start : start + self.states]
                tile = self.tile(target, train, rows, pools, operands[selected[0]], len(selected), invariants)
                items = tuple(operands[i] for i in selected)
                if not tile.batchable:
                    values, counts = self.math.serial(
                        items, target, tuple(indices[i] for i in selected), batch, plan=tile.plan
                    )
                else:
                    values, counts = tile.run(items, slots[:, [indices[i] for i in selected]], pools, batch.stream)
                results.update(zip(selected, values, strict=True))
                counters.append(counts)
        return tuple(results[i] for i in range(len(operands))), torch.stack(counters).sum(0)

    def tile(
        self,
        target: int,
        train: bool,
        rows: int,
        pools: tuple[torch.Tensor, ...],
        operand: ProjectionOperands,
        count: int,
        invariants: Tensors,
    ) -> OverlayTile:
        key = target, train, rows
        bounded = self.budget is not None and self.budget.resident_bytes > 0 and rows in self.capacities()
        capture = bounded and pools[0].is_cuda
        if key in self.tiles:
            tile = self.tiles[key]
            tile.invariants = invariants
            if tile.capture is not None and self.budget is not None:
                tile.capture.budget = replace(self.budget, resident_bytes=self.available(key))
            return tile
        tile = OverlayTile(self.math, pools, operand, self.states if bounded else count, rows, train, invariants)
        if not tile.batchable:
            self.tiles[key] = tile
            return tile
        self.batchable = True
        if capture and self.budget is not None:
            remaining = self.available(key)
            with FakeTensorMode(allow_non_fake_inputs=True):
                capture_bytes = measure(partial(tile.body, operand.base_weight)).capture
            needed = capture_bytes + tile.buffer_bytes
            if needed <= remaining:
                tile.capture = StepCapture(replace(self.budget, resident_bytes=remaining), self.states)
                self.tiles[key] = tile

        return tile

    def available(self, key: tuple[int, bool, int]) -> int:
        """An invalidated shape may reuse its budget, but never another live graph's bytes."""
        assert self.budget is not None
        return self.budget.resident_bytes - sum(
            tile.capture.resident_bytes
            for shape, tile in self.tiles.items()
            if shape != key and tile.capture is not None
        )


class OverlayTile:
    """Fixed mathematical inputs and capture state, with no request or page ownership."""

    def __init__(
        self,
        math: OverlayTransition,
        pools: tuple[torch.Tensor, ...],
        operand: ProjectionOperands,
        states: int,
        rows: int,
        train: bool,
        invariants: Tensors,
    ) -> None:
        self.math, self.pools, self.train = math, pools, train
        self.invariants = invariants
        chunk = math.chunk
        assert chunk is not None
        values = dict(zip(math.buffers, (pool[0] for pool in pools), strict=True))
        self.plan = (
            chunk.prepare(
                self.arguments(
                    values, operand.hidden, operand.desired, operand.embedding, operand.base_weight, operand.prompt_end
                )
            )
            if train
            else None
        )
        # Serial execution retains its prepared plan without padded tensor capacity.
        if not self.batchable:
            states = rows = 0
        self.prompt_end = operand.prompt_end.new_zeros((states,))
        self.table = torch.zeros((2 * len(pools) + 1, states), device=pools[0].device, dtype=torch.int64)
        self.hidden = operand.hidden.new_zeros((states, rows, operand.hidden.shape[1]))
        self.desired = operand.desired.new_zeros((states, rows, operand.desired.shape[1]))
        self.embedding = operand.embedding.new_zeros((states, rows, operand.embedding.shape[-1]))
        self.sources = torch.arange(states, device=pools[0].device)
        self.staging = PinnedStaging()
        self.capture: StepCapture | None = None
        self.output: tuple[torch.Tensor, ...] = ()

    @property
    def batchable(self) -> bool:
        """Dense source GEMMs retain their selected reduction order on the serial path."""
        if self.plan is None:
            return self.math.operator is not ParameterOverlayOperator.DENSE_ADD or isinstance(
                self.math.chunk, GradientUpdate
            )
        return (
            self.math.operator is ParameterOverlayOperator.HEAD_ADD
            if self.plan.analysis is None
            else self.plan.analysis.kind == "additive"
        )

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        return self.hidden, self.desired, self.embedding, self.sources, self.prompt_end

    @property
    def buffer_bytes(self) -> int:
        return sum(value.untyped_storage().nbytes() for value in (self.table, *self.buffers))

    def arguments(
        self,
        values: Tensors,
        hidden: torch.Tensor,
        desired: torch.Tensor,
        embedding: torch.Tensor,
        base: torch.Tensor,
        prompt_end: torch.Tensor,
    ) -> Tensors:
        inputs = {
            **values,
            **self.invariants,
            "input": hidden,
            "target": desired,
            "embedding": embedding,
            "weight": base,
            "prompt_end": prompt_end,
        }
        return {key: inputs[source] for key, source in self.math.arguments.items()}

    def one(
        self,
        values: Tensors,
        hidden: torch.Tensor,
        desired: torch.Tensor,
        embedding: torch.Tensor,
        count: torch.Tensor,
        base: torch.Tensor,
    ):
        chunk = self.math.chunk
        assert chunk is not None
        if self.train:
            assert isinstance(chunk, ChunkUpdate) and self.plan is not None
            result = chunk.counted(
                self.plan, count, self.arguments(values, hidden, desired, embedding, base, self.prompt_end)
            )
        else:
            result = UpdateOutput.unchanged(self.math.apply(hidden, values))
        return values, result.output, result.changed, result.update_count, result.trained_rows

    def body(self, base: torch.Tensor) -> None:
        """Gather, map the authored step, and write only the table's issued destinations."""
        width = len(self.pools)
        values = {
            name: pool.index_select(0, self.table[i])
            for i, (name, pool) in enumerate(zip(self.math.buffers, self.pools, strict=True))
        }
        if self.train:
            values["_pending_length"].masked_fill_(self.table[width] < 0, 0)
        chunk = self.math.chunk
        if self.train and isinstance(chunk, GradientUpdate):
            result = chunk.counted(
                None,
                self.table[-1],
                self.arguments(values, self.hidden, self.desired, self.embedding, base, self.prompt_end),
            )
            fields, output, changed, updates, trained = (
                values,
                result.output,
                result.changed,
                result.update_count,
                result.trained_rows,
            )
        else:
            fields, output, changed, updates, trained = torch.vmap(self.one, in_dims=(0, 0, 0, 0, 0, None))(
                values, self.hidden, self.desired, self.embedding, self.table[-1], base
            )
        self.output = (
            output,
            changed,
            updates,
            trained,
            fields["_pending_length"],
            torch.stack((updates.sum(), trained.sum())),
        )
        if self.train:
            for i, (name, pool) in enumerate(zip(self.math.buffers, self.pools, strict=True)):
                copy_pages(fields[name], pool, self.sources, self.table[width + i])

    def run(
        self,
        operands: tuple[ProjectionOperands, ...],
        slots: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
        pools: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream | None,
    ) -> tuple[tuple[UpdateOutput, ...], torch.Tensor]:
        """Fence staging through PinnedStaging and detach outputs before the next tile replay."""
        self.pools = pools
        base = operands[0].base_weight
        if self.capture is not None:
            self.capture.order(stream)
        width, count = len(pools), len(operands)
        with self.staging.fill(self.table) as host:
            host[:width, :count] = slots
            host[width : 2 * width] = -1
            if self.train:
                host[width : 2 * width, :count] = slots
            host[-1, :count] = tuple(len(o.hidden) for o in operands)
        self.hidden.zero_()
        self.desired.zero_()
        self.embedding.zero_()
        self.prompt_end.zero_()
        for i, operand in enumerate(operands):
            self.hidden[i, : len(operand.hidden)].copy_(operand.hidden)
            self.desired[i, : len(operand.hidden)].copy_(operand.desired)
            if "embedding" in self.math.arguments.values():
                self.embedding[i, : len(operand.hidden)].copy_(operand.embedding)
            self.prompt_end[i].copy_(operand.prompt_end)
        if self.capture is None:
            self.body(base)
        else:
            self.capture.run(
                partial(self.body, base),
                self.table,
                self.table[width : 2 * width],
                (
                    *tuple(pool.data_ptr() for pool in pools),
                    base.data_ptr(),
                    *(value.data_ptr() for value in self.invariants.values()),
                ),
                partial(self.staging.copy_to, self.table),
                self.buffers,
            )
        output, changed, updates, trained, lengths, counters = (value.clone() for value in self.output)
        return tuple(
            self.math.validate(
                UpdateOutput(
                    output[i, : len(o.hidden)],
                    changed[i],
                    updates[i],
                    trained[i],
                    {"tail_rows": lengths[i]} if self.train else {},
                ),
                o,
            )
            for i, o in enumerate(operands)
        ), counters


class DecodeOverlay:
    """Device-counted one-row transitions over private slots, with one update branch per request bucket."""

    def __init__(
        self,
        math: OverlayTransition,
        target: int,
        pools: tuple[torch.Tensor, ...],
        invariants: Tensors,
        requests: int,
        policy: TensorPolicy,
        key_fields: tuple[RequestKeyField, ...],
    ) -> None:
        self.key_fields = key_fields
        self.math, self.target, self.pools, self.policy = math, target, pools, policy
        self.invariants = dict(invariants)
        self.table = torch.zeros((len(pools) + 1, requests), device=pools[0].device, dtype=torch.long)
        self.source = self.table.new_zeros((1,))
        self.staging = PinnedStaging(capacity=self.table.numel(), slots=2)
        self.host = np.empty(0, dtype=np.int64)
        self.outputs: dict[int, tuple[UpdateOutput, ...]] = {}
        self.plan: PreparedScan | None = None
        self.read_full: Callable[..., torch.Tensor] | None = None
        self.read_row: Callable[..., torch.Tensor] | None = None

    @property
    def buffers(self) -> list[torch.Tensor]:
        """All mutable storage passed through the model's opaque projection op."""
        return [*self.pools, self.table, *self.invariants.values()]

    def reserve(self) -> None:
        """Acquire the staging slot before the invocation opens private pages."""
        self.host = self.staging.take(self.table.numel()).reshape(self.table.shape)

    def prepare(self, batch: TensorBatch, identities: tuple[RequestIdentity, ...]) -> None:
        """Upload the task's private slots; padded requests have no write destination."""
        width = len(self.math.buffers)
        batch = batch.for_resource(self.math.resource)
        indices = {
            row.identity: index for row, index in zip(batch.selection.rows, batch.selection.state_indices, strict=True)
        }
        fields = tuple(batch.field_slots(self.target * width + i) for i in range(width))
        self.host[-1, : len(identities)] = 1
        for i, (_, slots) in enumerate(fields):
            self.host[i, : len(identities)] = tuple(slots[indices[identity]] for identity in identities)
        self.staging.copy_to(self.table)
        for name, value in self.math.invariant_values(self.target, batch).items():
            self.invariants[name].copy_(value)

    def values(self, index: int) -> dict[str, torch.Tensor]:
        """Gather one request without combining its source GEMMs with other requests."""
        slots = self.table[:-1, index]
        return {
            name: pool.index_select(0, slots[i : i + 1])[0]
            for i, (name, pool) in enumerate(zip(self.math.buffers, self.pools, strict=True))
        }

    def arguments(
        self, values: Tensors, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> Tensors:
        """Bind the same state and model operands for counted chunks and decode completion."""
        inputs = {
            **values,
            **self.invariants,
            "input": hidden,
            "target": desired,
            "embedding": embedding,
            "weight": base,
            "prompt_end": hidden.new_zeros((), dtype=torch.bool),
        }
        return {key: inputs[source] for key, source in self.math.arguments.items()}

    def prepare_scan(self, arguments: Tensors) -> None:
        """Cut readouts and prepare the ordinary scan before any capacity graph is captured."""
        chunk = self.math.chunk
        if self.plan is not None or not isinstance(chunk, ChunkUpdate):
            return
        carry, *columns = chunk.arguments
        samples = (
            arguments[carry],
            *(arguments[f"_pending_rows_{name}"] for name in columns),
            *(arguments[name] for name in chunk.invariants),
        )
        specs = tuple((t.shape, t.stride(), t.dtype, t.device) for t in samples)
        self.plan = PreparedScan(chunk_step(chunk.step, specs, self.policy.allow_numerical_transforms), chunk.chunk)
        self.read_full = chunk.readout(arguments)
        if self.policy.allow_numerical_transforms:
            self.read_row = row_readout(chunk.step, specs, len(columns))

    def store(self, index: int, values: Tensors, active: torch.Tensor) -> None:
        """Write only issued destinations, leaving padding and inactive carries untouched."""
        slots = self.table[:-1, index]
        for i, (name, pool) in enumerate(zip(self.math.buffers, self.pools, strict=True)):
            copy_pages(values[name].unsqueeze(0), pool, self.source, slots[i : i + 1].masked_fill(~active, -1))

    def run(
        self, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> torch.Tensor:
        """Read incoming carries, append arriving rows, then update the device-selected completion bucket."""
        if isinstance(self.math.chunk, ChunkUpdate) and self.plan is None:
            arguments = self.arguments(self.values(0), hidden[:1], desired[:1], embedding[:1], base)
            self.prepare_scan(arguments)
        if self.read_full is None:
            results = tuple(self.counted(index, hidden, desired, embedding, base) for index in range(len(hidden)))
        else:
            results_list: list[UpdateOutput] = []
            local = RequestKeyField.REQUEST_ID in self.key_fields
            for index in range(len(hidden)):
                results_list.append(self.decode_row(index, hidden, desired, embedding, base))
                if not local:
                    # Shared keys retain read/update order between requests in the same forward.
                    self.complete_bucket((index,), results_list, hidden[:0], desired[:0], embedding[:0], base)
            if local:
                self.complete_bucket(
                    tuple(range(len(hidden))), results_list, hidden[:0], desired[:0], embedding[:0], base
                )
            results = tuple(results_list)
        self.outputs[len(hidden)] = results
        return torch.cat(tuple(result.output for result in results))

    def complete_bucket(
        self,
        indices: tuple[int, ...],
        results: list[UpdateOutput],
        hidden: torch.Tensor,
        desired: torch.Tensor,
        embedding: torch.Tensor,
        base: torch.Tensor,
    ) -> None:
        """Use the shared CUDA IF machinery once for requests whose updates can share temporary memory."""
        with device_branch(torch.stack(tuple(results[index].changed for index in indices)).any()):
            for index in indices:
                self.complete_row(index, results[index].changed, hidden, desired, embedding, base)

    def counted(
        self, index: int, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> UpdateOutput:
        """Keep opaque readouts and non-chunk updates on their ordinary tensor path."""
        values, count = self.values(index), self.table[-1, index]
        chunk = self.math.chunk
        if chunk is not None:
            values["_pending_length"].masked_fill_(count == 0, 0)
        arguments = self.arguments(
            values, hidden[index : index + 1], desired[index : index + 1], embedding[index : index + 1], base
        )
        if isinstance(chunk, ChunkUpdate):
            assert self.plan is not None
            result = chunk.counted(self.plan, count, arguments)
        elif isinstance(chunk, GradientUpdate):
            result = chunk.advance(arguments, count, batched=False)
        else:
            result = self.math.function(**arguments)
        self.store(index, values, count > 0)
        return result

    def complete_row(
        self,
        index: int,
        complete: torch.Tensor,
        hidden: torch.Tensor,
        desired: torch.Tensor,
        embedding: torch.Tensor,
        base: torch.Tensor,
    ) -> None:
        """Share one conditional allocator across requests while the scan masks each incomplete carry."""
        chunk = self.math.chunk
        assert isinstance(chunk, ChunkUpdate) and self.plan is not None
        values = self.values(index)
        arguments = self.arguments(values, hidden, desired, embedding, base)
        carry, *columns = chunk.arguments
        tails = tuple(arguments[f"_pending_rows_{name}"] for name in columns)
        # The store supplies the per-request mask; a second full-matrix where inside
        # the scan would retain another dense temporary in every conditional graph.
        updated, _ = execute_scan(
            self.plan, arguments[carry], tails, tuple(arguments[name] for name in chunk.invariants), None
        )
        values[self.math.arguments[carry]] = updated
        length = values["_pending_length"]
        full = length.new_full((), chunk.chunk)
        save_tail(tails, tails, length, full, full)
        self.store(index, values, complete)

    def decode_row(
        self, index: int, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> UpdateOutput:
        """Append a device-indexed row and read its chunk's incoming carry at the permitted GEMM shape."""
        chunk = self.math.chunk
        assert isinstance(chunk, ChunkUpdate) and self.read_full is not None
        carry, *columns = chunk.arguments
        slots, count = self.table[:-1, index], self.table[-1, index]
        fields = {name: i for i, name in enumerate(self.math.buffers)}
        carry_field, length_field = fields[self.math.arguments[carry]], fields["_pending_length"]
        state = self.pools[carry_field].index_select(0, slots[carry_field : carry_field + 1])[0]
        length = (
            self.pools[length_field]
            .index_select(0, slots[length_field : length_field + 1])[0]
            .masked_fill(count == 0, 0)
        )
        inputs = {
            **self.invariants,
            "input": hidden[index : index + 1],
            "target": desired[index : index + 1],
            "embedding": embedding[index : index + 1],
            "weight": base,
        }
        rows = tuple(inputs[self.math.arguments[name]] for name in columns)
        invariants = tuple(inputs[self.math.arguments[name]] for name in chunk.invariants)
        for name, row in zip(columns, rows, strict=True):
            field = fields[f"_pending_rows_{name}"]
            destination = (slots[field] * chunk.chunk + length).reshape(1).masked_fill(count == 0, -1)
            copy_pages(row, self.pools[field].flatten(0, 1), self.source, destination)
        if self.read_row is None:
            tails = tuple(
                self.pools[fields[f"_pending_rows_{name}"]].index_select(
                    0, slots[fields[f"_pending_rows_{name}"]].reshape(1)
                )[0]
                for name in columns
            )
            output = self.read_full(state, *tails, *invariants).index_select(0, length.reshape(1))
        else:
            output = self.read_row(state, *rows, *invariants)
        valid = length + count
        complete, remaining = (valid == chunk.chunk) & (count > 0), valid % chunk.chunk
        copy_pages(
            remaining.reshape(1),
            self.pools[length_field],
            self.source,
            slots[length_field : length_field + 1].masked_fill(count == 0, -1),
        )
        updates = complete.to(torch.int64)
        return UpdateOutput(output, complete, updates, updates * chunk.chunk, {"tail_rows": remaining})

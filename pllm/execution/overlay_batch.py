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
from pllm.execution.overlay_task import ChunkUpdate, GradientUpdate
from pllm.execution.ptts.slot_pools import Slots
from pllm.execution.scan import execute_scan
from pllm.execution.scan_analysis import DenseReadout, chunk_step, dense_readout, row_readout
from pllm.execution.scan_execution import PreparedScan
from pllm.execution.step_capture import StepCapture
from pllm.execution.tensor_frame import Tensors, UpdateOutput
from pllm.execution.tensor_program import TensorBatch
from pllm.execution.torch_private import measure
from pllm.kernels.page_copy import append_tail, clear_tail, copy_pages
from pllm.kernels.paged_projection import paged_dense_mv

if TYPE_CHECKING:
    from pllm.execution.overlay_task import OverlayTransition


def store_fields(
    names: tuple[str, ...],
    pools: tuple[torch.Tensor, ...],
    values: Tensors,
    sources: torch.Tensor,
    destinations: torch.Tensor,
) -> None:
    """Copy each field's `sources` rows to its row of page `destinations`; a negative destination is skipped."""
    for name, pool, destination in zip(names, pools, destinations, strict=True):
        copy_pages(values[name], pool, sources, destination)


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
    ) -> tuple[UpdateOutput, ...]:
        """Bucket actual rows, then tile states; oversized rows stay eager at native size."""
        train = operands[0].phase in self.math.phases
        if train and len(set(indices)) < len(indices):
            # Bucketing by length could move a later encounter ahead of the same key's earlier write.
            return tuple(
                self.run((operand,), target, (index,), batch)[0]
                for operand, index in zip(operands, indices, strict=True)
            )
        groups: dict[int, list[int]] = defaultdict(list)
        for i, operand in enumerate(operands):
            rows = len(operand.hidden)
            groups[next((size for size in self.capacities() if rows <= size), rows)].append(i)
        results: dict[int, UpdateOutput] = {}
        width = len(self.math.buffers)
        fields = tuple(batch.field_slots(target * width + i) for i in range(width))
        pools = tuple(pool for pool, _ in fields)
        slots = np.array(tuple(slots for _, slots in fields), dtype=np.int64)
        invariants = self.math.invariant_values(target, batch)
        for rows, members in groups.items():
            for start in range(0, len(members), self.states):
                selected = members[start : start + self.states]
                tile = self.tile(target, train, rows, pools, operands[selected[0]], len(selected), invariants)
                items = tuple(operands[i] for i in selected)
                states = tuple(indices[i] for i in selected)
                if tile.batchable:
                    reads, writes = self.page_slots(target, slots[:, states], batch, train)
                    values = tile.run(items, reads, writes, pools, batch.stream)
                else:
                    values = self.serial(items, target, slots[:, states], pools, batch, tile.plan)
                results.update(zip(selected, values, strict=True))
        return tuple(results[i] for i in range(len(operands)))

    def page_slots(self, target: int, slots: Slots, batch: TensorBatch, train: bool) -> tuple[Slots, Slots]:
        width = len(self.math.buffers)
        pairs = np.array(
            [[batch.overlay_slots(target * width + i, slot, train) for slot in field] for i, field in enumerate(slots)],
            dtype=np.int64,
        )
        return pairs[:, :, 0], pairs[:, :, 1]

    def serial(
        self,
        operands: tuple[ProjectionOperands, ...],
        target: int,
        slots: Slots,
        pools: tuple[torch.Tensor, ...],
        batch: TensorBatch,
        plan: PreparedScan | None,
    ) -> tuple[UpdateOutput, ...]:
        """Resolve slots in encounter order so a shared key observes the preceding update."""
        math, chunk = self.math, self.math.chunk
        rows = {row.identity: row for row in batch.selection.rows}
        invariants = math.invariant_values(target, batch)
        results: list[UpdateOutput] = []
        for i, operand in enumerate(operands):
            train = operand.phase in math.phases
            reads, writes = self.page_slots(target, slots[:, i : i + 1], batch, train)
            values = {name: pool[src] for name, pool, src in zip(math.buffers, pools, reads[:, 0], strict=True)}
            values.update(invariants)
            if not train or not isinstance(chunk, ChunkUpdate):
                results.append(math.result(operand, values))
                continue
            assert plan is not None
            dest = {
                name: pool[dst] for name, pool, dst in zip(math.buffers, pools, writes[:, 0], strict=True) if dst >= 0
            }
            destinations = {key: dest[name] for key, name in math.arguments.items() if name in dest}
            arguments = math.inputs(operand, values)
            row = rows[operand.identity]
            offset = math.chunk_offset(row, len(operand.hidden)) if row.phase is operand.phase else None
            if offset is None and batch.deferred is None:
                results.append(math.result(operand, values))
            elif offset is None:
                count = values["_pending_length"].new_full((), len(operand.hidden))
                results.append(chunk.counted(chunk.prepare(arguments), count, arguments, destinations))
            else:
                results.append(chunk.contiguous(arguments, plan, offset, destinations))
        return tuple(results)

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
        self.plan = chunk.prepare(math.inputs(operand, {**values, **invariants})) if train else None
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
                self.plan,
                count,
                self.math.bind({**values, **self.invariants}, hidden, desired, embedding, base, self.prompt_end),
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
            values["_pending_length"].masked_fill_(self.table[-1] == 0, 0)
        chunk = self.math.chunk
        if self.train and isinstance(chunk, GradientUpdate):
            result = chunk.advance(
                self.math.bind(
                    {**values, **self.invariants}, self.hidden, self.desired, self.embedding, base, self.prompt_end
                ),
                self.table[-1],
                batched=True,
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
        self.output = output, changed, updates, trained, fields["_pending_length"]
        if self.train:
            store_fields(self.math.buffers, self.pools, fields, self.sources, self.table[width : 2 * width])

    def run(
        self,
        operands: tuple[ProjectionOperands, ...],
        reads: Slots,
        writes: Slots,
        pools: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream | None,
    ) -> tuple[UpdateOutput, ...]:
        """Fence staging through PinnedStaging and detach outputs before the next tile replay."""
        self.pools = pools
        base = operands[0].base_weight
        if self.capture is not None:
            self.capture.order(stream)
        width, count = len(pools), len(operands)
        with self.staging.fill(self.table) as host:
            host[:width, :count] = reads
            host[:width, count:] = 0
            host[width : 2 * width] = -1
            host[width : 2 * width, :count] = writes
            host[-1, :count] = tuple(len(o.hidden) for o in operands)
            host[-1, count:] = 0
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
        output, changed, updates, trained, lengths = (value.clone() for value in self.output)
        return tuple(
            UpdateOutput(
                output[i, : len(o.hidden)],
                changed[i],
                updates[i],
                trained[i],
                {"tail_rows": lengths[i]} if self.train else {},
            )
            for i, o in enumerate(operands)
        )


class DecodeOverlay:
    """Device-counted decode reads each incoming carry and scans only its completing lane."""

    def __init__(
        self,
        math: OverlayTransition,
        target: int,
        pools: tuple[torch.Tensor, ...],
        invariants: Tensors,
        requests: int,
        policy: TensorPolicy,
        key_fields: tuple[RequestKeyField, ...],
        *,
        mutable_invariants: frozenset[str] | None = None,
        batch_invariant: bool = False,
    ) -> None:
        self.key_fields = key_fields
        self.math, self.target, self.pools, self.policy = math, target, pools, policy
        self.invariants = dict(invariants)
        self.mutable_invariants = frozenset(invariants) if mutable_invariants is None else mutable_invariants
        self.batch_invariant = batch_invariant
        width = len(pools)
        self.tail_fields = (
            tuple(math.buffers.index(f"_pending_rows_{name}") for name in math.chunk.arguments[1:])
            if isinstance(math.chunk, ChunkUpdate)
            else ()
        )
        self.tail_bounds = {
            i: torch.zeros((len(pools[i]), 2), device=pools[i].device, dtype=torch.long) for i in self.tail_fields
        }
        for bounds in self.tail_bounds.values():
            bounds[:, 0].fill_(-1)
        self.table = torch.zeros(
            (2 * width + len(self.tail_fields) + 1, requests), device=pools[0].device, dtype=torch.long
        )
        self.table[width : 2 * width].fill_(-1)
        self.source = self.table.new_zeros((1,))
        self.sources = torch.arange(requests, device=pools[0].device, dtype=torch.long)
        self.staging = PinnedStaging(capacity=self.table.numel(), slots=2)
        self.host = np.empty(0, dtype=np.int64)
        self.outputs: dict[int, tuple[UpdateOutput, ...]] = {}
        self.plan: PreparedScan | None = None
        self.read_full: Callable[..., torch.Tensor] | None = None
        self.read_row: Callable[..., torch.Tensor] | None = None
        self.dense_readout: DenseReadout | None = None
        self.prompt_end = self.table.new_zeros((), dtype=torch.bool)

    @property
    def buffers(self) -> list[torch.Tensor]:
        """All mutable storage passed through the model's opaque projection op."""
        return [
            *self.pools,
            self.table,
            *self.tail_bounds.values(),
            *(self.invariants[name] for name in self.mutable_invariants),
        ]

    def reserve(self) -> None:
        """Acquire the staging slot before the invocation opens private pages."""
        self.host = self.staging.take(self.table.numel()).reshape(self.table.shape)

    def prepare(
        self,
        batch: TensorBatch,
        identities: tuple[RequestIdentity, ...],
        generations: Slots | None,
    ) -> None:
        """Upload read and writable slots; shared carries and padded lanes get -1 write destinations."""
        width = len(self.math.buffers)
        batch = batch.for_resource(self.math.resource)
        indices = {
            row.identity: int(index)
            for row, index in zip(batch.selection.rows, batch.selection.state_indices, strict=True)
        }
        fields = tuple(batch.field_slots(self.target * width + i) for i in range(width))
        self.host[width : 2 * width] = -1
        self.host[-1, : len(identities)] = 1
        for i, (_, slots) in enumerate(fields):
            field_id = self.target * width + i
            for lane, identity in enumerate(identities):
                index = indices[identity]
                self.host[i, lane], self.host[width + i, lane] = batch.overlay_slots(field_id, slots[index], True)
        if generations is not None:
            states = [indices[identity] for identity in identities]
            for j, field in enumerate(self.tail_fields):
                self.host[2 * width + j, : len(states)] = generations[states, self.target * width + field]
        self.staging.copy_to(self.table)
        if self.mutable_invariants:
            values = self.math.invariant_values(self.target, batch)
            for name in self.mutable_invariants:
                self.invariants[name].copy_(values[name])

    def page(self, index: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A request's readable page of one field: the field's pool and a one-element slot index."""
        field = self.math.buffers.index(name)
        return self.pools[field], self.table[field, index : index + 1]

    def write_page(self, index: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A request's writable destination slot (-1 when the field is shared or idle)."""
        field = self.math.buffers.index(name)
        return self.pools[field], self.table[len(self.pools) + field, index : index + 1]

    def read(self, index: int, name: str) -> torch.Tensor:
        pool, slot = self.page(index, name)
        return pool.index_select(0, slot)[0]

    def values(self, index: int) -> dict[str, torch.Tensor]:
        """Gather one request without combining its source GEMMs with other requests."""
        return {name: self.read(index, name) for name in self.math.buffers}

    def arguments(
        self, values: Tensors, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> Tensors:
        """Decode rows never end a prompt."""
        return self.math.bind({**values, **self.invariants}, hidden, desired, embedding, base, self.prompt_end)

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
            if not self.batch_invariant:
                self.dense_readout = dense_readout(chunk.step, specs, len(columns))

    def store(self, index: int, values: Tensors, active: torch.Tensor) -> None:
        """Write only issued destinations, leaving shared carries and padding untouched."""
        width = len(self.pools)
        store_fields(
            self.math.buffers,
            self.pools,
            {name: value.unsqueeze(0) for name, value in values.items()},
            self.source,
            self.table[width : 2 * width, index : index + 1].masked_fill(~active, -1),
        )

    def append(self, field: int, lanes: slice, rows: torch.Tensor, lengths: torch.Tensor) -> None:
        """Copy only the live prefix, append arriving rows and clear stale rows from recycled slots."""
        width = len(self.pools)
        slot = self.tail_fields.index(field)
        append_tail(
            self.pools[field],
            rows,
            self.tail_bounds[field],
            self.table[field, lanes],
            self.table[width + field, lanes],
            lengths,
            self.table[2 * width + slot, lanes],
        )

    def inherit_carry(self, field: int, lanes: slice, complete: torch.Tensor) -> None:
        """A device-counted lane may own a fresh carry even when it finishes no chunk."""
        src, dst = self.table[field, lanes], self.table[len(self.pools) + field, lanes]
        inherit = ~complete & (dst >= 0) & (src != dst)
        with device_branch(inherit.any()):
            copy_pages(self.pools[field], self.pools[field], src, dst.masked_fill(~inherit, -1))

    def tail(self, index: int, name: str) -> torch.Tensor:
        pool, slot = self.write_page(index, f"_pending_rows_{name}")
        return pool.index_select(0, slot.clamp_min(0))[0]

    def decode_batch(
        self, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[UpdateOutput, ...]]:
        """Stream paged dense readouts and append tails across all decode lanes at once."""
        chunk, readout_spec = self.math.chunk, self.dense_readout
        assert isinstance(chunk, ChunkUpdate) and readout_spec is not None
        carry, *columns = chunk.arguments
        width, count = len(self.pools), len(hidden)
        counts = self.table[-1, :count]
        idle = counts == 0
        bound = self.arguments({}, hidden, desired, embedding[:count], base)
        carry_field = self.math.buffers.index(self.math.arguments[carry])
        output = paged_dense_mv(
            bound[columns[readout_spec.column]],
            self.pools[carry_field],
            self.table[carry_field, :count],
            counts,
            readout_spec.cast_input,
            readout_spec.scale,
        )
        len_field = self.math.buffers.index("_pending_length")
        lengths = self.pools[len_field].index_select(0, self.table[len_field, :count]).masked_fill(idle, 0)
        sources = self.sources[:count]
        for name in columns:
            tail_field = self.math.buffers.index(f"_pending_rows_{name}")
            self.append(tail_field, slice(0, count), bound[name], lengths)
        valid = lengths + counts
        complete, remaining = (valid == chunk.chunk) & ~idle, valid % chunk.chunk
        self.inherit_carry(carry_field, slice(0, count), complete)
        copy_pages(
            remaining, self.pools[len_field], sources, self.table[width + len_field, :count].masked_fill(idle, -1)
        )
        updates = complete.to(torch.int64)
        trained = updates * chunk.chunk
        results = [
            UpdateOutput(output[i : i + 1], complete[i], updates[i], trained[i], {"tail_rows": remaining[i]})
            for i in range(count)
        ]
        for index, result in enumerate(results):
            self.complete_row(index, result.changed, tuple(bound[key] for key in chunk.invariants))
        return output, tuple(results)

    def run(
        self, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> torch.Tensor:
        """Read incoming carries, append arriving rows, then update the device-selected completion bucket."""
        if isinstance(self.math.chunk, ChunkUpdate) and self.plan is None:
            samples = {name: pool[0] for name, pool in zip(self.math.buffers, self.pools, strict=True)}
            arguments = self.arguments(samples, hidden[:1], desired[:1], embedding[:1], base)
            self.prepare_scan(arguments)
        local = RequestKeyField.REQUEST_ID in self.key_fields
        if local and self.dense_readout is not None:
            output, results = self.decode_batch(hidden, desired, embedding, base)
            self.outputs[len(hidden)] = results
            return output
        if self.read_full is None:
            results = tuple(self.counted(index, hidden, desired, embedding, base) for index in range(len(hidden)))
        else:
            results = tuple(self.decode_row(index, hidden, desired, embedding, base) for index in range(len(hidden)))
        self.outputs[len(hidden)] = results
        return torch.cat(tuple(result.output for result in results))

    @torch.inference_mode(False)
    def counted(
        self, index: int, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> UpdateOutput:
        """Gather ordinary tensors before autograd enables gradient recording inside the projection."""
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

    def complete_row(self, index: int, complete: torch.Tensor, invariants: tuple[torch.Tensor, ...]) -> None:
        """Gate the gathers and scan together; cold execution still needs masked stores."""
        chunk = self.math.chunk
        assert isinstance(chunk, ChunkUpdate) and self.plan is not None
        carry, *columns = chunk.arguments
        name = self.math.arguments[carry]
        with device_branch(complete):
            tails = tuple(self.tail(index, column) for column in columns)
            updated, _ = execute_scan(self.plan, self.read(index, name), tails, invariants, None)
            pool, slot = self.write_page(index, name)
            copy_pages(updated.unsqueeze(0), pool, self.source, slot.masked_fill(~complete, -1))
            for field in self.tail_fields:
                slot = self.table[len(self.pools) + field, index : index + 1].masked_fill(~complete, -1)
                clear_tail(self.pools[field], self.tail_bounds[field], slot)

    def decode_row(
        self, index: int, hidden: torch.Tensor, desired: torch.Tensor, embedding: torch.Tensor, base: torch.Tensor
    ) -> UpdateOutput:
        """Append a device-indexed row and read its chunk's incoming carry at the permitted GEMM shape."""
        chunk = self.math.chunk
        assert isinstance(chunk, ChunkUpdate) and self.read_full is not None
        carry, *columns = chunk.arguments
        count = self.table[-1, index]
        idle = count == 0
        state = self.read(index, self.math.arguments[carry])
        length = self.read(index, "_pending_length").masked_fill(idle, 0)
        bound = self.arguments(
            {}, hidden[index : index + 1], desired[index : index + 1], embedding[index : index + 1], base
        )
        rows = tuple(bound[name] for name in columns)
        invariants = tuple(bound[name] for name in chunk.invariants)
        for name, row in zip(columns, rows, strict=True):
            field = self.math.buffers.index(f"_pending_rows_{name}")
            self.append(field, slice(index, index + 1), row, length.reshape(1))
        if self.read_row is None:
            tails = tuple(self.tail(index, name) for name in columns)
            output = self.read_full(state, *tails, *invariants).index_select(0, length.reshape(1))
        else:
            output = self.read_row(state, *rows, *invariants)
        valid = length + count
        complete, remaining = (valid == chunk.chunk) & ~idle, valid % chunk.chunk
        pool, slot = self.write_page(index, "_pending_length")
        copy_pages(remaining.reshape(1), pool, self.source, slot.masked_fill(idle, -1))
        self.inherit_carry(self.math.buffers.index(self.math.arguments[carry]), slice(index, index + 1), complete)
        self.complete_row(index, complete, invariants)
        updates = complete.to(torch.int64)
        return UpdateOutput(output, complete, updates, updates * chunk.chunk, {"tail_rows": remaining})

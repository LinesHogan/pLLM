"""Bounded overlay tiles over PTTS-issued slots; the task retains all page ownership."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.parameter_overlay import CapturedParameterOperand
from pllm.contracts.semantic_graph import TaskResourceFact
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.step_capture import StepCapture
from pllm.execution.tensor_frame import Tensors, TensorUpdate
from pllm.execution.tensor_program import TensorBatch
from pllm.execution.torch_private import measure
from pllm.kernels.page_copy import copy_pages

if TYPE_CHECKING:
    from pllm.execution.overlay_task import OverlayTransition


class OverlayBatch:
    """At most eight captured shapes per target: four row capacities and two phases."""

    states = 8

    def __init__(self, math: OverlayTransition, entitlement: TaskResourceFact | None = None) -> None:
        self.math, self.entitlement = math, entitlement
        self.tiles: dict[tuple[int, bool, int], OverlayTile | None] = {}
        self.batchable = False

    def resident(self, capture_bytes: int) -> int:
        """Charge the existing task budget; limit overlay residency to 512 MiB per task."""
        return min(capture_bytes, 512 << 20) if self.entitlement is not None and self.batchable else 0

    def capacities(self) -> tuple[int, ...]:
        assert self.math.chunk is not None
        chunk = self.math.chunk.chunk
        return tuple(dict.fromkeys((1, chunk, 4 * chunk, 8 * chunk)))

    def run(
        self,
        operands: tuple[CapturedParameterOperand, ...],
        target: int,
        indices: tuple[int, ...],
        batch: TensorBatch,
    ) -> tuple[tuple[TensorUpdate, ...], torch.Tensor]:
        """Bucket actual rows, then tile states; oversized rows stay eager at native size."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i, operand in enumerate(operands):
            rows = len(operand.hidden)
            groups[next((size for size in self.capacities() if rows <= size), rows)].append(i)
        results: dict[int, TensorUpdate] = {}
        counters: list[torch.Tensor] = []
        width = len(self.math.buffers)
        fields = tuple(batch.field_slots(target * width + i) for i in range(width))
        pools = tuple(pool for pool, _ in fields)
        slots = np.array(tuple(slots for _, slots in fields))
        train = operands[0].phase in self.math.phases
        for rows, members in groups.items():
            for start in range(0, len(members), self.states):
                selected = members[start : start + self.states]
                tile = self.tile(target, train, rows, pools, operands[0], len(selected))
                items = tuple(operands[i] for i in selected)
                if tile.plan is not None and tile.plan.analysis is not None:
                    values, counts = self.math.serial(items, target, tuple(indices[i] for i in selected), batch)
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
        operand: CapturedParameterOperand,
        count: int,
    ) -> OverlayTile:
        key = target, train, rows
        bounded = self.entitlement is not None and self.entitlement.resident_bytes > 0 and rows in self.capacities()
        capture = bounded and pools[0].is_cuda
        tile = self.tiles.get(key)
        if tile is not None:
            if tile.capture is not None and self.entitlement is not None:
                tile.capture.entitlement = replace(self.entitlement, resident_bytes=self.available(key))
            return tile
        tile = OverlayTile(self.math, pools, operand, self.states if bounded else count, rows, train)
        self.batchable |= tile.plan is None or tile.plan.analysis is None
        if capture and key not in self.tiles and self.entitlement is not None:
            remaining = self.available(key)
            if tile.plan is None or tile.plan.analysis is None:
                with FakeTensorMode(allow_non_fake_inputs=True):
                    workspace = measure(tile.body).workspace
                # Graph allocator segments and the BLAS handle round beyond tensor storage.
                needed = workspace + tile.buffer_bytes + (4 << 20)
                if needed <= remaining:
                    tile.capture = StepCapture(replace(self.entitlement, resident_bytes=remaining), self.states)
                    self.tiles[key] = tile
                else:
                    self.tiles[key] = None
        return tile

    def available(self, key: tuple[int, bool, int]) -> int:
        """An invalidated shape may reuse its budget, but never another live graph's bytes."""
        assert self.entitlement is not None
        return self.entitlement.resident_bytes - sum(
            tile.capture.resident_bytes
            for shape, tile in self.tiles.items()
            if shape != key and tile is not None and tile.capture is not None
        )


class OverlayTile:
    """Fixed mathematical inputs and capture state, with no request or page ownership."""

    def __init__(
        self,
        math: OverlayTransition,
        pools: tuple[torch.Tensor, ...],
        operand: CapturedParameterOperand,
        states: int,
        rows: int,
        train: bool,
    ) -> None:
        self.math, self.pools, self.train = math, pools, train
        self.table = torch.zeros((2 * len(pools) + 1, states), device=pools[0].device, dtype=torch.int64)
        self.hidden = operand.hidden.new_zeros((states, rows, operand.hidden.shape[1]))
        self.desired = operand.desired.new_zeros((states, rows, operand.desired.shape[1]))
        self.sources = torch.arange(states, device=pools[0].device)
        self.staging = PinnedStaging()
        self.capture: StepCapture | None = None
        self.output: tuple[torch.Tensor, ...] = ()
        chunk = math.chunk
        assert chunk is not None
        values = dict(zip(math.buffers, (pool[0] for pool in pools), strict=True))
        self.plan = chunk.prepare(self.arguments(values, self.hidden[0], self.desired[0])) if train else None

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        return self.hidden, self.desired, self.sources

    @property
    def buffer_bytes(self) -> int:
        return sum(value.untyped_storage().nbytes() for value in (self.table, *self.buffers))

    def arguments(self, values: Tensors, hidden: torch.Tensor, desired: torch.Tensor) -> Tensors:
        inputs = {**values, "input": hidden, "target": desired}
        return {key: inputs[source] for key, source in self.math.arguments.items()}

    def one(self, values: Tensors, hidden: torch.Tensor, desired: torch.Tensor, count: torch.Tensor):
        chunk = self.math.chunk
        assert chunk is not None
        if self.train:
            assert self.plan is not None
            result = chunk.counted(self.plan, count, self.arguments(values, hidden, desired))
        else:
            result = TensorUpdate.unchanged(self.math.apply(hidden, values))
        return values, result.output, result.changed, result.update_count, result.trained_rows

    def body(self) -> None:
        """Gather, map the authored step, and write only the table's issued destinations."""
        width = len(self.pools)
        values = {
            name: pool.index_select(0, self.table[i])
            for i, (name, pool) in enumerate(zip(self.math.buffers, self.pools, strict=True))
        }
        if self.train:
            values["_pending_length"].masked_fill_(self.table[-1] == 0, 0)
        fields, output, changed, updates, trained = torch.vmap(self.one)(
            values, self.hidden, self.desired, self.table[-1]
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
        operands: tuple[CapturedParameterOperand, ...],
        slots: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
        pools: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream | None,
    ) -> tuple[tuple[TensorUpdate, ...], torch.Tensor]:
        """Fence staging through PinnedStaging and detach outputs before the next tile replay."""
        self.pools = pools
        if self.capture is not None:
            self.capture.order(stream)
        host = (
            self.staging.take(self.table.numel()).view(self.table.shape)
            if self.table.is_cuda
            else torch.zeros_like(self.table, device="cpu")
        )
        width, count = len(pools), len(operands)
        host.numpy()[:width, :count] = slots
        host[width : 2 * width].fill_(-1)
        if self.train:
            host.numpy()[width : 2 * width, :count] = slots
        host.numpy()[-1, :count] = tuple(len(o.hidden) for o in operands)
        refill = partial(self.staging.copy_to, self.table) if self.table.is_cuda else partial(self.table.copy_, host)
        refill()
        self.hidden.zero_()
        self.desired.zero_()
        for i, operand in enumerate(operands):
            self.hidden[i, : len(operand.hidden)].copy_(operand.hidden)
            self.desired[i, : len(operand.hidden)].copy_(operand.desired)
        if self.capture is None:
            self.body()
        else:
            self.capture.run(
                self.body,
                self.table,
                self.table[width : 2 * width],
                tuple(pool.data_ptr() for pool in pools),
                stream,
                partial(self.staging.copy_to, self.table),
                self.buffers,
            )
        output, changed, updates, trained, lengths, counters = (value.clone() for value in self.output)
        return tuple(
            self.math.validate(
                TensorUpdate(
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

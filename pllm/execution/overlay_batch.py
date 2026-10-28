"""Bounded overlay tiles over PTTS-issued slots; the task retains all page ownership."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from pllm.contracts.parameter_overlay import CapturedParameterOperand
from pllm.execution.cuda_stream import upload
from pllm.execution.scan_execution import PreparedScan
from pllm.execution.tensor_frame import Tensors, TensorUpdate
from pllm.execution.tensor_program import TensorBatch
from pllm.kernels.page_copy import copy_pages

if TYPE_CHECKING:
    from pllm.execution.overlay_task import OverlayTransition


class OverlayBatch:
    """Execute at most eight states per tile, independent of admission and prompt limits."""

    states = 8

    def __init__(self, math: OverlayTransition) -> None:
        self.math = math

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
    ) -> tuple[TensorUpdate, ...]:
        """Bucket actual rows, then tile states; oversized rows stay eager at native size."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i, operand in enumerate(operands):
            rows = len(operand.hidden)
            groups[next((size for size in self.capacities() if rows <= size), rows)].append(i)
        results: dict[int, TensorUpdate] = {}
        for rows, members in groups.items():
            for start in range(0, len(members), self.states):
                tile = members[start : start + self.states]
                selected = tuple(operands[i] for i in tile)
                values = self.tile(selected, target, tuple(indices[i] for i in tile), batch, rows)
                results.update(zip(tile, values, strict=True))
        return tuple(results[i] for i in range(len(operands)))

    def tile(
        self,
        operands: tuple[CapturedParameterOperand, ...],
        target: int,
        indices: tuple[int, ...],
        batch: TensorBatch,
        rows: int,
    ) -> tuple[TensorUpdate, ...]:
        width = len(self.math.buffers)
        fields = tuple(batch.field_slots(target * width + i) for i in range(width))
        pools = tuple(pool for pool, _ in fields)
        slots = upload(tuple(tuple(slots[i] for i in indices) for _, slots in fields), pools[0].device)
        fields = tuple(pool.index_select(0, row) for pool, row in zip(pools, slots, strict=True))
        values = dict(zip(self.math.buffers, fields, strict=True))
        hidden = torch.stack(
            tuple(torch.nn.functional.pad(o.hidden, (0, 0, 0, rows - len(o.hidden))) for o in operands)
        )
        desired = torch.stack(
            tuple(torch.nn.functional.pad(o.desired, (0, 0, 0, rows - len(o.hidden))) for o in operands)
        )
        counts = upload(tuple(len(o.hidden) for o in operands), pools[0].device)
        train = operands[0].phase in self.math.phases
        chunk = self.math.chunk
        assert chunk is not None
        arguments = self.arguments(values, hidden, desired)
        plan = chunk.prepare({name: value[0] for name, value in arguments.items()}) if train else None
        if plan is not None and plan.analysis is not None:
            return self.math.serial(operands, target, indices, batch)
        result = torch.vmap(self.body(plan, train))(values, hidden, desired, counts)
        fields, output, changed, updates, trained = result
        if train:
            sources = torch.arange(len(operands), device=pools[0].device)
            for name, pool, destinations in zip(self.math.buffers, pools, slots, strict=True):
                copy_pages(fields[name], pool, sources, destinations)
        return tuple(
            TensorUpdate(
                output[i, : len(o.hidden)],
                changed[i],
                updates[i],
                trained[i],
                {"tail_rows": fields["_pending_length"][i]},
            )
            for i, o in enumerate(operands)
        )

    def arguments(self, values: Tensors, hidden: torch.Tensor, desired: torch.Tensor) -> Tensors:
        inputs = {**values, "input": hidden, "target": desired}
        return {key: inputs[source] for key, source in self.math.arguments.items()}

    def body(self, plan: PreparedScan | None, train: bool):
        """Map only the prepared mathematical body; no storage inspection occurs inside vmap."""

        def one(values: Tensors, hidden: torch.Tensor, desired: torch.Tensor, count: torch.Tensor):
            chunk = self.math.chunk
            assert chunk is not None
            if train:
                assert plan is not None
                result = chunk.counted(plan, count, self.arguments(values, hidden, desired))
            else:
                result = TensorUpdate.unchanged(self.math.apply(hidden, values))
            return values, result.output, result.changed, result.update_count, result.trained_rows

        return one

"""Device-indexed chunk schedules for ModelSites' request memories.

PTTS owns the storage and its lifetimes; this module uploads issued write slots and
maps the ordinary counted chunk scan over bounded, compacted request buckets.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx

from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_graph import device_branch
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.model_sites import ResidualRewrite
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.scan_execution import PreparedScan
from pllm.execution.state_execution import StateTaskContext
from pllm.execution.torch_private import compile_math, compiling_method
from pllm.kernels.page_copy import copy_pages

_FIELDS = ("carry", "_pending_rows_x", "_pending_rows_y", "_pending_length")


def _request_spans(rows: PackedPhaseRowBatch) -> tuple[tuple[int, int], ...]:
    if not rows.total_rows:
        return ()
    identities = rows.identity_indices
    starts = np.append(0, np.flatnonzero(identities[1:] != identities[:-1]) + 1)
    stops = np.append(starts[1:], rows.total_rows)
    first: list[int] = starts.tolist()
    last: list[int] = stops.tolist()
    return tuple(zip(first, last, strict=True))


class SiteMemorySchedule:
    """Stable PTTS pools and one uploaded table of request spans and private write slots."""

    def __init__(self, rewrite: ResidualRewrite, arena: PageBundleArena, requests: int, rows: int) -> None:
        memory = rewrite.memory
        assert memory is not None
        self.memory = memory
        self.name = rewrite.name
        self.numerics = arena.plan.timeline.tensor_policy.allow_numerical_transforms
        self.update = memory.update(rewrite, rewrite.places[0])
        self.views = (memory.carry.view, *memory.pending)
        self.pools = tuple(
            view.read(arena.field_pool(memory.carry.resource, view.field), pooled=True) for view in self.views
        )
        self.table = torch.zeros((2 + len(self.views), requests), dtype=torch.long, device=self.pools[0].device)
        self.rows = torch.zeros(rows, dtype=torch.long, device=self.pools[0].device)
        self.staging = PinnedStaging()
        self.row_staging = PinnedStaging()
        self._counts: tuple[int, ...] | None = None
        self._buckets: dict[tuple[int, int], Callable[..., tuple[dict[str, Tensor], Tensor]]] = {}

    def prepare(self, rows: PackedPhaseRowBatch, context: StateTaskContext) -> None:
        """Upload host scheduling facts and the exact slots authorized by the pending PTTS write."""
        resource = self.memory.carry.resource
        write = context.pending_write(resource)
        assert write is not None
        selection = context.selection.for_resource(resource)
        indices = dict(zip((row.identity for row in context.selection.rows), selection.state_indices, strict=True))
        spans = _request_spans(rows)
        packed: dict[StateRef, list[int]] = {}
        for start, end in spans:
            ref = selection.refs[int(indices[rows.identity(start)])]
            packed.setdefault(ref, []).extend(range(start, end))
        positions = write.before_refs.positions(StateRefs.of(tuple(packed)))
        _, slots = context.arena.slot_matrix(write, write.after_refs)
        self._counts = tuple(map(len, packed.values()))
        with self.staging.fill(self.table) as host:
            offset = 0
            for index, group in enumerate(packed.values()):
                host[0, index], host[1, index] = offset, len(group)
                host[2:, index] = slots[positions[index], [view.field for view in self.views]]
                offset += len(group)
        with self.row_staging.fill(self.rows) as host:
            host[: sum(self._counts)] = tuple(i for group in packed.values() for i in group)

    def __call__(self, stream: Tensor, branch: Tensor, invariants: tuple[Tensor, ...]) -> Tensor:
        # Buckets overwrite every live row; padded model lanes have no consumer.
        written = torch.empty_like(stream)
        table = self.table[:, : min(len(stream), self.table.shape[1])]
        widths = (*tuple(2**p for p in range((len(stream) - 1).bit_length())), len(stream))
        self.buckets(table, (1,), 0, stream, branch, written, invariants)
        if len(stream) > 1:
            # One shared conditional keeps library scratch bounded across capture sizes.
            with device_branch((table[1] > 1).any()):
                self.buckets(table, widths[1:], 1, stream, branch, written, invariants)
        return written

    def buckets(
        self,
        table: Tensor,
        widths: tuple[int, ...],
        lower: int,
        stream: Tensor,
        branch: Tensor,
        written: Tensor,
        invariants: tuple[Tensor, ...],
    ) -> None:
        """Compact bounded request spans; one-row decode skips all wider buckets."""
        lengths = table[1]
        capturing = stream.is_cuda and torch.cuda.is_current_stream_capturing()
        # A bucket of width w has at most capacity / (previous width + 1)
        # requests. Thus padding stays bounded even for a large prefill graph.
        for width in widths:
            if not capturing and self._counts is not None and not any(lower < count <= width for count in self._counts):
                lower = width
                continue
            active = (lengths > lower) & (lengths <= width)
            order = torch.argsort(~active, stable=True)[: min(len(lengths), len(stream) // (lower + 1))]
            self.bucket(
                table.index_select(1, order),
                active.index_select(0, order),
                width,
                stream,
                branch,
                written,
                invariants,
            )
            lower = width

    def one(
        self, values: dict[str, Tensor], x: Tensor, y: Tensor, count: Tensor, invariants: tuple[Tensor, ...]
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Run the shared counted scan without changing the authored chunk's shapes."""
        values.update({name: value for name, value in zip(self.update.invariants, invariants, strict=True)})
        values.update(x=x, y=y)
        result = self.update.counted(PreparedScan(self.update.step, self.update.chunk), count, values)
        return {name: values[name] for name in _FIELDS}, result.output

    def compute(
        self,
        fields: dict[str, Tensor],
        x: Tensor,
        y: Tensor,
        count: Tensor,
        invariants: tuple[Tensor, ...],
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Fuse the mapped chunk math with explicit tensor inputs before model capture."""
        mapped = torch.vmap(self.one, in_dims=(0, 0, 0, 0, None))
        if not x.is_cuda or not self.numerics:
            return mapped(fields, x, y, count, invariants)
        key = len(x), x.shape[1]
        if key not in self._buckets:
            with compiling_method("model-site", self.name):
                body = make_fx(
                    lambda *args: mapped(*args),
                    tracing_mode="fake",
                    _allow_non_fake_inputs=True,
                )(fields, x, y, count, invariants)
                # FX module error formatting walks a traceback generator under Dynamo and masks runtime OOMs.
                function = compile_math(body.forward)
                result = function(fields, x, y, count, invariants)
                self._buckets[key] = function
                return result
        return self._buckets[key](fields, x, y, count, invariants)

    def bucket(
        self,
        table: Tensor,
        active: Tensor,
        width: int,
        stream: Tensor,
        branch: Tensor,
        written: Tensor,
        invariants: tuple[Tensor, ...],
    ) -> None:
        """Gather compacted requests, scan padded rows, and scatter only active outputs and writes."""
        fields = {
            name: pool.index_select(0, slots) for name, pool, slots in zip(_FIELDS, self.pools, table[2:], strict=True)
        }
        fields["_pending_length"].masked_fill_(~active, 0)
        positions = table[0, :, None] + torch.arange(width, device=stream.device)
        count = torch.where(active, table[1], 0)
        valid = torch.arange(width, device=stream.device)[None] < count[:, None]
        indices = self.rows.index_select(0, positions.clamp_max(len(stream) - 1).flatten())
        x, y = (value.index_select(0, indices).view(len(active), width, -1) for value in (stream, branch))
        fields, output = self.compute(fields, x, y, count, invariants)
        sources = torch.arange(len(active), device=stream.device)
        for name, pool, slots in zip(_FIELDS, self.pools, table[2:], strict=True):
            copy_pages(fields[name], pool, sources, slots.masked_fill(~active, -1))
        copy_pages(
            output.flatten(0, 1),
            written,
            torch.arange(indices.numel(), device=stream.device),
            indices.masked_fill(~valid.flatten(), -1),
        )

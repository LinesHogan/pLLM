"""Bind causal logical accesses to borrowed paged KV tensors before any KV write.

Used by the tokenwise replay backend. This is an
invocation's logical access record, not a page owner or a runtime scheduler.
The backend must keep its cache and page tables stable until the invocation
completes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity


@dataclass(frozen=True, slots=True)
class PagedLayer:
    name: str
    cache: torch.Tensor
    slots: torch.Tensor


@dataclass(frozen=True, slots=True)
class PagedAccessPlan:
    """One invocation's logical KV intervals and the device write slot of every packed row.

    Page tables and slot mappings come from vLLM's own allocator and stay on the device;
    like the attention kernels, the plan trusts their contents and checks only the rows.
    """

    layers: Mapping[str, PagedLayer]
    ranges: Mapping[tuple[str, int], tuple[int, int]]

    def cell(self, layer: str, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor]:
        """Device block and offset indices of the row's KV write, shaped (1,) to avoid host reads."""
        paged = self.layers[layer]
        slot = paged.slots[row.packed_row_index : row.packed_row_index + 1].long()
        return slot // paged.cache.shape[2], slot % paged.cache.shape[2]


def bind_paged_access(rows: tuple[PackedPhaseRow, ...], layers: tuple[PagedLayer, ...]) -> PagedAccessPlan:
    """Check that each request's rows are consecutive positions and record their intervals.

    A layer reads [0, token) and replaces token; its read of the newly written
    current KV happens within that same layer call.
    """
    identities: dict[int, RequestIdentity] = {}
    ordinals: dict[RequestIdentity, int] = {}
    limits: dict[int, int] = {}
    row_ids: set[int] = set()
    for row in rows:
        if (
            row.packed_row_index in row_ids
            or identities.setdefault(row.request_ordinal, row.identity) != row.identity
            or ordinals.setdefault(row.identity, row.request_ordinal) != row.request_ordinal
            or row.request_ordinal in limits
            and row.token_offset != limits[row.request_ordinal]
        ):
            raise ValueError("KV rows require unique packed identities and consecutive request positions")
        row_ids.add(row.packed_row_index)
        limits[row.request_ordinal] = row.token_offset + 1
    if any(not layer.name for layer in layers) or len({layer.name for layer in layers}) != len(layers):
        raise ValueError("paged layers require distinct logical names")
    if any(layer.cache.ndim != 4 or layer.slots.ndim != 1 for layer in layers):
        raise ValueError("paged access requires [blocks, heads, block, content] KV and a packed slot vector")
    intervals: dict[int, tuple[int, int]] = {}
    for row in rows:
        first, end = intervals.get(row.request_ordinal, (row.token_offset, row.token_offset + 1))
        intervals[row.request_ordinal] = (min(first, row.token_offset), max(end, row.token_offset + 1))
    return PagedAccessPlan(
        MappingProxyType({layer.name: layer for layer in layers}),
        MappingProxyType(
            {(layer.name, request): interval for layer in layers for request, interval in intervals.items()}
        ),
    )

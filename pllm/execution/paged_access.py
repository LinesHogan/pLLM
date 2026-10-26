"""Bind causal logical accesses to borrowed paged KV tensors before any KV write.

Used by the tokenwise replay backend and region resources. This is an
invocation's logical access record, not a page owner or a runtime scheduler.
The backend must keep its cache and page tables stable until the invocation
completes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_program import AccessMode, BorrowedResource
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


@dataclass(frozen=True, slots=True)
class PagedResourceView:
    """A checked logical projection of the original paged access plan."""

    resource: BorrowedResource
    plan: PagedAccessPlan
    layers: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        domain = self.resource.domain
        if len(domain) != 3 or any(type(size) is not int or size < 1 for size in domain):
            raise ValueError("paged resource requires request, layer and position domains")
        ordinals = tuple(ordinal for ordinal, _ in self.layers)
        names = tuple(name for _, name in self.layers)
        if (
            len(set(ordinals)) != len(ordinals)
            or len(set(names)) != len(names)
            or set(names) != set(self.plan.layers)
            or any(type(ordinal) is not int or not 0 <= ordinal < domain[1] for ordinal in ordinals)
        ):
            raise ValueError("paged resource layer projection differs from its original plan")
        if any(
            not 0 <= request < domain[0] or not 0 <= first < end <= domain[2]
            for (_, request), (first, end) in self.plan.ranges.items()
        ):
            raise ValueError("paged resource rows exceed the declared logical domain")

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        if len(ranges) != 3 or type(mode) is not AccessMode:
            raise ValueError("paged resource access requires three indexed dimensions and a mode")
        if any(not 0 <= first <= end <= size for (first, end), size in zip(ranges, self.resource.domain, strict=True)):
            raise ValueError("paged resource access exceeds its declared domain")
        if any(first == end for first, end in ranges):
            return
        layers = dict(self.layers)
        requests, ordinals, positions = ranges
        for ordinal in range(*ordinals):
            if ordinal not in layers:
                raise ValueError("paged resource access names a layer outside its original proof")
            for request in range(*requests):
                interval = self.plan.ranges.get((layers[ordinal], request))
                if interval is None:
                    raise ValueError("paged resource access names a request outside its original proof")
                first, end = interval
                if not (0 if mode is AccessMode.READ else first) <= positions[0] <= positions[1] <= end:
                    raise ValueError("paged resource access exceeds its original read/write interval")


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

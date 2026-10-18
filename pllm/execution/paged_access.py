"""Bind causal logical accesses to borrowed paged tensors before any KV write.

This is an invocation's physical access evidence, not a page owner or a runtime
scheduler. The backend must keep its original cache and page tables stable until
the original invocation completes. Shared read-only prefixes are permitted.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from types import MappingProxyType

import torch

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_program import AccessMode, BorrowedResource
from pllm.contracts.request_identity import RequestIdentity


@dataclass(frozen=True, slots=True)
class PagedLayer:
    name: str
    cache: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor


@dataclass(frozen=True, slots=True)
class CausalAccess:
    """One logical interval and its exact strided byte projection for one head."""

    layer: str
    request: RequestIdentity
    head: int
    first: int
    count: int
    device: str
    address: int
    stride: int
    width: int

    @property
    def end(self) -> int:
        return self.address + (self.count - 1) * self.stride + self.width

    def conflicts(self, write: "CausalAccess") -> bool:
        # A current-token write is contiguous. Find which strided read cells it
        # intersects without expanding the history into individual token nodes.
        first = max(0, (write.address - self.width - self.address) // self.stride + 1)
        last = min(self.count - 1, (write.end - 1 - self.address) // self.stride)
        if first > last:
            return False
        return not (
            first == last
            and self.layer == write.layer
            and self.request == write.request
            and self.head == write.head
            and self.first + first == write.first
            and self.address + first * self.stride == write.address
            and self.width == write.width
        )


@dataclass(frozen=True, slots=True)
class PagedAccessPlan:
    reads: tuple[CausalAccess, ...]
    writes: tuple[CausalAccess, ...]
    _slots: Mapping[tuple[str, int], tuple[PackedPhaseRow, int, int]]
    layers: tuple[str, ...]
    ranges: Mapping[tuple[str, int], tuple[int, int]]

    def slot(self, layer: str, row: PackedPhaseRow) -> tuple[int, int]:
        original, block, offset = self._slots[(layer, row.packed_row_index)]
        if original != row:
            raise ValueError("KV access differs from the original packed row")
        return block, offset


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


def validate_paged_views(views: tuple[PagedResourceView, ...], protected: tuple[tuple[str, int, int], ...]) -> None:
    """Check separation between resource names and every original arena allocation."""
    if len(views) > 1:
        reads = tuple(
            replace(access, layer=str((view.resource.name, access.layer)))
            for view in views
            for access in view.plan.reads
        )
        writes = tuple(
            replace(access, layer=str((view.resource.name, access.layer)))
            for view in views
            for access in view.plan.writes
        )
        _validate_aliases(reads, writes)
    for view in views:
        for access in (*view.plan.reads, *view.plan.writes):
            if any(
                device == access.device and start < access.end and access.address < start + size
                for device, start, size in protected
            ):
                raise ValueError("borrowed paged access overlaps original arena storage")


def _validate_aliases(reads: tuple[CausalAccess, ...], writes: tuple[CausalAccess, ...]) -> None:
    # Sweep physical envelopes, then use exact strided intersection within the
    # overlapping pages. This handles HND/NHD and layers interleaved in one root.
    accesses = (*reads, *writes)
    devices = {access.device for access in accesses}
    for device in devices:
        ordered = sorted((access.address, index) for index, access in enumerate(accesses) if access.device == device)
        active: dict[int, CausalAccess] = {}
        active_writes: dict[int, CausalAccess] = {}
        ends: list[tuple[int, int]] = []
        for address, index in ordered:
            while ends and ends[0][0] <= address:
                _, expired = heappop(ends)
                active.pop(expired, None)
                active_writes.pop(expired, None)
            access = accesses[index]
            if index >= len(reads):
                if any(other.conflicts(access) for other in active.values()):
                    raise ValueError("KV write aliases a different logical layer, request, head or token")
                active_writes[index] = access
            elif any(access.conflicts(write) for write in active_writes.values()):
                raise ValueError("KV history aliases another logical token's write")
            active[index] = access
            heappush(ends, (access.end, index))


def bind_paged_access(rows: tuple[PackedPhaseRow, ...], layers: tuple[PagedLayer, ...]) -> PagedAccessPlan:
    """Verify page/slot agreement and writable-cell separation for this invocation.

    Cache payloads are borrowed. Only integer metadata is copied to the host,
    once per distinct table/view, and histories stay compressed as page spans.
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
    hosts: dict[tuple[str, int, tuple[int, ...], tuple[int, ...], torch.dtype], torch.Tensor] = {}

    def host(value: torch.Tensor) -> torch.Tensor:
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError("page tables and slots require integer metadata")
        key = (str(value.device), value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
        if key not in hosts:
            hosts[key] = value.detach().cpu()
        return hosts[key]

    reads: list[CausalAccess] = []
    writes: list[CausalAccess] = []
    slots: dict[tuple[str, int], tuple[PackedPhaseRow, int, int]] = {}
    for layer in layers:
        cache = layer.cache
        if cache.ndim != 4 or min(cache.shape) < 1 or cache.stride(3) != 1:
            raise ValueError("paged access requires packed contiguous KV content lanes")
        extent = 1
        for stride, size in sorted(zip(cache.stride(), cache.shape, strict=True)):
            if size > 1 and stride < extent:
                raise ValueError("paged KV layout has overlapping logical elements")
            extent += (size - 1) * stride
        if layer.block_table.ndim != 2 or layer.slots.ndim != 1:
            raise ValueError("paged access requires a request block table and packed slot vector")
        blocks, heads, block_size, content = cache.shape
        request_count = max(limits, default=-1) + 1
        page_count = max(((length + block_size - 1) // block_size for length in limits.values()), default=0)
        table = host(layer.block_table[:request_count, :page_count])
        mapping = host(layer.slots[: max(row_ids, default=-1) + 1])
        element = cache.element_size()
        base, device = cache.data_ptr(), str(cache.device)
        strides = cache.stride()

        def access(request: int, head: int, first: int, count: int, block: int, offset: int) -> CausalAccess:
            return CausalAccess(
                layer.name,
                identities[request],
                head,
                first,
                count,
                device,
                base + element * (block * strides[0] + head * strides[1] + offset * strides[2]),
                (strides[2] if block_size > 1 else content) * element,
                content * element,
            )

        pages: dict[int, tuple[int, ...]] = {}
        for request, length in limits.items():
            count = (length + block_size - 1) // block_size
            if request >= table.shape[0] or count > table.shape[1]:
                raise ValueError("KV history exceeds its original request page table")
            physical = tuple(int(value.item()) for value in table[request, :count])
            if any(not 0 <= block < blocks for block in physical):
                raise ValueError("KV page table names a block outside the original cache")
            if len(set(physical)) != len(physical):
                raise ValueError("one request aliases different logical KV pages")
            pages[request] = physical
            # Union of every causal read in this batch: the longest prefix.
            # Include the current cell as a same-identity read for backup/restore.
            for logical, block in enumerate(physical):
                first = logical * block_size
                count = min(block_size, length - first)
                reads.extend(access(request, head, first, count, block, 0) for head in range(heads))
        for row in rows:
            if row.packed_row_index >= mapping.shape[0]:
                raise ValueError("packed KV slot is outside the original mapping")
            physical = pages[row.request_ordinal]
            page, offset = divmod(row.token_offset, block_size)
            block = physical[page]
            if int(mapping[row.packed_row_index].item()) != block * block_size + offset:
                raise ValueError("KV write slot differs from its causal page-table position")
            slots[(layer.name, row.packed_row_index)] = (row, block, offset)
            writes.extend(
                access(row.request_ordinal, head, row.token_offset, 1, block, offset) for head in range(heads)
            )
    ranges: dict[tuple[str, int], tuple[int, int]] = {}
    for (layer_name, _), (row, _, _) in slots.items():
        key = (layer_name, row.request_ordinal)
        first, end = ranges.get(key, (row.token_offset, row.token_offset + 1))
        ranges[key] = (min(first, row.token_offset), max(end, row.token_offset + 1))
    result = PagedAccessPlan(
        tuple(reads),
        tuple(writes),
        MappingProxyType(slots),
        tuple(layer.name for layer in layers),
        MappingProxyType(ranges),
    )
    _validate_aliases(result.reads, result.writes)
    return result

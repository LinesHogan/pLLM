"""Read-only tensor families borrowed from the original invocation owner."""

from dataclasses import dataclass
from math import prod
from typing import Protocol

import torch

from pllm.contracts.region_program import AccessMode, BorrowedResource
from pllm.execution.paged_access import PagedResourceView


class MutableTensorView(Protocol):
    """A backend's physical separation proof for indexed mutable resources."""

    @property
    def resource(self) -> BorrowedResource: ...
    def validate(self, protected: tuple[tuple[str, int, int], ...]) -> tuple[tuple[str, int, int], ...]: ...
    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None: ...


@dataclass(frozen=True, slots=True)
class _TensorIdentity:
    tensor: torch.Tensor
    device: str
    address: int
    bytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int
    dtype: torch.dtype
    version: int | None

    @classmethod
    def bind(cls, tensor: torch.Tensor) -> "_TensorIdentity":
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise ValueError("read-only resource requires real strided tensor storage")
        storage = tensor.untyped_storage()
        return cls(
            tensor,
            str(tensor.device),
            storage.data_ptr(),
            storage.nbytes(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            int(tensor.storage_offset()),
            tensor.dtype,
            None if tensor.is_inference() else tensor._version,
        )

    def check(self) -> None:
        actual = _TensorIdentity.bind(self.tensor)
        if (self.device, self.address, self.bytes, self.shape, self.stride, self.offset, self.dtype, self.version) != (
            actual.device,
            actual.address,
            actual.bytes,
            actual.shape,
            actual.stride,
            actual.offset,
            actual.dtype,
            actual.version,
        ):
            raise ValueError("read-only resource changed after binding its original tensor")


@dataclass(frozen=True, slots=True)
class ReadonlyTensorCell:
    indices: tuple[int, ...]
    identities: tuple[_TensorIdentity, ...]

    @classmethod
    def bind(cls, indices: tuple[int, ...], tensors: tuple[torch.Tensor, ...]) -> "ReadonlyTensorCell":
        return cls(indices, tuple(_TensorIdentity.bind(tensor) for tensor in tensors))


@dataclass(frozen=True, slots=True)
class ReadonlyTensorView:
    """Bindings promise no writes; physical checks reject aliases with writable owners.

    Inference tensors have no version counter. Their original backend must keep
    contents stable through invocation completion, as it does for its model.
    No tensor is cloned and these references confer no publication authority.
    """

    resource: BorrowedResource
    cells: tuple[ReadonlyTensorCell, ...]

    def __post_init__(self) -> None:
        keys = tuple(cell.indices for cell in self.cells)
        domain = self.resource.domain
        if (
            len(set(keys)) != len(keys)
            or len(keys) != prod(domain)
            or any(
                len(key) != len(domain)
                or any(type(i) is not int or not 0 <= i < n for i, n in zip(key, domain, strict=True))
                for key in keys
            )
        ):
            raise ValueError("read-only tensor bindings must cover the declared resource family")

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        if mode is not AccessMode.READ:
            raise ValueError("read-only tensor resource cannot authorize a write")
        if len(ranges) != len(self.resource.domain) or any(
            not 0 <= lo <= hi <= size for (lo, hi), size in zip(ranges, self.resource.domain, strict=True)
        ):
            raise ValueError("read-only tensor access exceeds its declared domain")

    def validate(self) -> None:
        """Check physical identities once at the invocation binding boundary."""
        for cell in self.cells:
            for identity in cell.identities:
                identity.check()

    def tensor(self, indices: tuple[int, ...], field: int) -> torch.Tensor:
        self.check(AccessMode.READ, tuple((coordinate, coordinate + 1) for coordinate in indices))
        cell = next((cell for cell in self.cells if cell.indices == indices), None)
        if cell is None or type(field) is not int or not 0 <= field < len(cell.identities):
            raise ValueError("tensor borrow has no matching original resource field")
        return cell.identities[field].tensor.detach()

    @property
    def allocation_ranges(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            sorted(
                {
                    (identity.device, identity.address, identity.bytes)
                    for cell in self.cells
                    for identity in cell.identities
                    if identity.bytes
                }
            )
        )


def validate_readonly_views(
    views: tuple[ReadonlyTensorView, ...],
    paged: tuple[PagedResourceView, ...],
    protected: tuple[tuple[str, int, int], ...],
) -> None:
    for view in views:
        view.validate()
        for device, address, size in view.allocation_ranges:
            if any(
                device == other and address < start + count and start < address + size
                for other, start, count in protected
            ):
                raise ValueError("read-only resource overlaps original arena storage")
            if any(
                device == write.device and address < write.end and write.address < address + size
                for original in paged
                for write in original.plan.writes
            ):
                raise ValueError("read-only resource aliases writable paged storage")


@dataclass(frozen=True, slots=True)
class TensorResourceView:
    """Independent mutable tensor cells borrowed from a backend, without payload copies.

    Each cell conservatively covers its entire tensor. This permits reordering
    different layers while retaining all causal KV order within each layer.
    """

    resource: BorrowedResource
    tensors: tuple[torch.Tensor, ...]

    def validate(self, protected: tuple[tuple[str, int, int], ...]) -> tuple[tuple[str, int, int], ...]:
        if self.resource.domain != (len(self.tensors),):
            raise ValueError("tensor resources need one complete tensor per logical cell")
        spans = list(protected)
        for tensor in self.tensors:
            if tensor.layout != torch.strided or tensor.device.type == "meta":
                raise ValueError("mutable tensor borrowing requires real strided storage")
            extent = 1
            for stride, size in sorted(zip(tensor.stride(), tensor.shape, strict=True)):
                if size > 1 and stride < extent:
                    raise ValueError("mutable tensor cells must not overlap themselves")
                extent += (size - 1) * stride
            device, address, size = str(tensor.device), tensor.data_ptr(), extent * tensor.element_size()
            if any(device == d and address < a + n and a < address + size for d, a, n in spans):
                raise ValueError("mutable tensor cells overlap another borrowed cell or PTTS allocation")
            spans.append((device, address, size))
        return tuple(spans[len(protected) :])

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        assert len(ranges) == 1 and 0 <= ranges[0][0] <= ranges[0][1] <= len(self.tensors)


@dataclass(frozen=True, slots=True)
class PagedAllocationView:
    """Request/layer KV cells backed by the original allocator's CPU block grants.

    Prefix sharing is excluded: disjoint allocated blocks prove request
    separation without reading device page tables. The backend must use the
    same grants for its GPU table/slot producer and retain that producer's order.
    This descriptor allocates, retires and remaps no pages.
    """

    resource: BorrowedResource
    caches: tuple[torch.Tensor, ...]
    blocks: tuple[tuple[int, ...], ...]

    def validate(self, protected: tuple[tuple[str, int, int], ...]) -> tuple[tuple[str, int, int], ...]:
        if len(self.resource.domain) != 2 or self.resource.domain[1] != len(self.caches):
            raise ValueError("paged allocation proof needs request and layer domains")
        if len(self.blocks) > self.resource.domain[0]:
            raise ValueError("paged allocation grants exceed the declared request count")
        flat = tuple(block for row in self.blocks for block in row)
        if any(not row for row in self.blocks) or len(flat) != len(set(flat)):
            raise ValueError("closed model windows require distinct nonshared request blocks")
        if any(cache.ndim != 4 or any(not 0 <= block < cache.shape[0] for block in flat) for cache in self.caches):
            raise ValueError("original KV allocation grants exceed the cache block domain")
        storage = BorrowedResource(self.resource.name, (len(self.caches),))
        return TensorResourceView(storage, self.caches).validate(protected)

    def check(self, mode: AccessMode, ranges: tuple[tuple[int, int], ...]) -> None:
        if len(ranges) != 2 or not (
            0 <= ranges[0][0] <= ranges[0][1] <= len(self.blocks)
            and 0 <= ranges[1][0] <= ranges[1][1] <= len(self.caches)
        ):
            raise ValueError("KV access is outside the original request/layer allocation grants")

"""Read-only tensor families borrowed from the original invocation owner."""

from dataclasses import dataclass
from math import prod

import torch

from pllm.contracts.tensor_borrow import AccessMode, BorrowedResource


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


def validate_readonly_views(views: tuple[ReadonlyTensorView, ...], protected: tuple[tuple[str, int, int], ...]) -> None:
    for view in views:
        view.validate()
        for device, address, size in view.allocation_ranges:
            if any(
                device == other and address < start + count and start < address + size
                for other, start, count in protected
            ):
                raise ValueError("read-only resource overlaps original arena storage")

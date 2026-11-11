"""Pages received from another process and their invocation bindings; never a second state owner.

PageBundleArena exports pages with `PageFieldTransfer`; the consuming process maps them with
`receive_field`, groups them in `BorrowedStateRoot`, and state_feed/state_execution use those roots.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

import torch

from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_borrow import BorrowedResource
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.page_bundle_planning import page_stride_bytes
from pllm.execution.ptts.slot_pools import PageSlot
from pllm.execution.shared_tensor import CudaIpcStorage
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView, validate_readonly_views

if TYPE_CHECKING:
    from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin


@dataclass(frozen=True, slots=True)
class PageFieldTransfer:
    plan_digest: str
    owner_id: int  # Original PagePin.pin_id; all remote grants are read-only.
    ref: StateRef
    resource: str
    field_id: LayoutFieldId
    page: PageSlot
    storage_offset: int
    cuda_storage: CudaIpcStorage | None = None


@dataclass(slots=True)
class BorrowedPage:
    """Read-only native alias; discard all aliases before close (not OS-enforced)."""

    descriptor: PageFieldTransfer
    _tensor: torch.Tensor | None
    connection: Connection
    stream: torch.cuda.Stream | None = None

    @property
    def tensor(self) -> torch.Tensor:
        assert self._tensor is not None, "read after discard()"
        return self._tensor

    def discard(self) -> None:
        """Drop an import after consumer teardown, without fabricating a read ACK."""
        if self.stream is not None:
            self.stream.synchronize()
        self._tensor = None

    def close(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()
        self._tensor = None
        self.connection.send_bytes(encode_boundary(self.descriptor).encode())


def receive_field(
    plan: PlanBundle, plan_digest: str, connection: Connection, *, stream: torch.cuda.Stream | None = None
) -> BorrowedPage:
    """Map a producer grant only when it belongs to this plan."""
    descriptor = decode_boundary(connection.recv_bytes(65536).decode(), PageFieldTransfer)
    if descriptor.plan_digest != plan_digest:
        raise ValueError("the page producer was built for another compilation; restart it from this model's plan")
    ((entry, provider),) = (
        (entry, provider)
        for entry in plan.tensor_layout_request.layouts
        for provider in plan.states.plans
        if entry.resource == provider.resource == descriptor.resource
    )
    assert stream is not None
    spec = entry.layout.fields[descriptor.field_id]
    size = spec.dtype.itemsize
    stride = page_stride_bytes(spec) // size
    alignment = max(spec.alignment_bytes // size, 1)
    storage = descriptor.cuda_storage
    tensor = (
        storage.map(
            (provider.page_capacities[descriptor.field_id] * stride + alignment - 1) * size,
            spec.dtype.torch,
            descriptor.storage_offset,
            spec.shape,
            stream,
        )
        if storage is not None
        else torch.empty(spec.shape, dtype=spec.dtype.torch, device=stream.device)
    )
    return BorrowedPage(descriptor, tensor, connection, stream)


class BorrowedStateRoot:
    """Group receive_field results. The receiving scope owns partial-receive failures.

    This is a trusted-program binding, not authentication: receive_field validates
    each grant first. No view or derived alias may escape invocation completion.
    The local pin identifies the consuming invocation, never remote ownership.
    CPU unit fixtures exercise view binding only, never CPU page transport.
    """

    def __init__(self, arena: PageBundleArena, resource: str, ref: StateRef, pages: tuple[BorrowedPage, ...]) -> None:
        self.resource, self.ref, self.pages = resource, ref, pages
        (layout,) = (entry.layout for entry in arena.plan.tensor_layout_request.layouts if entry.resource == resource)
        if (
            not pages
            or len(pages) != len(layout.fields)
            or any(
                page.descriptor.plan_digest != arena.plan_digest
                or page.descriptor.resource != resource
                or page.descriptor.ref != ref
                or page.descriptor.field_id != field.field_id
                or page.descriptor.owner_id != pages[0].descriptor.owner_id
                or page.stream != pages[0].stream
                for page, field in zip(pages, layout.fields, strict=True)
            )
        ):
            raise ValueError(
                "a fed state must arrive as every field of one state version from one producer of this model's plan; "
                "restart the producer from this engine's plan"
            )
        self.stream = pages[0].stream
        self._view: ReadonlyTensorView | None = ReadonlyTensorView(
            BorrowedResource(resource, ()), (ReadonlyTensorCell.bind((), tuple(page.tensor for page in pages)),)
        )
        self._claimed = False
        self._pin: PagePin | None = None

    def claim(self, protected: tuple[tuple[str, int, int], ...]) -> None:
        assert not self._claimed and self._view is not None
        validate_readonly_views((self._view,), protected)
        self._claimed = True

    def bind(self, pin: PagePin) -> None:
        assert self._claimed and self._pin is None and pin.stream == self.stream
        self._pin = pin

    def tensors(self, pin: PagePin) -> tuple[torch.Tensor, ...]:
        assert self._view is not None and self._pin is pin
        for page in self.pages:
            page.tensor  # Recheck original receiving stream and the live BorrowedPage.
        return tuple(self._view.tensor((), index) for index in range(len(self.pages)))

    def check_output(self, tensor: torch.Tensor) -> None:
        assert self._view is not None
        storage = tensor.untyped_storage()
        validate_readonly_views((self._view,), ((str(tensor.device), storage.data_ptr(), storage.nbytes()),))

    def discard(self) -> None:
        """End all aliases during terminal consumer cleanup without acknowledging success."""
        self._view = None
        for page in self.pages:
            page.discard()

    def close(self) -> None:
        assert self._view is not None
        self._view = None
        for page in self.pages:
            page.close()


def borrowed_root(roots: Sequence[BorrowedStateRoot], resource: str) -> BorrowedStateRoot | None:
    return next((root for root in roots if root.resource == resource), None)

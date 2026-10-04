"""Invocation bindings for already received pages; never a second state owner."""

from collections.abc import Sequence

import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.region_program import BorrowedResource
from pllm.contracts.state_ref import StateRef
from pllm.execution.ptts.page_bundle import BorrowedPage, PagePin
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView, validate_readonly_views


class BorrowedStateRoot:
    """Group receive_field results. The receiving scope owns partial-receive failures.

    This is a trusted-program binding, not authentication: receive_field validates
    each grant first. No view or derived alias may escape invocation completion.
    The local pin identifies the consuming invocation, never remote ownership.
    CPU unit fixtures exercise view binding only, never CPU page transport.
    """

    def __init__(self, plan: PlanBundle, resource: str, ref: StateRef, pages: tuple[BorrowedPage, ...]) -> None:
        self.resource, self.ref, self.pages = resource, ref, pages
        self.signature = plan.schedule_decision.signature
        layout, = (entry.layout for entry in plan.tensor_layout_request.layouts if entry.resource == resource)
        if (not pages or len(pages) != len(layout.fields)
            or any(page.descriptor.compilation_signature != self.signature or page.descriptor.resource != resource
                or page.descriptor.ref != ref or page.descriptor.field_id != field.field_id
                or page.descriptor.owner_id != pages[0].descriptor.owner_id or page.stream != pages[0].stream
                for page, field in zip(pages, layout.fields, strict=True))):
            raise ValueError("borrowed root requires every field of one exact grant owner and version")
        self.stream = pages[0].stream
        self._view: ReadonlyTensorView | None = ReadonlyTensorView(BorrowedResource(resource, ()),
            (ReadonlyTensorCell.bind((), tuple(page.tensor for page in pages)),))
        self._claimed = False
        self._pin: PagePin | None = None

    def claim(self, protected: tuple[tuple[str, int, int], ...]) -> None:
        if self._claimed or self._view is None:
            raise RuntimeError("borrowed root is closed or already staged")
        validate_readonly_views((self._view,), (), protected)
        self._claimed = True

    def bind(self, pin: PagePin) -> None:
        if not self._claimed or self._pin is not None or pin.stream != self.stream:
            raise RuntimeError("borrowed root requires one invocation on its receiving stream")
        self._pin = pin

    def tensors(self, pin: PagePin) -> tuple[torch.Tensor, ...]:
        if self._view is None or self._pin is not pin:
            raise RuntimeError("borrowed root is closed or belongs to another invocation")
        for page in self.pages:
            page.tensor  # Recheck original receiving stream and the live BorrowedPage.
        return tuple(self._view.tensor((), index) for index in range(len(self.pages)))

    def check_output(self, tensor: torch.Tensor) -> None:
        if self._view is None:
            raise RuntimeError("borrowed root is closed")
        storage = tensor.untyped_storage()
        validate_readonly_views((self._view,), (), ((str(tensor.device), storage.data_ptr(), storage.nbytes()),))

    def close(self) -> None:
        if self._view is None:
            raise RuntimeError("borrowed root is already closed")
        self._view = None
        for page in self.pages:
            page.close()


def borrowed_root(roots: Sequence[BorrowedStateRoot], resource: str) -> BorrowedStateRoot | None:
    return next((root for root in roots if root.resource == resource), None)

"""CUDA-graph capture over signed pages: views, kernel bindings and entitlement checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Callable

import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import TaskResourceFact
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.ptts.page_bundle_planning import DTYPE_BYTES
from pllm.kernels.paged_linear import PagedRows

if TYPE_CHECKING:
    from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin, PageWrite


@dataclass(slots=True)
class CapturedPageFields:
    """Physical views over caller-owned cold scratch, not another version store."""

    arena: PageBundleArena
    resource: str
    fields: tuple[LayoutFieldId, ...]
    indices: torch.Tensor
    buffers: tuple[torch.Tensor, ...]
    addresses: tuple[int, ...]
    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph
    access: PagePin | PageWrite
    refs: tuple[StateRef, ...]
    results: tuple[torch.Tensor, ...] = ()
    inputs: tuple[torch.Tensor, ...] = ()
    paged: bool = False

    def resident_bytes(self) -> int:
        """Cold graph-pool reservation plus owned page scratch; borrowed inputs are separate.

        This is a resource fact, not a capacity entitlement. A shared graph pool
        must be charged once by its owner, not once per view using that pool.
        """
        from pllm.execution.codec import cuda_graph_pool_bytes

        return cuda_graph_pool_bytes(self.graph.pool(), self.stream.device.index or 0) + sum(
            value.untyped_storage().nbytes() for value in (self.indices, *self.buffers)
        )


@dataclass(frozen=True, slots=True)
class PagedKernel:
    compute: Callable[[CapturedPageFields, tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]

    @classmethod
    def rows(
        cls,
        row_input: int,
        compute: Callable[[tuple[PagedRows, ...], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
    ) -> PagedKernel:
        def run(view: CapturedPageFields, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            if not 0 <= row_input < len(inputs):
                raise ValueError("paged capture requires a row-index input")
            return compute(view.arena.read_paged_captured(view, inputs[row_input]), inputs)

        return cls(run)


@dataclass(slots=True)
class KernelCapture:
    """A native kernel binding; input factories accept meta devices for allocation-free sizing."""

    resource: str
    fields: tuple[LayoutFieldId, ...]
    count: int
    inputs: Callable[[torch.device], tuple[torch.Tensor, ...]]
    compute: Callable[[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] | PagedKernel
    entitlement: TaskResourceFact
    writable: bool = True
    _view: CapturedPageFields | None = None

    @property
    def view(self) -> CapturedPageFields:
        if self._view is None:
            raise RuntimeError("kernel has not been bound to its signed arena")
        return self._view

    def check_inputs(self, plan: PlanBundle, inputs: tuple[torch.Tensor, ...]) -> int:
        if self.entitlement not in plan.timeline.task_resources:
            raise ValueError("capture entitlement is absent from the original signed plan")
        # Factories own distinct input buffers; meta tensors have no usable storage address.
        input_bytes = sum(tensor.untyped_storage().nbytes() for tensor in inputs)
        specs = next(
            item.layout.fields for item in plan.tensor_layout_request.layouts if item.resource == self.resource
        )
        scratch_bytes = self.count * (
            8 * len(self.fields)
            + (
                0
                if isinstance(self.compute, PagedKernel)
                else sum(prod(specs[field].shape) * DTYPE_BYTES[specs[field].dtype] for field in self.fields)
            )
        )
        if input_bytes + scratch_bytes > self.entitlement.resident_bytes:
            raise ValueError("capture inputs and field scratch exceed the signed resident bytes")
        return input_bytes

    def bind(self, arena: PageBundleArena) -> None:
        if self._view is not None or arena.device.type != "cuda":
            raise ValueError("capture requires an unbound kernel and a CUDA arena")
        self.check_inputs(arena.plan, self.inputs(torch.device("meta")))
        with torch.inference_mode(False):
            inputs = self.inputs(arena.device)
        input_bytes = self.check_inputs(arena.plan, inputs)
        stream = torch.cuda.Stream(device=arena.device)
        stream.wait_stream(torch.cuda.current_stream(arena.device))
        cold = arena.prepare_admission((self.resource,) * self.count, stream)
        publication = pin = None
        try:
            if self.writable:
                access, refs = cold, cold.after_refs
            else:
                publication = arena.publish(cold)
                arena.settle(publication)
                pin = arena.pin_exact(publication.after_refs, stream)
                access, refs = pin, pin.refs
            view = arena.capture_kernel(
                access, refs, self.fields, inputs, self.compute, preserve_fields=not self.writable
            )
            required = view.resident_bytes() + input_bytes
            if required > self.entitlement.resident_bytes:
                raise ValueError(
                    f"capture requires {required} resident bytes, signed {self.entitlement.resident_bytes}"
                )
            self._view = view
        finally:
            if pin is not None:
                arena.wait_release(arena.release_pin(pin))
            if publication is None:
                arena.abort(cold)
            else:
                arena.retire(publication.after_refs)
            arena.collect_completed(wait=True)

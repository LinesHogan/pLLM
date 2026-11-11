"""Slot-addressed field storage shared by versioned PTTS state and transient hidden rows.

SlotPools keeps the field tensors and CUDA event objects; Rust owns holds, generations and reuse fences.
PageBundleArena and vLLM hidden-row capture share the same numeric pool implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod

import numpy as np
import numpy.typing as npt
import torch

from pllm import _core
from pllm.contracts.tensor_layout import TensorLayoutSpec
from pllm.execution.admission import WindowBackpressure
from pllm.execution.ptts.page_bundle_planning import page_stride_bytes
from pllm.kernels.page_copy import copy_pages


@dataclass(frozen=True, slots=True)
class PageSlot:
    slot_id: int
    generation: int


Slots = npt.NDArray[np.int64]


class PageEvents:
    """Keep CUDA objects alive for the numeric event IDs retained by a Rust ledger."""

    def __init__(self) -> None:
        self._events: dict[int, torch.cuda.Event] = {}

    def add(self, event: torch.cuda.Event | None) -> int:
        """Register a completion without querying or waiting on it."""
        if event is None:
            return 0
        key = id(event)
        self._events[key] = event
        return key

    def get(self, key: int) -> torch.cuda.Event | None:
        """Resolve a retained event for an explicit page snapshot."""
        return self._events[key] if key else None

    def completed(self) -> Slots:
        """Poll fences; queries never wait for device work."""
        return np.fromiter((key for key, event in self._events.items() if event.query()), dtype=np.int64)

    def order(self, keys: Slots, stream: torch.cuda.Stream | None) -> None:
        """Enqueue the ledger's reuse and producer dependencies on the consuming stream."""
        for key in keys.tolist():
            assert stream is not None
            stream.wait_event(self._events[key])

    def retain(self, keys: Slots) -> None:
        """Discard event objects only after the ledger stops referring to them."""
        retained = set(keys.tolist())
        for key in tuple(self._events):
            if key not in retained:
                del self._events[key]


class SlotPools:
    """Stable field storage shared by versioned state and transient hidden rows.

    Allocate each producer batch from free slots; allocation owns the first hold
    and orders the writer after prior releases. Readers add holds before the
    producer releases, wait for producer completion, and release with their own
    completion events. Hold/release take [rows, fields] slot matrices; tensor/views
    check allocation generations. Callers serialize ownership changes and keep
    storage alive until GPU work completes.
    """

    def __init__(
        self,
        layout: TensorLayoutSpec,
        capacities: tuple[int, ...],
        device: torch.device,
        *,
        host: bool = False,
        map_host: Callable[[torch.Tensor], torch.Tensor] | None = None,
        writer_fields: tuple[int, ...] = (),
    ) -> None:
        self.layout = layout
        self.writer_fields = writer_fields
        self.host_tensors: list[torch.Tensor] = []
        tensors: list[torch.Tensor] = []
        for field, capacity in zip(layout.fields, capacities, strict=True):
            size = field.dtype.itemsize
            stride = page_stride_bytes(field) // size
            alignment = max(field.alignment_bytes // size, 1)
            count = capacity * stride + alignment - 1
            mapped = host and device.type == "cuda"
            try:
                pool = torch.zeros(
                    count, dtype=field.dtype.torch, device="cpu" if mapped else device, pin_memory=mapped
                )
                offset = (-pool.data_ptr() // size) % alignment
                pool = (
                    pool[offset : offset + capacity * stride]
                    .view(capacity, stride)[:, : prod(field.shape)]
                    .view(capacity, *field.shape)
                )
                if mapped:
                    assert map_host is not None
                    self.host_tensors.append(pool)
                    pool = map_host(pool)
            except RuntimeError as error:
                if not mapped:
                    raise
                raise ValueError(
                    "host state does not fit pinned memory or UVA is unavailable; "
                    'reduce the table or use placement="device"'
                ) from error
            tensors.append(pool)
        self.tensors = tuple(tensors)
        self.ledger = _core.SlotLedger(list(capacities))
        self._events = PageEvents()

    def free_count(self, field: int) -> int:
        """Slots available for allocation, possibly still awaiting GPU completion."""
        return self.ledger.free_count(field)

    def allocate(self, field: int, count: int, stream: torch.cuda.Stream | None) -> tuple[Slots, Slots]:
        """Lease free pages and enqueue the Rust ledger's prior-reader dependencies."""
        try:
            slots, generations, waits, retained = self.ledger.allocate(field, count, self._events.completed())
        except _core.PageCapacity as error:
            raise WindowBackpressure(str(error)) from error
        self._events.order(waits, stream)
        self._events.retain(retained)
        return slots, generations

    def tensor(self, field: int, page: PageSlot) -> torch.Tensor:
        """Borrow one live allocation; the caller retains its hold while using the view."""
        self.ledger.check_page(field, page.slot_id, page.generation)
        return self.tensors[field][page.slot_id]

    def copy_from(self, field: int, source: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> None:
        """Copy selected source rows into leased slots; negative destinations disable padded entries."""
        copy_pages(source, self.tensors[field], rows, slots)

    def views(self, slots: Slots, generations: Slots, *, host: bool = False) -> tuple[torch.Tensor, ...]:
        """One tensor view per field of a version, after checking none of its pages was reclaimed.

        An older version keeps no pages for its writer fields (AdamW's step and moments); their views are meta
        placeholders with the field's shape, which fail on any read.
        """
        live = self.ledger.live(slots, generations)
        assert all(alive or field in self.writer_fields for field, alive in enumerate(live))
        tensors = self.host_tensors if host and self.host_tensors else self.tensors
        return tuple(
            tensor[slot] if alive else torch.empty(tensor.shape[1:], dtype=tensor.dtype, device="meta")
            for tensor, slot, alive in zip(tensors, slots.tolist(), live, strict=True)
        )

    def hold(self, slots: Slots) -> None:
        """Add a batch of holders, including repeated shared pages."""
        self.ledger.hold(slots)

    def release(self, slots: Slots, completion: torch.cuda.Event | None = None) -> None:
        """Drop holders, retaining completion objects until Rust permits physical reuse."""
        completed = self._events.completed()
        retained = self.ledger.release(slots, self._events.add(completion), completed)
        self._events.retain(retained)

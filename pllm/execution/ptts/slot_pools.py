"""Slot-addressed field storage shared by versioned PTTS state and transient hidden rows.

SlotPools owns the page tensors of one layout and their holders, generations and reuse fences.
PageBundleArena (versioned state) and the vLLM hidden-row capture use it; nothing here knows versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
import numpy.typing as npt
import torch

from pllm.contracts.tensor_layout import TensorLayoutSpec
from pllm.execution.admission import WindowBackpressure
from pllm.execution.ptts.page_bundle_planning import page_stride_bytes
from pllm.kernels.page_copy import copy_pages


@dataclass(frozen=True, slots=True)
class PageSlot:
    slot_id: int
    generation: int


Slots = npt.NDArray[np.int64]


class SlotPools:
    """Stable field storage shared by versioned state and transient hidden rows.

    Allocate each producer batch from free slots; allocation owns the first hold
    and orders the writer after prior releases. Readers add holds before the
    producer releases, wait for producer completion, and release with their own
    completion events. Hold/release take [rows, fields] slot matrices; tensor/views
    check allocation generations. Callers serialize ownership changes and keep
    storage alive until GPU work completes.
    """

    def __init__(self, layout: TensorLayoutSpec, capacity: int, device: torch.device) -> None:
        self.layout = layout
        tensors: list[torch.Tensor] = []
        for field in layout.fields:
            size = field.dtype.itemsize
            stride = page_stride_bytes(field) // size
            alignment = max(field.alignment_bytes // size, 1)
            count = capacity * stride + alignment - 1
            pool = torch.zeros(count, dtype=field.dtype.torch, device=device)
            offset = (-pool.data_ptr() // size) % alignment
            tensors.append(
                pool[offset : offset + capacity * stride]
                .view(capacity, stride)[:, : prod(field.shape)]
                .view(capacity, *field.shape)
            )
        self.tensors = tuple(tensors)
        # Per field and slot: how many owners share the page (0 = free), and how often it was reused.
        self.holders = np.zeros((len(layout.fields), capacity), dtype=np.int32)
        self.generations = np.zeros((len(layout.fields), capacity), dtype=np.int64)
        self._field_offsets = np.arange(len(layout.fields), dtype=np.int64) * capacity
        # Each completion fences a batch of field-offset slot IDs, independent of current holders.
        self._readiness: dict[torch.cuda.Event, Slots] = {}

    def free_count(self, field: int) -> int:
        """Slots available for allocation, possibly still awaiting GPU completion."""
        return int(np.count_nonzero(self.holders[field] == 0))

    def allocate(self, field: int, count: int, stream: torch.cuda.Stream | None) -> tuple[Slots, Slots]:
        """Lease free pages, preferring ready slots before ordering reuse behind pending readers."""
        free = np.flatnonzero(self.holders[field] == 0)
        if len(free) < count:
            raise WindowBackpressure("Slot capacity is exhausted; release current holders before retrying")
        self._collect_completed()
        slots = free[:count]
        if self._readiness:
            fenced = np.isin(free + self._field_offsets[field], np.concatenate(tuple(self._readiness.values())))
            ready = free[~fenced][:count]
            reuse = free[fenced][: count - len(ready)]
            slots = np.concatenate((ready, reuse))
            if reuse.size:
                allocated = reuse + self._field_offsets[field]
                for event, pending in tuple(self._readiness.items()):
                    overlap = np.isin(pending, allocated, assume_unique=True, kind="sort")
                    if overlap.any():
                        assert stream is not None
                        stream.wait_event(event)
                        # Other slots may be reused on another stream and still need this fence.
                        remaining = pending[~overlap]
                        if remaining.size:
                            self._readiness[event] = remaining
                        else:
                            del self._readiness[event]
        self.holders[field, slots] = 1
        return slots, self.generations[field, slots]

    def share(self, field: int, slots: Slots) -> None:
        """Let more versions hold these pages (fields a transition does not write)."""
        np.add.at(self.holders[field], slots, 1)

    def tensor(self, field: int, page: PageSlot) -> torch.Tensor:
        """Borrow one live allocation; the caller retains its hold while using the view."""
        assert self.generations[field, page.slot_id] == page.generation and self.holders[field, page.slot_id] >= 1
        return self.tensors[field][page.slot_id]

    def copy_from(self, field: int, source: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor) -> None:
        """Copy selected source rows into leased slots; negative destinations disable padded entries."""
        copy_pages(source, self.tensors[field], rows, slots)

    def views(self, slots: Slots, generations: Slots) -> tuple[torch.Tensor, ...]:
        """One tensor view per field of a version, after checking none of its pages was reclaimed."""
        fields = np.arange(len(slots))
        assert (self.generations[fields, slots] == generations).all() and (self.holders[fields, slots] >= 1).all()
        return tuple(tensor[slot] for tensor, slot in zip(self.tensors, slots.tolist()))

    def _counts(self, slots: Slots) -> Slots:
        return np.bincount((slots + self._field_offsets).ravel(), minlength=self.holders.size).reshape(
            self.holders.shape
        )

    def _collect_completed(self) -> None:
        for event in tuple(self._readiness):
            if event.query():
                del self._readiness[event]

    def hold(self, slots: Slots) -> None:
        """Add a batch of holders, including repeated shared pages."""
        self.holders += self._counts(slots)

    def release(self, slots: Slots, completion: torch.cuda.Event | None = None) -> None:
        """Drop a batch of holders; only the last holder advances the allocation generation."""
        counts = self._counts(slots)
        if completion is not None:
            self._collect_completed()
            pending = np.flatnonzero(counts)
            if completion in self._readiness:
                pending = np.union1d(self._readiness[completion], pending)
            self._readiness[completion] = pending
        self.holders -= counts
        self.generations += (counts != 0) & (self.holders == 0)

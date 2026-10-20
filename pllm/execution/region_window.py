"""Fixed private tensor slots for the shared region executor, ordered by device events."""

from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol

import torch

from pllm.contracts.region_streams import RegionWindowSlot
from pllm.execution.region_values import Data

_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


class _StorageSlice(Protocol):
    """Typed slice boundary for PyTorch's unannotated storage API."""

    def __getitem__(self, index: slice) -> torch.UntypedStorage: ...


def _storage_slice(storage: _StorageSlice, offset: int, size: int) -> torch.UntypedStorage:
    return storage[offset : offset + size]


class WindowBackpressure(RuntimeError):
    """The caller may retry admission after the current owner releases its window."""


@dataclass(eq=False)
class _Slot:
    tensor: torch.Tensor
    backing: torch.Tensor
    producer: torch.cuda.Event | None = None
    readers: dict[int, torch.cuda.Event] = field(default_factory=lambda: dict[int, torch.cuda.Event]())

    @classmethod
    def allocate(cls, backing: torch.Tensor, dtype: torch.dtype, offset: int, size: int) -> "_Slot":
        """Give the slot its own zero-offset storage view without another allocation."""
        storage = _storage_slice(backing.untyped_storage(), offset, size)
        tensor = torch.empty(0, dtype=dtype, device=backing.device).set_(storage, 0, (size // dtype.itemsize,), (1,))
        return cls(tensor, backing)

    def order(self, stream: torch.cuda.Stream) -> None:
        if self.producer is not None:
            stream.wait_event(self.producer)
        for event in self.readers.values():
            stream.wait_event(event)


class RegionWindow:
    """One reserved arena; every physical overwrite waits for its last device users."""

    def __init__(self, slots: tuple[RegionWindowSlot, ...]) -> None:
        self._lock = Lock()
        self._slots: dict[int, tuple[_Slot, ...]] = {}
        for spec in slots:
            dtype = _DTYPES[spec.dtype]
            backing = torch.empty(spec.count * spec.stride_bytes, dtype=torch.uint8, device=spec.device)
            self._slots[spec.value_id] = tuple(
                _Slot.allocate(backing, dtype, i * spec.stride_bytes, spec.bytes) for i in range(spec.count)
            )
        self._free: dict[int, deque[int]] = {}
        self._live: dict[tuple[int, tuple[int, ...]], tuple[int, _Slot]] = {}
        self._reading: dict[int, _Slot] = {}
        self._reader_events: dict[int, torch.cuda.Event] = {}
        self.reuses = 0

    @contextmanager
    def reserve(self) -> Generator[None, None, None]:
        """Reserve the complete signed slot set before any input copy or cell issues."""
        if not self._lock.acquire(blocking=False):
            raise WindowBackpressure("region window is already reserved; retry admission asynchronously")
        self._free = {key: deque(range(len(slots))) for key, slots in self._slots.items()}
        self.reuses = 0
        self._reader_events = {}
        try:
            yield
        finally:
            self._live.clear()
            self._reading.clear()
            self._free.clear()
            self._lock.release()

    def store(self, point: tuple[int, tuple[int, ...]], value: torch.Tensor) -> torch.Tensor:
        """Materialize a private SSA leaf in a compiler-funded physical slot."""
        value_id = point[0]
        if value_id not in self._slots:
            return value  # A declared readonly borrow remains with its original owner.
        assert self._free[value_id], "source lifetime exceeded its signed window slots"
        index = self._free[value_id].popleft()
        slot = self._slots[value_id][index]
        stream = torch.cuda.current_stream(value.device) if value.is_cuda else None
        if stream is not None:
            slot.order(stream)
            slot.backing.record_stream(stream)
        self.reuses += int(slot.producer is not None)
        destination = slot.tensor.as_strided(value.shape, value.stride())
        try:
            destination.copy_(value)
        finally:
            slot.readers.clear()
            if stream is not None:
                if slot.producer is None:
                    slot.producer = torch.cuda.Event()
                # CUDA waits capture the record at issue time; all old waiters
                # have issued before a logically dead slot is overwritten.
                slot.producer.record(stream)
        self._live[point] = index, slot
        return destination

    def read(self, point: tuple[int, tuple[int, ...]], value: Data) -> None:
        held = self._live.get(point)
        if held is None:
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(torch.cuda.current_stream(value.device))
        else:
            _, slot = held
            if slot.tensor.is_cuda:
                stream = torch.cuda.current_stream(slot.tensor.device)
                slot.backing.record_stream(stream)
                if slot.producer is not None:
                    stream.wait_event(slot.producer)
                self._reading[id(slot)] = slot

    def finish_reads(self) -> None:
        """Record one last-reader event per issuing stream after its whole statement."""
        if self._reading:
            first = next(iter(self._reading.values()))
            stream = torch.cuda.current_stream(first.tensor.device)
            event = self._reader_events.get(stream.cuda_stream)
            if event is None:
                event = torch.cuda.Event()
                self._reader_events[stream.cuda_stream] = event
            event.record(stream)
            for slot in self._reading.values():
                slot.readers[stream.cuda_stream] = event
            self._reading.clear()

    def release(self, point: tuple[int, tuple[int, ...]]) -> None:
        held = self._live.pop(point, None)
        if held is not None:
            index, _ = held
            self._free[point[0]].append(index)

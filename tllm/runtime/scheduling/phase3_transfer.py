"""Python-owned pinned buffers and asynchronous transfers for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol, TypeVar, cast, runtime_checkable

import torch

from tllm.workgraph.phase3 import MemoryKind, Move, MoveKind


@runtime_checkable
class CopyFenceLike(Protocol):
    def query(self) -> bool: ...


class TensorLike(Protocol):
    shape: object
    dtype: object
    device: object


@runtime_checkable
class TensorCopyAdapter(Protocol):
    def allocate_pinned(self, shape: tuple[int, ...], dtype: str) -> object: ...

    def copy(
        self,
        kind: str,
        source: object,
        destination: object,
        *,
        staging: object | None = None,
    ) -> CopyFenceLike: ...


_T = TypeVar("_T")


def _instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        try:
            value = tuple(cast(object, value))  # type: ignore[call-overload]
        except TypeError as exc:
            raise TypeError("tensor shape must be iterable") from exc
    dimensions = tuple(
        _integer(dimension, "tensor shape dimension", minimum=1)
        for dimension in cast(tuple[object, ...], value)
    )
    if not dimensions:
        raise ValueError("tensor shape must be non-empty")
    return dimensions


@dataclass(slots=True, eq=False)
class PinnedMemorySlot:
    slot: int
    shape: tuple[int, ...]
    dtype: str
    tensor: object
    acquired: bool = field(default=True, init=False)
    retire_fence: CopyFenceLike | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.slot = _integer(self.slot, "PinnedMemorySlot.slot")
        self.shape = _shape(self.shape)
        if not isinstance(self.dtype, str) or not self.dtype:
            raise TypeError("PinnedMemorySlot.dtype must be a non-empty str")


class PinnedMemoryPool:
    """Bounded pinned slots; a retire fence owns a slot until completion."""

    __slots__ = ("_capacity", "_adapter", "_slots", "_lock")

    def __init__(self, *, capacity: int, adapter: TensorCopyAdapter) -> None:
        self._capacity = _integer(capacity, "PinnedMemoryPool.capacity", minimum=1)
        raw_adapter = cast(object, adapter)
        if not isinstance(raw_adapter, TensorCopyAdapter):
            raise TypeError("PinnedMemoryPool.adapter must implement TensorCopyAdapter")
        self._adapter = adapter
        self._slots: list[PinnedMemorySlot] = []
        self._lock = Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        with self._lock:
            return sum(slot.acquired for slot in self._slots)

    def acquire(self, *, shape: tuple[int, ...], dtype: str) -> PinnedMemorySlot:
        normalized_shape = _shape(shape)
        if not isinstance(dtype, str) or not dtype:
            raise TypeError("PinnedMemoryPool dtype must be a non-empty str")
        with self._lock:
            self._reap_locked()
            for slot in self._slots:
                if (
                    not slot.acquired
                    and slot.shape == normalized_shape
                    and slot.dtype == dtype
                ):
                    slot.acquired = True
                    slot.retire_fence = None
                    return slot
            if len(self._slots) >= self._capacity:
                raise BufferError("pinned-memory pool capacity is exhausted")
            tensor = self._adapter.allocate_pinned(normalized_shape, dtype)
            slot = PinnedMemorySlot(
                slot=len(self._slots),
                shape=normalized_shape,
                dtype=dtype,
                tensor=tensor,
            )
            self._slots.append(slot)
            return slot

    def retire(self, slot: PinnedMemorySlot, fence: CopyFenceLike) -> None:
        slot = _instance(slot, PinnedMemorySlot, "PinnedMemoryPool.slot")
        raw_fence = cast(object, fence)
        if not isinstance(raw_fence, CopyFenceLike):
            raise TypeError("PinnedMemoryPool fence must implement query()")
        with self._lock:
            if slot not in self._slots or not slot.acquired:
                raise ValueError("pinned-memory slot is not acquired from this pool")
            slot.retire_fence = fence

    def reap(self) -> None:
        with self._lock:
            self._reap_locked()

    def _reap_locked(self) -> None:
        for slot in self._slots:
            fence = slot.retire_fence
            if slot.acquired and fence is not None and fence.query():
                slot.acquired = False
                slot.retire_fence = None


@dataclass(slots=True)
class CopyHandle:
    moves: tuple[Move, ...]
    complete: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.moves), tuple) or any(
            not isinstance(move, Move) for move in cast(tuple[object, ...], self.moves)
        ):
            raise TypeError("CopyHandle.moves must be tuple[Move, ...]")
        if not self.moves:
            raise ValueError("CopyHandle.moves must be non-empty")


@dataclass(slots=True)
class _InFlightCopy:
    handle: CopyHandle
    source: TensorLike
    destination: TensorLike
    slot: PinnedMemorySlot | None
    index: int = 0
    fence: CopyFenceLike | None = None


class CopyExecutor:
    """Execute admitted one- or two-segment typed transfers asynchronously."""

    __slots__ = ("_pool", "_adapter", "_capacity", "_active", "_lock")

    def __init__(
        self,
        *,
        pool: PinnedMemoryPool,
        adapter: TensorCopyAdapter,
        capacity: int = 1,
    ) -> None:
        self._pool = _instance(pool, PinnedMemoryPool, "CopyExecutor.pool")
        raw_adapter = cast(object, adapter)
        if not isinstance(raw_adapter, TensorCopyAdapter):
            raise TypeError("CopyExecutor.adapter must implement TensorCopyAdapter")
        self._adapter = adapter
        self._capacity = _integer(capacity, "CopyExecutor.capacity", minimum=1)
        self._active: list[_InFlightCopy] = []
        self._lock = Lock()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._active)

    def submit(
        self,
        moves: tuple[Move, ...],
        *,
        source: TensorLike,
        destination: TensorLike,
    ) -> CopyHandle:
        raw_moves = cast(object, moves)
        if not isinstance(raw_moves, tuple) or any(
            not isinstance(move, Move) for move in cast(tuple[object, ...], raw_moves)
        ):
            raise TypeError("CopyExecutor.moves must be tuple[Move, ...]")
        typed_moves = cast(tuple[Move, ...], raw_moves)
        if not typed_moves or len(typed_moves) > 2:
            raise ValueError("CopyExecutor accepts one direct or two staged moves")
        if len(typed_moves) == 2 and not (
            typed_moves[0].kind == MoveKind.D2H
            and typed_moves[1].kind == MoveKind.H2D
            and typed_moves[0].destination == typed_moves[1].source
            and typed_moves[0].destination.memory == MemoryKind.PINNED_HOST
        ):
            raise ValueError("two-segment CopyExecutor routes must use pinned D2H then H2D")
        with self._lock:
            self._poll_locked()
            if len(self._active) >= self._capacity:
                raise BufferError("copy-engine capacity is exhausted")
            needs_staging = any(move.kind != MoveKind.P2P for move in typed_moves)
            slot: PinnedMemorySlot | None = None
            if needs_staging:
                shape = _shape(source.shape)
                dtype = str(source.dtype).removeprefix("torch.")
                slot = self._pool.acquire(shape=shape, dtype=dtype)
            handle = CopyHandle(typed_moves)
            active = _InFlightCopy(handle, source, destination, slot)
            self._submit_segment(active)
            self._active.append(active)
            return handle

    def poll(self) -> None:
        with self._lock:
            self._poll_locked()
        self._pool.reap()

    def _poll_locked(self) -> None:
        remaining: list[_InFlightCopy] = []
        for active in self._active:
            fence = active.fence
            if fence is None or not fence.query():
                remaining.append(active)
                continue
            if active.index + 1 < len(active.handle.moves):
                active.index += 1
                self._submit_segment(active)
                remaining.append(active)
                continue
            active.handle.complete = True
            if active.slot is not None:
                self._pool.retire(active.slot, fence)
        self._active = remaining

    def _submit_segment(self, active: _InFlightCopy) -> None:
        move = active.handle.moves[active.index]
        slot = active.slot
        if len(active.handle.moves) == 2:
            if slot is None:
                raise RuntimeError("host-staged transfer requires a pinned slot")
            if active.index == 0:
                source: object = active.source
                destination: object = slot.tensor
            else:
                source = slot.tensor
                destination = active.destination
        else:
            source = active.source
            destination = active.destination
        fence = self._adapter.copy(
            move.kind.value,
            source,
            destination,
            staging=None if slot is None else slot.tensor,
        )
        raw_fence = cast(object, fence)
        if not isinstance(raw_fence, CopyFenceLike):
            raise TypeError("copy adapter must return a fence implementing query()")
        active.fence = fence


class _TorchCopyFence:
    __slots__ = ("_event", "_finalize", "_finalized")

    def __init__(self, event: torch.cuda.Event, finalize: object | None = None) -> None:
        self._event = event
        self._finalize = finalize
        self._finalized = False

    def query(self) -> bool:
        complete = bool(self._event.query())
        if complete and not self._finalized and callable(self._finalize):
            cast(object, self._finalize)()  # type: ignore[operator]
            self._finalized = True
        return complete


class TorchTensorCopyAdapter:
    """Torch/CUDA adapter; runtime tensors and events never enter frozen plans."""

    _DTYPES: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }

    def allocate_pinned(self, shape: tuple[int, ...], dtype: str) -> torch.Tensor:
        try:
            torch_dtype = self._DTYPES[dtype]
        except KeyError as exc:
            raise ValueError(f"unsupported pinned tensor dtype {dtype!r}") from exc
        return torch.empty(shape, dtype=torch_dtype, device="cpu", pin_memory=True)

    def copy(
        self,
        kind: str,
        source: object,
        destination: object,
        *,
        staging: object | None = None,
    ) -> CopyFenceLike:
        source_tensor = _instance(source, torch.Tensor, "Torch copy source")
        destination_tensor = _instance(
            destination, torch.Tensor, "Torch copy destination"
        )
        staging_tensor = (
            None
            if staging is None
            else _instance(staging, torch.Tensor, "Torch copy staging")
        )
        finalize: object | None = None
        if kind == MoveKind.H2D.value:
            copy_source = source_tensor
            if staging_tensor is not None and source_tensor.device.type == "cpu":
                staging_tensor.copy_(source_tensor)
                copy_source = staging_tensor
            destination_tensor.copy_(copy_source, non_blocking=True)
        elif kind == MoveKind.D2H.value:
            if staging_tensor is not None and destination_tensor is not staging_tensor:
                staging_tensor.copy_(source_tensor, non_blocking=True)
                finalize = lambda: destination_tensor.copy_(staging_tensor)
            else:
                destination_tensor.copy_(source_tensor, non_blocking=True)
        elif kind == MoveKind.P2P.value:
            destination_tensor.copy_(source_tensor, non_blocking=True)
        else:
            raise ValueError("unsupported typed copy kind")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        return _TorchCopyFence(event, finalize)


__all__ = [
    "CopyExecutor",
    "CopyFenceLike",
    "CopyHandle",
    "PinnedMemoryPool",
    "PinnedMemorySlot",
    "TensorCopyAdapter",
    "TorchTensorCopyAdapter",
]

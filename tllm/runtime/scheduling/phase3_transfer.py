"""Python-owned pinned buffers and asynchronous transfers for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Protocol, TypeVar, cast, runtime_checkable

import torch

from tllm.workgraph.ir import NodeKey
from tllm.workgraph.phase3 import (
    CopyEngine,
    CopyEngineCapacityDesc,
    DeviceKind,
    MemoryKind,
    Move,
    MoveKind,
    Placement,
    PlacementPlan,
    ValueKey,
)


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


def _tensor_matches_placement(tensor: TensorLike, placement: Placement) -> bool:
    text = str(tensor.device)
    if placement.kind == DeviceKind.CPU:
        return text == "cpu" or text.startswith("cpu:")
    return text == f"cuda:{placement.index}"


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

    def release(self, slot: PinnedMemorySlot) -> None:
        """Return a slot when submission failed before producing a fence."""

        slot = _instance(slot, PinnedMemorySlot, "PinnedMemoryPool.slot")
        with self._lock:
            if slot not in self._slots or not slot.acquired:
                raise ValueError("pinned-memory slot is not acquired from this pool")
            if slot.retire_fence is not None:
                raise RuntimeError("a fenced pinned-memory slot must retire through reap()")
            slot.acquired = False

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
    error: Exception | None = field(default=None, init=False)

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

    __slots__ = ("_pool", "_adapter", "_capacities", "_active", "_lock")

    def __init__(
        self,
        *,
        pool: PinnedMemoryPool,
        adapter: TensorCopyAdapter,
        capacity: int = 1,
        capacities: tuple[CopyEngineCapacityDesc, ...] = (),
    ) -> None:
        self._pool = _instance(pool, PinnedMemoryPool, "CopyExecutor.pool")
        raw_adapter = cast(object, adapter)
        if not isinstance(raw_adapter, TensorCopyAdapter):
            raise TypeError("CopyExecutor.adapter must implement TensorCopyAdapter")
        self._adapter = adapter
        default_capacity = _integer(capacity, "CopyExecutor.capacity", minimum=1)
        raw_capacities = cast(object, capacities)
        if not isinstance(raw_capacities, tuple) or any(
            not isinstance(item, CopyEngineCapacityDesc)
            for item in cast(tuple[object, ...], raw_capacities)
        ):
            raise TypeError("CopyExecutor.capacities must be CopyEngineCapacityDesc tuple")
        typed_capacities = cast(tuple[CopyEngineCapacityDesc, ...], raw_capacities)
        if len({item.engine for item in typed_capacities}) != len(typed_capacities):
            raise ValueError("CopyExecutor capacities must have unique engines")
        self._capacities: dict[CopyEngine, int] = {
            engine: default_capacity for engine in CopyEngine
        }
        self._capacities.update(
            {item.engine: item.capacity for item in typed_capacities}
        )
        self._active: list[_InFlightCopy] = []
        self._lock = Lock()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def capacities(self) -> tuple[CopyEngineCapacityDesc, ...]:
        return tuple(
            CopyEngineCapacityDesc(engine, self._capacities[engine])
            for engine in CopyEngine
        )

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
            engine = typed_moves[0].engine
            if self._engine_in_flight(engine) >= self._capacities[engine]:
                raise BufferError("copy-engine capacity is exhausted")
            needs_staging = any(move.kind != MoveKind.P2P for move in typed_moves)
            slot: PinnedMemorySlot | None = None
            if needs_staging:
                shape = _shape(source.shape)
                dtype = str(source.dtype).removeprefix("torch.")
                slot = self._pool.acquire(shape=shape, dtype=dtype)
            handle = CopyHandle(typed_moves)
            active = _InFlightCopy(handle, source, destination, slot)
            try:
                self._submit_segment(active)
            except Exception:
                if slot is not None:
                    self._pool.release(slot)
                raise
            self._active.append(active)
            return handle

    def poll(self) -> None:
        with self._lock:
            self._poll_locked()
        self._pool.reap()

    def _poll_locked(self) -> None:
        remaining: list[_InFlightCopy] = []
        first_error: Exception | None = None
        for active in self._active:
            fence = active.fence
            if fence is None or not fence.query():
                remaining.append(active)
                continue
            if active.index + 1 < len(active.handle.moves):
                next_engine = active.handle.moves[active.index + 1].engine
                if self._engine_in_flight(next_engine, excluding=active) >= self._capacities[
                    next_engine
                ]:
                    remaining.append(active)
                    continue
                active.index += 1
                try:
                    self._submit_segment(active)
                except Exception as exc:
                    active.handle.error = exc
                    active.handle.complete = True
                    if active.slot is not None:
                        self._pool.release(active.slot)
                    if first_error is None:
                        first_error = exc
                else:
                    remaining.append(active)
                continue
            active.handle.complete = True
            if active.slot is not None:
                self._pool.retire(active.slot, fence)
        self._active = remaining
        if first_error is not None:
            raise first_error

    def _engine_in_flight(
        self,
        engine: CopyEngine,
        *,
        excluding: _InFlightCopy | None = None,
    ) -> int:
        return sum(
            active is not excluding
            and active.handle.moves[active.index].engine == engine
            for active in self._active
        )

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
        if not _tensor_matches_placement(source, move.source.placement):
            raise ValueError("source tensor device does not match scheduled move")
        if not _tensor_matches_placement(destination, move.destination.placement):
            raise ValueError("destination tensor device does not match scheduled move")
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


class PlacementPlanDispatcher:
    """Bind frozen plan routes and per-engine credits to the copy runtime."""

    __slots__ = ("plan", "_executor", "_routes")

    def __init__(
        self,
        *,
        plan: PlacementPlan,
        pool: PinnedMemoryPool,
        adapter: TensorCopyAdapter,
    ) -> None:
        raw_plan = _instance(plan, PlacementPlan, "PlacementPlanDispatcher.plan")
        self.plan = replace(raw_plan)
        self._executor = CopyExecutor(
            pool=pool,
            adapter=adapter,
            capacities=self.plan.copy_engines,
        )
        routes: dict[tuple[NodeKey, NodeKey, ValueKey], list[Move]] = {}
        for scheduled in self.plan.moves:
            if scheduled.producer is None or scheduled.consumer is None:
                raise ValueError("placement plan move lacks producer/consumer binding")
            key = (scheduled.producer, scheduled.consumer, scheduled.move.value)
            routes.setdefault(key, []).append(scheduled.move)
        self._routes = {
            key: tuple(moves) for key, moves in routes.items()
        }

    @property
    def in_flight(self) -> int:
        return self._executor.in_flight

    @property
    def capacities(self) -> tuple[CopyEngineCapacityDesc, ...]:
        return self._executor.capacities

    def submit(
        self,
        *,
        producer: NodeKey,
        consumer: NodeKey,
        value: ValueKey,
        source: TensorLike,
        destination: TensorLike,
    ) -> CopyHandle:
        key = (
            _instance(producer, NodeKey, "PlacementPlanDispatcher.producer"),
            _instance(consumer, NodeKey, "PlacementPlanDispatcher.consumer"),
            _instance(value, ValueKey, "PlacementPlanDispatcher.value"),
        )
        try:
            moves = self._routes[key]
        except KeyError as exc:
            raise KeyError("placement plan has no matching transfer route") from exc
        return self._executor.submit(
            moves,
            source=source,
            destination=destination,
        )

    def poll(self) -> None:
        self._executor.poll()


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
        cuda_device = (
            destination_tensor.device
            if destination_tensor.device.type == "cuda"
            else source_tensor.device
        )
        if cuda_device.type != "cuda":
            raise ValueError("typed H2D/D2H/P2P copy requires one CUDA endpoint")
        stream = torch.cuda.current_stream(cuda_device)
        with torch.cuda.stream(stream):
            if kind == MoveKind.H2D.value:
                copy_source = source_tensor
                if (
                    staging_tensor is not None
                    and source_tensor.device.type == "cpu"
                    and source_tensor is not staging_tensor
                ):
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
            event.record(stream)
        return _TorchCopyFence(event, finalize)


__all__ = [
    "CopyExecutor",
    "CopyFenceLike",
    "CopyHandle",
    "PinnedMemoryPool",
    "PinnedMemorySlot",
    "PlacementPlanDispatcher",
    "TensorCopyAdapter",
    "TorchTensorCopyAdapter",
]

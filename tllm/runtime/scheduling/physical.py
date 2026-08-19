"""Python-owned fence and buffer primitives for physical WorkGraph execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeVar, cast, runtime_checkable

import torch

from tllm.workgraph.frontend import Device
from tllm.workgraph.physical import (
    BufferBucketDesc,
    BufferOwnershipKind,
    FenceKey,
)


@runtime_checkable
class EventLike(Protocol):
    def record(self, stream: object) -> None: ...

    def query(self) -> bool: ...


@runtime_checkable
class StreamLike(Protocol):
    def wait_event(self, event: object) -> None: ...


@runtime_checkable
class GraphLike(Protocol):
    def replay(self) -> None: ...


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


@dataclass(slots=True)
class Fence:
    """Runtime event owned by Python and named by a frozen descriptor key."""

    key: FenceKey
    event: EventLike
    recorded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.key = _instance(self.key, FenceKey, "Fence.key")
        event = cast(object, self.event)
        if not isinstance(event, EventLike):
            raise TypeError("Fence.event must implement record(stream) and query()")

    def record(self, stream: StreamLike | None) -> None:
        self.event.record(cast(object, stream))
        self.recorded = True

    def wait(self, stream: StreamLike) -> None:
        stream.wait_event(cast(object, self.event))

    @property
    def complete(self) -> bool:
        return bool(self.event.query())


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


@dataclass(slots=True)
class BufferLease:
    """One runtime Tensor slot governed by ready/completion/retire fences."""

    bucket: BufferBucketDesc
    tensor: torch.Tensor
    plan_epoch: int
    slot: int
    ready_fence: Fence | None
    completion_fence: Fence
    retire_fence: Fence | None

    def __post_init__(self) -> None:
        self.bucket = _instance(self.bucket, BufferBucketDesc, "BufferLease.bucket")
        self.tensor = _instance(
            cast(object, self.tensor), torch.Tensor, "BufferLease.tensor"
        )
        self.plan_epoch = _integer(self.plan_epoch, "BufferLease.plan_epoch", minimum=1)
        self.slot = _integer(self.slot, "BufferLease.slot")
        if self.slot >= self.bucket.slots:
            raise ValueError("BufferLease.slot must reference a bucket slot")
        if tuple(int(dimension) for dimension in self.tensor.shape) != self.bucket.shape:
            raise ValueError("BufferLease.tensor shape differs from its buffer bucket")
        expected_dtype = _TORCH_DTYPES.get(self.bucket.dtype.name)
        if self.bucket.dtype.namespace != "torch" or expected_dtype is None:
            raise ValueError("BufferLease bucket dtype is not supported by the Python executor")
        if self.tensor.dtype != expected_dtype:
            raise ValueError("BufferLease.tensor dtype differs from its buffer bucket")
        expected_device = self.bucket.device
        if self.tensor.device.type != expected_device.kind.value:
            raise ValueError("BufferLease.tensor device differs from its buffer bucket")
        if expected_device.kind == Device.CUDA and (
            expected_device.index is not None
            and self.tensor.device.index != expected_device.index
        ):
            raise ValueError("BufferLease.tensor CUDA index differs from its buffer bucket")
        if self.ready_fence is not None:
            self.ready_fence = _instance(
                self.ready_fence, Fence, "BufferLease.ready_fence"
            )
        self.completion_fence = _instance(
            self.completion_fence, Fence, "BufferLease.completion_fence"
        )
        if self.retire_fence is not None:
            self.retire_fence = _instance(
                self.retire_fence, Fence, "BufferLease.retire_fence"
            )
        if (
            self.bucket.ownership == BufferOwnershipKind.BORROWED
            and self.ready_fence is None
        ):
            raise ValueError("borrowed BufferLease requires an explicit ready fence")
        if (
            self.bucket.ownership == BufferOwnershipKind.MANAGED
            and self.retire_fence is None
        ):
            raise ValueError("managed BufferLease requires an explicit retire fence")

    def wait_ready(self, stream: StreamLike) -> None:
        if self.ready_fence is not None:
            self.ready_fence.wait(stream)

    def publish_completion(self, stream: StreamLike | None) -> None:
        self.completion_fence.record(stream)

    def publish_retire(self, stream: StreamLike | None) -> None:
        if self.retire_fence is not None:
            self.retire_fence.record(stream)

    @property
    def reusable(self) -> bool:
        if not self.completion_fence.recorded:
            return False
        retire = self.retire_fence
        if retire is not None and not retire.recorded:
            return False
        if not self.completion_fence.complete:
            return False
        return retire is None or retire.complete


__all__ = [
    "BufferLease",
    "EventLike",
    "Fence",
    "GraphLike",
    "StreamLike",
]

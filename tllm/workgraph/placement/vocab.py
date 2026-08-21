#!/usr/bin/env python3
"""Device/memory/move vocabulary and core placement descriptors."""

from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar, cast
from tllm.workgraph.ir import ExecutorKey, NodeKey


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
class MemoryKind(StrEnum):
    HOST = "host"
    PINNED_HOST = "pinned_host"
    DEVICE = "device"
class MoveKind(StrEnum):
    H2D = "h2d"
    D2H = "d2h"
    P2P = "p2p"
class CopyEngine(StrEnum):
    H2D = "h2d"
    D2H = "d2h"
    P2P = "p2p"
class PlacementCompilationError(RuntimeError):
    """Raised when a frozen placement plan fails cross-language validation."""
@dataclass(frozen=True, slots=True, order=True)
class Placement:
    kind: DeviceKind
    index: int | None = None

    def __post_init__(self) -> None:
        kind = require_enum(cast(object, self.kind), DeviceKind, "Placement.kind")
        index = self.index
        if kind == DeviceKind.CPU:
            if index is not None:
                raise ValueError("CPU placement cannot declare a device index")
        elif index is None:
            raise ValueError("CUDA placement requires a device index")
        else:
            index = require_int(index, "Placement.index")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "index", index)

    def __str__(self) -> str:
        return self.kind.value if self.index is None else f"{self.kind.value}:{self.index}"
@dataclass(frozen=True, slots=True, order=True)
class ValueKey:
    name: str
    version: int

    def __post_init__(self) -> None:
        name = require_str(self.name, "ValueKey.name")
        if "/" in name or "\\" in name:
            raise ValueError("ValueKey.name must not contain '/' or '\\'")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "version",
            require_int(self.version, "ValueKey.version", minimum=1),
        )

    @property
    def value(self) -> str:
        return f"{self.name}@{self.version}"
@dataclass(frozen=True, slots=True)
class Residency:
    value: ValueKey
    placement: Placement
    memory: MemoryKind | None = None

    def __post_init__(self) -> None:
        value = require_instance(self.value, ValueKey, "Residency.value")
        placement = require_instance(self.placement, Placement, "Residency.placement")
        memory = self.memory
        if memory is None:
            memory = (
                MemoryKind.DEVICE
                if placement.kind == DeviceKind.CUDA
                else MemoryKind.HOST
            )
        else:
            memory = require_enum(cast(object, memory), MemoryKind, "Residency.memory")
        if placement.kind == DeviceKind.CUDA and memory != MemoryKind.DEVICE:
            raise ValueError("CUDA residency requires device memory")
        if placement.kind == DeviceKind.CPU and memory == MemoryKind.DEVICE:
            raise ValueError("CPU residency cannot use device memory")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "memory", memory)
_T = TypeVar("_T")
_EnumT = TypeVar("_EnumT", bound=StrEnum)
def require_instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value
def require_enum(value: object, expected: type[_EnumT], label: str) -> _EnumT:
    if isinstance(value, expected):
        return value
    if isinstance(value, str):
        try:
            return expected(value)
        except ValueError as exc:
            raise ValueError(f"{label} is unsupported") from exc
    raise TypeError(f"{label} must be {expected.__name__}")
def require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if not value or "0x" in value.lower() or "<" in value or ">" in value:
        raise ValueError(f"{label} must be a stable non-empty string")
    return value
def require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value
def require_executor_key(value: object, label: str) -> ExecutorKey:
    key = require_instance(value, ExecutorKey, label)
    components = (
        require_str(key.namespace, f"{label}.namespace"),
        require_str(key.name, f"{label}.name"),
        require_str(key.version, f"{label}.version"),
    )
    if any(":" in item or "@" in item for item in components):
        raise ValueError(f"{label} components must not contain ':' or '@'")
    return ExecutorKey(*components)
def require_node_key(value: object, label: str) -> NodeKey:
    key = require_instance(value, NodeKey, label)
    return NodeKey(
        require_str(key.namespace, f"{label}.namespace"),
        require_str(key.owner, f"{label}.owner"),
        require_int(key.ordinal, f"{label}.ordinal"),
    )
def require_tuple_of(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return cast(tuple[_T, ...], items)

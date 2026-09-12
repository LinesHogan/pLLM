"""Imported tensor storage and access facts, separate from algorithm semantics.

These describe the unfused exported graph. They are not device allocator sizes,
backend scratch bounds, cache misses, or measured HBM traffic.
"""

from dataclasses import dataclass
from enum import StrEnum


class TensorAccessKind(StrEnum):
    VIEW = "view"
    DENSE = "dense"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TensorStorage:
    device: str
    bytes: int


@dataclass(frozen=True, slots=True)
class TensorStorageView:
    name: str
    storage: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int
    dtype: str


@dataclass(frozen=True, slots=True)
class TensorAccess:
    # The node name also identifies its single tensor result.
    node: str
    operator: str
    kind: TensorAccessKind
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TensorAccessGraph:
    source_digest: str
    graph_digest: str
    storages: tuple[TensorStorage, ...]
    values: tuple[TensorStorageView, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    operations: tuple[TensorAccess, ...]


@dataclass(frozen=True, slots=True)
class TensorMemoryStep:
    node: str
    live_storages: tuple[int, ...]
    release_storages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TensorDeviceMemory:
    device: str
    # Input capacity is the imported private layout's capacity. Composing a
    # call with borrowed buffers must substitute their original owner's storage
    # capacity and aliases; shape/stride guards alone do not bound a view's base.
    input_bytes: int
    result_bytes: int
    peak_live_tensor_bytes: int
    peak_new_tensor_bytes: int
    # Each distinct operand value is counted once per known dense operation.
    # Views transfer zero bytes; overlapping/expanded values and cache reuse
    # require a later backend model. Unknown operations are listed separately.
    logical_read_bytes: int
    logical_write_bytes: int


@dataclass(frozen=True, slots=True)
class TensorMemoryProfile:
    access_digest: str
    devices: tuple[TensorDeviceMemory, ...]
    steps: tuple[TensorMemoryStep, ...]
    unknown_accesses: tuple[str, ...]
    signature: str

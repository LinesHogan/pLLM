"""Composition of source tensor lifetimes and original backend retention."""

from dataclasses import dataclass

from pllm.contracts.region_capacity import RegionCapacity
from pllm.contracts.region_retention import RegionRetention


@dataclass(frozen=True, slots=True)
class RegionPayloadDevice:
    device: str
    tensor_bytes_upper_bound: int
    retained_bytes_upper_bound: int
    peak_bytes_upper_bound: int


@dataclass(frozen=True, slots=True)
class RegionPayload:
    capacity: RegionCapacity
    retention: RegionRetention
    devices: tuple[RegionPayloadDevice, ...]
    signature: str

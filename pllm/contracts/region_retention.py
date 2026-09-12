"""Backend-owned allocations retained between source calls; no runtime owner."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetentionEndpoint:
    function: str
    argument: int


@dataclass(frozen=True, slots=True)
class RetainedAllocation:
    name: str
    acquire: RetentionEndpoint
    release: RetentionEndpoint
    device: str
    bytes: int


@dataclass(frozen=True, slots=True)
class RetentionBound:
    name: str
    device: str
    bytes_per_instance: int
    acquisitions: str
    releases: str
    live_instances: str
    peak_instances_upper_bound: int
    peak_bytes_upper_bound: int


@dataclass(frozen=True, slots=True)
class RegionRetention:
    program_digest: str
    schedule_map: str
    imports_digest: str
    allocations: tuple[RetentionBound, ...]
    signature: str

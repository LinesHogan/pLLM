"""Native placement and bounded phase completion, separate from source semantics."""

from dataclasses import dataclass

from pllm.contracts.region_bindings import AccessTable
from pllm.contracts.region_payload import RegionPayload


@dataclass(frozen=True, slots=True)
class RegionStreamDevice:
    device: str
    concurrent_scratch_bytes_upper_bound: int
    pending_phase_bytes_upper_bound: int
    peak_bytes_upper_bound: int


@dataclass(frozen=True, slots=True)
class RegionStreamWaits:
    lane: int
    predecessors: tuple[AccessTable, ...]


@dataclass(frozen=True, slots=True)
class RegionStreamPlan:
    version: int
    program_digest: str
    schedule_map: str
    axis: int
    lanes: int
    pipeline_depth: int
    phase_schedule: str
    placements: tuple[AccessTable, ...]
    waits: tuple[RegionStreamWaits, ...]
    host_fences: tuple[int, ...]
    payload: RegionPayload
    devices: tuple[RegionStreamDevice, ...]
    signature: str


@dataclass(frozen=True, slots=True)
class RegionStreamExploration:
    program_digest: str
    candidates: tuple[RegionStreamPlan, ...]
    diagnostics: tuple[str, ...]

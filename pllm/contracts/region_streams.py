"""Native placement and bounded phase completion, separate from source semantics."""

from dataclasses import dataclass

from pllm.contracts.region_bindings import AccessTable
from pllm.contracts.region_payload import RegionPayload


@dataclass(frozen=True, slots=True)
class RegionStreamDevice:
    device: str
    concurrent_scratch_bytes_upper_bound: int
    window_storage_bytes_upper_bound: int
    entry_copy_bytes_upper_bound: int
    result_copy_bytes_upper_bound: int
    peak_bytes_upper_bound: int


@dataclass(frozen=True, slots=True)
class RegionStreamWaits:
    lane: int
    predecessors: tuple[AccessTable, ...]


@dataclass(frozen=True, slots=True)
class RegionWindowSlot:
    """Compiler-bounded private value storage, reusable after its device readers."""

    value_id: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    count: int
    bytes: int
    stride_bytes: int


@dataclass(frozen=True, slots=True)
class RegionStateWindow:
    """Additional COW roots per cell; initial pins and current roots remain owned."""

    resource: str
    private_roots: int


@dataclass(frozen=True, slots=True)
class RegionStreamPlan:
    """One private slot bank with at most one in-flight page window per owner.

    Within that submission, pipeline_depth bounds active GPU phases. The signed
    source domain bounds receipt metadata; slot and state facts bound storage.
    """

    version: int
    program_digest: str
    schedule_map: str
    axis: int
    lanes: int
    pipeline_depth: int
    phase_schedule: str
    placements: tuple[AccessTable, ...]
    waits: tuple[RegionStreamWaits, ...]
    slots: tuple[RegionWindowSlot, ...]
    state_windows: tuple[RegionStateWindow, ...]
    payload: RegionPayload
    devices: tuple[RegionStreamDevice, ...]
    signature: str


@dataclass(frozen=True, slots=True)
class RegionStreamExploration:
    program_digest: str
    candidates: tuple[RegionStreamPlan, ...]
    diagnostics: tuple[str, ...]

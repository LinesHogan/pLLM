"""Measured candidate selection in AER; payload limits are not device budgets."""

from dataclasses import dataclass
from typing import Literal

from pllm.contracts.region_capacity import RegionCapacity
from pllm.contracts.region_payload import RegionPayload
from pllm.contracts.region_retention import RetainedAllocation
from pllm.contracts.region_imports import RegionTensorImport
from pllm.contracts.region_outlining import CallOutlineEvidence
from pllm.contracts.region_streams import RegionStreamPlan


@dataclass(frozen=True, slots=True)
class RegionSelectionPolicy:
    allow_numerical_transforms: bool = True
    schedule_policy: Literal["auto", "serial"] = "auto"


@dataclass(frozen=True, slots=True)
class RegionPayloadLimit:
    device: str
    bytes: int


@dataclass(frozen=True, slots=True)
class RegionCandidate:
    schedule_map: str
    imports: tuple[RegionTensorImport, ...]
    # End-to-end execution, including frame copies and completion. Measurements
    # rank this workload; they do not prove latency or memory for other inputs.
    elapsed_ns: tuple[int, ...]
    outlines: tuple[CallOutlineEvidence, ...] = ()
    stream_plan: RegionStreamPlan | None = None


@dataclass(frozen=True, slots=True)
class RegionTuningSize:
    name: str
    value: int


@dataclass(frozen=True, slots=True)
class RegionSelectionRequest:
    source_imports: tuple[RegionTensorImport, ...]
    # Candidate zero must be the source program's reference execution.
    candidates: tuple[RegionCandidate, ...]
    payload_limits: tuple[RegionPayloadLimit, ...]
    allow_numerical_transforms: bool
    measurement_context: str
    # One example workload for timing. The Program's full parameter domain
    # continues to govern legality and capacity; this never specializes it.
    tuning_sizes: tuple[RegionTuningSize, ...] = ()
    retention_imports: tuple[RetainedAllocation, ...] = ()


@dataclass(frozen=True, slots=True)
class RegionCandidateAssessment:
    payload: RegionPayload
    # Effective physical peak, including concurrent scratch when applicable.
    # payload remains the independently verifiable phase storage/retention proof.
    peak_bytes: tuple[RegionPayloadLimit, ...]
    median_ns: int | None
    feasible: bool
    # Changed imports that can execute somewhere in the full source domain.
    changed_functions: tuple[str, ...]

    @property
    def capacity(self) -> RegionCapacity:
        return self.payload.capacity


@dataclass(frozen=True, slots=True)
class RegionSelection:
    program_digest: str
    request_digest: str
    candidates: tuple[RegionCandidateAssessment, ...]
    selected: int
    signature: str

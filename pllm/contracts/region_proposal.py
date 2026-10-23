"""A fixed-source schedule proposal with its storage and physical evidence."""

from dataclasses import dataclass

from pllm.contracts.region_payload import RegionPayload
from pllm.contracts.region_program import Program
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.region_streams import RegionStateWindow, RegionStreamPlan
from pllm.contracts.visibility import StateVisibility

REGION_STREAMS_CAPABILITY = "execution/region_streams@1"


@dataclass(frozen=True, slots=True)
class RegionScheduleProposal:
    payload: RegionPayload
    stream_plan: RegionStreamPlan | None = None

    @property
    def peak_bytes(self) -> tuple[RegionPayloadLimit, ...]:
        devices = self.payload.devices if self.stream_plan is None else self.stream_plan.devices
        return tuple(RegionPayloadLimit(device.device, device.peak_bytes_upper_bound) for device in devices)


@dataclass(frozen=True, slots=True)
class ClosedWindowRequest:
    """Engine constraints on independent, invariant-only source scans."""

    counts: tuple[str, ...]
    quantum: int
    token_budget: int
    visibility: StateVisibility


@dataclass(frozen=True, slots=True)
class ClosedRegionWindow:
    """A verified source restriction; closing it leaves no live SSA continuation."""

    source_digest: str
    request: ClosedWindowRequest
    chunks: int
    program: Program
    proposal: RegionScheduleProposal
    state_windows: tuple[RegionStateWindow, ...]
    workspace_bytes: int
    diagnostics: tuple[str, ...]
    signature: str

"""A fixed-source schedule proposal with its storage and physical evidence."""

from dataclasses import dataclass

from pllm.contracts.region_payload import RegionPayload
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.region_streams import RegionStreamPlan


@dataclass(frozen=True, slots=True)
class RegionScheduleProposal:
    payload: RegionPayload
    stream_plan: RegionStreamPlan | None = None

    @property
    def peak_bytes(self) -> tuple[RegionPayloadLimit, ...]:
        devices = self.payload.devices if self.stream_plan is None else self.stream_plan.devices
        return tuple(RegionPayloadLimit(device.device, device.peak_bytes_upper_bound) for device in devices)

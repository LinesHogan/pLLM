"""Workload-independent physical facts for synchronous typed host execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    SchedulerTimeline,
    TimelineAnchor,
)


@dataclass(frozen=True, slots=True)
class CanonicalHostRuntimeFacts:
    device: torch.device
    plan_epoch: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.device) is not torch.device
            or self.device.type not in ("cpu", "cuda")
            or type(self.plan_epoch) is not int
            or self.plan_epoch < 1
        ):
            raise ValueError("canonical host runtime facts are not closed")

    def timeline(
        self,
        extra_capabilities: tuple[str, ...] = (),
    ) -> SchedulerTimeline:
        compute_capability = (
            "compute/cpu@1"
            if self.device.type == "cpu"
            else "compute/cuda@1"
        )
        return SchedulerTimeline(
            anchors=(
                TimelineAnchor(AnchorSymbol("host.input.ready"), 0),
                TimelineAnchor(AnchorSymbol("host.compute.end"), 1),
                TimelineAnchor(AnchorSymbol("host.output.publish"), 2),
                TimelineAnchor(AnchorSymbol("host.next"), 3),
            ),
            capabilities=tuple(
                sorted(
                    (
                        compute_capability,
                        "host/service@1",
                        *extra_capabilities,
                    )
                )
            ),
        )


__all__ = ["CanonicalHostRuntimeFacts"]

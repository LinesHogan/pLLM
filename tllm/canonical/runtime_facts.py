"""Workload-neutral canonical runtime facts for in-process request lifecycles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    SchedulerTimeline,
    TimelineAnchor,
)


@dataclass(frozen=True, slots=True)
class CanonicalRuntimeFacts:
    device: torch.device
    plan_epoch: int = 1

    def __post_init__(self) -> None:
        if self.device.type not in ("cpu", "cuda") or self.plan_epoch < 1:
            raise ValueError("canonical runtime facts are invalid")

    def timeline(
        self,
        extra_capabilities: tuple[str, ...] = (),
    ) -> SchedulerTimeline:
        compute = (
            "compute/cpu@1"
            if self.device.type == "cpu"
            else "compute/cuda@1"
        )
        return SchedulerTimeline(
            anchors=(
                TimelineAnchor(AnchorSymbol("request.prepare"), 0),
                TimelineAnchor(AnchorSymbol("model.execute"), 1),
                TimelineAnchor(AnchorSymbol("state.publish"), 2),
                TimelineAnchor(AnchorSymbol("sampler.pre_draw"), 3),
                TimelineAnchor(AnchorSymbol("sampler.complete"), 4),
                TimelineAnchor(AnchorSymbol("token.commit"), 5),
                TimelineAnchor(AnchorSymbol("request.finish"), 6),
                TimelineAnchor(AnchorSymbol("next.prepare"), 7),
            ),
            capabilities=tuple(
                sorted((compute, "host/service@1", *extra_capabilities))
            ),
        )


__all__ = ["CanonicalRuntimeFacts"]

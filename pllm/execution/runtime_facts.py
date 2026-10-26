"""Workload-neutral canonical runtime facts for in-process request lifecycles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    SchedulerTimeline,
    TimelineAnchor,
)


@dataclass(frozen=True, slots=True)
class RuntimeFacts:
    device: torch.device
    plan_epoch: int = 1

    def __post_init__(self) -> None:
        if self.device.type not in ("cpu", "cuda") or self.plan_epoch < 1:
            raise ValueError("canonical runtime facts are invalid")

    def timeline(
        self,
        extra_capabilities: tuple[str, ...] = (),
        *,
        layers: int = 0,
        internal: bool = False,
    ) -> SchedulerTimeline:
        compute = "compute/cpu@1" if self.device.type == "cpu" else "compute/cuda@1"
        return SchedulerTimeline(
            anchors=tuple(
                TimelineAnchor(AnchorSymbol(name), index)
                for index, name in enumerate(
                    (
                        "request.prepare",
                        "model.execute",
                        *(f"model.layer.{layer}" for layer in range(layers)),
                        *(("model.norm",) if layers else ()),
                        "state.publish",
                        "sampler.pre_draw",
                        "sampler.complete",
                        "token.commit",
                        "request.finish",
                        "next.prepare",
                    )
                )
            ),
            internal_points=internal,
            capabilities=tuple(sorted((compute, "host/service@1", *extra_capabilities))),
        )


__all__ = ["RuntimeFacts"]

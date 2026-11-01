"""Model timeline construction shared by backend adapters and workload fixtures."""

from __future__ import annotations

import torch

from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    SchedulerTimeline,
    TimelineAnchor,
)


def model_timeline(device: torch.device, capabilities: tuple[str, ...] = ()) -> SchedulerTimeline:
    """Build the model-step anchors and capabilities for a CPU or CUDA device."""
    if device.type not in ("cpu", "cuda"):
        raise ValueError("model timelines need a CPU or CUDA device")
    compute = "compute/cpu@1" if device.type == "cpu" else "compute/cuda@1"
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
        capabilities=tuple(sorted((compute, *capabilities))),
    )


__all__ = ["model_timeline"]

"""Pure timeline construction from a backend's declared capabilities."""

from __future__ import annotations

from tllm.backends.protocol import BackendCapabilities
from tllm.schedule.contracts import (
    BackendObservableEvent,
    MainWorkloadRegion,
    MainWorkloadTimeline,
    TimelineAnchor,
)


def export_main_workload_timeline(
    capabilities: BackendCapabilities,
    *,
    graph_mode: str,
    static_headroom_bytes: int,
    backend_version: str = "0.27.1",
) -> MainWorkloadTimeline:
    """Build pinned main-workload facts without touching backend-private data."""
    if graph_mode not in {"eager", "piecewise", "full"}:
        raise ValueError("graph_mode must be eager, piecewise, or full")
    capture_region = "" if graph_mode == "eager" else "model"
    anchors = (
        TimelineAnchor("step.begin", 0, backend_events=()),
        TimelineAnchor(
            "prepare.pre",
            1,
            backend_events=(BackendObservableEvent.PREPARE_INPUTS_PRE,),
        ),
        TimelineAnchor(
            "prepare.post",
            2,
            backend_events=(BackendObservableEvent.PREPARE_INPUTS_POST,),
        ),
        TimelineAnchor(
            "model.execute",
            3,
            backend_events=(BackendObservableEvent.EXECUTE_MODEL_PRE,),
        ),
        TimelineAnchor(
            "logits.post",
            4,
            visible_values=("model.hidden", "model.logits"),
            backend_events=(BackendObservableEvent.EXECUTE_MODEL_POST,),
        ),
        TimelineAnchor(
            "sampler.pre_draw",
            5,
            visible_values=("model.logits",),
            backend_events=(
                BackendObservableEvent.SAMPLE_TOKENS_PRE,
                BackendObservableEvent.SAMPLER_PRE_DRAW,
            ),
        ),
        TimelineAnchor(
            "token.commit",
            6,
            visible_values=("sample.token",),
            backend_events=(BackendObservableEvent.SAMPLE_TOKENS_POST,),
        ),
        TimelineAnchor(
            "step.end",
            7,
            backend_events=(BackendObservableEvent.SAMPLE_TOKENS_POST,),
        ),
        TimelineAnchor("next.prepare", 8, backend_events=()),
        TimelineAnchor("generate.end", 9, backend_events=()),
    )
    regions = (
        MainWorkloadRegion("prepare", "prepare.pre", "prepare.post"),
        MainWorkloadRegion("model", "model.execute", "logits.post", capture_region=capture_region),
        MainWorkloadRegion("logits", "logits.post", "sampler.pre_draw"),
        MainWorkloadRegion("sampler", "sampler.pre_draw", "token.commit"),
        MainWorkloadRegion("commit", "token.commit", "step.end"),
    )
    capability_names = set(capabilities.capability_names())
    if graph_mode == "eager":
        capability_names.add("eager_only")
        capability_names.update(
            capability.value for capability in capabilities.eager_extensions
        )
    capability_names.update(
        f"sampler/{stage.name.lower()}" for stage in capabilities.sampler_stages
    )
    return MainWorkloadTimeline(
        backend_id=capabilities.backend_id,
        backend_version=backend_version,
        runner_family=capabilities.runner_family,
        graph_mode=graph_mode,
        anchors=anchors,
        regions=regions,
        static_headroom_bytes=static_headroom_bytes,
        capabilities=tuple(sorted(capability_names)),
        schema_version=2,
    )


__all__ = ["export_main_workload_timeline"]

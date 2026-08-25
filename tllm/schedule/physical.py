"""Mechanical lowering from a validated schedule decision to runtime descriptors."""

from __future__ import annotations

import hashlib

from tllm.schedule.compiler import validate_schedule
from tllm.schedule.contracts import (
    CompileOptions,
    ExecutionRegion,
    MainWorkloadTimeline,
    RegionImplementationKind,
    ScheduleDecision,
    ScheduleGraph,
)
from tllm.workgraph.frontend import ArtifactKey
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.physical import (
    AnchorPlacementDesc,
    CapturePolicy,
    FallbackPolicy,
    FenceDependencyDesc,
    FenceKey,
    FenceKind,
    PhysicalFlowPlan,
    PhysicalNodePlan,
    StreamClass,
)


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def executor_key_for_implementation(implementation_id: str) -> ExecutorKey:
    """Return the stable runtime handle for one selected implementation."""

    if not isinstance(implementation_id, str) or not implementation_id:
        raise TypeError("implementation_id must be a non-empty str")
    return ExecutorKey(
        namespace="workload",
        name=_stable_digest("implementation", implementation_id),
        version="1",
    )


def _node_key(graph: ScheduleGraph, task_id: str) -> NodeKey:
    return NodeKey(
        namespace="workload_region",
        owner=_stable_digest(graph.workload_key, task_id),
        ordinal=0,
    )


def executor_key_for_region(region: ExecutionRegion) -> ExecutorKey:
    """Return the executor binding selected by an immutable Region decision."""

    if not isinstance(region, ExecutionRegion):
        raise TypeError("region must be ExecutionRegion")
    if region.implementation_kind is RegionImplementationKind.EXTERN:
        return executor_key_for_implementation(region.implementation_ids[0])
    return ExecutorKey(
        namespace="workload_region",
        name=_stable_digest("region-artifact", region.artifact_key),
        version="2",
    )


def artifact_key_for_region(region: ExecutionRegion) -> ArtifactKey:
    """Return the immutable artifact identity selected by a Region decision."""

    if not isinstance(region, ExecutionRegion):
        raise TypeError("region must be ExecutionRegion")
    return ArtifactKey(
        namespace="workload_region",
        name=_stable_digest("region", region.region_id),
        version=2,
        digest=region.artifact_key,
    )


def lower_physical_flow_plan(
    graph: ScheduleGraph,
    timeline: MainWorkloadTimeline,
    decision: ScheduleDecision,
    options: CompileOptions = CompileOptions(),
    *,
    plan_epoch: int = 1,
) -> PhysicalFlowPlan:
    """Mechanically bind one physical executor per authoritative Region."""

    validate_schedule(graph, timeline, decision, options)

    node_by_region = {
        region.region_id: _node_key(graph, region.region_id)
        for region in decision.execution_regions
    }
    incoming_fences: dict[str, list[FenceDependencyDesc]] = {
        region.region_id: [] for region in decision.execution_regions
    }
    for ordinal, fence in enumerate(decision.fences):
        incoming_fences[fence.target].append(
            FenceDependencyDesc(
                fence=FenceKey(
                    namespace="schedule",
                    name=f"fence-{_stable_digest(fence.fence_id)[:24]}",
                    ordinal=ordinal,
                ),
                producer=node_by_region[fence.source],
                consumer=node_by_region[fence.target],
                kind=FenceKind(fence.kind),
            )
        )

    nodes = tuple(
        PhysicalNodePlan(
            node_key=node_by_region[region.region_id],
            executor=executor_key_for_region(region),
            stream=StreamClass(region.stream.value),
            fences=tuple(incoming_fences[region.region_id]),
            artifact_key=artifact_key_for_region(region),
            capture_policy=CapturePolicy(region.capture_mode.value),
            fallback_policy=FallbackPolicy(region.fallback_policy.value),
        )
        for region in decision.execution_regions
    )
    physical = PhysicalFlowPlan(
        plan_epoch=plan_epoch,
        nodes=nodes,
        anchor_placements=tuple(
            AnchorPlacementDesc(
                node_key=node_by_region[region.region_id],
                after_anchor=region.placement_anchor,
            )
            for region in decision.execution_regions
        ),
        schedule_signature=decision.signature,
    )
    from tllm import core as _core

    _core.validate_physical_flow_plan(physical)
    return physical


__all__ = [
    "artifact_key_for_region",
    "executor_key_for_implementation",
    "executor_key_for_region",
    "lower_physical_flow_plan",
]

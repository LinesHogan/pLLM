"""Compiler middle-end schedule contracts.

The contract dataclasses load eagerly; the compiler entry points and the
physical lowering helpers resolve lazily via PEP 562 so that importing a
native-stack module (``tllm.schedule.contracts`` via ``tllm.workload``)
never pulls the legacy workgraph frontend (a ``tllm.schedule.physical``
dependency, gone at legacy retirement) into a compiler-only process.
"""

from typing import TYPE_CHECKING, Any

from tllm.schedule.contracts import (
    AffineRegionSchedule,
    BackendObservableEvent,
    BackendRegionBinderStage,
    CompileOptions,
    CostBreakdown,
    ExecutionRegion,
    HiddenPhase,
    HiddenLayerKind,
    HiddenSite,
    MainWorkloadRegion,
    MainWorkloadTimeline,
    MemoryAllocation,
    MemoryPlan,
    MaterializedValue,
    RegionInputBinding,
    RegionInputKind,
    RegionImplementationKind,
    RegionCompilerAdapter,
    RegionAutogradPolicy,
    RegionCompileCommand,
    RegionCaptureMode,
    RegionStreamClass,
    RegionFallbackPolicy,
    RegionOutputBinding,
    RegionOutputKind,
    RegionPort,
    RegionTaskInvocation,
    RegionTaskPortBinding,
    ResidualResourceBinding,
    ResourceReservation,
    RuntimeAdvancePlan,
    ScheduleAffineAccess,
    ScheduleAffineExpr,
    ScheduleDecision,
    ScheduleDomain,
    ScheduleEdge,
    ScheduleEdgeKind,
    ScheduleEffect,
    ScheduleEffectKind,
    ScheduleInput,
    ScheduleInputKind,
    ScheduleOutput,
    ScheduleOutputKind,
    ScheduleState,
    ScheduleFence,
    ScheduleGraph,
    ScheduleTask,
    ScheduleTree,
    ScheduleValue,
    TaskPlacement,
    TaskTiming,
    TimelineAnchor,
)

if TYPE_CHECKING:
    from tllm.schedule.compiler import compile_schedule, validate_schedule
    from tllm.schedule.coordinator import (
        WorkloadCompilation,
        WorkloadCompilerCoordinator,
        PlanEpochAllocator,
        install_workload_compilation,
        workload_compilation_cache_key,
    )
    from tllm.schedule.physical import (
        executor_key_for_implementation,
        executor_key_for_region,
        lower_physical_flow_plan,
    )


def __getattr__(name: str) -> Any:
    if name in {"compile_schedule", "validate_schedule"}:
        from tllm.schedule import compiler

        return getattr(compiler, name)
    if name in {
        "executor_key_for_implementation",
        "executor_key_for_region",
        "lower_physical_flow_plan",
    }:
        from tllm.schedule import physical

        return getattr(physical, name)
    if name in {
        "WorkloadCompilation",
        "WorkloadCompilerCoordinator",
        "PlanEpochAllocator",
        "install_workload_compilation",
        "workload_compilation_cache_key",
    }:
        from tllm.schedule import coordinator

        return getattr(coordinator, name)
    raise AttributeError(f"module 'tllm.schedule' has no attribute {name!r}")

__all__ = [
    "AffineRegionSchedule",
    "BackendObservableEvent",
    "BackendRegionBinderStage",
    "CompileOptions",
    "CostBreakdown",
    "ExecutionRegion",
    "HiddenPhase",
    "HiddenLayerKind",
    "HiddenSite",
    "MainWorkloadRegion",
    "MainWorkloadTimeline",
    "MemoryAllocation",
    "MemoryPlan",
    "MaterializedValue",
    "RegionInputBinding",
    "RegionInputKind",
    "RegionImplementationKind",
    "RegionCompilerAdapter",
    "RegionAutogradPolicy",
    "RegionCompileCommand",
    "RegionCaptureMode",
    "RegionStreamClass",
    "RegionFallbackPolicy",
    "RegionOutputBinding",
    "RegionOutputKind",
    "RegionPort",
    "RegionTaskInvocation",
    "RegionTaskPortBinding",
    "ResidualResourceBinding",
    "ResourceReservation",
    "RuntimeAdvancePlan",
    "ScheduleAffineAccess",
    "ScheduleAffineExpr",
    "ScheduleDecision",
    "ScheduleDomain",
    "ScheduleEdge",
    "ScheduleEdgeKind",
    "ScheduleEffect",
    "ScheduleEffectKind",
    "ScheduleInput",
    "ScheduleInputKind",
    "ScheduleOutput",
    "ScheduleOutputKind",
    "ScheduleState",
    "ScheduleFence",
    "ScheduleGraph",
    "ScheduleTask",
    "ScheduleTree",
    "ScheduleValue",
    "TaskPlacement",
    "TaskTiming",
    "TimelineAnchor",
    "WorkloadCompilation",
    "WorkloadCompilerCoordinator",
    "PlanEpochAllocator",
    "compile_schedule",
    "executor_key_for_implementation",
    "executor_key_for_region",
    "install_workload_compilation",
    "lower_physical_flow_plan",
    "validate_schedule",
    "workload_compilation_cache_key",
]

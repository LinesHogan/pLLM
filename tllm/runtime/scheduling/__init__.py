#!/usr/bin/env python3
"""Shared scheduling infrastructure for off-critical-path consumer work."""

from tllm.runtime.scheduling.deferred import DeferredWork, DeferredWorkQueue
from tllm.runtime.scheduling.physical import BufferLease, Fence
from tllm.runtime.scheduling.plan_store import (
    PlanResources,
    PlanSnapshotStore,
    RuntimeEvent,
    RuntimeGraph,
    RuntimeStream,
    select_buffer_bucket,
)
from tllm.runtime.scheduling.workgraph_executor import (
    PhysicalExecutionContext,
    PhysicalExecutionError,
    PhysicalExecutionKind,
    PhysicalExecutionStats,
    PhysicalExecutor,
    PhysicalStepResult,
    SpecializationRequestDesc,
    WorkGraphRuntimeStats,
    WorkGraphRuntimeStatsReader,
    runtime_stats_from_executor,
)

__all__ = [
    "BufferLease",
    "DeferredWork",
    "DeferredWorkQueue",
    "Fence",
    "PlanResources",
    "PlanSnapshotStore",
    "PhysicalExecutionContext",
    "PhysicalExecutionError",
    "PhysicalExecutionKind",
    "PhysicalExecutionStats",
    "PhysicalExecutor",
    "PhysicalStepResult",
    "SpecializationRequestDesc",
    "WorkGraphRuntimeStats",
    "WorkGraphRuntimeStatsReader",
    "RuntimeEvent",
    "RuntimeGraph",
    "RuntimeStream",
    "select_buffer_bucket",
    "runtime_stats_from_executor",
]

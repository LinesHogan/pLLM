#!/usr/bin/env python3
"""Producer plan: registry, compiler, and contracts."""

from tllm.producers.base import (
    BindContext,
    BoundProducer,
    HookPoint,
    PortSink,
    Producer,
)
from tllm.producers.registry import (
    ExecutionPlan,
    ExecutionPlanError,
    PlannedProducer,
    ProducerEntry,
    ProducerRegistry,
    compile_execution_plan,
)

__all__ = [
    "BindContext",
    "BoundProducer",
    "ExecutionPlan",
    "ExecutionPlanError",
    "HookPoint",
    "PlannedProducer",
    "PortSink",
    "Producer",
    "ProducerEntry",
    "ProducerRegistry",
    "compile_execution_plan",
]

#!/usr/bin/env python3
"""Producer plan: registry, compiler, and contracts."""

from tllm.producers.base import (
    HookPoint,
    Producer,
    ProducerDescriptor,
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
    "ExecutionPlan",
    "ExecutionPlanError",
    "HookPoint",
    "PlannedProducer",
    "Producer",
    "ProducerDescriptor",
    "ProducerEntry",
    "ProducerRegistry",
    "compile_execution_plan",
]

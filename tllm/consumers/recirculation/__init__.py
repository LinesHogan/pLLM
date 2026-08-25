#!/usr/bin/env python3
"""Residual-stream Recirculation consumer."""

from tllm.consumers.recirculation.config import RecirculationConfig
from tllm.consumers.recirculation.consumer import RecirculationConsumer
from tllm.consumers.recirculation.reference import (
    SerialRecirculationResult,
    run_serial_recirculation,
)
from tllm.consumers.recirculation.region_workload import (
    RecirculationExecutionRequest,
    RecirculationExecutionResult,
    RecirculationRegionResources,
    RecirculationRegionWorkload,
    build_recirculation_region_workload,
    run_exact_recirculation_region,
)
from tllm.consumers.recirculation.vllm_region_workload import (
    PREFILL_BINDER_IMPLEMENTATION_ID,
    VllmRecirculationRegionWorkload,
    build_vllm_recirculation_region_workload,
)
from tllm.workgraph.recurrence import JacobiPolicy

__all__ = [
    "JacobiPolicy",
    "RecirculationConfig",
    "RecirculationConsumer",
    "RecirculationExecutionRequest",
    "RecirculationExecutionResult",
    "RecirculationRegionResources",
    "RecirculationRegionWorkload",
    "VllmRecirculationRegionWorkload",
    "PREFILL_BINDER_IMPLEMENTATION_ID",
    "SerialRecirculationResult",
    "build_recirculation_region_workload",
    "build_vllm_recirculation_region_workload",
    "run_exact_recirculation_region",
    "run_serial_recirculation",
]

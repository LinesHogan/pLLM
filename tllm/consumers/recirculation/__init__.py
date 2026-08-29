#!/usr/bin/env python3
"""Residual-stream Recirculation consumer."""

from tllm.consumers.recirculation.reference import (
    SerialRecirculationResult,
    run_serial_recirculation,
)
from tllm.consumers.recirculation.native_backend import (
    PREFILL_BINDER_IMPLEMENTATION_ID,
)
from tllm.consumers.recirculation.workload import (
    RecirculationConfig,
    RecirculationWorkload,
)

__all__ = [
    "RecirculationConfig",
    "RecirculationWorkload",
    "PREFILL_BINDER_IMPLEMENTATION_ID",
    "SerialRecirculationResult",
    "run_serial_recirculation",
]

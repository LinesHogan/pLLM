#!/usr/bin/env python3
"""Common runtime hook helpers for signed native backend events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from tllm.backends.protocol import BackendObservableEvent

if TYPE_CHECKING:
    from tllm.runtime.workload_execution import NativeWorkloadExecution


class RuntimeWithWorkloadExecution(Protocol):
    workload_execution: "NativeWorkloadExecution | None"
    event_step_id: int


def _advance_native_workload(
    runtime: RuntimeWithWorkloadExecution,
    event: BackendObservableEvent | None,
) -> None:
    execution = runtime.workload_execution
    if execution is None or event is None:
        return
    execution.advance_backend_event(event)


def dispatch_runtime_event(
    *,
    runtime: RuntimeWithWorkloadExecution,
    event: BackendObservableEvent,
) -> int:
    """Advance a signed backend event through native workload execution."""
    _advance_native_workload(runtime, event)
    return 0

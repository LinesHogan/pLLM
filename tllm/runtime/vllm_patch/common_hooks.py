#!/usr/bin/env python3
"""Common runtime hook helpers for signed native backend events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence

import torch

from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RunnerLike, RuntimeContext
from tllm.backends.protocol import BackendObservableEvent
from tllm.runtime.vllm_patch.compilation import is_torch_compiling

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


def _runner_uses_cudagraph(runner: RunnerLike) -> bool:
    use_cuda_graph = getattr(runner, "use_cuda_graph", None)
    if use_cuda_graph is True:
        return True

    cfg = getattr(runner, "compilation_config", None)
    if cfg is None:
        return False
    try:
        level = int(getattr(cfg, "level", 0))
    except Exception:
        level = 0
    mode_raw = getattr(cfg, "cudagraph_mode", 0)
    if mode_raw is None:
        mode = 0
    else:
        mode = 0
        value = getattr(mode_raw, "value", mode_raw)
        try:
            mode = int(value)
        except Exception:
            if isinstance(value, tuple):
                try:
                    mode = 1 if any(int(x) != 0 for x in value) else 0
                except Exception:
                    mode = 1
            else:
                name = str(getattr(mode_raw, "name", value)).strip().upper()
                mode = 0 if name in {"", "0", "NONE"} else 1
    return (level > 0) or (mode > 0)


def _resolve_device(runner: RunnerLike) -> torch.device:
    device = getattr(runner, "device", None)
    if isinstance(device, torch.device):
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_runtime_context(
    *,
    runner: RunnerLike,
    event_name: str,
    predecessor_events: Sequence[object] = (),
) -> RuntimeContext:
    device = _resolve_device(runner)
    main_stream = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            main_stream = torch.cuda.current_stream(device=device)
        except Exception:
            main_stream = None
    return RuntimeContext(
        runner=runner,
        model=getattr(runner, "model", None),
        device=device,
        main_stream=main_stream,
        is_compiling=is_torch_compiling(),
        uses_cudagraph=_runner_uses_cudagraph(runner),
        event_name=event_name,
        predecessor_events=tuple(predecessor_events),
    )


def dispatch_runtime_event(
    *,
    runtime: RuntimeWithWorkloadExecution,
    runner: RunnerLike,
    event: BackendObservableEvent,
    phase: Optional[str] = None,
    layer_path: Optional[str] = None,
    capture_enabled: Optional[bool] = None,
    batch: Optional[HiddenBatch] = None,
    batch_factory: Optional[Callable[[], Optional[HiddenBatch]]] = None,
    predecessor_events: Sequence[object] = (),
) -> int:
    """Advance a signed backend event through native workload execution."""
    return _dispatch_runtime_event(
        runtime=runtime,
        runner=runner,
        event_name=event.value,
        native_event=event,
        phase=phase,
        layer_path=layer_path,
        capture_enabled=capture_enabled,
        batch=batch,
        batch_factory=batch_factory,
        predecessor_events=predecessor_events,
    )


def dispatch_legacy_runtime_event(
    *,
    runtime: RuntimeWithWorkloadExecution,
    runner: RunnerLike,
    event_name: str,
    phase: Optional[str] = None,
    layer_path: Optional[str] = None,
    capture_enabled: Optional[bool] = None,
    batch: Optional[HiddenBatch] = None,
    batch_factory: Optional[Callable[[], Optional[HiddenBatch]]] = None,
    predecessor_events: Sequence[object] = (),
) -> int:
    """Legacy event-consumer dispatch has been retired."""
    _ = (
        runtime,
        runner,
        event_name,
        phase,
        layer_path,
        capture_enabled,
        batch,
        batch_factory,
        predecessor_events,
    )
    return 0


def _dispatch_runtime_event(
    *,
    runtime: RuntimeWithWorkloadExecution,
    runner: RunnerLike,
    event_name: str,
    native_event: BackendObservableEvent | None,
    phase: Optional[str] = None,
    layer_path: Optional[str] = None,
    capture_enabled: Optional[bool] = None,
    batch: Optional[HiddenBatch] = None,
    batch_factory: Optional[Callable[[], Optional[HiddenBatch]]] = None,
    predecessor_events: Sequence[object] = (),
) -> int:
    """Advance one signed runtime event; retired consumers receive nothing."""
    _ = (
        runner,
        event_name,
        phase,
        layer_path,
        capture_enabled,
        batch,
        batch_factory,
        predecessor_events,
    )
    _advance_native_workload(runtime, native_event)
    return 0

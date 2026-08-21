#!/usr/bin/env python3
"""Common runtime hook helpers for consumer dispatch."""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

import torch

from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.runtime_context import RunnerLike, RuntimeContext
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.consumer_compat import dispatch_consumer_event
from tllm.runtime.scheduling.deferred import DeferredWorkQueue
from tllm.runtime.vllm_patch.compilation import is_torch_compiling


class RuntimeWithDispatchPlan(Protocol):
    dispatch_plan: DispatchPlan | None
    deferred_work_queue: DeferredWorkQueue
    event_step_id: int


def _advance_distributed_collective_window(
    runtime: RuntimeWithDispatchPlan,
    event_name: str,
) -> None:
    from tllm.runtime.state import ResidualRuntimeState
    from tllm.workgraph.distributed import CollectiveWindow

    if not isinstance(runtime, ResidualRuntimeState):
        return
    window = {
        "prepare_inputs.pre": CollectiveWindow.PREPARE,
        "execute_model.pre": CollectiveWindow.BEFORE_COMPUTE,
        "execute_model.post": CollectiveWindow.AFTER_COMPUTE,
        "sample_tokens.post": CollectiveWindow.STEP_END,
    }.get(event_name)
    store = runtime.distributed_plan_store
    step = runtime.distributed_prepared_step_id
    if window is not None and store is not None and step >= 0:
        store.executor(step).advance_window(step, window)


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
    runtime: RuntimeWithDispatchPlan,
    runner: RunnerLike,
    event_name: str,
    phase: Optional[str] = None,
    layer_path: Optional[str] = None,
    capture_enabled: Optional[bool] = None,
    batch: Optional[HiddenBatch] = None,
    batch_factory: Optional[Callable[[], Optional[HiddenBatch]]] = None,
    predecessor_events: Sequence[object] = (),
) -> int:
    """Dispatch one runtime event to registered consumers, if any."""
    _advance_distributed_collective_window(runtime, event_name)
    effective_predecessors = tuple(predecessor_events)
    if (
        runtime.deferred_work_queue
        and not effective_predecessors
        and event_name == "prepare_inputs.pre"
    ):
        device = _resolve_device(runner)
        if device.type == "cuda" and torch.cuda.is_available():
            stream = torch.cuda.current_stream(device=device)
            deadline_fence = torch.cuda.Event(blocking=False)
            deadline_fence.record(stream)
            effective_predecessors = (deadline_fence,)
    if runtime.deferred_work_queue:
        from tllm.runtime.ports.residual_bundle_dispatch import drain_deferred_port_bundles

        drain_deferred_port_bundles(
            runtime=runtime,
            step_id=int(runtime.event_step_id),
            hook=event_name,
            expired_only=event_name == "prepare_inputs.pre",
            predecessor_events=effective_predecessors,
        )
    plan = getattr(runtime, "dispatch_plan", None)
    if plan is None:
        return 0

    if capture_enabled is None:
        capture_enabled = not _runner_uses_cudagraph(runner)

    targets = plan.select(
        event_name=event_name,
        phase=phase,
        layer_path=layer_path,
        capture_enabled=bool(capture_enabled),
    )
    if not targets:
        return 0

    ctx = build_runtime_context(
        runner=runner,
        event_name=event_name,
        predecessor_events=effective_predecessors,
    )
    payload = batch
    if payload is None and batch_factory is not None:
        payload = batch_factory()

    for target in targets:
        dispatch_consumer_event(
            consumer=target.consumer,
            payload=payload,
            event_name=event_name,
            ctx=ctx,
        )
    return len(targets)


def dispatch_port_bundle(
    *,
    runtime: RuntimeWithDispatchPlan,
    runner: RunnerLike,
    bundle: PortBundle,
    window: str,
) -> int:
    """Dispatch one assembled bundle to flow-based consumers."""
    plan = getattr(runtime, "dispatch_plan", None)
    if plan is None:
        return 0

    targets = [target for target in plan.flow_targets() if str(target.flow.window) == str(window)]
    if not targets:
        return 0

    ctx = build_runtime_context(runner=runner, event_name=f"flow:{window}")
    for target in targets:
        target.consumer.consume_flow(target.flow_ordinal, bundle, ctx)
    return len(targets)

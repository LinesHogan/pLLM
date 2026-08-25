"""Generic vLLM binding for Rust-signed backend Regions.

Native workload execution reaches this module only through signed backend
events.  It binds a local slot to the exact stage identity selected by Rust;
it does not classify a workload, interpret recurrence operation names, or
construct a schedule.  Historical dispatch-plan replay remains isolated in
``model_region_legacy`` and is loaded only when no native execution exists.
"""

from __future__ import annotations

from typing import Any

from tllm.runtime.backend_region_binders import (
    BackendRegionStageContext,
    BackendRegionStagerRegistry,
)
from tllm.runtime.workload_execution import BackendObservableEvent


def _legacy() -> object:
    from tllm.backends.vllm import model_region_legacy

    return model_region_legacy


def _worker_registry(runtime: object) -> BackendRegionStagerRegistry | None:
    registry = getattr(runtime, "backend_region_stager_registry", None)
    if registry is None:
        return None
    if not isinstance(registry, BackendRegionStagerRegistry):
        raise TypeError("backend_region_stager_registry must be BackendRegionStagerRegistry")
    return registry


def stage_signed_backend_region_event(
    *,
    runtime: object,
    runner: object,
    event: BackendObservableEvent,
    num_scheduled_tokens: object | None,
    input_batch: object | None = None,
    step_id: object | None = None,
) -> bool:
    """Stage exactly the backend bindings selected by the signed event record."""

    if not isinstance(event, BackendObservableEvent):
        raise TypeError("backend Region event must be BackendObservableEvent")
    execution = getattr(runtime, "workload_execution", None)
    if execution is None:
        return False
    stages = execution.backend_region_binder_stages(event)
    if not stages:
        return False
    registry = _worker_registry(runtime)
    # Preserve the raw backend value: BackendRegionStageContext owns the
    # non-coercing signed step-id check.
    resolved_step_id = getattr(runtime, "event_step_id", 0) if step_id is None else step_id
    for stage in stages:
        try:
            slot = execution.backend_region_binders[stage.implementation_id]
        except KeyError as exc:
            raise RuntimeError(
                "signed backend Region has no local binder slot for "
                f"{stage.implementation_id!r}"
            ) from exc
        context = BackendRegionStageContext(
            implementation_id=stage.implementation_id,
            backend_event=stage.backend_event.value,
            region_id=stage.region_id,
            slot=slot,
            runtime=runtime,
            runner=runner,
            step_id=resolved_step_id,
            num_scheduled_tokens=num_scheduled_tokens,
            input_batch=input_batch,
        )
        if registry is None:
            slot.stage_from_backend(context)
        else:
            registry.stage(context)
    return True


def prepare_model_region_step(
    *,
    runtime: object,
    runner: object,
    num_scheduled_tokens: object | None,
    input_batch: object | None = None,
) -> object | None:
    """Compatibility entry point; native stages only at their signed event."""

    if getattr(runtime, "workload_execution", None) is not None:
        return None
    return _legacy().prepare_model_region_step(
        runtime=runtime,
        runner=runner,
        num_scheduled_tokens=num_scheduled_tokens,
        input_batch=input_batch,
    )


def abort_model_region_step(runtime: object, step_id: int) -> bool:
    execution = getattr(runtime, "workload_execution", None)
    if execution is not None:
        from tllm.backends.vllm.typed_layer_replay import abort_typed_layer_replay

        aborted = False
        for binder in execution.backend_region_binders.values():
            aborted = bool(binder.abort(step_id)) or aborted
        return abort_typed_layer_replay(runtime) or aborted
    return _legacy().abort_model_region_step(runtime, step_id)


def install_model_region_executor(*, binding: object, runtime: object) -> object:
    return _legacy().install_model_region_executor(binding=binding, runtime=runtime)


def restore_model_region_executors(binding: object) -> int:
    return _legacy().restore_model_region_executors(binding)


def restore_runtime_model_region(runtime: object) -> int:
    if getattr(runtime, "workload_execution", None) is not None:
        from tllm.backends.vllm.typed_layer_replay import restore_typed_layer_replay

        return restore_typed_layer_replay(runtime)
    return _legacy().restore_runtime_model_region(runtime)


def __getattr__(name: str) -> Any:
    """Keep legacy test/import compatibility without loading it on native paths."""

    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_legacy(), name)


__all__ = [
    "abort_model_region_step",
    "install_model_region_executor",
    "prepare_model_region_step",
    "restore_model_region_executors",
    "restore_runtime_model_region",
    "stage_signed_backend_region_event",
]

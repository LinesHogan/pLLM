"""Generic vLLM binding for Rust-signed backend Regions.

Native workload execution reaches this module only through signed backend
events. It binds a local slot to the exact stage identity selected by Rust;
it does not classify a workload, interpret recurrence operation names, or
construct a schedule.
"""

from __future__ import annotations

from tllm.runtime.backend_region_binders import (
    BackendRegionStageContext,
    BackendRegionStagerRegistry,
)
from tllm.runtime.workload_execution import BackendObservableEvent


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

    execution = getattr(runtime, "workload_execution", None)
    if execution is None:
        return False
    stages = execution.backend_region_binder_stages(event)
    if not stages:
        return False
    registry = getattr(runtime, "backend_region_stager_registry", None)
    if registry is not None and not isinstance(registry, BackendRegionStagerRegistry):
        raise TypeError(
            "backend_region_stager_registry must be BackendRegionStagerRegistry"
        )
    raw_step_id = (
        getattr(runtime, "event_step_id", 0) if step_id is None else step_id
    )
    if type(raw_step_id) is not int or raw_step_id < 0:
        raise ValueError("step_id must be a non-negative int")
    resolved_step_id = raw_step_id
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

    del runtime, runner, num_scheduled_tokens, input_batch
    return None


def abort_model_region_step(runtime: object, step_id: int) -> bool:
    execution = getattr(runtime, "workload_execution", None)
    if execution is not None:
        from tllm.backends.vllm.typed_layer_replay import abort_typed_layer_replay

        aborted = False
        for binder in execution.backend_region_binders.values():
            aborted = bool(binder.abort(step_id)) or aborted
        return abort_typed_layer_replay(runtime) or aborted
    return False


def install_model_region_executor(*, binding: object, runtime: object) -> object:
    del binding, runtime
    return None


def restore_model_region_executors(binding: object) -> int:
    del binding
    return 0


def restore_runtime_model_region(runtime: object) -> int:
    if getattr(runtime, "workload_execution", None) is not None:
        from tllm.backends.vllm.typed_layer_replay import restore_typed_layer_replay

        return restore_typed_layer_replay(runtime)
    return 0


__all__ = [
    "abort_model_region_step",
    "install_model_region_executor",
    "prepare_model_region_step",
    "restore_model_region_executors",
    "restore_runtime_model_region",
    "stage_signed_backend_region_event",
]

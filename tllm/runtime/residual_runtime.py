#!/usr/bin/env python3
"""Residual-stream runtime host: public API and hook-facing "core" module.

This module is intentionally thin. The implementation lives in focused
modules:

* ``tllm.runtime.state`` — ``ResidualRuntimeConfig`` / ``ResidualRuntimeState``
  and the shared ``RUNTIME`` object;
* ``tllm.runtime.resources`` — sampler precompute buffers, caches, streams;
* ``tllm.runtime.stats`` — statistics containers and collectors;
* ``tllm.runtime.lifecycle`` — runner patch installation and ``make_llm``;
* ``tllm.runtime.vllm_patch.port_runtime_hooks`` — the hook bodies, driven
  with this module as the ``core`` handle.

The module-level ``GPUModelRunner`` / ``_ORIG_*`` / ``_PATCH_INSTALLED``
globals stay here because backends and hook modules address them through the
``core`` module object.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import torch

from tllm.common.state import ensure_v1_env, pick_common_attn_metadata, resolve_object_by_path
from tllm.ports.consumer import BaseConsumer
from tllm.contracts.consumer_stats import TrainingStats, merge_training_stats_maps
from tllm.contracts.request_identity import resolve_prompt_sample

# Backward-compatible alias: ESamp's stats name, canonical type is TrainingStats.
ESampStats = TrainingStats
from tllm.contracts.runtime_context import RunnerLike
from tllm.common.decode_localization import compute_decode_localization
from tllm.runtime import lifecycle as _lifecycle
from tllm.runtime import stats as _stats
from tllm.runtime.config import normalize_adaptation_stream_mode
from tllm.runtime.consumer_registry import ConsumerRegistry
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.consumer_compat import synchronize as synchronize_consumer
from tllm.runtime.resources import (
    SamplerPrecomputeBuffers,
    SamplerPrecomputeCache,
    SamplerPrecomputeState,
)
from tllm.runtime.state import (
    RUNTIME,
    ResidualRuntimeConfig,
    ResidualRuntimeState,
    RuntimeConsumer,
)
from tllm.runtime.stats import DistillerTimingStats, GraphDebugStats, PathHotspotStats, PlacementStats

if TYPE_CHECKING:
    from vllm import LLM
    from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStats

ensure_v1_env()

GPUModelRunner = None
_ORIG_LOAD_MODEL = None
_ORIG_PREPARE_INPUTS = None
_ORIG_EXECUTE_MODEL = None
_PATCH_INSTALLED = False

MODEL_HOOK_FLAG = "esamp_hook_installed"
MODEL_HOOK_SPEC_ATTR = "esamp_hook_spec"


def runtime_patch_installed() -> bool:
    return _PATCH_INSTALLED


def active_consumer_id() -> str | None:
    consumer = RUNTIME.consumer
    return None if consumer is None else str(consumer.consumer_id)



def _ensure_dispatch_registry() -> ConsumerRegistry:
    if RUNTIME.consumer_registry is None:
        RUNTIME.consumer_registry = ConsumerRegistry()
    return RUNTIME.consumer_registry


def _active_dispatch_consumers() -> list[BaseConsumer[object]]:
    consumers = _ensure_dispatch_registry().consumers()
    if isinstance(RUNTIME.consumer, BaseConsumer):
        consumers = consumers + [RUNTIME.consumer]
    return consumers


def _refresh_dispatch_plan() -> None:
    consumers = _active_dispatch_consumers()
    if not consumers:
        RUNTIME.dispatch_plan = None
        RUNTIME.execution_plan = None
        RUNTIME.producer_executor = None
        return
    plan = DispatchPlan.build(consumers)
    RUNTIME.dispatch_plan = None if plan.is_empty() else plan
    # Fail fast at startup when a consumer reads a migrated port kind that no
    # registered producer can serve (kinds still served by legacy hardcoded
    # producers are not enforced yet).
    from tllm.runtime.execution_plan import compile_default_plan

    execution_plan = compile_default_plan(consumers=consumers)
    from tllm.runtime.producer_executor import (
        ProducerExecutor,
        default_producer_materializers,
    )

    RUNTIME.execution_plan = execution_plan
    RUNTIME.producer_executor = ProducerExecutor.bind(
        plan=execution_plan,
        registry=default_producer_materializers(),
    )
    RUNTIME.placement_stats.capture_resident_stage_count = max(
        RUNTIME.placement_stats.capture_resident_stage_count,
        len(execution_plan.capture_resident_producer_ids),
)


def prepare_physical_workgraph_step(step_id: int) -> None:
    """Publish staged physical epochs at the existing prepare safe point."""

    step = int(step_id)
    if step < 0:
        raise ValueError("physical step id must be non-negative")
    previous = RUNTIME.physical_prepared_step_id
    if previous >= 0 and previous != step:
        workload_store = RUNTIME.workload_execution_store
        if workload_store is not None:
            workload_store.finish_step(previous)
            workload_store.collect_retired()
    workload_store = RUNTIME.workload_execution_store
    if workload_store is not None:
        pending_shapes = workload_store.current.peek_shape_specialization_requests()
        if pending_shapes:
            from tllm.runtime.runtime import active_runtime

            if active_runtime().apply_region_shape_specializations(pending_shapes):
                drained = workload_store.current.drain_shape_specialization_requests()
                if drained != pending_shapes:
                    raise RuntimeError(
                        "native shape specialization queue changed during safe-point compilation"
                    )
        workload_execution = workload_store.prepare_step(step)
        RUNTIME.workload_execution = workload_execution
        RUNTIME.workload_compilation = workload_execution.compilation
        RUNTIME.physical_executor = workload_execution
        RUNTIME.native_schedule_graph = workload_execution.compilation.graph
        RUNTIME.native_workloads_pending = False
        workload_store.collect_retired()
    else:
        workload_execution = RUNTIME.workload_execution
        if workload_execution is not None:
            workload_execution.prepare_step(step)
    RUNTIME.physical_prepared_step_id = step


def read_workgraph_runtime_stats() -> "WorkGraphRuntimeStats":
    """Aggregate non-resetting physical executor and deadline statistics."""

    from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStats

    snapshots: list[WorkGraphRuntimeStats] = []
    physical_executor = RUNTIME.physical_executor
    if physical_executor is not None:
        snapshots.insert(0, physical_executor.read_workgraph_runtime_stats())
    local = WorkGraphRuntimeStats(
        active_plan_epoch=max(
            (snapshot.active_plan_epoch for snapshot in snapshots),
            default=0,
        ),
        prepare_count=sum(snapshot.prepare_count for snapshot in snapshots),
        execute_count=sum(snapshot.execute_count for snapshot in snapshots),
        finish_count=sum(snapshot.finish_count for snapshot in snapshots),
        replay_count=sum(snapshot.replay_count for snapshot in snapshots),
        eager_count=sum(snapshot.eager_count for snapshot in snapshots),
        fallback_count=sum(snapshot.fallback_count for snapshot in snapshots),
        fence_wait_count=sum(snapshot.fence_wait_count for snapshot in snapshots),
        fence_record_count=sum(snapshot.fence_record_count for snapshot in snapshots),
        bucket_miss_count=sum(snapshot.bucket_miss_count for snapshot in snapshots),
        specialization_request_count=sum(
            snapshot.specialization_request_count for snapshot in snapshots
        ),
        active_leases=sum(snapshot.active_leases for snapshot in snapshots),
        retired_plans=sum(snapshot.retired_plans for snapshot in snapshots),
        deadline_flush_count=int(RUNTIME.placement_stats.deadline_flush_count),
    )
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return WorkGraphRuntimeStats.merge(
        (local, *(reply.workgraph for reply in plane.pull_stats("workgraph", sync=False)))
    )


def publish_workload_compilation(
    compilation: object | None,
    *,
    bindings: object | None = None,
) -> None:
    """Publish an immutable result and activate only fully bound native plans."""

    if compilation is not None:
        from tllm.schedule.coordinator import WorkloadCompilation

        if not isinstance(compilation, WorkloadCompilation):
            raise TypeError("workload compilation must be WorkloadCompilation or None")
        from tllm.runtime.workload_execution import build_native_workload_execution
        from tllm.runtime.workload_execution import WorkloadExecutionBindings

        if bindings is not None and not isinstance(bindings, WorkloadExecutionBindings):
            raise TypeError("workload runtime bindings must be WorkloadExecutionBindings")
        if bindings is None and RUNTIME.native_workloads_pending:
            raise RuntimeError(
                "native workload publication requires explicit bindings"
            )

        execution = build_native_workload_execution(
            compilation,
            runtime=RUNTIME,
            # A WorkloadExecutionBindings object is the complete native
            # authority contract.  Do not even instantiate the legacy
            # registry while binding it.
            consumers=() if bindings is not None else _active_dispatch_consumers(),
            bindings=bindings,
        )
    else:
        if bindings is not None:
            raise ValueError("workload runtime bindings require a compilation")
        execution = None
    from tllm.runtime.workload_execution import NativeWorkloadExecutionStore

    store = RUNTIME.workload_execution_store
    previous = RUNTIME.workload_execution
    if compilation is None:
        if store is not None:
            store.close()
        elif previous is not None:
            previous.close()
        if RUNTIME.physical_executor is previous:
            RUNTIME.physical_executor = None
        RUNTIME.workload_execution_store = None
        RUNTIME.workload_compilation = None
        RUNTIME.workload_execution = None
        return
    if execution is None:
        if store is not None:
            store.close()
        elif previous is not None:
            previous.close()
        if RUNTIME.physical_executor is previous:
            RUNTIME.physical_executor = None
        RUNTIME.workload_execution_store = None
        RUNTIME.workload_compilation = compilation
        RUNTIME.workload_execution = None
        return
    if (
        previous is not None
        and previous.plan_epoch == execution.plan_epoch
        and previous.compilation.decision.signature
        == execution.compilation.decision.signature
        and previous.artifact_report == execution.artifact_report
    ):
        execution.close()
        if store is None:
            store = NativeWorkloadExecutionStore(previous)
            RUNTIME.workload_execution_store = store
        RUNTIME.workload_compilation = compilation
        RUNTIME.workload_execution = previous
        RUNTIME.physical_executor = previous
        return
    if store is None:
        if previous is not None and previous is not execution:
            store = NativeWorkloadExecutionStore(previous)
            store.stage(execution)
            RUNTIME.workload_execution_store = store
            return
        store = NativeWorkloadExecutionStore(execution)
        RUNTIME.workload_execution_store = store
        RUNTIME.workload_compilation = compilation
        RUNTIME.workload_execution = execution
        RUNTIME.physical_executor = execution
        return
    store.stage(execution)


def advance_workload_anchor(anchor: str) -> bool:
    """Advance the native plan at one backend-exported observable anchor."""

    execution = RUNTIME.workload_execution
    if execution is None:
        return False
    return execution.advance_anchor(str(anchor))


def register_dispatch_consumer(consumer: BaseConsumer[object]) -> None:
    _ensure_dispatch_registry().register(consumer)
    _refresh_dispatch_plan()


def replace_dispatch_consumers(consumers: Sequence[BaseConsumer[Any]]) -> None:
    _ensure_dispatch_registry().replace_all(consumers)
    _refresh_dispatch_plan()


def clear_dispatch_consumers() -> None:
    _ensure_dispatch_registry().clear()
    _refresh_dispatch_plan()


def set_runtime_consumer(consumer: BaseConsumer[Any] | None) -> None:
    RUNTIME.consumer = consumer
    _refresh_dispatch_plan()


def runner_uses_compilation_or_cudagraph(runner: RunnerLike) -> bool:
    """Legacy predicate kept for call sites; semantics now live in
    ``tllm.runtime.graph_policy.resolve_graph_mode`` (PIECEWISE/FULL ->
    True)."""
    from tllm.runtime import graph_policy as _graph_policy

    mode = _graph_policy.resolve_graph_mode(runner)
    if mode != _graph_policy.GraphMode.EAGER:
        return True

    from tllm.runtime.vllm_adapter import legacy_runner_graph_enabled

    return legacy_runner_graph_enabled(runner)


def infer_hidden_dtype(layer: torch.nn.Module) -> torch.dtype:
    for p in layer.parameters():
        return p.dtype
    for b in layer.buffers():
        return b.dtype
    return torch.float32


def resolve_module_by_path_with_fallback(model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
    from tllm.common.path_resolution import candidate_capture_paths

    errors: list[str] = []
    for candidate in candidate_capture_paths(path):
        try:
            obj = resolve_object_by_path(model, candidate)
        except RuntimeError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if not isinstance(obj, torch.nn.Module):
            errors.append(f"{candidate}: resolved to non-module {type(obj).__name__}")
            continue
        return obj, candidate

    details = "; ".join(errors) if errors else "no candidate"
    raise RuntimeError(
        f"Cannot resolve layer path `{path}` from model root `{type(model).__name__}`. Tried: {details}"
    )


_runner_uses_compilation_or_cudagraph = runner_uses_compilation_or_cudagraph
_infer_hidden_dtype = infer_hidden_dtype
_resolve_module_by_path_with_fallback = resolve_module_by_path_with_fallback


def _resolve_prompt_sample_for_req_id(req_id: str) -> tuple[int, int]:
    return resolve_prompt_sample(
        req_id,
        prompt_indices=RUNTIME.reqid_to_promptidx,
        sample_indices=RUNTIME.reqid_to_sampleidx,
    )


def configure_runtime(
    *,
    graph_scratch_rows: int,
    tap_layer_paths: Sequence[str],
    source_layer_path: str,
    target_layer_path: str,
    adaptation_pipeline_slots: int = 4,
    adaptation_stream_mode: str = "dual",
    adaptation_stream_priority: int = 0,
    compact_capture_lane: bool = False,
) -> None:
    """Configure runtime-owned mechanisms (tap topology, streams, capture lanes).

    Consumer-algorithm settings (distiller dims/lr/beta, model bank policy,
    training toggles, tracing) are NOT accepted here — they belong to each
    consumer's own config, applied via the
    consumer's support workflow.
    """
    cfg = RUNTIME.config
    cfg.graph_scratch_rows = max(0, int(graph_scratch_rows))
    cfg.tap_layer_paths = [str(path).strip() for path in tap_layer_paths if str(path).strip()]
    cfg.source_layer_path = str(source_layer_path).strip()
    cfg.target_layer_path = str(target_layer_path).strip()
    cfg.adaptation_pipeline_slots = max(1, int(adaptation_pipeline_slots))
    cfg.adaptation_stream_mode = normalize_adaptation_stream_mode(adaptation_stream_mode)
    cfg.adaptation_stream_priority = int(adaptation_stream_priority)
    cfg.compact_capture_lane = bool(compact_capture_lane)

    if not cfg.source_layer_path:
        raise RuntimeError("source_layer_path must be non-empty")
    if not cfg.target_layer_path:
        raise RuntimeError("target_layer_path must be non-empty")

    from tllm.runtime.ports.residual_bindings import default_raw_paths_from_config

    RUNTIME.residual_raw_paths = default_raw_paths_from_config(cfg)
    precompute = RUNTIME.sampler_precompute
    precompute.timing_enabled = os.getenv("TLLM_TRACE_DISTILLER_TIMING", "") == "1"
    precompute.precompute_event_pairs = []
    precompute.wait_event_pairs = []
    precompute.fallback_event_pairs = []
    precompute.port_publish_attempt_count = 0
    precompute.port_publish_hit_count = 0
    precompute.schedule_attempt_count = 0
    precompute.schedule_hit_count = 0
    RUNTIME.path_hotspot_enabled = os.getenv("TLLM_TRACE_PATH_HOTSPOTS", "") == "1"
    RUNTIME.path_hotspot_cpu_ms = {}
    RUNTIME.path_hotspot_counts = {}
    _refresh_dispatch_plan()
    RUNTIME.reset_step()


def set_esamp_training_enabled(enabled: bool) -> None:
    """Toggle legacy ESamp directly or publish native ESamp through an epoch."""
    from tllm.runtime.runtime import active_runtime

    runtime = active_runtime()
    if RUNTIME.consumer is not None:
        RUNTIME.consumer.set_enabled(enabled)
        _refresh_dispatch_plan()
    elif runtime.has_native_workloads and (
        RUNTIME.workload_execution is not None
        or RUNTIME.workload_execution_store is not None
    ):
        from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec
        from tllm.runtime.mp.control import active_control_plane

        runtime.transition_native_workload_enabled(
            bool(enabled), plane=active_control_plane()
        )
        refresh_exported_bootstrap_spec()
        return
    elif RUNTIME.workload_execution is not None:
        raise RuntimeError(
            "native ESamp enable changes require the owning coordinator runtime"
        )
    from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec
    from tllm.runtime.mp.control import active_control_plane

    refresh_exported_bootstrap_spec()
    plane = active_control_plane()
    if plane is not None:
        plane.apply_enable_flags({"esamp_training_enabled": bool(enabled)})


def synchronize_esamp() -> None:
    if RUNTIME.consumer is not None:
        synchronize_consumer(RUNTIME.consumer)
    elif RUNTIME.workload_execution is not None:
        RUNTIME.workload_execution.synchronize()
    registry = RUNTIME.consumer_registry
    if registry is not None:
        for consumer in registry.consumers():
            synchronize_consumer(consumer)

    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is not None:
        plane.synchronize()


def read_and_reset_esamp_stats(sync: bool = True) -> TrainingStats:
    if RUNTIME.consumer is not None:
        local = RUNTIME.consumer.read_and_reset_stats(sync=sync)
    elif RUNTIME.workload_execution is not None:
        local = RUNTIME.workload_execution.read_and_reset_stats(sync=sync)
    else:
        local = TrainingStats(loss_avg=0.0, loss_count=0)
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return TrainingStats.merge([local, *(reply.esamp for reply in plane.pull_stats("esamp", sync=sync))])


def read_and_reset_esamp_per_request_stats(sync: bool = True) -> Dict[int, TrainingStats]:
    if RUNTIME.consumer is not None:
        local = dict(RUNTIME.consumer.read_and_reset_per_request_stats(sync=sync))
    elif RUNTIME.workload_execution is not None:
        local = dict(
            RUNTIME.workload_execution.read_and_reset_per_request_stats(sync=sync)
        )
    else:
        local = {}
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return merge_training_stats_maps(
        [local, *(reply.esamp_per_request for reply in plane.pull_stats("esamp_per_request", sync=sync))]
    )


def read_and_reset_distiller_timing_stats(sync: bool = True) -> DistillerTimingStats:
    local = _stats.read_and_reset_distiller_timing_stats(RUNTIME, sync=sync)
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return DistillerTimingStats.merge(
        [local, *(reply.distiller_timing for reply in plane.pull_stats("distiller_timing", sync=sync))]
    )


def record_path_hotspot_cpu(name: str, ms: float) -> None:
    _stats.record_path_hotspot_cpu(RUNTIME, name, ms)


def read_and_reset_path_hotspot_stats(sync: bool = True) -> PathHotspotStats:
    local = _stats.read_and_reset_path_hotspot_stats(RUNTIME, sync=sync)
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return PathHotspotStats.merge(
        [local, *(reply.path_hotspot for reply in plane.pull_stats("path_hotspot", sync=sync))]
    )


def read_and_reset_placement_stats() -> PlacementStats:
    local = _stats.read_and_reset_placement_stats(RUNTIME)
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return PlacementStats.merge([local, *(reply.placement for reply in plane.pull_stats("placement"))])


def read_graph_debug_stats(*, mode: str = "model_bank") -> GraphDebugStats:
    return _stats.read_graph_debug_stats(RUNTIME.consumer, mode=mode)





def _wrapped_load_model(self: object, *args: object, **kwargs: object) -> object:
    from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

    return _hooks.wrapped_load_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def _wrapped_prepare_inputs(self: object, scheduler_output: object) -> object:
    from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

    return _hooks.wrapped_prepare_inputs(core=sys.modules[__name__], runner=self, scheduler_output=scheduler_output)


def _wrapped_execute_model(self: object, *args: object, **kwargs: object) -> object:
    from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

    return _hooks.wrapped_execute_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def install_runner_patch() -> None:
    _lifecycle.install_runner_patch(sys.modules[__name__])


def make_llm(*args: object, **kwargs: object) -> "LLM":
    return _lifecycle.make_llm(sys.modules[__name__], *args, **kwargs)


__all__ = [
    "DistillerTimingStats",
    "ESampStats",
    "TrainingStats",
    "GraphDebugStats",
    "PathHotspotStats",
    "PlacementStats",
    "MODEL_HOOK_FLAG",
    "MODEL_HOOK_SPEC_ATTR",
    "RUNTIME",
    "ResidualRuntimeConfig",
    "ResidualRuntimeState",
    "RuntimeConsumer",
    "SamplerPrecomputeBuffers",
    "SamplerPrecomputeCache",
    "SamplerPrecomputeState",
    "clear_dispatch_consumers",
    "compute_decode_localization",
    "configure_runtime",
    "infer_hidden_dtype",
    "make_llm",
    "pick_common_attn_metadata",
    "read_and_reset_distiller_timing_stats",
    "read_and_reset_path_hotspot_stats",
    "read_and_reset_placement_stats",
    "read_graph_debug_stats",
    "read_and_reset_esamp_per_request_stats",
    "read_and_reset_esamp_stats",
    "register_dispatch_consumer",
    "replace_dispatch_consumers",
    "resolve_module_by_path_with_fallback",
    "record_path_hotspot_cpu",
    "runner_uses_compilation_or_cudagraph",
    "set_runtime_consumer",
    "set_esamp_training_enabled",
    "synchronize_esamp",
]

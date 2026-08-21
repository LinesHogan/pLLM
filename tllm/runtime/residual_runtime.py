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
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import torch

from tllm.common.state import ensure_v1_env, pick_common_attn_metadata, resolve_object_by_path
from tllm.consumers.base import BaseConsumer
from tllm.contracts.consumer_stats import TrainingStats, merge_training_stats_maps
from tllm.contracts.request_identity import resolve_prompt_sample

# Backward-compatible alias: ESamp's stats name, canonical type is TrainingStats.
ESampStats = TrainingStats
from tllm.contracts.runtime_context import RunnerLike
from tllm.producer.decode import compute_decode_localization
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
from tllm.runtime.state import RUNTIME, ResidualRuntimeConfig, ResidualRuntimeState, RuntimeConsumer
from tllm.runtime.stats import DistillerTimingStats, GraphDebugStats, PathHotspotStats, PlacementStats
from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

if TYPE_CHECKING:
    from tllm.workgraph.anchors import BackendResolution
    from tllm.workgraph.frontend import CapturedFlow
    from tllm.workgraph.hierarchy import CaptureAttachment
    from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStats
    from tllm.runtime.specialization import RuntimeSpecializationSpec
    from tllm.runtime.scheduling.phase3_transfer import (
        CopyHandle,
        PlacementRuntimeBinding,
        TensorLike,
    )
    from tllm.runtime.scheduling.phase4_collective import (
        CollectiveHandle,
        DistributedRuntimeBinding,
    )
    from tllm.runtime.scheduling.phase5_adaptive import AdaptiveRuntimeBinding
    from tllm.workgraph.ir import NodeKey
    from tllm.workgraph.placement import ValueKey
    from tllm.workgraph.distributed import CollectiveKey

ensure_v1_env()

GPUModelRunner = None
_ORIG_LOAD_MODEL = None
_ORIG_PREPARE_INPUTS = None
_ORIG_EXECUTE_MODEL = None
_PATCH_INSTALLED = False

MODEL_HOOK_FLAG = "esamp_hook_installed"
MODEL_HOOK_SPEC_ATTR = "esamp_hook_spec"


def _ensure_runner_binding() -> None:
    _lifecycle.ensure_runner_binding(sys.modules[__name__])


def _ensure_dispatch_registry() -> ConsumerRegistry:
    if RUNTIME.consumer_registry is None:
        RUNTIME.consumer_registry = ConsumerRegistry()
    return RUNTIME.consumer_registry


def _active_dispatch_consumers() -> list[BaseConsumer[object]]:
    consumers = _ensure_dispatch_registry().consumers()
    if isinstance(RUNTIME.consumer, BaseConsumer):
        consumers = consumers + [RUNTIME.consumer]
    return consumers


def _collect_workgraph_capture_metadata(
    consumers: Sequence[BaseConsumer[object]],
) -> tuple[tuple["CapturedFlow", ...], tuple["CaptureAttachment", ...]]:
    captures = tuple(
        capture
        for consumer in consumers
        for capture in consumer.workgraph_captures()
    )
    attachments = tuple(
        attachment
        for consumer in consumers
        for attachment in consumer.workgraph_attachments()
    )
    return captures, attachments


def _refresh_dispatch_plan() -> None:
    consumers = _active_dispatch_consumers()
    if not consumers:
        RUNTIME.dispatch_plan = None
        RUNTIME.execution_plan = None
        RUNTIME.workgraph_snapshot = None
        RUNTIME.workgraph_captures = ()
        RUNTIME.workgraph_attachments = ()
        RUNTIME.workgraph_refresh_pending = False
        RUNTIME.physical_plan_store = None
        _clear_physical_capture_regions()
        RUNTIME.producer_executor = None
        return
    captures, attachments = _collect_workgraph_capture_metadata(consumers)
    RUNTIME.workgraph_captures = captures
    RUNTIME.workgraph_attachments = attachments
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
    # Shadow-only compatibility metadata. The runtime dispatch path continues
    # to consume DispatchPlan/ExecutionPlan directly and never reads this
    # snapshot from a per-step hook.  Once a typed backend has been resolved,
    # retain the previous concrete snapshot until the replacement can be
    # compiled.  Persisting an unresolved intermediate would make one logical
    # plan rebuild consume two reconciliation epochs.
    if RUNTIME.dispatch_plan is None:
        RUNTIME.workgraph_snapshot = None
        RUNTIME.workgraph_refresh_pending = False
    elif RUNTIME.workgraph_backend_resolution is None:
        from tllm.workgraph.compiler import compile_shadow_snapshot

        RUNTIME.workgraph_snapshot = compile_shadow_snapshot(
            consumers=consumers,
            dispatch_plan=plan,
            execution_plan=execution_plan,
            previous=RUNTIME.workgraph_snapshot,
            captures=captures,
            attachments=attachments,
        )
        RUNTIME.workgraph_refresh_pending = True
    elif not plan.requires_graph_mode_refresh(RUNTIME.workgraph_backend_resolution.graph_mode):
        RUNTIME.workgraph_refresh_pending = True
        _compile_concrete_workgraph_snapshot()
    else:
        RUNTIME.workgraph_refresh_pending = True
    RUNTIME.producer_executor = ProducerExecutor.bind(
        plan=execution_plan,
        registry=default_producer_materializers(),
    )
    RUNTIME.placement_stats.capture_resident_stage_count = max(
        RUNTIME.placement_stats.capture_resident_stage_count,
        len(execution_plan.capture_resident_producer_ids),
    )


def _compile_concrete_workgraph_snapshot() -> None:
    """Compile one concrete shadow snapshot from already-authoritative plans.

    This cold path deliberately has no access to runner instances or runtime
    payloads. ``DispatchPlan`` and ``ExecutionPlan`` remain the only inputs
    used to describe scheduling; this result is never consulted for dispatch.
    """

    resolution = RUNTIME.workgraph_backend_resolution
    plan = RUNTIME.dispatch_plan
    execution_plan = RUNTIME.execution_plan
    if resolution is None:
        raise RuntimeError("WorkGraph refresh requires a resolved backend context")
    if plan is None or execution_plan is None:
        raise RuntimeError("WorkGraph refresh requires DispatchPlan and ExecutionPlan")

    from tllm.workgraph.anchors import backend_snapshot_from_resolution
    from tllm.workgraph.compiler import compile_phase1a

    # ``lower_plans`` binds the existing authoritative plans directly; it
    # intentionally does not inspect consumer instances at this boundary.
    RUNTIME.workgraph_snapshot = compile_phase1a(
        consumers=(),
        dispatch_plan=plan,
        execution_plan=execution_plan,
        backend=backend_snapshot_from_resolution(resolution),
        previous=RUNTIME.workgraph_snapshot,
        captures=RUNTIME.workgraph_captures,
        attachments=RUNTIME.workgraph_attachments,
    )
    _stage_physical_workgraph_snapshot()
    RUNTIME.workgraph_refresh_pending = False


def _stage_physical_workgraph_snapshot() -> None:
    """Lower and stage one concrete physical descriptor snapshot.

    Runtime streams, events, leases, and graphs are attached by the executor
    resource factory when an executable region is activated.  Publishing the
    frozen plan here keeps compilation off the step path and gives every step
    a stable epoch even while those resources are still legacy-owned.
    """

    snapshot = RUNTIME.workgraph_snapshot
    if snapshot is None:
        RUNTIME.physical_plan_store = None
        _clear_physical_capture_regions()
        return
    backend = snapshot.backend
    if backend is None or backend.graph_mode == "full":
        # FULL remains a typed Phase 1 diagnostic and an explicit legacy
        # boundary.  It must never acquire a partially usable physical plan.
        RUNTIME.physical_plan_store = None
        _clear_physical_capture_regions()
        return

    from tllm.runtime.scheduling.plan_store import PlanResources, PlanSnapshotStore
    from tllm.workgraph.physical import compile_physical_flow_plan

    physical_plan = compile_physical_flow_plan(snapshot)
    placement_plan = None
    placement_dispatcher = None
    placement_implementations = ()
    adaptive_plan = None
    adaptive_scheduler = None
    adaptive_flow_bindings = ()
    placement_binding = RUNTIME.placement_runtime_binding
    if placement_binding is not None:
        (
            placement_plan,
            placement_dispatcher,
            placement_implementations,
        ) = placement_binding.compile(physical_plan)
    adaptive_binding = RUNTIME.adaptive_runtime_binding
    if adaptive_binding is not None:
        if placement_plan is None:
            raise RuntimeError(
                "adaptive WorkGraph compilation requires a Phase 3 placement binding"
            )
        (
            adaptive_plan,
            adaptive_scheduler,
            adaptive_flow_bindings,
        ) = adaptive_binding.compile(placement_plan)
    resources = PlanResources(
        plan=physical_plan,
        placement_plan=placement_plan,
        placement_dispatcher=placement_dispatcher,
        placement_implementations=placement_implementations,
        adaptive_plan=adaptive_plan,
        adaptive_scheduler=adaptive_scheduler,
        adaptive_flow_bindings=adaptive_flow_bindings,
    )
    store = RUNTIME.physical_plan_store
    if store is None:
        RUNTIME.physical_plan_store = PlanSnapshotStore(resources)
        _activate_physical_capture_regions(snapshot)
        return
    current_epoch = store.current.plan.plan_epoch
    if resources.plan.plan_epoch <= current_epoch:
        _activate_physical_capture_regions(snapshot)
        return
    store.stage(resources)
    _activate_physical_capture_regions(snapshot)


def _activate_physical_capture_regions(snapshot: object) -> None:
    """Publish migrated regions through the typed consumer interface."""

    from tllm.workgraph.compiler import Phase1aPlan
    from tllm.workgraph.ir import NodeKey
    from tllm.workgraph.physical import compile_physical_capture_region

    if not isinstance(snapshot, Phase1aPlan):
        raise TypeError("physical capture activation requires Phase1aPlan")
    hierarchy = snapshot.hierarchy
    if hierarchy is None:
        return
    from tllm.workgraph.frontend import CaptureStatus

    executable_artifacts = {
        root.artifact_key
        for root in hierarchy.roots
        if root.status == CaptureStatus.CAPTURED and len(root.children) == 1
    }
    consumers = _active_dispatch_consumers()
    for consumer in consumers:
        declared_executors = consumer.workgraph_region_executors()
        executor_descs = {
            descriptor.artifact_key: descriptor
            for descriptor in declared_executors
        }
        if len(executor_descs) != len(declared_executors):
            raise ValueError("consumer declares duplicate WorkGraph region executors")
        owned_nodes = {
            NodeKey.consumer_flow(consumer.consumer_id, ordinal)
            for ordinal, _flow in enumerate(consumer.flows())
        }
        owned_attachments = tuple(
            attachment
            for attachment in hierarchy.attachments
            if attachment.node_key in owned_nodes
            and attachment.artifact_key in executable_artifacts
        )
        owned_artifacts = {attachment.artifact_key for attachment in owned_attachments}
        undeclared_artifacts = set(executor_descs).difference(owned_artifacts)
        if undeclared_artifacts:
            raise ValueError(
                "consumer declares WorkGraph executors without matching attachments"
            )
        executable_attachments = tuple(
            attachment
            for attachment in owned_attachments
            if attachment.artifact_key in executor_descs
        )
        regions = tuple(
            compile_physical_capture_region(
                snapshot,
                attachment.artifact_key,
                executor_descs[attachment.artifact_key],
            )
            for attachment in executable_attachments
        )
        placement_binding = RUNTIME.placement_runtime_binding
        if placement_binding is None:
            consumer.activate_workgraph_regions(regions)
            continue
        from tllm.runtime.scheduling.plan_store import PlanResources

        placement_resources: list[PlanResources] = []
        for region in regions:
            placement_plan, dispatcher, implementations = placement_binding.compile(
                region.plan,
                value_flows=(),
            )
            placement_resources.append(
                PlanResources(
                    plan=region.plan,
                    placement_plan=placement_plan,
                    placement_dispatcher=dispatcher,
                    placement_implementations=implementations,
                )
            )
        consumer.activate_workgraph_region_resources(
            regions,
            tuple(placement_resources),
        )


def _clear_physical_capture_regions() -> None:
    for consumer in _active_dispatch_consumers():
        consumer.activate_workgraph_regions(())


def prepare_physical_workgraph_step(step_id: int) -> None:
    """Publish staged physical epochs at the existing prepare safe point."""

    step = int(step_id)
    if step < 0:
        raise ValueError("physical WorkGraph step id must be non-negative")
    previous = RUNTIME.physical_prepared_step_id
    distributed_previous = RUNTIME.distributed_prepared_step_id
    distributed_store = RUNTIME.distributed_plan_store
    if distributed_previous >= 0 and distributed_previous != step:
        if distributed_store is not None:
            distributed_store.poll()
            distributed_store.finish_step(distributed_previous)
            distributed_store.collect_retired()
        RUNTIME.distributed_prepared_step_id = -1
    store = RUNTIME.physical_plan_store
    if previous >= 0 and previous != step:
        if store is not None:
            store.active(previous).poll_transfers()
            store.finish_step(previous)
            store.collect_retired()
        for consumer in _active_dispatch_consumers():
            consumer.finish_workgraph_step(previous)
    for consumer in _active_dispatch_consumers():
        consumer.drain_workgraph_specializations()
    if store is not None:
        resources = store.prepare_step(step)
        if resources is not None:
            resources.poll_transfers()
            resources.reset_adaptive_reservations()
    for consumer in _active_dispatch_consumers():
        consumer.prepare_workgraph_step(step)
    if distributed_store is not None:
        distributed_store.prepare_step(step)
        RUNTIME.distributed_prepared_step_id = step
    RUNTIME.physical_prepared_step_id = step


def prepare_workgraph_step(step_id: int) -> None:
    """Compatibility-free public name used by runner safe-point adapters."""

    prepare_physical_workgraph_step(step_id)


def configure_workgraph_placement(
    binding: "PlacementRuntimeBinding | None",
) -> None:
    """Install or remove typed Phase 3 cold-compilation inputs."""

    from tllm.runtime.scheduling.phase3_transfer import PlacementRuntimeBinding

    if binding is not None and not isinstance(binding, PlacementRuntimeBinding):
        raise TypeError(
            "configure_workgraph_placement binding must be PlacementRuntimeBinding"
        )
    if binding is None and RUNTIME.adaptive_runtime_binding is not None:
        raise RuntimeError(
            "cannot remove WorkGraph placement while adaptive planning is configured"
        )
    store = RUNTIME.physical_plan_store
    if store is not None and not store.closeable:
        raise RuntimeError(
            "cannot replace WorkGraph placement binding while an epoch is active"
        )
    _clear_physical_capture_regions()
    RUNTIME.placement_runtime_binding = binding
    RUNTIME.physical_plan_store = None
    RUNTIME.physical_prepared_step_id = -1
    if RUNTIME.workgraph_snapshot is not None:
        _stage_physical_workgraph_snapshot()


def configure_workgraph_adaptive(
    binding: "AdaptiveRuntimeBinding | None",
) -> None:
    """Install typed Phase 5 cold-compilation inputs and explicit flow bindings."""

    from tllm.runtime.scheduling.phase5_adaptive import AdaptiveRuntimeBinding

    if binding is not None and not isinstance(binding, AdaptiveRuntimeBinding):
        raise TypeError(
            "configure_workgraph_adaptive binding must be AdaptiveRuntimeBinding"
        )
    if binding is not None and RUNTIME.placement_runtime_binding is None:
        raise RuntimeError(
            "adaptive WorkGraph configuration requires a placement binding"
        )
    store = RUNTIME.physical_plan_store
    if store is not None and not store.closeable:
        raise RuntimeError(
            "cannot replace WorkGraph adaptive binding while an epoch is active"
        )
    RUNTIME.adaptive_runtime_binding = binding
    RUNTIME.physical_plan_store = None
    RUNTIME.physical_prepared_step_id = -1
    if RUNTIME.workgraph_snapshot is not None:
        _stage_physical_workgraph_snapshot()


def configure_workgraph_distributed(
    binding: "DistributedRuntimeBinding | None",
) -> None:
    """Install typed Phase 4 topology and rank-local runtime bindings."""

    from tllm.runtime.scheduling.phase4_collective import DistributedRuntimeBinding

    if binding is not None and not isinstance(binding, DistributedRuntimeBinding):
        raise TypeError(
            "configure_workgraph_distributed binding must be DistributedRuntimeBinding"
        )
    store = RUNTIME.distributed_plan_store
    if store is not None and not store.closeable:
        raise RuntimeError(
            "cannot replace WorkGraph distributed binding while an epoch is active"
        )
    RUNTIME.distributed_runtime_binding = binding
    RUNTIME.distributed_plan_store = None
    RUNTIME.distributed_prepared_step_id = -1


def publish_workgraph_distributed(*, plan_epoch: int) -> None:
    """Compile and stage one distributed plan on the explicit cold path."""

    from tllm.runtime.scheduling.phase4_collective import (
        DistributedPlanResources,
        DistributedPlanSnapshotStore,
    )

    binding = RUNTIME.distributed_runtime_binding
    if binding is None:
        raise RuntimeError("distributed WorkGraph publication requires a binding")
    plan, executor = binding.compile(plan_epoch)
    resources = DistributedPlanResources(plan, executor)
    store = RUNTIME.distributed_plan_store
    if store is None:
        RUNTIME.distributed_plan_store = DistributedPlanSnapshotStore(resources)
    else:
        store.stage(resources)


def launch_workgraph_collective(
    step_id: int,
    *,
    collective: "CollectiveKey",
    payload: object,
) -> "CollectiveHandle":
    """Launch only the next collective approved by the prepared rank plan."""

    step = int(step_id)
    if step != RUNTIME.distributed_prepared_step_id:
        raise RuntimeError("collective launch requires the currently prepared step")
    store = RUNTIME.distributed_plan_store
    if store is None:
        raise RuntimeError("collective launch requires a distributed plan")
    return store.executor(step).launch(
        step,
        collective=collective,
        payload=payload,
    )


def poll_workgraph_collectives() -> int:
    """Poll Python-owned async collective fences without global synchronization."""

    store = RUNTIME.distributed_plan_store
    if store is None:
        return 0
    completed = store.poll()
    store.collect_retired()
    return completed


def finish_workgraph_distributed_step(step_id: int) -> None:
    """Close one distributed step after its full frozen local sequence launched."""

    step = int(step_id)
    store = RUNTIME.distributed_plan_store
    if store is None:
        return
    store.poll()
    store.finish_step(step)
    store.collect_retired()
    if RUNTIME.distributed_prepared_step_id == step:
        RUNTIME.distributed_prepared_step_id = -1


def dispatch_workgraph_value(
    *,
    producer: "NodeKey",
    consumer: "NodeKey",
    value: "ValueKey",
    source: "TensorLike",
    destination: "TensorLike",
) -> "CopyHandle":
    """Submit one transfer from the placement snapshot acquired for this step."""

    store = RUNTIME.physical_plan_store
    step = RUNTIME.physical_prepared_step_id
    if store is None or step < 0:
        raise RuntimeError("WorkGraph placement dispatch requires a prepared step")
    resources = store.active(step)
    dispatcher = resources.placement_dispatcher
    if dispatcher is None:
        raise RuntimeError("prepared WorkGraph epoch has no placement dispatcher")
    return dispatcher.submit(
        producer=producer,
        consumer=consumer,
        value=value,
        source=source,
        destination=destination,
    )


def poll_workgraph_transfers() -> None:
    """Poll asynchronous transfers owned by the currently prepared epoch."""

    store = RUNTIME.physical_plan_store
    step = RUNTIME.physical_prepared_step_id
    if store is None or step < 0:
        return
    store.active(step).poll_transfers()


def read_workgraph_runtime_stats() -> "WorkGraphRuntimeStats":
    """Aggregate non-resetting physical executor and deadline statistics."""

    from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStats

    snapshots = [
        consumer.read_workgraph_runtime_stats()
        for consumer in _active_dispatch_consumers()
    ]
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


def record_workgraph_backend_resolution(*, resolution: "BackendResolution") -> None:
    """Record one typed backend resolution and refresh non-placement plans.

    Drivers call this only after vLLM has resolved its effective graph mode.
    Plans whose fusion placement depends on that mode wait for the existing
    dispatch-side placement refresh so its counters and behavior remain
    unchanged.
    """

    previous_resolution = RUNTIME.workgraph_backend_resolution
    RUNTIME.workgraph_backend_resolution = resolution
    plan = RUNTIME.dispatch_plan
    if plan is None:
        return
    if previous_resolution == resolution and not RUNTIME.workgraph_refresh_pending:
        return
    if plan.requires_graph_mode_refresh(resolution.graph_mode):
        RUNTIME.workgraph_refresh_pending = True
        return
    RUNTIME.workgraph_refresh_pending = True
    # Placement is already current for this graph mode.  A changed typed
    # backend descriptor therefore needs only the cold shadow refresh, not a
    # second dispatch-side placement transition.
    _compile_concrete_workgraph_snapshot()


def finalize_workgraph_snapshot_after_placement_refresh() -> None:
    """Finish the one-time refresh after existing placement/fusion work."""

    if RUNTIME.workgraph_refresh_pending:
        _compile_concrete_workgraph_snapshot()


def refresh_workgraph_capture_metadata() -> None:
    """Publish cold-path consumer captures without rebuilding runtime plans."""

    consumers = _active_dispatch_consumers()
    captures, attachments = _collect_workgraph_capture_metadata(consumers)
    if (
        captures == RUNTIME.workgraph_captures
        and attachments == RUNTIME.workgraph_attachments
    ):
        return

    RUNTIME.workgraph_captures = captures
    RUNTIME.workgraph_attachments = attachments
    plan = RUNTIME.dispatch_plan
    execution_plan = RUNTIME.execution_plan
    if plan is None or execution_plan is None:
        RUNTIME.workgraph_snapshot = None
        RUNTIME.workgraph_refresh_pending = False
        return

    resolution = RUNTIME.workgraph_backend_resolution
    if resolution is None:
        from tllm.workgraph.compiler import compile_shadow_snapshot

        RUNTIME.workgraph_snapshot = compile_shadow_snapshot(
            consumers=consumers,
            dispatch_plan=plan,
            execution_plan=execution_plan,
            previous=RUNTIME.workgraph_snapshot,
            captures=captures,
            attachments=attachments,
        )
        RUNTIME.workgraph_refresh_pending = True
        return

    RUNTIME.workgraph_refresh_pending = True
    if not plan.requires_graph_mode_refresh(resolution.graph_mode):
        _compile_concrete_workgraph_snapshot()


def cold_specialize_workgraph(spec: "RuntimeSpecializationSpec") -> None:
    """Publish backend-derived capture examples before the first runtime step."""

    from tllm.runtime.specialization import RuntimeSpecializationSpec

    if not isinstance(spec, RuntimeSpecializationSpec):
        raise TypeError("cold WorkGraph specialization requires RuntimeSpecializationSpec")
    for consumer in _active_dispatch_consumers():
        consumer.cold_specialize_workgraph(spec)
    refresh_workgraph_capture_metadata()


def register_dispatch_consumer(consumer: BaseConsumer) -> None:
    _ensure_dispatch_registry().register(consumer)
    _refresh_dispatch_plan()


def replace_dispatch_consumers(consumers: Sequence[BaseConsumer]) -> None:
    _ensure_dispatch_registry().replace_all(consumers)
    _refresh_dispatch_plan()


def clear_dispatch_consumers() -> None:
    _ensure_dispatch_registry().clear()
    _refresh_dispatch_plan()


def set_runtime_consumer(consumer: BaseConsumer | None) -> None:
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

    # Structural fallback for runners that only expose the legacy flags.
    use_cuda_graph = getattr(runner, "use_cuda_graph", None)
    if use_cuda_graph is True:
        return True

    cfg = getattr(runner, "compilation_config", None)
    if cfg is None:
        return False

    level_raw = getattr(cfg, "level", 0)
    try:
        level = int(level_raw)
    except Exception:
        level = 0
    return level > 0


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


    return False


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
    fast_weight_enabled: bool = False,
    fast_weight_layers: Sequence[int] = (0,),
    fast_weight_chunk_size: int = 4096,
    fast_weight_lr: float = 1e-3,
    fast_weight_mode: str = "exact",
    fast_weight_decode_update_enabled: bool = False,
    fast_weight_key_mode: str = "static",
    fast_weight_static_request_id: str = "__tllm_fast_weight_static__",
    fast_weight_max_slots: int = 64,
    fast_weight_max_bytes: int = 0,
    fast_weight_fail_on_unsupported: bool = False,
) -> None:
    """Configure runtime-owned mechanisms (tap topology, streams, capture lanes).

    Consumer-algorithm settings (distiller dims/lr/beta, model bank policy,
    training toggles, tracing) are NOT accepted here — they belong to each
    consumer's own config (e.g. ``ESampConsumerConfig``), applied via the
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
    old_fast_weight_enabled = bool(cfg.fast_weight_enabled)
    cfg.fast_weight_enabled = bool(fast_weight_enabled)
    if old_fast_weight_enabled and not cfg.fast_weight_enabled:
        from tllm.runtime.fast_weight.runtime import restore_fast_weight_wrappers

        restore_fast_weight_wrappers(RUNTIME)
    cfg.fast_weight_layers = tuple(int(layer) for layer in fast_weight_layers)
    cfg.fast_weight_chunk_size = max(1, int(fast_weight_chunk_size))
    cfg.fast_weight_lr = float(fast_weight_lr)
    from tllm.runtime.fast_weight.state import normalize_fast_weight_mode

    cfg.fast_weight_mode = normalize_fast_weight_mode(fast_weight_mode)
    cfg.fast_weight_decode_update_enabled = bool(fast_weight_decode_update_enabled)
    cfg.fast_weight_key_mode = str(fast_weight_key_mode).strip() or "static"
    cfg.fast_weight_static_request_id = (
        str(fast_weight_static_request_id).strip() or "__tllm_fast_weight_static__"
    )
    cfg.fast_weight_max_slots = max(0, int(fast_weight_max_slots))
    cfg.fast_weight_max_bytes = max(0, int(fast_weight_max_bytes))
    cfg.fast_weight_fail_on_unsupported = bool(fast_weight_fail_on_unsupported)

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
    """Toggle training on the active consumer (config lives on the consumer)."""
    if RUNTIME.consumer is not None:
        RUNTIME.consumer.set_enabled(enabled)
        _refresh_dispatch_plan()
    from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec
    from tllm.runtime.mp.control import active_control_plane

    refresh_exported_bootstrap_spec()
    plane = active_control_plane()
    if plane is not None:
        plane.apply_enable_flags({"esamp_training_enabled": bool(enabled)})


def set_test_time_lora_enabled(enabled: bool) -> None:
    RUNTIME.test_time_lora_enabled = bool(enabled)
    from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec
    from tllm.runtime.mp.control import active_control_plane

    refresh_exported_bootstrap_spec()
    plane = active_control_plane()
    if plane is not None:
        plane.apply_enable_flags({"test_time_lora_enabled": bool(enabled)})


def set_fast_weight_enabled(
    enabled: bool,
    *,
    layers: Sequence[int] | None = None,
    chunk_size: int | None = None,
    learning_rate: float | None = None,
    mode: str | None = None,
    decode_update_enabled: bool | None = None,
    key_mode: str | None = None,
    max_slots: int | None = None,
    max_bytes: int | None = None,
    fail_on_unsupported: bool | None = None,
) -> None:
    normalized_mode = None
    if mode is not None:
        from tllm.runtime.fast_weight.state import normalize_fast_weight_mode

        normalized_mode = normalize_fast_weight_mode(mode)
    RUNTIME.config.fast_weight_enabled = bool(enabled)
    if not bool(enabled):
        from tllm.runtime.fast_weight.runtime import restore_fast_weight_wrappers

        restore_fast_weight_wrappers(RUNTIME)
        RUNTIME.fast_weight_current_token_batch_meta = None
        RUNTIME.fast_weight_phase = "prefill"
    if layers is not None:
        RUNTIME.config.fast_weight_layers = tuple(int(layer) for layer in layers)
    if chunk_size is not None:
        RUNTIME.config.fast_weight_chunk_size = max(1, int(chunk_size))
    if learning_rate is not None:
        RUNTIME.config.fast_weight_lr = float(learning_rate)
    if normalized_mode is not None:
        RUNTIME.config.fast_weight_mode = normalized_mode
    if decode_update_enabled is not None:
        RUNTIME.config.fast_weight_decode_update_enabled = bool(decode_update_enabled)
    if key_mode is not None:
        RUNTIME.config.fast_weight_key_mode = str(key_mode).strip() or "static"
    if max_slots is not None:
        RUNTIME.config.fast_weight_max_slots = max(0, int(max_slots))
    if max_bytes is not None:
        RUNTIME.config.fast_weight_max_bytes = max(0, int(max_bytes))
    if fail_on_unsupported is not None:
        RUNTIME.config.fast_weight_fail_on_unsupported = bool(fail_on_unsupported)
    from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec
    from tllm.runtime.mp.control import active_control_plane

    refresh_exported_bootstrap_spec()
    plane = active_control_plane()
    if plane is not None:
        plane.set_fast_weight_enabled(
            enabled,
            layers=layers,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
            mode=mode,
            decode_update_enabled=decode_update_enabled,
            key_mode=key_mode,
            max_slots=max_slots,
            max_bytes=max_bytes,
            fail_on_unsupported=fail_on_unsupported,
        )


def synchronize_esamp() -> None:
    from tllm.runtime.ports.residual_bundle_dispatch import (
        synchronize_deferred_port_bundles,
    )

    synchronize_deferred_port_bundles(runtime=RUNTIME)
    if RUNTIME.consumer is not None:
        synchronize_consumer(RUNTIME.consumer)
    registry = RUNTIME.consumer_registry
    if registry is not None:
        for consumer in registry.consumers():
            synchronize_consumer(consumer)

    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is not None:
        plane.synchronize()


def read_and_reset_esamp_stats(sync: bool = True) -> TrainingStats:
    local = (
        TrainingStats(loss_avg=0.0, loss_count=0)
        if RUNTIME.consumer is None
        else RUNTIME.consumer.read_and_reset_stats(sync=sync)
    )
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return TrainingStats.merge([local, *(reply.esamp for reply in plane.pull_stats("esamp", sync=sync))])


def read_and_reset_esamp_per_request_stats(sync: bool = True) -> Dict[int, TrainingStats]:
    local = {} if RUNTIME.consumer is None else RUNTIME.consumer.read_and_reset_per_request_stats(sync=sync)
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
    return _stats.read_graph_debug_stats(getattr(RUNTIME, "consumer", None), mode=mode)


def _prepare_decode_localization(runner: RunnerLike, out: tuple[object, ...]) -> None:
    _hooks.prepare_decode_localization(core=sys.modules[__name__], runner=runner, out=out)


def _build_tap_path_list() -> list[str]:
    return _hooks.build_tap_path_list(core=sys.modules[__name__])


def _setup_runtime_hooks_if_active(runner: RunnerLike) -> None:
    _hooks.setup_runtime_hooks_if_active(core=sys.modules[__name__], runner=runner)


def _wrapped_load_model(self: object, *args: object, **kwargs: object) -> object:
    return _hooks.wrapped_load_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def _wrapped_prepare_inputs(self: object, scheduler_output: object) -> object:
    return _hooks.wrapped_prepare_inputs(core=sys.modules[__name__], runner=self, scheduler_output=scheduler_output)


def _wrapped_execute_model(self: object, *args: object, **kwargs: object) -> object:
    return _hooks.wrapped_execute_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def install_runner_patch() -> None:
    _lifecycle.install_runner_patch(sys.modules[__name__])


def make_llm(*args: object, **kwargs: object) -> object:
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
    "configure_workgraph_adaptive",
    "configure_workgraph_distributed",
    "configure_workgraph_placement",
    "dispatch_workgraph_value",
    "infer_hidden_dtype",
    "make_llm",
    "finish_workgraph_distributed_step",
    "launch_workgraph_collective",
    "pick_common_attn_metadata",
    "poll_workgraph_transfers",
    "poll_workgraph_collectives",
    "publish_workgraph_distributed",
    "read_and_reset_distiller_timing_stats",
    "read_and_reset_path_hotspot_stats",
    "read_and_reset_placement_stats",
    "read_graph_debug_stats",
    "read_and_reset_esamp_per_request_stats",
    "read_and_reset_esamp_stats",
    "refresh_workgraph_capture_metadata",
    "register_dispatch_consumer",
    "replace_dispatch_consumers",
    "resolve_module_by_path_with_fallback",
    "record_path_hotspot_cpu",
    "runner_uses_compilation_or_cudagraph",
    "set_runtime_consumer",
    "set_esamp_training_enabled",
    "set_fast_weight_enabled",
    "set_test_time_lora_enabled",
    "synchronize_esamp",
]

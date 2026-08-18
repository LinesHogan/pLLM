#!/usr/bin/env python3
"""Generic residual bundle dispatch helpers."""

from __future__ import annotations

import time
from typing import List, Protocol

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease, DeviceTensorWriteSink
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.ports.base import ConsumerFlow, PortKind, TimingContract
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.ports.base import PortRead
from tllm.runtime.dispatch_plan import DispatchPlan, FlowDispatchTarget
from tllm.runtime.ports.frame import Ownership, PortFrame
from tllm.runtime.ports.assembler import BundleAssembler
from tllm.runtime.ports.post_generate_dispatch import is_post_generate_flow
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.decode_runtime_metadata import active_request_prompt_sample_metadata
from tllm.runtime.ports import residual_bindings as _residual_bindings
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey
from tllm.runtime.vllm_patch import common_hooks as _common_hooks
from tllm.runtime.consumer_compat import on_step_end as call_consumer_step_end
from tllm.contracts.runtime_context import RunnerLike
from tllm.contracts.runtime_context import RuntimeContext
from tllm.runtime.scheduling.deferred import DeferredWork, DeferredWorkQueue
from tllm.runtime.fusion import FusionExecutor
from tllm.runtime.timing_debug import (
    TimingDebugMode,
    assert_bundle_matches,
    fingerprint_bundle,
    timing_debug_mode,
)
from tllm.runtime.stats import PlacementStats
from tllm.runtime.instrumentation import path_hotspots_enabled, record_path_hotspot_cpu


class PortDispatchRuntimeLike(Protocol):
    decode_count: int
    decode_prompt_idxs: list[int]
    decode_sample_idxs: list[int]
    decode_request_ids: list[str]
    tap_decode_hidden: dict[str, torch.Tensor]
    tap_decode_hidden_compact: dict[str, torch.Tensor]
    decode_compact_count: int
    decode_compact_row_ids: tuple[int, ...]
    residual_bindings: dict[str, ResidualPathBinding]
    write_leases: dict[WriteLeaseKey, ResidualWriteLease]
    event_step_id: int
    dispatch_plan: DispatchPlan | None
    deferred_work_queue: DeferredWorkQueue
    placement_stats: PlacementStats
    fusion_executor: FusionExecutor
    path_hotspot_enabled: bool


class PortDispatchCoreLike(Protocol):
    RUNTIME: PortDispatchRuntimeLike

    def record_path_hotspot_cpu(self, name: str, ms: float) -> None: ...


def _compact_residual_hidden_for_read(
    *,
    core: PortDispatchCoreLike,
    read: PortRead,
    active: int,
    row_ids: tuple[int, ...] | None,
) -> torch.Tensor | None:
    if row_ids is None:
        return None
    if int(core.RUNTIME.decode_compact_count) != int(active):
        return None
    if tuple(core.RUNTIME.decode_compact_row_ids) != tuple(row_ids):
        return None
    locator = read.locator
    if not isinstance(locator, ResidualLocator):
        return None
    resolved_path = _residual_bindings.resolved_path_for_locator(core.RUNTIME.residual_bindings, locator)
    if resolved_path is None:
        return None
    compact_map = core.RUNTIME.tap_decode_hidden_compact
    hidden = compact_map.get(resolved_path)
    if not isinstance(hidden, torch.Tensor) or int(hidden.shape[0]) < int(active):
        return None
    return hidden[:active]


def _regular_row_view(hidden: torch.Tensor, row_ids: tuple[int, ...]) -> torch.Tensor | None:
    if not row_ids:
        return hidden[:0]
    if hidden.ndim != 2:
        return None
    if len(row_ids) == 1:
        start = int(row_ids[0])
        if start < 0 or start >= int(hidden.shape[0]):
            return None
        return hidden[start : start + 1]
    step = int(row_ids[1]) - int(row_ids[0])
    if step <= 0:
        return None
    start = int(row_ids[0])
    last = int(row_ids[-1])
    if start < 0 or last >= int(hidden.shape[0]):
        return None
    for left, right in zip(row_ids, row_ids[1:]):
        if int(right) - int(left) != step:
            return None
    return hidden.as_strided(
        (len(row_ids), int(hidden.shape[1])),
        (int(hidden.stride(0)) * step, int(hidden.stride(1))),
        storage_offset=int(hidden.storage_offset()) + start * int(hidden.stride(0)),
    )


def _request_meta_payload(core: PortDispatchCoreLike, active: int) -> list[dict[str, object]]:
    return _request_meta_view(core, active).as_legacy_dicts()


def _request_meta_view(core: PortDispatchCoreLike, active: int) -> RowBatchMeta:
    request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(core.RUNTIME, active)
    return RowBatchMeta(
        request_ids=tuple(str(request_ids[row_i]) for row_i in range(active)),
        prompt_idxs=tuple(int(prompt_idxs[row_i]) for row_i in range(active)),
        sample_idxs=tuple(int(sample_idxs[row_i]) for row_i in range(active)),
        phase="decode",
        engine_step_id=int(core.RUNTIME.event_step_id),
        row_ids=tuple(range(active)),
    )


def _compact_first_per_prompt_from_runtime(core: PortDispatchCoreLike, active: int) -> tuple[RowBatchMeta, tuple[int, ...]]:
    runtime = core.RUNTIME
    request_ids = runtime.decode_request_ids
    prompt_idxs = runtime.decode_prompt_idxs
    sample_idxs = runtime.decode_sample_idxs
    if len(request_ids) < active or len(prompt_idxs) < active or len(sample_idxs) < active:
        raise RuntimeError(
            "decode runtime metadata is inconsistent: "
            f"active={active} request_ids={len(request_ids)} prompt_idxs={len(prompt_idxs)} sample_idxs={len(sample_idxs)}"
        )
    seen: set[int] = set()
    keep: list[int] = []
    compact_request_ids: list[str] = []
    compact_prompt_idxs: list[int] = []
    compact_sample_idxs: list[int] = []
    for row_i in range(active):
        prompt_key = int(prompt_idxs[row_i])
        if prompt_key < 0 or prompt_key in seen:
            continue
        seen.add(prompt_key)
        keep.append(int(row_i))
        compact_request_ids.append(str(request_ids[row_i]))
        compact_prompt_idxs.append(prompt_key)
        compact_sample_idxs.append(int(sample_idxs[row_i]))
    row_ids = tuple(keep)
    compact = RowBatchMeta(
        request_ids=tuple(compact_request_ids),
        prompt_idxs=tuple(compact_prompt_idxs),
        sample_idxs=tuple(compact_sample_idxs),
        phase="decode",
        engine_step_id=int(runtime.event_step_id),
        row_compaction="first_per_prompt",
        row_ids=row_ids,
    )
    return compact, row_ids


def _request_meta_payload_from_view(view: RowBatchMeta) -> list[dict[str, object]]:
    return [
        {
            "request_id": view.request_ids[row_i],
            "prompt_idx": view.prompt_idxs[row_i],
            "sample_idx": view.sample_idxs[row_i],
            "phase": view.phase,
            "engine_step_id": int(view.engine_step_id),
        }
        for row_i in range(len(view.request_ids))
    ]


def build_explicit_step_scope_port_bundle(
    *,
    flow: ConsumerFlow,
    phase: str,
    engine_step_id: int,
    row_meta: RowBatchMeta,
    entries: dict[str, object],
    write_outputs: dict[str, DeviceTensorWriteSink] | None = None,
) -> PortBundle:
    """Build a step-scope bundle from explicit entries for any runtime phase."""
    if tuple(flow.bundle_key) != ("engine_step_id", "phase"):
        raise RuntimeError("explicit step-scope bundles require bundle_key=('engine_step_id', 'phase')")
    phase_value = str(phase)
    if str(row_meta.phase) != phase_value:
        raise RuntimeError("row metadata phase does not match requested bundle phase")
    if int(row_meta.engine_step_id) != int(engine_step_id):
        raise RuntimeError("row metadata engine_step_id does not match requested bundle step")

    out: dict[str, object] = {}
    lease_entries: dict[str, torch.Tensor] = {}
    delivery = str(flow.delivery)
    for read in flow.reads:
        name = str(read.role).strip() or str(read.kind.value)
        if read.kind is PortKind.REQUEST_META:
            out[name] = row_meta
            continue
        if read.kind is PortKind.RESIDUAL_STREAM:
            locator = read.locator
            if not isinstance(locator, ResidualLocator):
                raise RuntimeError(f"flow read `{name}` has an invalid residual locator")
            read_phase = str(locator.phase)
            if read_phase != phase_value:
                raise RuntimeError(f"flow read `{name}` is for phase `{read_phase}`, not `{phase_value}`")
        if name not in entries:
            raise RuntimeError(f"explicit step-scope bundle missing required entry `{name}`")
        payload = entries[name]
        if delivery == "device_lease":
            if read.kind is not PortKind.RESIDUAL_STREAM or not isinstance(payload, torch.Tensor):
                raise RuntimeError("device_lease explicit bundle supports residual_stream and request_meta reads")
            lease_entries[name] = payload
        else:
            out[name] = payload

    if delivery == "device_lease":
        out["device_lease"] = DeviceTensorLease(
            entries=lease_entries,
            active_rows=len(row_meta.request_ids),
            ownership=flow.ownership,
            lifetime=flow.lease_lifetime,
            outputs={} if write_outputs is None else write_outputs,
        )

    request_id = row_meta.request_ids[0] if row_meta.request_ids else ""
    sample_idx = int(row_meta.sample_idxs[0]) if row_meta.sample_idxs else 0
    return PortBundle(
        key=BundleKey(
            engine_step_id=int(engine_step_id),
            phase=phase_value,
            request_id=str(request_id),
            sample_idx=sample_idx,
        ),
        entries=out,
        version=int(engine_step_id),
    )


def _entry_for_flow_read(
    *,
    core: PortDispatchCoreLike,
    read: PortRead,
    request_meta_payload: list[dict[str, object]],
    active: int,
    row_ids: tuple[int, ...] | None = None,
) -> tuple[str, object] | None:
    name = str(read.role).strip() or str(read.kind.value)
    if read.kind is PortKind.REQUEST_META:
        return name, request_meta_payload
    if read.kind is not PortKind.RESIDUAL_STREAM:
        return None
    compact_hidden = _compact_residual_hidden_for_read(core=core, read=read, active=active, row_ids=row_ids)
    if compact_hidden is not None:
        return name, compact_hidden
    locator = read.locator
    if not isinstance(locator, ResidualLocator):
        return None
    resolved_path = _residual_bindings.resolved_path_for_locator(core.RUNTIME.residual_bindings, locator)
    if resolved_path is None:
        return None
    hidden = core.RUNTIME.tap_decode_hidden.get(resolved_path)
    if hidden is None or not isinstance(hidden, torch.Tensor):
        return None
    if row_ids is None:
        return name, hidden[:active]
    if not row_ids:
        return name, hidden[:0]
    if row_ids == tuple(range(len(row_ids))):
        return name, hidden[: len(row_ids)]
    view = _regular_row_view(hidden, row_ids)
    if view is not None:
        return name, view
    row_tensor = torch.as_tensor(row_ids, device=hidden.device, dtype=torch.long)
    return name, hidden.index_select(0, row_tensor)


def build_decode_port_frames(*, core: PortDispatchCoreLike, layer_path: str) -> List[PortFrame]:
    decode_buf = core.RUNTIME.tap_decode_hidden.get(str(layer_path))
    if not isinstance(decode_buf, torch.Tensor):
        return []

    active = int(core.RUNTIME.decode_count)
    if active <= 0:
        return []

    binding = core.RUNTIME.residual_bindings.get(str(layer_path))
    if binding is None:
        return []
    locator = binding.locator
    request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(core.RUNTIME, active)
    frames: List[PortFrame] = []
    request_meta_locator = RequestMeta.read().locator
    include_request_meta = bool(binding.include_request_meta)

    for row_i in range(active):
        prompt_idx = prompt_idxs[row_i]
        sample_idx = sample_idxs[row_i]
        request_id = request_ids[row_i]
        key = BundleKey(
            engine_step_id=int(core.RUNTIME.event_step_id),
            phase="decode",
            request_id=request_id,
            sample_idx=sample_idx,
        )
        frames.append(
            PortFrame(
                key=key,
                kind=ResidualStream.KIND,
                locator=locator,
                payload=decode_buf[row_i],
                ownership=Ownership.BORROWED,
                ready_window="same_step",
                version=int(core.RUNTIME.event_step_id),
            )
        )
        if include_request_meta:
            frames.append(
                PortFrame(
                    key=key,
                    kind=RequestMeta.KIND,
                    locator=request_meta_locator,
                    payload={
                        "request_id": request_id,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "phase": "decode",
                        "engine_step_id": int(core.RUNTIME.event_step_id),
                    },
                    ownership=Ownership.STAGED,
                    ready_window="same_step",
                    version=int(core.RUNTIME.event_step_id),
                )
            )
    return frames


def _write_outputs_for_flow(
    *,
    core: PortDispatchCoreLike,
    flow: ConsumerFlow,
    consumer_id: str,
) -> dict[str, DeviceTensorWriteSink]:
    if not flow.writes:
        return {}
    writer_id = str(consumer_id).strip()
    if not writer_id:
        raise RuntimeError("device write outputs require the dispatch target consumer_id")
    outputs: dict[str, DeviceTensorWriteSink] = {}
    for role, write in zip(flow.output_fields, flow.writes):
        if write.kind is not PortKind.RESIDUAL_STREAM or not isinstance(write.locator, ResidualLocator):
            raise RuntimeError(
                f"device_lease output {role!r} has no runtime materializer for port kind {write.kind.value!r}"
            )
        layer_path = _residual_bindings.resolved_path_for_locator(
            core.RUNTIME.residual_bindings,
            write.locator,
        )
        if not layer_path:
            raise RuntimeError(f"device_lease output {role!r} has no resolved residual layer")
        key = WriteLeaseKey(
            layer_path=str(layer_path),
            writer_id=writer_id,
            order=int(write.order),
        )
        lease = core.RUNTIME.write_leases.get(key)
        if lease is None:
            raise RuntimeError(f"device_lease output {role!r} has no allocated runtime write lease")
        outputs[role] = lease
    return outputs


def build_step_scope_port_bundle(
    *,
    core: PortDispatchCoreLike,
    flow: ConsumerFlow,
    consumer_id: str = "",
) -> PortBundle | None:
    if tuple(flow.bundle_key) != ("engine_step_id", "phase"):
        return None
    active = int(core.RUNTIME.decode_count)
    if active <= 0:
        return None

    entries: dict[str, object] = {}
    row_ids: tuple[int, ...] | None = None
    row_compaction = str(flow.row_compaction)
    if row_compaction == "first_per_prompt":
        request_meta_view, row_ids = _compact_first_per_prompt_from_runtime(core, active)
        active = len(row_ids)
        if active <= 0:
            raise RuntimeError(
                "first_per_prompt row compaction requires resolved request identity "
                "for at least one active decode row"
            )
    elif row_compaction != "none":
        raise RuntimeError(f"unsupported row_compaction mode `{row_compaction}`")
    else:
        request_meta_view = _request_meta_view(core, active)

    row_cap = int(flow.max_bundle_rows)
    if row_cap > 0 and active > row_cap:
        raise RuntimeError(
            f"flow row cap exceeded: active_rows={active} max_bundle_rows={row_cap}. "
            "Increase the flow capacity or reduce the workload; tLLM does not silently drop residual rows."
        )

    delivery = str(flow.delivery)
    if delivery == "device_lease":
        lease_entries: dict[str, torch.Tensor] = {}
        for read in flow.reads:
            entry = _entry_for_flow_read(
                core=core,
                read=read,
                request_meta_payload=[],
                active=active,
                row_ids=row_ids,
            )
            if entry is None:
                name = str(read.role).strip() or str(read.kind.value)
                raise RuntimeError(f"active flow bundle missing required entry `{name}`")
            name, payload = entry
            if read.kind is PortKind.REQUEST_META:
                entries[name] = request_meta_view
                continue
            if read.kind is not PortKind.RESIDUAL_STREAM or not isinstance(payload, torch.Tensor):
                raise RuntimeError("device_lease delivery currently supports residual_stream and request_meta reads")
            lease_entries[name] = payload
        lease = DeviceTensorLease(
            entries=lease_entries,
            active_rows=int(active),
            ownership=flow.ownership,
            lifetime=flow.lease_lifetime,
            outputs=_write_outputs_for_flow(
                core=core,
                flow=flow,
                consumer_id=consumer_id,
            ),
        )
        entries["device_lease"] = lease
        request_id = request_meta_view.request_ids[0] if request_meta_view.request_ids else ""
        sample_idx = int(request_meta_view.sample_idxs[0]) if request_meta_view.sample_idxs else 0
    else:
        request_meta_payload = _request_meta_payload_from_view(request_meta_view)
        for read in flow.reads:
            entry = _entry_for_flow_read(
                core=core,
                read=read,
                request_meta_payload=request_meta_payload,
                active=active,
                row_ids=row_ids,
            )
            if entry is None:
                name = str(read.role).strip() or str(read.kind.value)
                raise RuntimeError(f"active flow bundle missing required entry `{name}`")
            name, payload = entry
            entries[name] = payload
        request_id = request_meta_view.request_ids[0] if request_meta_view.request_ids else ""
        sample_idx = int(request_meta_view.sample_idxs[0]) if request_meta_view.sample_idxs else 0

    return PortBundle(
        key=BundleKey(
            engine_step_id=int(core.RUNTIME.event_step_id),
            phase="decode",
            request_id=str(request_id),
            sample_idx=sample_idx,
        ),
        entries=entries,
        version=int(core.RUNTIME.event_step_id),
    )


def _flow_due_for_step(*, flow: ConsumerFlow, step_id: int) -> bool:
    stride = max(1, int(flow.dispatch_every_n_steps))
    return (int(step_id) % stride) == 0


def _consume_flow_target(
    runtime: PortDispatchRuntimeLike,
    target: FlowDispatchTarget,
    bundle: PortBundle,
    ctx: RuntimeContext,
) -> None:
    executor = runtime.fusion_executor
    if executor.execute_if_fused(
        target=target,
        bundle=bundle,
        context=ctx,
        stats=runtime.placement_stats,
    ):
        return
    target.consumer.consume_bundle(bundle, ctx)
    if str(target.flow.window) != "background":
        call_consumer_step_end(target.consumer, ctx)


def _enqueue_flow_target(
    *,
    queue: DeferredWorkQueue,
    runtime: PortDispatchRuntimeLike,
    target: FlowDispatchTarget,
    bundle: PortBundle,
    ctx: RuntimeContext,
    step_id: int,
) -> None:
    lease = bundle.entries.get("device_lease")
    if str(target.flow.delivery) == "device_lease" and not isinstance(
        lease, DeviceTensorLease
    ):
        raise RuntimeError(
            f"device_lease flow {target.flow_id!r} produced a legacy frame bundle; "
            f"bundle_key={tuple(target.flow.bundle_key)!r} active_rows=0 "
            f"entries={tuple(bundle.entries)!r}"
        )
    timing = target.flow.timing
    assert isinstance(timing, TimingContract)
    max_defer = max(1, int(timing.max_defer_steps))
    # Port bundles may retain borrowed decode scratch. Completing no later than
    # the next prepare hook prevents storage reuse even when the semantic
    # max_defer budget is larger.
    deadline_step = min(int(step_id) + max_defer, int(step_id) + 1)
    declared_fingerprint = (
        fingerprint_bundle(bundle)
        if timing_debug_mode() == TimingDebugMode.CROSS_CHECK
        else None
    )
    queue.enqueue(
        DeferredWork(
            flow_id=target.flow_id,
            bundle=bundle,
            context=ctx,
            run=lambda queued_bundle, queued_ctx, dispatch_target=target: _consume_flow_target(
                runtime, dispatch_target, queued_bundle, queued_ctx
            ),
            enqueued_step=int(step_id),
            eligible_step=int(step_id),
            eligible_hook=target.placement_hook,
            deadline_step=deadline_step,
            deadline_hook="prepare_inputs.pre",
            verify=(
                None
                if declared_fingerprint is None
                else lambda queued_bundle, expected=declared_fingerprint, flow_id=target.flow_id: assert_bundle_matches(
                    expected, queued_bundle, flow_id=flow_id
                )
            ),
        )
    )


def drain_deferred_port_bundles(
    *,
    runtime: PortDispatchRuntimeLike,
    step_id: int,
    hook: str,
    expired_only: bool = False,
    predecessor_events: tuple[object, ...] = (),
) -> int:
    queue = runtime.deferred_work_queue
    if not queue:
        return 0
    started = time.perf_counter()
    if expired_only:
        drained = queue.flush_expired(
            step_id=int(step_id),
            hook=str(hook),
            predecessor_events=predecessor_events,
        )
        runtime.placement_stats.deadline_flush_count += drained
    else:
        drained = queue.drain_on_bubble(
            step_id=int(step_id),
            hook=str(hook),
            predecessor_events=predecessor_events,
        )
        runtime.placement_stats.bubble_drain_count += drained
        runtime.placement_stats.bubble_drain_ms += (time.perf_counter() - started) * 1000.0
    return drained


def retire_step_device_leases(*, core: PortDispatchCoreLike, runner: RunnerLike) -> int:
    """Fence step leases before the next model forward can overwrite taps."""
    plan = core.RUNTIME.dispatch_plan
    if plan is None:
        return 0
    ctx = _common_hooks.build_runtime_context(
        runner=runner,
        event_name="prepare_inputs.post:retire_step_leases",
    )
    retired = 0
    seen: set[int] = set()
    for target in plan.flow_targets():
        if str(target.flow.delivery) != "device_lease":
            continue
        if str(target.flow.lease_lifetime) != "step":
            continue
        consumer_key = id(target.consumer)
        if consumer_key in seen:
            continue
        seen.add(consumer_key)
        target.consumer.retire_device_lease(ctx)
        retired += 1
    if retired:
        core.RUNTIME.placement_stats.step_lease_retire_count += retired
    return retired


def dispatch_decode_port_bundles(*, core: PortDispatchCoreLike, runner: RunnerLike) -> int:
    plan = core.RUNTIME.dispatch_plan
    if plan is None:
        return 0

    from tllm.runtime.graph_policy import resolve_graph_mode

    if plan.requires_graph_mode_resolution() and plan.ensure_graph_mode(resolve_graph_mode(runner)):
        core.RUNTIME.placement_stats.fusion_eligible_count += sum(
            len(group) for group in plan.fusion_groups
        )
        core.RUNTIME.placement_stats.fusion_fallback_count += len(plan.fusion_rejections)
    core.RUNTIME.fusion_executor.configure(plan)

    step_id = int(core.RUNTIME.event_step_id)
    targets = [
        target
        for target in plan.flow_targets()
        if (not is_post_generate_flow(target.flow)) and _flow_due_for_step(flow=target.flow, step_id=step_id)
    ]
    if not targets:
        return 0

    dispatched = 0
    trace_hotspots = path_hotspots_enabled(core)
    t_ctx0 = time.perf_counter() if trace_hotspots else 0.0
    ctx = _common_hooks.build_runtime_context(runner=runner, event_name="flow:decode")
    t_ctx1 = time.perf_counter() if trace_hotspots else 0.0
    record_path_hotspot_cpu(core, "dispatch_bundles.context_cpu", t_ctx0, t_ctx1)
    frame_targets: List[FlowDispatchTarget] = []
    for target in targets:
        t_build0 = time.perf_counter() if trace_hotspots else 0.0
        direct_bundle = build_step_scope_port_bundle(
            core=core,
            flow=target.flow,
            consumer_id=str(target.consumer.consumer_id),
        )
        t_build1 = time.perf_counter() if trace_hotspots else 0.0
        record_path_hotspot_cpu(core, "dispatch_bundles.step_bundle_cpu", t_build0, t_build1)
        if direct_bundle is not None:
            t_consume0 = time.perf_counter() if trace_hotspots else 0.0
            if target.placement_hook == "execute_model.post":
                core.RUNTIME.placement_stats.pinned_dispatch_count += 1
                _consume_flow_target(core.RUNTIME, target, direct_bundle, ctx)
                t_consume1 = time.perf_counter() if trace_hotspots else 0.0
                record_path_hotspot_cpu(core, "dispatch_bundles.consume_bundle_cpu", t_consume0, t_consume1)
            else:
                core.RUNTIME.placement_stats.moved_dispatch_count += 1
                _enqueue_flow_target(
                    queue=core.RUNTIME.deferred_work_queue,
                    runtime=core.RUNTIME,
                    target=target,
                    bundle=direct_bundle,
                    ctx=ctx,
                    step_id=step_id,
                )
                core.RUNTIME.placement_stats.enqueued_count += 1
                core.RUNTIME.placement_stats.max_queue_depth = max(
                    core.RUNTIME.placement_stats.max_queue_depth,
                    core.RUNTIME.deferred_work_queue.depth,
                )
                t_consume1 = time.perf_counter() if trace_hotspots else 0.0
                record_path_hotspot_cpu(core, "dispatch_bundles.consume_bundle_cpu", t_consume0, t_consume1)
            dispatched += 1
        else:
            frame_targets.append(target)

    if not frame_targets:
        return dispatched

    frames: List[PortFrame] = []
    for layer_path in _residual_bindings.tap_paths(core.RUNTIME.residual_bindings):
        layer_key = str(layer_path).strip()
        if not layer_key:
            continue
        frames.extend(build_decode_port_frames(core=core, layer_path=layer_key))
    if not frames:
        return dispatched

    for target in frame_targets:
        assembler = BundleAssembler(target.flow)
        for frame in frames:
            bundles = assembler.push(frame)
            for bundle in bundles:
                if target.placement_hook == "execute_model.post":
                    core.RUNTIME.placement_stats.pinned_dispatch_count += 1
                    target.consumer.consume_bundle(bundle, ctx)
                else:
                    core.RUNTIME.placement_stats.moved_dispatch_count += 1
                    _enqueue_flow_target(
                        queue=core.RUNTIME.deferred_work_queue,
                        runtime=core.RUNTIME,
                        target=target,
                        bundle=bundle,
                        ctx=ctx,
                        step_id=step_id,
                    )
                    core.RUNTIME.placement_stats.enqueued_count += 1
                    core.RUNTIME.placement_stats.max_queue_depth = max(
                        core.RUNTIME.placement_stats.max_queue_depth,
                        core.RUNTIME.deferred_work_queue.depth,
                    )
                dispatched += 1
        for bundle in assembler.finalize_pending():
            if target.placement_hook == "execute_model.post":
                core.RUNTIME.placement_stats.pinned_dispatch_count += 1
                target.consumer.consume_bundle(bundle, ctx)
            else:
                core.RUNTIME.placement_stats.moved_dispatch_count += 1
                _enqueue_flow_target(
                    queue=core.RUNTIME.deferred_work_queue,
                    runtime=core.RUNTIME,
                    target=target,
                    bundle=bundle,
                    ctx=ctx,
                    step_id=step_id,
                )
                core.RUNTIME.placement_stats.enqueued_count += 1
                core.RUNTIME.placement_stats.max_queue_depth = max(
                    core.RUNTIME.placement_stats.max_queue_depth,
                    core.RUNTIME.deferred_work_queue.depth,
                )
            dispatched += 1
        if target.placement_hook == "execute_model.post" and str(target.flow.window) != "background":
            call_consumer_step_end(target.consumer, ctx)
    return dispatched

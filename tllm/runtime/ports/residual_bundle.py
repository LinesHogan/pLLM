#!/usr/bin/env python3
"""Residual port-bundle construction for hook-owned direct dispatch."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import torch

from tllm.contracts.gpu_stage import DeviceTensorLease, DeviceTensorWriteSink
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RunnerLike, RuntimeContext
from tllm.ports.base import ConsumerFlow, PortKind, PortRead
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.runtime.consumer_compat import on_step_end as call_consumer_step_end
from tllm.runtime.decode_runtime_metadata import active_request_prompt_sample_metadata
from tllm.runtime.dispatch_plan import DispatchPlan, FlowDispatchTarget
from tllm.runtime.fusion import FusionExecutor
from tllm.runtime.instrumentation import path_hotspots_enabled, record_path_hotspot_cpu
from tllm.runtime.ports import residual_bindings as _residual_bindings
from tllm.runtime.ports.assembler import BundleAssembler
from tllm.runtime.ports.frame import Ownership, PortFrame
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey
from tllm.runtime.ports.post_generate_dispatch import is_post_generate_flow
from tllm.runtime.stats import PlacementStats
from tllm.runtime.vllm_patch import common_hooks as _common_hooks

if TYPE_CHECKING:
    from tllm.runtime.workload_execution import NativeWorkloadExecution


class ResidualBundleRuntimeLike(Protocol):
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
    placement_stats: PlacementStats
    fusion_executor: FusionExecutor
    path_hotspot_enabled: bool
    workload_execution: "NativeWorkloadExecution | None"


class ResidualBundleCoreLike(Protocol):
    RUNTIME: ResidualBundleRuntimeLike

    def record_path_hotspot_cpu(self, name: str, ms: float) -> None: ...


def _compact_residual_hidden_for_read(
    *,
    core: ResidualBundleCoreLike,
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
    resolved_path = _residual_bindings.resolved_path_for_locator(
        core.RUNTIME.residual_bindings, locator
    )
    if resolved_path is None:
        return None
    hidden = core.RUNTIME.tap_decode_hidden_compact.get(resolved_path)
    if not isinstance(hidden, torch.Tensor) or int(hidden.shape[0]) < int(active):
        return None
    return hidden[:active]


def _regular_row_view(
    hidden: torch.Tensor, row_ids: tuple[int, ...]
) -> torch.Tensor | None:
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


def _request_meta_view(core: ResidualBundleCoreLike, active: int) -> RowBatchMeta:
    request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(
        core.RUNTIME, active
    )
    return RowBatchMeta(
        request_ids=tuple(str(request_ids[row_i]) for row_i in range(active)),
        prompt_idxs=tuple(int(prompt_idxs[row_i]) for row_i in range(active)),
        sample_idxs=tuple(int(sample_idxs[row_i]) for row_i in range(active)),
        phase="decode",
        engine_step_id=int(core.RUNTIME.event_step_id),
        row_ids=tuple(range(active)),
    )


def _compact_first_per_prompt_from_runtime(
    core: ResidualBundleCoreLike, active: int
) -> tuple[RowBatchMeta, tuple[int, ...]]:
    runtime = core.RUNTIME
    request_ids = runtime.decode_request_ids
    prompt_idxs = runtime.decode_prompt_idxs
    sample_idxs = runtime.decode_sample_idxs
    if (
        len(request_ids) < active
        or len(prompt_idxs) < active
        or len(sample_idxs) < active
    ):
        raise RuntimeError(
            "decode runtime metadata is inconsistent: "
            f"active={active} request_ids={len(request_ids)} "
            f"prompt_idxs={len(prompt_idxs)} sample_idxs={len(sample_idxs)}"
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


def _entry_for_flow_read(
    *,
    core: ResidualBundleCoreLike,
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
    compact_hidden = _compact_residual_hidden_for_read(
        core=core, read=read, active=active, row_ids=row_ids
    )
    if compact_hidden is not None:
        return name, compact_hidden
    locator = read.locator
    if not isinstance(locator, ResidualLocator):
        return None
    resolved_path = _residual_bindings.resolved_path_for_locator(
        core.RUNTIME.residual_bindings, locator
    )
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


def build_decode_port_frames(
    *, core: ResidualBundleCoreLike, layer_path: str
) -> list[PortFrame]:
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
    request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(
        core.RUNTIME, active
    )
    frames: list[PortFrame] = []
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
    core: ResidualBundleCoreLike,
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
        if write.kind is not PortKind.RESIDUAL_STREAM or not isinstance(
            write.locator, ResidualLocator
        ):
            raise RuntimeError(
                f"device_lease output {role!r} has no runtime materializer "
                f"for port kind {write.kind.value!r}"
            )
        layer_path = _residual_bindings.resolved_path_for_locator(
            core.RUNTIME.residual_bindings,
            write.locator,
        )
        if not layer_path:
            raise RuntimeError(
                f"device_lease output {role!r} has no resolved residual layer"
            )
        key = WriteLeaseKey(
            layer_path=str(layer_path),
            writer_id=writer_id,
            order=int(write.order),
        )
        lease = core.RUNTIME.write_leases.get(key)
        if lease is None:
            raise RuntimeError(
                f"device_lease output {role!r} has no allocated runtime write lease"
            )
        outputs[role] = lease
    return outputs


def build_step_scope_port_bundle(
    *,
    core: ResidualBundleCoreLike,
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

    if str(flow.delivery) == "device_lease":
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
            if read.kind is not PortKind.RESIDUAL_STREAM or not isinstance(
                payload, torch.Tensor
            ):
                raise RuntimeError(
                    "device_lease delivery currently supports residual_stream "
                    "and request_meta reads"
                )
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
        request_id = (
            request_meta_view.request_ids[0] if request_meta_view.request_ids else ""
        )
        sample_idx = (
            int(request_meta_view.sample_idxs[0])
            if request_meta_view.sample_idxs
            else 0
        )
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
        request_id = (
            request_meta_view.request_ids[0] if request_meta_view.request_ids else ""
        )
        sample_idx = (
            int(request_meta_view.sample_idxs[0])
            if request_meta_view.sample_idxs
            else 0
        )

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
    runtime: ResidualBundleRuntimeLike,
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
    target.consumer.consume_flow(target.flow_ordinal, bundle, ctx)
    if str(target.flow.window) != "background":
        call_consumer_step_end(target.consumer, ctx)


def _reject_moved_flow(target: FlowDispatchTarget) -> None:
    if target.placement_hook == "execute_model.post":
        return
    raise RuntimeError(
        f"flow {target.flow_id!r} resolved to moved hook {target.placement_hook!r}; "
        "deferred/adaptive WorkGraph dispatch was removed, so moved flows must "
        "enter native workload execution instead of compatibility dispatch"
    )


def retire_step_device_leases(
    *, core: ResidualBundleCoreLike, runner: RunnerLike
) -> int:
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


def dispatch_decode_port_bundles(
    *, core: ResidualBundleCoreLike, runner: RunnerLike
) -> int:
    plan = core.RUNTIME.dispatch_plan
    if plan is None:
        return 0

    from tllm.runtime.graph_policy import resolve_graph_mode

    placement_refreshed = plan.requires_graph_mode_resolution() and plan.ensure_graph_mode(
        resolve_graph_mode(runner)
    )
    if placement_refreshed:
        core.RUNTIME.placement_stats.fusion_eligible_count += sum(
            len(group) for group in plan.fusion_groups
        )
        core.RUNTIME.placement_stats.fusion_fallback_count += len(
            plan.fusion_rejections
        )
    core.RUNTIME.fusion_executor.configure(plan)

    step_id = int(core.RUNTIME.event_step_id)
    native_execution = core.RUNTIME.workload_execution
    targets = [
        target
        for target in plan.flow_targets()
        if (not is_post_generate_flow(target.flow))
        and _flow_due_for_step(flow=target.flow, step_id=step_id)
        and not (
            native_execution is not None
            and native_execution.owns_consumer(target.consumer)
        )
    ]
    if not targets:
        return 0
    for target in targets:
        _reject_moved_flow(target)

    dispatched = 0
    trace_hotspots = path_hotspots_enabled(core)
    t_ctx0 = time.perf_counter() if trace_hotspots else 0.0
    ctx = _common_hooks.build_runtime_context(runner=runner, event_name="flow:decode")
    t_ctx1 = time.perf_counter() if trace_hotspots else 0.0
    record_path_hotspot_cpu(core, "dispatch_bundles.context_cpu", t_ctx0, t_ctx1)

    frame_targets: list[FlowDispatchTarget] = []
    for target in targets:
        t_build0 = time.perf_counter() if trace_hotspots else 0.0
        direct_bundle = build_step_scope_port_bundle(
            core=core,
            flow=target.flow,
            consumer_id=str(target.consumer.consumer_id),
        )
        t_build1 = time.perf_counter() if trace_hotspots else 0.0
        record_path_hotspot_cpu(
            core, "dispatch_bundles.step_bundle_cpu", t_build0, t_build1
        )
        if direct_bundle is None:
            frame_targets.append(target)
            continue
        t_consume0 = time.perf_counter() if trace_hotspots else 0.0
        core.RUNTIME.placement_stats.pinned_dispatch_count += 1
        _consume_flow_target(core.RUNTIME, target, direct_bundle, ctx)
        t_consume1 = time.perf_counter() if trace_hotspots else 0.0
        record_path_hotspot_cpu(
            core, "dispatch_bundles.consume_bundle_cpu", t_consume0, t_consume1
        )
        dispatched += 1

    if not frame_targets:
        return dispatched

    frames: list[PortFrame] = []
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
            for bundle in assembler.push(frame):
                core.RUNTIME.placement_stats.pinned_dispatch_count += 1
                target.consumer.consume_flow(target.flow_ordinal, bundle, ctx)
                dispatched += 1
        for bundle in assembler.finalize_pending():
            core.RUNTIME.placement_stats.pinned_dispatch_count += 1
            target.consumer.consume_flow(target.flow_ordinal, bundle, ctx)
            dispatched += 1
        if str(target.flow.window) != "background":
            call_consumer_step_end(target.consumer, ctx)
    return dispatched


__all__ = [
    "build_decode_port_frames",
    "build_step_scope_port_bundle",
    "dispatch_decode_port_bundles",
    "retire_step_device_leases",
]

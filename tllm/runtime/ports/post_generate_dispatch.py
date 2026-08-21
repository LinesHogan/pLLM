#!/usr/bin/env python3
"""Post-generate flow dispatch helpers."""

from __future__ import annotations

from typing import Protocol

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RunnerLike
from tllm.ports.base import ConsumerFlow, PortKind, PortRead, PortWrite
from tllm.runtime import reward_outcome_runtime
from tllm.runtime import retry_control_runtime
from tllm.runtime import update_barrier_runtime
from tllm.runtime.consumer_compat import on_step_end as call_consumer_step_end
from tllm.runtime.dispatch_plan import DispatchPlan, FlowDispatchTarget
from tllm.runtime.generation_trace_runtime import GenerationTraceBatch
from tllm.runtime.vllm_patch import common_hooks as _common_hooks
from tllm.runtime.append_store import RuntimeAppendSink


class PostGenerateRuntimeLike(Protocol):
    dispatch_plan: DispatchPlan | None


class RewardOutcomeSink(RuntimeAppendSink[reward_outcome_runtime.RewardOutcomeBatch]):
    """Write-only sink exposed for RewardOutcome post-generate flow writes."""

    def __init__(self, *, runtime: object, mode: str = "append") -> None:
        super().__init__(
            runtime=runtime,
            mode=mode,
            port_name="reward outcome",
            writer=reward_outcome_runtime.write_reward_outcomes,
        )


class RetryControlSink(RuntimeAppendSink[retry_control_runtime.RetryControlBatch]):
    """Write-only sink exposed for RetryControl post-generate flow writes."""

    def __init__(self, *, runtime: object, mode: str = "append") -> None:
        super().__init__(
            runtime=runtime,
            mode=mode,
            port_name="retry control",
            writer=retry_control_runtime.write_retry_controls,
        )


class UpdateBarrierSink(RuntimeAppendSink[update_barrier_runtime.UpdateBarrierBatch]):
    """Write-only sink exposed for UpdateBarrier post-generate flow writes."""

    def __init__(self, *, runtime: object, mode: str = "append") -> None:
        super().__init__(
            runtime=runtime,
            mode=mode,
            port_name="update barrier",
            writer=update_barrier_runtime.write_update_barriers,
        )


def is_post_generate_flow(flow: ConsumerFlow) -> bool:
    return any(read.kind == PortKind.GENERATION_TRACE for read in flow.reads) or any(
        write.kind == PortKind.REWARD_OUTCOME
        or write.kind == PortKind.RETRY_CONTROL
        or write.kind == PortKind.UPDATE_BARRIER
        for write in flow.writes
    )


def _role_or_kind(item: PortRead | PortWrite) -> str:
    role = str(getattr(item, "role", "")).strip()
    return role or str(item.kind.value)


def _require_generation_trace(runtime: object) -> GenerationTraceBatch:
    trace = getattr(runtime, "last_generation_trace", None)
    if not isinstance(trace, GenerationTraceBatch):
        raise RuntimeError("post_generate flow requires runtime.last_generation_trace")
    return trace


def _generation_group_id(trace: GenerationTraceBatch) -> str:
    group_ids = {entry.group_id for entry in trace.entries}
    if len(group_ids) > 1:
        raise RuntimeError("post_generate bundle cannot combine multiple generation_group_id values")
    return next(iter(group_ids), "")


def _entry_for_read(*, runtime: object, read: PortRead) -> tuple[str, object]:
    if read.kind == PortKind.GENERATION_TRACE:
        return _role_or_kind(read), _require_generation_trace(runtime)
    raise RuntimeError(f"post_generate flow cannot read `{read.kind.value}`")


def _entry_for_write(*, runtime: object, write: PortWrite) -> tuple[str, object]:
    if write.kind == PortKind.REWARD_OUTCOME:
        mode_raw = getattr(write, "mode", "")
        if not isinstance(mode_raw, str):
            raise TypeError("reward outcome write mode must be str")
        mode = mode_raw or "append"
        return _role_or_kind(write), RewardOutcomeSink(runtime=runtime, mode=mode)
    if write.kind == PortKind.RETRY_CONTROL:
        mode_raw = getattr(write, "mode", "")
        if not isinstance(mode_raw, str):
            raise TypeError("retry control write mode must be str")
        mode = mode_raw or "append"
        return _role_or_kind(write), RetryControlSink(runtime=runtime, mode=mode)
    if write.kind == PortKind.UPDATE_BARRIER:
        mode_raw = getattr(write, "mode", "")
        if not isinstance(mode_raw, str):
            raise TypeError("update barrier write mode must be str")
        mode = mode_raw or "append"
        return _role_or_kind(write), UpdateBarrierSink(runtime=runtime, mode=mode)
    raise RuntimeError(f"post_generate flow cannot write `{write.kind.value}`")


def build_post_generate_port_bundle(*, runtime: object, flow: ConsumerFlow) -> PortBundle:
    if not is_post_generate_flow(flow):
        raise RuntimeError("flow does not declare a post_generate port")
    if tuple(flow.bundle_key) not in {(), ("generation_group_id", "phase")}:
        raise RuntimeError("post_generate flows require bundle_key=() or ('generation_group_id', 'phase')")

    trace = (
        _require_generation_trace(runtime)
        if "generation_group_id" in flow.bundle_key
        or any(read.kind == PortKind.GENERATION_TRACE for read in flow.reads)
        else None
    )
    entries: dict[str, object] = {}
    for read in flow.reads:
        name, payload = _entry_for_read(runtime=runtime, read=read)
        entries[name] = payload
    for write in flow.writes:
        name, payload = _entry_for_write(runtime=runtime, write=write)
        entries[name] = payload

    return PortBundle(
        key=BundleKey(
            engine_step_id=int(getattr(runtime, "event_step_id", 0) or 0),
            phase="post_generate",
            request_id="",
            sample_idx=0,
            generation_group_id=_generation_group_id(trace) if trace is not None else "",
        ),
        entries=entries,
    )


def _post_generate_targets(plan: DispatchPlan) -> list[FlowDispatchTarget]:
    return [target for target in plan.flow_targets() if is_post_generate_flow(target.flow)]


def dispatch_post_generate_port_bundles(*, runtime: PostGenerateRuntimeLike, runner: RunnerLike) -> int:
    plan = getattr(runtime, "dispatch_plan", None)
    if plan is None:
        return 0

    targets = _post_generate_targets(plan)
    if not targets:
        return 0

    ctx = _common_hooks.build_runtime_context(runner=runner, event_name="flow:post_generate")
    dispatched = 0
    for target in targets:
        bundle = build_post_generate_port_bundle(runtime=runtime, flow=target.flow)
        target.consumer.consume_flow(target.flow_ordinal, bundle, ctx)
        dispatched += 1
        if str(target.flow.window) != "background":
            call_consumer_step_end(target.consumer, ctx)
    return dispatched

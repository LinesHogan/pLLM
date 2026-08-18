#!/usr/bin/env python3
"""Runtime executor for compiler-selected consumer fusion groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from tllm.consumers.base import FusionGroupHandle
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RuntimeContext

if TYPE_CHECKING:
    from tllm.runtime.dispatch_plan import DispatchPlan, FlowDispatchTarget


class FusionStats(Protocol):
    fusion_execution_count: int
    fusion_replay_count: int
    fusion_eager_fallback_count: int


_InvocationKey = tuple[BundleKey, int]


@dataclass(slots=True)
class _PendingInvocation:
    context: RuntimeContext
    bundles: dict[str, PortBundle] = field(default_factory=dict)


@dataclass(slots=True)
class _FusionGroup:
    flow_ids: tuple[str, ...]
    handle: FusionGroupHandle | None
    pending: dict[_InvocationKey, _PendingInvocation] = field(default_factory=dict)


class FusionExecutor:
    """Bind every compiler group to one executable runtime artifact.

    Member dispatches only deliver their bundles here. Once all bundles for an
    invocation are present, the group handle executes exactly once. A compiler
    group without a handle is an explicit eager fallback; it is never reported
    as fused execution.
    """

    __slots__ = ("_configuration_key", "_groups_by_flow")

    def __init__(self) -> None:
        self._configuration_key: tuple[int, tuple[tuple[str, ...], ...]] | None = None
        self._groups_by_flow: dict[str, _FusionGroup] = {}

    @staticmethod
    def _resolve_handle(
        *,
        flow_ids: tuple[str, ...],
        targets: tuple["FlowDispatchTarget", ...],
    ) -> FusionGroupHandle | None:
        handles: list[FusionGroupHandle] = []
        seen_consumers: set[int] = set()
        for target in targets:
            consumer_key = id(target.consumer)
            if consumer_key in seen_consumers:
                continue
            seen_consumers.add(consumer_key)
            handle = target.consumer.fusion_group_handle(flow_ids)
            if handle is None or any(handle is existing for existing in handles):
                continue
            handles.append(handle)
        if len(handles) > 1:
            raise RuntimeError(
                f"compiler fusion group {flow_ids!r} resolved to multiple group handles"
            )
        return handles[0] if handles else None

    def configure(self, plan: "DispatchPlan") -> None:
        groups = tuple(tuple(str(flow_id) for flow_id in group) for group in plan.fusion_groups)
        configuration_key = (id(plan), groups)
        if configuration_key == self._configuration_key:
            return

        targets_by_id = {target.flow_id: target for target in plan.flow_targets()}
        groups_by_flow: dict[str, _FusionGroup] = {}
        for flow_ids in groups:
            if not flow_ids:
                raise RuntimeError("compiler emitted an empty fusion group")
            try:
                targets = tuple(targets_by_id[flow_id] for flow_id in flow_ids)
            except KeyError as exc:
                raise RuntimeError(
                    f"compiler fusion group references unknown flow {exc.args[0]!r}"
                ) from exc
            group = _FusionGroup(
                flow_ids=flow_ids,
                handle=self._resolve_handle(flow_ids=flow_ids, targets=targets),
            )
            for flow_id in flow_ids:
                if flow_id in groups_by_flow:
                    raise RuntimeError(
                        f"compiler flow {flow_id!r} belongs to multiple fusion groups"
                    )
                groups_by_flow[flow_id] = group

        self._configuration_key = configuration_key
        self._groups_by_flow = groups_by_flow

    def execute_if_fused(
        self,
        *,
        target: "FlowDispatchTarget",
        bundle: PortBundle,
        context: RuntimeContext,
        stats: FusionStats,
    ) -> bool:
        group = self._groups_by_flow.get(target.flow_id)
        if group is None:
            return False
        handle = group.handle
        if handle is None:
            stats.fusion_eager_fallback_count += 1
            return False

        invocation_key = (bundle.key, int(bundle.version))
        pending = group.pending.get(invocation_key)
        if pending is None:
            pending = _PendingInvocation(context=context)
            group.pending[invocation_key] = pending
        if target.flow_id in pending.bundles:
            raise RuntimeError(
                f"fusion group {group.flow_ids!r} received flow {target.flow_id!r} "
                f"twice for bundle invocation {invocation_key!r}"
            )
        pending.bundles[target.flow_id] = bundle
        if len(pending.bundles) < len(group.flow_ids):
            return True

        del group.pending[invocation_key]
        ordered_bundles = {
            flow_id: pending.bundles[flow_id]
            for flow_id in group.flow_ids
        }
        before = int(handle.replay_count())
        handle.execute(ordered_bundles, pending.context)
        after = int(handle.replay_count())
        stats.fusion_execution_count += 1
        if after > before:
            stats.fusion_replay_count += after - before
        else:
            stats.fusion_eager_fallback_count += 1
        return True

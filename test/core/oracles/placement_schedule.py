#!/usr/bin/env python3
"""REFERENCE oracle — 仅作 Rust core 对照基准,禁止运行时引用。

Reference implementation of the B3 dependency-aware placement compiler
(``tllm._core.compile_placement_schedule``). Sunset plan: once the Rust core
shows oracle matrix equivalence + dual-runner parity for two consecutive
development cycles, delete this file and evolve only in Rust.
"""

from __future__ import annotations

from collections.abc import Sequence

from tllm.ports.address import PortAddressDesc, port_address_label
from tllm.runtime.placement import PlacementFlowDesc, VersionedStateResourceDesc

HOOK_POINTS = (
    "load_model.pre",
    "load_model.post",
    "prepare_inputs.pre",
    "prepare_inputs.post",
    "layer.pre",
    "layer.post",
    "block.end",
    "stack.end",
    "execute_model.pre",
    "execute_model.post",
    "sample_tokens.pre",
    "sampler.pre_draw",
    "sample_tokens.post",
)
IN_CAPTURE_HOOKS = ("layer.pre", "layer.post", "block.end", "stack.end")
MOBILITIES = ("pinned", "deferrable", "advanceable", "free")
DEADLINES = ("step_end", "next_prepare", "generate_end")
OBSERVATIONS = ("at_declared_window", "latest_at_execution", "stale_ok")
GRAPH_MODES = ("eager", "piecewise", "full")
LEASE_LIFETIMES = ("consume_call", "step")
DEFAULT_BUBBLE_SCORES = {
    "sampler.pre_draw": 120,
    "sample_tokens.pre": 100,
    "prepare_inputs.post": 60,
    "sample_tokens.post": 40,
    "execute_model.post": 20,
    "execute_model.pre": 10,
}


class OraclePlacementError(RuntimeError):
    pass


def _qr(value: str | PortAddressDesc | VersionedStateResourceDesc) -> str:
    if isinstance(value, PortAddressDesc):
        text = port_address_label(value)
    elif isinstance(value, VersionedStateResourceDesc):
        text = value.label
    else:
        text = value
    return f"'{text}'"


def _qr_list(v) -> str:
    return "[" + ", ".join(_qr(x) for x in v) + "]"


def _dependency_graph(flows: Sequence[PlacementFlowDesc]):
    count = len(flows)
    successors = [set() for _ in flows]
    predecessors = [set() for _ in flows]
    flow_ids: set[str] = set()
    read_sets: list[set[PortAddressDesc]] = []
    write_orders: list[dict[PortAddressDesc, int]] = []

    def same_resource(
        left: VersionedStateResourceDesc,
        right: VersionedStateResourceDesc,
    ) -> bool:
        return (
            left.kind == right.kind
            and left.namespace == right.namespace
            and left.name == right.name
        )

    def add_edge(source: int, target: int) -> None:
        if source == target or target in successors[source]:
            return
        successors[source].add(target)
        predecessors[target].add(source)

    for flow in flows:
        if flow.flow_id in flow_ids:
            raise OraclePlacementError(
                f"duplicate flow_id {_qr(flow.flow_id)} in placement input"
            )
        flow_ids.add(flow.flow_id)
        read_sets.append(set(flow.reads))
        orders: dict[PortAddressDesc, int] = {}
        for write in flow.writes:
            if write.port in orders:
                raise OraclePlacementError(
                    f"flow {_qr(flow.flow_id)} writes port {_qr(write.port)} more than once"
                )
            orders[write.port] = write.order
        write_orders.append(orders)
        resources: set[VersionedStateResourceDesc] = set()
        for write in flow.resource_writes:
            if write.resource in resources:
                raise OraclePlacementError(
                    f"flow {_qr(flow.flow_id)} writes state resource "
                    f"{_qr(write.resource)} more than once"
                )
            resources.add(write.resource)

    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            left = flows[left_index]
            right = flows[right_index]
            same_consumer = bool(left.consumer_id) and left.consumer_id == right.consumer_id
            raw_or_war = bool(
                (set(write_orders[left_index]) & read_sets[right_index])
                or (read_sets[left_index] & set(write_orders[right_index]))
            )
            if same_consumer or raw_or_war:
                add_edge(left_index, right_index)

            for left_write in left.writes:
                right_order = write_orders[right_index].get(left_write.port)
                if right_order is None:
                    continue
                if left_write.order == right_order:
                    raise OraclePlacementError(
                        f"flows {_qr(left.flow_id)} and {_qr(right.flow_id)} write port "
                        f"{_qr(left_write.port)} with duplicate order {left_write.order}"
                    )
                source, target = (
                    (left_index, right_index)
                    if left_write.order < right_order
                    else (right_index, left_index)
                )
                add_edge(source, target)

            for left_read in left.resource_reads:
                for right_write in right.resource_writes:
                    resource = right_write.resource
                    if not same_resource(left_read, resource):
                        continue
                    source, target = (
                        (left_index, right_index)
                        if left_read.version < resource.version
                        else (right_index, left_index)
                    )
                    add_edge(source, target)

            for right_read in right.resource_reads:
                for left_write in left.resource_writes:
                    resource = left_write.resource
                    if not same_resource(right_read, resource):
                        continue
                    source, target = (
                        (left_index, right_index)
                        if resource.version <= right_read.version
                        else (right_index, left_index)
                    )
                    add_edge(source, target)

            for left_write in left.resource_writes:
                resource = left_write.resource
                for right_write in right.resource_writes:
                    right_resource = right_write.resource
                    if not same_resource(resource, right_resource):
                        continue
                    if resource.version == right_resource.version:
                        raise OraclePlacementError(
                            f"flows {_qr(left.flow_id)} and {_qr(right.flow_id)} write state resource "
                            f"{_qr(right_resource)} with duplicate version {resource.version}"
                        )
                    source, target = (
                        (left_index, right_index)
                        if resource.version < right_resource.version
                        else (right_index, left_index)
                    )
                    add_edge(source, target)
    return successors, predecessors


def _topological_order(
    flows: Sequence[PlacementFlowDesc],
    successors: Sequence[set[int]],
    predecessors: Sequence[set[int]],
) -> list[int]:
    remaining_incoming = [len(edges) for edges in predecessors]
    ready = {index for index, incoming in enumerate(remaining_incoming) if incoming == 0}
    topo: list[int] = []
    while ready:
        index = min(ready)
        ready.remove(index)
        topo.append(index)
        for successor in sorted(successors[index]):
            remaining_incoming[successor] -= 1
            if remaining_incoming[successor] == 0:
                ready.add(successor)
    if len(topo) != len(flows):
        cycle_ids = [flows[index].flow_id for index in range(len(flows)) if remaining_incoming[index] > 0]
        raise OraclePlacementError(
            f"placement dependency cycle among flows {_qr_list(cycle_ids)}"
        )
    return topo


def oracle_execution_order(flows: Sequence[PlacementFlowDesc]) -> list[str]:
    successors, predecessors = _dependency_graph(flows)
    return [flow_id for flow_id in (flows[index].flow_id for index in _topological_order(flows, successors, predecessors))]


def oracle_compile_placement_schedule(
    flows: Sequence[PlacementFlowDesc],
    graph_mode: str,
    hook_costs: Sequence[object] | None = None,
):
    """Return a dependency-safe schedule ``[(flow_id, hook)]`` or raise.

    ``flows`` also carries plain consumer_id/read/write dependency descriptors.
    """
    if graph_mode not in GRAPH_MODES:
        raise OraclePlacementError(
            f"unsupported graph mode {_qr(graph_mode)}; expected one of {_qr_list(GRAPH_MODES)}"
        )
    placements = []
    for flow in flows:
        flow_id = flow.flow_id
        declared_hook = flow.declared_hook
        mobility = flow.mobility or "pinned"
        deadline = flow.deadline or "step_end"
        max_defer = int(flow.max_defer_steps or 0)
        observes = flow.observes_version or "at_declared_window"
        if declared_hook not in HOOK_POINTS:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unknown hook point {_qr(declared_hook)}; "
                f"expected one of {_qr_list(HOOK_POINTS)}"
            )
        if flow.earliest_hook is not None and flow.earliest_hook not in HOOK_POINTS:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unknown earliest hook {_qr(flow.earliest_hook)}; "
                f"expected one of {_qr_list(HOOK_POINTS)}"
            )
        for allowed_hook in flow.allowed_hooks:
            if allowed_hook not in HOOK_POINTS:
                raise OraclePlacementError(
                    f"flow {_qr(flow_id)} allows unknown hook point {_qr(allowed_hook)}; "
                    f"expected one of {_qr_list(HOOK_POINTS)}"
                )
        if mobility not in MOBILITIES:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unsupported timing mobility {_qr(mobility)}; "
                f"expected one of {_qr_list(MOBILITIES)}"
            )
        if deadline not in DEADLINES:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unsupported timing deadline {_qr(deadline)}; "
                f"expected one of {_qr_list(DEADLINES)}"
            )
        if observes not in OBSERVATIONS:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unsupported version observation {_qr(observes)}; "
                f"expected one of {_qr_list(OBSERVATIONS)}"
            )
        if max_defer < 0:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares negative max_defer_steps {max_defer}"
            )
        if mobility == "pinned":
            if max_defer != 0:
                raise OraclePlacementError(
                    f"flow {_qr(flow_id)} is pinned but declares max_defer_steps={max_defer}"
                )
            if observes != "at_declared_window":
                raise OraclePlacementError(
                    f"flow {_qr(flow_id)} is pinned but declares observes_version={_qr(observes)}"
                )
        if mobility in ("deferrable", "free") and max_defer < 1:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} is {mobility} but declares max_defer_steps={max_defer} (must be >=1)"
            )
        if mobility in ("deferrable", "free") and deadline == "step_end":
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} is {mobility} with deadline 'step_end'; "
                "deferral is across steps, so this admits no movement"
            )
        if mobility == "advanceable" and observes == "at_declared_window":
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} is advanceable but observes 'at_declared_window'; "
                "running earlier can never observe that version"
            )
        if flow.lease_lifetime not in LEASE_LIFETIMES:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares unsupported lease_lifetime {_qr(flow.lease_lifetime)}; "
                f"expected one of {_qr_list(LEASE_LIFETIMES)}"
            )
        if mobility != "pinned" and flow.delivery == "device_lease":
            if flow.ownership != "runtime_lease" or flow.lease_lifetime != "step":
                raise OraclePlacementError(
                    f"flow {_qr(flow_id)} uses timing-movable device_lease delivery; this requires "
                    "ownership='runtime_lease' and lease_lifetime='step'"
                )
            if deadline != "next_prepare":
                raise OraclePlacementError(
                    f"flow {_qr(flow_id)} uses a timing-movable step-lifetime device lease but declares "
                    f"deadline {_qr(deadline)}; deadline 'next_prepare' is required"
                )
        if graph_mode != "eager" and mobility != "pinned" and declared_hook in IN_CAPTURE_HOOKS:
            raise OraclePlacementError(
                f"flow {_qr(flow_id)} declares mobility {_qr(mobility)} at in-capture hook "
                f"{_qr(declared_hook)} under graph mode {_qr(graph_mode)}; "
                "timing movement across the capture boundary requires the full placement compiler — "
                "pin the flow or use graph mode 'eager'"
            )
        placements.append((flow_id, declared_hook))

    successors, predecessors = _dependency_graph(flows)
    topo = _topological_order(flows, successors, predecessors)
    count = len(flows)

    lower = [0] * count
    upper = [len(HOOK_POINTS) - 1] * count
    for index, flow in enumerate(flows):
        declared = HOOK_POINTS.index(flow.declared_hook)
        earliest = declared if flow.earliest_hook is None else HOOK_POINTS.index(flow.earliest_hook)
        mobility = flow.mobility or "pinned"
        if mobility == "pinned":
            lower[index] = declared
            upper[index] = declared
        elif mobility == "deferrable":
            lower[index] = declared
        elif mobility == "advanceable":
            lower[index] = min(earliest, declared)
            upper[index] = declared
        elif mobility == "free":
            lower[index] = min(earliest, declared)
    for index in reversed(topo):
        for successor in successors[index]:
            upper[index] = min(upper[index], upper[successor])

    score_overrides = {
        str(cost.hook): int(cost.bubble_score)
        for cost in (hook_costs or ())
    }
    selected = [0] * count
    for index in topo:
        for predecessor in predecessors[index]:
            lower[index] = max(lower[index], selected[predecessor])
        flow = flows[index]
        declared = HOOK_POINTS.index(flow.declared_hook)
        allowed = set(flow.allowed_hooks) if flow.allowed_hooks else set(HOOK_POINTS)
        movable_step_lease = (
            (flow.mobility or "pinned") != "pinned"
            and flow.delivery == "device_lease"
            and flow.ownership == "runtime_lease"
            and flow.lease_lifetime == "step"
        )
        best: tuple[int, int, int] | None = None
        if lower[index] <= upper[index]:
            for hook_index in range(lower[index], upper[index] + 1):
                hook = HOOK_POINTS[hook_index]
                if hook not in allowed:
                    continue
                if (
                    movable_step_lease
                    and hook_index != declared
                    and (
                        flow.declared_hook != "execute_model.post"
                        or hook not in ("sample_tokens.pre", "sampler.pre_draw", "sample_tokens.post")
                    )
                ):
                    continue
                if (
                    graph_mode != "eager"
                    and hook in IN_CAPTURE_HOOKS
                    and hook_index != declared
                ):
                    continue
                bubble = score_overrides.get(hook, DEFAULT_BUBBLE_SCORES.get(hook, 0))
                score = (
                    bubble,
                    -abs(declared - hook_index),
                    hook_index,
                )
                if best is None or score > best:
                    best = score
        if best is None:
            raise OraclePlacementError(
                f"flow {_qr(flow.flow_id)} has no legal placement after dependency, mobility, graph, "
                "and backend-hook constraints"
            )
        selected[index] = best[2]
        placements[index] = (flow.flow_id, HOOK_POINTS[best[2]])
    return placements

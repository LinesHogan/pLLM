"""Typed lowering from existing runtime plans to the shared Phase 1a IR."""

from __future__ import annotations

from typing import Sequence, cast

from tllm.ports.consumer import BaseConsumer
from tllm.ports.address import PortAddressDesc, port_address_desc
from tllm.ports.base import (
    ConsumerFlow,
    PortKind,
    PortRead,
    PortWrite,
    TimingContract,
    TimingMobility,
    VersionObservation,
)
from tllm.ports.port_id import PortId
from tllm.producers.registry import ExecutionPlan, PlannedProducer
from tllm.workgraph.dispatch import DispatchPlanLike
from tllm.workgraph.anchors import (
    BackendCompatibilityError,
    anchor_desc,
    is_runtime_non_anchor_hook,
    validate_backend_snapshot,
)
from tllm.workgraph.ir import (
    AccessDesc,
    AccessMode,
    AnchorDesc,
    BackendSnapshot,
    BoundaryKind,
    DataCentricIR,
    EdgeKind,
    ExecutionNodeDesc,
    ExecutorKey,
    LeaseDesc,
    NodeKind,
    NodeKey,
    GraphSafetyKind,
    ResidenceDesc,
    RuntimePlanBinding,
    SemanticGraph,
    SemanticNodeDesc,
    TimingDesc,
    TimingDeadlineKind,
    TimingMobilityKind,
    TimingWindowKind,
    ValueDesc,
    VersionObservationKind,
)


class WorkGraphError(RuntimeError):
    """Deterministic contract or plan-parity failure."""

    def __init__(self, code: str, message: str, *, node_key: str = "") -> None:
        self.code = str(code)
        self.node_key = str(node_key)
        suffix = f" node={self.node_key!r}" if self.node_key else ""
        super().__init__(f"[{self.code}]{suffix} {message}")


def _address_value(port: PortRead | PortWrite) -> ValueDesc:
    address = port_address_desc(port)
    return ValueDesc(
        value_id=f"address-sha256:{address.identity}",
        version=1,
        port=address,
    )


def _port_value(port: PortId, *, kind: str = "producer_port") -> ValueDesc:
    return ValueDesc(
        value_id=f"port:{port}",
        version=int(port.version),
        port=port,
    )


def _phase(address: PortAddressDesc) -> str:
    for field in address.locator_fields:
        if field.path == ("phase",):
            return field.value
    return ""


def _flow_node(
    *,
    flow_id: str,
    consumer_id: str,
    flow: ConsumerFlow,
    placement_hook: str,
    backend: BackendSnapshot,
    strict_native: bool,
) -> tuple[SemanticNodeDesc, tuple[ValueDesc, ...], tuple[str, ...], AnchorDesc | None]:
    raw_timing = flow.timing
    timing_contract = cast(TimingContract, raw_timing)
    timing = TimingDesc(
        window=TimingWindowKind(str(flow.window)),
        mobility=TimingMobilityKind(timing_contract.mobility.value),
        deadline=TimingDeadlineKind(timing_contract.deadline.value),
        max_defer_steps=timing_contract.max_defer_steps,
        observes_version=VersionObservationKind(timing_contract.observes_version.value),
    )
    ordinal = int(flow_id.rsplit("#", 1)[1])
    node_key = NodeKey.consumer_flow(consumer_id, ordinal)
    anchor_note: str | None = None
    runtime_hook_boundary = False
    try:
        anchor: AnchorDesc | None = anchor_desc(backend, placement_hook)
    except BackendCompatibilityError:
        if not is_runtime_non_anchor_hook(placement_hook):
            raise WorkGraphError(
                "WG121_UNSUPPORTED_ANCHOR",
                f"selected runtime hook {placement_hook!r} is not a supported "
                "backend anchor or known legacy runtime boundary",
                node_key=flow_id,
            )
        # Preserve the authoritative selected hook without inventing a
        # different lifecycle point. The runtime plan remains the executor.
        anchor = None
        runtime_hook_boundary = True
        anchor_note = f"{flow_id}:legacy_runtime_hook:{placement_hook}"
    values: list[ValueDesc] = []
    reads: list[AccessDesc] = []
    writes: list[AccessDesc] = []
    for read in flow.reads:
        value = _address_value(read)
        values.append(value)
        reads.append(
            AccessDesc(
                node_key=node_key,
                value=value,
                mode=AccessMode.READ,
                port=value.port,
                subset=(str(read.role),) if str(read.role) else (),
            )
        )
    for write in flow.writes:
        value = _address_value(write)
        values.append(value)
        writes.append(
            AccessDesc(
                node_key=node_key,
                value=value,
                mode=AccessMode.WRITE,
                port=value.port,
                subset=(str(write.role),) if str(write.role) else (),
            )
        )
    legacy_reasons: list[str] = []
    if runtime_hook_boundary:
        if strict_native:
            raise WorkGraphError(
                "WG121_UNSUPPORTED_ANCHOR",
                "native Phase 1a nodes cannot attach to a layer-owned runtime hook",
                node_key=flow_id,
            )
        legacy_reasons.append("runtime_hook_boundary")
    if timing_contract.observes_version != VersionObservation.AT_DECLARED_WINDOW:
        if strict_native:
            raise WorkGraphError(
                "WG110_UNSUPPORTED_VERSION",
                "Phase 1a native nodes require observes_version='at_declared_window'",
                node_key=flow_id,
            )
        legacy_reasons.append("non_declared_version_observation")
    kind = NodeKind.COMPUTE
    executor = ExecutorKey("consumer", consumer_id)
    if timing_contract.mobility != TimingMobility.PINNED:
        if strict_native:
            raise WorkGraphError(
                "WG111_UNSUPPORTED_MOBILITY",
                "Phase 1a native nodes require mobility='pinned'",
                node_key=flow_id,
            )
        legacy_reasons.append("non_pinned_flow")
    if backend.graph_mode == "full":
        if strict_native:
            raise WorkGraphError(
                "WG120_UNSUPPORTED_GRAPH_MODE",
                "Phase 1a native nodes support only eager and piecewise graph modes",
                node_key=flow_id,
            )
        legacy_reasons.append("full_graph_mode_legacy_only")
    legacy_reason = ",".join(legacy_reasons) if legacy_reasons else None
    if legacy_reason is not None:
        executor = ExecutorKey("legacy", consumer_id)
    graph_safety = GraphSafetyKind.EAGER_ONLY
    node = SemanticNodeDesc(
        node_key=node_key,
        kind=kind,
        executor=executor,
        reads=tuple(reads),
        writes=tuple(writes),
        effects=("opaque_legacy_effect",),
        timing=timing,
        anchor=None,
        graph_safety=graph_safety,
        source=flow_id,
        legacy_boundary=BoundaryKind.LEGACY if legacy_reason is not None else None,
    )
    node_notes: list[str] = []
    if legacy_reason is not None:
        node_notes.append(f"{flow_id}:legacy_boundary:{legacy_reason}")
    if anchor_note is not None:
        node_notes.append(anchor_note)
    return node, tuple(values), tuple(node_notes), anchor


def _producer_node(
    planned: PlannedProducer,
    backend: BackendSnapshot,
) -> tuple[SemanticNodeDesc, tuple[ValueDesc, ...], tuple[AnchorDesc, ...], tuple[str, ...]]:
    producer_id = planned.entry.producer_id
    node_key = NodeKey.producer(producer_id)
    outputs = tuple(_port_value(port) for port in planned.output_ports)
    reads = tuple(
        AccessDesc(
            node_key=node_key,
            value=_port_value(port),
            mode=AccessMode.READ,
            port=port,
        )
        for port in planned.depends_on
    )
    writes = tuple(
        AccessDesc(node_key=node_key, value=value, mode=AccessMode.WRITE, port=value.port)
        for value in outputs
    )
    ordered_hooks = tuple(sorted(hook.value for hook in planned.emits_at))
    anchors: list[AnchorDesc] = []
    node_notes: list[str] = []
    legacy_boundary = False
    for hook in ordered_hooks:
        try:
            anchors.append(anchor_desc(backend, hook))
        except BackendCompatibilityError:
            if not is_runtime_non_anchor_hook(hook):
                raise WorkGraphError(
                    "WG121_UNSUPPORTED_ANCHOR",
                    f"producer {producer_id!r} selects unsupported hook "
                    f"{hook!r}",
                    node_key=str(node_key),
                )
            # A provider may materialize at a layer hook that is not a Phase
            # 1a external anchor. Preserve the hook in its effect contract;
            # execution still remains owned by the existing producer plan.
            legacy_boundary = True
            node_notes.append(f"{producer_id}:legacy_runtime_hook:{hook}")
    if len(ordered_hooks) != 1:
        # A single WorkNode cannot faithfully mean "materialize at each of
        # these hook invocations". Until stage lowering exists, retain the
        # authoritative producer plan as a conservative opaque boundary.
        legacy_boundary = True
        node_notes.append(f"{producer_id}:legacy_multi_hook:{','.join(ordered_hooks)}")
    native_anchors = tuple(anchors) if not legacy_boundary and len(anchors) == 1 else ()
    effects = tuple(f"emits_at:{hook}" for hook in ordered_hooks)
    if legacy_boundary:
        effects = (*effects, "opaque_legacy_effect")
    node = SemanticNodeDesc(
        node_key=node_key,
        kind=NodeKind.PROVIDER,
        executor=ExecutorKey("legacy" if legacy_boundary else "producer", producer_id),
        reads=reads,
        writes=writes,
        effects=effects,
        anchor=None,
        graph_safety=GraphSafetyKind(planned.entry.producer.graph_safety.name.lower()),
        source=producer_id,
        legacy_boundary=BoundaryKind.LEGACY if legacy_boundary else None,
    )
    return node, outputs, native_anchors, tuple(node_notes)


def lower_plans(
    *,
    consumers: Sequence[BaseConsumer[object]],
    dispatch_plan: DispatchPlanLike,
    execution_plan: ExecutionPlan,
    backend: BackendSnapshot,
    strict_native: bool = False,
) -> tuple[SemanticGraph, DataCentricIR, RuntimePlanBinding, tuple[str, ...], tuple[str, ...]]:
    """Lower authoritative runtime plans without selecting a new schedule."""
    del consumers  # declarations are already represented by the authoritative plans
    try:
        validate_backend_snapshot(backend)
    except BackendCompatibilityError as exc:
        raise WorkGraphError("WG121_INVALID_BACKEND", str(exc)) from exc
    nodes: list[SemanticNodeDesc] = []
    values_by_key: dict[str, ValueDesc] = {}
    notes: list[str] = []
    legacy: list[str] = []
    attached_anchors: list[tuple[NodeKey, AnchorDesc]] = []
    flow_by_id: dict[str, ConsumerFlow] = {}
    for target in dispatch_plan.flow_targets():
        flow_by_id[target.flow_id] = target.flow
        node, values, node_notes, anchor = _flow_node(
            flow_id=target.flow_id,
            consumer_id=target.flow_id.rsplit("#", 1)[0],
            flow=target.flow,
            placement_hook=target.placement_hook,
            backend=backend,
            strict_native=strict_native,
        )
        nodes.append(node)
        if anchor is not None:
            attached_anchors.append((node.node_key, anchor))
        notes.extend(node_notes)
        if node.legacy_boundary is not None:
            legacy.append(node.source)
        for value in values:
            values_by_key.setdefault(value.key, value)

    for planned in execution_plan.producers:
        node, values, anchors, node_notes = _producer_node(planned, backend)
        nodes.append(node)
        attached_anchors.extend((node.node_key, anchor) for anchor in anchors)
        notes.extend(node_notes)
        if node.legacy_boundary is not None:
            legacy.append(node.source)
        for value in values:
            values_by_key.setdefault(value.key, value)
        for value in planned.depends_on:
            dependency_value = _port_value(value)
            values_by_key.setdefault(dependency_value.key, dependency_value)

    # Anchors are first-class semantic/data nodes.  Each attached flow or
    # provider points from its fixed backend lifecycle node, but no lifecycle
    # ordering is invented beyond that fixed attachment in Phase 1a.
    anchors_by_key = {
        anchor.node_key: anchor
        for _attached_node_key, anchor in attached_anchors
    }
    for anchor_key in sorted(anchors_by_key, key=str):
        anchor = anchors_by_key[anchor_key]
        nodes.append(
            SemanticNodeDesc(
                node_key=anchor.node_key,
                kind=NodeKind.ANCHOR,
                anchor=anchor,
                source=f"anchor:{anchor.node_key}",
            )
        )

    flow_order = tuple(target.flow_id for target in dispatch_plan.flow_targets())
    producer_order = tuple(planned.entry.producer_id for planned in execution_plan.producers)
    node_by_source = {node.source: node for node in nodes}
    edges: list[tuple[NodeKey, NodeKey, EdgeKind]] = []
    for left, right in zip(flow_order, flow_order[1:]):
        edges.append((node_by_source[left].node_key, node_by_source[right].node_key, EdgeKind.CONTROL))
    for left, right in zip(producer_order, producer_order[1:]):
        edges.append((node_by_source[left].node_key, node_by_source[right].node_key, EdgeKind.CONTROL))
    for attached_node_key, anchor in attached_anchors:
        edges.append((anchor.node_key, attached_node_key, EdgeKind.ANCHOR))

    selected_outputs: list[tuple[str, PortKind, tuple[str, ...]]] = []
    for planned in execution_plan.producers:
        producer_id = planned.entry.producer_id
        for spec in planned.entry.producer.outputs:
            if execution_plan.port_sources.get(spec.id) == producer_id:
                selected_outputs.append((producer_id, spec.kind, tuple(spec.phases)))
    emitted_binding_edges: set[tuple[NodeKey, NodeKey]] = set()
    emitted_binding_notes: set[str] = set()
    for flow_id, flow in flow_by_id.items():
        for read in flow.reads:
            address = port_address_desc(read)
            phase = _phase(address)
            matches = tuple(
                sorted(
                    {
                        producer_id
                        for producer_id, kind, phases in selected_outputs
                        if kind == read.kind and (not phase or not phases or phase in phases)
                    }
                )
            )
            if len(matches) == 1:
                edge = (node_by_source[matches[0]].node_key, node_by_source[flow_id].node_key)
                if edge not in emitted_binding_edges:
                    # Kind/phase matching is only a coarse runtime-plan
                    # diagnostic.  The producer PortId and consumer address
                    # remain distinct values, so this is not provenance.
                    edges.append((*edge, EdgeKind.CONTROL))
                    emitted_binding_edges.add(edge)
                note = f"{flow_id}:coarse_kind_phase_binding:{matches[0]}"
                if note not in emitted_binding_notes:
                    notes.append(note)
                    emitted_binding_notes.add(note)
            elif len(matches) > 1:
                note = f"{flow_id}:coarse_kind_phase_binding:ambiguous"
                if note not in emitted_binding_notes:
                    notes.append(note)
                    emitted_binding_notes.add(note)

    values = tuple(sorted(values_by_key.values(), key=lambda value: value.key))
    graph = SemanticGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        values=values,
        backend=backend,
        graph_mode=backend.graph_mode,
    )
    residences = tuple(
        ResidenceDesc(value=value, device=backend.device, rank=backend.rank)
        for value in values
    )
    accesses = tuple(
        access
        for node in graph.nodes
        for access in (*node.reads, *node.writes)
    )
    leases = tuple(
        LeaseDesc(
            value=access.value,
            storage="backend_owned",
            live_start=str(node.node_key),
            live_end=str(node.node_key),
            stable_address=False,
            materialized=False,
        )
        for node in graph.nodes
        if node.kind == NodeKind.COMPUTE and node.legacy_boundary is not None
        for access in node.reads
    )
    data_nodes = tuple(
        ExecutionNodeDesc(
            node_key=node.node_key,
            semantic_key=node.node_key,
            kind=node.kind,
            executor=node.executor,
            accesses=tuple((*node.reads, *node.writes)),
            residence_keys=tuple(access.value.key for access in (*node.reads, *node.writes)),
            lease_keys=tuple(
                lease.value.key for lease in leases if lease.live_start == str(node.node_key)
            ),
            device=backend.device,
            rank=backend.rank,
            anchor=node.anchor,
            graph_safety=node.graph_safety,
            fusion_group=next(
                (
                    "fusion/" + ",".join(group)
                    for group in dispatch_plan.fusion_groups
                    if node.source in group
                ),
                "",
            ),
        )
        for node in graph.nodes
    )
    data = DataCentricIR(
        nodes=data_nodes,
        values=values,
        accesses=accesses,
        residences=residences,
        moves=(),
        leases=leases,
        edges=graph.edges,
        backend=backend,
        graph_mode=backend.graph_mode,
    )
    binding = RuntimePlanBinding(
        dispatch_flow_ids=flow_order,
        dispatch_hooks=tuple((target.flow_id, target.placement_hook) for target in dispatch_plan.flow_targets()),
        dispatch_fusion_groups=tuple(dispatch_plan.fusion_groups),
        execution_producer_ids=producer_order,
        producer_hooks=tuple(
            (planned.entry.producer_id, tuple(sorted(hook.value for hook in planned.emits_at)))
            for planned in execution_plan.producers
        ),
        capture_resident_producer_ids=tuple(execution_plan.capture_resident_producer_ids),
        execution_port_sources=tuple(
            sorted((str(port), str(producer_id)) for port, producer_id in execution_plan.port_sources.items())
        ),
    )
    return graph, data, binding, tuple(legacy), tuple(notes)


__all__ = [
    "RuntimePlanBinding",
    "WorkGraphError",
    "lower_plans",
]

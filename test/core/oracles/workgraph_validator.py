"""Small Python differential oracle for the Phase 1a Rust validator."""

from __future__ import annotations

from collections import Counter, defaultdict

from tllm.ports.address import PortAddressDesc
from tllm.ports.port_id import PortId


_NODE_KINDS = {
    "anchor",
    "compute",
    "provider",
    "transfer",
    "collective",
    "barrier",
    "state",
}
_GRAPH_SAFETY_KINDS = {"eager_only", "capture_safe", "replay_stateful"}
_TIMING_WINDOWS = {"background", "same_step", "next_step", "out_of_band"}
_TIMING_DEADLINES = {"step_end", "next_prepare", "generate_end"}


def _key(node_key: object) -> str:
    if node_key.namespace == "anchor":
        return f"{node_key.namespace}/{node_key.owner}"
    return f"{node_key.namespace}/{node_key.owner}/{node_key.ordinal}"


def _port_signature(port: object | None) -> tuple[object, ...] | None:
    if port is None:
        return None
    if isinstance(port, PortId):
        return ("port_id", port.namespace, port.name, port.version)
    if isinstance(port, PortAddressDesc):
        return (
            "port_address",
            port.kind,
            port.locator_type,
            tuple((field.path, field.value_type, field.value) for field in port.locator_fields),
        )
    raise ValueError("port reference")


def _value_key(value: object) -> str:
    return f"{value.value_id}@{value.version}"


def _value_signature(value: object) -> tuple[object, ...]:
    return (
        value.value_id,
        value.version,
        _port_signature(value.port),
        tuple(value.shape),
        value.dtype,
        value.layout,
        value.alias_set,
    )


def _access_signature(access: object) -> tuple[object, ...]:
    return (
        _key(access.node_key),
        _value_signature(access.value),
        str(access.mode),
        _port_signature(access.port),
        tuple(access.subset),
        access.consistency,
    )


def _access_counts(accesses: object) -> Counter[tuple[object, ...]]:
    return Counter(_access_signature(access) for access in accesses)


def _backend_signature(backend: object) -> tuple[object, ...]:
    return (
        backend.backend_id,
        backend.runner_family,
        backend.graph_mode,
        backend.backend_version,
        tuple(backend.capabilities),
        backend.device,
        backend.rank,
        backend.tp_size,
        backend.pp_size,
        backend.dp_size,
        tuple(backend.topology),
    )


def _validate_backend(backend: object, graph_mode: str) -> None:
    if not backend.backend_id or "/" in backend.backend_id or "\\" in backend.backend_id:
        raise ValueError("backend")
    if not backend.backend_version.startswith("0.27."):
        raise ValueError("backend version family")
    if backend.runner_family not in {"mrv1", "mrv2", "unresolved"}:
        raise ValueError("runner family")
    if backend.graph_mode not in {"eager", "piecewise"} or backend.graph_mode != graph_mode:
        raise ValueError("graph mode")
    if (
        not backend.device
        or backend.rank < 0
        or backend.tp_size < 1
        or backend.pp_size < 1
        or backend.dp_size < 1
    ):
        raise ValueError("backend placement")


def _edge_set(edges: object) -> set[tuple[str, str, str]]:
    return {(_key(source), _key(target), str(kind)) for source, target, kind in edges}


def _validate_edges(node_keys: list[str], edges: object) -> tuple[str, ...]:
    known = set(node_keys)
    seen: set[tuple[str, str, str]] = set()
    successors: dict[str, set[str]] = defaultdict(set)
    indegree = {key: 0 for key in node_keys}
    for source, target, kind in edges:
        edge = (_key(source), _key(target), str(kind))
        if edge[0] not in known or edge[1] not in known:
            raise ValueError("unknown node")
        if edge[2] not in {"data", "control", "effect", "lease", "collective", "anchor"}:
            raise ValueError("edge kind")
        if edge in seen:
            raise ValueError("duplicate edge")
        if edge[0] == edge[1]:
            raise ValueError("cycle")
        seen.add(edge)
        if edge[1] not in successors[edge[0]]:
            successors[edge[0]].add(edge[1])
            indegree[edge[1]] += 1
    ready = sorted(key for key, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        key = ready.pop(0)
        order.append(key)
        for successor in sorted(successors[key]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(node_keys):
        raise ValueError("cycle")
    return tuple(order)


def _validate_access(access: object, values: set[str], owner: str | None = None) -> None:
    if _value_key(access.value) not in values:
        raise ValueError("unknown value")
    if str(access.mode) not in {"read", "write", "read_write"}:
        raise ValueError("access mode")
    if not access.consistency:
        raise ValueError("access consistency")
    if owner is not None and _key(access.node_key) != owner:
        raise ValueError("access node")
    if access.port is not None and access.value.port is not None:
        if _port_signature(access.port) != _port_signature(access.value.port):
            raise ValueError("access port")


def _validate_anchor(anchor: object, backend: object, graph_mode: str, anchors: dict[str, object]) -> None:
    anchor_key = f"anchor/{anchor.backend}/{anchor.runner}/{anchor.graph_mode}/{anchor.hook}"
    if (
        anchor.backend != backend.backend_id
        or anchor.runner != backend.runner_family
        or anchor.graph_mode != graph_mode
        or anchor.hook
        not in {
            "prepare_inputs.pre",
            "prepare_inputs.post",
            "execute_model.pre",
            "execute_model.post",
            "sample_tokens.pre",
            "sampler.pre_draw",
            "sample_tokens.post",
        }
        or not anchor.externally_owned
        or anchor.mobility != "pinned"
        or not anchor.service_class
    ):
        raise ValueError("anchor")
    previous = anchors.get(anchor_key)
    if previous is not None and previous != anchor:
        raise ValueError("duplicate anchor")
    anchors[anchor_key] = anchor


def _validate_explicit_anchor(node: object, layer: str) -> None:
    if node.anchor is None:
        if node.kind == "anchor":
            raise ValueError(f"{layer} anchor")
        return
    anchor_key = f"anchor/{node.anchor.backend}/{node.anchor.runner}/{node.anchor.graph_mode}/{node.anchor.hook}"
    if node.kind != "anchor" or _key(node.node_key) != anchor_key:
        raise ValueError(f"{layer} anchor")


def _validate_node_domain(node: object, *, execution: bool) -> None:
    if node.kind not in _NODE_KINDS:
        raise ValueError("node kind")
    if node.graph_safety not in _GRAPH_SAFETY_KINDS:
        raise ValueError("graph safety")
    executor = node.executor
    if executor is not None and (
        not executor.namespace
        or not executor.name
        or not executor.version
        or "/" in executor.namespace
        or "/" in executor.name
    ):
        raise ValueError("executor")
    if not execution and node.legacy_boundary not in {None, "legacy_boundary"}:
        raise ValueError("legacy boundary")
    if execution and (not node.device or node.rank < 0):
        raise ValueError("execution placement")


def _validate_anchor_edges(nodes: object, edges: object, layer: str) -> None:
    kinds = {_key(node.node_key): node.kind for node in nodes}
    declared = {key for key, kind in kinds.items() if kind == "anchor"}
    attached: set[str] = set()
    for source, target, kind in edges:
        if str(kind) != "anchor":
            continue
        source_key = _key(source)
        target_key = _key(target)
        if kinds.get(source_key) != "anchor" or kinds.get(target_key) == "anchor":
            raise ValueError(f"{layer} anchor edge")
        attached.add(source_key)
    if declared != attached:
        raise ValueError(f"{layer} anchor attachment")


def validate_workgraph(semantic: object, data: object) -> tuple[str, ...]:
    if semantic.schema_version != 1 or data.schema_version != 1:
        raise ValueError("schema_version")
    if semantic.backend is None or data.backend is None:
        raise ValueError("backend")
    if semantic.graph_mode not in {"eager", "piecewise"}:
        raise ValueError("graph mode")
    if data.graph_mode != semantic.graph_mode:
        raise ValueError("graph mode")
    _validate_backend(semantic.backend, semantic.graph_mode)
    _validate_backend(data.backend, semantic.graph_mode)
    if _backend_signature(data.backend) != _backend_signature(semantic.backend):
        raise ValueError("backend parity")

    node_keys = [_key(node.node_key) for node in semantic.nodes]
    if len(node_keys) != len(set(node_keys)):
        raise ValueError("duplicate node")
    values = {_value_key(value) for value in semantic.values}
    if len(values) != len(semantic.values):
        raise ValueError("duplicate value")
    semantic_values = {_value_signature(value) for value in semantic.values}
    anchors: dict[str, object] = {}
    for node in semantic.nodes:
        _validate_node_domain(node, execution=False)
        legacy = node.legacy_boundary is not None
        if node.timing.window not in _TIMING_WINDOWS:
            raise ValueError("timing window")
        if node.timing.deadline not in _TIMING_DEADLINES:
            raise ValueError("timing deadline")
        if node.timing.mobility not in {"pinned", "deferrable", "advanceable", "free"}:
            raise ValueError("mobility")
        if node.timing.observes_version not in {
            "at_declared_window",
            "latest_at_execution",
            "stale_ok",
        }:
            raise ValueError("observation")
        if node.timing.max_defer_steps < 0:
            raise ValueError("max_defer_steps")
        if not legacy and node.timing.mobility != "pinned":
            raise ValueError("WorkGraph-native")
        if not legacy and node.timing.observes_version != "at_declared_window":
            raise ValueError("at_declared_window")
        _validate_explicit_anchor(node, "semantic")
        for access in (*node.reads, *node.writes):
            _validate_access(access, values, _key(node.node_key))
        if node.anchor is not None:
            _validate_anchor(node.anchor, semantic.backend, semantic.graph_mode, anchors)
    order = _validate_edges(node_keys, semantic.edges)
    _validate_anchor_edges(semantic.nodes, semantic.edges, "semantic")

    data_nodes = {_key(node.node_key) for node in data.nodes}
    if len(data_nodes) != len(data.nodes):
        raise ValueError("duplicate execution node")
    data_values = {_value_key(value) for value in data.values}
    if len(data_values) != len(data.values):
        raise ValueError("duplicate value")
    if set(node_keys) != data_nodes:
        raise ValueError("node parity")
    if values != data_values:
        raise ValueError("value parity")
    if semantic_values != {_value_signature(value) for value in data.values}:
        raise ValueError("value metadata parity")
    if data.moves:
        raise ValueError("Move descriptors")
    for node in data.nodes:
        _validate_node_domain(node, execution=True)
        if node.device != data.backend.device or node.rank != data.backend.rank:
            raise ValueError("execution placement parity")
        key = _key(node.node_key)
        if node.semantic_key is None or _key(node.semantic_key) != key:
            raise ValueError("semantic key")
        _validate_explicit_anchor(node, "execution")
        for access in node.accesses:
            _validate_access(access, data_values, key)
        if node.move_keys:
            raise ValueError("Move descriptors")
        if node.anchor is not None:
            _validate_anchor(node.anchor, semantic.backend, semantic.graph_mode, anchors)
    for access in data.accesses:
        _validate_access(access, data_values)
        if _key(access.node_key) not in data_nodes:
            raise ValueError("unknown node")

    residences = {_value_key(residence.value) for residence in data.residences}
    leases = {_value_key(lease.value) for lease in data.leases}
    if len(residences) != len(data.residences):
        raise ValueError("duplicate residence")
    if len(leases) != len(data.leases):
        raise ValueError("duplicate lease")
    for residence in data.residences:
        if residence.device != data.backend.device or residence.rank != data.backend.rank:
            raise ValueError("residence placement parity")
    for node in data.nodes:
        if set(node.residence_keys) - residences:
            raise ValueError("unknown residence")
        if set(node.lease_keys) - leases:
            raise ValueError("unknown lease")
    _validate_edges(list(data_nodes), data.edges)
    _validate_anchor_edges(data.nodes, data.edges, "execution")
    if _edge_set(semantic.edges) != _edge_set(data.edges):
        raise ValueError("edge parity")

    semantic_anchors = {_key(node.node_key): node.anchor for node in semantic.nodes}
    data_anchors = {_key(node.node_key): node.anchor for node in data.nodes}
    if semantic_anchors != data_anchors:
        raise ValueError("anchor parity")

    for semantic_node in semantic.nodes:
        execution = next(node for node in data.nodes if _key(node.node_key) == _key(semantic_node.node_key))
        if (
            semantic_node.kind != execution.kind
            or semantic_node.executor != execution.executor
            or semantic_node.graph_safety != execution.graph_safety
        ):
            raise ValueError("node projection parity")

    semantic_accesses = _access_counts(
        access for node in semantic.nodes for access in (*node.reads, *node.writes)
    )
    execution_accesses = _access_counts(access for node in data.nodes for access in node.accesses)
    if semantic_accesses != execution_accesses:
        raise ValueError("execution access parity")
    global_accesses = _access_counts(data.accesses)
    if semantic_accesses != global_accesses:
        raise ValueError("global access parity")
    for semantic_node in semantic.nodes:
        key = _key(semantic_node.node_key)
        declared = _access_counts((*semantic_node.reads, *semantic_node.writes))
        execution = next(node for node in data.nodes if _key(node.node_key) == key)
        if declared != _access_counts(execution.accesses):
            raise ValueError("execution node access parity")
        node_global = _access_counts(
            access for access in data.accesses if _key(access.node_key) == key
        )
        if declared != node_global:
            raise ValueError("global node access parity")
    return order

"""Pure Python WorkGraph descriptors and conservative declaration lowering.

This module deliberately models *metadata*.  A descriptor may mention a
logical port, device, or lease, but it must never contain a Tensor, stream,
event, pointer, or backend-private runtime object.  The physical executor is
owned by the runtime and is outside this first IR slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar, cast

from tllm.ports.address import PortAddressDesc
from tllm.ports.port_id import PortId


PortRef = PortAddressDesc | PortId


class NodeKind(StrEnum):
    ANCHOR = "anchor"
    COMPUTE = "compute"
    PROVIDER = "provider"
    TRANSFER = "transfer"
    COLLECTIVE = "collective"
    BARRIER = "barrier"
    STATE = "state"


class AccessMode(StrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class BoundaryKind(StrEnum):
    LEGACY = "legacy_boundary"


class TimingMobilityKind(StrEnum):
    PINNED = "pinned"
    DEFERRABLE = "deferrable"
    ADVANCEABLE = "advanceable"
    FREE = "free"


class TimingWindowKind(StrEnum):
    BACKGROUND = "background"
    SAME_STEP = "same_step"
    NEXT_STEP = "next_step"
    OUT_OF_BAND = "out_of_band"


class TimingDeadlineKind(StrEnum):
    STEP_END = "step_end"
    NEXT_PREPARE = "next_prepare"
    GENERATE_END = "generate_end"


class VersionObservationKind(StrEnum):
    AT_DECLARED_WINDOW = "at_declared_window"
    LATEST_AT_EXECUTION = "latest_at_execution"
    STALE_OK = "stale_ok"


class WorkGraphDiagnosticCode(StrEnum):
    """Stable, machine-readable Phase 1a diagnostic classifications."""

    UNSUPPORTED_GRAPH_MODE = "WG120_UNSUPPORTED_GRAPH_MODE"


@dataclass(frozen=True, slots=True)
class WorkGraphDiagnostic:
    """Pure-data diagnostic retained with a compatibility snapshot."""

    code: WorkGraphDiagnosticCode
    message: str
    detail: str = ""

    def __post_init__(self) -> None:
        code = _require_enum(self.code, WorkGraphDiagnosticCode, "WorkGraphDiagnostic.code")
        object.__setattr__(self, "code", code)
        object.__setattr__(
            self,
            "message",
            _require_str(self.message, "WorkGraphDiagnostic.message", nonempty=True),
        )
        object.__setattr__(self, "detail", _require_str(self.detail, "WorkGraphDiagnostic.detail"))


class GraphSafetyKind(StrEnum):
    EAGER_ONLY = "eager_only"
    CAPTURE_SAFE = "capture_safe"
    REPLAY_STATEFUL = "replay_stateful"


class EdgeKind(StrEnum):
    DATA = "data"
    CONTROL = "control"
    EFFECT = "effect"
    LEASE = "lease"
    COLLECTIVE = "collective"
    ANCHOR = "anchor"


_EnumT = TypeVar("_EnumT", bound=StrEnum)
_InstanceT = TypeVar("_InstanceT")


def _require_str(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if nonempty and not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be bool")
    return value


def _require_instance(value: object, expected_type: type[_InstanceT], label: str) -> _InstanceT:
    if not isinstance(value, expected_type):
        raise TypeError(f"{label} must be {expected_type.__name__}")
    return value


def _require_port_ref(value: object, label: str, *, allow_none: bool = False) -> PortRef | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, (PortAddressDesc, PortId)):
        suffix = " or None" if allow_none else ""
        raise TypeError(f"{label} must be PortAddressDesc, PortId{suffix}")
    return value


def _require_tuple(values: object, label: str, item_type: type[Any]) -> tuple[Any, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be tuple")
    typed_values = cast(tuple[object, ...], values)
    if any(not isinstance(value, item_type) for value in typed_values):
        raise TypeError(f"{label} must contain only {item_type.__name__}")
    return cast(tuple[Any, ...], typed_values)


def _require_tuple_str(values: object, label: str) -> tuple[str, ...]:
    return tuple(_require_tuple(values, label, str))


def _require_tuple_pairs_str(values: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be tuple")
    pairs: list[tuple[str, str]] = []
    for value in cast(tuple[object, ...], values):
        if not isinstance(value, tuple):
            raise TypeError(f"{label} must contain two-string tuples")
        pair = cast(tuple[object, ...], value)
        if len(pair) != 2:
            raise TypeError(f"{label} must contain two-string tuples")
        first, second = pair
        pairs.append(
            (
                _require_str(first, f"{label} key", nonempty=True),
                _require_str(second, f"{label} value", nonempty=True),
            )
        )
    return tuple(pairs)


def _require_enum(value: object, enum_type: type[_EnumT], label: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{label} must be {enum_type.__name__} or str")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {label} {value!r}") from exc


_SUPPORTED_RUNNER_FAMILIES = frozenset({"mrv1", "mrv2", "unresolved"})
_SUPPORTED_GRAPH_MODES = frozenset({"eager", "piecewise", "full"})
_BACKEND_VERSION_PREFIX = "0.27."
_SUPPORTED_ANCHOR_HOOKS = frozenset(
    {
        "prepare_inputs.pre",
        "prepare_inputs.post",
        "execute_model.pre",
        "execute_model.post",
        "sample_tokens.pre",
        "sampler.pre_draw",
        "sample_tokens.post",
    }
)


def _stable_segment(value: str, label: str) -> str:
    result = _require_str(value, label, nonempty=True)
    if "/" in result or "\\" in result:
        raise ValueError(f"{label} must not contain '/' or '\\\\'")
    return result


def _validate_edges(
    nodes: tuple["NodeKey", ...],
    edges: object,
) -> tuple[tuple["NodeKey", "NodeKey", "EdgeKind"], ...]:
    if not isinstance(edges, tuple):
        raise TypeError("graph edges must be tuple")
    known = set(nodes)
    normalized: list[tuple[NodeKey, NodeKey, EdgeKind]] = []
    seen: set[tuple[NodeKey, NodeKey, EdgeKind]] = set()
    for edge_spec in cast(tuple[object, ...], edges):
        if not isinstance(edge_spec, tuple):
            raise TypeError("graph edges must contain (NodeKey, NodeKey, EdgeKind) tuples")
        edge_parts = cast(tuple[object, ...], edge_spec)
        if len(edge_parts) != 3:
            raise TypeError("graph edges must contain (NodeKey, NodeKey, EdgeKind) tuples")
        source = _require_instance(edge_parts[0], NodeKey, "graph edge source")
        target = _require_instance(edge_parts[1], NodeKey, "graph edge target")
        kind = edge_parts[2]
        if source not in known or target not in known:
            raise ValueError(f"graph edge references unknown node: {source} -> {target}")
        edge = (source, target, _require_enum(kind, EdgeKind, "graph edge kind"))
        if edge in seen:
            raise ValueError(f"duplicate graph edge: {source} -> {target} ({edge[2]})")
        seen.add(edge)
        normalized.append(edge)

    adjacency: dict[NodeKey, tuple[NodeKey, ...]] = {node: () for node in nodes}
    successors: dict[NodeKey, list[NodeKey]] = {node: [] for node in nodes}
    for source, target, _kind in normalized:
        successors[source].append(target)
    adjacency = {node: tuple(successors[node]) for node in nodes}
    state: dict[NodeKey, int] = {node: 0 for node in nodes}

    def visit(node: NodeKey) -> None:
        state[node] = 1
        for successor in adjacency[node]:
            if state[successor] == 1:
                raise ValueError(f"graph contains a cycle through {successor}")
            if state[successor] == 0:
                visit(successor)
        state[node] = 2

    for node in nodes:
        if state[node] == 0:
            visit(node)
    return tuple(normalized)


def _validate_anchor_edges(
    nodes: tuple["SemanticNodeDesc", ...] | tuple["ExecutionNodeDesc", ...],
    edges: tuple[tuple["NodeKey", "NodeKey", "EdgeKind"], ...],
    label: str,
) -> None:
    node_by_key = {node.node_key: node for node in nodes}
    anchor_keys = {
        node.node_key
        for node in nodes
        if node.kind == NodeKind.ANCHOR
    }
    attached: set[NodeKey] = set()
    for source, target, kind in edges:
        if kind != EdgeKind.ANCHOR:
            continue
        if node_by_key[source].kind != NodeKind.ANCHOR:
            raise ValueError(f"{label} Anchor edge source must be an AnchorNode")
        if node_by_key[target].kind == NodeKind.ANCHOR:
            raise ValueError(f"{label} Anchor edge target must not be an AnchorNode")
        attached.add(source)
    if attached != anchor_keys:
        raise ValueError(f"{label} AnchorNodes must each attach through an Anchor edge")


@dataclass(frozen=True, slots=True, order=True)
class NodeKey:
    """Stable logical node identity.

    ``str(NodeKey(...))`` is the persisted form.  The three components are
    intentionally independent of object addresses, request ids, Tensor reprs
    and source line numbers.
    """

    namespace: str
    owner: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        namespace = _require_str(self.namespace, "NodeKey.namespace", nonempty=True)
        owner = _require_str(self.owner, "NodeKey.owner", nonempty=True)
        if "/" in namespace:
            raise ValueError("NodeKey.namespace must not contain '/'")
        if namespace == "anchor":
            segments = owner.split("/")
            if len(segments) != 4 or any(not segment or "\\" in segment for segment in segments):
                raise ValueError(
                    "anchor NodeKey.owner must contain exactly four non-empty '/'-separated segments"
                )
        else:
            owner = _stable_segment(owner, "NodeKey.owner")
        ordinal = _require_int(self.ordinal, "NodeKey.ordinal")
        if ordinal < 0:
            raise ValueError("NodeKey.ordinal must be a non-negative int")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "ordinal", ordinal)

    @property
    def value(self) -> str:
        if self.namespace == "anchor":
            return f"{self.namespace}/{self.owner}"
        return f"{self.namespace}/{self.owner}/{self.ordinal}"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def consumer_flow(cls, consumer_id: str, ordinal: int) -> "NodeKey":
        return cls("consumer_flow", consumer_id, ordinal)

    @classmethod
    def producer(cls, producer_id: str) -> "NodeKey":
        return cls("producer", producer_id, 0)

    @classmethod
    def anchor(cls, backend: str, runner: str, graph_mode: str, hook: str) -> "NodeKey":
        return cls("anchor", f"{backend}/{runner}/{graph_mode}/{hook}", 0)


@dataclass(frozen=True, slots=True, order=True)
class ExecutorKey:
    """Stable identity for an implementation, without storing its callback."""

    namespace: str
    name: str
    version: str = "1"

    def __post_init__(self) -> None:
        namespace = _require_str(self.namespace, "ExecutorKey.namespace", nonempty=True)
        name = _require_str(self.name, "ExecutorKey.name", nonempty=True)
        version = _require_str(self.version, "ExecutorKey.version", nonempty=True)
        if "/" in namespace or "/" in name:
            raise ValueError("ExecutorKey namespace/name must not contain '/'")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class BackendSnapshot:
    """CPU-visible backend capability snapshot used by a plan revision."""

    backend_id: str
    runner_family: str = ""
    graph_mode: str = "eager"
    backend_version: str = ""
    capabilities: tuple[str, ...] = ()
    device: str = "cuda"
    rank: int = 0
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    topology: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_id", _stable_segment(self.backend_id, "backend_id"))
        runner_family = _require_str(self.runner_family, "BackendSnapshot.runner_family")
        graph_mode = _require_str(self.graph_mode, "BackendSnapshot.graph_mode", nonempty=True)
        backend_version = _require_str(
            self.backend_version,
            "BackendSnapshot.backend_version",
            nonempty=True,
        )
        capabilities = _require_tuple_str(self.capabilities, "BackendSnapshot.capabilities")
        device = _require_str(self.device, "BackendSnapshot.device", nonempty=True)
        topology = _require_tuple_str(self.topology, "BackendSnapshot.topology")
        rank = _require_int(self.rank, "BackendSnapshot.rank")
        tp_size = _require_int(self.tp_size, "BackendSnapshot.tp_size")
        pp_size = _require_int(self.pp_size, "BackendSnapshot.pp_size")
        dp_size = _require_int(self.dp_size, "BackendSnapshot.dp_size")
        if rank < 0:
            raise ValueError("BackendSnapshot.rank must be >= 0")
        if tp_size < 1:
            raise ValueError("BackendSnapshot.tp_size must be >= 1")
        if pp_size < 1:
            raise ValueError("BackendSnapshot.pp_size must be >= 1")
        if dp_size < 1:
            raise ValueError("BackendSnapshot.dp_size must be >= 1")
        if runner_family not in _SUPPORTED_RUNNER_FAMILIES:
            raise ValueError(
                "BackendSnapshot.runner_family must be one of 'mrv1', 'mrv2', or 'unresolved'"
            )
        if graph_mode not in _SUPPORTED_GRAPH_MODES:
            raise ValueError(
                "BackendSnapshot.graph_mode must be one of 'eager', 'piecewise', or 'full'"
            )
        if not backend_version.startswith(_BACKEND_VERSION_PREFIX):
            raise ValueError(
                "BackendSnapshot.backend_version must belong to the supported vLLM 0.27.x family"
            )
        object.__setattr__(self, "runner_family", runner_family)
        object.__setattr__(self, "graph_mode", graph_mode)
        object.__setattr__(self, "backend_version", backend_version)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "pp_size", pp_size)
        object.__setattr__(self, "dp_size", dp_size)

@dataclass(frozen=True, slots=True)
class AnchorDesc:
    """A fixed, externally-owned backend lifecycle point."""

    backend: str
    runner: str
    graph_mode: str
    hook: str
    externally_owned: bool = True
    mobility: str = "pinned"
    service_class: str = "critical"

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _stable_segment(self.backend, "AnchorDesc.backend"))
        object.__setattr__(self, "runner", _stable_segment(self.runner, "AnchorDesc.runner"))
        object.__setattr__(self, "graph_mode", _stable_segment(self.graph_mode, "AnchorDesc.graph_mode"))
        object.__setattr__(self, "hook", _stable_segment(self.hook, "AnchorDesc.hook"))
        if not _require_bool(self.externally_owned, "AnchorDesc.externally_owned"):
            raise ValueError("WorkGraph anchors must be externally_owned")
        mobility = _require_str(self.mobility, "AnchorDesc.mobility", nonempty=True)
        if mobility != "pinned":
            raise ValueError("WorkGraph anchors must be pinned")
        service_class = _require_str(self.service_class, "AnchorDesc.service_class", nonempty=True)
        if self.hook not in _SUPPORTED_ANCHOR_HOOKS:
            raise ValueError(f"unsupported WorkGraph anchor hook {self.hook!r}")
        object.__setattr__(self, "externally_owned", True)
        object.__setattr__(self, "mobility", mobility)
        object.__setattr__(self, "service_class", service_class)

    @property
    def node_key(self) -> NodeKey:
        return NodeKey.anchor(self.backend, self.runner, self.graph_mode, self.hook)


@dataclass(frozen=True, slots=True)
class TimingDesc:
    """Serializable timing contract, detached from ConsumerFlow objects."""

    window: TimingWindowKind = TimingWindowKind.SAME_STEP
    mobility: TimingMobilityKind = TimingMobilityKind.PINNED
    deadline: TimingDeadlineKind = TimingDeadlineKind.STEP_END
    max_defer_steps: int = 0
    observes_version: VersionObservationKind = VersionObservationKind.AT_DECLARED_WINDOW

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "window",
            _require_enum(self.window, TimingWindowKind, "TimingDesc.window"),
        )
        object.__setattr__(
            self,
            "mobility",
            _require_enum(self.mobility, TimingMobilityKind, "TimingDesc.mobility"),
        )
        object.__setattr__(
            self,
            "deadline",
            _require_enum(self.deadline, TimingDeadlineKind, "TimingDesc.deadline"),
        )
        object.__setattr__(
            self,
            "observes_version",
            _require_enum(
                self.observes_version,
                VersionObservationKind,
                "TimingDesc.observes_version",
            ),
        )
        max_defer_steps = _require_int(self.max_defer_steps, "TimingDesc.max_defer_steps")
        if max_defer_steps < 0:
            raise ValueError("TimingDesc.max_defer_steps must be non-negative")
        object.__setattr__(self, "max_defer_steps", max_defer_steps)


@dataclass(frozen=True, slots=True)
class ValueDesc:
    """Logical value/version identity; no payload is retained."""

    value_id: str
    version: int = 1
    port: PortRef | None = None
    shape: tuple[str, ...] = ()
    dtype: str = ""
    layout: str = ""
    alias_set: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value_id",
            _require_str(self.value_id, "ValueDesc.value_id", nonempty=True),
        )
        version = _require_int(self.version, "ValueDesc.version")
        if version < 1:
            raise ValueError("ValueDesc.version must be >= 1")
        port = _require_port_ref(self.port, "ValueDesc.port", allow_none=True)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "shape", _require_tuple_str(self.shape, "ValueDesc.shape"))
        object.__setattr__(self, "dtype", _require_str(self.dtype, "ValueDesc.dtype"))
        object.__setattr__(self, "layout", _require_str(self.layout, "ValueDesc.layout"))
        object.__setattr__(self, "alias_set", _require_str(self.alias_set, "ValueDesc.alias_set"))

    @property
    def key(self) -> str:
        return f"{self.value_id}@{self.version}"


@dataclass(frozen=True, slots=True)
class AccessDesc:
    node_key: NodeKey
    value: ValueDesc
    mode: AccessMode
    port: PortRef | None = None
    subset: tuple[str, ...] = ()
    consistency: str = "exact"

    def __post_init__(self) -> None:
        node_key = _require_instance(self.node_key, NodeKey, "AccessDesc.node_key")
        value = _require_instance(self.value, ValueDesc, "AccessDesc.value")
        object.__setattr__(self, "mode", _require_enum(self.mode, AccessMode, "AccessDesc.mode"))
        port = _require_port_ref(self.port, "AccessDesc.port", allow_none=True)
        if port is not None and value.port is not None and port != value.port:
            raise ValueError("AccessDesc.port must match ValueDesc.port")
        object.__setattr__(self, "node_key", node_key)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "subset", _require_tuple_str(self.subset, "AccessDesc.subset"))
        object.__setattr__(
            self,
            "consistency",
            _require_str(self.consistency, "AccessDesc.consistency", nonempty=True),
        )


@dataclass(frozen=True, slots=True)
class ResidenceDesc:
    value: ValueDesc
    device: str
    rank: int = 0
    memory_kind: str = "backend_owned"
    valid_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            _require_instance(self.value, ValueDesc, "ResidenceDesc.value"),
        )
        object.__setattr__(
            self,
            "device",
            _require_str(self.device, "ResidenceDesc.device", nonempty=True),
        )
        object.__setattr__(self, "rank", _require_int(self.rank, "ResidenceDesc.rank"))
        object.__setattr__(
            self,
            "memory_kind",
            _require_str(self.memory_kind, "ResidenceDesc.memory_kind", nonempty=True),
        )
        version = _require_int(self.valid_version, "ResidenceDesc.valid_version")
        if version < 1:
            raise ValueError("ResidenceDesc.valid_version must be >= 1")
        object.__setattr__(self, "valid_version", version)

    @property
    def key(self) -> str:
        """Phase 1a reference key for the value's same-worker residence."""

        return self.value.key


@dataclass(frozen=True, slots=True)
class MoveDesc:
    value: ValueDesc
    source: ResidenceDesc
    destination: ResidenceDesc
    subset: tuple[str, ...] = ()
    layout: str = ""
    engine: str = ""
    asynchronous: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _require_instance(self.value, ValueDesc, "MoveDesc.value"))
        object.__setattr__(
            self,
            "source",
            _require_instance(self.source, ResidenceDesc, "MoveDesc.source"),
        )
        object.__setattr__(
            self,
            "destination",
            _require_instance(self.destination, ResidenceDesc, "MoveDesc.destination"),
        )
        object.__setattr__(self, "subset", _require_tuple_str(self.subset, "MoveDesc.subset"))
        object.__setattr__(self, "layout", _require_str(self.layout, "MoveDesc.layout"))
        object.__setattr__(self, "engine", _require_str(self.engine, "MoveDesc.engine"))
        object.__setattr__(self, "asynchronous", _require_bool(self.asynchronous, "MoveDesc.asynchronous"))


@dataclass(frozen=True, slots=True)
class LeaseDesc:
    value: ValueDesc
    storage: str = "backend_owned"
    live_start: str = ""
    live_end: str = ""
    ready_fence: str = ""
    retire_fence: str = ""
    reader_count: int = 0
    stable_address: bool = False
    materialized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _require_instance(self.value, ValueDesc, "LeaseDesc.value"))
        object.__setattr__(self, "storage", _require_str(self.storage, "LeaseDesc.storage", nonempty=True))
        object.__setattr__(self, "live_start", _require_str(self.live_start, "LeaseDesc.live_start"))
        object.__setattr__(self, "live_end", _require_str(self.live_end, "LeaseDesc.live_end"))
        object.__setattr__(self, "ready_fence", _require_str(self.ready_fence, "LeaseDesc.ready_fence"))
        object.__setattr__(self, "retire_fence", _require_str(self.retire_fence, "LeaseDesc.retire_fence"))
        readers = _require_int(self.reader_count, "LeaseDesc.reader_count")
        if readers < 0:
            raise ValueError("LeaseDesc.reader_count must be non-negative")
        object.__setattr__(self, "reader_count", readers)
        object.__setattr__(self, "stable_address", _require_bool(self.stable_address, "LeaseDesc.stable_address"))
        materialized = _require_bool(self.materialized, "LeaseDesc.materialized")
        object.__setattr__(self, "materialized", materialized)
        if materialized:
            raise ValueError("Phase 1a leases cannot contain materialized runtime resources")

    @property
    def key(self) -> str:
        """Phase 1a reference key for a metadata-only value lease."""

        return self.value.key


@dataclass(frozen=True, slots=True)
class SemanticNodeDesc:
    node_key: NodeKey
    kind: NodeKind = NodeKind.COMPUTE
    executor: ExecutorKey | None = None
    reads: tuple[AccessDesc, ...] = ()
    writes: tuple[AccessDesc, ...] = ()
    effects: tuple[str, ...] = ()
    timing: TimingDesc = TimingDesc()
    anchor: AnchorDesc | None = None
    graph_safety: GraphSafetyKind = GraphSafetyKind.EAGER_ONLY
    source: str = ""
    legacy_boundary: BoundaryKind | None = None
    shape_guard: tuple[str, ...] = ()

    @property
    def key(self) -> NodeKey:
        return self.node_key

    def __post_init__(self) -> None:
        node_key = _require_instance(self.node_key, NodeKey, "SemanticNodeDesc.node_key")
        kind = _require_enum(self.kind, NodeKind, "SemanticNodeDesc.kind")
        executor = (
            None
            if self.executor is None
            else _require_instance(self.executor, ExecutorKey, "SemanticNodeDesc.executor")
        )
        reads = _require_tuple(self.reads, "SemanticNodeDesc.reads", AccessDesc)
        writes = _require_tuple(self.writes, "SemanticNodeDesc.writes", AccessDesc)
        timing = _require_instance(self.timing, TimingDesc, "SemanticNodeDesc.timing")
        anchor = (
            None
            if self.anchor is None
            else _require_instance(self.anchor, AnchorDesc, "SemanticNodeDesc.anchor")
        )
        graph_safety = _require_enum(
            self.graph_safety,
            GraphSafetyKind,
            "SemanticNodeDesc.graph_safety",
        )
        if kind == NodeKind.ANCHOR:
            if executor is not None:
                raise ValueError("Anchor SemanticNodeDesc must not have an executor")
            if anchor is None or node_key != anchor.node_key:
                raise ValueError("Anchor SemanticNodeDesc must use its AnchorDesc node_key")
        elif anchor is not None:
            raise ValueError("only Anchor SemanticNodeDesc may hold an AnchorDesc")
        object.__setattr__(self, "node_key", node_key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "executor", executor)
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "timing", timing)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "effects", _require_tuple_str(self.effects, "SemanticNodeDesc.effects"))
        object.__setattr__(self, "source", _require_str(self.source, "SemanticNodeDesc.source"))
        object.__setattr__(
            self,
            "shape_guard",
            _require_tuple_str(self.shape_guard, "SemanticNodeDesc.shape_guard"),
        )
        object.__setattr__(self, "graph_safety", graph_safety)
        if self.legacy_boundary is not None:
            object.__setattr__(
                self,
                "legacy_boundary",
                _require_enum(
                    self.legacy_boundary,
                    BoundaryKind,
                    "SemanticNodeDesc.legacy_boundary",
                ),
            )


@dataclass(frozen=True, slots=True)
class SemanticGraph:
    nodes: tuple[SemanticNodeDesc, ...]
    edges: tuple[tuple[NodeKey, NodeKey, EdgeKind], ...] = ()
    values: tuple[ValueDesc, ...] = ()
    backend: BackendSnapshot | None = None
    graph_mode: str = "eager"
    schema_version: int = 1

    def __post_init__(self) -> None:
        nodes = _require_tuple(self.nodes, "SemanticGraph.nodes", SemanticNodeDesc)
        schema_version = _require_int(self.schema_version, "SemanticGraph.schema_version")
        if schema_version != 1:
            raise ValueError(f"unsupported SemanticGraph.schema_version {self.schema_version!r}")
        keys = [node.node_key for node in nodes]
        if len(keys) != len(set(keys)):
            raise ValueError("SemanticGraph contains duplicate node keys")
        edges = _validate_edges(tuple(keys), self.edges)
        values = _require_tuple(self.values, "SemanticGraph.values", ValueDesc)
        value_keys = [value.key for value in values]
        if len(value_keys) != len(set(value_keys)):
            raise ValueError("SemanticGraph contains duplicate value keys")
        object.__setattr__(self, "nodes", nodes)
        backend = (
            None
            if self.backend is None
            else _require_instance(self.backend, BackendSnapshot, "SemanticGraph.backend")
        )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(
            self,
            "graph_mode",
            _require_str(self.graph_mode, "SemanticGraph.graph_mode", nonempty=True),
        )
        _validate_anchor_edges(nodes, edges, "SemanticGraph")
        value_key_set = set(value_keys)
        for node in nodes:
            for access in (*node.reads, *node.writes):
                if access.node_key != node.node_key:
                    raise ValueError(
                        f"SemanticGraph access node {access.node_key} does not match "
                        f"owner node {node.node_key}"
                    )
                if access.value.key not in value_key_set:
                    raise ValueError(
                        f"SemanticGraph access references unknown value {access.value.key!r}"
                    )

    def node(self, key: NodeKey) -> SemanticNodeDesc:
        for node in self.nodes:
            if node.node_key == key:
                return node
        raise KeyError(key)


@dataclass(frozen=True, slots=True)
class ExecutionNodeDesc:
    node_key: NodeKey
    semantic_key: NodeKey | None = None
    kind: NodeKind = NodeKind.COMPUTE
    executor: ExecutorKey | None = None
    accesses: tuple[AccessDesc, ...] = ()
    residence_keys: tuple[str, ...] = ()
    move_keys: tuple[str, ...] = ()
    lease_keys: tuple[str, ...] = ()
    device: str = "cuda"
    rank: int = 0
    anchor: AnchorDesc | None = None
    graph_safety: GraphSafetyKind = GraphSafetyKind.EAGER_ONLY
    fusion_group: str = ""

    @property
    def key(self) -> NodeKey:
        return self.node_key

    def __post_init__(self) -> None:
        node_key = _require_instance(self.node_key, NodeKey, "ExecutionNodeDesc.node_key")
        semantic_key = node_key if self.semantic_key is None else _require_instance(
            self.semantic_key,
            NodeKey,
            "ExecutionNodeDesc.semantic_key",
        )
        if semantic_key != node_key:
            raise ValueError("Phase 1a ExecutionNodeDesc.semantic_key must equal node_key")
        kind = _require_enum(self.kind, NodeKind, "ExecutionNodeDesc.kind")
        executor = (
            None
            if self.executor is None
            else _require_instance(self.executor, ExecutorKey, "ExecutionNodeDesc.executor")
        )
        anchor = (
            None
            if self.anchor is None
            else _require_instance(self.anchor, AnchorDesc, "ExecutionNodeDesc.anchor")
        )
        graph_safety = _require_enum(
            self.graph_safety,
            GraphSafetyKind,
            "ExecutionNodeDesc.graph_safety",
        )
        if kind == NodeKind.ANCHOR:
            if executor is not None:
                raise ValueError("Anchor ExecutionNodeDesc must not have an executor")
            if anchor is None or node_key != anchor.node_key:
                raise ValueError("Anchor ExecutionNodeDesc must use its AnchorDesc node_key")
        elif anchor is not None:
            raise ValueError("only Anchor ExecutionNodeDesc may hold an AnchorDesc")
        object.__setattr__(self, "node_key", node_key)
        object.__setattr__(self, "semantic_key", semantic_key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "executor", executor)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "graph_safety", graph_safety)
        object.__setattr__(self, "accesses", _require_tuple(self.accesses, "ExecutionNodeDesc.accesses", AccessDesc))
        object.__setattr__(
            self,
            "residence_keys",
            _require_tuple_str(self.residence_keys, "ExecutionNodeDesc.residence_keys"),
        )
        object.__setattr__(self, "move_keys", _require_tuple_str(self.move_keys, "ExecutionNodeDesc.move_keys"))
        object.__setattr__(self, "lease_keys", _require_tuple_str(self.lease_keys, "ExecutionNodeDesc.lease_keys"))
        object.__setattr__(
            self,
            "device",
            _require_str(self.device, "ExecutionNodeDesc.device", nonempty=True),
        )
        object.__setattr__(self, "rank", _require_int(self.rank, "ExecutionNodeDesc.rank"))
        object.__setattr__(self, "fusion_group", _require_str(self.fusion_group, "ExecutionNodeDesc.fusion_group"))


@dataclass(frozen=True, slots=True)
class DataCentricIR:
    nodes: tuple[ExecutionNodeDesc, ...]
    values: tuple[ValueDesc, ...] = ()
    accesses: tuple[AccessDesc, ...] = ()
    residences: tuple[ResidenceDesc, ...] = ()
    moves: tuple[MoveDesc, ...] = ()
    leases: tuple[LeaseDesc, ...] = ()
    edges: tuple[tuple[NodeKey, NodeKey, EdgeKind], ...] = ()
    backend: BackendSnapshot | None = None
    graph_mode: str = "eager"
    schema_version: int = 1

    def __post_init__(self) -> None:
        nodes = _require_tuple(self.nodes, "DataCentricIR.nodes", ExecutionNodeDesc)
        schema_version = _require_int(self.schema_version, "DataCentricIR.schema_version")
        if schema_version != 1:
            raise ValueError(f"unsupported DataCentricIR.schema_version {self.schema_version!r}")
        keys = [node.node_key for node in nodes]
        if len(keys) != len(set(keys)):
            raise ValueError("DataCentricIR contains duplicate node keys")
        moves = _require_tuple(self.moves, "DataCentricIR.moves", MoveDesc)
        if moves:
            raise ValueError("Phase 1a DataCentricIR must not contain MoveDesc")
        edges = _validate_edges(tuple(keys), self.edges)
        object.__setattr__(self, "nodes", nodes)
        values = _require_tuple(self.values, "DataCentricIR.values", ValueDesc)
        accesses = _require_tuple(self.accesses, "DataCentricIR.accesses", AccessDesc)
        residences = _require_tuple(self.residences, "DataCentricIR.residences", ResidenceDesc)
        leases = _require_tuple(self.leases, "DataCentricIR.leases", LeaseDesc)
        backend = (
            None
            if self.backend is None
            else _require_instance(self.backend, BackendSnapshot, "DataCentricIR.backend")
        )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "accesses", accesses)
        object.__setattr__(self, "residences", residences)
        object.__setattr__(self, "moves", moves)
        object.__setattr__(self, "leases", leases)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(
            self,
            "graph_mode",
            _require_str(self.graph_mode, "DataCentricIR.graph_mode", nonempty=True),
        )
        _validate_anchor_edges(nodes, edges, "DataCentricIR")
        value_key_list = [value.key for value in self.values]
        residence_key_list = [residence.key for residence in self.residences]
        lease_key_list = [lease.key for lease in self.leases]
        value_keys = set(value_key_list)
        residence_keys = set(residence_key_list)
        lease_keys = set(lease_key_list)
        if len(value_key_list) != len(value_keys):
            raise ValueError("DataCentricIR contains duplicate value keys")
        if len(residence_key_list) != len(residence_keys):
            raise ValueError("DataCentricIR contains duplicate residence keys")
        if len(lease_key_list) != len(lease_keys):
            raise ValueError("DataCentricIR contains duplicate lease keys")
        for residence in self.residences:
            if residence.value.key not in value_keys:
                raise ValueError(
                    f"DataCentricIR residence references unknown value {residence.value.key!r}"
                )
        for lease in self.leases:
            if lease.value.key not in value_keys:
                raise ValueError(
                    f"DataCentricIR lease references unknown value {lease.value.key!r}"
                )
        for node in nodes:
            if node.move_keys:
                raise ValueError(
                    f"DataCentricIR node {node.node_key} references unsupported Move descriptors"
                )
            for access in node.accesses:
                if access.node_key != node.node_key:
                    raise ValueError(
                        f"DataCentricIR access node {access.node_key} does not match "
                        f"owner node {node.node_key}"
                    )
                if access.value.key not in value_keys:
                    raise ValueError(
                        f"DataCentricIR access references unknown value {access.value.key!r}"
                    )
            unknown_residences = set(node.residence_keys) - residence_keys
            if unknown_residences:
                raise ValueError(
                    f"DataCentricIR node {node.node_key} references unknown residences "
                    f"{sorted(unknown_residences)!r}"
                )
            unknown_leases = set(node.lease_keys) - lease_keys
            if unknown_leases:
                raise ValueError(
                    f"DataCentricIR node {node.node_key} references unknown leases "
                    f"{sorted(unknown_leases)!r}"
                )
        for access in self.accesses:
            if access.node_key not in keys:
                raise ValueError(
                    f"DataCentricIR access references unknown node {access.node_key}"
                )
            if access.value.key not in value_keys:
                raise ValueError(
                    f"DataCentricIR access references unknown value {access.value.key!r}"
                )
        if self.backend is not None:
            for node in nodes:
                if node.device != self.backend.device or node.rank != self.backend.rank:
                    raise ValueError(
                        f"DataCentricIR node {node.node_key} placement differs from backend snapshot"
                    )
            for residence in self.residences:
                if residence.device != self.backend.device or residence.rank != self.backend.rank:
                    raise ValueError(
                        f"DataCentricIR residence {residence.value.key!r} placement differs from backend snapshot"
                    )


@dataclass(frozen=True, slots=True)
class RuntimePlanBinding:
    """Typed runtime-plan facts that participate in the shadow plan key."""

    dispatch_flow_ids: tuple[str, ...]
    dispatch_hooks: tuple[tuple[str, str], ...]
    execution_producer_ids: tuple[str, ...]
    producer_hooks: tuple[tuple[str, tuple[str, ...]], ...]
    dispatch_fusion_groups: tuple[tuple[str, ...], ...] = ()
    execution_port_sources: tuple[tuple[str, str], ...] = ()
    capture_resident_producer_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dispatch_flow_ids",
            _require_tuple_str(self.dispatch_flow_ids, "RuntimePlanBinding.dispatch_flow_ids"),
        )
        object.__setattr__(
            self,
            "dispatch_hooks",
            _require_tuple_pairs_str(self.dispatch_hooks, "RuntimePlanBinding.dispatch_hooks"),
        )
        object.__setattr__(
            self,
            "execution_producer_ids",
            _require_tuple_str(
                self.execution_producer_ids,
                "RuntimePlanBinding.execution_producer_ids",
            ),
        )
        producer_hook_specs = _require_tuple(
            self.producer_hooks,
            "RuntimePlanBinding.producer_hooks",
            tuple,
        )
        producer_hooks: list[tuple[str, tuple[str, ...]]] = []
        for item in producer_hook_specs:
            item_parts = cast(tuple[object, ...], item)
            if len(item_parts) != 2:
                raise TypeError(
                    "RuntimePlanBinding.producer_hooks must contain (str, tuple[str, ...])"
                )
            producer_id, hooks = item_parts
            producer_hooks.append(
                (
                    _require_str(
                        producer_id,
                        "RuntimePlanBinding.producer_hooks producer_id",
                        nonempty=True,
                    ),
                    _require_tuple_str(
                        hooks,
                        "RuntimePlanBinding.producer_hooks hooks",
                    ),
                )
            )
        object.__setattr__(self, "producer_hooks", tuple(producer_hooks))
        fusion_groups = _require_tuple(
            self.dispatch_fusion_groups,
            "RuntimePlanBinding.dispatch_fusion_groups",
            tuple,
        )
        object.__setattr__(
            self,
            "dispatch_fusion_groups",
            tuple(
                _require_tuple_str(group, "RuntimePlanBinding.dispatch_fusion_groups group")
                for group in fusion_groups
            ),
        )
        object.__setattr__(
            self,
            "execution_port_sources",
            _require_tuple_pairs_str(
                self.execution_port_sources,
                "RuntimePlanBinding.execution_port_sources",
            ),
        )
        object.__setattr__(
            self,
            "capture_resident_producer_ids",
            _require_tuple_str(
                self.capture_resident_producer_ids,
                "RuntimePlanBinding.capture_resident_producer_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class CompiledWorkGraph:
    semantic: SemanticGraph
    data_centric: DataCentricIR
    structural_signature: str = ""
    epoch: int = 0
    diagnostics: tuple[str, ...] = ()

    @property
    def ir(self) -> DataCentricIR:
        return self.data_centric

    @property
    def signature(self) -> str:
        return self.structural_signature

    def __post_init__(self) -> None:
        semantic = _require_instance(self.semantic, SemanticGraph, "CompiledWorkGraph.semantic")
        data_centric = _require_instance(
            self.data_centric,
            DataCentricIR,
            "CompiledWorkGraph.data_centric",
        )
        if semantic.backend != data_centric.backend:
            raise ValueError("CompiledWorkGraph semantic/data backend snapshots must match")
        if semantic.graph_mode != data_centric.graph_mode:
            raise ValueError("CompiledWorkGraph semantic/data graph modes must match")
        semantic_nodes = {node.node_key: node for node in semantic.nodes}
        data_nodes = {node.node_key: node for node in data_centric.nodes}
        if set(semantic_nodes) != set(data_nodes):
            raise ValueError("CompiledWorkGraph semantic/data node keys must match")
        for node_key, semantic_node in semantic_nodes.items():
            data_node = data_nodes[node_key]
            if (
                semantic_node.kind != data_node.kind
                or semantic_node.executor != data_node.executor
                or semantic_node.graph_safety != data_node.graph_safety
                or semantic_node.anchor != data_node.anchor
            ):
                raise ValueError(
                    f"CompiledWorkGraph semantic/data node projection differs for {node_key}"
                )
        if frozenset(semantic.values) != frozenset(data_centric.values):
            raise ValueError("CompiledWorkGraph semantic/data value descriptors must match")
        if frozenset(semantic.edges) != frozenset(data_centric.edges):
            raise ValueError("CompiledWorkGraph semantic/data edges must match")
        object.__setattr__(self, "semantic", semantic)
        object.__setattr__(self, "data_centric", data_centric)
        _require_str(
            self.structural_signature,
            "CompiledWorkGraph.structural_signature",
        )
        epoch = _require_int(self.epoch, "CompiledWorkGraph.epoch")
        if epoch < 0:
            raise ValueError("CompiledWorkGraph.epoch must be non-negative")
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(
            self,
            "diagnostics",
            _require_tuple_str(self.diagnostics, "CompiledWorkGraph.diagnostics"),
        )

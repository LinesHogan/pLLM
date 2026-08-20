"""Frozen distributed topology, state, and collective planning contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar, cast

from tllm.workgraph.frontend import StateScope
from tllm.workgraph.phase3 import Placement, ValueKey


class RankRole(StrEnum):
    MODEL = "model"
    HELPER = "helper"


class ParallelGroupKind(StrEnum):
    TENSOR = "tensor_parallel"
    PIPELINE = "pipeline_parallel"
    DATA = "data_parallel"
    HELPER = "helper"
    GLOBAL = "global"


class CollectiveKind(StrEnum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    SEND_RECV = "send_recv"
    BARRIER = "barrier"


class CollectiveBackend(StrEnum):
    NCCL = "nccl"
    GLOO = "gloo"


class CollectiveWindow(StrEnum):
    PREPARE = "prepare"
    BEFORE_COMPUTE = "before_compute"
    AFTER_COMPUTE = "after_compute"
    STEP_END = "step_end"


class ShardKind(StrEnum):
    REPLICATED = "replicated"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    DATA = "data"


_COLLECTIVE_WINDOW_ORDER = {
    CollectiveWindow.PREPARE: 0,
    CollectiveWindow.BEFORE_COMPUTE: 1,
    CollectiveWindow.AFTER_COMPUTE: 2,
    CollectiveWindow.STEP_END: 3,
}


class DistributedCompilationError(RuntimeError):
    """Raised when Rust rejects a Python-compiled distributed plan."""


_T = TypeVar("_T")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _instance(value: object, expected: type[_T], label: str) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


def _enum(value: object, expected: type[_EnumT], label: str) -> _EnumT:
    if isinstance(value, expected):
        return value
    if isinstance(value, str):
        try:
            return expected(value)
        except ValueError as exc:
            raise ValueError(f"{label} is unsupported") from exc
    raise TypeError(f"{label} must be {expected.__name__}")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if (
        not value
        or "0x" in value.lower()
        or "<" in value
        or ">" in value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{label} must be a stable non-empty segment")
    return value


def _tuple_of(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return cast(tuple[_T, ...], items)


@dataclass(frozen=True, slots=True, order=True)
class StateKey:
    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _text(self.namespace, "StateKey.namespace"))
        object.__setattr__(self, "name", _text(self.name, "StateKey.name"))
        object.__setattr__(
            self,
            "version",
            _integer(self.version, "StateKey.version", minimum=1),
        )

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class ShardSpec:
    kind: ShardKind
    partitions: int = 1
    axis: int | None = None

    def __post_init__(self) -> None:
        kind = _enum(cast(object, self.kind), ShardKind, "ShardSpec.kind")
        partitions = _integer(
            self.partitions,
            "ShardSpec.partitions",
            minimum=1,
        )
        axis = self.axis
        if kind == ShardKind.REPLICATED:
            if partitions != 1 or axis is not None:
                raise ValueError(
                    "replicated ShardSpec requires one partition and no axis"
                )
        elif kind in (ShardKind.TENSOR, ShardKind.DATA):
            if axis is None:
                raise ValueError("tensor/data ShardSpec requires an axis")
            axis = _integer(axis, "ShardSpec.axis")
        elif axis is not None:
            raise ValueError("pipeline ShardSpec does not use a tensor axis")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "axis", axis)


@dataclass(frozen=True, slots=True)
class PlacementAffinity:
    ranks: tuple[int, ...] = ()
    node_ids: tuple[str, ...] = ()
    role: RankRole | None = None

    def __post_init__(self) -> None:
        raw_ranks = cast(object, self.ranks)
        if not isinstance(raw_ranks, tuple):
            raise TypeError("PlacementAffinity.ranks must be tuple")
        ranks = tuple(
            _integer(rank, "PlacementAffinity.rank")
            for rank in cast(tuple[object, ...], raw_ranks)
        )
        if len(set(ranks)) != len(ranks):
            raise ValueError("PlacementAffinity.ranks must be unique")
        raw_node_ids = cast(object, self.node_ids)
        if not isinstance(raw_node_ids, tuple):
            raise TypeError("PlacementAffinity.node_ids must be tuple")
        node_ids = tuple(
            _text(node_id, "PlacementAffinity.node_id")
            for node_id in cast(tuple[object, ...], raw_node_ids)
        )
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("PlacementAffinity.node_ids must be unique")
        role = self.role
        if role is not None:
            role = _enum(cast(object, role), RankRole, "PlacementAffinity.role")
        object.__setattr__(self, "ranks", tuple(sorted(ranks)))
        object.__setattr__(self, "node_ids", tuple(sorted(node_ids)))
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class StateRequirementDesc:
    key: StateKey
    scope: StateScope
    shard: ShardSpec
    affinity: PlacementAffinity = PlacementAffinity()

    def __post_init__(self) -> None:
        key = _instance(self.key, StateKey, "StateRequirementDesc.key")
        scope = _enum(cast(object, self.scope), StateScope, "StateRequirementDesc.scope")
        if scope == StateScope.NONE:
            raise ValueError("distributed state cannot use StateScope.NONE")
        shard = _instance(self.shard, ShardSpec, "StateRequirementDesc.shard")
        affinity = _instance(
            self.affinity,
            PlacementAffinity,
            "StateRequirementDesc.affinity",
        )
        if scope == StateScope.SHARD and shard.kind == ShardKind.REPLICATED:
            raise ValueError("StateScope.SHARD requires a sharded ShardSpec")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "shard", shard)
        object.__setattr__(self, "affinity", affinity)


@dataclass(frozen=True, slots=True)
class TopologyRankDesc:
    rank: int
    node_id: str
    local_rank: int
    placement: Placement
    role: RankRole = RankRole.MODEL
    tp_rank: int | None = 0
    pp_rank: int | None = 0
    dp_rank: int | None = 0

    def __post_init__(self) -> None:
        rank = _integer(self.rank, "TopologyRankDesc.rank")
        node_id = _text(self.node_id, "TopologyRankDesc.node_id")
        local_rank = _integer(self.local_rank, "TopologyRankDesc.local_rank")
        placement = _instance(
            self.placement,
            Placement,
            "TopologyRankDesc.placement",
        )
        role = _enum(cast(object, self.role), RankRole, "TopologyRankDesc.role")
        coordinates = (self.tp_rank, self.pp_rank, self.dp_rank)
        if role == RankRole.MODEL:
            if any(item is None for item in coordinates):
                raise ValueError("model ranks require TP/PP/DP coordinates")
            tp_rank, pp_rank, dp_rank = (
                _integer(cast(int, item), "TopologyRankDesc coordinate")
                for item in coordinates
            )
        else:
            if any(item is not None for item in coordinates):
                raise ValueError("helper ranks must not declare TP/PP/DP coordinates")
            tp_rank = pp_rank = dp_rank = None
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "local_rank", local_rank)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "tp_rank", tp_rank)
        object.__setattr__(self, "pp_rank", pp_rank)
        object.__setattr__(self, "dp_rank", dp_rank)


@dataclass(frozen=True, slots=True)
class DistributedTopology:
    ranks: tuple[TopologyRankDesc, ...]
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1

    def __post_init__(self) -> None:
        ranks = _tuple_of(self.ranks, TopologyRankDesc, "DistributedTopology.ranks")
        tp_size = _integer(self.tp_size, "DistributedTopology.tp_size", minimum=1)
        pp_size = _integer(self.pp_size, "DistributedTopology.pp_size", minimum=1)
        dp_size = _integer(self.dp_size, "DistributedTopology.dp_size", minimum=1)
        if not ranks:
            raise ValueError("DistributedTopology requires at least one rank")
        rank_ids = tuple(item.rank for item in ranks)
        if len(set(rank_ids)) != len(rank_ids) or set(rank_ids) != set(range(len(ranks))):
            raise ValueError("DistributedTopology ranks must be unique and contiguous")
        local_devices = tuple((item.node_id, item.local_rank) for item in ranks)
        if len(set(local_devices)) != len(local_devices):
            raise ValueError("DistributedTopology local rank identities must be unique")
        model = tuple(item for item in ranks if item.role == RankRole.MODEL)
        if len(model) != tp_size * pp_size * dp_size:
            raise ValueError("model rank count must equal tp_size * pp_size * dp_size")
        expected_coordinates = {
            (tp, pp, dp)
            for dp in range(dp_size)
            for pp in range(pp_size)
            for tp in range(tp_size)
        }
        actual_coordinates = {
            (item.tp_rank, item.pp_rank, item.dp_rank) for item in model
        }
        if actual_coordinates != expected_coordinates:
            raise ValueError("model ranks must exactly cover the TP/PP/DP mesh")
        object.__setattr__(self, "ranks", tuple(sorted(ranks, key=lambda item: item.rank)))
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "pp_size", pp_size)
        object.__setattr__(self, "dp_size", dp_size)

    def rank(self, rank: int) -> TopologyRankDesc:
        return self.ranks[_integer(rank, "DistributedTopology.rank")]

    def groups(self, kind: ParallelGroupKind) -> tuple[tuple[int, ...], ...]:
        kind = _enum(cast(object, kind), ParallelGroupKind, "DistributedTopology.group")
        model = tuple(item for item in self.ranks if item.role == RankRole.MODEL)
        if kind == ParallelGroupKind.GLOBAL:
            return (tuple(item.rank for item in model),)
        if kind == ParallelGroupKind.HELPER:
            helpers = tuple(
                item.rank for item in self.ranks if item.role == RankRole.HELPER
            )
            return () if not helpers else (helpers,)
        grouped: dict[tuple[int, int], list[TopologyRankDesc]] = {}
        for item in model:
            assert item.tp_rank is not None
            assert item.pp_rank is not None
            assert item.dp_rank is not None
            if kind == ParallelGroupKind.TENSOR:
                group_key = (item.dp_rank, item.pp_rank)
                order = item.tp_rank
            elif kind == ParallelGroupKind.PIPELINE:
                group_key = (item.dp_rank, item.tp_rank)
                order = item.pp_rank
            else:
                group_key = (item.pp_rank, item.tp_rank)
                order = item.dp_rank
            grouped.setdefault(group_key, []).append(item)
            grouped[group_key].sort(
                key=lambda candidate: (
                    candidate.tp_rank
                    if kind == ParallelGroupKind.TENSOR
                    else candidate.pp_rank
                    if kind == ParallelGroupKind.PIPELINE
                    else candidate.dp_rank,
                    candidate.rank,
                )
            )
            _ = order
        return tuple(
            tuple(item.rank for item in grouped[key]) for key in sorted(grouped)
        )


@dataclass(frozen=True, slots=True)
class StateResidenceDesc:
    key: StateKey
    rank: int
    placement: Placement
    shard_index: int
    replica_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _instance(self.key, StateKey, "StateResidenceDesc.key"))
        object.__setattr__(self, "rank", _integer(self.rank, "StateResidenceDesc.rank"))
        object.__setattr__(
            self,
            "placement",
            _instance(self.placement, Placement, "StateResidenceDesc.placement"),
        )
        object.__setattr__(
            self,
            "shard_index",
            _integer(self.shard_index, "StateResidenceDesc.shard_index"),
        )
        object.__setattr__(
            self,
            "replica_index",
            _integer(self.replica_index, "StateResidenceDesc.replica_index"),
        )


@dataclass(frozen=True, slots=True, order=True)
class CommunicatorKey:
    kind: ParallelGroupKind
    group_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(cast(object, self.kind), ParallelGroupKind, "CommunicatorKey.kind"),
        )
        object.__setattr__(
            self,
            "group_index",
            _integer(self.group_index, "CommunicatorKey.group_index"),
        )

    @property
    def value(self) -> str:
        return f"{self.kind.value}:{self.group_index}"


@dataclass(frozen=True, slots=True)
class CommunicatorDesc:
    key: CommunicatorKey
    ranks: tuple[int, ...]
    backend: CollectiveBackend = CollectiveBackend.NCCL

    def __post_init__(self) -> None:
        key = _instance(self.key, CommunicatorKey, "CommunicatorDesc.key")
        raw_ranks = cast(object, self.ranks)
        if not isinstance(raw_ranks, tuple):
            raise TypeError("CommunicatorDesc.ranks must be tuple")
        ranks = tuple(
            _integer(rank, "CommunicatorDesc.rank")
            for rank in cast(tuple[object, ...], raw_ranks)
        )
        if not ranks or len(set(ranks)) != len(ranks):
            raise ValueError("CommunicatorDesc ranks must be non-empty and unique")
        backend = _enum(
            cast(object, self.backend),
            CollectiveBackend,
            "CommunicatorDesc.backend",
        )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "backend", backend)


@dataclass(frozen=True, slots=True, order=True)
class CollectiveKey:
    namespace: str
    name: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace",
            _text(self.namespace, "CollectiveKey.namespace"),
        )
        object.__setattr__(self, "name", _text(self.name, "CollectiveKey.name"))
        object.__setattr__(
            self,
            "ordinal",
            _integer(self.ordinal, "CollectiveKey.ordinal"),
        )

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}/{self.ordinal}"


@dataclass(frozen=True, slots=True)
class CollectiveRequestDesc:
    key: CollectiveKey
    kind: CollectiveKind
    group: ParallelGroupKind
    group_index: int = 0
    backend: CollectiveBackend = CollectiveBackend.NCCL
    window: CollectiveWindow = CollectiveWindow.AFTER_COMPUTE
    value: ValueKey | None = None
    depends_on: tuple[CollectiveKey, ...] = ()
    root_rank: int | None = None

    def __post_init__(self) -> None:
        key = _instance(self.key, CollectiveKey, "CollectiveRequestDesc.key")
        kind = _enum(cast(object, self.kind), CollectiveKind, "CollectiveRequestDesc.kind")
        group = _enum(
            cast(object, self.group),
            ParallelGroupKind,
            "CollectiveRequestDesc.group",
        )
        group_index = _integer(
            self.group_index,
            "CollectiveRequestDesc.group_index",
        )
        backend = _enum(
            cast(object, self.backend),
            CollectiveBackend,
            "CollectiveRequestDesc.backend",
        )
        window = _enum(
            cast(object, self.window),
            CollectiveWindow,
            "CollectiveRequestDesc.window",
        )
        value = self.value
        if value is not None:
            value = _instance(value, ValueKey, "CollectiveRequestDesc.value")
        dependencies = _tuple_of(
            self.depends_on,
            CollectiveKey,
            "CollectiveRequestDesc.depends_on",
        )
        if len(set(dependencies)) != len(dependencies) or key in dependencies:
            raise ValueError("collective dependencies must be unique and non-self")
        root_rank = self.root_rank
        if kind in (CollectiveKind.BROADCAST, CollectiveKind.SEND_RECV):
            if root_rank is None:
                raise ValueError("broadcast/send_recv requires root_rank")
            root_rank = _integer(root_rank, "CollectiveRequestDesc.root_rank")
        elif root_rank is not None:
            raise ValueError("only broadcast/send_recv may declare root_rank")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "group_index", group_index)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(self, "root_rank", root_rank)


@dataclass(frozen=True, slots=True)
class CollectiveDesc:
    key: CollectiveKey
    kind: CollectiveKind
    communicator: CommunicatorKey
    sequence: int
    window: CollectiveWindow
    value: ValueKey | None = None
    depends_on: tuple[CollectiveKey, ...] = ()
    root_rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _instance(self.key, CollectiveKey, "CollectiveDesc.key"))
        object.__setattr__(
            self,
            "kind",
            _enum(cast(object, self.kind), CollectiveKind, "CollectiveDesc.kind"),
        )
        object.__setattr__(
            self,
            "communicator",
            _instance(self.communicator, CommunicatorKey, "CollectiveDesc.communicator"),
        )
        object.__setattr__(
            self,
            "sequence",
            _integer(self.sequence, "CollectiveDesc.sequence"),
        )
        object.__setattr__(
            self,
            "window",
            _enum(cast(object, self.window), CollectiveWindow, "CollectiveDesc.window"),
        )
        if self.value is not None:
            object.__setattr__(
                self,
                "value",
                _instance(self.value, ValueKey, "CollectiveDesc.value"),
            )
        dependencies = _tuple_of(
            self.depends_on,
            CollectiveKey,
            "CollectiveDesc.depends_on",
        )
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("CollectiveDesc dependencies must be unique")
        object.__setattr__(self, "depends_on", dependencies)
        if self.root_rank is not None:
            object.__setattr__(
                self,
                "root_rank",
                _integer(self.root_rank, "CollectiveDesc.root_rank"),
            )


@dataclass(frozen=True, slots=True)
class LocalCollectiveSchedule:
    rank: int
    collectives: tuple[CollectiveKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank", _integer(self.rank, "LocalCollectiveSchedule.rank"))
        collectives = _tuple_of(
            self.collectives,
            CollectiveKey,
            "LocalCollectiveSchedule.collectives",
        )
        if len(set(collectives)) != len(collectives):
            raise ValueError("LocalCollectiveSchedule collectives must be unique")
        object.__setattr__(self, "collectives", collectives)


def _eligible_ranks(
    topology: DistributedTopology,
    affinity: PlacementAffinity,
) -> tuple[TopologyRankDesc, ...]:
    candidates = tuple(
        item
        for item in topology.ranks
        if (affinity.ranks == () or item.rank in affinity.ranks)
        and (affinity.node_ids == () or item.node_id in affinity.node_ids)
        and (
            item.role == (affinity.role or RankRole.MODEL)
        )
    )
    if affinity.ranks and any(rank >= len(topology.ranks) for rank in affinity.ranks):
        raise ValueError("state affinity references a rank outside topology")
    if not candidates:
        raise ValueError("state affinity has no eligible topology rank")
    return candidates


def _compile_state_residences(
    topology: DistributedTopology,
    requirements: tuple[StateRequirementDesc, ...],
) -> tuple[StateResidenceDesc, ...]:
    result: list[StateResidenceDesc] = []
    for requirement in sorted(requirements, key=lambda item: item.key.value):
        if requirement.scope == StateScope.REQUEST:
            raise ValueError("request-scoped state cannot enter a distributed plan")
        shard = requirement.shard
        expected_partitions = {
            ShardKind.TENSOR: topology.tp_size,
            ShardKind.PIPELINE: topology.pp_size,
            ShardKind.DATA: topology.dp_size,
        }.get(shard.kind, 1)
        if shard.partitions != expected_partitions:
            raise ValueError("ShardSpec partitions must match its topology axis")
        eligible = _eligible_ranks(topology, requirement.affinity)
        if shard.kind == ShardKind.REPLICATED:
            for replica_index, rank in enumerate(eligible):
                result.append(
                    StateResidenceDesc(
                        requirement.key,
                        rank.rank,
                        rank.placement,
                        0,
                        replica_index,
                    )
                )
            continue
        def shard_coordinates(rank: TopologyRankDesc) -> tuple[int, tuple[int, int]]:
            assert rank.tp_rank is not None
            assert rank.pp_rank is not None
            assert rank.dp_rank is not None
            if shard.kind == ShardKind.TENSOR:
                return rank.tp_rank, (rank.pp_rank, rank.dp_rank)
            if shard.kind == ShardKind.PIPELINE:
                return rank.pp_rank, (rank.tp_rank, rank.dp_rank)
            return rank.dp_rank, (rank.tp_rank, rank.pp_rank)

        groups: dict[tuple[int, int], list[TopologyRankDesc]] = {}
        for rank in eligible:
            if rank.role != RankRole.MODEL:
                raise ValueError("sharded state requires model-rank affinity")
            _shard_index, replica_key = shard_coordinates(rank)
            groups.setdefault(replica_key, []).append(rank)
        expected_indices = set(range(shard.partitions))
        for ranks in groups.values():
            actual_indices = {
                shard_coordinates(rank)[0] for rank in ranks
            }
            if len(ranks) != shard.partitions or actual_indices != expected_indices:
                raise ValueError(
                    "eligible state ranks must exactly cover each topology shard axis"
                )
        replica_indices = {
            key: index for index, key in enumerate(sorted(groups))
        }
        for rank in eligible:
            shard_index, replica_key = shard_coordinates(rank)
            result.append(
                StateResidenceDesc(
                    requirement.key,
                    rank.rank,
                    rank.placement,
                    shard_index,
                    replica_indices[replica_key],
                )
            )
    return tuple(result)


def _collective_order(
    requests: tuple[CollectiveRequestDesc, ...],
) -> tuple[CollectiveRequestDesc, ...]:
    by_key = {item.key: item for item in requests}
    if len(by_key) != len(requests):
        raise ValueError("collective request keys must be unique")
    if any(
        dependency not in by_key
        for item in requests
        for dependency in item.depends_on
    ):
        raise ValueError("collective request references an unknown dependency")
    if any(
        _COLLECTIVE_WINDOW_ORDER[by_key[dependency].window]
        > _COLLECTIVE_WINDOW_ORDER[item.window]
        for item in requests
        for dependency in item.depends_on
    ):
        raise ValueError("collective dependency cannot move backwards in lifecycle time")
    remaining = {key: set(item.depends_on) for key, item in by_key.items()}
    ordered: list[CollectiveRequestDesc] = []
    while remaining:
        ready = sorted(
            (key for key, dependencies in remaining.items() if not dependencies),
            key=lambda key: (
                _COLLECTIVE_WINDOW_ORDER[by_key[key].window],
                key.value,
            ),
        )
        if not ready:
            raise ValueError("collective dependencies contain a cycle")
        key = ready[0]
        ordered.append(by_key[key])
        del remaining[key]
        for dependencies in remaining.values():
            dependencies.discard(key)
    return tuple(ordered)


def _compile_communicators(
    topology: DistributedTopology,
    requests: tuple[CollectiveRequestDesc, ...],
) -> tuple[CommunicatorDesc, ...]:
    by_key: dict[CommunicatorKey, CommunicatorDesc] = {}
    for request in requests:
        groups = topology.groups(request.group)
        if request.group_index >= len(groups):
            raise ValueError("collective group index is outside topology")
        ranks = groups[request.group_index]
        if request.kind == CollectiveKind.SEND_RECV and len(ranks) != 2:
            raise ValueError("send_recv requires an exact two-rank communicator")
        if request.root_rank is not None and request.root_rank not in ranks:
            raise ValueError("collective root rank is outside its communicator")
        if request.backend == CollectiveBackend.NCCL and any(
            topology.rank(rank).placement.kind.value != "cuda" for rank in ranks
        ):
            raise ValueError("NCCL communicators require CUDA topology ranks")
        key = CommunicatorKey(request.group, request.group_index)
        descriptor = CommunicatorDesc(key, ranks, request.backend)
        previous = by_key.setdefault(key, descriptor)
        if previous != descriptor:
            raise ValueError("one communicator group cannot use conflicting backends")
    return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True, slots=True)
class DistributedFlowPlan:
    plan_epoch: int
    topology: DistributedTopology
    state_requirements: tuple[StateRequirementDesc, ...] = ()
    state_residences: tuple[StateResidenceDesc, ...] = ()
    collective_requests: tuple[CollectiveRequestDesc, ...] = ()
    communicators: tuple[CommunicatorDesc, ...] = ()
    collectives: tuple[CollectiveDesc, ...] = ()
    local_schedules: tuple[LocalCollectiveSchedule, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        epoch = _integer(self.plan_epoch, "DistributedFlowPlan.plan_epoch", minimum=1)
        topology = _instance(
            self.topology,
            DistributedTopology,
            "DistributedFlowPlan.topology",
        )
        requirements = _tuple_of(
            self.state_requirements,
            StateRequirementDesc,
            "DistributedFlowPlan.state_requirements",
        )
        residences = _tuple_of(
            self.state_residences,
            StateResidenceDesc,
            "DistributedFlowPlan.state_residences",
        )
        requests = _tuple_of(
            self.collective_requests,
            CollectiveRequestDesc,
            "DistributedFlowPlan.collective_requests",
        )
        communicators = _tuple_of(
            self.communicators,
            CommunicatorDesc,
            "DistributedFlowPlan.communicators",
        )
        collectives = _tuple_of(
            self.collectives,
            CollectiveDesc,
            "DistributedFlowPlan.collectives",
        )
        schedules = _tuple_of(
            self.local_schedules,
            LocalCollectiveSchedule,
            "DistributedFlowPlan.local_schedules",
        )
        if self.schema_version != 1:
            raise ValueError("DistributedFlowPlan.schema_version must be 1")
        if len({item.key for item in requirements}) != len(requirements):
            raise ValueError("distributed state requirement keys must be unique")
        expected_residences = _compile_state_residences(topology, requirements)
        if residences != expected_residences:
            raise ValueError(
                "DistributedFlowPlan state residences must exactly match requirements"
            )
        ordered_requests = _collective_order(requests)
        if requests != ordered_requests:
            raise ValueError(
                "DistributedFlowPlan collective requests must use canonical order"
            )
        expected_communicators = _compile_communicators(topology, ordered_requests)
        if communicators != expected_communicators:
            raise ValueError(
                "DistributedFlowPlan communicators must exactly match collective demand"
            )
        expected_collectives = tuple(
            CollectiveDesc(
                key=request.key,
                kind=request.kind,
                communicator=CommunicatorKey(request.group, request.group_index),
                sequence=sequence,
                window=request.window,
                value=request.value,
                depends_on=request.depends_on,
                root_rank=request.root_rank,
            )
            for sequence, request in enumerate(ordered_requests)
        )
        if collectives != expected_collectives:
            raise ValueError(
                "DistributedFlowPlan collectives must use canonical global order"
            )
        communicator_by_key = {item.key: item for item in communicators}
        expected_schedules = tuple(
            LocalCollectiveSchedule(
                rank.rank,
                tuple(
                    collective.key
                    for collective in collectives
                    if rank.rank
                    in communicator_by_key[collective.communicator].ranks
                ),
            )
            for rank in topology.ranks
        )
        if schedules != expected_schedules:
            raise ValueError(
                "DistributedFlowPlan local schedules must project global collective order"
            )
        object.__setattr__(self, "plan_epoch", epoch)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "state_requirements", requirements)
        object.__setattr__(self, "state_residences", residences)
        object.__setattr__(self, "collective_requests", ordered_requests)
        object.__setattr__(self, "communicators", communicators)
        object.__setattr__(self, "collectives", collectives)
        object.__setattr__(self, "local_schedules", schedules)


def compile_distributed_flow_plan(
    *,
    plan_epoch: int,
    topology: DistributedTopology,
    state_requirements: tuple[StateRequirementDesc, ...] = (),
    collective_requests: tuple[CollectiveRequestDesc, ...] = (),
) -> DistributedFlowPlan:
    topology = _instance(
        topology,
        DistributedTopology,
        "compile_distributed_flow_plan.topology",
    )
    requirements = _tuple_of(
        state_requirements,
        StateRequirementDesc,
        "compile_distributed_flow_plan.state_requirements",
    )
    requests = _tuple_of(
        collective_requests,
        CollectiveRequestDesc,
        "compile_distributed_flow_plan.collective_requests",
    )
    ordered_requests = _collective_order(requests)
    communicators = _compile_communicators(topology, ordered_requests)
    communicator_by_key = {item.key: item for item in communicators}
    collectives = tuple(
        CollectiveDesc(
            request.key,
            request.kind,
            CommunicatorKey(request.group, request.group_index),
            sequence,
            request.window,
            request.value,
            request.depends_on,
            request.root_rank,
        )
        for sequence, request in enumerate(ordered_requests)
    )
    schedules = tuple(
        LocalCollectiveSchedule(
            rank.rank,
            tuple(
                collective.key
                for collective in collectives
                if rank.rank
                in communicator_by_key[collective.communicator].ranks
            ),
        )
        for rank in topology.ranks
    )
    result = DistributedFlowPlan(
        plan_epoch=_integer(plan_epoch, "compile_distributed_flow_plan.plan_epoch", minimum=1),
        topology=topology,
        state_requirements=tuple(
            sorted(requirements, key=lambda item: item.key.value)
        ),
        state_residences=_compile_state_residences(topology, requirements),
        collective_requests=ordered_requests,
        communicators=communicators,
        collectives=collectives,
        local_schedules=schedules,
    )
    try:
        from tllm import core as _core

        _core.validate_distributed_flow_plan(result)
    except (TypeError, ValueError) as exc:
        raise DistributedCompilationError(
            "Rust distributed-plan descriptor validation failed"
        ) from exc
    return result


__all__ = [
    "CollectiveBackend",
    "CollectiveDesc",
    "CollectiveKey",
    "CollectiveKind",
    "CollectiveRequestDesc",
    "CollectiveWindow",
    "CommunicatorDesc",
    "CommunicatorKey",
    "DistributedCompilationError",
    "DistributedFlowPlan",
    "DistributedTopology",
    "LocalCollectiveSchedule",
    "ParallelGroupKind",
    "PlacementAffinity",
    "RankRole",
    "ShardKind",
    "ShardSpec",
    "StateKey",
    "StateRequirementDesc",
    "StateResidenceDesc",
    "TopologyRankDesc",
    "compile_distributed_flow_plan",
]

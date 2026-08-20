"""Typed local placement, transfer, and admission contracts for WorkGraph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import Protocol, TypeVar, cast, runtime_checkable

from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.physical import PhysicalFlowPlan


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class MemoryKind(StrEnum):
    HOST = "host"
    PINNED_HOST = "pinned_host"
    DEVICE = "device"


class MoveKind(StrEnum):
    H2D = "h2d"
    D2H = "d2h"
    P2P = "p2p"


class CopyEngine(StrEnum):
    H2D = "h2d"
    D2H = "d2h"
    P2P = "p2p"


class PlacementCompilationError(RuntimeError):
    """Raised when a frozen Phase 3 plan fails cross-language validation."""


_MAX_EXACT_PLACEMENT_ASSIGNMENTS = 100_000


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


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if not value or "0x" in value.lower() or "<" in value or ">" in value:
        raise ValueError(f"{label} must be a stable non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be int")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _executor_key(value: object, label: str) -> ExecutorKey:
    key = _instance(value, ExecutorKey, label)
    components = (
        _text(key.namespace, f"{label}.namespace"),
        _text(key.name, f"{label}.name"),
        _text(key.version, f"{label}.version"),
    )
    if any(":" in item or "@" in item for item in components):
        raise ValueError(f"{label} components must not contain ':' or '@'")
    return ExecutorKey(*components)


def _node_key(value: object, label: str) -> NodeKey:
    key = _instance(value, NodeKey, label)
    return NodeKey(
        _text(key.namespace, f"{label}.namespace"),
        _text(key.owner, f"{label}.owner"),
        _integer(key.ordinal, f"{label}.ordinal"),
    )


def _tuple_of(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return cast(tuple[_T, ...], items)


@dataclass(frozen=True, slots=True, order=True)
class Placement:
    kind: DeviceKind
    index: int | None = None

    def __post_init__(self) -> None:
        kind = _enum(cast(object, self.kind), DeviceKind, "Placement.kind")
        index = self.index
        if kind == DeviceKind.CPU:
            if index is not None:
                raise ValueError("CPU placement cannot declare a device index")
        elif index is None:
            raise ValueError("CUDA placement requires a device index")
        else:
            index = _integer(index, "Placement.index")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "index", index)

    def __str__(self) -> str:
        return self.kind.value if self.index is None else f"{self.kind.value}:{self.index}"


@dataclass(frozen=True, slots=True, order=True)
class ValueKey:
    name: str
    version: int

    def __post_init__(self) -> None:
        name = _text(self.name, "ValueKey.name")
        if "/" in name or "\\" in name:
            raise ValueError("ValueKey.name must not contain '/' or '\\'")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "version",
            _integer(self.version, "ValueKey.version", minimum=1),
        )

    @property
    def value(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class Residency:
    value: ValueKey
    placement: Placement
    memory: MemoryKind | None = None

    def __post_init__(self) -> None:
        value = _instance(self.value, ValueKey, "Residency.value")
        placement = _instance(self.placement, Placement, "Residency.placement")
        memory = self.memory
        if memory is None:
            memory = (
                MemoryKind.DEVICE
                if placement.kind == DeviceKind.CUDA
                else MemoryKind.HOST
            )
        else:
            memory = _enum(cast(object, memory), MemoryKind, "Residency.memory")
        if placement.kind == DeviceKind.CUDA and memory != MemoryKind.DEVICE:
            raise ValueError("CUDA residency requires device memory")
        if placement.kind == DeviceKind.CPU and memory == MemoryKind.DEVICE:
            raise ValueError("CPU residency cannot use device memory")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "memory", memory)


@dataclass(frozen=True, slots=True)
class TransferRequest:
    value: ValueKey
    source: Residency
    destination: Residency
    byte_count: int = 0

    def __post_init__(self) -> None:
        value = _instance(self.value, ValueKey, "TransferRequest.value")
        source = _instance(self.source, Residency, "TransferRequest.source")
        destination = _instance(
            self.destination, Residency, "TransferRequest.destination"
        )
        if source.value != value or destination.value != value:
            raise ValueError("transfer residencies must reference the requested value")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(
            self,
            "byte_count",
            _integer(self.byte_count, "TransferRequest.byte_count"),
        )


@dataclass(frozen=True, slots=True)
class Move:
    value: ValueKey
    source: Residency
    destination: Residency
    kind: MoveKind
    byte_count: int = 0
    asynchronous: bool = True

    def __post_init__(self) -> None:
        value = _instance(self.value, ValueKey, "Move.value")
        source = _instance(self.source, Residency, "Move.source")
        destination = _instance(self.destination, Residency, "Move.destination")
        kind = _enum(cast(object, self.kind), MoveKind, "Move.kind")
        if source.value != value or destination.value != value:
            raise ValueError("Move residencies must reference Move.value")
        if source.placement == destination.placement:
            raise ValueError("Move source and destination must differ")
        source_kind = source.placement.kind
        destination_kind = destination.placement.kind
        expected = (
            MoveKind.H2D
            if source_kind == DeviceKind.CPU and destination_kind == DeviceKind.CUDA
            else MoveKind.D2H
            if source_kind == DeviceKind.CUDA and destination_kind == DeviceKind.CPU
            else MoveKind.P2P
            if source_kind == DeviceKind.CUDA
            and destination_kind == DeviceKind.CUDA
            else None
        )
        if kind != expected:
            raise ValueError("Move kind does not match its typed placements")
        if not isinstance(cast(object, self.asynchronous), bool):
            raise TypeError("Move.asynchronous must be bool")
        if not self.asynchronous:
            raise ValueError("Phase 3 Move must be asynchronous")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "byte_count",
            _integer(self.byte_count, "Move.byte_count"),
        )

    @property
    def engine(self) -> CopyEngine:
        return CopyEngine(self.kind.value)


@dataclass(frozen=True, slots=True)
class DeviceTopology:
    peer_access: frozenset[tuple[int, int]]
    cuda_devices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        peers = cast(object, self.peer_access)
        if not isinstance(peers, frozenset):
            raise TypeError("DeviceTopology.peer_access must be frozenset")
        normalized: set[tuple[int, int]] = set()
        for edge in cast(frozenset[object], peers):
            if not isinstance(edge, tuple):
                raise TypeError("peer access entries must be (source, destination)")
            pair = cast(tuple[object, ...], edge)
            if len(pair) != 2:
                raise TypeError("peer access entries must be (source, destination)")
            source = _integer(pair[0], "peer source")
            destination = _integer(pair[1], "peer destination")
            if source == destination:
                raise ValueError("peer access cannot contain a self edge")
            normalized.add((source, destination))
        devices = tuple(
            _integer(device, "DeviceTopology.cuda_devices")
            for device in self.cuda_devices
        )
        if len(set(devices)) != len(devices):
            raise ValueError("DeviceTopology.cuda_devices must be unique")
        if not devices and normalized:
            devices = tuple(sorted({index for edge in normalized for index in edge}))
        if any(
            source not in devices or destination not in devices
            for source, destination in normalized
        ):
            raise ValueError("peer access references an undeclared CUDA device")
        object.__setattr__(self, "peer_access", frozenset(normalized))
        object.__setattr__(self, "cuda_devices", tuple(sorted(devices)))

    def supports_peer(self, source: Placement, destination: Placement) -> bool:
        if source.kind != DeviceKind.CUDA or destination.kind != DeviceKind.CUDA:
            return False
        if source.index is None or destination.index is None:
            return False
        return (source.index, destination.index) in self.peer_access

    def contains(self, placement: Placement) -> bool:
        placement = _instance(placement, Placement, "DeviceTopology.placement")
        if placement.kind == DeviceKind.CPU:
            return True
        assert placement.index is not None
        return placement.index in self.cuda_devices


def compile_moves(
    requests: tuple[TransferRequest, ...],
    *,
    topology: DeviceTopology,
) -> tuple[Move, ...]:
    requests = _tuple_of(requests, TransferRequest, "compile_moves.requests")
    topology = _instance(topology, DeviceTopology, "compile_moves.topology")
    moves: list[Move] = []
    for request in requests:
        source = request.source
        destination = request.destination
        if not topology.contains(source.placement) or not topology.contains(
            destination.placement
        ):
            raise ValueError("transfer references a placement outside DeviceTopology")
        if source.placement == destination.placement:
            continue
        if (
            source.placement.kind == DeviceKind.CUDA
            and destination.placement.kind == DeviceKind.CUDA
        ):
            if topology.supports_peer(source.placement, destination.placement):
                moves.append(
                    Move(
                        request.value,
                        source,
                        destination,
                        MoveKind.P2P,
                        request.byte_count,
                    )
                )
                continue
            host = Residency(
                request.value,
                Placement(DeviceKind.CPU),
                MemoryKind.PINNED_HOST,
            )
            moves.extend(
                (
                    Move(
                        request.value,
                        source,
                        host,
                        MoveKind.D2H,
                        request.byte_count,
                    ),
                    Move(
                        request.value,
                        host,
                        destination,
                        MoveKind.H2D,
                        request.byte_count,
                    ),
                )
            )
            continue
        kind = (
            MoveKind.H2D
            if source.placement.kind == DeviceKind.CPU
            else MoveKind.D2H
        )
        moves.append(
            Move(
                request.value,
                source,
                destination,
                kind,
                request.byte_count,
            )
        )
    return tuple(moves)


@runtime_checkable
class RuntimeImplementation(Protocol):
    def execute(self, *args: object) -> object: ...


@dataclass(frozen=True, slots=True, order=True)
class SemanticOracleKey:
    namespace: str
    name: str
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _text(self.namespace, "SemanticOracleKey.namespace"))
        object.__setattr__(self, "name", _text(self.name, "SemanticOracleKey.name"))
        object.__setattr__(self, "version", _text(self.version, "SemanticOracleKey.version"))

    @property
    def value(self) -> str:
        return f"{self.namespace}:{self.name}@{self.version}"


@runtime_checkable
class SemanticOracle(Protocol):
    def equivalent(self, reference: object, candidate: object) -> bool: ...


@dataclass(frozen=True, slots=True)
class SemanticOracleSpec:
    key: SemanticOracleKey
    oracle: SemanticOracle

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            _instance(self.key, SemanticOracleKey, "SemanticOracleSpec.key"),
        )
        if not isinstance(cast(object, self.oracle), SemanticOracle):
            raise TypeError("SemanticOracleSpec.oracle must implement equivalent()")


@dataclass(frozen=True, slots=True)
class SemanticOracleVerificationDesc:
    logical_executor: ExecutorKey
    oracle_key: SemanticOracleKey
    implementations: tuple[ExecutorKey, ...]
    case_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_executor",
            _instance(
                self.logical_executor,
                ExecutorKey,
                "SemanticOracleVerificationDesc.logical_executor",
            ),
        )
        object.__setattr__(
            self,
            "oracle_key",
            _instance(
                self.oracle_key,
                SemanticOracleKey,
                "SemanticOracleVerificationDesc.oracle_key",
            ),
        )
        implementations = _tuple_of(
            self.implementations,
            ExecutorKey,
            "SemanticOracleVerificationDesc.implementations",
        )
        if not implementations or len(set(implementations)) != len(implementations):
            raise ValueError("semantic oracle implementations must be non-empty and unique")
        object.__setattr__(self, "implementations", implementations)
        object.__setattr__(
            self,
            "case_count",
            _integer(
                self.case_count,
                "SemanticOracleVerificationDesc.case_count",
                minimum=1,
            ),
        )


class SemanticOracleMismatch(RuntimeError):
    """Raised when cross-placement implementations are not proven equivalent."""


@dataclass(frozen=True, slots=True)
class DurationProfile:
    microseconds: int
    samples: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "microseconds",
            _integer(self.microseconds, "DurationProfile.microseconds", minimum=1),
        )
        object.__setattr__(
            self,
            "samples",
            _integer(self.samples, "DurationProfile.samples", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class TransferProfile:
    """Conservative per-engine duration snapshot used by the list scheduler."""

    h2d: DurationProfile | None = None
    d2h: DurationProfile | None = None
    p2p: DurationProfile | None = None
    fallback: DurationProfile = DurationProfile(16)

    def __post_init__(self) -> None:
        for name, value in (
            ("h2d", self.h2d),
            ("d2h", self.d2h),
            ("p2p", self.p2p),
        ):
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _instance(value, DurationProfile, f"TransferProfile.{name}"),
                )
        object.__setattr__(
            self,
            "fallback",
            _instance(self.fallback, DurationProfile, "TransferProfile.fallback"),
        )

    def for_engine(self, engine: CopyEngine) -> tuple[DurationProfile, bool]:
        engine = _enum(cast(object, engine), CopyEngine, "TransferProfile.engine")
        profile = {
            CopyEngine.H2D: self.h2d,
            CopyEngine.D2H: self.d2h,
            CopyEngine.P2P: self.p2p,
        }[engine]
        return (self.fallback, True) if profile is None else (profile, False)


@dataclass(frozen=True, slots=True, order=True)
class MoveKey:
    value: ValueKey
    ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _instance(self.value, ValueKey, "MoveKey.value"))
        object.__setattr__(
            self,
            "ordinal",
            _integer(self.ordinal, "MoveKey.ordinal"),
        )

    @property
    def text(self) -> str:
        return f"{self.value.value}/move/{self.ordinal}"


@dataclass(frozen=True, slots=True)
class ScheduledMove:
    key: MoveKey
    move: Move
    engine: CopyEngine
    start_at: int
    end_at: int
    producer: NodeKey | None = None
    consumer: NodeKey | None = None
    depends_on: tuple[MoveKey, ...] = ()
    used_fallback: bool = False

    def __post_init__(self) -> None:
        key = _instance(self.key, MoveKey, "ScheduledMove.key")
        move = _instance(self.move, Move, "ScheduledMove.move")
        engine = _enum(cast(object, self.engine), CopyEngine, "ScheduledMove.engine")
        start = _integer(self.start_at, "ScheduledMove.start_at")
        end = _integer(self.end_at, "ScheduledMove.end_at", minimum=1)
        dependencies = _tuple_of(
            self.depends_on, MoveKey, "ScheduledMove.depends_on"
        )
        producer = self.producer
        consumer = self.consumer
        if (producer is None) != (consumer is None):
            raise ValueError("ScheduledMove producer and consumer must be declared together")
        if producer is not None and consumer is not None:
            producer = _node_key(producer, "ScheduledMove.producer")
            consumer = _node_key(consumer, "ScheduledMove.consumer")
            if producer == consumer:
                raise ValueError("ScheduledMove producer and consumer must differ")
        if key.value != move.value:
            raise ValueError("ScheduledMove key must reference its Move value")
        if engine != move.engine:
            raise ValueError("ScheduledMove engine must match Move kind")
        if end <= start:
            raise ValueError("ScheduledMove.end_at must be greater than start_at")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("ScheduledMove dependencies must be unique")
        if not isinstance(cast(object, self.used_fallback), bool):
            raise TypeError("ScheduledMove.used_fallback must be bool")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "move", move)
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "start_at", start)
        object.__setattr__(self, "end_at", end)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "consumer", consumer)
        object.__setattr__(self, "depends_on", dependencies)


def schedule_moves(
    moves: tuple[Move, ...],
    *,
    profile: TransferProfile,
    start_at: int = 0,
    capacities: tuple[CopyEngineCapacityDesc, ...] = (),
    ordinal_start: int = 0,
) -> tuple[ScheduledMove, ...]:
    """Schedule typed transfers on bounded directional copy engines."""

    moves = _tuple_of(moves, Move, "schedule_moves.moves")
    profile = _instance(profile, TransferProfile, "schedule_moves.profile")
    start = _integer(start_at, "schedule_moves.start_at")
    ordinal = _integer(ordinal_start, "schedule_moves.ordinal_start")
    capacities = _tuple_of(
        capacities, CopyEngineCapacityDesc, "schedule_moves.capacities"
    )
    capacity_by_engine: dict[CopyEngine, int] = {
        CopyEngine.H2D: 1,
        CopyEngine.D2H: 1,
        CopyEngine.P2P: 1,
    }
    declared_engines: set[CopyEngine] = set()
    for descriptor in capacities:
        if descriptor.engine in declared_engines:
            raise ValueError("copy engine capacities must be unique")
        declared_engines.add(descriptor.engine)
        capacity_by_engine[descriptor.engine] = descriptor.capacity
    lanes: dict[CopyEngine, list[int]] = {
        engine: [start] * capacity for engine, capacity in capacity_by_engine.items()
    }
    pinned_ready: dict[Residency, tuple[int, MoveKey]] = {}
    scheduled: list[ScheduledMove] = []
    for offset, move in enumerate(moves):
        engine = move.engine
        engine_lanes = lanes[engine]
        lane = min(range(len(engine_lanes)), key=lambda index: (engine_lanes[index], index))
        ready = start
        dependencies: tuple[MoveKey, ...] = ()
        if move.source.memory == MemoryKind.PINNED_HOST:
            staged = pinned_ready.get(move.source)
            if staged is not None:
                ready, dependency = staged
                dependencies = (dependency,)
        begin = max(ready, engine_lanes[lane])
        duration, used_fallback = profile.for_engine(engine)
        end = begin + duration.microseconds
        key = MoveKey(move.value, ordinal + offset)
        item = ScheduledMove(
            key=key,
            move=move,
            engine=engine,
            start_at=begin,
            end_at=end,
            depends_on=dependencies,
            used_fallback=used_fallback,
        )
        scheduled.append(item)
        engine_lanes[lane] = end
        if move.destination.memory == MemoryKind.PINNED_HOST:
            pinned_ready[move.destination] = (end, key)
    return tuple(scheduled)


@dataclass(frozen=True, slots=True)
class TransferProfileDesc:
    """Frozen linear transfer model collected outside the planner."""

    kind: MoveKind
    source: Placement
    destination: Placement
    bytes_per_microsecond: int
    fixed_latency_microseconds: int = 1
    samples: int = 1

    def __post_init__(self) -> None:
        kind = _enum(cast(object, self.kind), MoveKind, "TransferProfileDesc.kind")
        source = _instance(self.source, Placement, "TransferProfileDesc.source")
        destination = _instance(
            self.destination, Placement, "TransferProfileDesc.destination"
        )
        probe = Move(
            ValueKey("profile", 1),
            Residency(ValueKey("profile", 1), source),
            Residency(ValueKey("profile", 1), destination),
            kind,
        )
        del probe
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(
            self,
            "bytes_per_microsecond",
            _integer(
                self.bytes_per_microsecond,
                "TransferProfileDesc.bytes_per_microsecond",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "fixed_latency_microseconds",
            _integer(
                self.fixed_latency_microseconds,
                "TransferProfileDesc.fixed_latency_microseconds",
            ),
        )
        object.__setattr__(
            self,
            "samples",
            _integer(self.samples, "TransferProfileDesc.samples", minimum=1),
        )

    def duration(self, byte_count: int) -> int:
        size = _integer(byte_count, "TransferProfileDesc.byte_count")
        transfer = (size + self.bytes_per_microsecond - 1) // self.bytes_per_microsecond
        return max(1, self.fixed_latency_microseconds + transfer)


@dataclass(frozen=True, slots=True)
class CopyEngineCapacityDesc:
    engine: CopyEngine
    capacity: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "engine",
            _enum(cast(object, self.engine), CopyEngine, "CopyEngineCapacityDesc.engine"),
        )
        object.__setattr__(
            self,
            "capacity",
            _integer(self.capacity, "CopyEngineCapacityDesc.capacity", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class ValueFlowDesc:
    value: ValueKey
    producer: NodeKey
    consumer: NodeKey
    byte_count: int

    def __post_init__(self) -> None:
        value = _instance(self.value, ValueKey, "ValueFlowDesc.value")
        producer = _node_key(self.producer, "ValueFlowDesc.producer")
        consumer = _node_key(self.consumer, "ValueFlowDesc.consumer")
        if producer == consumer:
            raise ValueError("ValueFlowDesc producer and consumer must differ")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "consumer", consumer)
        object.__setattr__(
            self,
            "byte_count",
            _integer(self.byte_count, "ValueFlowDesc.byte_count", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class ImplementationSpec:
    key: ExecutorKey
    placement: Placement
    implementation: RuntimeImplementation
    logical_executor: ExecutorKey | None = None
    duration: DurationProfile = DurationProfile(1)
    oracle_key: SemanticOracleKey | None = None

    def __post_init__(self) -> None:
        key = _executor_key(self.key, "ImplementationSpec.key")
        placement = _instance(
            self.placement, Placement, "ImplementationSpec.placement"
        )
        implementation = cast(object, self.implementation)
        if not isinstance(implementation, RuntimeImplementation):
            raise TypeError("ImplementationSpec.implementation must implement execute")
        logical = self.logical_executor
        if logical is None:
            logical = key
        else:
            logical = _executor_key(logical, "ImplementationSpec.logical_executor")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "logical_executor", logical)
        object.__setattr__(
            self,
            "duration",
            _instance(self.duration, DurationProfile, "ImplementationSpec.duration"),
        )
        if self.oracle_key is not None:
            object.__setattr__(
                self,
                "oracle_key",
                _instance(
                    self.oracle_key,
                    SemanticOracleKey,
                    "ImplementationSpec.oracle_key",
                ),
            )


@dataclass(frozen=True, slots=True)
class ImplementationDesc:
    key: ExecutorKey
    logical_executor: ExecutorKey
    placement: Placement
    duration: DurationProfile
    oracle_key: SemanticOracleKey | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _executor_key(self.key, "ImplementationDesc.key"))
        object.__setattr__(
            self,
            "logical_executor",
            _executor_key(
                self.logical_executor,
                "ImplementationDesc.logical_executor",
            ),
        )
        object.__setattr__(
            self,
            "placement",
            _instance(self.placement, Placement, "ImplementationDesc.placement"),
        )
        object.__setattr__(
            self,
            "duration",
            _instance(self.duration, DurationProfile, "ImplementationDesc.duration"),
        )
        if self.oracle_key is not None:
            object.__setattr__(
                self,
                "oracle_key",
                _instance(
                    self.oracle_key,
                    SemanticOracleKey,
                    "ImplementationDesc.oracle_key",
                ),
            )


class ImplementationRegistry:
    """Control-plane registry; runtime callables never enter frozen plans."""

    __slots__ = (
        "_specs",
        "_by_logical",
        "_oracles",
        "_verified",
        "_verification_cases",
    )

    def __init__(self) -> None:
        self._specs: dict[ExecutorKey, ImplementationSpec] = {}
        self._by_logical: dict[ExecutorKey, list[ImplementationSpec]] = {}
        self._oracles: dict[SemanticOracleKey, SemanticOracle] = {}
        self._verified: dict[ExecutorKey, SemanticOracleVerificationDesc] = {}
        self._verification_cases: dict[
            ExecutorKey, tuple[tuple[object, ...], ...]
        ] = {}

    def register_oracle(self, spec: SemanticOracleSpec) -> None:
        spec = _instance(spec, SemanticOracleSpec, "ImplementationRegistry.oracle")
        if spec.key in self._oracles:
            raise ValueError(f"duplicate semantic oracle key {spec.key.value}")
        self._oracles[spec.key] = spec.oracle

    def register(self, spec: ImplementationSpec) -> None:
        spec = _instance(spec, ImplementationSpec, "ImplementationRegistry.spec")
        if spec.key in self._specs:
            raise ValueError(f"duplicate implementation key {spec.key.value}")
        self._specs[spec.key] = spec
        assert spec.logical_executor is not None
        self._by_logical.setdefault(spec.logical_executor, []).append(spec)
        self._by_logical[spec.logical_executor].sort(
            key=lambda candidate: (
                candidate.duration.microseconds,
                str(candidate.placement),
                candidate.key.value,
            )
        )
        self._verified.pop(spec.logical_executor, None)
        self._verification_cases.pop(spec.logical_executor, None)

    def resolve(self, key: ExecutorKey) -> RuntimeImplementation:
        key = _executor_key(key, "ImplementationRegistry.key")
        return self._specs[key].implementation

    def candidates(self, logical_executor: ExecutorKey) -> tuple[ImplementationDesc, ...]:
        logical_executor = _executor_key(
            logical_executor,
            "ImplementationRegistry.logical_executor",
        )
        specs = tuple(self._by_logical.get(logical_executor, ()))
        placements = {spec.placement for spec in specs}
        if len(placements) > 1:
            oracle_keys = {spec.oracle_key for spec in specs}
            if None in oracle_keys or len(oracle_keys) != 1:
                raise SemanticOracleMismatch(
                    "cross-placement implementations require one shared semantic oracle"
                )
            verification = self._verified.get(logical_executor)
            oracle_key = cast(SemanticOracleKey, next(iter(oracle_keys)))
            cases = self._verification_cases.get(logical_executor)
            implementation_keys = tuple(spec.key for spec in specs)
            if (
                verification is None
                or verification.oracle_key != oracle_key
                or verification.implementations != implementation_keys
                or cases is None
            ):
                raise SemanticOracleMismatch(
                    "cross-placement implementations require semantic oracle verification"
                )
            self._verify_current_semantics(
                specs=specs,
                oracle_key=oracle_key,
                cases=cases,
            )
        return tuple(
            ImplementationDesc(
                key=spec.key,
                logical_executor=logical_executor,
                placement=spec.placement,
                duration=spec.duration,
                oracle_key=spec.oracle_key,
            )
            for spec in specs
        )

    def verify_semantics(
        self,
        logical_executor: ExecutorKey,
        *,
        cases: tuple[tuple[object, ...], ...],
    ) -> SemanticOracleVerificationDesc:
        logical_executor = _executor_key(
            logical_executor,
            "ImplementationRegistry.logical_executor",
        )
        specs = tuple(self._by_logical.get(logical_executor, ()))
        if len(specs) < 2 or len({spec.placement for spec in specs}) < 2:
            raise ValueError("semantic verification requires cross-placement implementations")
        raw_cases = cast(object, cases)
        if not isinstance(raw_cases, tuple) or not raw_cases or any(
            not isinstance(case, tuple) for case in cast(tuple[object, ...], raw_cases)
        ):
            raise TypeError("semantic oracle cases must be a non-empty tuple of tuples")
        typed_cases = cast(tuple[tuple[object, ...], ...], raw_cases)
        oracle_keys = {spec.oracle_key for spec in specs}
        if None in oracle_keys or len(oracle_keys) != 1:
            raise SemanticOracleMismatch(
                "cross-placement implementations require one shared semantic oracle"
            )
        oracle_key = cast(SemanticOracleKey, next(iter(oracle_keys)))
        self._verify_current_semantics(
            specs=specs,
            oracle_key=oracle_key,
            cases=typed_cases,
        )
        verification = SemanticOracleVerificationDesc(
            logical_executor=logical_executor,
            oracle_key=oracle_key,
            implementations=tuple(spec.key for spec in specs),
            case_count=len(typed_cases),
        )
        self._verified[logical_executor] = verification
        self._verification_cases[logical_executor] = typed_cases
        return verification

    def _verify_current_semantics(
        self,
        *,
        specs: tuple[ImplementationSpec, ...],
        oracle_key: SemanticOracleKey,
        cases: tuple[tuple[object, ...], ...],
    ) -> None:
        try:
            oracle = self._oracles[oracle_key]
        except KeyError as exc:
            raise SemanticOracleMismatch(
                f"semantic oracle {oracle_key.value} is not registered"
            ) from exc
        reference = specs[0]
        for case_index, case in enumerate(cases):
            expected = reference.implementation.execute(*case)
            for candidate in specs[1:]:
                actual = candidate.implementation.execute(*case)
                if not oracle.equivalent(expected, actual):
                    raise SemanticOracleMismatch(
                        "semantic oracle mismatch for "
                        f"{candidate.key.value} on case {case_index}"
                    )

    def keys(self) -> tuple[ExecutorKey, ...]:
        return tuple(sorted(self._specs, key=lambda key: key.value))

    def frozen(self) -> tuple[ImplementationDesc, ...]:
        return tuple(
            ImplementationDesc(
                key=spec.key,
                logical_executor=cast(ExecutorKey, spec.logical_executor),
                placement=spec.placement,
                duration=spec.duration,
                oracle_key=spec.oracle_key,
            )
            for spec in sorted(self._specs.values(), key=lambda item: item.key.value)
        )


@dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    key: str
    priority: int
    ready_at: int
    depends_on: tuple[str, ...] = ()
    duration: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _text(self.key, "ScheduleCandidate.key"))
        if not isinstance(cast(object, self.priority), int) or isinstance(
            cast(object, self.priority), bool
        ):
            raise TypeError("ScheduleCandidate.priority must be int")
        object.__setattr__(
            self,
            "ready_at",
            _integer(self.ready_at, "ScheduleCandidate.ready_at"),
        )
        raw_dependencies = cast(object, self.depends_on)
        if not isinstance(raw_dependencies, tuple) or any(
            not isinstance(key, str) or not key
            for key in cast(tuple[object, ...], raw_dependencies)
        ):
            raise TypeError("ScheduleCandidate.depends_on must be tuple[str, ...]")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("ScheduleCandidate dependencies must be unique")
        object.__setattr__(
            self,
            "duration",
            _integer(self.duration, "ScheduleCandidate.duration", minimum=1),
        )


def schedule_candidates(
    candidates: Sequence[ScheduleCandidate],
) -> tuple[ScheduleCandidate, ...]:
    typed = cast(tuple[object, ...], tuple(candidates))
    if any(not isinstance(candidate, ScheduleCandidate) for candidate in typed):
        raise TypeError("schedule candidates must be ScheduleCandidate")
    candidates_typed = cast(tuple[ScheduleCandidate, ...], typed)
    by_key = {candidate.key: candidate for candidate in candidates_typed}
    if len(by_key) != len(candidates_typed):
        raise ValueError("schedule candidate keys must be unique")
    if any(
        dependency not in by_key
        for candidate in candidates_typed
        for dependency in candidate.depends_on
    ):
        raise ValueError("schedule candidate references an unknown dependency")
    successors: dict[str, list[str]] = {key: [] for key in by_key}
    for candidate in candidates_typed:
        for dependency in candidate.depends_on:
            successors[dependency].append(candidate.key)

    visiting: set[str] = set()
    critical_lengths: dict[str, int] = {}

    def critical_length(key: str) -> int:
        cached = critical_lengths.get(key)
        if cached is not None:
            return cached
        if key in visiting:
            raise ValueError("schedule candidate dependencies contain a cycle")
        visiting.add(key)
        child_lengths = tuple(critical_length(child) for child in successors[key])
        visiting.remove(key)
        result = by_key[key].duration + (max(child_lengths) if child_lengths else 0)
        critical_lengths[key] = result
        return result

    for key in sorted(by_key):
        critical_length(key)
    remaining = set(by_key)
    completed: set[str] = set()
    ordered: list[ScheduleCandidate] = []
    while remaining:
        ready = [
            by_key[key]
            for key in remaining
            if set(by_key[key].depends_on).issubset(completed)
        ]
        if not ready:
            raise ValueError("schedule candidate dependencies contain a cycle")
        chosen = min(
            ready,
            key=lambda candidate: (
                candidate.ready_at,
                -critical_lengths[candidate.key],
                -candidate.priority,
                candidate.key,
            ),
        )
        ordered.append(chosen)
        remaining.remove(chosen.key)
        completed.add(chosen.key)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class ImplementationSelectionDesc:
    node_key: NodeKey
    logical_executor: ExecutorKey
    key: ExecutorKey
    placement: Placement
    duration: DurationProfile
    oracle_key: SemanticOracleKey | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_key",
            _node_key(self.node_key, "ImplementationSelectionDesc.node_key"),
        )
        object.__setattr__(
            self,
            "logical_executor",
            _executor_key(
                self.logical_executor,
                "ImplementationSelectionDesc.logical_executor",
            ),
        )
        if self.oracle_key is not None:
            object.__setattr__(
                self,
                "oracle_key",
                _instance(
                    self.oracle_key,
                    SemanticOracleKey,
                    "ImplementationSelectionDesc.oracle_key",
                ),
            )
        object.__setattr__(
            self,
            "key",
            _executor_key(self.key, "ImplementationSelectionDesc.key"),
        )
        object.__setattr__(
            self,
            "placement",
            _instance(
                self.placement, Placement, "ImplementationSelectionDesc.placement"
            ),
        )
        object.__setattr__(
            self,
            "duration",
            _instance(
                self.duration,
                DurationProfile,
                "ImplementationSelectionDesc.duration",
            ),
        )


@dataclass(frozen=True, slots=True)
class PlacedNodeDesc:
    node_key: NodeKey
    implementation: ExecutorKey
    placement: Placement
    start_at: int
    end_at: int
    depends_on: tuple[NodeKey, ...] = ()

    def __post_init__(self) -> None:
        node_key = _node_key(self.node_key, "PlacedNodeDesc.node_key")
        implementation = _executor_key(
            self.implementation, "PlacedNodeDesc.implementation"
        )
        placement = _instance(self.placement, Placement, "PlacedNodeDesc.placement")
        start = _integer(self.start_at, "PlacedNodeDesc.start_at")
        end = _integer(self.end_at, "PlacedNodeDesc.end_at", minimum=1)
        dependencies = _tuple_of(
            self.depends_on, NodeKey, "PlacedNodeDesc.depends_on"
        )
        if end <= start:
            raise ValueError("PlacedNodeDesc.end_at must be greater than start_at")
        if node_key in dependencies or len(set(dependencies)) != len(dependencies):
            raise ValueError("PlacedNodeDesc dependencies must be unique and acyclic")
        object.__setattr__(self, "node_key", node_key)
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "start_at", start)
        object.__setattr__(self, "end_at", end)
        object.__setattr__(self, "depends_on", dependencies)


@dataclass(frozen=True, slots=True)
class PlacementLeaseDesc:
    residency: Residency
    live_start: int
    live_end: int

    def __post_init__(self) -> None:
        residency = _instance(
            self.residency, Residency, "PlacementLeaseDesc.residency"
        )
        start = _integer(self.live_start, "PlacementLeaseDesc.live_start")
        end = _integer(self.live_end, "PlacementLeaseDesc.live_end")
        if end < start:
            raise ValueError("PlacementLeaseDesc.live_end must be >= live_start")
        object.__setattr__(self, "residency", residency)
        object.__setattr__(self, "live_start", start)
        object.__setattr__(self, "live_end", end)


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    plan_epoch: int
    selections: tuple[ImplementationSelectionDesc, ...]
    schedule: tuple[PlacedNodeDesc, ...]
    topology: DeviceTopology
    value_flows: tuple[ValueFlowDesc, ...] = ()
    residencies: tuple[Residency, ...] = ()
    moves: tuple[ScheduledMove, ...] = ()
    leases: tuple[PlacementLeaseDesc, ...] = ()
    copy_engines: tuple[CopyEngineCapacityDesc, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        epoch = _integer(self.plan_epoch, "PlacementPlan.plan_epoch", minimum=1)
        selections = _tuple_of(
            self.selections,
            ImplementationSelectionDesc,
            "PlacementPlan.selections",
        )
        schedule = _tuple_of(self.schedule, PlacedNodeDesc, "PlacementPlan.schedule")
        residencies = _tuple_of(
            self.residencies, Residency, "PlacementPlan.residencies"
        )
        moves = _tuple_of(self.moves, ScheduledMove, "PlacementPlan.moves")
        leases = _tuple_of(self.leases, PlacementLeaseDesc, "PlacementPlan.leases")
        copy_engines = _tuple_of(
            self.copy_engines,
            CopyEngineCapacityDesc,
            "PlacementPlan.copy_engines",
        )
        raw_topology = _instance(
            self.topology, DeviceTopology, "PlacementPlan.topology"
        )
        value_flows = _tuple_of(
            self.value_flows, ValueFlowDesc, "PlacementPlan.value_flows"
        )
        topology = DeviceTopology(
            peer_access=raw_topology.peer_access,
            cuda_devices=raw_topology.cuda_devices,
        )

        def canonical_placement(value: Placement) -> Placement:
            return Placement(value.kind, value.index)

        def canonical_value(value: ValueKey) -> ValueKey:
            return ValueKey(value.name, value.version)

        def canonical_residency(value: Residency) -> Residency:
            return Residency(
                canonical_value(value.value),
                canonical_placement(value.placement),
                value.memory,
            )

        def canonical_oracle(
            value: SemanticOracleKey | None,
        ) -> SemanticOracleKey | None:
            if value is None:
                return None
            return SemanticOracleKey(value.namespace, value.name, value.version)

        selections = tuple(
            ImplementationSelectionDesc(
                node_key=item.node_key,
                logical_executor=item.logical_executor,
                key=item.key,
                placement=canonical_placement(item.placement),
                duration=DurationProfile(
                    item.duration.microseconds, item.duration.samples
                ),
                oracle_key=canonical_oracle(item.oracle_key),
            )
            for item in selections
        )
        schedule = tuple(
            PlacedNodeDesc(
                node_key=item.node_key,
                implementation=item.implementation,
                placement=canonical_placement(item.placement),
                start_at=item.start_at,
                end_at=item.end_at,
                depends_on=tuple(
                    _node_key(dependency, "PlacedNodeDesc.depends_on")
                    for dependency in item.depends_on
                ),
            )
            for item in schedule
        )
        residencies = tuple(canonical_residency(item) for item in residencies)
        moves = tuple(
            ScheduledMove(
                key=MoveKey(canonical_value(item.key.value), item.key.ordinal),
                move=Move(
                    value=canonical_value(item.move.value),
                    source=canonical_residency(item.move.source),
                    destination=canonical_residency(item.move.destination),
                    kind=item.move.kind,
                    byte_count=item.move.byte_count,
                    asynchronous=item.move.asynchronous,
                ),
                engine=item.engine,
                start_at=item.start_at,
                end_at=item.end_at,
                producer=item.producer,
                consumer=item.consumer,
                depends_on=tuple(
                    MoveKey(
                        canonical_value(dependency.value), dependency.ordinal
                    )
                    for dependency in item.depends_on
                ),
                used_fallback=item.used_fallback,
            )
            for item in moves
        )
        leases = tuple(
            PlacementLeaseDesc(
                canonical_residency(item.residency),
                item.live_start,
                item.live_end,
            )
            for item in leases
        )
        copy_engines = tuple(
            CopyEngineCapacityDesc(item.engine, item.capacity)
            for item in copy_engines
        )
        value_flows = tuple(
            ValueFlowDesc(
                canonical_value(item.value),
                item.producer,
                item.consumer,
                item.byte_count,
            )
            for item in value_flows
        )
        if self.schema_version != 1:
            raise ValueError("PlacementPlan.schema_version must be 1")
        selection_nodes = tuple(item.node_key for item in selections)
        schedule_nodes = tuple(item.node_key for item in schedule)
        if len(set(selection_nodes)) != len(selection_nodes):
            raise ValueError("PlacementPlan selections must have unique nodes")
        selections_by_logical: dict[ExecutorKey, list[ImplementationSelectionDesc]] = {}
        for selection in selections:
            selections_by_logical.setdefault(selection.logical_executor, []).append(selection)
        for logical_selections in selections_by_logical.values():
            if len({item.placement for item in logical_selections}) > 1:
                oracle_keys = {item.oracle_key for item in logical_selections}
                if None in oracle_keys or len(oracle_keys) != 1:
                    raise ValueError(
                        "cross-placement selections require one semantic oracle key"
                    )
        if any(
            not topology.contains(item.placement) for item in selections
        ):
            raise ValueError("PlacementPlan selection is outside its topology")
        if len(set(schedule_nodes)) != len(schedule_nodes):
            raise ValueError("PlacementPlan schedule must have unique nodes")
        if set(selection_nodes) != set(schedule_nodes):
            raise ValueError("PlacementPlan selections and schedule must cover the same nodes")
        selection_by_node = {item.node_key: item for item in selections}
        positions = {node: index for index, node in enumerate(schedule_nodes)}
        if len(
            {(item.value, item.producer, item.consumer) for item in value_flows}
        ) != len(value_flows):
            raise ValueError("PlacementPlan value flows must be unique")
        flow_producers: dict[ValueKey, NodeKey] = {}
        for flow in value_flows:
            if (
                flow.producer not in positions
                or flow.consumer not in positions
                or positions[flow.producer] >= positions[flow.consumer]
            ):
                raise ValueError("PlacementPlan value flow must follow its schedule")
            previous = flow_producers.setdefault(flow.value, flow.producer)
            if previous != flow.producer:
                raise ValueError("PlacementPlan ValueKey must have one producer")
        for item in schedule:
            selected = selection_by_node[item.node_key]
            if (
                selected.key != item.implementation
                or selected.placement != item.placement
            ):
                raise ValueError("PlacementPlan schedule differs from its selection")
            if item.end_at - item.start_at != selected.duration.microseconds:
                raise ValueError(
                    "PlacementPlan schedule duration differs from its selection"
                )
            if any(
                dependency not in positions
                or positions[dependency] >= positions[item.node_key]
                for dependency in item.depends_on
            ):
                raise ValueError("PlacementPlan schedule dependencies must be ordered")
            if any(
                schedule[positions[dependency]].end_at > item.start_at
                for dependency in item.depends_on
            ):
                raise ValueError(
                    "PlacementPlan schedule dependencies must finish before launch"
                )
        if len(set(residencies)) != len(residencies):
            raise ValueError("PlacementPlan residencies must be unique")
        if any(
            not topology.contains(item.placement) for item in residencies
        ):
            raise ValueError("PlacementPlan residency is outside its topology")
        residency_set = set(residencies)
        expected_residencies: set[Residency] = set()
        for flow in value_flows:
            required = (
                Residency(flow.value, selection_by_node[flow.producer].placement),
                Residency(flow.value, selection_by_node[flow.consumer].placement),
            )
            expected_residencies.update(required)
            if any(item not in residency_set for item in required):
                raise ValueError(
                    "PlacementPlan value flow is missing a required residency"
                )
        if any(
            item.move.source not in residency_set
            or item.move.destination not in residency_set
            for item in moves
        ):
            raise ValueError("PlacementPlan move references an unknown residency")
        expected_residencies.update(
            residency
            for item in moves
            for residency in (item.move.source, item.move.destination)
        )
        if residency_set != expected_residencies:
            raise ValueError(
                "PlacementPlan residencies must exactly cover value flows and routes"
            )
        schedule_by_node = {item.node_key: item for item in schedule}
        if any(item.producer is None or item.consumer is None for item in moves):
            raise ValueError("PlacementPlan moves require explicit producer and consumer nodes")
        for item in moves:
            assert item.producer is not None and item.consumer is not None
            if (
                item.move.kind == MoveKind.P2P
                and not topology.supports_peer(
                    item.move.source.placement,
                    item.move.destination.placement,
                )
            ):
                raise ValueError("PlacementPlan P2P move is outside its topology")
            producer = schedule_by_node.get(item.producer)
            consumer = schedule_by_node.get(item.consumer)
            if producer is None or consumer is None:
                raise ValueError("PlacementPlan move references an unknown scheduled node")
            if (
                positions[item.producer] >= positions[item.consumer]
                or item.start_at < producer.end_at
                or item.end_at > consumer.start_at
            ):
                raise ValueError("PlacementPlan move must complete between producer and consumer")
        routes: dict[
            tuple[NodeKey, NodeKey, ValueKey], list[ScheduledMove]
        ] = {}
        for item in moves:
            assert item.producer is not None and item.consumer is not None
            routes.setdefault(
                (item.producer, item.consumer, item.move.value), []
            ).append(item)
        for (producer_key, consumer_key, _value), route in routes.items():
            producer_placement = selection_by_node[producer_key].placement
            consumer_placement = selection_by_node[consumer_key].placement
            if route[0].move.source.placement != producer_placement:
                raise ValueError("PlacementPlan route source differs from its producer")
            if route[-1].move.destination.placement != consumer_placement:
                raise ValueError("PlacementPlan route destination differs from its consumer")
            if route[0].depends_on:
                raise ValueError("PlacementPlan route head must not depend on another move")
            if producer_placement == consumer_placement:
                raise ValueError("same-placement values must not have a Move route")
            if producer_placement.kind == DeviceKind.CPU:
                expected_kinds = (MoveKind.H2D,)
            elif consumer_placement.kind == DeviceKind.CPU:
                expected_kinds = (MoveKind.D2H,)
            elif topology.supports_peer(producer_placement, consumer_placement):
                expected_kinds = (MoveKind.P2P,)
            else:
                expected_kinds = (MoveKind.D2H, MoveKind.H2D)
            if tuple(item.move.kind for item in route) != expected_kinds:
                raise ValueError("PlacementPlan route differs from its topology matrix")
            if len(route) == 2:
                if (
                    route[0].move.destination != route[1].move.source
                    or route[0].move.destination.memory != MemoryKind.PINNED_HOST
                    or route[1].depends_on != (route[0].key,)
                ):
                    raise ValueError(
                        "PlacementPlan staged route must use one continuous pinned chain"
                    )
        expected_routes = {
            (flow.producer, flow.consumer, flow.value)
            for flow in value_flows
            if selection_by_node[flow.producer].placement
            != selection_by_node[flow.consumer].placement
        }
        if set(routes) != expected_routes:
            raise ValueError(
                "PlacementPlan routes must exactly cover cross-placement value flows"
            )
        move_positions = {item.key: index for index, item in enumerate(moves)}
        if len(move_positions) != len(moves):
            raise ValueError("PlacementPlan moves must have unique keys")
        if any(
            dependency not in move_positions
            or move_positions[dependency] >= move_positions[item.key]
            for item in moves
            for dependency in item.depends_on
        ):
            raise ValueError("PlacementPlan move dependencies must be ordered")
        if any(
            moves[move_positions[dependency]].end_at > item.start_at
            for item in moves
            for dependency in item.depends_on
        ):
            raise ValueError("PlacementPlan move dependencies must finish before launch")
        for item in moves:
            if item.move.source.memory == MemoryKind.PINNED_HOST and not any(
                moves[move_positions[dependency]].move.destination == item.move.source
                for dependency in item.depends_on
            ):
                raise ValueError("pinned-host H2D requires its D2H dependency")
        if any(lease.residency not in residency_set for lease in leases):
            raise ValueError("PlacementPlan lease references an unknown residency")
        lease_residencies = tuple(lease.residency for lease in leases)
        if len(set(lease_residencies)) != len(lease_residencies):
            raise ValueError("PlacementPlan leases must have unique residencies")
        if set(lease_residencies) != residency_set:
            raise ValueError("every PlacementPlan residency requires one lease")
        lease_by_residency = {lease.residency: lease for lease in leases}
        for flow in value_flows:
            producer = schedule_by_node[flow.producer]
            consumer = schedule_by_node[flow.consumer]
            for residency in (
                Residency(flow.value, selection_by_node[flow.producer].placement),
                Residency(flow.value, selection_by_node[flow.consumer].placement),
            ):
                lease = lease_by_residency[residency]
                if (
                    lease.live_start > producer.end_at
                    or lease.live_end < consumer.end_at
                ):
                    raise ValueError(
                        "PlacementPlan value-flow lease must cover producer to consumer"
                    )
        if any(
            lease_by_residency[residency].live_start > item.start_at
            or lease_by_residency[residency].live_end < item.end_at
            for item in moves
            for residency in (item.move.source, item.move.destination)
        ):
            raise ValueError("PlacementPlan lease does not cover its scheduled move")
        engines = tuple(item.engine for item in copy_engines)
        if len(set(engines)) != len(engines):
            raise ValueError("PlacementPlan copy engines must be unique")
        capacity_by_engine = {
            item.engine: item.capacity for item in copy_engines
        }
        if any(item.engine not in capacity_by_engine for item in moves):
            raise ValueError("PlacementPlan move references an undeclared copy engine")
        for item in moves:
            overlapping = sum(
                other.engine == item.engine
                and other.start_at < item.end_at
                and other.end_at > item.start_at
                for other in moves
            )
            if overlapping > capacity_by_engine[item.engine]:
                raise ValueError("PlacementPlan moves exceed copy-engine capacity")
        object.__setattr__(self, "plan_epoch", epoch)
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "value_flows", value_flows)
        object.__setattr__(self, "residencies", residencies)
        object.__setattr__(self, "moves", moves)
        object.__setattr__(self, "leases", leases)
        object.__setattr__(self, "copy_engines", copy_engines)


def _residency_sort_key(residency: Residency) -> tuple[str, int, str, int]:
    placement = residency.placement
    memory = cast(MemoryKind, residency.memory)
    return (
        residency.value.name,
        residency.value.version,
        f"{placement.kind.value}:{memory.value}",
        -1 if placement.index is None else placement.index,
    )


def _profiled_move_duration(
    move: Move,
    *,
    aggregate: TransferProfile,
    profiles: tuple[TransferProfileDesc, ...],
) -> tuple[DurationProfile, bool]:
    for profile in profiles:
        if (
            profile.kind == move.kind
            and profile.source == move.source.placement
            and profile.destination == move.destination.placement
        ):
            return DurationProfile(
                profile.duration(move.byte_count),
                samples=profile.samples,
            ), False
    return aggregate.for_engine(move.engine)


def compile_placement_plan(
    physical: PhysicalFlowPlan,
    *,
    registry: ImplementationRegistry,
    topology: DeviceTopology,
    value_flows: tuple[ValueFlowDesc, ...] = (),
    transfer_profile: TransferProfile = TransferProfile(),
    transfer_profiles: tuple[TransferProfileDesc, ...] = (),
    copy_engines: tuple[CopyEngineCapacityDesc, ...] = (),
    placement_overrides: Mapping[ExecutorKey, ExecutorKey] | None = None,
) -> PlacementPlan:
    """Compile one deterministic local CPU/GPU placement and transfer plan."""

    physical = _instance(physical, PhysicalFlowPlan, "compile_placement_plan.physical")
    registry = _instance(
        registry, ImplementationRegistry, "compile_placement_plan.registry"
    )
    topology = _instance(topology, DeviceTopology, "compile_placement_plan.topology")
    flows = _tuple_of(value_flows, ValueFlowDesc, "compile_placement_plan.value_flows")
    transfer_profile = _instance(
        transfer_profile, TransferProfile, "compile_placement_plan.transfer_profile"
    )
    route_profiles = _tuple_of(
        transfer_profiles,
        TransferProfileDesc,
        "compile_placement_plan.transfer_profiles",
    )
    profile_keys = tuple(
        (profile.kind, profile.source, profile.destination)
        for profile in route_profiles
    )
    if len(set(profile_keys)) != len(profile_keys):
        raise ValueError("compile_placement_plan transfer profiles must be unique")
    capacities = _tuple_of(
        copy_engines,
        CopyEngineCapacityDesc,
        "compile_placement_plan.copy_engines",
    )
    if not capacities:
        capacities = tuple(CopyEngineCapacityDesc(engine) for engine in CopyEngine)
    if len({item.engine for item in capacities}) != len(capacities):
        raise ValueError("compile_placement_plan copy engines must be unique")

    overrides: dict[ExecutorKey, ExecutorKey] = {}
    if placement_overrides is not None:
        for logical, implementation in placement_overrides.items():
            overrides[_instance(logical, ExecutorKey, "placement override logical key")] = _instance(
                implementation,
                ExecutorKey,
                "placement override implementation key",
            )

    nodes = {node.node_key: node for node in physical.nodes}
    positions = {node.node_key: index for index, node in enumerate(physical.nodes)}
    if any(
        flow.producer not in nodes
        or flow.consumer not in nodes
        or positions[flow.producer] >= positions[flow.consumer]
        for flow in flows
    ):
        raise ValueError("value flow must follow the physical DAG")
    if len({(flow.value, flow.producer, flow.consumer) for flow in flows}) != len(flows):
        raise ValueError("value flows must be unique")
    producer_by_value: dict[ValueKey, NodeKey] = {}
    for flow in flows:
        previous_producer = producer_by_value.setdefault(flow.value, flow.producer)
        if previous_producer != flow.producer:
            raise ValueError(
                "each ValueKey must have exactly one producer; use distinct value versions"
            )
    logical_executors = {node.executor for node in physical.nodes}
    if any(logical not in logical_executors for logical in overrides):
        raise ValueError("placement override references an unknown logical executor")

    candidate_cache: dict[ExecutorKey, tuple[ImplementationDesc, ...]] = {}

    def available_candidates(logical: ExecutorKey) -> tuple[ImplementationDesc, ...]:
        cached = candidate_cache.get(logical)
        if cached is not None:
            return cached
        available = tuple(
            sorted(
                (
                    candidate
                    for candidate in registry.candidates(logical)
                    if topology.contains(candidate.placement)
                ),
                key=lambda candidate: (
                    candidate.key.value,
                    candidate.placement.kind.value,
                    -1
                    if candidate.placement.index is None
                    else candidate.placement.index,
                    candidate.duration.microseconds,
                ),
            )
        )
        candidate_cache[logical] = available
        return available

    dependencies: dict[NodeKey, set[NodeKey]] = {
        node.node_key: {fence.producer for fence in node.fences}
        for node in physical.nodes
    }
    for flow in flows:
        dependencies[flow.consumer].add(flow.producer)

    choices_by_node: dict[NodeKey, tuple[ImplementationDesc, ...]] = {}
    for node in physical.nodes:
        node_choices = available_candidates(node.executor)
        if not node_choices:
            raise ValueError(f"no implementation registered for {node.executor.value}")
        override = overrides.get(node.executor)
        if override is not None:
            node_choices = tuple(
                item for item in node_choices if item.key == override
            )
        if not node_choices:
            raise ValueError("placement override is not registered for its logical executor")
        choices_by_node[node.node_key] = node_choices

    schedule_inputs = tuple(
        ScheduleCandidate(
            key=node.node_key.value,
            priority=0,
            ready_at=0,
            depends_on=tuple(
                dependency.value for dependency in sorted(dependencies[node.node_key])
            ),
            duration=min(
                item.duration.microseconds
                for item in choices_by_node[node.node_key]
            ),
        )
        for node in physical.nodes
    )
    node_by_text = {node.node_key.value: node for node in physical.nodes}
    node_order = tuple(
        node_by_text[item.key] for item in schedule_candidates(schedule_inputs)
    )

    assignment_count = 1
    for node in node_order:
        assignment_count *= len(choices_by_node[node.node_key])
        if assignment_count > _MAX_EXACT_PLACEMENT_ASSIGNMENTS:
            raise PlacementCompilationError(
                "exact placement assignment search exceeds the Phase 3 bound; "
                "reduce implementation candidates or provide placement overrides"
            )

    capacity_by_engine = {item.engine: item.capacity for item in capacities}

    def assignment_score(
        assignment: tuple[ImplementationDesc, ...],
    ) -> tuple[int, int, int, tuple[str, ...]]:
        selected = {
            node.node_key: choice
            for node, choice in zip(node_order, assignment, strict=True)
        }
        engine_lanes = {
            engine: [0] * capacity
            for engine, capacity in capacity_by_engine.items()
        }
        placement_available: dict[Placement, int] = {}
        node_end: dict[NodeKey, int] = {}
        transfer_work = 0
        compute_work = 0
        makespan = 0
        for node in node_order:
            choice = selected[node.node_key]
            dependency_end = max(
                (node_end[item] for item in dependencies[node.node_key]),
                default=0,
            )
            inbound_end = dependency_end
            for flow in sorted(
                (item for item in flows if item.consumer == node.node_key),
                key=lambda item: (item.value.value, item.producer.value),
            ):
                producer = selected[flow.producer]
                route = compile_moves(
                    (
                        TransferRequest(
                            flow.value,
                            Residency(flow.value, producer.placement),
                            Residency(flow.value, choice.placement),
                            flow.byte_count,
                        ),
                    ),
                    topology=topology,
                )
                ready = node_end[flow.producer]
                for move in route:
                    lanes = engine_lanes[move.engine]
                    lane = min(
                        range(len(lanes)), key=lambda index: (lanes[index], index)
                    )
                    begin = max(ready, lanes[lane])
                    duration, _used_fallback = _profiled_move_duration(
                        move,
                        aggregate=transfer_profile,
                        profiles=route_profiles,
                    )
                    ready = begin + duration.microseconds
                    lanes[lane] = ready
                    transfer_work += duration.microseconds
                inbound_end = max(inbound_end, ready)
            start = max(
                dependency_end,
                inbound_end,
                placement_available.get(choice.placement, 0),
            )
            end = start + choice.duration.microseconds
            node_end[node.node_key] = end
            placement_available[choice.placement] = end
            compute_work += choice.duration.microseconds
            makespan = max(makespan, end)
        signature = tuple(choice.key.value for choice in assignment)
        return makespan, transfer_work, compute_work, signature

    assignments = product(
        *(choices_by_node[node.node_key] for node in node_order)
    )
    chosen_assignment = min(assignments, key=assignment_score)
    selections_by_node: dict[NodeKey, ImplementationSelectionDesc] = {}
    for node, chosen in zip(node_order, chosen_assignment, strict=True):
        selections_by_node[node.node_key] = ImplementationSelectionDesc(
            node_key=node.node_key,
            logical_executor=node.executor,
            key=chosen.key,
            placement=chosen.placement,
            duration=chosen.duration,
            oracle_key=chosen.oracle_key,
        )

    engine_lanes: dict[CopyEngine, list[int]] = {
        engine: [0] * capacity for engine, capacity in capacity_by_engine.items()
    }
    placement_available: dict[Placement, int] = {}
    scheduled_nodes: list[PlacedNodeDesc] = []
    scheduled_by_node: dict[NodeKey, PlacedNodeDesc] = {}
    scheduled_moves: list[ScheduledMove] = []
    residencies: set[Residency] = set()

    for node in node_order:
        selected = selections_by_node[node.node_key]
        dependency_end = max(
            (scheduled_by_node[item].end_at for item in dependencies[node.node_key]),
            default=0,
        )
        inbound_end = dependency_end
        for flow in sorted(
            (item for item in flows if item.consumer == node.node_key),
            key=lambda item: (item.value.value, item.producer.value),
        ):
            producer = selections_by_node[flow.producer]
            source = Residency(flow.value, producer.placement)
            destination = Residency(flow.value, selected.placement)
            residencies.update((source, destination))
            route = compile_moves(
                (TransferRequest(flow.value, source, destination, flow.byte_count),),
                topology=topology,
            )
            ready = scheduled_by_node[flow.producer].end_at
            previous_key: MoveKey | None = None
            for move in route:
                lanes = engine_lanes[move.engine]
                lane = min(range(len(lanes)), key=lambda index: (lanes[index], index))
                begin = max(ready, lanes[lane])
                duration, used_fallback = _profiled_move_duration(
                    move,
                    aggregate=transfer_profile,
                    profiles=route_profiles,
                )
                end = begin + duration.microseconds
                key = MoveKey(move.value, len(scheduled_moves))
                scheduled = ScheduledMove(
                    key=key,
                    move=move,
                    engine=move.engine,
                    start_at=begin,
                    end_at=end,
                    producer=flow.producer,
                    consumer=flow.consumer,
                    depends_on=() if previous_key is None else (previous_key,),
                    used_fallback=used_fallback,
                )
                scheduled_moves.append(scheduled)
                residencies.update((move.source, move.destination))
                lanes[lane] = end
                ready = end
                previous_key = key
            inbound_end = max(inbound_end, ready)
        start = max(
            dependency_end,
            inbound_end,
            placement_available.get(selected.placement, 0),
        )
        scheduled = PlacedNodeDesc(
            node_key=node.node_key,
            implementation=selected.key,
            placement=selected.placement,
            start_at=start,
            end_at=start + selected.duration.microseconds,
            depends_on=tuple(sorted(dependencies[node.node_key])),
        )
        scheduled_nodes.append(scheduled)
        scheduled_by_node[node.node_key] = scheduled
        placement_available[selected.placement] = scheduled.end_at

    lease_ranges: dict[Residency, tuple[int, int]] = {}
    for flow in flows:
        producer_node = scheduled_by_node[flow.producer]
        consumer_node = scheduled_by_node[flow.consumer]
        producer = selections_by_node[flow.producer]
        consumer = selections_by_node[flow.consumer]
        source = Residency(flow.value, producer.placement)
        destination = Residency(flow.value, consumer.placement)
        for residency, start, end in (
            (source, producer_node.end_at, consumer_node.end_at),
            (destination, producer_node.end_at, consumer_node.end_at),
        ):
            current = lease_ranges.get(residency)
            lease_ranges[residency] = (
                start if current is None else min(start, current[0]),
                end if current is None else max(end, current[1]),
            )
    for move in scheduled_moves:
        if move.move.destination.memory == MemoryKind.PINNED_HOST:
            lease_ranges[move.move.destination] = (move.start_at, move.end_at)
        elif move.move.source.memory == MemoryKind.PINNED_HOST:
            current = lease_ranges.get(move.move.source, (move.start_at, move.start_at))
            lease_ranges[move.move.source] = (current[0], max(current[1], move.end_at))

    ordered_residencies = tuple(sorted(residencies, key=_residency_sort_key))
    leases = tuple(
        PlacementLeaseDesc(residency, *lease_ranges[residency])
        for residency in ordered_residencies
        if residency in lease_ranges
    )
    result = PlacementPlan(
        plan_epoch=physical.plan_epoch,
        selections=tuple(
            selections_by_node[node.node_key] for node in node_order
        ),
        schedule=tuple(scheduled_nodes),
        topology=topology,
        value_flows=flows,
        residencies=ordered_residencies,
        moves=tuple(scheduled_moves),
        leases=leases,
        copy_engines=capacities,
    )
    try:
        from tllm import core as _core

        _core.validate_placement_plan(result)
    except (TypeError, ValueError) as exc:
        raise PlacementCompilationError(
            "Rust placement-plan descriptor validation failed"
        ) from exc
    return result


@dataclass(frozen=True, slots=True)
class Lease:
    key: str
    live_start: int
    live_end: int
    ready: bool
    retired: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _text(self.key, "Lease.key"))
        start = _integer(self.live_start, "Lease.live_start")
        end = _integer(self.live_end, "Lease.live_end")
        if end < start:
            raise ValueError("Lease.live_end must be >= live_start")
        if not isinstance(cast(object, self.ready), bool) or not isinstance(
            cast(object, self.retired), bool
        ):
            raise TypeError("Lease ready/retired flags must be bool")


@dataclass(frozen=True, slots=True)
class CopyAdmission:
    source: str
    destination: str
    at_step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _text(self.source, "CopyAdmission.source"))
        object.__setattr__(
            self,
            "destination",
            _text(self.destination, "CopyAdmission.destination"),
        )
        object.__setattr__(
            self,
            "at_step",
            _integer(self.at_step, "CopyAdmission.at_step"),
        )


def admit_reuse(lease: Lease, *, live_start: int, live_end: int) -> bool:
    lease = _instance(lease, Lease, "admit_reuse.lease")
    start = _integer(live_start, "admit_reuse.live_start")
    end = _integer(live_end, "admit_reuse.live_end")
    if end < start:
        raise ValueError("reuse live_end must be >= live_start")
    return lease.retired or end < lease.live_start or start > lease.live_end


def admit_copy(copy: CopyAdmission, *, leases: tuple[Lease, ...]) -> bool:
    copy = _instance(copy, CopyAdmission, "admit_copy.copy")
    leases = _tuple_of(leases, Lease, "admit_copy.leases")
    return all(
        lease.key != copy.destination
        or lease.retired
        or not (lease.live_start <= copy.at_step <= lease.live_end)
        for lease in leases
    )


class CopyEngineAdmission:
    """Bounded copy credits released only by retired leases."""

    __slots__ = ("_capacity", "_active")

    def __init__(self, capacity: int = 1) -> None:
        self._capacity = _integer(capacity, "CopyEngineAdmission.capacity", minimum=1)
        self._active: dict[str, Lease] = {}

    def admit(self, copy: CopyAdmission, *, leases: tuple[Lease, ...]) -> bool:
        self.reap(leases)
        if len(self._active) >= self._capacity or not admit_copy(copy, leases=leases):
            return False
        matching = next(
            (lease for lease in leases if lease.key == copy.destination),
            None,
        )
        if matching is None:
            return False
        self._active[matching.key] = matching
        return True

    def reap(self, leases: tuple[Lease, ...]) -> None:
        current = {lease.key: lease for lease in leases}
        self._active = {
            key: lease
            for key, lease in self._active.items()
            if key in current and not current[key].retired
        }


@runtime_checkable
class CopyRuntime(Protocol):
    def copy(self, kind: str, source: str, destination: str) -> None: ...


def execute_moves(moves: Iterable[Move], *, runtime: CopyRuntime) -> None:
    if not isinstance(cast(object, runtime), CopyRuntime):
        raise TypeError("execute_moves runtime must implement copy")
    for move in moves:
        move = _instance(move, Move, "execute_moves.move")
        runtime.copy(
            move.kind.value,
            str(move.source.placement),
            str(move.destination.placement),
        )


__all__ = [
    "CopyAdmission",
    "CopyEngine",
    "CopyEngineAdmission",
    "CopyEngineCapacityDesc",
    "DeviceKind",
    "DeviceTopology",
    "DurationProfile",
    "ImplementationDesc",
    "ImplementationRegistry",
    "ImplementationSelectionDesc",
    "ImplementationSpec",
    "Lease",
    "MemoryKind",
    "Move",
    "MoveKey",
    "MoveKind",
    "PlacedNodeDesc",
    "Placement",
    "PlacementCompilationError",
    "PlacementLeaseDesc",
    "PlacementPlan",
    "Residency",
    "ScheduleCandidate",
    "ScheduledMove",
    "SemanticOracle",
    "SemanticOracleKey",
    "SemanticOracleMismatch",
    "SemanticOracleSpec",
    "SemanticOracleVerificationDesc",
    "TransferProfile",
    "TransferProfileDesc",
    "TransferRequest",
    "ValueFlowDesc",
    "ValueKey",
    "admit_copy",
    "admit_reuse",
    "compile_moves",
    "compile_placement_plan",
    "execute_moves",
    "schedule_candidates",
    "schedule_moves",
]

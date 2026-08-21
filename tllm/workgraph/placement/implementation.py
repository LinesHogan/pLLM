#!/usr/bin/env python3
"""Implementation registry, profiles, and move scheduling for WorkGraph placement."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.placement.transfer import Move
from tllm.workgraph.placement.vocab import (
    CopyEngine,
    MemoryKind,
    MoveKind,
    Placement,
    Residency,
    ValueKey,
    require_enum,
    require_executor_key,
    require_instance,
    require_int,
    require_node_key,
    require_str,
    require_tuple_of,
)


@runtime_checkable
class RuntimeImplementation(Protocol):
    def execute(self, *args: object) -> object: ...
@dataclass(frozen=True, slots=True, order=True)
class SemanticOracleKey:
    namespace: str
    name: str
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", require_str(self.namespace, "SemanticOracleKey.namespace"))
        object.__setattr__(self, "name", require_str(self.name, "SemanticOracleKey.name"))
        object.__setattr__(self, "version", require_str(self.version, "SemanticOracleKey.version"))

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
            require_instance(self.key, SemanticOracleKey, "SemanticOracleSpec.key"),
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
            require_instance(
                self.logical_executor,
                ExecutorKey,
                "SemanticOracleVerificationDesc.logical_executor",
            ),
        )
        object.__setattr__(
            self,
            "oracle_key",
            require_instance(
                self.oracle_key,
                SemanticOracleKey,
                "SemanticOracleVerificationDesc.oracle_key",
            ),
        )
        implementations = require_tuple_of(
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
            require_int(
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
            require_int(self.microseconds, "DurationProfile.microseconds", minimum=1),
        )
        object.__setattr__(
            self,
            "samples",
            require_int(self.samples, "DurationProfile.samples", minimum=1),
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
                    require_instance(value, DurationProfile, f"TransferProfile.{name}"),
                )
        object.__setattr__(
            self,
            "fallback",
            require_instance(self.fallback, DurationProfile, "TransferProfile.fallback"),
        )

    def for_engine(self, engine: CopyEngine) -> tuple[DurationProfile, bool]:
        engine = require_enum(cast(object, engine), CopyEngine, "TransferProfile.engine")
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
        object.__setattr__(self, "value", require_instance(self.value, ValueKey, "MoveKey.value"))
        object.__setattr__(
            self,
            "ordinal",
            require_int(self.ordinal, "MoveKey.ordinal"),
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
        key = require_instance(self.key, MoveKey, "ScheduledMove.key")
        move = require_instance(self.move, Move, "ScheduledMove.move")
        engine = require_enum(cast(object, self.engine), CopyEngine, "ScheduledMove.engine")
        start = require_int(self.start_at, "ScheduledMove.start_at")
        end = require_int(self.end_at, "ScheduledMove.end_at", minimum=1)
        dependencies = require_tuple_of(
            self.depends_on, MoveKey, "ScheduledMove.depends_on"
        )
        producer = self.producer
        consumer = self.consumer
        if (producer is None) != (consumer is None):
            raise ValueError("ScheduledMove producer and consumer must be declared together")
        if producer is not None and consumer is not None:
            producer = require_node_key(producer, "ScheduledMove.producer")
            consumer = require_node_key(consumer, "ScheduledMove.consumer")
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

    moves = require_tuple_of(moves, Move, "schedule_moves.moves")
    profile = require_instance(profile, TransferProfile, "schedule_moves.profile")
    start = require_int(start_at, "schedule_moves.start_at")
    ordinal = require_int(ordinal_start, "schedule_moves.ordinal_start")
    capacities = require_tuple_of(
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
        kind = require_enum(cast(object, self.kind), MoveKind, "TransferProfileDesc.kind")
        source = require_instance(self.source, Placement, "TransferProfileDesc.source")
        destination = require_instance(
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
            require_int(
                self.bytes_per_microsecond,
                "TransferProfileDesc.bytes_per_microsecond",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "fixed_latency_microseconds",
            require_int(
                self.fixed_latency_microseconds,
                "TransferProfileDesc.fixed_latency_microseconds",
            ),
        )
        object.__setattr__(
            self,
            "samples",
            require_int(self.samples, "TransferProfileDesc.samples", minimum=1),
        )

    def duration(self, byte_count: int) -> int:
        size = require_int(byte_count, "TransferProfileDesc.byte_count")
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
            require_enum(cast(object, self.engine), CopyEngine, "CopyEngineCapacityDesc.engine"),
        )
        object.__setattr__(
            self,
            "capacity",
            require_int(self.capacity, "CopyEngineCapacityDesc.capacity", minimum=1),
        )
@dataclass(frozen=True, slots=True)
class ValueFlowDesc:
    value: ValueKey
    producer: NodeKey
    consumer: NodeKey
    byte_count: int

    def __post_init__(self) -> None:
        value = require_instance(self.value, ValueKey, "ValueFlowDesc.value")
        producer = require_node_key(self.producer, "ValueFlowDesc.producer")
        consumer = require_node_key(self.consumer, "ValueFlowDesc.consumer")
        if producer == consumer:
            raise ValueError("ValueFlowDesc producer and consumer must differ")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "consumer", consumer)
        object.__setattr__(
            self,
            "byte_count",
            require_int(self.byte_count, "ValueFlowDesc.byte_count", minimum=1),
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
        key = require_executor_key(self.key, "ImplementationSpec.key")
        placement = require_instance(
            self.placement, Placement, "ImplementationSpec.placement"
        )
        implementation = cast(object, self.implementation)
        if not isinstance(implementation, RuntimeImplementation):
            raise TypeError("ImplementationSpec.implementation must implement execute")
        logical = self.logical_executor
        if logical is None:
            logical = key
        else:
            logical = require_executor_key(logical, "ImplementationSpec.logical_executor")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "logical_executor", logical)
        object.__setattr__(
            self,
            "duration",
            require_instance(self.duration, DurationProfile, "ImplementationSpec.duration"),
        )
        if self.oracle_key is not None:
            object.__setattr__(
                self,
                "oracle_key",
                require_instance(
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
        object.__setattr__(self, "key", require_executor_key(self.key, "ImplementationDesc.key"))
        object.__setattr__(
            self,
            "logical_executor",
            require_executor_key(
                self.logical_executor,
                "ImplementationDesc.logical_executor",
            ),
        )
        object.__setattr__(
            self,
            "placement",
            require_instance(self.placement, Placement, "ImplementationDesc.placement"),
        )
        object.__setattr__(
            self,
            "duration",
            require_instance(self.duration, DurationProfile, "ImplementationDesc.duration"),
        )
        if self.oracle_key is not None:
            object.__setattr__(
                self,
                "oracle_key",
                require_instance(
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
        spec = require_instance(spec, SemanticOracleSpec, "ImplementationRegistry.oracle")
        if spec.key in self._oracles:
            raise ValueError(f"duplicate semantic oracle key {spec.key.value}")
        self._oracles[spec.key] = spec.oracle

    def register(self, spec: ImplementationSpec) -> None:
        spec = require_instance(spec, ImplementationSpec, "ImplementationRegistry.spec")
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
        key = require_executor_key(key, "ImplementationRegistry.key")
        return self._specs[key].implementation

    def candidates(self, logical_executor: ExecutorKey) -> tuple[ImplementationDesc, ...]:
        logical_executor = require_executor_key(
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
        logical_executor = require_executor_key(
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
        object.__setattr__(self, "key", require_str(self.key, "ScheduleCandidate.key"))
        if not isinstance(cast(object, self.priority), int) or isinstance(
            cast(object, self.priority), bool
        ):
            raise TypeError("ScheduleCandidate.priority must be int")
        object.__setattr__(
            self,
            "ready_at",
            require_int(self.ready_at, "ScheduleCandidate.ready_at"),
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
            require_int(self.duration, "ScheduleCandidate.duration", minimum=1),
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
            require_node_key(self.node_key, "ImplementationSelectionDesc.node_key"),
        )
        object.__setattr__(
            self,
            "logical_executor",
            require_executor_key(
                self.logical_executor,
                "ImplementationSelectionDesc.logical_executor",
            ),
        )
        if self.oracle_key is not None:
            object.__setattr__(
                self,
                "oracle_key",
                require_instance(
                    self.oracle_key,
                    SemanticOracleKey,
                    "ImplementationSelectionDesc.oracle_key",
                ),
            )
        object.__setattr__(
            self,
            "key",
            require_executor_key(self.key, "ImplementationSelectionDesc.key"),
        )
        object.__setattr__(
            self,
            "placement",
            require_instance(
                self.placement, Placement, "ImplementationSelectionDesc.placement"
            ),
        )
        object.__setattr__(
            self,
            "duration",
            require_instance(
                self.duration,
                DurationProfile,
                "ImplementationSelectionDesc.duration",
            ),
        )

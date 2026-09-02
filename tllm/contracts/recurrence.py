#!/usr/bin/env python3
"""Finite recurrence contracts and semantics-preserving schedule synthesis.

Consumers declare dependence, not execution tricks.  Exact schedules preserve
every edge.  Approximation transforms (currently Jacobi) are explicit and
produce a new graph with snapshot-version reads.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from tllm.backends.protocol import ModelRegionCapabilities


class RecurrencePass(StrEnum):
    NORMAL = "normal"
    RECURRENT = "recurrent"
    MIX = "mix"


class DependenceKind(StrEnum):
    DATA = "data"
    CONTROL = "control"
    CARRIED = "carried"
    KV_WRITE_READ = "kv_write_read"
    KV_OVERWRITE_READ = "kv_overwrite_read"
    SNAPSHOT = "snapshot"


class VersionVisibility(StrEnum):
    CURRENT = "current"
    PREVIOUS_ROUND = "previous_round"


@dataclass(frozen=True, slots=True, order=True)
class IterationPoint:
    token: int
    layer: int
    pass_name: str
    round: int = 0

    def __post_init__(self) -> None:
        for name in ("token", "layer", "round"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"IterationPoint.{name} must be int")
            if name != "layer" and value < 0:
                raise ValueError(f"IterationPoint.{name} must be non-negative")
        if not isinstance(self.pass_name, str) or not self.pass_name:  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise ValueError("IterationPoint.pass_name must be non-empty")


@dataclass(frozen=True, slots=True)
class RecurrenceOp:
    op_id: str
    executor: str
    point: IterationPoint

    def __post_init__(self) -> None:
        if not isinstance(self.op_id, str) or not self.op_id:  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise ValueError("RecurrenceOp.op_id must be non-empty")
        if not isinstance(self.executor, str) or not self.executor:  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise ValueError("RecurrenceOp.executor must be non-empty")
        if not isinstance(self.point, IterationPoint):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise TypeError("RecurrenceOp.point must be IterationPoint")


@dataclass(frozen=True, slots=True)
class RecurrenceEdge:
    source: str
    target: str
    kind: DependenceKind
    distance: tuple[int, ...] = ()
    visibility: VersionVisibility = VersionVisibility.CURRENT

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("RecurrenceEdge endpoints must be non-empty")
        object.__setattr__(self, "kind", DependenceKind(self.kind))
        object.__setattr__(self, "visibility", VersionVisibility(self.visibility))
        distance = tuple(self.distance)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in distance):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise TypeError("RecurrenceEdge.distance must contain ints")
        if self.kind is DependenceKind.CARRIED:
            if not distance or not any(value > 0 for value in distance) or any(value < 0 for value in distance):
                raise ValueError("carried dependence requires a non-negative, positive distance")
        object.__setattr__(self, "distance", distance)


@dataclass(frozen=True, slots=True)
class RecurrenceGraph:
    ops: tuple[RecurrenceOp, ...]
    edges: tuple[RecurrenceEdge, ...]

    def __post_init__(self) -> None:
        ops = tuple(self.ops)
        edges = tuple(self.edges)
        ids = tuple(op.op_id for op in ops)
        if len(ids) != len(set(ids)):
            raise ValueError("recurrence graph contains duplicate op ids")
        known = set(ids)
        for edge in edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"recurrence edge references unknown op: {edge.source}->{edge.target}")
            if edge.source == edge.target:
                raise ValueError("recurrence graph contains a self edge")
        object.__setattr__(self, "ops", ops)
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True, slots=True)
class OrderedRowFusionCapability:
    internalized_edges: frozenset[DependenceKind]
    max_rows: int = 2

    def __post_init__(self) -> None:
        normalized = frozenset(DependenceKind(item) for item in self.internalized_edges)
        if self.max_rows < 2:
            raise ValueError("ordered-row fusion requires max_rows >= 2")
        object.__setattr__(self, "internalized_edges", normalized)


@dataclass(frozen=True, slots=True)
class RecurrenceSchedule:
    groups: tuple[tuple[str, ...], ...]
    kind: str
    fusion_rejections: tuple[tuple[str, str, str], ...] = ()

    @property
    def execution_order(self) -> tuple[str, ...]:
        return tuple(op_id for group in self.groups for op_id in group)


@dataclass(frozen=True, slots=True)
class RecurrenceRequirements:
    """Backend-independent effects declared by one recurrent computation."""

    phase: str = "prefill"
    effects: frozenset[DependenceKind] = frozenset()
    jacobi: "JacobiPolicy | None" = None
    jacobi_carried_kinds: frozenset[DependenceKind] = frozenset(
        {DependenceKind.CARRIED}
    )
    prefer_ordered_row_fusion: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase:  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise ValueError("recurrence phase must be non-empty")
        effects = frozenset(DependenceKind(kind) for kind in self.effects)
        carried = frozenset(
            DependenceKind(kind) for kind in self.jacobi_carried_kinds
        )
        if self.jacobi is not None and not isinstance(self.jacobi, JacobiPolicy):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise TypeError("recurrence jacobi policy must be JacobiPolicy or None")
        if self.jacobi is not None and not carried:
            raise ValueError("Jacobi recurrence requires snapshot-carried effects")
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "jacobi_carried_kinds", carried)


@dataclass(frozen=True, slots=True)
class CompiledRecurrence:
    graph: RecurrenceGraph
    schedule: RecurrenceSchedule
    phase: str
    approximation: str = "exact"


class RecurrenceAdmissionError(RuntimeError):
    """Raised before model execution when a recurrence cannot be materialized."""


def _graph_mode_name(graph_mode: object) -> str:
    if isinstance(graph_mode, Enum):
        return str(graph_mode.name).lower()
    if isinstance(graph_mode, str):
        return graph_mode.lower()
    raise TypeError("graph_mode must be an enum or string")


def validate_recurrence_admission(
    requirements: RecurrenceRequirements,
    capabilities: "ModelRegionCapabilities | None",
    *,
    graph_mode: object,
    tp_size: int,
) -> None:
    """Validate a recurrence against typed backend facts, without a runner."""

    if not isinstance(requirements, RecurrenceRequirements):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
        raise TypeError("requirements must be RecurrenceRequirements")
    if capabilities is None:
        raise RecurrenceAdmissionError(
            "active recurrence requires backend model-region support"
        )
    if requirements.phase not in capabilities.phases:
        raise RecurrenceAdmissionError(
            f"backend model-region does not support phase {requirements.phase!r}"
        )
    if not capabilities.replay:
        raise RecurrenceAdmissionError(
            "active recurrence requires model-region replay"
        )
    if (
        DependenceKind.KV_OVERWRITE_READ in requirements.effects
        and not capabilities.kv_overwrite
    ):
        raise RecurrenceAdmissionError(
            "active recurrence requires KV overwrite-before-read support"
        )
    if requirements.jacobi is not None and not capabilities.kv_snapshot:
        raise RecurrenceAdmissionError(
            "Jacobi recurrence requires backend KV snapshot support"
        )
    mode = _graph_mode_name(graph_mode)
    if capabilities.graph_safety.name == "EAGER_ONLY" and mode != "eager":
        raise RecurrenceAdmissionError(
            f"backend model-region is eager-only, got graph mode {mode!r}"
        )
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
        raise TypeError("tp_size must be a positive int")
    if tp_size not in capabilities.tensor_parallel_sizes:
        raise RecurrenceAdmissionError(
            f"backend model-region does not support TP={tp_size}"
        )


def compile_recurrence(
    graph: RecurrenceGraph,
    requirements: RecurrenceRequirements,
    capabilities: "ModelRegionCapabilities | None",
    *,
    graph_mode: object,
    tp_size: int,
) -> CompiledRecurrence:
    """Compile one finite recurrence with Rust as the schedule authority."""

    if not isinstance(graph, RecurrenceGraph):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
        raise TypeError("graph must be RecurrenceGraph")
    validate_recurrence_admission(
        requirements,
        capabilities,
        graph_mode=graph_mode,
        tp_size=tp_size,
    )
    actual_effects = frozenset(edge.kind for edge in graph.edges)
    undeclared = actual_effects - requirements.effects
    if undeclared:
        names = ", ".join(sorted(kind.value for kind in undeclared))
        raise RecurrenceAdmissionError(
            f"recurrence graph contains undeclared dependence effects: {names}"
        )

    approximation = "exact"
    compiled_graph = graph
    if requirements.jacobi is not None:
        compiled_graph = jacobi_transform(
            graph,
            rounds=requirements.jacobi.max_rounds,
            carried_kinds=requirements.jacobi_carried_kinds,
        )
        approximation = "jacobi"

    assert capabilities is not None
    fusion = (
        capabilities.ordered_row_fusion
        if requirements.prefer_ordered_row_fusion
        else None
    )
    internalized = () if fusion is None else tuple(fusion.internalized_edges)
    max_rows = 2 if fusion is None else int(fusion.max_rows)
    from tllm import _core

    try:
        rust = _core.compile_recurrence_schedule(
            compiled_graph.ops,
            compiled_graph.edges,
            internalized,
            max_rows,
        )
    except (RuntimeError, ValueError) as exc:
        raise RecurrenceAdmissionError(str(exc)) from exc
    schedule = RecurrenceSchedule(
        groups=tuple(tuple(str(op_id) for op_id in group) for group in rust.groups),
        kind=str(rust.kind),
        fusion_rejections=tuple(
            (str(source), str(target), str(reason))
            for source, target, reason in rust.fusion_rejections
        ),
    )
    return CompiledRecurrence(
        graph=compiled_graph,
        schedule=schedule,
        phase=requirements.phase,
        approximation=approximation,
    )


def _stable_topological_order(graph: RecurrenceGraph) -> tuple[str, ...]:
    ordinal: dict[str, int] = {
        op.op_id: index for index, op in enumerate(graph.ops)
    }
    indegree: dict[str, int] = {op.op_id: 0 for op in graph.ops}
    successors: dict[str, list[str]] = {op.op_id: [] for op in graph.ops}
    for edge in graph.edges:
        indegree[edge.target] += 1
        successors[edge.source].append(edge.target)
    ready: list[str] = sorted(
        (op_id for op_id, degree in indegree.items() if degree == 0),
        key=lambda op_id: ordinal[op_id],
    )
    order: list[str] = []
    while ready:
        op_id = ready.pop(0)
        order.append(op_id)
        for target in successors[op_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
        ready.sort(key=lambda candidate: ordinal[candidate])
    if len(order) != len(graph.ops):
        raise ValueError("recurrence instance graph contains a cycle")
    return tuple(order)


def compile_serial_schedule(graph: RecurrenceGraph) -> RecurrenceSchedule:
    return RecurrenceSchedule(
        groups=tuple((op_id,) for op_id in _stable_topological_order(graph)),
        kind="serial",
    )


def _contracted_schedule(
    graph: RecurrenceGraph,
    pairs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, ...], ...]:
    group_of: dict[str, int] = {}
    groups: list[tuple[str, ...]] = []
    paired = {op_id for pair in pairs for op_id in pair}
    for pair in pairs:
        index = len(groups)
        groups.append(pair)
        for op_id in pair:
            group_of[op_id] = index
    for op in graph.ops:
        if op.op_id in paired:
            continue
        group_of[op.op_id] = len(groups)
        groups.append((op.op_id,))

    indegree = [0] * len(groups)
    successors: list[set[int]] = [set() for _ in groups]
    for edge in graph.edges:
        source = group_of[edge.source]
        target = group_of[edge.target]
        if source == target:
            continue
        if target not in successors[source]:
            successors[source].add(target)
            indegree[target] += 1
    ordinal = {op.op_id: index for index, op in enumerate(graph.ops)}
    group_ordinal = [min(ordinal[op_id] for op_id in group) for group in groups]
    ready = sorted((index for index, degree in enumerate(indegree) if degree == 0), key=group_ordinal.__getitem__)
    result: list[tuple[str, ...]] = []
    while ready:
        index = ready.pop(0)
        result.append(groups[index])
        for target in successors[index]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
        ready.sort(key=group_ordinal.__getitem__)
    if len(result) != len(groups):
        raise ValueError("ordered-row fusion would create a cyclic contracted graph")
    return tuple(result)


def compile_wavefront_schedule(
    graph: RecurrenceGraph,
    capability: OrderedRowFusionCapability,
) -> RecurrenceSchedule:
    """Fuse dependence-connected rows when the backend internalizes the edge.

    This deliberately contains no Recirculation identifiers.  A pair is legal
    only when it shares an executor, the source row precedes the target row,
    and the backend declares the dependence kind as an in-kernel ordered edge.
    """

    by_id = {op.op_id: op for op in graph.ops}
    used: set[str] = set()
    pairs: list[tuple[str, str]] = []
    rejections: list[tuple[str, str, str]] = []
    for edge in graph.edges:
        if edge.kind not in capability.internalized_edges:
            continue
        source = by_id[edge.source]
        target = by_id[edge.target]
        if source.executor != target.executor:
            rejections.append((edge.source, edge.target, "executor_mismatch"))
            continue
        if source.op_id in used or target.op_id in used:
            rejections.append((edge.source, edge.target, "row_already_fused"))
            continue
        if source.point.token >= target.point.token:
            rejections.append((edge.source, edge.target, "noncausal_row_order"))
            continue
        if source.point.layer != target.point.layer:
            rejections.append((edge.source, edge.target, "layer_mismatch"))
            continue
        pairs.append((source.op_id, target.op_id))
        used.update((source.op_id, target.op_id))
    groups = _contracted_schedule(graph, tuple(pairs))
    return RecurrenceSchedule(
        groups=groups,
        kind="wavefront" if pairs else "serial",
        fusion_rejections=tuple(rejections),
    )


@dataclass(frozen=True, slots=True)
class JacobiPolicy:
    max_rounds: int = 3
    tolerance: float = 1e-3
    fail_if_not_converged: bool = False

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("Jacobi max_rounds must be positive")
        if self.tolerance < 0.0:
            raise ValueError("Jacobi tolerance must be non-negative")


def jacobi_transform(
    graph: RecurrenceGraph,
    *,
    rounds: int,
    carried_kinds: frozenset[DependenceKind] = frozenset({DependenceKind.CARRIED}),
) -> RecurrenceGraph:
    """Expand a recurrence into explicit rounds with previous-round reads."""

    if rounds < 1:
        raise ValueError("Jacobi rounds must be positive")
    normalized_kinds = frozenset(DependenceKind(kind) for kind in carried_kinds)
    if not normalized_kinds:
        raise ValueError("Jacobi transform requires at least one carried dependence kind")
    ops: list[RecurrenceOp] = []
    edges: list[RecurrenceEdge] = []
    for round_idx in range(rounds):
        for op in graph.ops:
            ops.append(
                replace(
                    op,
                    op_id=f"r{round_idx}/{op.op_id}",
                    point=replace(op.point, round=round_idx),
                )
            )
        for edge in graph.edges:
            if edge.kind in normalized_kinds:
                if round_idx == 0:
                    continue
                edges.append(
                    RecurrenceEdge(
                        source=f"r{round_idx - 1}/{edge.source}",
                        target=f"r{round_idx}/{edge.target}",
                        kind=DependenceKind.SNAPSHOT,
                        distance=(1,),
                        visibility=VersionVisibility.PREVIOUS_ROUND,
                    )
                )
            else:
                edges.append(
                    replace(
                        edge,
                        source=f"r{round_idx}/{edge.source}",
                        target=f"r{round_idx}/{edge.target}",
                    )
                )
    return RecurrenceGraph(tuple(ops), tuple(edges))


def graph_from_iterables(
    ops: Iterable[RecurrenceOp], edges: Iterable[RecurrenceEdge]
) -> RecurrenceGraph:
    return RecurrenceGraph(tuple(ops), tuple(edges))


__all__ = [
    "CompiledRecurrence",
    "DependenceKind",
    "IterationPoint",
    "JacobiPolicy",
    "OrderedRowFusionCapability",
    "RecurrenceAdmissionError",
    "RecurrenceEdge",
    "RecurrenceGraph",
    "RecurrenceOp",
    "RecurrencePass",
    "RecurrenceRequirements",
    "RecurrenceSchedule",
    "VersionVisibility",
    "compile_recurrence",
    "compile_serial_schedule",
    "compile_wavefront_schedule",
    "graph_from_iterables",
    "jacobi_transform",
    "validate_recurrence_admission",
]

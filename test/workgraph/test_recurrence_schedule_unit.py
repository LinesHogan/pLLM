#!/usr/bin/env python3
"""Recurrence scheduling is algorithm-independent and Rust/Python equivalent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import pytest

from tllm import _core
from tllm.consumers.recirculation.graph import build_recirculation_graph
from tllm.consumers.recirculation.workload import RecirculationConfig
from tllm.workgraph.recurrence import (
    DependenceKind,
    IterationPoint,
    JacobiPolicy,
    OrderedRowFusionCapability,
    RecurrenceEdge,
    RecurrenceGraph,
    RecurrenceOp,
    VersionVisibility,
    compile_serial_schedule,
    compile_wavefront_schedule,
    jacobi_transform,
)


StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class _JacobiResult(Generic[StateT]):
    state: StateT
    rounds: int
    residual: float
    converged: bool


def _run_jacobi(
    initial: StateT,
    *,
    step: Callable[[StateT], StateT],
    residual: Callable[[StateT, StateT], float],
    policy: JacobiPolicy,
) -> _JacobiResult[StateT]:
    current = initial
    last_residual = float("inf")
    for round_idx in range(1, policy.max_rounds + 1):
        candidate = step(current)
        last_residual = float(residual(current, candidate))
        if last_residual < 0.0:
            raise ValueError("Jacobi residual must be non-negative")
        current = candidate
        if last_residual <= policy.tolerance:
            return _JacobiResult(current, round_idx, last_residual, True)
    if policy.fail_if_not_converged:
        raise RuntimeError(
            f"Jacobi iteration did not converge after {policy.max_rounds} rounds; "
            f"residual={last_residual} tolerance={policy.tolerance}"
        )
    return _JacobiResult(current, policy.max_rounds, last_residual, False)


def _capability() -> OrderedRowFusionCapability:
    return OrderedRowFusionCapability(
        frozenset({DependenceKind.KV_OVERWRITE_READ}), max_rows=2
    )


def _rust(graph: RecurrenceGraph):
    return _core.compile_recurrence_schedule(
        graph.ops,
        graph.edges,
        [DependenceKind.KV_OVERWRITE_READ.value],
        2,
    )


def test_recirculation_wavefront_is_derived_from_effect_edges() -> None:
    graph = build_recirculation_graph(
        RecirculationConfig(source_layer=2, destination_layer=0),
        token_count=3,
        layer_count=4,
    )
    python = compile_wavefront_schedule(graph, _capability())
    rust = _rust(graph)

    assert python.kind == rust.kind == "wavefront"
    assert python.groups == tuple(tuple(group) for group in rust.groups)
    assert python.execution_order == tuple(rust.execution_order)
    assert (
        "t0/recurrent/l1",
        "t1/normal/l1",
    ) in python.groups
    assert (
        "t1/recurrent/l3",
        "t2/normal/l3",
    ) in python.groups
    assert all(len(group) <= 2 for group in python.groups)


def test_second_temporal_algorithm_uses_the_same_fusion_rule() -> None:
    # No Recirculation names or mix node: a generic recurrent state update
    # overwrites the state consumed by the next token's same executor.
    graph = RecurrenceGraph(
        ops=(
            RecurrenceOp("state/0", "temporal_cell", IterationPoint(0, 0, "update")),
            RecurrenceOp("read/1", "temporal_cell", IterationPoint(1, 0, "read")),
        ),
        edges=(
            RecurrenceEdge(
                "state/0",
                "read/1",
                DependenceKind.KV_OVERWRITE_READ,
                distance=(1,),
            ),
        ),
    )
    assert compile_wavefront_schedule(graph, _capability()).groups == (
        ("state/0", "read/1"),
    )
    assert tuple(tuple(group) for group in _rust(graph).groups) == (
        ("state/0", "read/1"),
    )


def test_serial_schedule_preserves_every_dependency() -> None:
    graph = build_recirculation_graph(
        RecirculationConfig(source_layer=2, destination_layer=0),
        token_count=2,
        layer_count=3,
    )
    schedule = compile_serial_schedule(graph)
    position = {op_id: index for index, op_id in enumerate(schedule.execution_order)}
    assert all(position[edge.source] < position[edge.target] for edge in graph.edges)


def test_jacobi_is_an_explicit_snapshot_transform() -> None:
    graph = RecurrenceGraph(
        ops=(
            RecurrenceOp("produce", "cell", IterationPoint(0, 0, "produce")),
            RecurrenceOp("consume", "cell", IterationPoint(1, 0, "consume")),
        ),
        edges=(
            RecurrenceEdge(
                "produce", "consume", DependenceKind.CARRIED, distance=(1,)
            ),
        ),
    )
    transformed = jacobi_transform(graph, rounds=3)
    snapshot_edges = [edge for edge in transformed.edges if edge.kind is DependenceKind.SNAPSHOT]
    assert len(snapshot_edges) == 2
    assert all(edge.visibility is VersionVisibility.PREVIOUS_ROUND for edge in snapshot_edges)
    assert snapshot_edges[0].source == "r0/produce"
    assert snapshot_edges[0].target == "r1/consume"


def test_jacobi_runtime_reports_nonconvergence_and_strict_failure() -> None:
    result = _run_jacobi(
        0.0,
        step=lambda value: value + 1.0,
        residual=lambda old, new: abs(new - old),
        policy=JacobiPolicy(max_rounds=3, tolerance=0.1),
    )
    assert result.state == 3.0
    assert result.rounds == 3
    assert not result.converged
    with pytest.raises(RuntimeError, match="did not converge"):
        _run_jacobi(
            0.0,
            step=lambda value: value + 1.0,
            residual=lambda old, new: abs(new - old),
            policy=JacobiPolicy(max_rounds=2, tolerance=0.1, fail_if_not_converged=True),
        )


def test_invalid_carried_distance_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive distance"):
        RecurrenceEdge("a", "b", DependenceKind.CARRIED, distance=(0,))

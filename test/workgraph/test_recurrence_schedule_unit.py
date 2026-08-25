#!/usr/bin/env python3
"""Recurrence scheduling is algorithm-independent and Rust/Python equivalent."""

from __future__ import annotations

import pytest

from tllm import _core
from tllm.backends.protocol import (
    BackendCapabilities,
    GraphSafety,
    ModelRegionCapabilities,
    OrderedRowFusionCapabilities,
)
from tllm.consumers.recirculation import RecirculationConfig, RecirculationConsumer
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.graph_policy import GraphMode
from tllm.workgraph.jacobi import run_jacobi
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


def _backend(
    *,
    ordered: bool = True,
    snapshot: bool = True,
    tp_sizes: frozenset[int] = frozenset({1}),
) -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="test-model-region",
        runner_family="mrv1",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        model_region=ModelRegionCapabilities(
            phases=frozenset({"prefill"}),
            replay=True,
            kv_snapshot=snapshot,
            kv_overwrite=True,
            ordered_row_fusion=(
                OrderedRowFusionCapabilities(
                    internalized_edges=frozenset(
                        {DependenceKind.KV_OVERWRITE_READ.value}
                    ),
                    max_rows=2,
                )
                if ordered
                else None
            ),
            graph_safety=GraphSafety.EAGER_ONLY,
            tensor_parallel_sizes=tp_sizes,
        ),
    )


def test_recirculation_wavefront_is_derived_from_effect_edges() -> None:
    graph = RecirculationConsumer(
        RecirculationConfig(source_layer=2, destination_layer=0)
    ).recurrence_graph(token_count=3, layer_count=4)
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
    graph = RecirculationConsumer(
        RecirculationConfig(source_layer=2, destination_layer=0)
    ).recurrence_graph(token_count=2, layer_count=3)
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
    result = run_jacobi(
        0.0,
        step=lambda value: value + 1.0,
        residual=lambda old, new: abs(new - old),
        policy=JacobiPolicy(max_rounds=3, tolerance=0.1),
    )
    assert result.state == 3.0
    assert result.rounds == 3
    assert not result.converged
    with pytest.raises(RuntimeError, match="did not converge"):
        run_jacobi(
            0.0,
            step=lambda value: value + 1.0,
            residual=lambda old, new: abs(new - old),
            policy=JacobiPolicy(max_rounds=2, tolerance=0.1, fail_if_not_converged=True),
        )


def test_invalid_carried_distance_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive distance"):
        RecurrenceEdge("a", "b", DependenceKind.CARRIED, distance=(0,))


def test_dispatch_compiler_selects_wavefront_only_from_backend_capability() -> None:
    consumer = RecirculationConsumer(
        RecirculationConfig(source_layer=2, destination_layer=0)
    )
    plan = DispatchPlan.build([consumer])

    assert not plan.is_empty()
    assert len(plan.recurrence_targets()) == 1
    compiled = plan.compile_recurrences(
        token_count=3,
        layer_count=4,
        backend_capabilities=_backend(ordered=True),
        graph_mode=GraphMode.EAGER,
    )[0][1]
    assert compiled.approximation == "exact"
    assert compiled.schedule.kind == "wavefront"

    serial = plan.compile_recurrences(
        token_count=3,
        layer_count=4,
        backend_capabilities=_backend(ordered=False),
        graph_mode=GraphMode.EAGER,
    )[0][1]
    assert serial.schedule.kind == "serial"


def test_jacobi_is_compiled_as_explicit_snapshot_approximation() -> None:
    plan = DispatchPlan.build(
        [
            RecirculationConsumer(
                RecirculationConfig(
                    source_layer=2,
                    destination_layer=0,
                    jacobi=JacobiPolicy(max_rounds=2),
                )
            )
        ]
    )
    compiled = plan.compile_recurrences(
        token_count=2,
        layer_count=3,
        backend_capabilities=_backend(ordered=False),
        graph_mode=GraphMode.EAGER,
    )[0][1]

    assert compiled.approximation == "jacobi"
    assert any(
        edge.kind is DependenceKind.SNAPSHOT
        and edge.visibility is VersionVisibility.PREVIOUS_ROUND
        for edge in compiled.graph.edges
    )


@pytest.mark.parametrize(
    ("backend", "mode", "tp_size", "message"),
    [
        (
            BackendCapabilities(
                backend_id="no-region",
                runner_family="mrv1",
                sampler_stages=frozenset(),
                native_min_p=False,
                tuple_sampler_return=False,
                split_execute_and_sample=True,
            ),
            GraphMode.EAGER,
            1,
            "model-region",
        ),
        (_backend(), GraphMode.PIECEWISE, 1, "eager-only"),
        (_backend(), GraphMode.EAGER, 2, "TP=2"),
    ],
)
def test_recurrence_backend_admission_fails_closed(
    backend: BackendCapabilities,
    mode: GraphMode,
    tp_size: int,
    message: str,
) -> None:
    plan = DispatchPlan.build(
        [RecirculationConsumer(RecirculationConfig(source_layer=2, destination_layer=0))]
    )
    with pytest.raises(RuntimeError, match=message):
        plan.validate_recurrence_admission(
            backend_capabilities=backend,
            graph_mode=mode,
            tp_size=tp_size,
        )

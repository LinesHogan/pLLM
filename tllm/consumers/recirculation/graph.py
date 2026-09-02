"""Exact Recirculation dependence graph construction."""

from __future__ import annotations

from tllm.consumers.recirculation.spec import RecirculationSettings
from tllm.contracts.recurrence import (
    DependenceKind,
    IterationPoint,
    RecurrenceEdge,
    RecurrenceGraph,
    RecurrenceOp,
)


def _normal(token: int, layer: int) -> str:
    return f"t{token}/normal/l{layer}"


def _mix(token: int) -> str:
    return f"t{token}/mix"


def _recurrent(token: int, layer: int) -> str:
    return f"t{token}/recurrent/l{layer}"


def build_recirculation_graph(
    config: RecirculationSettings,
    *,
    token_count: int,
    layer_count: int,
) -> RecurrenceGraph:
    source_layer, destination_layer = config.resolved_layers(layer_count)
    ops: list[RecurrenceOp] = []
    edges: list[RecurrenceEdge] = []
    for token in range(token_count):
        for layer in range(layer_count):
            normal = _normal(token, layer)
            ops.append(
                RecurrenceOp(
                    normal,
                    f"model_layer:{layer}",
                    IterationPoint(token, layer, "normal"),
                )
            )
            if layer:
                edges.append(
                    RecurrenceEdge(_normal(token, layer - 1), normal, DependenceKind.DATA)
                )
            if token:
                prior = (
                    _normal(token - 1, layer)
                    if layer <= destination_layer
                    else _recurrent(token - 1, layer)
                )
                edges.append(
                    RecurrenceEdge(
                        prior,
                        normal,
                        DependenceKind.KV_WRITE_READ
                        if layer <= destination_layer
                        else DependenceKind.KV_OVERWRITE_READ,
                        distance=(1,),
                    )
                )
        mix = _mix(token)
        ops.append(
            RecurrenceOp(
                mix,
                "recirculation_mix",
                IterationPoint(token, destination_layer, "mix"),
            )
        )
        edges.extend(
            RecurrenceEdge(value, mix, DependenceKind.DATA)
            for value in (
                _normal(token, destination_layer),
                _normal(token, source_layer),
                _normal(token, layer_count - 1),
            )
        )
        for layer in range(destination_layer + 1, layer_count):
            recurrent = _recurrent(token, layer)
            ops.append(
                RecurrenceOp(
                    recurrent,
                    f"model_layer:{layer}",
                    IterationPoint(token, layer, "recurrent"),
                )
            )
            previous = mix if layer == destination_layer + 1 else _recurrent(token, layer - 1)
            edges.extend(
                (
                    RecurrenceEdge(previous, recurrent, DependenceKind.DATA),
                    RecurrenceEdge(
                        _normal(token, layer), recurrent, DependenceKind.CONTROL
                    ),
                )
            )
    return RecurrenceGraph(tuple(ops), tuple(edges))


__all__ = ["build_recirculation_graph"]

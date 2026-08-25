"""Backend-independent exact Recirculation graph construction."""

from __future__ import annotations

from tllm.contracts.recirculation import RecirculationPrefillBindingConfig
from tllm.workgraph.recurrence import (
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


def build_exact_recirculation_graph(
    config: RecirculationPrefillBindingConfig,
    *,
    token_count: int,
    layer_count: int,
) -> RecurrenceGraph:
    """Build the finite dependence graph without selecting its schedule."""

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    source_layer, destination_layer = config.resolved_layers(layer_count)
    ops: list[RecurrenceOp] = []
    edges: list[RecurrenceEdge] = []
    for token in range(token_count):
        for layer in range(layer_count):
            ops.append(
                RecurrenceOp(
                    op_id=_normal(token, layer),
                    executor=f"model_layer:{layer}",
                    point=IterationPoint(token, layer, "normal"),
                )
            )
            if layer > 0:
                edges.append(
                    RecurrenceEdge(
                        _normal(token, layer - 1),
                        _normal(token, layer),
                        DependenceKind.DATA,
                    )
                )
            if token > 0:
                prior = (
                    _normal(token - 1, layer)
                    if layer <= destination_layer
                    else _recurrent(token - 1, layer)
                )
                edges.append(
                    RecurrenceEdge(
                        prior,
                        _normal(token, layer),
                        DependenceKind.KV_WRITE_READ
                        if layer <= destination_layer
                        else DependenceKind.KV_OVERWRITE_READ,
                        distance=(1,),
                    )
                )
        ops.append(
            RecurrenceOp(
                op_id=_mix(token),
                executor="recirculation_mix",
                point=IterationPoint(token, destination_layer, "mix"),
            )
        )
        for dependency in (
            _normal(token, destination_layer),
            _normal(token, source_layer),
            _normal(token, layer_count - 1),
        ):
            edges.append(RecurrenceEdge(dependency, _mix(token), DependenceKind.DATA))
        for layer in range(destination_layer + 1, layer_count):
            ops.append(
                RecurrenceOp(
                    op_id=_recurrent(token, layer),
                    executor=f"model_layer:{layer}",
                    point=IterationPoint(token, layer, "recurrent"),
                )
            )
            previous = (
                _mix(token)
                if layer == destination_layer + 1
                else _recurrent(token, layer - 1)
            )
            edges.append(
                RecurrenceEdge(previous, _recurrent(token, layer), DependenceKind.DATA)
            )
            edges.append(
                RecurrenceEdge(
                    _normal(token, layer),
                    _recurrent(token, layer),
                    DependenceKind.CONTROL,
                )
            )
    return RecurrenceGraph(tuple(ops), tuple(edges))


__all__ = ["build_exact_recirculation_graph"]

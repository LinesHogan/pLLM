"""Typed Recirculation stage declarations compiled and signed by Rust."""

from __future__ import annotations

import tllm._core as _core

from tllm.contracts.typed_replay import (
    VLLM_FINAL_NORM_PRIMITIVE,
    VLLM_LAYER_REPLAY_PRIMITIVE,
)
from tllm.consumers.recirculation.graph import build_recirculation_graph
from tllm.consumers.recirculation.spec import RecirculationSettings
from tllm.workgraph.recurrence import RecurrenceGraph

PREFILL_BINDER_IMPLEMENTATION_ID = "tllm.recirculation:bind_prefill@1"
RECIRCULATION_MIX_PRIMITIVE = "tllm.recirculation:typed_mix@1"
SEED_STATE_BINDING_KIND = "tllm.vllm:seed-state@1"
LAYER_REPLAY_BINDING_KIND = "tllm.vllm:layer-replay@1"
MIX_BINDING_KIND = "tllm.recirculation:mix@1"
FINAL_NORM_BINDING_KIND = "tllm.vllm:final-norm@1"
NORMALIZED_OUTPUT_BINDING_KIND = "tllm.vllm:normalized-output@1"


def _normal(token: int, layer: int) -> str:
    return f"t{token}/normal/l{layer}"


def _recurrent(token: int, layer: int) -> str:
    return f"t{token}/recurrent/l{layer}"


def _mix(token: int) -> str:
    return f"t{token}/mix"


def _output(token: int) -> str:
    return f"recirculation/output/{token}"


def _recurrence(
    graph: RecurrenceGraph,
    config: RecirculationSettings,
    *,
    row_count: int,
    layer_count: int,
) -> _core.TypedRecurrence:
    source_layer, destination_layer = config.resolved_layers(layer_count)
    operations: list[_core.TypedOperation] = []
    for op in graph.ops:
        point = op.point
        call_resource = f"vllm_call/{op.op_id}"
        if point.pass_name == "normal":
            operands = (
                [_core.TypedOperand.seed(f"seed/{op.op_id}")]
                if point.layer == 0
                else [
                    _core.TypedOperand.region_value(
                        _normal(point.token, point.layer - 1)
                    )
                ]
            )
            operands.append(_core.TypedOperand.backend_resource(call_resource))
            primitive_key = VLLM_LAYER_REPLAY_PRIMITIVE
        elif point.pass_name == "recurrent":
            previous = (
                _mix(point.token)
                if point.layer == destination_layer + 1
                else _recurrent(point.token, point.layer - 1)
            )
            operands = [
                _core.TypedOperand.region_value(previous),
                _core.TypedOperand.backend_resource(call_resource),
            ]
            primitive_key = VLLM_LAYER_REPLAY_PRIMITIVE
        elif point.pass_name == "mix":
            operands = [
                _core.TypedOperand.region_value(
                    _normal(point.token, source_layer)
                ),
                _core.TypedOperand.region_value(
                    _normal(point.token, destination_layer)
                ),
                _core.TypedOperand.backend_resource(
                    f"recirculation_mix/{op.op_id}"
                ),
            ]
            primitive_key = RECIRCULATION_MIX_PRIMITIVE
        else:
            raise RuntimeError(
                f"Recirculation declaration has unsupported pass {point.pass_name!r}"
            )
        operations.append(
            _core.TypedOperation(
                op.op_id,
                primitive_key,
                PREFILL_BINDER_IMPLEMENTATION_ID,
                operands,
                (op.op_id,),
            )
        )

    for token in range(row_count):
        operations.append(
            _core.TypedOperation(
                f"vllm_norm/{token}",
                VLLM_FINAL_NORM_PRIMITIVE,
                PREFILL_BINDER_IMPLEMENTATION_ID,
                (
                    _core.TypedOperand.region_value(
                        _normal(token, layer_count - 1)
                    ),
                    _core.TypedOperand.backend_resource("vllm_final_norm"),
                ),
                (_output(token),),
            )
        )

    dependencies = [
        _core.TypedDependency(edge.source, edge.target, edge.kind.value)
        for edge in graph.edges
    ]
    dependencies.extend(
        _core.TypedDependency(
            _normal(token, layer_count - 1),
            f"vllm_norm/{token}",
            "data",
        )
        for token in range(row_count)
    )
    return _core.TypedRecurrence(
        f"recirculation.vllm-prefill/rows-{row_count}",
        PREFILL_BINDER_IMPLEMENTATION_ID,
        _core.TypedStageAdmission.prefill_reject(),
        _core.TypedStageFusionPolicy.serial(),
        operations,
        dependencies,
        tuple(_output(token) for token in range(row_count)),
    )


def build_stage_definitions(
    config: RecirculationSettings,
    *,
    max_row_count: int,
    layer_count: int,
) -> tuple[_core.TypedRecurrenceStageDefinition, ...]:
    definitions: list[_core.TypedRecurrenceStageDefinition] = []
    for row_count in range(1, max_row_count + 1):
        graph = build_recirculation_graph(
            config,
            token_count=row_count,
            layer_count=layer_count,
        )
        seed_bindings: list[_core.TypedStageBinding] = []
        resource_bindings = [
            _core.TypedStageResourceBinding(
                "vllm_final_norm", FINAL_NORM_BINDING_KIND, ()
            )
        ]
        for op in graph.ops:
            point = op.point
            if point.pass_name == "normal" and point.layer == 0:
                seed_bindings.append(
                    _core.TypedStageBinding(
                        f"seed/{op.op_id}", SEED_STATE_BINDING_KIND, (point.token,)
                    )
                )
            if point.pass_name in {"normal", "recurrent"}:
                resource_bindings.append(
                    _core.TypedStageResourceBinding(
                        f"vllm_call/{op.op_id}",
                        LAYER_REPLAY_BINDING_KIND,
                        (
                            point.token,
                            point.layer,
                            0 if point.pass_name == "normal" else 1,
                        ),
                    )
                )
            elif point.pass_name == "mix":
                resource_bindings.append(
                    _core.TypedStageResourceBinding(
                        f"recirculation_mix/{op.op_id}",
                        MIX_BINDING_KIND,
                        (point.token,),
                    )
                )
        output_bindings = tuple(
            _core.TypedStageOutputBinding(
                f"result/{token}",
                _output(token),
                NORMALIZED_OUTPUT_BINDING_KIND,
                (token,),
            )
            for token in range(row_count)
        )
        definitions.append(
            _core.TypedRecurrenceStageDefinition(
                row_count,
                _recurrence(
                    graph,
                    config,
                    row_count=row_count,
                    layer_count=layer_count,
                ),
                seed_bindings,
                resource_bindings,
                output_bindings,
            )
        )
    return tuple(definitions)


__all__ = [
    "FINAL_NORM_BINDING_KIND",
    "LAYER_REPLAY_BINDING_KIND",
    "MIX_BINDING_KIND",
    "NORMALIZED_OUTPUT_BINDING_KIND",
    "PREFILL_BINDER_IMPLEMENTATION_ID",
    "RECIRCULATION_MIX_PRIMITIVE",
    "SEED_STATE_BINDING_KIND",
    "build_stage_definitions",
]

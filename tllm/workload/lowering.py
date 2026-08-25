"""Lower workload declarations into the unified scheduler graph."""

from __future__ import annotations

import hashlib

from tllm.schedule.contracts import (
    ScheduleAffineAccess,
    ScheduleAffineExpr,
    ScheduleDomain,
    ScheduleEdge,
    ScheduleEdgeKind,
    ScheduleEffect,
    ScheduleEffectKind,
    ScheduleGraph,
    HiddenLayerKind as ScheduleHiddenLayerKind,
    ScheduleInput,
    ScheduleInputKind,
    ScheduleOutput,
    ScheduleOutputKind,
    ScheduleState,
    ScheduleTask,
    ScheduleValue,
    TaskTiming,
)
from tllm.workload.ir import EffectKind, ValueSourceKind, WorkloadEdgeKind, WorkloadIR


def lower_schedule_graph(
    workloads: tuple[WorkloadIR, ...],
    *,
    persistent_state_bytes: int = 0,
) -> ScheduleGraph:
    """Combine workload IRs without selecting order, placement, or buffers."""

    if not isinstance(workloads, tuple) or any(
        not isinstance(workload, WorkloadIR) for workload in workloads
    ):
        raise TypeError("workloads must be tuple[WorkloadIR, ...]")
    if not isinstance(persistent_state_bytes, int) or isinstance(persistent_state_bytes, bool):
        raise TypeError("persistent_state_bytes must be int")
    if persistent_state_bytes < 0:
        raise ValueError("persistent_state_bytes must be non-negative")
    semantic_keys = tuple(workload.semantic_key for workload in workloads)
    combined_key = hashlib.sha256("\n".join(semantic_keys).encode("ascii")).hexdigest()
    multiple = len(workloads) > 1

    def task_id(workload: WorkloadIR, name: str) -> str:
        return f"{workload.workload_id}/{name}" if multiple else name

    def state_id(workload: WorkloadIR, name: str) -> str:
        return f"{workload.workload_id}/{name}" if multiple else name

    tasks: list[ScheduleTask] = []
    edges: list[ScheduleEdge] = []
    values: list[ScheduleValue] = []
    states: list[ScheduleState] = []
    edge_kind = {
        WorkloadEdgeKind.DATA: ScheduleEdgeKind.DATA,
        WorkloadEdgeKind.EFFECT: ScheduleEdgeKind.EFFECT,
        WorkloadEdgeKind.CONTROL: ScheduleEdgeKind.CONTROL,
    }
    effect_kind = {
        EffectKind.READ: ScheduleEffectKind.STATE_READ,
        EffectKind.CURRENT: ScheduleEffectKind.STATE_READ,
        EffectKind.TRANSITION: ScheduleEffectKind.STATE_TRANSITION,
        EffectKind.NEXT: ScheduleEffectKind.VERSION_PUBLISH,
    }
    for workload in workloads:
        states.extend(
            ScheduleState(
                resource=state_id(workload, state.name),
                scope=state.scope.value,
                request_key=state.request_key,
                transition=state.transition,
            )
            for state in workload.states
        )
        for task in workload.tasks:
            tasks.append(
                ScheduleTask(
                    task_id(workload, task.name),
                    timing=TaskTiming(task.timing.after, task.timing.before),
                    implementation_ids=tuple(
                        implementation.key for implementation in task.implementations
                    ),
                    implementation_kinds=tuple(
                        implementation.kind.value
                        for implementation in task.implementations
                    ),
                    implementation_capabilities=tuple(
                        tuple(sorted(set(implementation.capabilities)))
                        for implementation in task.implementations
                    ),
                    required_capabilities=tuple(
                        sorted(
                            {
                                capability
                                for implementation in task.implementations
                                for capability in implementation.capabilities
                                if capability != "torch.export"
                            }
                        )
                    ),
                    capture_safe=(
                        not task.effects
                        and all(
                            implementation.kind.value == "captured"
                            for implementation in task.implementations
                        )
                    ),
                    effects=tuple(
                        ScheduleEffect(
                            effect_kind[effect.kind],
                            state_id(workload, effect.state),
                        )
                        for effect in task.effects
                    ),
                    domain=ScheduleDomain(
                        name=task.domain.name,
                        indices=task.domain.indices,
                        bounds=task.domain.bounds,
                        parameters=task.domain.parameters,
                    ),
                    accesses=tuple(
                        ScheduleAffineAccess(
                            value=access.value,
                            indices=tuple(
                                ScheduleAffineExpr(
                                    expression.coefficients,
                                    expression.constant,
                                )
                                for expression in access.indices
                            ),
                            mode=access.mode,
                        )
                        for access in task.accesses
                    ),
                    inputs=tuple(
                        ScheduleInput(
                            name=task_input.name,
                            kind=ScheduleInputKind(task_input.kind.value),
                            source=(
                                task_id(workload, task_input.source)
                                if task_input.kind is ValueSourceKind.RESULT
                                else task_input.source
                            ),
                            output=task_input.output,
                            layer=task_input.layer,
                            layer_kind=ScheduleHiddenLayerKind(
                                task_input.layer_kind or "none"
                            ),
                            site=task_input.site,
                            phase=task_input.phase,
                        )
                        for task_input in task.inputs
                    ),
                    outputs=tuple(
                        ScheduleOutput(
                            name=output.name,
                            kind=ScheduleOutputKind(output.kind.value),
                            target=output.target,
                        )
                        for output in task.outputs
                    ),
                )
            )
        for edge in workload.edges:
            edges.append(
                ScheduleEdge(
                    task_id(workload, edge.source),
                    task_id(workload, edge.target),
                    edge_kind[edge.kind],
                )
            )
        for value in workload.values:
            values.append(
                ScheduleValue(
                    value_id=value.value_id,
                    producer=task_id(workload, value.producer),
                    producer_output=value.output,
                    consumers=tuple(task_id(workload, consumer) for consumer in value.consumers),
                )
            )
    return ScheduleGraph(
        workload_key=combined_key,
        tasks=tuple(tasks),
        edges=tuple(edges),
        values=tuple(values),
        states=tuple(states),
        persistent_state_bytes=persistent_state_bytes,
        schema_version=2,
    )


__all__ = ["lower_schedule_graph"]

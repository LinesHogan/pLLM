"""Lower workload declarations into the unified scheduler graph."""

from __future__ import annotations

import hashlib

from tllm.schedule.contracts import (
    ScheduleAffineAccess,
    ScheduleAffineExpr,
    ScheduleCapability,
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
    ScheduleStateStorage,
    ScheduleTask,
    ScheduleValue,
    TaskTiming,
)
from tllm.workload.ir import (
    EffectKind,
    StateStorage,
    ValueSourceKind,
    WorkloadEdgeKind,
    WorkloadIR,
)


_IMPLEMENTATION_EVIDENCE = frozenset(
    {
        "runtime.async_host_safe",
        "runtime.optimizer_transition",
        "runtime.side_stream_safe",
        "torch.export",
    }
)


def lower_schedule_graph(
    workloads: tuple[WorkloadIR, ...],
    *,
    persistent_state_bytes: int = 0,
) -> ScheduleGraph:
    """Combine workload IRs without selecting order, placement, or buffers."""

    if persistent_state_bytes < 0:
        raise ValueError("persistent_state_bytes must be non-negative")
    has_native_v4 = any(workload.schema_version == 4 for workload in workloads)
    has_native_v3 = any(workload.schema_version == 3 for workload in workloads)
    if has_native_v4 and any(
        workload.schema_version != 4 for workload in workloads
    ):
        raise ValueError(
            "cannot combine native WorkloadIR v4 with an older WorkloadIR"
        )
    if has_native_v3 and any(
        workload.schema_version < 3 for workload in workloads
    ):
        raise ValueError(
            "cannot combine native WorkloadIR v3 with legacy WorkloadIR"
        )
    graph_schema_version = 5 if has_native_v4 else 4 if has_native_v3 else 3
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
        EffectKind.TRANSITION_MAYBE: ScheduleEffectKind.STATE_TRANSITION_MAYBE,
        EffectKind.NEXT: ScheduleEffectKind.VERSION_PUBLISH,
        EffectKind.EXTERN_IO: ScheduleEffectKind.EXTERN_IO,
    }
    for workload in workloads:
        state_transitions = {
            state.name: state.transition for state in workload.states
        }
        # extern_io names a process-global channel, not workload state: it is
        # never prefixed, so two workloads naming the same channel share it.
        extern_resources = {
            effect.state
            for task in workload.tasks
            for effect in task.effects
            if effect.kind is EffectKind.EXTERN_IO
        }
        states.extend(
            ScheduleState(
                resource=state_id(workload, state.name),
                scope=state.scope.value,
                request_key=state.request_key,
                storage=ScheduleStateStorage(state.storage.value),
                transition=state.transition,
            )
            for state in workload.states
        )
        for task in workload.tasks:
            common_capabilities: set[str] = {
                capability
                for capability in task.implementations[0].capabilities
                if capability not in _IMPLEMENTATION_EVIDENCE
            }
            for implementation in task.implementations[1:]:
                common_capabilities.intersection_update(
                    capability
                    for capability in implementation.capabilities
                    if capability not in _IMPLEMENTATION_EVIDENCE
                )
            if any(
                task_input.kind is ValueSourceKind.PACKED_REQUEST_ROWS
                for task_input in task.inputs
            ):
                common_capabilities.add(
                    ScheduleCapability.PACKED_PHASE_ROWS.value
                )
            if any(
                task_input.kind is ValueSourceKind.PACKED_HIDDEN
                for task_input in task.inputs
            ):
                common_capabilities.add(
                    ScheduleCapability.PACKED_HIDDEN_BLOCK_OUTPUT.value
                )
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
                    # This graph-level field contains only requirements that
                    # every candidate shares. Rust selects one candidate
                    # against the timeline and derives the Region's actual
                    # requirements from that selected declaration.
                    required_capabilities=tuple(
                        sorted(common_capabilities)
                    ),
                    capture_safe=(
                        all(
                            implementation.kind.value == "captured"
                            for implementation in task.implementations
                        )
                        and all(
                            effect.kind in (EffectKind.READ, EffectKind.CURRENT)
                            and state_transitions.get(effect.state) == "optimizer"
                            and next(
                                state.storage
                                for state in workload.states
                                if state.name == effect.state
                            )
                            is StateStorage.OPTIMIZER_STABLE
                            for effect in task.effects
                        )
                        and all(
                            "runtime.async_host_safe"
                            not in implementation.capabilities
                            for implementation in task.implementations
                        )
                        and all(
                            task_input.kind
                            not in (
                                ValueSourceKind.PACKED_REQUEST_ROWS,
                                ValueSourceKind.PACKED_HIDDEN,
                            )
                            for task_input in task.inputs
                        )
                    ),
                    effects=tuple(
                        ScheduleEffect(
                            effect_kind[effect.kind],
                            effect.state
                            if effect.kind is EffectKind.EXTERN_IO
                            else state_id(workload, effect.state),
                            selection_port=effect.selection_port,
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
                                else state_id(workload, task_input.source)
                                if task_input.kind
                                is ValueSourceKind.STATE_SELECTION
                                else task_input.source
                            ),
                            output=task_input.output,
                            layer=task_input.layer,
                            layer_kind=ScheduleHiddenLayerKind(
                                task_input.layer_kind or "none"
                            ),
                            site=task_input.site,
                            phase=task_input.phase,
                            packed_rows_port=task_input.packed_rows_port,
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
            if (
                multiple
                and edge.kind is WorkloadEdgeKind.EFFECT
                and edge.detail in extern_resources
            ):
                # Shared extern channels serialize across workloads; those
                # edges are re-derived over the combined task list below.
                continue
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
    if multiple:
        # Re-derive extern channel serialization across the combined graph in
        # declaration order, mirroring the signed graph's derivation: a shared
        # channel serializes every adjacent pair of tasks that touch it.
        extern_previous: dict[str, str] = {}
        for workload in workloads:
            for task in workload.tasks:
                for effect in task.effects:
                    if effect.kind is not EffectKind.EXTERN_IO:
                        continue
                    current = task_id(workload, task.name)
                    previous = extern_previous.get(effect.state)
                    if previous is not None:
                        edges.append(
                            ScheduleEdge(previous, current, ScheduleEdgeKind.EFFECT)
                        )
                    extern_previous[effect.state] = current
    return ScheduleGraph(
        workload_key=combined_key,
        tasks=tuple(tasks),
        edges=tuple(edges),
        values=tuple(values),
        states=tuple(states),
        persistent_state_bytes=persistent_state_bytes,
        schema_version=graph_schema_version,
        # Non-exact policies are signed into the graph so the Decision can
        # explain which Regions run under approximation parity.  Criteria
        # stay host-side (declaration-only) by design.
        workload_approximations=tuple(
            sorted(
                (workload.workload_id, workload.approximation.value)
                for workload in workloads
                if workload.approximation.value != "exact"
            )
        ),
    )


__all__ = ["lower_schedule_graph"]

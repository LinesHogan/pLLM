"""Read-only lowering adapters for the migration-only Program frontend."""

from __future__ import annotations

from collections.abc import Mapping

from tllm.program.decl import EdgeKind as ProgramEdgeKind
from tllm.program.decl import StateSpec, declare_program
from tllm.workgraph.frontend import (
    FlowProgramDecl,
    HiddenSource,
    PromptIdxKey,
    RequestStateKey,
    ResultRef,
    SinkRef,
)
from tllm.workload.ir import (
    ApproximationPolicy,
    EffectIR,
    EffectKind,
    HiddenLayerKind,
    HiddenPhase,
    HiddenSite,
    ImplementationIR,
    ImplementationKind,
    OptimizerSpec,
    StateIR,
    StateScope,
    TaskIR,
    TaskInputIR,
    TaskOutputIR,
    TimingIR,
    ValueIR,
    ValueSourceKind,
    ValueTargetKind,
    WorkloadEdge,
    WorkloadEdgeKind,
    WorkloadIR,
)


def _state_ir(name: str, spec: StateSpec) -> StateIR:
    old_scope = str(spec.scope)
    scope = StateScope(old_scope)
    if isinstance(spec.key, PromptIdxKey):
        key = ("prompt_idx",)
    elif isinstance(spec.key, RequestStateKey):
        key = tuple(field.value for field in spec.key.fields)
    else:
        key = ()
    optimizer = spec.optimizer
    optimizer_type = optimizer if isinstance(optimizer, type) else type(optimizer)
    optimizer_spec = None
    if optimizer is not None:
        optimizer_spec = OptimizerSpec(optimizer_type.__name__.lower())
    return StateIR(
        name=name,
        scope=scope,
        request_key=key,
        parameters=(f"legacy:{name}",),
        optimizer=optimizer_spec,
    )


def lower_program(
    instance: object,
    *,
    states: Mapping[str, StateSpec],
    workload_id: str | None = None,
) -> WorkloadIR:
    """Project a validated Program into the new dependence/effect vocabulary."""

    declaration = declare_program(instance, states=states)
    identifier = workload_id or f"legacy.program.{type(instance).__module__}.{type(instance).__qualname__}"
    task_irs: list[TaskIR] = []
    values: list[ValueIR] = []
    for node in declaration.nodes:
        node_decl = node.decl
        inputs: list[TaskInputIR] = []
        for name, source in node_decl.inputs:
            if isinstance(source, HiddenSource):
                inputs.append(
                    TaskInputIR(
                        name,
                        ValueSourceKind.HIDDEN,
                        f"hidden:{source.layer}:{source.site}:{source.phase}",
                        layer=str(source.layer),
                        layer_kind=HiddenLayerKind.INDEX.value,
                        site=HiddenSite(source.site),
                        phase=HiddenPhase(source.phase),
                    )
                )
            elif isinstance(source, ResultRef):
                inputs.append(
                    TaskInputIR(
                        name,
                        ValueSourceKind.RESULT,
                        source.flow,
                        output=source.output,
                    )
                )
            else:
                inputs.append(TaskInputIR(name, ValueSourceKind.PROVIDED, name))
        outputs: list[TaskOutputIR] = []
        if isinstance(node_decl, FlowProgramDecl):
            for name, target in node_decl.outputs:
                if isinstance(target, SinkRef):
                    outputs.append(TaskOutputIR(name, ValueTargetKind.SINK, target.name))
                else:
                    outputs.append(TaskOutputIR(name, ValueTargetKind.TEMPORARY, f"temporary:{name}"))
        effects: tuple[EffectIR, ...] = ()
        if node_decl.state is not None:
            effect_kind = EffectKind.READ if isinstance(node_decl, FlowProgramDecl) else EffectKind.TRANSITION
            effects = (EffectIR(effect_kind, node_decl.state),)
        implementation_kind = (
            ImplementationKind.CAPTURED
            if isinstance(node_decl, FlowProgramDecl)
            else ImplementationKind.EXTERN
        )
        task_irs.append(
            TaskIR(
                name=node.name,
                ordinal=node.ordinal,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                effects=effects,
                timing=TimingIR("step.begin", "step.end"),
                implementations=(
                    ImplementationIR(
                        implementation_kind,
                        f"legacy.program:{identifier}.{node.name}@1",
                        capabilities=("legacy_adapter",),
                    ),
                ),
            )
        )
    for value in declaration.values:
        values.append(
            ValueIR(
                value_id=f"{identifier}:{value.producer}:{value.name}",
                producer=value.producer,
                output=value.name,
                consumers=value.consumers,
            )
        )
    edge_kind = {
        ProgramEdgeKind.DATA: WorkloadEdgeKind.DATA,
        ProgramEdgeKind.CONTROL: WorkloadEdgeKind.CONTROL,
        ProgramEdgeKind.STATE: WorkloadEdgeKind.EFFECT,
    }
    edges = tuple(
        WorkloadEdge(edge.source, edge.target, edge_kind[edge.kind], edge.detail)
        for edge in declaration.edges
        if edge.kind is not ProgramEdgeKind.STATE
    )
    # Program STATE edges point to a state name rather than a task. The new
    # effect is already carried directly on each task and ordering is inferred
    # only between state accesses, so these binding edges are intentionally not
    # copied as task edges.
    state_accesses: dict[str, list[tuple[str, EffectKind]]] = {}
    inferred_edges = list(edges)
    for task_ir in task_irs:
        for effect in task_ir.effects:
            accesses = state_accesses.setdefault(effect.state, [])
            if accesses and (
                effect.kind is EffectKind.TRANSITION
                or accesses[-1][1] is EffectKind.TRANSITION
            ):
                inferred_edges.append(
                    WorkloadEdge(
                        accesses[-1][0],
                        task_ir.name,
                        WorkloadEdgeKind.EFFECT,
                        effect.state,
                    )
                )
            accesses.append((task_ir.name, effect.kind))
    return WorkloadIR(
        workload_id=identifier,
        states=tuple(_state_ir(state.name, state.spec) for state in declaration.states),
        tasks=tuple(task_irs),
        values=tuple(values),
        edges=tuple(inferred_edges),
        schedule=declaration.schedule,
        approximation=ApproximationPolicy.EXACT,
    )


def dependence_effect_fingerprint(ir: WorkloadIR) -> tuple[tuple[str, str, str, str], ...]:
    """Stable projection used by old/new frontend differential tests."""

    effects = tuple(
        (task.name, effect.state, "effect", effect.kind.value)
        for task in ir.tasks
        for effect in task.effects
    )
    edges = tuple(
        (edge.source, edge.target, edge.kind.value, edge.detail) for edge in ir.edges
    )
    return tuple(sorted((*effects, *edges)))


__all__ = ["dependence_effect_fingerprint", "lower_program"]

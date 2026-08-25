#!/usr/bin/env python3
"""Static Program declaration: discovery, validation, and IR.

``declare_program`` inspects an instance whose methods carry Program
metadata (``@flow(inputs/outputs/state)`` / ``@extern(inputs/after/deadline)``
from ``tllm.workgraph.frontend``) and produces a validated, deterministically
ordered ``ProgramDecl``. Every validation failure raises ``WorkGraphError``
with a PG### code (see docs/program_adr.md); there is no silent fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch

from tllm.workgraph.frontend import (
    ExternProgramDecl,
    FlowProgramDecl,
    LocalRef,
    PromptIdxKey,
    RequestStateKey,
    ResultRef,
    SinkRef,
    StateScope,
)
from tllm.workgraph.lowering import WorkGraphError

_SUPPORTED_OPTIMIZERS: tuple[type[torch.optim.Optimizer], ...] = (
    torch.optim.SGD,
    torch.optim.Adam,
    torch.optim.AdamW,
)


def _optimizer_kind(optimizer: object) -> type[torch.optim.Optimizer] | None:
    """Return the optimizer class, or None when no optimizer is declared."""
    if optimizer is None:
        return None
    if isinstance(optimizer, type):
        if issubclass(optimizer, torch.optim.Optimizer):
            return optimizer
        raise WorkGraphError("PG160", f"unsupported optimizer {optimizer!r}")
    if isinstance(optimizer, torch.optim.Optimizer):
        return type(optimizer)
    raise WorkGraphError("PG160", f"unsupported optimizer {optimizer!r}")


@dataclass(frozen=True, slots=True)
class StateSpec:
    """One named optimizer/state bank owned by a Program."""

    scope: StateScope
    key: PromptIdxKey | RequestStateKey | None = None
    optimizer: object = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, StateScope):
            raise TypeError("StateSpec.scope must be a StateScope")
        if self.scope not in (StateScope.REQUEST, StateScope.REPLICA):
            raise WorkGraphError(
                "PG100", f"StateSpec.scope must be REQUEST or REPLICA, got {self.scope!r}"
            )
        if self.scope is StateScope.REQUEST and not isinstance(
            self.key, (PromptIdxKey, RequestStateKey)
        ):
            raise WorkGraphError(
                "PG100",
                "REQUEST-scoped state requires key=request_prompt_idx() or request_state_key(...)"
            )
        kind = _optimizer_kind(self.optimizer)
        if kind is not None and kind not in _SUPPORTED_OPTIMIZERS:
            raise WorkGraphError(
                "PG160",
                f"unsupported optimizer {kind.__name__}; supported: SGD, Adam, AdamW",
            )


@dataclass(frozen=True, slots=True)
class ProgramNode:
    """One declared function, in class-definition order."""

    name: str
    ordinal: int
    kind: str  # "flow" | "extern"
    function: Any
    decl: FlowProgramDecl | ExternProgramDecl


@dataclass(frozen=True, slots=True)
class ProgramValue:
    """One named flow output and the externs that consume it."""

    name: str
    producer: str  # node name
    ref: SinkRef | LocalRef
    consumers: tuple[str, ...]  # extern node names, ordinal order


@dataclass(frozen=True, slots=True)
class ProgramState:
    """One named state bank and the nodes bound to it."""

    name: str
    spec: StateSpec
    nodes: tuple[str, ...]  # node names declaring this state, ordinal order


class EdgeKind(StrEnum):
    DATA = "data"  # flow named output -> extern input (result ref)
    CONTROL = "control"  # extern.after ordering constraint
    STATE = "state"  # node -> state bank binding


@dataclass(frozen=True, slots=True)
class ProgramEdge:
    kind: EdgeKind
    source: str  # node name (STATE: the bound node)
    target: str  # node name (STATE: the state name)
    detail: str = ""  # DATA: "<output>-><param>"; CONTROL/STATE: ""


@dataclass(frozen=True, slots=True)
class ProgramDiagnostic:
    """Non-fatal observation (PG3xx); fatal validation raises WorkGraphError."""

    code: str
    message: str
    node: str = ""


@dataclass(frozen=True, slots=True)
class ProgramDecl:
    """Validated static IR for one Program instance (no CUDA objects)."""

    nodes: tuple[ProgramNode, ...]
    values: tuple[ProgramValue, ...]
    states: tuple[ProgramState, ...]
    edges: tuple[ProgramEdge, ...]
    schedule: tuple[str, ...]  # deterministic topological order of node names
    diagnostics: tuple[ProgramDiagnostic, ...]


def _discover_nodes(instance: object) -> tuple[ProgramNode, ...]:
    nodes: list[ProgramNode] = []
    for name, member in vars(type(instance)).items():
        try:
            flow_decl: object = member.flow_program  # type: ignore[attr-defined]
        except AttributeError:
            flow_decl = None
        try:
            extern_decl: object = member.extern_program  # type: ignore[attr-defined]
        except AttributeError:
            extern_decl = None
        if isinstance(flow_decl, FlowProgramDecl):
            nodes.append(
                ProgramNode(
                    name=name,
                    ordinal=len(nodes),
                    kind="flow",
                    function=getattr(instance, name),
                    decl=flow_decl,
                )
            )
        elif isinstance(extern_decl, ExternProgramDecl):
            nodes.append(
                ProgramNode(
                    name=name,
                    ordinal=len(nodes),
                    kind="extern",
                    function=getattr(instance, name),
                    decl=extern_decl,
                )
            )
    return tuple(nodes)


def _dependencies(node: ProgramNode) -> tuple[str, ...]:
    if not isinstance(node.decl, ExternProgramDecl):
        return ()
    deps: list[str] = list(node.decl.after)
    for _, value in node.decl.inputs:
        if isinstance(value, ResultRef):
            deps.append(value.flow)
    return tuple(dict.fromkeys(deps))


def declare_program(
    instance: object,
    *,
    states: Mapping[str, StateSpec],
) -> ProgramDecl:
    """Discover and validate the Program declared on ``instance``.

    Raises ``WorkGraphError`` with codes PG100 (unknown reference), PG110
    (dependency cycle), PG120 (duplicate sink producer), PG160 (unsupported
    optimizer). On failure nothing is registered or returned.
    """
    if not isinstance(states, Mapping):
        raise TypeError("states must be a mapping of name to StateSpec")
    state_map: dict[str, StateSpec] = {}
    for name, spec in states.items():
        if not isinstance(name, str) or not name:
            raise TypeError("state names must be non-empty strings")
        if not isinstance(spec, StateSpec):
            raise TypeError(f"states[{name!r}] must be a StateSpec")
        state_map[name] = spec

    nodes = _discover_nodes(instance)
    by_name = {node.name: node for node in nodes}

    for node in nodes:
        decl_state = node.decl.state
        if decl_state is not None and decl_state not in state_map:
            raise WorkGraphError(
                "PG100",
                f"node {node.name!r} references unknown state {decl_state!r}",
            )
        if isinstance(node.decl, FlowProgramDecl):
            continue
        for dep in _dependencies(node):
            if dep not in by_name:
                raise WorkGraphError(
                    "PG100",
                    f"node {node.name!r} references unknown flow {dep!r}",
                )
        for _, value in node.decl.inputs:
            if not isinstance(value, ResultRef):
                continue
            target = by_name.get(value.flow)
            if isinstance(target, ProgramNode) and isinstance(target.decl, FlowProgramDecl):
                output_names = {name for name, _ in target.decl.outputs}
                if value.output not in output_names:
                    raise WorkGraphError(
                        "PG100",
                        f"node {node.name!r} references unknown output "
                        f"{value.output!r} of flow {value.flow!r}",
                    )

    sink_producers: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node.decl, FlowProgramDecl):
            continue
        for output_name, ref in node.decl.outputs:
            if not isinstance(ref, SinkRef):
                continue
            prior = sink_producers.get(ref.name)
            if prior is not None:
                raise WorkGraphError(
                    "PG120",
                    f"sink {ref.name!r} has two producers: {prior!r} and {node.name!r}",
                )
            sink_producers[ref.name] = node.name

    # Deterministic topological order: Kahn's algorithm, ties broken by
    # class-definition ordinal.
    indegree: dict[str, int] = {node.name: 0 for node in nodes}
    dependents: dict[str, list[str]] = {node.name: [] for node in nodes}
    for node in nodes:
        for dep in _dependencies(node):
            indegree[node.name] += 1
            dependents[dep].append(node.name)
    ready = sorted(
        (node for node in nodes if indegree[node.name] == 0),
        key=lambda node: node.ordinal,
    )
    schedule: list[str] = []
    while ready:
        node = ready.pop(0)
        schedule.append(node.name)
        for dependent in dependents[node.name]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(by_name[dependent])
        ready.sort(key=lambda item: item.ordinal)
    if len(schedule) != len(nodes):
        remaining = sorted(name for name, degree in indegree.items() if degree > 0)
        raise WorkGraphError(
            "PG110",
            f"dependency cycle among externs: {', '.join(remaining)}",
        )

    # Named values: every flow output, with its extern consumers in ordinal
    # order.
    consumers_of: dict[tuple[str, str], list[str]] = {}
    edges: list[ProgramEdge] = []
    for node in nodes:
        if not isinstance(node.decl, ExternProgramDecl):
            continue
        for dep in node.decl.after:
            edges.append(ProgramEdge(kind=EdgeKind.CONTROL, source=dep, target=node.name))
        for param, value in node.decl.inputs:
            if isinstance(value, ResultRef):
                consumers_of.setdefault((value.flow, value.output), []).append(node.name)
                edges.append(
                    ProgramEdge(
                        kind=EdgeKind.DATA,
                        source=value.flow,
                        target=node.name,
                        detail=f"{value.output}->{param}",
                    )
                )
    values: list[ProgramValue] = []
    for node in nodes:
        if not isinstance(node.decl, FlowProgramDecl):
            continue
        for output_name, ref in node.decl.outputs:
            consumers = tuple(
                sorted(
                    consumers_of.get((node.name, output_name), ()),
                    key=lambda name: by_name[name].ordinal,
                )
            )
            values.append(
                ProgramValue(
                    name=output_name,
                    producer=node.name,
                    ref=ref,
                    consumers=consumers,
                )
            )

    # State banks with their bound nodes; STATE edges record the binding.
    state_nodes: dict[str, list[str]] = {name: [] for name in state_map}
    for node in nodes:
        decl_state = node.decl.state
        if decl_state is not None:
            state_nodes[decl_state].append(node.name)
            edges.append(
                ProgramEdge(kind=EdgeKind.STATE, source=node.name, target=decl_state)
            )
    program_states = tuple(
        ProgramState(name=name, spec=state_map[name], nodes=tuple(members))
        for name, members in state_nodes.items()
    )

    diagnostics: list[ProgramDiagnostic] = []
    for value in values:
        if isinstance(value.ref, LocalRef) and not value.consumers:
            diagnostics.append(
                ProgramDiagnostic(
                    code="PG300",
                    message=(
                        f"flow {value.producer!r} output {value.name!r} is local "
                        "and never consumed"
                    ),
                    node=value.producer,
                )
            )
    for state in program_states:
        if not state.nodes:
            diagnostics.append(
                ProgramDiagnostic(
                    code="PG301",
                    message=f"state {state.name!r} is declared but no node uses it",
                )
            )

    return ProgramDecl(
        nodes=nodes,
        values=tuple(values),
        states=program_states,
        edges=tuple(edges),
        schedule=tuple(schedule),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "EdgeKind",
    "ProgramDecl",
    "ProgramDiagnostic",
    "ProgramEdge",
    "ProgramNode",
    "ProgramState",
    "ProgramValue",
    "StateSpec",
    "declare_program",
]

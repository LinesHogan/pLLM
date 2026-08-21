#!/usr/bin/env python3
"""Stage slicing: split one exported Program flow into hook-level stages.

The slice works only on the strict-export SSA facts produced by
``tllm.workgraph.capture.capture_module_flow``. Every operator's earliest
hook is the maximum availability level of its data inputs (declaration order
of ``flow.inputs`` defines the arrival order; parameters/buffers/constants
are always available at level 0). Operators are scheduled deterministically
at their earliest legal level (DAG list scheduling, definition-order
tie-break).

Cross-stage values: when a later stage consumes a value produced earlier,
the producer chain is *recomputed* in the later stage if every operator on
the chain is pure (capture-safe, no RNG, no mutation — mutation never
survives strict capture). Operators carrying ``Effect.RNG`` or any non-pure
effect are pinned: they are computed exactly once at their earliest level
and their results cross stages as owned values.

Failure is fail-closed: any export/declaration mismatch raises
``WorkGraphError("PG130", ...)``; nothing partially lowers.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import cast
import operator as _operator

import torch
from torch.fx.node import Argument, map_arg

from tllm.workgraph.capture import capture_module_flow
from tllm.workgraph.frontend import (
    ArtifactKey,
    CapturedFlow,
    CaptureStatus,
    DynamicDim,
    Effect,
    FlowProgramDecl,
    ShapeGuard,
)
from tllm.workgraph.lowering import WorkGraphError


@dataclass(frozen=True, slots=True)
class StageValueDesc:
    """One owned value crossing stage boundaries inside a flow."""

    value_id: str  # "leaf/{ordinal}@1"
    producer_stage: int
    consumer_stages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProgramStage:
    """One hook-level slice of a flow's exported graph."""

    flow: str
    flow_ordinal: int
    stage_ordinal: int
    hook_level: int
    placeholder_ordinals: tuple[int, ...]  # user inputs arriving at this stage
    operator_ordinals: tuple[int, ...]  # operators first computed in this stage
    recomputed_ordinals: tuple[int, ...]  # pure operators recomputed in this stage
    input_values: tuple[str, ...]  # cross-stage values read here
    output_values: tuple[str, ...]  # values leaving this stage
    flow_outputs: tuple[str, ...]  # declared output names produced here


@dataclass(frozen=True, slots=True)
class FlowStagePlan:
    """Frozen stage IR for one sliced flow (no CUDA objects)."""

    flow: str
    flow_ordinal: int
    artifact_key: ArtifactKey
    shape_guard: ShapeGuard
    output_names: tuple[str, ...]  # declared outputs in return order
    stages: tuple[ProgramStage, ...]
    cross_stage_values: tuple[StageValueDesc, ...]
    hook_count: int


def _require_module_bound(node_function: object) -> tuple[torch.nn.Module, FunctionType]:
    """Recover (instance, raw function) from a decorated bound method."""
    if not isinstance(node_function, MethodType):
        raise WorkGraphError(
            "PG130", "Program flow capture requires a bound nn.Module method"
        )
    instance = node_function.__self__
    raw = inspect.unwrap(node_function.__func__)
    if not isinstance(instance, torch.nn.Module) or not isinstance(raw, FunctionType):
        raise WorkGraphError(
            "PG130", "Program flow capture requires a bound nn.Module method"
        )
    return instance, raw


def _broadcast_getitem_aliases(
    graph_nodes: tuple[torch.fx.Node, ...],
    node_ordinal: dict[torch.fx.Node, int],
) -> dict[int, int]:
    """Map ``getitem(broadcast_tensors(xs), i)`` ordinals to ``xs[i]``.

    ``F.mse_loss`` and friends insert ``broadcast_tensors``; the returned
    element is a stride view of one input, so for availability and execution
    the getitem is an alias of that input, not a consumer of the whole
    broadcast group.
    """
    aliases: dict[int, int] = {}
    broadcast = torch.ops.aten.broadcast_tensors.default
    for ordinal, node in enumerate(graph_nodes):
        if node.op != "call_function" or node.target is not _operator.getitem:
            continue
        if len(node.args) != 2:
            continue
        source, index = node.args[0], node.args[1]
        if not isinstance(source, torch.fx.Node) or source.target != broadcast:
            continue
        if not isinstance(index, int):
            continue
        elements = source.args[0]
        if not isinstance(elements, (tuple, list)) or index >= len(elements):
            continue
        element = elements[index]
        if isinstance(element, torch.fx.Node):
            aliases[ordinal] = node_ordinal[element]
    return aliases


class StagedFlowExecutor:
    """Eager stage-by-stage interpreter for one sliced flow.

    Reference executor for CPU verification and the P3 fallback lane: runs
    the exported graph stage by stage, honoring availability gating (a user
    input is only loaded when its hook level is reached), recomputing pure
    chains, and threading cross-stage values through a plain environment.
    """

    def __init__(
        self,
        plan: FlowStagePlan,
        exported_program: torch.export.ExportedProgram,
        instance: torch.nn.Module,
        aliases: dict[int, int],
    ) -> None:
        self._plan = plan
        self._graph_nodes = tuple(exported_program.graph.nodes)
        self._aliases = aliases
        self._constants = exported_program.constants
        self._instance = instance
        signature = exported_program.graph_signature
        # Placeholder node name -> ("input", position) | ("state", target).
        self._placeholder_sources: dict[str, tuple[str, object]] = {}
        user_inputs = [
            spec for spec in signature.input_specs if spec.kind.name == "USER_INPUT"
        ]
        user_index = {spec.arg.name: index for index, spec in enumerate(user_inputs)}
        for spec in signature.input_specs:
            if spec.kind.name == "USER_INPUT":
                self._placeholder_sources[spec.arg.name] = (
                    "input",
                    user_index[spec.arg.name],
                )
            else:
                self._placeholder_sources[spec.arg.name] = ("state", spec.target)

    def _resolve_state(self, target: object) -> torch.Tensor:
        name = str(target)
        prefix = "_program_instance."
        if name.startswith(prefix):
            value: object = self._instance
            for segment in name[len(prefix):].split("."):
                value = getattr(value, segment)
            if torch.is_tensor(value):
                return value
        constant = self._constants.get(name)
        if torch.is_tensor(constant):
            return constant
        raise WorkGraphError("PG130", f"cannot resolve lifted state target {name!r}")

    def run(self, *user_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Execute all stages eagerly; returns declared outputs by name.

        Each stage loads only the placeholders arriving at its hook level and
        evaluates its owned operators plus its recomputed pure chain, so the
        execution genuinely re-runs recomputed operators.
        """
        plan = self._plan
        input_count = sum(
            1 for source in self._placeholder_sources.values() if source[0] == "input"
        )
        if len(user_inputs) != input_count:
            raise WorkGraphError(
                "PG130",
                f"flow expects {input_count} user inputs, got {len(user_inputs)}",
            )
        env: dict[torch.fx.Node, object] = {}
        output_positions: list[torch.fx.Node] = []
        for node in self._graph_nodes:
            if node.op == "output":
                _flatten_nodes(node.args, output_positions)

        for stage in plan.stages:
            wanted = (
                set(stage.placeholder_ordinals)
                | set(stage.operator_ordinals)
                | set(stage.recomputed_ordinals)
            )
            for ordinal, node in enumerate(self._graph_nodes):
                if ordinal not in wanted:
                    continue
                if node.op == "placeholder":
                    kind, source = self._placeholder_sources[node.name]
                    env[node] = (
                        user_inputs[cast(int, source)]
                        if kind == "input"
                        else self._resolve_state(source)
                    )
                    continue
                if ordinal in self._aliases:
                    env[node] = env[self._graph_nodes[self._aliases[ordinal]]]
                    continue
                if node.op == "call_function":
                    args = map_arg(node.args, lambda dep: cast(Argument, env[dep]))
                    kwargs = map_arg(node.kwargs, lambda dep: cast(Argument, env[dep]))
                    env[node] = node.target(*args, **kwargs)  # type: ignore[operator]
                    continue
                raise WorkGraphError(
                    "PG130", f"unsupported exported node kind {node.op!r} in stage"
                )

        results: dict[str, torch.Tensor] = {}
        if len(output_positions) != len(plan.output_names):
            raise WorkGraphError("PG130", "stage plan lost the graph output tree")
        for position, name in enumerate(plan.output_names):
            node = output_positions[position]
            if node not in env:
                raise WorkGraphError("PG130", f"stage plan lost output {name!r}")
            results[name] = cast(torch.Tensor, env[node])
        return results


def _flatten_nodes(value: object, out: list[torch.fx.Node]) -> None:
    if isinstance(value, torch.fx.Node):
        out.append(value)
    elif isinstance(value, (tuple, list)):
        for item in cast(tuple[object, ...], value):
            _flatten_nodes(item, out)
    elif value is None:
        return
    else:
        raise WorkGraphError("PG130", "export produced a non-Tensor output")


def _pure_chain(
    start: int,
    *,
    operator_effects: dict[int, tuple[Effect, ...]],
    operator_inputs: dict[int, tuple[int, ...]],
    level_by_ordinal: dict[int, int],
    stage: int,
    memo: dict[int, frozenset[int] | None],
) -> frozenset[int] | None:
    """Operators to recompute for value ``start`` in ``stage``, or None.

    A chain is recomputable only when every operator on it is pure; the
    returned set contains the transitive pure ancestors first computed
    before ``stage`` (placeholders are leaves and never recomputed).
    """
    if start in memo:
        return memo[start]
    if start not in operator_effects:
        memo[start] = frozenset()  # placeholder: always available, never moved
        return memo[start]
    if operator_effects[start] != (Effect.PURE,):
        memo[start] = None
        return None
    collected: set[int] = (
        {start} if level_by_ordinal[start] < stage else set()
    )
    for input_ordinal in operator_inputs[start]:
        chain = _pure_chain(
            input_ordinal,
            operator_effects=operator_effects,
            operator_inputs=operator_inputs,
            level_by_ordinal=level_by_ordinal,
            stage=stage,
            memo=memo,
        )
        if chain is None:
            memo[start] = None
            return None
        collected.update(chain)
    memo[start] = frozenset(collected)
    return memo[start]


def slice_flow_stages(
    *,
    flow: str,
    flow_ordinal: int,
    decl: FlowProgramDecl,
    function: object,
    example_args: tuple[object, ...],
    dynamic_dims: tuple[DynamicDim, ...] = (),
) -> tuple[FlowStagePlan, StagedFlowExecutor]:
    """Capture ``function`` and slice its exported graph into hook stages.

    ``example_args`` are cold-specialization examples (real hidden sizes);
    ``dynamic_dims`` reuse the bounded ``DynamicDim`` specialization surface.
    Raises ``WorkGraphError`` PG130 on any unsupported export graph.
    """
    instance, raw = _require_module_bound(function)
    captured, exported_program = capture_module_flow(
        instance,
        raw,
        example_args,
        {},
        dynamic_dims=dynamic_dims,
    )
    if captured.status != CaptureStatus.CAPTURED or exported_program is None:
        codes = ", ".join(diagnostic.code.value for diagnostic in captured.diagnostics)
        raise WorkGraphError(
            "PG130",
            f"flow {flow!r} did not produce a supported export graph ({codes})",
        )
    return _slice_captured(
        flow=flow,
        flow_ordinal=flow_ordinal,
        decl=decl,
        captured=captured,
        exported_program=exported_program,
        instance=instance,
    )


def _slice_captured(
    *,
    flow: str,
    flow_ordinal: int,
    decl: FlowProgramDecl,
    captured: CapturedFlow,
    exported_program: torch.export.ExportedProgram,
    instance: torch.nn.Module,
) -> tuple[FlowStagePlan, StagedFlowExecutor]:
    graph_nodes = tuple(exported_program.graph.nodes)
    node_ordinal = {node: ordinal for ordinal, node in enumerate(graph_nodes)}
    aliases = _broadcast_getitem_aliases(graph_nodes, node_ordinal)
    artifact = captured.leaf_artifact
    if artifact is None:
        raise WorkGraphError("PG130", "captured flow has no leaf artifact")
    operator_effects = {
        operator.ordinal: operator.effects for operator in artifact.operators
    }
    operator_inputs = {
        operator.ordinal: operator.inputs for operator in artifact.operators
    }

    # Availability level per declared input (declaration order = arrival
    # order); parameters/buffers/constants are level 0. v1 requires every
    # user input to be declared.
    signature = exported_program.graph_signature
    user_input_names = [
        spec.arg.name for spec in signature.input_specs if spec.kind.name == "USER_INPUT"
    ]
    if len(user_input_names) != len(decl.inputs):
        raise WorkGraphError(
            "PG130",
            f"flow {flow!r} export has {len(user_input_names)} user inputs, "
            f"declaration has {len(decl.inputs)}",
        )

    level_by_ordinal: dict[int, int] = {}
    for ordinal, node in enumerate(graph_nodes):
        if ordinal in aliases:
            level_by_ordinal[ordinal] = level_by_ordinal[aliases[ordinal]]
        elif node.op == "placeholder":
            level_by_ordinal[ordinal] = (
                user_input_names.index(node.name) if node.name in user_input_names else 0
            )
        elif node.op == "call_function":
            level_by_ordinal[ordinal] = max(
                (level_by_ordinal[node_ordinal[dep]] for dep in node.all_input_nodes),
                default=0,
            )
        elif node.op == "output":
            continue
        else:
            raise WorkGraphError(
                "PG130", f"flow {flow!r} export contains unsupported node {node.op!r}"
            )

    # Graph outputs (flattened, positional) map to declared output names in
    # return order.
    output_nodes = [node for node in graph_nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise WorkGraphError("PG130", f"flow {flow!r} export has no single output node")
    flat_outputs: list[torch.fx.Node] = []
    _flatten_nodes(output_nodes[0].args, flat_outputs)
    declared_outputs = tuple(name for name, _ref in decl.outputs)
    if len(flat_outputs) != len(declared_outputs):
        raise WorkGraphError(
            "PG130",
            f"flow {flow!r} returns {len(flat_outputs)} values, "
            f"declaration has {len(declared_outputs)} outputs",
        )
    output_levels = tuple(
        level_by_ordinal[node_ordinal[node]] for node in flat_outputs
    )

    # Recompute analysis: an operator consumed by a later stage is recomputed
    # there when its whole chain is pure; otherwise it crosses as an owned
    # value.
    recomputed_into: dict[int, set[int]] = {}
    cross_stage: dict[int, set[int]] = {}
    memo: dict[int, frozenset[int] | None] = {}
    for ordinal, node in enumerate(graph_nodes):
        if node.op != "call_function":
            continue
        stage = level_by_ordinal[ordinal]
        for dep in node.all_input_nodes:
            dep_ordinal = node_ordinal[dep]
            if level_by_ordinal[dep_ordinal] >= stage:
                continue
            chain = _pure_chain(
                dep_ordinal,
                operator_effects=operator_effects,
                operator_inputs=operator_inputs,
                level_by_ordinal=level_by_ordinal,
                stage=stage,
                memo=memo,
            )
            if chain is None:
                cross_stage.setdefault(dep_ordinal, set()).add(stage)
            else:
                recomputed_into.setdefault(stage, set()).update(chain)

    hook_count = max(level_by_ordinal.values(), default=0) + 1
    stages: list[ProgramStage] = []
    for stage_level in range(hook_count):
        placeholders = tuple(
            ordinal
            for ordinal, node in enumerate(graph_nodes)
            if node.op == "placeholder" and level_by_ordinal[ordinal] == stage_level
        )
        owned = tuple(
            ordinal
            for ordinal, node in enumerate(graph_nodes)
            if node.op == "call_function" and level_by_ordinal[ordinal] == stage_level
        )
        recomputed = tuple(sorted(recomputed_into.get(stage_level, set())))
        inputs = tuple(
            f"leaf/{ordinal}@1"
            for ordinal in sorted(cross_stage)
            if stage_level in cross_stage[ordinal]
        )
        leaving = tuple(
            ordinal
            for ordinal in sorted(cross_stage)
            if level_by_ordinal[ordinal] == stage_level
        )
        flow_outputs = tuple(
            declared_outputs[position]
            for position, level in enumerate(output_levels)
            if level == stage_level
        )
        output_ordinals = tuple(
            node_ordinal[flat_outputs[position]]
            for position, level in enumerate(output_levels)
            if level == stage_level
        )
        stages.append(
            ProgramStage(
                flow=flow,
                flow_ordinal=flow_ordinal,
                stage_ordinal=stage_level,
                hook_level=stage_level,
                placeholder_ordinals=placeholders,
                operator_ordinals=owned,
                recomputed_ordinals=recomputed,
                input_values=inputs,
                output_values=tuple(
                    dict.fromkeys(
                        (
                            *(f"leaf/{ordinal}@1" for ordinal in output_ordinals),
                            *(f"leaf/{ordinal}@1" for ordinal in leaving),
                        )
                    )
                ),
                flow_outputs=flow_outputs,
            )
        )

    cross_values = tuple(
        StageValueDesc(
            value_id=f"leaf/{ordinal}@1",
            producer_stage=level_by_ordinal[ordinal],
            consumer_stages=tuple(sorted(cross_stage[ordinal])),
        )
        for ordinal in sorted(cross_stage)
    )
    plan = FlowStagePlan(
        flow=flow,
        flow_ordinal=flow_ordinal,
        artifact_key=captured.artifact_key,
        shape_guard=captured.shape_guard,
        output_names=declared_outputs,
        stages=tuple(stages),
        cross_stage_values=cross_values,
        hook_count=hook_count,
    )
    return plan, StagedFlowExecutor(plan, exported_program, instance, aliases)


__all__ = [
    "FlowStagePlan",
    "ProgramStage",
    "StageValueDesc",
    "StagedFlowExecutor",
    "slice_flow_stages",
]

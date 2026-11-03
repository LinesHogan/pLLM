"""Model-side tensor effects: parameter overlays, imported overlays, activation patches.

Module functions declare effects that run inside or alongside the model forward and their
request-state transitions on the TensorBinding supplied by workload authoring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch

from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    ActivationPatchOperator,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
    ModelProbe,
)
from pllm.contracts.codec import boundary_digest
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.host_batch import StateImport
from pllm.contracts.parameter_overlay import (
    FieldView,
    ProjectionDelta,
)
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    MathContract,
    PackedRowsInputSource,
    SemanticEffect,
    SemanticInput,
    SemanticTask,
    StateEffectKind,
    StateScope,
    TaskResultInputSource,
    TaskSymbol,
    TokenProgramEffect,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.contracts.visibility import VisibilityKind, VisibilitySource
from pllm.execution.binding_base import Field, overlay_fields
from pllm.execution.custom_forward import CUSTOM_FORWARD_IMPLEMENTATION
from pllm.execution.declaration import Call, State, When, native
from pllm.execution.model_sites import PatchInput, ResidualRewrite, activation_write
from pllm.execution.overlay_task import ChunkUpdate, GradientUpdate, OverlayTransition, overlay_apply
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.tensor_frame import (
    TensorFrame,
    Tensors,
    UpdateOutput,
)
from pllm.execution.tensor_program import TaskOutput, TensorBatch
from pllm.execution.tensor_task import FieldWriter, TensorTask
from pllm.execution.token_program import ApplyUpdates, Cell, Layers, Over, Segment, TokenProgram, Tokens, TokenTensor

if TYPE_CHECKING:
    from pllm.execution.tensor_binding import TensorBinding


def _bind_visibility(b: TensorBinding, call: Call, source: VisibilitySource) -> None:
    """Attach the read domain; plan validation derives and checks its relation once."""
    b.update_task(call.symbol, lambda task: replace(task, visibility=source))


def declare_parameter_overlay(
    b: TensorBinding,
    name: str,
    effect: ProjectionDelta,
    fields: Mapping[str, Field],
    *,
    initialize: Callable[[Tensors, torch.Generator], None] | None,
    keys: tuple[RequestKeyField, ...],
    phases: tuple[HiddenPhase, ...],
    identity: str,
    function: Callable[..., UpdateOutput],
    arguments: Mapping[str, str],
    visibility: VisibilityKind,
    equations_digest: str,
    invariants: Mapping[str, tuple[str, tuple[FieldView, ...]]],
    reads: tuple[State, ...],
) -> None:
    """Declare a parameter overlay and its request-state transition."""
    if b.host:
        raise ValueError("@pllm.train_in_forward trains inside a model forward; compile this workload with make_llm")
    if b.effects.in_forward is not None:
        raise ValueError(
            f"declare one @pllm.train_in_forward method per workload; {b.effects.in_forward} already is one"
        )
    layout = overlay_fields(
        fields,
        effect,
        b.dimensions,
        lambda key, value: (
            b.dimensions.dtype if value.dtype is None else TensorDType(str(value.dtype).removeprefix("torch."))
        ),
    )
    targets = effect.targets
    b.overlay_targets = targets
    b.parameter_phases = phases
    state = b.state(
        effect.state,
        layout,
        scope=StateScope.REQUEST,
        keys=keys,
    )
    if initialize is not None:

        def init(root: tuple[torch.Tensor, ...], generator: torch.Generator) -> None:
            """Initialize each target's copy of the fields; they are laid out target by target."""
            for start in range(0, len(root), len(fields)):
                initialize(dict(zip(fields, root[start : start + len(fields)], strict=True)), generator)

        b.initializers[str(state.declaration.resource)] = init

    apply = partial(overlay_apply, effect)

    def select(data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
        active = (
            {row.identity for row in data.step.rows.rows if row.phase in phases}
            if visibility == "exact"
            else {operand.identity for operand in data.step.operands if operand.phase in phases}
        )
        return tuple(
            sorted(
                {
                    index
                    for row, index in zip(
                        selection.rows, selection.for_resource(state.declaration.resource).state_indices, strict=True
                    )
                    if row.identity in active
                }
            )
        )

    source = VisibilitySource(
        visibility,
        "overlay",
        equations_digest,
        (str(state.declaration.resource),) * len(targets),
        0,
        "carry" if visibility == "exact" else "transition",
    )
    math = OverlayTransition(
        tuple(fields),
        targets,
        phases,
        apply,
        function,
        arguments,
        source,
        TaskSymbol(name),
        effect.operator,
        state.declaration.resource,
        invariants,
        function if isinstance(function, (ChunkUpdate, GradientUpdate)) else None,
    )
    b.overlay_math = math
    transition = b.compute(
        name,
        inputs={"operands": b.input("parameter.operands", "dense_pairs", "dense_operand_pairs@1")},
        outputs=("transition",),
        reads=tuple(read.version() for read in reads),
        changes=state,
        effect=StateEffectKind.TRANSITION_MAYBE,
        when=When.IN_FORWARD,
        function=math,
        select=select,
        implementations=(
            native(identity + ":cpu", "compute/cpu@1", "transition/full_root@1"),
            native(identity + ":cuda", "compute/cuda@1", "state/fenced_reclaim@1", "transition/full_root@1"),
        ),
    )
    _bind_visibility(b, transition, source)
    if visibility == "exact":
        b.operations[transition.symbol] = replace(
            b.operations[transition.symbol], version="private", select_fields=math.written_fields
        )
    b.publish(
        transition,
        state,
        implementations=(
            native(
                "state:cuda_publish_parameter_overlay@1",
                "compute/cuda@1",
                "state/fenced_reclaim@1",
                "transition/full_root@1",
            ),
            native("state:publish_parameter_overlay@1", "compute/cpu@1", "transition/full_root@1"),
        ),
    )
    views = effect.field_bindings(
        lambda name: state.view(name, b.dimensions), reads_weight="weight" in arguments.values()
    )
    b.overlay(views, transition)


def declare_site_memory(b: TensorBinding, name: str, state: State) -> None:
    """A stateful custom residual's transition: before the forward it copies each request's memory into a private
    write, which the method's hook advances chunk by chunk; the publication makes it visible after the forward."""
    transition = b.compute(
        f"{name}.memory",
        inputs={"memory": state.read()},
        outputs=("transition",),
        changes=state,
        effect=StateEffectKind.TRANSITION_MAYBE,
        when=When.BEFORE_FORWARD,
        function=_open_memory,
        implementations=(native(f"site.memory:{name}@1", "transition/full_root@1"),),
    )
    b.publish(transition, state)


def _open_memory(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
    """The private write itself is the transition; the forward's site hook fills it."""
    return TaskOutput()


def write_result(b: TensorBinding, source: Call, state: State, fields: tuple[str, ...], when: When) -> Call:
    """Bind a computed tuple to one atomic private transition."""
    writer = FieldWriter(
        source,
        tuple(b.field_views[state.declaration.resource, key] for key in fields),
        state.declaration.resource,
        b.tensor_tasks[source.symbol].phase,
    )
    update = b.compute(
        f"{source.symbol}.write",
        inputs={name: source[name] for name in source.outputs},
        outputs=("updated",),
        changes=state,
        effect=StateEffectKind.TRANSITION_MAYBE,
        when=when,
        implementations=(native(f"tensor.write:{source.symbol}@1", "transition/full_root@1"),),
    )

    _write_task(b, update, writer)
    return update


def _write_task(b: TensorBinding, update: Call, writer: FieldWriter) -> None:
    b.tensor_tasks[update.symbol] = TensorTask(
        update.symbol, lambda: (), {}, {}, MathContract(), (), {}, writer.phase, writer.resource, writer
    )


def import_update(b: TensorBinding, state: State, identity: str) -> Call:
    """One BEFORE_FORWARD import of externally produced field values into private roots."""
    writer = FieldWriter(None, b.imports[str(state.declaration.resource)], state.declaration.resource, None)
    update = b.compute(
        "update",
        inputs={"parameters": b.input("input", "state", StateImport.schema)},
        outputs=("updated",),
        changes=state,
        effect=StateEffectKind.TRANSITION_MAYBE,
        when=When.BEFORE_FORWARD,
        implementations=(native(identity, "transition/full_root@1"),),
    )
    _write_task(b, update, writer)
    return update


def declare_activation_patch(
    b: TensorBinding,
    name: str,
    state: State,
    *,
    layers: tuple[int, ...],
    operator: ActivationPatchOperator,
    first_tokens: int,
    last_tokens: int,
    decode: DecodeWindow = DecodeWindow.NONE,
    decode_refresh: DecodeRefresh = DecodeRefresh.INHERIT,
    on_missing: MissingPolicy = MissingPolicy.STOP,
) -> None:
    """Read logical state views at activation sites."""
    width = 1 if operator is ActivationPatchOperator.ADD else 3
    views = tuple(view for (resource, _), view in b.field_views.items() if resource == state.declaration.resource)
    if (
        not layers
        or tuple(sorted(set(layers))) != layers
        or layers[0] < 0
        or min(first_tokens, last_tokens) < 0
        or len(views) != len(layers) * width
    ):
        raise ValueError("activation fields must cover each ordered layer and operator")
    for start in range(0, len(views), width):
        _check_activation_fields(b, state, operator, views[start : start + width])
    # The patch owns each layer's MLP residual update: another custom residual there is a conflict.
    b.sites.add(
        ResidualRewrite(
            name,
            "mlp",
            tuple(("mlp", layer) for layer in layers),
            activation_write,
            (("patch", PatchInput(state.declaration.resource)),),
        )
    )
    b.activation_patch(
        tuple(
            ActivationPatchBinding(
                state.declaration.resource,
                layer,
                views[index * width : (index + 1) * width],
                first_tokens,
                last_tokens,
                operator,
                decode=decode,
                decode_refresh=decode_refresh,
                on_missing=on_missing,
            )
            for index, layer in enumerate(layers)
        )
    )


def _check_activation_fields(
    b: TensorBinding, state: State, operator: ActivationPatchOperator, views: tuple[FieldView, ...]
) -> None:
    """Reject one layer's activation fields the operator cannot apply, naming the fields to fix."""
    hidden, names = b.dimensions.hidden_size, tuple(state.fields[view.field].name for view in views)
    shapes = tuple(view.shape for view in views)
    if operator is ActivationPatchOperator.ADD:
        rule, valid = f"({hidden},)", shapes == ((hidden,),)
        allowed = (b.dimensions.dtype, TensorDType.FLOAT32, TensorDType.FLOAT64)
    else:
        rank = shapes[0][0] if shapes[0] else 0
        rule = f"(rank, {hidden}), (rank,) and ({hidden}, rank) with 0 < rank <= {hidden}"
        valid = 0 < rank <= hidden and shapes == ((rank, hidden), (rank,), (hidden, rank))
        allowed = (b.dimensions.dtype,)
    if not valid:
        raise ValueError(f"activation fields {names} must be shaped {rule} for {operator}; got {shapes}")
    (layout,) = (entry.layout for entry in b.layouts.layouts if entry.resource == state.declaration.resource)
    for name, view in zip(names, views, strict=True):
        if layout.fields[view.field].dtype not in allowed:
            raise ValueError(
                f"activation field {name!r} must hold one of {allowed} for {operator}; set dtype=torch.{allowed[0]}"
            )


def declare_custom_forward(
    b: TensorBinding,
    forwards: tuple[tuple[Tokens, Over, Callable[[Layers], torch.Tensor]], ...],
    position_states: tuple[tuple[str, float], ...],
    publications: tuple[Call, ...],
) -> None:
    """Trace the workload's @pllm.custom_forward methods and sign their complete model/tensor source before planning.

    The forward reads every version in `publications`: the writes of states its activation patches read.
    """
    tokens: Tokens = forwards[0][0] if len(forwards) == 1 else "all"
    composed = b.effects.in_forward is not None
    _check_custom_forward(b)
    if position_states and (composed or b.effects.activation_bindings):
        raise ValueError("PositionState needs a @pllm.custom_forward without train_in_forward or an activation patch")
    b.position_states = position_states
    symbol = _attach_custom_forward(b, composed, TokenProgramEffect(b.dimensions.layer_count, tokens, position_states))
    if publications:
        b.update_task(symbol, partial(_read_publications, b, publications))
    probe = b.probe
    assert probe is not None and probe.causal_domain is not None
    captured = _trace_programs(probe, forwards, position_states)
    programs = {
        role: program for selected, program in captured for role in ("prefill", "decode") if selected in (role, "all")
    }
    _check_adaptive(b, captured)
    _count_program_work(b, tuple(program for _, program in captured))
    applies_updates = any(isinstance(step, ApplyUpdates) for _, program in captured for step in program.steps)
    if composed:
        select = partial(_program_states, b.parameter_phases, b.custom_forward_updates, tokens)
        b.operations[symbol] = replace(b.operations[symbol], select=select)
    elif applies_updates:
        raise ValueError("apply_updates needs one trained parameter overlay on the same workload")
    b.custom_forward_programs = programs
    program_effect = TokenProgramEffect(
        probe.causal_domain.layers,
        tokens,
        position_states,
        boundary_digest(
            tuple(
                (
                    role,
                    program.over,
                    tuple(str(step.graph.graph) if isinstance(step, Cell) else step for step in program.steps),
                )
                for role, program in programs.items()
            )
        ),
        len(forwards) > 1 or any(program.role_sensitive for program in programs.values()),
        applies_updates,
        any(program.over == "span" for program in programs.values()),
        any(program.adaptive for program in programs.values()),
    )
    b.update_task(symbol, lambda task: replace(task, custom_forward=program_effect))


def _check_adaptive(b: TensorBinding, programs: tuple[tuple[Tokens, TokenProgram], ...]) -> None:
    """Reject declarations whose row or layer dependencies adaptive decode cannot yet replay."""
    for tokens, program in programs:
        if not program.adaptive:
            continue
        if tokens != "decode":
            raise ValueError(
                'active= currently needs @pllm.custom_forward(tokens="decode"); keep prompt layers unmasked'
            )
        for step in program.steps:
            if not isinstance(step, Segment) or step.active is None:
                continue
            for target in b.overlay_targets:
                if target.layer is not None and step.start <= target.layer < step.stop:
                    raise ValueError(
                        f"train_in_forward targets layer {target.layer}, which tokens skip; target layers below {step.start} or remove active="
                    )
            for rewrite in b.sites.rewrites:
                if rewrite.memory is not None and any(step.start <= layer < step.stop for _, layer in rewrite.places):
                    raise ValueError(
                        f"{rewrite.name} has state on layers tokens skip; move it below {step.start} or remove active="
                    )
                if any(step.start <= source.layer < step.stop for source in rewrite.earlier):
                    raise ValueError(f"{rewrite.name} reads a layer tokens skip; read layers below {step.start}")
        committed = {
            layer
            for step in program.steps
            if isinstance(step, Segment) and step.kv == "commit"
            for layer in range(step.start, step.stop)
        }
        missing = set(range(b.dimensions.layer_count)) - committed
        if missing:
            raise ValueError(f"adaptive forward leaves KV holes at layers {sorted(missing)}; commit every layer")


def _check_custom_forward(b: TensorBinding) -> None:
    """Reject effects whose backend hooks index the forward's packed rows, which a custom forward replaces with its
    own row waves and layer visits."""
    for task in b.graph.tasks:
        if any(isinstance(port.source, HiddenInputSource) for port in task.inputs):
            raise ValueError(
                f"{task.symbol} reads hidden rows (pllm.hidden, pllm.residual, pllm.layer_logits or "
                "pllm.final_hidden), which a @pllm.custom_forward does not produce: it may run a layer several "
                "times or not at all per token; move this method to a workload without the custom forward"
            )
    adapters = {binding.resource for binding in b.effects.overlay_bindings} - {
        str(effect.resource) for task in b.graph.tasks if task.symbol == b.effects.in_forward for effect in task.effects
    }
    if adapters:
        raise ValueError(
            f"pllm.model.Adapter reading {sorted(adapters)} corrects projections for the model's own packed rows; "
            "inside a @pllm.custom_forward only the @pllm.train_in_forward state is applied, so train the adapter "
            "with train_in_forward or drop the custom forward"
        )


def _read_publications(b: TensorBinding, publications: tuple[Call, ...], task: SemanticTask) -> SemanticTask:
    """Make the custom forward read the published versions of the states its activation patches read."""
    published = tuple(next(item for item in b.graph.tasks if item.symbol == call.symbol) for call in publications)
    return replace(
        task,
        inputs=(
            *task.inputs,
            *(
                SemanticInput(f"published_{index}", TaskResultInputSource(item.symbol, item.outputs[0].name))
                for index, item in enumerate(published)
            ),
        ),
        effects=(
            *task.effects,
            *(SemanticEffect(item.effects[-1].resource, StateEffectKind.READ, item.symbol) for item in published),
        ),
    )


def _attach_custom_forward(b: TensorBinding, composed: bool, effect: TokenProgramEffect) -> TaskSymbol:
    """Run the custom forward as the trained overlay's transition, or as a standalone forward; return its task."""
    if not composed:
        forward = b.compute(
            "forward",
            inputs={"rows": b.rows()},
            outputs=("normal_hidden",),
            when=When.IN_FORWARD,
            custom_forward=effect,
            implementations=(native(CUSTOM_FORWARD_IMPLEMENTATION, "model/custom_forward@1"),),
        )
        b.custom_forward(forward.symbol)
        return forward.symbol
    symbol = b.effects.in_forward
    assert symbol is not None
    b.custom_forward(symbol)
    b.update_task(
        symbol,
        lambda task: replace(
            task,
            custom_forward=effect,
            inputs=(*task.inputs, SemanticInput("rows", PackedRowsInputSource())),
            implementations=tuple(
                replace(
                    implementation,
                    required_capabilities=tuple(
                        sorted((*implementation.required_capabilities, "model/custom_forward@1"))
                    ),
                )
                for implementation in task.implementations
            ),
        ),
    )
    b.operations[symbol] = replace(b.operations[symbol], version="private")
    return symbol


def _trace_programs(
    probe: ModelProbe,
    forwards: tuple[tuple[Tokens, Over, Callable[[Layers], torch.Tensor]], ...],
    position_states: tuple[tuple[str, float], ...],
) -> tuple[tuple[Tokens, TokenProgram], ...]:
    """Trace each @pllm.custom_forward method into a token program over the model's layers."""
    assert probe.causal_domain is not None
    tensor = TokenTensor(
        (1, probe.hidden_size),
        probe.tensor_dtype.torch,
        probe.device_kind.value if probe.device_index is None else f"{probe.device_kind.value}:{probe.device_index}",
    )
    names = tuple(name for name, _ in position_states)
    return tuple(
        (selected, TokenProgram.trace(function, tensor, probe.causal_domain.layers, names, over))
        for selected, over, function in forwards
    )


def _count_program_work(b: TensorBinding, programs: tuple[TokenProgram, ...]) -> None:
    """Fund packed rows for each model segment, and note whether apply_updates follows a targeted layer."""
    assert b.probe is not None
    segments = sum(isinstance(step, Segment) for program in programs for step in program.steps)
    b.custom_forward_rows = b.probe.max_packed_rows * max(1, segments)
    for program in programs:
        pending = False
        for step in program.steps:
            if isinstance(step, Segment):
                pending |= any(
                    target.layer is not None and step.start <= target.layer < step.stop for target in b.overlay_targets
                )
            elif isinstance(step, ApplyUpdates):
                b.custom_forward_updates |= pending
                pending = False


def _program_states(
    phases: tuple[HiddenPhase, ...],
    updates: bool,
    tokens: Tokens,
    data: TensorFrame,
    selection: RequestStateSelection,
) -> tuple[int, ...]:
    """The overlay states a custom-forward step transitions: rows in an overlay phase that the program does not skip."""
    return tuple(
        sorted(
            {
                index
                for row, index in zip(selection.rows, selection.state_indices, strict=True)
                if row.phase in phases and (updates or not (tokens == "all" or row.phase.value == tokens))
            }
        )
    )

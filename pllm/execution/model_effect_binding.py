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

from pllm.contracts.auxiliary import StateTensorUpdate
from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    ActivationPatchOperator,
    AttachmentKind,
    AttachmentTaskRole,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.codec import boundary_digest
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    ParameterEffect,
    ParameterOverlayFieldBinding,
)
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.semantic_graph import (
    MathContract,
    PackedRowsInputSource,
    SemanticInput,
    StateEffectKind,
    StateScope,
    TaskSymbol,
    TokenProgramEffect,
)
from pllm.contracts.visibility import Exact, Visibility, VisibilitySource
from pllm.execution.binding_base import Buffer, overlay_fields
from pllm.execution.declaration import Call, State, When, native
from pllm.execution.overlay_task import ChunkUpdate, GradientUpdate, OverlayTransition, overlay_apply
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.replay import TOKENWISE_REPLAY_IMPLEMENTATION
from pllm.execution.tensor_frame import (
    TensorFrame,
    Tensors,
    TensorUpdate,
)
from pllm.execution.tensor_task import FieldWriter, TensorTask
from pllm.execution.token_program import ApplyUpdates, Cell, Layers, Over, Segment, TokenProgram, Tokens, TokenTensor

if TYPE_CHECKING:
    from pllm.execution.tensor_binding import TensorBinding


def _bind_visibility(b: TensorBinding, call: Call, source: VisibilitySource) -> None:
    """Attach the read domain; plan validation derives and checks its relation once."""
    b.graph = replace(
        b.graph,
        tasks=tuple(replace(task, visibility=source) if task.symbol == call.symbol else task for task in b.graph.tasks),
    )


def declare_parameter_overlay(
    b: TensorBinding,
    name: str,
    effect: ParameterEffect,
    buffers: Mapping[str, Buffer],
    *,
    initialize: Callable[[Tensors, torch.Generator], None] | None,
    namespace: str,
    keys: tuple[RequestKeyField, ...],
    phases: tuple[HiddenPhase, ...],
    identity: str,
    function: Callable[..., TensorUpdate],
    arguments: Mapping[str, str],
    visibility: Visibility,
    equations_digest: str,
) -> None:
    """Declare a parameter overlay and its request-state transition."""
    if b.graph.tasks or b.graph.states or b.host:
        raise ValueError("ordinary overlay requires a sole request state and model transition")
    fields = overlay_fields(buffers, effect)
    targets, applied = effect.targets, effect.applied_fields
    dimensions = b.dimensions
    if len(dimensions.target_shapes) != len(targets):
        raise ValueError("overlay targets differ from backend specialization")
    b.overlay_targets = targets
    b.parameter_phases = phases
    state = b.state(
        effect.state,
        fields,
        namespace=namespace,
        scope=StateScope.REQUEST,
        keys=keys,
    )
    if initialize is not None:

        def init(root: tuple[torch.Tensor, ...], generator: torch.Generator) -> None:
            """Initialize each target's buffers; fields are laid out target by target."""
            for start in range(0, len(root), len(buffers)):
                initialize(dict(zip(buffers, root[start : start + len(buffers)], strict=True)), generator)

        b.initializers[str(state.declaration.resource)] = init

    apply = partial(overlay_apply, effect)

    def select(data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
        active = {operand.identity for operand in data.step.operands if operand.phase in phases}
        return tuple(
            sorted(
                {
                    index
                    for row, index in zip(selection.rows, selection.state_indices, strict=True)
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
        "carry" if visibility == Exact() else "transition",
    )
    math = OverlayTransition(
        tuple(buffers),
        targets,
        phases,
        apply,
        function,
        arguments,
        source,
        TaskSymbol(name),
        effect.operator,
        function if isinstance(function, (ChunkUpdate, GradientUpdate)) else None,
    )
    b.overlay_math = math
    transition = b.compute(
        name,
        inputs={"operands": b.input("parameter.operands", "dense_pairs", "canonical.dense_operand_pairs@1")},
        outputs=("transition",),
        changes=state,
        effect=StateEffectKind.TRANSITION_MAYBE,
        when=When.MODEL_UPDATE,
        function=math,
        select=select,
        implementations=(
            native(identity + ":cpu", "compute/cpu@1", "transition/full_root@1"),
            native(identity + ":cuda", "compute/cuda@1", "state/fenced_reclaim@1", "transition/full_root@1"),
        ),
    )
    _bind_visibility(b, transition, source)
    if visibility == Exact():
        b.operations[transition.symbol] = replace(b.operations[transition.symbol], version="private")
    publication = b.publish(
        transition,
        state,
        implementations=(
            native(
                "canonical.state:cuda_publish_parameter_overlay@1",
                "compute/cuda@1",
                "state/fenced_reclaim@1",
                "transition/full_root@1",
            ),
            native("canonical.state:publish_parameter_overlay@1", "compute/cpu@1", "transition/full_root@1"),
        ),
    )
    b.overlay(
        state,
        (transition, publication),
        tuple(
            ParameterOverlayFieldBinding(
                target,
                effect.operator,
                tuple(state.view(f"{key}_{i}", b.dimensions) for key in applied),
                str(state.declaration.resource),
                effect.scale,
            )
            for i, target in enumerate(targets)
        ),
    )


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
    """One BEFORE_MODEL import of externally produced field values into private roots."""
    writer = FieldWriter(None, b.imports[str(state.declaration.resource)], state.declaration.resource, None)
    update = b.compute(
        "update",
        inputs={"parameters": b.input("input", "state", StateTensorUpdate.schema)},
        outputs=("updated",),
        changes=state,
        effect=StateEffectKind.TRANSITION_MAYBE,
        when=When.BEFORE_MODEL,
        implementations=(native(identity, "transition/full_root@1"),),
    )
    _write_task(b, update, writer)
    return update


def declare_activation_patch(
    b: TensorBinding,
    state: State,
    producer: tuple[Call, Call] | None,
    *,
    layers: tuple[int, ...],
    operator: ActivationPatchOperator,
    first_tokens: int,
    last_tokens: int,
    decode: DecodeWindow = DecodeWindow.NONE,
    decode_refresh: DecodeRefresh = DecodeRefresh.INHERIT,
    on_missing: MissingPolicy = MissingPolicy.STOP,
) -> None:
    """Read logical state views at activation sites; replay/HOLD also bind their publication dependency."""
    width = 1 if operator is ActivationPatchOperator.ADD else 3
    views = tuple(view for (resource, _), view in b.field_views.items() if resource == state.declaration.resource)
    if (
        b.host
        or not layers
        or tuple(sorted(set(layers))) != layers
        or layers[0] < 0
        or min(first_tokens, last_tokens) < 0
        or len(views) != len(layers) * width
    ):
        raise ValueError("activation fields must cover each ordered layer and operator")
    b.overlay(
        state,
        producer,
        activations=tuple(
            ActivationPatchBinding(
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
        ),
    )


def declare_layer_route(
    b: TensorBinding,
    routes: tuple[tuple[Tokens, Over, Callable[[Layers], torch.Tensor]], ...],
    position_states: tuple[tuple[str, float], ...] = (),
) -> None:
    """Trace a user route and sign its complete model/tensor source before planning."""
    tokens: Tokens = routes[0][0] if len(routes) == 1 else "all"
    activation = b.attachment is not None and b.attachment.kind is AttachmentKind.ACTIVATION_PATCH
    composed = (
        b.attachment is not None
        and b.attachment.kind is AttachmentKind.PARAMETER_OVERLAY
        and b.overlay_math is not None
    )
    if b.host or (not activation and not composed and (b.attachment is not None or b.graph.tasks or b.graph.states)):
        raise ValueError("layer_route requires a standalone effect, Activation or one trained parameter overlay")
    if position_states and (activation or composed):
        raise ValueError("use PositionState on a pure route without learned overlays or activations")
    b.position_states = position_states
    effect = TokenProgramEffect(b.dimensions.layer_count, tokens, position_states)
    if composed:
        assert b.attachment is not None
        symbol = next(symbol for role, symbol in b.attachment.tasks if role is AttachmentTaskRole.TRANSITION)
        b.graph = replace(
            b.graph,
            tasks=tuple(
                replace(
                    task,
                    replay=effect,
                    inputs=(*task.inputs, SemanticInput("rows", PackedRowsInputSource())),
                    implementations=tuple(
                        replace(
                            implementation,
                            required_capabilities=tuple(
                                sorted((*implementation.required_capabilities, "model/tokenwise_replay@1"))
                            ),
                        )
                        for implementation in task.implementations
                    ),
                )
                if task.symbol == symbol
                else task
                for task in b.graph.tasks
            ),
        )
        b.operations[symbol] = replace(b.operations[symbol], version="private")
    else:
        forward = b.compute(
            "forward",
            inputs={"rows": b.rows()},
            outputs=("normal_hidden",),
            when=When.MODEL_UPDATE,
            replay=effect,
            implementations=(native(TOKENWISE_REPLAY_IMPLEMENTATION, "model/tokenwise_replay@1"),),
        )
        symbol = forward.symbol
        b.model_effect(forward, AttachmentKind.MODEL_REPLAY)

    if b.probe is not None:
        probe = b.probe
        assert b.attachment is not None
        assert probe.causal_domain is not None
        tensor = TokenTensor(
            (1, probe.hidden_size),
            probe.tensor_dtype.torch,
            "cpu" if probe.device_kind.value == "cpu" else f"cuda:{probe.device_ordinal}",
        )
        captured = tuple(
            (
                selected,
                TokenProgram.capture(
                    function, tensor, probe.causal_domain.layers, tuple(name for name, _ in position_states), over
                ),
            )
            for selected, over, function in routes
        )
        programs = {
            role: program
            for selected, program in captured
            for role in ("prefill", "decode")
            if selected in (role, "all")
        }
        steps = tuple(step for _, program in captured for step in program.steps)
        b.route_rows = probe.max_packed_rows * max(1, sum(isinstance(step, Segment) for step in steps))
        for _, program in captured:
            pending = False
            for step in program.steps:
                if isinstance(step, Segment):
                    pending |= any(
                        target.layer is not None and step.start <= target.layer < step.stop
                        for target in b.overlay_targets
                    )
                elif isinstance(step, ApplyUpdates):
                    b.route_updates |= pending
                    pending = False
        if composed:
            phases, updates = b.parameter_phases, b.route_updates

            def select(data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
                return tuple(
                    sorted(
                        {
                            index
                            for row, index in zip(selection.rows, selection.state_indices, strict=True)
                            if row.phase in phases and (updates or not (tokens == "all" or row.phase.value == tokens))
                        }
                    )
                )

            b.operations[symbol] = replace(b.operations[symbol], select=select)
        if any(isinstance(step, ApplyUpdates) for step in steps) and not composed:
            raise ValueError("apply_updates needs one trained parameter overlay on the same workload")
        b.route_program = programs
        b.graph = replace(
            b.graph,
            tasks=tuple(
                replace(
                    task,
                    replay=TokenProgramEffect(
                        probe.causal_domain.layers,
                        tokens,
                        position_states,
                        boundary_digest(
                            tuple(
                                (
                                    role,
                                    program.over,
                                    tuple(
                                        str(step.graph.graph) if isinstance(step, Cell) else step
                                        for step in program.steps
                                    ),
                                )
                                for role, program in programs.items()
                            )
                        ),
                        len(routes) > 1 or any(program.role_sensitive for program in programs.values()),
                        any(isinstance(step, ApplyUpdates) for step in steps),
                        any(program.over == "span" for program in programs.values()),
                    ),
                )
                if task.symbol == symbol
                else task
                for task in b.graph.tasks
            ),
        )

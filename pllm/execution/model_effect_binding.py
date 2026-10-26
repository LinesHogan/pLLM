"""Model-side tensor effects: parameter overlays, imported overlays, activation patches.

ModelEffectBinding declares effects that run inside or alongside the model forward and their
request-state transitions. Used by TensorBinding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import isfinite
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
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    FieldView,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterTarget,
)
from pllm.contracts.region_program import TensorType
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.semantic_graph import (
    PackedRowsInputSource,
    SemanticInput,
    StateEffectKind,
    StateResource,
    StateScope,
    TokenProgramEffect,
)
from pllm.contracts.visibility import Exact, Visibility, VisibilitySource
from pllm.execution.codec import boundary_digest
from pllm.execution.declaration import Call, State, When, native
from pllm.execution.overlay_task import OverlayTransition
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.replay import TOKENWISE_REPLAY_IMPLEMENTATION
from pllm.execution.tensor_frame import (
    TensorFrame,
    Tensors,
    TensorUpdate,
    finite,
)
from pllm.execution.tensor_program import (
    TaskOutput,
    TensorBatch,
)
from pllm.execution.token_program import ApplyUpdates, Cell, Layers, Over, Segment, TokenProgram, Tokens
from pllm.kernels.head_overlay import head_correction

if TYPE_CHECKING:
    pass


from pllm.execution.binding_base import Buffer, TensorBindingBase, buffer_field


@dataclass(frozen=True)
class FieldWriter:
    """Write a task result into its declared field; used by the write lowering."""

    source: Call
    fields: tuple[FieldView, ...]
    resource: StateResource
    phase: HiddenPhase | None

    def __call__(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        selected = self.select(data, batch.selection)
        if not selected:
            return TaskOutput()
        instance = batch.selection.for_resource(self.resource).keys == (None,)
        values = tuple(batch.results[self.source.symbol, name] for name in self.source.outputs)
        for view, value in zip(self.fields, values, strict=True):
            target = view.read(batch.roots[selected[0]][view.field])
            if (
                value.shape != ((*(() if instance else (len(selected),)), *view.shape))
                or value.dtype != target.dtype
                or value.device != target.device
                or not finite(value)
            ):
                raise ValueError(
                    f"write to {self.resource} needs finite fields with their declared shape, dtype, device and request axis"
                )
        return TaskOutput(
            field_writes=tuple(
                (view, value.unsqueeze(0) if instance else value)
                for view, value in zip(self.fields, values, strict=True)
            ),
            changed=selected,
            update_count=len(selected),
        )

    def select(self, data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
        """Select only this producer's eligible resource rows."""
        group = selection.for_resource(self.resource)
        if self.phase is not None and not data.step.rows.producer_rows(self.phase):
            return ()
        if group.keys == (None,):
            return tuple(selection.refs.index(ref) for ref in group.refs)
        indices = {row.identity: index for row, index in zip(selection.rows, group.state_indices, strict=True)}
        return tuple(dict.fromkeys(indices[row.identity] for row in data.step.rows.producer_rows(self.phase)))


def check_overlay_fields(
    buffers: Mapping[str, Buffer],
    applied: tuple[str, ...],
    targets: tuple[ParameterTarget, ...],
    operator: ParameterOverlayOperator,
    scale: float,
) -> None:
    arity = 2 if operator is ParameterOverlayOperator.LOW_RANK_ADD else 1
    if (
        not targets
        or tuple(sorted(set(targets))) != targets
        or len(applied) != arity
        or len(set(applied)) != arity
        or any(key not in buffers for key in applied)
        or not isfinite(scale)
    ):
        raise ValueError("overlay requires ordered unique targets, exact operator fields and a finite scale")


class ModelEffectBinding(TensorBindingBase):
    """Declare model-side effects and their request-state transitions."""

    position_states: tuple[tuple[str, float], ...] = ()

    def _bind_visibility(self, call: Call, source: VisibilitySource) -> None:
        """Attach the read domain; plan validation derives and checks its relation once."""
        self.graph = replace(
            self.graph,
            tasks=tuple(
                replace(task, visibility=source) if task.symbol == call.symbol else task for task in self.graph.tasks
            ),
        )

    def parameter_overlay(
        self,
        name: str,
        state_name: str,
        buffers: Mapping[str, Buffer],
        *,
        initialize: Callable[[Tensors, torch.Generator], None] | None,
        targets: tuple[ParameterTarget, ...],
        applied: tuple[str, ...],
        namespace: str,
        keys: tuple[RequestKeyField, ...],
        phases: tuple[HiddenPhase, ...],
        identity: str,
        operator: ParameterOverlayOperator,
        scale: float,
        function: Callable[[Tensors], TensorUpdate],
        visibility: Visibility,
        equations_digest: str,
    ) -> None:
        if self.graph.tasks or self.graph.states or self.host:
            raise ValueError("ordinary overlay requires a sole request state and model transition")
        check_overlay_fields(buffers, applied, targets, operator, scale)
        dimensions = self.dimensions
        if dimensions is not None:
            if len(dimensions.target_shapes) != len(targets):
                raise ValueError("overlay targets differ from backend specialization")
        self.overlay_targets = targets
        self.parameter_phases = phases
        state = self.state(
            state_name,
            tuple(
                buffer_field(f"{key}_{index}", buffer, lambda m, i=index: m.target_shapes[i])
                for index in range(len(targets))
                for key, buffer in buffers.items()
            ),
            namespace=namespace,
            scope=StateScope.REQUEST,
            keys=keys,
        )
        if initialize is not None:

            def init(root: tuple[torch.Tensor, ...], generator: torch.Generator) -> None:
                """Initialize each target's buffers; fields are laid out target by target."""
                for start in range(0, len(root), len(buffers)):
                    initialize(dict(zip(buffers, root[start : start + len(buffers)], strict=True)), generator)

            self.initializers[str(state.declaration.resource)] = init

        def apply(hidden: torch.Tensor, values: Tensors) -> torch.Tensor:
            if operator is ParameterOverlayOperator.HEAD_ADD:
                return head_correction(hidden, values[applied[0]])
            delta = (
                values[applied[0]]
                if operator is ParameterOverlayOperator.DENSE_ADD
                else values[applied[1]] @ values[applied[0]]
            )
            return hidden @ delta.T

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
        math = OverlayTransition(tuple(buffers), targets, phases, apply, function, source)
        self.overlay_math = math
        transition = self.compute(
            name,
            inputs={"operands": self.input("parameter.operands", "dense_pairs", "canonical.dense_operand_pairs@1")},
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
        self._bind_visibility(transition, source)
        if visibility == Exact():
            self.operations[transition.symbol] = replace(self.operations[transition.symbol], version="private")
        publication = self.publish(
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
        self.overlay(
            state,
            (transition, publication),
            tuple(
                ParameterOverlayFieldBinding(
                    target,
                    operator,
                    tuple(state.view(f"{key}_{i}", self.dimensions) for key in applied),
                    str(state.declaration.resource),
                    scale,
                )
                for i, target in enumerate(targets)
            ),
        )

    def write_result(self, source: Call, state: State, fields: tuple[str, ...], when: When) -> Call:
        """Bind a computed tuple to one atomic private transition."""
        task = self.tensor_tasks.get(source.symbol)
        writer = FieldWriter(
            source,
            tuple(self.field_views[state.declaration.resource, key] for key in fields)
            if self.dimensions is not None
            else (),
            state.declaration.resource,
            None if task is None else task.phase,
        )
        update = self.compute(
            f"{source.symbol}.write",
            inputs={name: source[name] for name in source.outputs},
            outputs=("updated",),
            changes=state,
            effect=StateEffectKind.TRANSITION_MAYBE,
            when=when,
            function=writer,
            select=writer.select,
            implementations=(native(f"tensor.write:{source.symbol}@1", "transition/full_root@1"),),
        )

        self.operations[update.symbol] = replace(
            self.operations[update.symbol], optimizer=None, mode="transition", copy_fields=None
        )
        return update

    def import_update(self, state: State, identity: str) -> Call:
        """One BEFORE_MODEL import of externally produced field values into private roots."""

        def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
            with torch.no_grad():
                if any(not finite(source) for _, values in data.replacements for source in values):
                    raise ValueError("replacement tensors differ from the signed layout or finite-value contract")
                for index, values in data.replacements:
                    for view, source in zip(self.imports[str(state.declaration.resource)], values, strict=True):
                        view.read(batch.roots[index][view.field]).copy_(source)
            return TaskOutput(
                update_count=len(data.replacements), changed=tuple(index for index, _ in data.replacements)
            )

        update = self.compute(
            "update",
            inputs={"parameters": self.input("input", "state", StateTensorUpdate.schema)},
            outputs=("updated",),
            changes=state,
            effect=StateEffectKind.TRANSITION_MAYBE,
            when=When.BEFORE_MODEL,
            function=execute,
            select=lambda data, selection: tuple(index for index, _ in data.replacements),
            implementations=(native(identity, "transition/full_root@1"),),
        )
        self.operations[update.symbol] = replace(
            self.operations[update.symbol], optimizer=None, mode="transition", copy_fields=None
        )
        return update

    def activation_patch(
        self,
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
        views = tuple(
            view for (resource, _), view in self.field_views.items() if resource == state.declaration.resource
        )
        if (
            self.host
            or not layers
            or tuple(sorted(set(layers))) != layers
            or layers[0] < 0
            or min(first_tokens, last_tokens) < 0
            or self.dimensions is not None
            and len(views) != len(layers) * width
        ):
            raise ValueError("activation fields must cover each ordered layer and operator")
        self.overlay(
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

    def layer_route(
        self,
        routes: tuple[tuple[Tokens, Over, Callable[[Layers], torch.Tensor]], ...],
        position_states: tuple[tuple[str, float], ...] = (),
    ) -> None:
        """Trace a user route and sign its complete model/tensor source before planning."""
        tokens: Tokens = routes[0][0] if len(routes) == 1 else "all"
        activation = self.attachment is not None and self.attachment.kind is AttachmentKind.ACTIVATION_PATCH
        composed = (
            self.attachment is not None
            and self.attachment.kind is AttachmentKind.PARAMETER_OVERLAY
            and self.overlay_math is not None
        )
        if self.host or (
            not activation and not composed and (self.attachment is not None or self.graph.tasks or self.graph.states)
        ):
            raise ValueError("layer_route requires a standalone effect, Activation or one trained parameter overlay")
        if position_states and (activation or composed):
            raise ValueError("use PositionState on a pure route without learned overlays or activations")
        self.position_states = position_states
        effect = (
            None
            if self.dimensions is None
            else TokenProgramEffect(self.dimensions.layer_count, tokens, position_states)
        )
        if composed:
            assert self.attachment is not None
            symbol = next(symbol for role, symbol in self.attachment.tasks if role is AttachmentTaskRole.TRANSITION)
            self.graph = replace(
                self.graph,
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
                    for task in self.graph.tasks
                ),
            )
            self.operations[symbol] = replace(self.operations[symbol], version="private")
        else:
            forward = self.compute(
                "forward",
                inputs={"rows": self.rows()},
                outputs=("normal_hidden",),
                when=When.MODEL_UPDATE,
                replay=effect,
                implementations=(native(TOKENWISE_REPLAY_IMPLEMENTATION, "model/tokenwise_replay@1"),),
            )
            symbol = forward.symbol
            self.model_effect(forward, AttachmentKind.MODEL_REPLAY)

        if self.probe is not None:
            probe = self.probe
            assert self.attachment is not None
            assert probe.causal_domain is not None
            tensor = TensorType(
                (1, probe.hidden_size),
                "torch." + probe.tensor_dtype.value,
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
            self.route_rows = probe.max_packed_rows * max(1, sum(isinstance(step, Segment) for step in steps))
            for _, program in captured:
                pending = False
                for step in program.steps:
                    if isinstance(step, Segment):
                        pending |= any(step.start <= target.layer < step.stop for target in self.overlay_targets)
                    elif isinstance(step, ApplyUpdates):
                        self.route_updates |= pending
                        pending = False
            if composed:
                phases, updates = self.parameter_phases, self.route_updates

                def select(data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
                    return tuple(
                        sorted(
                            {
                                index
                                for row, index in zip(selection.rows, selection.state_indices, strict=True)
                                if row.phase in phases
                                and (updates or not (tokens == "all" or row.phase.value == tokens))
                            }
                        )
                    )

                self.operations[symbol] = replace(self.operations[symbol], select=select)
            if any(isinstance(step, ApplyUpdates) for step in steps) and not composed:
                raise ValueError("apply_updates needs one trained parameter overlay on the same workload")
            self.route_program = programs
            self.graph = replace(
                self.graph,
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
                        ),
                    )
                    if task.symbol == symbol
                    else task
                    for task in self.graph.tasks
                ),
            )

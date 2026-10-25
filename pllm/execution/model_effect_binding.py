"""Model-side tensor effects: parameter overlays, imported overlays, activation patches.

ModelEffectBinding declares effects that run inside or alongside the model forward and their
request-state transitions. Used by TensorBinding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import isfinite, prod
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
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterTarget,
)
from pllm.contracts.region_program import BorrowedResource, TensorType
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.semantic_graph import (
    PackedRowsInputSource,
    SemanticInput,
    StateEffectKind,
    StateResource,
    StateScope,
    TaskSourceRegion,
    TokenProgramEffect,
)
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.contracts.visibility import Exact, Visibility, VisibilitySource
from pllm.execution.declaration import Call, Dimensions, State, When, native
from pllm.execution.decoder_region import decoder_activations
from pllm.execution.overlay_task import OverlayTransition
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.region_storage import import_region_functions
from pllm.execution.replay import TOKENWISE_REPLAY_IMPLEMENTATION
from pllm.execution.shared_tensor import tensor_dtype
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
from pllm.execution.token_program import ApplyUpdates, Layers, Segment, TokenProgram, Tokens
from pllm.kernels.head_overlay import head_correction

if TYPE_CHECKING:
    pass


from pllm.execution.binding_base import Buffer, TensorBindingBase, buffer_field


@dataclass(frozen=True)
class FieldWriter:
    """Write a task result into its declared field; used by the write lowering."""

    source: Call
    field: LayoutFieldId
    resource: StateResource

    def __call__(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        value = batch.results[self.source.symbol, "value"]
        if value.is_meta:
            target = batch.parameter_root(self.resource)[self.field]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"write to {self.resource} needs shape {tuple(target.shape)} and dtype {target.dtype}")
        if not finite(value):
            raise ValueError("write needs a finite tensor result")
        return TaskOutput(
            field_writes=((self.field, value.unsqueeze(0)),), changed=self.select(data, batch.selection), update_count=1
        )

    def select(self, data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
        """Select the target resource without updating other local states."""
        return tuple(selection.refs.index(ref) for ref in selection.for_resource(self.resource).refs)


def _check_overlay_fields(
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


def _check_overlay_shapes(
    buffers: Mapping[str, Buffer], applied: tuple[str, ...], dimensions: Dimensions, operator: ParameterOverlayOperator
) -> None:
    def shape_of(key: str, inputs: int, outputs: int) -> tuple[int, ...]:
        buffer = buffers[key]
        return buffer.shape(inputs, outputs) if callable(buffer.shape) else buffer.shape

    for inputs, outputs in dimensions.target_shapes:
        if operator is ParameterOverlayOperator.DENSE_ADD:
            matched = shape_of(applied[0], inputs, outputs) == (outputs, inputs)
        elif operator is ParameterOverlayOperator.HEAD_ADD:
            shape = shape_of(applied[0], inputs, outputs)
            matched = (
                len(shape) == 3
                and shape[0] > 0
                and shape[1] == shape[2] > 0
                and shape[0] * shape[1] == inputs == outputs
            )
        else:
            down, up = (shape_of(key, inputs, outputs) for key in applied)
            matched = (
                len(down) == 2 and len(up) == 2 and 1 <= down[0] == up[1] and down[1] == inputs and up[0] == outputs
            )
        dtypes = (
            (torch.float32,)
            if operator is ParameterOverlayOperator.HEAD_ADD
            else (None, tensor_dtype(dimensions.dtype))
        )
        if not matched or any(buffers[key].dtype not in dtypes for key in applied):
            raise ValueError("overlay fields must match their operator, target shape and dtype")


class ModelEffectBinding(TensorBindingBase):
    """Declare model-side effects and their request-state transitions."""

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
        _check_overlay_fields(buffers, applied, targets, operator, scale)
        dimensions = self.dimensions
        if dimensions is not None:
            if len(dimensions.target_shapes) != len(targets):
                raise ValueError("overlay targets differ from backend specialization")
            _check_overlay_shapes(buffers, applied, dimensions, operator)
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
            transition,
            publication,
            tuple(
                ParameterOverlayFieldBinding(
                    target,
                    operator,
                    tuple(state.field(f"{key}_{i}") for key in applied),
                    str(state.declaration.resource),
                    scale,
                )
                for i, target in enumerate(targets)
            ),
        )

    def write_result(self, source: Call, state: State, field: str) -> Call:
        """Lower one computed tensor to a private field update."""
        writer = FieldWriter(source, state.field(field), state.declaration.resource)
        return self.compute(
            f"{source.symbol}.write",
            inputs={"value": source["value"]},
            outputs=("updated",),
            changes=state,
            effect=StateEffectKind.TRANSITION,
            when=When.BEFORE_MODEL,
            function=writer,
            select=writer.select,
            implementations=(native(f"tensor.write:{source.symbol}:{field}@1", "transition/full_root@1"),),
        )

    def import_update(self, state: State, identity: str) -> Call:
        """One BEFORE_MODEL import of externally produced field values into private roots."""

        def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
            with torch.no_grad():
                if any(not finite(source) for _, values in data.replacements for source in values):
                    raise ValueError("replacement tensors differ from the signed layout or finite-value contract")
                for index, values in data.replacements:
                    for target, source in zip(batch.roots[index], values, strict=True):
                        target.copy_(source)
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
        self.operations[update.symbol] = replace(self.operations[update.symbol], copy_fields=())
        return update

    def import_overlay(
        self,
        state_name: str,
        buffers: Mapping[str, Buffer],
        *,
        targets: tuple[ParameterTarget, ...],
        applied: tuple[str, ...],
        namespace: str,
        keys: tuple[RequestKeyField, ...],
        scope: StateScope,
        operator: ParameterOverlayOperator,
        scale: float,
    ) -> None:
        """Externally produced overlay fields; the model consumption is identical to trained overlays."""
        if self.graph.tasks or self.graph.states or self.host:
            raise ValueError("imported overlay requires a sole model state and effect")
        _check_overlay_fields(buffers, applied, targets, operator, scale)
        dimensions = self.dimensions
        if dimensions is not None:
            if len(dimensions.target_shapes) != len(targets):
                raise ValueError("overlay targets differ from backend specialization")
            _check_overlay_shapes(buffers, applied, dimensions, operator)
        self.overlay_targets = targets
        self.overlay_import = True
        state = self.state(
            state_name,
            tuple(
                buffer_field(f"{key}_{index}", buffer, lambda m, i=index: m.target_shapes[i])
                for index in range(len(targets))
                for key, buffer in buffers.items()
            ),
            namespace=namespace,
            scope=scope,
            keys=keys,
        )
        update = self.import_update(state, "canonical.overlay:import@1")
        publication = self.publish(update, state, when=When.BEFORE_MODEL)
        self.overlay(
            state,
            update,
            publication,
            tuple(
                ParameterOverlayFieldBinding(
                    target,
                    operator,
                    tuple(state.field(f"{key}_{i}") for key in applied),
                    str(state.declaration.resource),
                    scale,
                )
                for i, target in enumerate(targets)
            ),
        )

    def buffer_state(
        self,
        name: str,
        buffers: Mapping[str, Buffer],
        *,
        namespace: str,
        scope: StateScope,
        keys: tuple[RequestKeyField, ...],
    ) -> State:
        """Allocate named model-width fields independently of their effect."""
        return self.state(
            name,
            tuple(
                buffer_field(key, buffer, lambda m: (m.hidden_size, m.hidden_size)) for key, buffer in buffers.items()
            ),
            namespace=namespace,
            scope=scope,
            keys=keys,
        )

    def activation_patch(
        self,
        state: State,
        update: Call,
        publication: Call,
        *,
        layers: tuple[int, ...],
        operator: ActivationPatchOperator,
        first_tokens: int,
        last_tokens: int,
        decode: DecodeWindow = DecodeWindow.NONE,
        decode_refresh: DecodeRefresh = DecodeRefresh.INHERIT,
        on_missing: MissingPolicy = MissingPolicy.STOP,
    ) -> None:
        """Attach an activation to a declared transition and its publication."""
        width = 1 if operator is ActivationPatchOperator.ADD else 3
        if (
            self.host
            or not layers
            or tuple(sorted(set(layers))) != layers
            or layers[0] < 0
            or min(first_tokens, last_tokens) < 0
            or len(state.fields) != len(layers) * width
        ):
            raise ValueError("activation fields must cover each ordered layer and operator")
        if self.dimensions is not None:
            hidden = self.dimensions.hidden_size
            (layout,) = (entry.layout for entry in self.layouts.layouts if entry.resource == state.declaration.resource)
            for index in range(len(layers)):
                fields = layout.fields[index * width : (index + 1) * width]
                shapes = tuple(field.shape for field in fields)
                match shapes:
                    case ((size,),) if operator is ActivationPatchOperator.ADD and size == hidden:
                        dtype = TensorDType.FLOAT32
                    case ((rank, size), (bias,), (outputs, rotation)) if (
                        operator is ActivationPatchOperator.LOW_RANK_AFFINE
                        and 1 <= rank <= hidden
                        and size == outputs == hidden
                        and rank == bias == rotation
                    ):
                        dtype = self.dimensions.dtype
                    case _:
                        raise ValueError("activation fields differ from the operator's rank and hidden dimensions")
                if any(field.dtype is not dtype for field in fields):
                    raise ValueError("activation fields differ from the operator's dtype")
        self.overlay(
            state,
            update,
            publication,
            activations=tuple(
                ActivationPatchBinding(
                    layer,
                    tuple(state.field(state.fields[index * width + offset].name) for offset in range(width)),
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

    def layer_route(self, function: Callable[[Layers], torch.Tensor], tokens: Tokens) -> None:
        """Trace a user route and sign its complete model/tensor source before planning."""
        if any(task.replay is not None for task in self.graph.tasks):
            raise ValueError("compose one layer_route with one trained overlay")
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
        effect = None if self.dimensions is None else TokenProgramEffect(self.dimensions.layer_count, tokens)
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
            activations = decoder_activations(
                self.attachment.activation_bindings, self.layouts, self.attachment.state_resource, tensor.device
            )
            captured = TokenProgram.capture(function, tensor, probe.causal_domain.layers)
            self.route_rows = probe.max_packed_rows * max(1, sum(isinstance(step, Segment) for step in captured.steps))
            pending = False
            for step in captured.steps:
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
            program, self.route_functions = captured.lower(
                tensor,
                probe.causal_domain,
                probe.max_packed_rows,
                activations,
                None if not composed else BorrowedResource("overlay", (self.overlay_bytes(captured),)),
            )
            self.graph = replace(
                self.graph,
                source_regions=(
                    TaskSourceRegion(
                        symbol, "layer_route", program, import_region_functions(program, self.route_functions)
                    ),
                ),
            )

    def overlay_bytes(self, program: TokenProgram) -> int:
        """Fund token staging, captured operands and per-boundary device effects from source bounds."""
        assert self.attachment is not None and self.dimensions is not None and self.probe is not None
        fields = tuple(field for layout in self.layouts.layouts for field in layout.layout.fields)

        return (
            3
            * sum(
                prod(field.shape) * (8 if field.dtype in (TensorDType.INT64, TensorDType.FLOAT64) else 4) + 8
                for field in fields
            )
            + self.probe.max_packed_rows
            * sum(
                16 * (inputs + outputs) + 128
                for target, (inputs, outputs) in zip(self.overlay_targets, self.dimensions.target_shapes, strict=True)
                for step in program.steps
                if isinstance(step, Segment) and step.start <= target.layer < step.stop
            )
            + 1024
        )

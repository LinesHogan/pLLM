"""Model-side tensor effects: ModelScan, parameter overlays, imported overlays, activation patches.

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
    BackendAttachmentSpec,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ModelScan,
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterTarget,
)
from pllm.contracts.region_imports import NativeTensorImport, region_import_name
from pllm.contracts.region_proposal import ClosedWindowRequest
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    ConditionalLayerReplay,
    SemanticEffect,
    StateEffectKind,
    StateScope,
    TaskSourceRegion,
)
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.contracts.visibility import Exact, Visibility, VisibilitySource
from pllm.execution.declaration import Call, Dimensions, State, When, native
from pllm.execution.model_scan import ChunkTraining, model_scan_source, target_resource
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.replay import TOKENWISE_REPLAY_IMPLEMENTATION, Recurrence
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.tensor_frame import (
    OverlayEntry,
    OverlayTargetResult,
    TensorFrame,
    TensorMath,
    Tensors,
    TensorUpdate,
    finite,
)
from pllm.execution.tensor_program import (
    PageUpdate,
    PoolSGD,
    TaskOutput,
    TensorBatch,
)
from pllm.kernels.adamw import AdamWConfig
from pllm.kernels.head_overlay import head_correction

if TYPE_CHECKING:
    pass


from pllm.execution.binding_base import SGD, Buffer, TensorBindingBase, buffer_field


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


@dataclass(frozen=True, slots=True)
class OverlayTransition:
    """Request/target adaptation shared by source and explicitly lowered math.

    Replacing this immutable callable during binding changes only the imported
    mathematics. The original TensorOperation still owns private-page access.
    """

    buffers: tuple[str, ...]
    targets: tuple[ParameterTarget, ...]
    phases: tuple[HiddenPhase, ...]
    apply: Callable[[torch.Tensor, Tensors], torch.Tensor]
    function: Callable[[Tensors], TensorUpdate]
    visibility: VisibilitySource

    def __call__(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        operands = {
            (operand.identity, operand.target): operand
            for operand in data.step.operands
            if self.visibility.in_forward or operand.phase in self.phases
        }
        if not operands:
            return TaskOutput()
        start = len(data.entries)
        changed: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for row, index in zip(batch.selection.rows, batch.selection.state_indices, strict=True):
                results: list[OverlayTargetResult] = []
                for target_index, target in enumerate(self.targets):
                    if (row.identity, target) not in operands:
                        continue
                    operand = operands[row.identity, target]
                    values = dict(
                        zip(
                            self.buffers,
                            batch.roots[index][
                                len(self.buffers) * target_index : len(self.buffers) * (target_index + 1)
                            ],
                            strict=True,
                        )
                    )
                    result = self.result(operand, values)
                    results.append(OverlayTargetResult(target, result))
                    changed[index] = result.changed if index not in changed else changed[index] | result.changed
                if results:
                    data.entries.append(
                        OverlayEntry(row.identity, batch.selection.refs[index], batch.after_refs[index], tuple(results))
                    )
        return TaskOutput(
            residuals={
                (entry.identity, target.target): target.update.output
                for entry in data.entries[start:]
                for target in entry.targets
            },
            changed=tuple(changed),
            update_mask=torch.stack(tuple(changed.values())),
            counters=torch.stack(
                tuple(
                    torch.stack((target.update.update_count, target.update.trained_rows))
                    for entry in data.entries[start:]
                    for target in entry.targets
                )
            ).sum(0),
        )

    def result(self, operand: CapturedParameterOperand, values: Tensors) -> TensorUpdate:
        """Validate the author's result at its tensor boundary without observing CUDA values."""
        result = (
            self.function({**values, "input": operand.hidden, "target": operand.desired, "weight": operand.base_weight})
            if operand.phase in self.phases
            else TensorUpdate.unchanged(self.apply(operand.hidden, values))
        )
        if (
            result.output.shape != operand.desired.shape
            or result.output.dtype not in (operand.hidden.dtype, torch.float32)
            or result.output.device != operand.hidden.device
        ):
            raise ValueError("overlay output differs from its target shape, dtype or device")
        if any(
            value.shape != () or value.device != operand.hidden.device or value.dtype != dtype
            for value, dtype in (
                (result.changed, torch.bool),
                (result.update_count, torch.int64),
                (result.trained_rows, torch.int64),
            )
        ):
            raise ValueError("overlay counters need a bool changed flag and int64 counts on the operand device")
        return result


class ModelEffectBinding(TensorBindingBase):
    """Declare model-side effects and their request-state transitions."""

    def scan_model(
        self,
        name: str,
        scan: ModelScan,
        buffers: Mapping[str, Buffer],
        namespace: str,
        initialize: Callable[[Mapping[str, torch.Tensor], torch.Generator], None],
        loss: TensorMath,
        optimizer: AdamWConfig | SGD,
        keys: tuple[RequestKeyField, ...],
        equations_digest: str,
    ) -> None:
        """Declare one request-owned scan and its projection consumer."""
        effect = scan.apply
        _check_overlay_fields(buffers, effect.factors, effect.targets, effect.operator, effect.scale)
        if self.graph.tasks or self.graph.states or self.host:
            raise ValueError("ModelScan needs its own request state and model execution")
        if self.dimensions is not None:
            _check_overlay_shapes(buffers, effect.factors, self.dimensions, effect.operator)
        if self.dimensions is not None and any(
            target.layer >= self.dimensions.layer_count for target in effect.targets
        ):
            raise ValueError("ModelScan targets must be inside the backend's complete layer domain")
        self.overlay_targets = effect.targets
        states = tuple(
            self.state(
                target_resource(effect.state, target.layer),
                tuple(
                    buffer_field(key, buffer, lambda model, i=i: model.target_shapes[i])
                    for key, buffer in buffers.items()
                ),
                namespace=namespace,
                scope=StateScope.REQUEST,
                keys=keys,
                optimizer=optimizer if isinstance(optimizer, AdamWConfig) else None,
            )
            for i, target in enumerate(effect.targets)
        )
        stride = 5 if states[0].optimizer is not None else 1
        parameters = tuple(range(0, len(states[0].fields), stride))

        def init(root: tuple[torch.Tensor, ...], generator: torch.Generator) -> None:
            initialize(dict(zip(buffers, (root[i] for i in parameters), strict=True)), generator)

        self.initializers.update((str(state.declaration.resource), init) for state in states)
        forward = self.compute(
            name,
            inputs={"rows": self.rows()},
            reads=tuple(SemanticEffect(state.declaration.resource, StateEffectKind.UPDATE_MAYBE) for state in states),
            when=When.MODEL_UPDATE,
            implementations=(native("model.chunk_scan@1", "model/chunk_scan@1"),),
        )
        self._bind_visibility(
            forward,
            VisibilitySource(
                scan.visibility,
                "model_scan",
                equations_digest,
                tuple(str(state.declaration.resource) for state in states),
                scan.chunk_size,
                "include",
            ),
        )
        if self.probe is not None:
            assert self.dimensions is not None
            program = model_scan_source(
                scan, self.layouts, self.probe, self.dimensions.layer_count, parameters, self.max_requests
            )
            catalog = {region_import_name(item): item for item in self.probe.region_imports}
            self.graph = replace(
                self.graph,
                source_regions=(
                    TaskSourceRegion(
                        forward.symbol,
                        "model",
                        program,
                        tuple(
                            catalog.get(function.name, NativeTensorImport(function))
                            for function in sorted(program.functions, key=lambda function: function.name)
                        ),
                        ClosedWindowRequest(
                            tuple(f"prefill{i}" for i in range(self.max_requests)),
                            scan.chunk_size,
                            self.probe.max_packed_rows,
                        ),
                    ),
                ),
            )
        self.model_scan = (
            scan,
            ChunkTraining(
                loss,
                tuple(buffers),
                parameters,
                PageUpdate(optimizer, len(buffers))
                if isinstance(optimizer, AdamWConfig)
                else PoolSGD(optimizer.lr, len(buffers)),
            ),
        )
        self._attach(
            BackendAttachmentSpec(
                AttachmentKind.MODEL_SCAN,
                ((AttachmentTaskRole.COMPUTE, forward.symbol),),
                AnchorSymbol("sampler.pre_draw"),
                states[0].declaration.resource,
                tuple(
                    ParameterOverlayFieldBinding(
                        target,
                        effect.operator,
                        tuple(LayoutFieldId(i) for i in parameters),
                        target_resource(effect.state, target.layer),
                        effect.scale,
                    )
                    for target in effect.targets
                ),
                model_scan_chunk_size=scan.chunk_size,
            )
        )

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
        targets: tuple[ParameterTarget, ...],
        applied: tuple[str, ...],
        namespace: str,
        keys: tuple[RequestKeyField, ...],
        phases: tuple[HiddenPhase, ...],
        replay: ConditionalLayerReplay | None,
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
        transition = self.compute(
            name,
            inputs={"operands": self.input("parameter.operands", "dense_pairs", "canonical.dense_operand_pairs@1")},
            outputs=("transition",),
            changes=state,
            effect=StateEffectKind.TRANSITION_MAYBE,
            when=When.MODEL_UPDATE,
            function=OverlayTransition(tuple(buffers), targets, phases, apply, function, source),
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
        if replay is not None:
            self.replay(publication, state, replay)

    def _import_update(self, state: State, identity: str) -> Call:
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
        update = self._import_update(state, "canonical.overlay:import@1")
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

    def activation_patch(
        self,
        name: str,
        buffers: Mapping[str, Buffer],
        *,
        layers: tuple[int, ...],
        operator: ActivationPatchOperator,
        first_tokens: int,
        last_tokens: int,
        scope: StateScope,
        keys: tuple[RequestKeyField, ...],
        namespace: str,
        decode: DecodeWindow = DecodeWindow.NONE,
        decode_refresh: DecodeRefresh = DecodeRefresh.INHERIT,
        on_missing: MissingPolicy = MissingPolicy.STOP,
    ) -> None:
        replay = (
            self.attachment is not None
            and self.attachment.kind is AttachmentKind.MODEL_REPLAY
            and not self.attachment.activation_bindings
            and len(self.graph.tasks) == 1
            and not self.graph.states
        )
        if self.host or (not replay and (self.graph.tasks or self.graph.states)):
            raise ValueError("activation replacement requires a sole model state and effect")
        width = 1 if operator is ActivationPatchOperator.ADD else 3
        if (
            not layers
            or tuple(sorted(set(layers))) != layers
            or layers[0] < 0
            or min(first_tokens, last_tokens) < 0
            or len(buffers) != len(layers) * width
        ):
            raise ValueError("activation fields must cover each ordered layer and operator")
        state = self.state(
            name,
            tuple(
                buffer_field(key, buffer, lambda m: (m.hidden_size, m.hidden_size)) for key, buffer in buffers.items()
            ),
            namespace=namespace,
            scope=scope,
            keys=keys,
        )
        if self.dimensions is not None:
            hidden = self.dimensions.hidden_size
            layout = self.layouts.layouts[0].layout
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
        update = self._import_update(state, "canonical.activation:import@1")
        publication = self.publish(update, state, when=When.BEFORE_MODEL)
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

    def recurrence(self, effect: Recurrence) -> None:
        activation = self.attachment is not None and self.attachment.kind is AttachmentKind.ACTIVATION_PATCH
        if self.host or (not activation and (self.attachment is not None or self.graph.tasks or self.graph.states)):
            raise ValueError("tokenwise recurrence requires a standalone model effect")
        forward = self.compute(
            "forward",
            inputs={"rows": self.rows()},
            outputs=("normal_hidden",),
            when=When.MODEL_UPDATE,
            replay=None if self.dimensions is None else effect.effect(self.dimensions.layer_count),
            implementations=(native(TOKENWISE_REPLAY_IMPLEMENTATION, "model/tokenwise_replay@1"),),
        )
        self.model_effect(forward, AttachmentKind.MODEL_REPLAY)

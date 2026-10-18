"""Tensor decorators bound once to the original graph and invocation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite, prod
from typing import TYPE_CHECKING, overload

import torch

from pllm.contracts.auxiliary import StateTensorUpdate, WeightedCEBatch
from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    ActivationPatchOperator,
    AttachmentKind,
    BackendPlacementProbe,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding, ParameterOverlayOperator, ParameterTarget
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback, SamplerStage
from pllm.contracts.semantic_graph import (
    ConditionalLayerReplay,
    HiddenInputSource,
    HostPayloadSchemaId,
    MathContract,
    PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource,
    SemanticEffect,
    SemanticInput,
    SemanticInputSource,
    SemanticOutputKind,
    StateEffectKind,
    StateScope,
    StateSelectionInputSource,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.execution.codec import register_gradient_hook
from pllm.execution.declaration import Builder, Call, Dimensions, Field, Shape, State, When, native
from pllm.execution.ptts.page_borrow import borrowed_root
from pllm.execution.ptts.page_bundle import PagePublication
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.replay import TOKENWISE_REPLAY_IMPLEMENTATION, Recurrence
from pllm.execution.shared_tensor import tensor_dtype
from pllm.execution.tensor_program import InvocationOutput, Observation, TaskOutput, TensorBatch
from pllm.execution.trace import MAX_SNAPSHOT_BYTES
from pllm.kernels.adamw import AdamWConfig

if TYPE_CHECKING:
    from pllm.authoring import TensorStepInput

Tensors = Mapping[str, torch.Tensor]
TensorMath = Callable[[Tensors], torch.Tensor]
TensorReader = Callable[["TensorFrame", TensorBatch], torch.Tensor]


def select_rows(value: torch.Tensor, rows: tuple[int, ...]) -> torch.Tensor:
    """Borrow consecutive original rows; pack only an irregular selection."""
    if not rows:
        return value[:0]
    if rows == tuple(range(rows[0], rows[0] + len(rows))):
        return value.narrow(0, rows[0], len(rows))
    return value[list(rows)]


def _flat_gradient(fields: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return fields[0].flatten() if len(fields) == 1 else torch.cat(tuple(value.flatten() for value in fields))


def _check_overlay_fields(
    buffers: Mapping[str, Buffer],
    applied: tuple[str, ...],
    targets: tuple[ParameterTarget, ...],
    operator: ParameterOverlayOperator,
    scale: float,
) -> None:
    arity = 1 if operator is ParameterOverlayOperator.DENSE_ADD else 2
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
        else:
            down, up = (shape_of(key, inputs, outputs) for key in applied)
            matched = (
                len(down) == 2 and len(up) == 2 and 1 <= down[0] == up[1] and down[1] == inputs and up[0] == outputs
            )
        if not matched or any(buffers[key].dtype not in (None, tensor_dtype(dimensions.dtype)) for key in applied):
            raise ValueError("overlay fields must match their operator, target shape and dtype")


def _gradient_rows(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Give this consumer its own gradient edge without copying the forward value.
    result = value.view_as(value)
    register_gradient_hook(result, lambda gradient: gradient.masked_fill(~mask, 0))
    return result


@dataclass(frozen=True)
class SGD:
    lr: float

    def __post_init__(self) -> None:
        if not isfinite(self.lr) or self.lr <= 0:
            raise ValueError("SGD requires a positive finite learning rate")


@dataclass(frozen=True)
class Buffer:
    shape: tuple[int, ...] | Callable[[int, int], tuple[int, ...]]
    dtype: torch.dtype | None = None


def _buffer_field(name: str, buffer: Buffer, dimensions: Callable[[Dimensions], tuple[int, int]]) -> Field:
    shape = buffer.shape

    def size(model: Dimensions) -> tuple[int, ...]:
        return shape(*dimensions(model)) if callable(shape) else shape

    return Field(
        name,
        size,
        (lambda model: model.dtype) if buffer.dtype is None else TensorDType(str(buffer.dtype).removeprefix("torch.")),
        8 if buffer.dtype is torch.int64 else 16,
    )


@dataclass(frozen=True)
class TensorUpdate:
    output: torch.Tensor
    changed: bool = False
    update_count: int = 0
    trained_rows: int = 0
    metrics: Mapping[str, int] = field(default_factory=lambda: dict[str, int]())


@dataclass(frozen=True, slots=True)
class OverlayTargetResult:
    target: ParameterTarget
    update: TensorUpdate


@dataclass(frozen=True, slots=True)
class OverlayEntry:
    identity: RequestIdentity
    before_ref: StateRef
    after_ref: StateRef
    targets: tuple[OverlayTargetResult, ...]

    @property
    def changed(self) -> bool:
        return any(target.update.changed for target in self.targets)


@dataclass(frozen=True, slots=True)
class TensorPort:
    source: SemanticInputSource
    read: TensorReader
    rowwise: bool = True


def take_rows(values: torch.Tensor, rows: Sequence[int], device_rows: torch.Tensor | None = None) -> torch.Tensor:
    """`values[rows]` along dim 0: a view when the rows repeat one row or run consecutively, else one gather."""
    first = rows[0] if rows else 0
    if len(set(rows)) == 1:
        return values[first : first + 1].expand(len(rows), *values.shape[1:])
    if tuple(rows) == tuple(range(first, first + len(rows))):
        return values.narrow(0, first, len(rows))
    if device_rows is None:
        device_rows = torch.tensor(rows, dtype=torch.long, device=values.device)
    return values.index_select(0, device_rows)


_COLUMNS = ("packed", "sampler", "samples", "states", "versions")


@dataclass(frozen=True, slots=True)
class RowIndex:
    """Per-row index columns of one invocation; entry i describes `TensorFrame.rows[i]`.

    The columns are built on the host and copied to the device once per invocation, so reading
    many tensor inputs costs one small host-to-device copy.
    """

    packed: tuple[int, ...]  # row in the packed hidden batch
    sampler: tuple[int, ...]  # row in the sampler batch; -1 if the row was not sampled
    samples: tuple[int, ...]  # sample index of the row's request
    states: tuple[int, ...]  # position of the row's state in the selection; 0 without state
    versions: tuple[int, ...]  # version of that state; 0 without state
    device: torch.Tensor  # the five columns above, stacked on the device

    def column(self, name: str) -> torch.Tensor:
        return self.device[_COLUMNS.index(name)]

    def take(self, values: torch.Tensor, name: str) -> torch.Tensor:
        """Rows of `values` selected by one column, in operation-row order."""
        rows: tuple[int, ...] = getattr(self, name)
        if -1 in rows:
            raise ValueError(f"some rows have no {name} row")
        return take_rows(values, rows, self.column(name))

    @classmethod
    def build(cls, data: "TensorFrame", batch: TensorBatch, device: torch.device) -> "RowIndex":
        links = {link.packed_row_index: link.sampler_row_index for link in data.step.sampler_rows.links}
        selection = batch.selection
        states = (
            tuple(selection.row_states[row.identity] for row in data.rows)
            if selection.groups
            else (0,) * len(data.rows)
        )
        columns = (
            tuple(row.packed_row_index for row in data.rows),
            tuple(links.get(row.packed_row_index, -1) for row in data.rows),
            tuple(row.identity.sample_idx for row in data.rows),
            states,
            tuple(int(selection.refs[index].version_id) for index in states) if selection.groups else states,
        )
        return cls(*columns, device=torch.tensor(columns, dtype=torch.long).to(device))


def row_index(data: "TensorFrame", batch: TensorBatch, device: torch.device) -> RowIndex:
    """The invocation's row tables, built on first use."""
    if data.index is None:
        data.index = RowIndex.build(data, batch, device)
    return data.index


def _field_rows(data: "TensorFrame", batch: TensorBatch, field: int, start: int, size: int) -> torch.Tensor:
    """`field[start:start+size]` of each row's state, `[rows, size]`; copies only that slice."""
    pool, slots = batch.field_slots(field)
    view = pool.reshape(len(pool), -1).narrow(1, start, size)
    rows = tuple(slots[state] for state in row_index(data, batch, pool.device).states)
    return take_rows(view, rows)


def _state_rows(batch: TensorBatch, field: int) -> torch.Tensor:
    """One row per selected state: `[states, *field_shape]`, a view when there is a single state."""
    pool, slots = batch.field_slots(field)
    if len(slots) == 1:
        return pool[slots[0] : slots[0] + 1]
    return pool.index_select(0, torch.tensor(slots, dtype=torch.long, device=pool.device))


class _LeafRoots(Sequence[tuple[torch.Tensor, ...]]):
    """State roots whose first field reads the differentiable leaf; other fields are borrowed lazily."""

    __slots__ = ("_leaf", "_roots")

    def __init__(self, leaf: torch.Tensor, roots: Sequence[tuple[torch.Tensor, ...]]) -> None:
        self._leaf, self._roots = leaf, roots

    def __len__(self) -> int:
        return len(self._roots)

    @overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[torch.Tensor, ...], ...]: ...

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return (self._leaf[index], *self._roots[index][1:])


def state_leaf(data: "TensorFrame", batch: TensorBatch) -> torch.Tensor:
    """The invocation's differentiable parameters, created on first use."""
    if data.leaf is None:
        data.leaf = _state_rows(batch, 0).detach().requires_grad_(True)
    return data.leaf


@dataclass(slots=True)
class TensorFrame:
    step: TensorStepInput
    rows: tuple[PackedPhaseRow, ...]
    columns: int | None = None
    entries: list[OverlayEntry] = field(default_factory=lambda: list[OverlayEntry]())
    parameters: tuple[torch.Tensor, ...] = ()
    derivative_sources: dict[tuple[str, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    input_gradients: dict[tuple[str, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    parameter_gradient_fields: dict[StateRef, tuple[torch.Tensor, ...]] = field(
        default_factory=lambda: dict[StateRef, tuple[torch.Tensor, ...]]()
    )
    pending_update: TaskOutput | None = None
    batch_inputs: tuple[torch.Tensor, ...] = ()
    selected: tuple[int, ...] = ()
    trained_rows: int = 0
    replacements: tuple[tuple[int, tuple[torch.Tensor, ...]], ...] = ()
    index: RowIndex | None = None
    # Differentiable [states, parameters] copy of the flat optimizer field, shared by the
    # invocation's differentiable tasks and cleared by the training task that consumes it.
    leaf: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TensorResult:
    value: torch.Tensor | None
    invocation: InvocationOutput
    entries: tuple[OverlayEntry, ...] = ()
    input_gradients: Mapping[tuple[str, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    parameter_gradient_fields: Mapping[StateRef, tuple[torch.Tensor, ...]] = field(
        default_factory=lambda: dict[StateRef, tuple[torch.Tensor, ...]]()
    )

    @property
    def parameter_gradients(self) -> Mapping[StateRef, torch.Tensor]:
        """Explicit flat inspection; execution retains the original gradient fields."""
        return {ref: _flat_gradient(values) for ref, values in self.parameter_gradient_fields.items()}

    @property
    def loss(self) -> torch.Tensor | None:
        return self.invocation.loss

    @property
    def guidance(self) -> PackedSamplerGuidanceBatch | None:
        return self.invocation.guidance

    @property
    def publication(self) -> PagePublication | None:
        return self.invocation.publication


TensorOutput = TensorResult | torch.Tensor | None


def _feedback(read: Callable[[SamplerFeedback], torch.Tensor]) -> TensorPort:
    def resolve(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        feedback = batch.feedback
        if feedback is None:
            raise RuntimeError("tensor function requires completed sampler feedback")
        values = read(feedback)
        return row_index(data, batch, values.device).take(values, "sampler")

    return TensorPort(SamplerFeedbackInputSource(), resolve)


def training_observations(
    data: TensorFrame, batch: TensorBatch, losses: torch.Tensor, inputs: Tensors, rows: tuple[int, ...]
) -> tuple[Observation, ...]:
    if not batch.observing:
        return ()
    return tuple(
        Observation(name, tensor, data.rows[row], ref, data.step.rows.step_id, gradient)
        for position, row in enumerate(rows)
        for index in (batch.selection.row_states[data.rows[row].identity],)
        for ref in (batch.selection.refs[index],)
        for name, tensor, gradient in (
            *((name, tensor[position], False) for name, tensor in inputs.items()),
            ("loss", losses if losses.ndim == 0 else losses[index], False),
            ("parameters.grad", _flat_gradient(data.parameter_gradient_fields[ref]), True),
        )
    )


@torch.no_grad()
def weighted_training_output(
    data: TensorFrame,
    batch: TensorBatch,
    losses: torch.Tensor,
    gradients: tuple[torch.Tensor, ...],
    fields: tuple[str, ...],
    steps: int,
    parameters: tuple[torch.Tensor, ...] = (),
) -> TaskOutput:
    for index, state in enumerate(data.selected):
        if parameters:
            for target, value in zip(batch.roots[state], parameters, strict=True):
                target.copy_(value[index])
        data.parameter_gradient_fields[batch.selection.refs[state]] = tuple(value[index] for value in gradients)
    observations = (
        tuple(
            Observation(
                name,
                value,
                ref=ref,
                iteration=steps - 1,
                identities=tuple(
                    row.identity
                    for row, position in zip(batch.selection.rows, batch.selection.state_indices, strict=True)
                    if position == state
                ),
                gradient=name.endswith(".grad"),
            )
            for index, state in enumerate(data.selected)
            for ref in (batch.selection.refs[state],)
            for name, value in (
                ("loss", losses[index]),
                *zip((f"{field}.grad" for field in fields), data.parameter_gradient_fields[ref], strict=True),
            )
        )
        if batch.observing
        else ()
    )
    data.batch_inputs = data.parameters = ()
    return TaskOutput(
        loss=losses,
        loss_count=len(data.selected) * steps,
        update_count=len(data.selected),
        trained_rows=data.trained_rows,
        changed=data.selected,
        observations=observations,
    )


def _port(source: SemanticInputSource | TensorPort) -> TensorPort:
    match source:
        case TensorPort():
            return source
        case HiddenInputSource(layer=layer, site=site, phase=HiddenPhase.DECODE):

            def hidden(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
                (snapshot,) = (
                    item
                    for item in data.step.hidden_batch(batch.stream).snapshots
                    if item.resource.locator_key == (str(layer), "index", site.value)
                )
                index = row_index(data, batch, snapshot.hidden.device)
                return index.take(snapshot.hidden, "packed")[:, : data.columns]

            return TensorPort(source, hidden)
        case PackedSamplerRowsInputSource():

            def samples(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
                device = data.step.hidden_batch(batch.stream).snapshots[0].hidden.device
                return row_index(data, batch, device).column("samples")

            return TensorPort(source, samples)
        case SamplerFeedbackInputSource():
            return _feedback(lambda feedback: feedback.base_logits)
        case TaskResultInputSource(task=task, output=output):
            return TensorPort(source, lambda data, batch: batch.results[task, output])
        case _:
            raise ValueError("unsupported ordinary tensor input; use an explicit native binding")


@dataclass(frozen=True, slots=True)
class OverlayTransition:
    """Request/target adaptation shared by source and explicitly lowered math.

    Replacing this immutable callable during binding changes only the imported
    mathematics. The original TensorOperation still owns private-page access.
    """

    buffers: tuple[str, ...]
    targets: tuple[ParameterTarget, ...]
    phases: tuple[HiddenPhase, ...]
    effective: Callable[[Tensors], torch.Tensor]
    function: Callable[[Tensors], TensorUpdate]

    def __call__(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        operands = {(operand.identity, operand.target): operand for operand in data.step.operands}
        changed: set[int] = set()
        with torch.no_grad():
            for row, index in zip(batch.selection.rows, batch.selection.state_indices, strict=True):
                results: list[OverlayTargetResult] = []
                for target_index, target in enumerate(self.targets):
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
                    if operand.phase in self.phases:
                        result = self.function(
                            {
                                **values,
                                "input": operand.hidden,
                                "target": operand.desired,
                                "weight": operand.base_weight,
                            }
                        )
                    else:
                        result = TensorUpdate(operand.hidden @ (operand.base_weight + self.effective(values)).T)
                    if (
                        result.output.shape != operand.desired.shape
                        or result.output.dtype != operand.hidden.dtype
                        or result.output.device != operand.hidden.device
                    ):
                        raise ValueError("overlay output differs from its target shape, dtype or device")
                    if min(result.update_count, result.trained_rows) < 0:
                        raise ValueError("overlay update counters must be nonnegative")
                    results.append(OverlayTargetResult(target, result))
                    if result.changed:
                        changed.add(index)
                data.entries.append(
                    OverlayEntry(row.identity, batch.selection.refs[index], batch.after_refs[index], tuple(results))
                )
        return TaskOutput(
            changed=tuple(sorted(changed)),
            update_count=sum(target.update.update_count for entry in data.entries for target in entry.targets),
            trained_rows=sum(target.update.trained_rows for entry in data.entries for target in entry.targets),
        )


class TensorBinding(Builder[TensorFrame]):
    """Authoring projections only; tensor results remain invocation-local."""

    def __init__(self, name: str, digest: str, dimensions: Dimensions | None = None, *, host: bool = False) -> None:
        super().__init__(name, digest, dimensions)
        self.host = host
        self.initializers: dict[str, Callable[[tuple[torch.Tensor, ...], torch.Generator], None]] = {}
        self.weighted_training = False
        self.probe: BackendPlacementProbe | None = None
        self.sample_limit: int | None = None
        self.overlay_targets: tuple[ParameterTarget, ...] = ()
        self.overlay_import = False

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
        state = self.state(
            state_name,
            tuple(
                _buffer_field(f"{key}_{index}", buffer, lambda m, i=index: m.target_shapes[i])
                for index in range(len(targets))
                for key, buffer in buffers.items()
            ),
            namespace=namespace,
            scope=StateScope.REQUEST,
            keys=keys,
        )

        def effective(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
            if operator is ParameterOverlayOperator.DENSE_ADD:
                return values[applied[0]] * scale
            return values[applied[1]] @ values[applied[0]] * scale

        def select(data: TensorFrame, selection: RequestStateSelection) -> tuple[int, ...]:
            return tuple(
                sorted(
                    {
                        index
                        for row, index in zip(selection.rows, selection.state_indices, strict=True)
                        if any(
                            operand.identity == row.identity and operand.phase in phases
                            for operand in data.step.operands
                        )
                    }
                )
            )

        transition = self.compute(
            name,
            inputs={"operands": self.input("parameter.operands", "dense_pairs", "canonical.dense_operand_pairs@1")},
            outputs=("transition",),
            changes=state,
            effect=StateEffectKind.TRANSITION_MAYBE,
            when=When.MODEL_UPDATE,
            function=OverlayTransition(tuple(buffers), targets, phases, effective, function),
            select=select,
            implementations=(
                native(identity + ":cpu", "compute/cpu@1", "transition/full_root@1"),
                native(identity + ":cuda", "compute/cuda@1", "state/fenced_reclaim@1", "transition/full_root@1"),
            ),
        )
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
                    target, operator, tuple(state.field(f"{key}_{i}") for key in applied), scale
                )
                for i, target in enumerate(targets)
            ),
        )
        if replay is not None:
            self.replay(publication, state, replay)

    def lower_overlay(self, name: str, function: Callable[[Tensors], TensorUpdate]) -> None:
        """Bind explicit native math after its author has verified the source contract."""
        symbol = TaskSymbol(name)
        operation = self.operations[symbol]
        if not isinstance(operation.function, OverlayTransition):
            raise ValueError("overlay lowering requires an original overlay transition")
        self.operations[symbol] = replace(operation, function=replace(operation.function, function=function))

    def _import_update(self, state: State, identity: str) -> Call:
        """One BEFORE_MODEL import of externally produced field values into private roots."""

        def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
            with torch.no_grad():
                if any(not torch.isfinite(source).all() for _, values in data.replacements for source in values):
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
                _buffer_field(f"{key}_{index}", buffer, lambda m, i=index: m.target_shapes[i])
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
                    target, operator, tuple(state.field(f"{key}_{i}") for key in applied), scale
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
                _buffer_field(key, buffer, lambda m: (m.hidden_size, m.hidden_size)) for key, buffer in buffers.items()
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

    @staticmethod
    def sampled_tokens() -> TensorPort:
        return _feedback(lambda feedback: feedback.sampled_tokens)

    def parameters(
        self,
        name: str,
        shapes: Mapping[str, Shape],
        *,
        optimizer: AdamWConfig | SGD | None,
        namespace: str = "",
        initialize: Callable[[Tensors, torch.Generator], None] | None = None,
        scope: StateScope = StateScope.INSTANCE,
        dtype: TensorDType = TensorDType.FLOAT32,
        alignment: int = 16,
        keys: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,),
        memory: Mapping[str, Shape] = {},
    ) -> tuple[State, dict[str, SemanticInputSource | TensorPort]]:
        if self.graph.states:
            raise ValueError("ordinary parameters use one instance state with named tensor views")
        if set(memory) & set(shapes):
            raise ValueError("memory fields duplicate a parameter field")
        if memory and optimizer is None:
            raise ValueError("memory fields ride on the training transition and require an optimizer")

        def sizes(model: Dimensions) -> tuple[tuple[int, ...], ...]:
            return tuple(shape(model) if callable(shape) else shape for shape in shapes.values())

        # Without an optimizer the fields are the layout directly: readable, importable, never updated.
        raw = not isinstance(optimizer, AdamWConfig)
        memory_fields = tuple(Field(key, shape, dtype, alignment) for key, shape in memory.items())
        state = self.state(
            name,
            tuple(Field(key, shape, dtype, alignment) for key, shape in shapes.items())
            if raw
            else (Field("parameters", lambda m: (sum(prod(shape) for shape in sizes(m)),), dtype, alignment),),
            namespace=namespace,
            optimizer=optimizer if isinstance(optimizer, AdamWConfig) else None,
            scope=scope,
            keys=keys if scope is StateScope.REQUEST else (),
            memory=memory_fields,
        )
        dimensions = self.dimensions
        if dimensions is None:
            return state, {key: state.read() for key in (*shapes, *memory)}
        views: dict[str, Callable[[tuple[torch.Tensor, ...]], torch.Tensor]] = {}
        # Where each named view lives: (field index, start, size, shape); start is None for whole fields.
        locations: dict[str, tuple[int, int | None, tuple[int, ...]]] = {}
        offset = 0
        for index, (key, shape) in enumerate(zip(shapes, sizes(dimensions), strict=True)):
            size = prod(shape)

            def view(
                root: tuple[torch.Tensor, ...],
                start: int = offset,
                size: int = size,
                shape: tuple[int, ...] = shape,
                index: int = index,
            ) -> torch.Tensor:
                return root[index] if raw else root[0].narrow(0, start, size).view(shape)

            views[key] = view
            locations[key] = (index, None, shape) if raw else (0, offset, shape)
            offset += size
        # Memory fields follow the optimizer-expanded parameter fields.
        base = len(shapes) if raw else 5
        for index, (key, memory_shape) in enumerate(memory.items()):

            def memory_view(root: tuple[torch.Tensor, ...], index: int = base + index) -> torch.Tensor:
                return root[index]

            views[key] = memory_view
            locations[key] = (
                base + index,
                None,
                memory_shape(dimensions) if callable(memory_shape) else memory_shape,
            )

        if initialize is not None:

            def init(root: tuple[torch.Tensor, ...], generator: torch.Generator) -> None:
                initialize({key: view(root) for key, view in views.items()}, generator)

            self.initializers[str(state.declaration.resource)] = init

        def port(view: Callable[[tuple[torch.Tensor, ...]], torch.Tensor]) -> TensorPort:
            return TensorPort(
                state.read(),
                lambda data, batch: view(batch.parameter_root(str(state.declaration.resource))),
                rowwise=False,
            )

        if scope is StateScope.REQUEST:

            def request_port(key: str) -> TensorPort:
                field_index, start, shape = locations[key]

                def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
                    size = prod(shape)
                    if field_index == 0 and data.leaf is not None:
                        leaf = data.leaf.narrow(1, start or 0, size)
                        rows = row_index(data, batch, leaf.device).take(leaf, "states")
                    else:
                        rows = _field_rows(data, batch, field_index, start or 0, size)
                    return rows.view(len(rows), *shape)

                return TensorPort(state.read(), read)

            return state, {key: request_port(key) for key in views}

        return state, {key: port(view) for key, view in views.items()}

    def weighted_train(
        self,
        name: str,
        state: State,
        *,
        identity: str,
        optimizer: SGD,
        steps: int,
        reset: Callable[[Tensors, torch.Generator], None],
        function: TensorMath,
        arguments: Mapping[str, tuple[str, str]],
    ) -> Call:
        if steps < 1 or self.weighted_training or self.host:
            raise ValueError("weighted training requires one model training sink and positive steps")
        self.weighted_training = True
        fields = tuple(field.name for field in state.fields)
        dimensions = self.dimensions
        prepare_function: Callable[[TensorFrame, TensorBatch], TaskOutput] | None = None
        if dimensions is not None:
            limit = next((fact for fact in self.resources if fact.task == name), None)
            if limit is not None and len(limit.input_shapes) != 1:
                raise ValueError("weighted training requires one signed input shape")
            shape = None if limit is None or not limit.input_shapes else limit.input_shapes[0]
            if state.declaration.scope is StateScope.REQUEST and shape is None:
                raise ValueError("independent training requires signed [states, rows, hidden] input bounds")
            if shape is not None and (
                len(shape) != 3
                or min(shape) < 1
                or shape[2] != dimensions.hidden_size
                or state.declaration.scope is StateScope.INSTANCE
                and shape[0] != 1
            ):
                raise ValueError("weighted training input shape differs from its state and hidden dimensions")

            def prepare(data: TensorFrame, view: TensorBatch) -> TaskOutput:
                batch = data.step.auxiliary
                if batch is None:
                    return TaskOutput()
                if not isinstance(batch, WeightedCEBatch):
                    raise TypeError("weighted training requires its declared weighted-CE input")
                shared = view.selection.keys == (None,)
                if shared and batch.keys:
                    raise ValueError("weighted CE keys require distinct co-resident original request states")
                selected = (0,)
                if not shared:
                    if (
                        not batch.keys
                        or len(set(batch.keys)) != len(batch.keys)
                        or len({key.fields for key in batch.keys}) != 1
                    ):
                        raise ValueError("weighted CE keys require distinct co-resident original request states")
                    batch_fields = batch.keys[0].fields
                    # Batch keys name rows under their own projection, independent of
                    # how the state is keyed (request, prompt/sample, route, ...).
                    row_states: dict[RequestStateKey, int] = {}
                    for row, index in zip(view.selection.rows, view.selection.state_indices, strict=True):
                        row_states.setdefault(RequestStateKey.project(row.identity, batch_fields), index)
                    selected = tuple(row_states[key] for key in batch.keys if key in row_states)
                    if len(selected) != len(batch.keys) or len(set(selected)) != len(selected):
                        raise ValueError("weighted CE keys require distinct co-resident original request states")
                tensors = (batch.hidden, batch.base_logits, batch.labels, batch.weights)
                hidden, logits, labels, weights = (
                    tensor.detach().unsqueeze(0) if shared else tensor.detach() for tensor in tensors
                )
                root = view.roots[selected[0]]
                if (
                    hidden.ndim != 3
                    or hidden.shape[0] != len(selected)
                    or hidden.shape[2] != dimensions.hidden_size
                    or logits.shape != (*hidden.shape[:2], dimensions.vocab_size)
                    or labels.shape != hidden.shape[:2]
                    or weights.shape != labels.shape
                    or labels.dtype != torch.long
                    or any(tensor.device != root[0].device for tensor in tensors)
                    or any(tensor.dtype != root[0].dtype for tensor in (hidden, logits, weights))
                    or shape is not None
                    and any(actual > bound for actual, bound in zip(hidden.shape, shape, strict=True))
                ):
                    raise ValueError("weighted-CE batch differs from signed parameters or input bounds")
                if not torch.isfinite(weights).all():
                    raise ValueError("weighted CE weights must be finite")
                valid = (labels >= 0) & (labels < dimensions.vocab_size) & (weights > 0)
                counts = valid.sum(-1).cpu()
                if not counts.all():
                    raise ValueError("weighted CE requires at least one valid weighted label per state")
                data.selected, data.trained_rows = selected, int(counts.sum().item())
                if data.trained_rows != valid.numel():
                    hidden, logits = (tensor.masked_fill(~valid.unsqueeze(-1), 0) for tensor in (hidden, logits))
                    labels, weights = (tensor.masked_fill(~valid, 0) for tensor in (labels, weights))
                if not torch.isfinite(hidden).all() or not torch.isfinite(logits).all():
                    raise ValueError("weighted-CE active rows must be finite")
                data.batch_inputs = tuple(
                    tensor.clone() if torch.is_inference(tensor) else tensor
                    for tensor in (hidden, logits, labels, weights)
                )
                data.parameters = tuple(tensor.new_empty((len(selected), *tensor.shape)) for tensor in root)
                with torch.no_grad():
                    reset(dict(zip(fields, data.parameters, strict=True)), torch.Generator(device=root[0].device))
                return TaskOutput()

            prepare_function = prepare
        prepared = self.compute(
            f"__{name}_prepare",
            inputs={"state": state.read(), "batch": self.input("training", "batch", WeightedCEBatch.schema)},
            outputs=("batch",),
            when=When.BEFORE_SAMPLE,
            function=prepare_function,
            implementations=(native("canonical.weighted_batch:prepare@1"),),
        )

        def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
            if not data.selected:
                return TaskOutput()
            with torch.enable_grad():
                parameters = {
                    key: tensor.detach().requires_grad_() for key, tensor in zip(fields, data.parameters, strict=True)
                }
                external: dict[str, torch.Tensor] = dict(
                    zip(("hidden", "base_logits", "labels", "weights"), data.batch_inputs, strict=True)
                )
                inputs = {
                    key: (parameters if kind == "parameter" else external)[field]
                    for key, (kind, field) in arguments.items()
                }

                def step() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
                    losses = function(inputs)
                    if losses.shape != (len(data.selected),):
                        raise ValueError("weighted_state_mean_then_mean requires one mean loss per selected state")
                    gradients = (
                        torch.autograd.grad(losses.mean(), tuple(parameters.values()), materialize_grads=True)
                        if losses.requires_grad
                        else tuple(torch.zeros_like(parameter) for parameter in parameters.values())
                    )
                    with torch.no_grad():
                        for parameter, gradient in zip(parameters.values(), gradients, strict=True):
                            parameter.add_(gradient, alpha=-optimizer.lr)
                    return losses.detach(), gradients

                losses, gradients = step()
                for _ in range(steps - 1):
                    losses, gradients = step()
            return weighted_training_output(data, batch, losses, gradients, fields, steps, data.parameters)

        transition = self.compute(
            name,
            inputs={"state": state.read(), "batch": TaskResultInputSource(prepared.symbol, "batch")},
            outputs=("publication",),
            result_schema="canonical.page_publication@1",
            changes=state,
            when=When.AFTER_SAMPLE,
            math=MathContract(
                parameter_gradients=(state.declaration.resource,),
                detached_inputs=("batch",),
                forward_rows="batch",
                parameter_gradient_rows="batch",
                update_states="state",
                reduction="weighted_state_mean_then_mean",
            ),
            function=execute,
            select=lambda data, selection: data.selected,
            implementations=(native(identity),),
        )
        self.operations[transition.symbol] = replace(
            self.operations[transition.symbol], mode="transition", copy_fields=(), publish=True
        )
        return transition

    def parameter_rows(self, source: SemanticInputSource | TensorPort) -> SemanticInputSource | TensorPort:
        if self.dimensions is None:
            return source
        port = _port(source)
        if port.rowwise:
            return port

        def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
            value = port.read(data, batch)
            return value.unsqueeze(0).expand(len(data.rows), *value.shape)

        return TensorPort(port.source, read)

    @staticmethod
    def state_metadata(state: State, *, version: bool) -> TensorPort:
        def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
            # Remote instance metadata never indexes local refs; test_video covers colliding lineage IDs.
            borrowed = borrowed_root(batch.borrowed_roots, str(state.declaration.resource))
            if borrowed is not None:
                tensors = batch.parameter_root(str(state.declaration.resource))
                return torch.full(
                    (len(data.rows),),
                    int(borrowed.ref.version_id) if version else 0,
                    device=tensors[0].device,
                    dtype=torch.long,
                )
            index = row_index(data, batch, batch.roots[0][0].device)
            return index.column("versions" if version else "states")

        return TensorPort(state.read(), read)

    def tensor(
        self,
        name: str,
        *,
        inputs: Mapping[str, SemanticInputSource | TensorPort],
        when: When = When.BEFORE_SAMPLE,
        identity: str = "",
        update: State | None = None,
        math: MathContract | None = None,
        output: str = "value",
        captures: Mapping[str, LayoutFieldId] = {},
    ) -> Callable[[TensorMath], Call]:
        sources = {key: value.source if isinstance(value, TensorPort) else value for key, value in inputs.items()}
        if captures and update is None:
            raise ValueError("memory captures require the task's own training transition")
        if any(key not in sources for key in captures):
            raise ValueError("memory captures must name a declared tensor input")
        if update is not None and not any(
            isinstance(source, StateSelectionInputSource) and source.resource == update.declaration.resource
            for source in sources.values()
        ):
            if "__update_state" in sources:
                raise ValueError("__update_state is reserved for the optimizer's state selection")
            sources["__update_state"] = update.read()
        if update is not None and (update.optimizer is None or tuple(self.graph.states) != (update.declaration,)):
            raise ValueError("ordinary training requires its sole declared instance optimizer state")
        policy = MathContract() if math is None else math
        reads = tuple(
            SemanticEffect(resource, StateEffectKind.READ)
            for resource in dict.fromkeys(
                source.resource
                for source in sources.values()
                if isinstance(source, StateSelectionInputSource)
                and (update is None or source.resource != update.declaration.resource)
            )
        )

        def decorate(function: TensorMath) -> Call:
            symbol = TaskSymbol(name)
            execute: Callable[[TensorFrame, TensorBatch], TaskOutput] | None = None
            # Discovery builds only the graph; executable readers require real model facts.
            if self.dimensions is not None:
                ports = {key: _port(value) for key, value in inputs.items()}
                domains = tuple(
                    dict.fromkeys(
                        key
                        for key in (
                            policy.forward_rows,
                            policy.input_gradient_rows,
                            policy.parameter_gradient_rows,
                            policy.update_states,
                        )
                        if key and not isinstance(sources[key], StateSelectionInputSource)
                    )
                )
                if policy.parameter_gradient_rows and any(
                    not port.rowwise
                    for port in ports.values()
                    if isinstance(port.source, StateSelectionInputSource)
                    and port.source.resource in policy.parameter_gradients
                ):
                    raise ValueError(
                        "parameter gradient rows require Parameter(..., per_row=True) for shared parameters"
                    )

                def values(data: TensorFrame, batch: TensorBatch) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
                    if policy.parameter_gradients:
                        leaf = state_leaf(data, batch)
                        batch = replace(batch, roots=_LeafRoots(leaf, batch.roots))
                    inputs = {key: ports[key].read(data, batch) for key in domains}
                    for key in domains:
                        mask = inputs[key]
                        if mask.dtype != torch.bool or mask.shape != (len(data.rows),):
                            raise ValueError("row domain must be a boolean mask over the original operation rows")
                    rows = (
                        tuple(int(index.item()) for index in inputs[policy.forward_rows].nonzero().flatten().cpu())
                        if policy.forward_rows
                        else tuple(range(len(data.rows)))
                    )
                    selected = (
                        replace(data, rows=tuple(data.rows[index] for index in rows), index=None)
                        if policy.forward_rows
                        else data
                    )
                    for key, port in ports.items():
                        source = sources[key]
                        state_rows = isinstance(source, StateSelectionInputSource) and port.rowwise
                        value = inputs[key] if key in domains else port.read(selected if state_rows else data, batch)
                        differentiable = key not in policy.detached_inputs and (
                            source.resource in policy.parameter_gradients
                            if isinstance(source, StateSelectionInputSource)
                            else key in policy.input_gradients
                        )
                        if not differentiable:
                            value = value.detach()
                        elif not isinstance(source, (StateSelectionInputSource, TaskResultInputSource)):
                            if not value.is_floating_point():
                                raise ValueError("input gradients require floating tensors")
                            value = value.detach().requires_grad_(True)
                            data.derivative_sources[str(symbol), key] = value
                        if policy.forward_rows and port.rowwise and key not in domains and not state_rows:
                            mask = inputs[policy.forward_rows]
                            if value.ndim == 0 or value.shape[0] != len(data.rows) or value.device != mask.device:
                                raise ValueError("forward rows require aligned logical row tensors on the same device")
                            value = select_rows(value, rows)
                        domain = (
                            policy.parameter_gradient_rows
                            if isinstance(source, StateSelectionInputSource)
                            else policy.input_gradient_rows
                        )
                        if differentiable and domain:
                            mask = select_rows(inputs[domain], rows) if policy.forward_rows else inputs[domain]
                            if (
                                not port.rowwise
                                or value.ndim == 0
                                or value.shape[0] != len(rows)
                                or mask.device != value.device
                            ):
                                raise ValueError("gradient rows require aligned logical row tensors on the same device")
                            if value.requires_grad:
                                value = _gradient_rows(value, mask.reshape(len(rows), *([1] * (value.ndim - 1))))
                        inputs[key] = value
                    return inputs, rows

                def evaluate(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
                    with torch.set_grad_enabled(bool(policy.input_gradients or policy.parameter_gradients)):
                        inputs, rows = values(data, batch)
                        result = function(inputs)
                        # Borrowed inputs cannot escape as result aliases; test_video exercises rejection.
                        for borrowed in batch.borrowed_roots:
                            borrowed.check_output(result)
                        if policy.forward_rows:
                            if result.ndim == 0 or result.shape[0] != len(rows):
                                raise ValueError("selected forward functions must preserve their logical row axis")
                            if len(rows) != len(data.rows):
                                result = result.new_zeros((len(data.rows), *result.shape[1:])).index_copy_(
                                    0, inputs[policy.forward_rows].nonzero().flatten(), result
                                )
                    batch.results[symbol, output] = result
                    return TaskOutput()

                execute = evaluate
                if update is not None:

                    def training(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
                        with torch.enable_grad():
                            inputs, rows = values(data, batch)
                            selected = rows
                            if policy.update_states in domains:
                                selected = tuple(
                                    int(index.item())
                                    for index in inputs[policy.update_states].nonzero().flatten().cpu()
                                )
                            indices = batch.selection.row_states
                            active = tuple(sorted({indices[data.rows[row].identity] for row in selected}))
                            if not rows:
                                outcome = TaskOutput(
                                    gradients=tuple(
                                        torch.zeros_like(state_leaf(data, batch)[index]) for index in active
                                    ),
                                    gradient_indices=active,
                                    update_count=len(active),
                                )
                            else:
                                loss = function(inputs)
                                objective = loss
                                if policy.reduction == "state_mean":
                                    if loss.shape != (len(batch.roots),):
                                        raise ValueError(
                                            "state_mean requires one mean loss per original selected state"
                                        )
                                    objective = loss.sum()
                                elif loss.ndim != 0:
                                    raise ValueError(
                                        "optimizer functions must return a scalar loss or declare state_mean"
                                    )
                                leaf = state_leaf(data, batch)
                                targets = (leaf, *data.derivative_sources.values())
                                derivatives = (
                                    torch.autograd.grad(objective, targets, allow_unused=True, materialize_grads=True)
                                    if objective.requires_grad
                                    else tuple(torch.zeros_like(target) for target in targets)
                                )
                                state_gradients = derivatives[0].detach()
                                gradients = tuple(state_gradients[index] for index in active)
                                data.parameter_gradient_fields.update(
                                    (ref, (value,))
                                    for ref, value in zip(batch.selection.refs, state_gradients.unbind(), strict=True)
                                )
                                data.input_gradients.update(
                                    (key, value.detach())
                                    for key, value in zip(data.derivative_sources, derivatives[1:], strict=True)
                                )
                                batch.results[symbol, output] = loss = loss.detach()
                                outcome = TaskOutput(
                                    loss=loss,
                                    gradients=gradients,
                                    gradient_indices=active,
                                    loss_count=len({indices[data.rows[row].identity] for row in rows})
                                    if policy.reduction == "state_mean"
                                    else 1,
                                    update_count=len(active),
                                    trained_rows=len(rows),
                                    observations=training_observations(
                                        data,
                                        batch,
                                        loss,
                                        {
                                            key: inputs[key]
                                            for key in inputs
                                            if isinstance(sources[key], HiddenInputSource)
                                        },
                                        rows,
                                    ),
                                )
                                if captures:
                                    # Each lineage's memory receives its last selected row's value.
                                    per_lineage = {indices[data.rows[row].identity]: row for row in rows}
                                    outcome = replace(
                                        outcome,
                                        field_writes=tuple(
                                            (
                                                field_id,
                                                torch.stack(
                                                    tuple(inputs[key][per_lineage[index]].detach() for index in active)
                                                ),
                                            )
                                            for key, field_id in captures.items()
                                        ),
                                    )
                        # validate_gradients binds every derivative path to this sole sink.
                        data.derivative_sources.clear()
                        data.leaf = None
                        for key, value in batch.results.items():
                            batch.results[key] = value.detach()
                        return outcome

                    execute = training
            return self.compute(
                name,
                inputs=sources,
                outputs=(output,),
                changes=update,
                math=math,
                result_schema="canonical.tensor_training_result@1" if update is not None else "",
                when=When.HOST if self.host else when,
                reads=reads,
                function=execute,
                implementations=(native(identity or f"{self.graph.workload_id}:{name}@1"),),
            )

        return decorate

    def validate_gradients(self, training: Call | None) -> None:
        tasks = {task.symbol: task for task in self.graph.tasks}
        retained = {
            task.symbol
            for task in tasks.values()
            if task.math is not None and (task.math.input_gradients or task.math.parameter_gradients)
        }
        if training is None:
            if retained:
                raise ValueError("differentiable operations require an explicit training sink")
            return
        visited: set[TaskSymbol] = set()
        resources: set[str] = set()

        def visit(symbol: TaskSymbol) -> None:
            if symbol in visited:
                return
            visited.add(symbol)
            task = tasks[symbol]
            if task.math is None:
                return
            resources.update(task.math.parameter_gradients)
            for port in task.inputs:
                if port.name in task.math.input_gradients and isinstance(port.source, TaskResultInputSource):
                    visit(port.source.task)

        visit(training.symbol)
        if retained - visited or not resources:
            raise ValueError("training must consume the declared differentiable parameter paths")

    def stage_update(self, loss: Call, state: State) -> Call:
        """Separate the original loss result from its private optimizer mutation."""
        task = next(task for task in self.graph.tasks if task.symbol == loss.symbol)
        if task.math is None or task.math.forward_rows or task.math.update_states == "__update_states":
            raise ValueError("staged state_mean requires an unconditional state update")
        if self.dimensions is not None:
            operation = self.operations[loss.symbol]

            def evaluate(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
                output = operation.function(data, batch)
                data.pending_update = output
                return TaskOutput(loss=output.loss, loss_count=output.loss_count, observations=output.observations)

            self.operations[loss.symbol] = replace(operation, function=evaluate, mode="read", optimizer=None)
        self.graph = replace(
            self.graph,
            tasks=tuple(
                replace(
                    task,
                    effects=tuple(replace(effect, kind=StateEffectKind.READ) for effect in task.effects),
                    math=None if task.math is None else replace(task.math, update_states=""),
                )
                if task.symbol == loss.symbol
                else task
                for task in self.graph.tasks
            ),
        )

        def update(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
            output = data.pending_update
            if output is None:
                raise RuntimeError("optimizer requires its planned loss result")
            data.pending_update = None
            return replace(output, loss=None, loss_count=0, observations=())

        transition = self.compute(
            "train" if loss.symbol != "train" else "train.update",
            inputs={"gradient": loss[loss.outputs[0]], "state": state.read()},
            outputs=("publication",),
            result_schema="canonical.page_publication@1",
            changes=state,
            when=When.AFTER_SAMPLE,
            math=MathContract(update_states="state"),
            function=update,
        )
        self.graph = replace(
            self.graph,
            tasks=tuple(
                replace(entry, timing=task.timing) if entry.symbol == transition.symbol else entry
                for entry in self.graph.tasks
            ),
        )
        return transition

    def logits(
        self,
        prediction: Call,
        training: Call,
        state: State,
        *,
        stage: SamplerStage = SamplerStage.PACKED_PRE_FILTER_ADD,
        beta: float = 0.0,
    ) -> None:
        if (
            stage is SamplerStage.PACKED_PRE_FILTER_ADD
            and self.dimensions is not None
            and self.dimensions.vocab_size < 1
        ):
            raise ValueError("logits binding requires the backend vocabulary size")
        if stage is SamplerStage.PACKED_POST_FILTER_EXACT:
            update = next(task for task in self.graph.tasks if task.symbol == training.symbol)
            if update.math is None or update.math.forward_rows or update.math.update_states == "__update_states":
                raise ValueError("post-filter guidance requires an unconditional request-state update")
            self.graph = replace(
                self.graph,
                tasks=tuple(
                    replace(
                        task,
                        effects=tuple(
                            replace(effect, kind=StateEffectKind.UPDATE)
                            if effect.resource == state.declaration.resource
                            else effect
                            for effect in task.effects
                        ),
                    )
                    if task.symbol == training.symbol
                    else task
                    for task in self.graph.tasks
                ),
            )
        prediction_task = next(task for task in self.graph.tasks if task.symbol == prediction.symbol)
        domain = "" if prediction_task.math is None else prediction_task.math.forward_rows
        mask = _port(next(port.source for port in prediction_task.inputs if port.name == domain)) if domain else None
        self.graph = replace(
            self.graph,
            tasks=tuple(
                replace(
                    task,
                    inputs=(
                        *task.inputs,
                        SemanticInput("packed_rows", self.rows()),
                        *(
                            (SemanticInput("__sampler_rows", self.sampler_rows()),)
                            if not any(isinstance(port.source, PackedSamplerRowsInputSource) for port in task.inputs)
                            else ()
                        ),
                    ),
                )
                if task.symbol == prediction.symbol
                else task
                for task in self.graph.tasks
            ),
        )
        if self.dimensions is not None:
            operation = self.operations[prediction.symbol]
            width = (
                self.dimensions.hidden_size
                if stage is SamplerStage.PACKED_POST_FILTER_EXACT
                else self.dimensions.vocab_size
            )

            def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
                operation.function(data, batch)
                result = batch.results[prediction.symbol, prediction.outputs[0]]
                if result.shape != (len(data.rows), width):
                    raise ValueError("sampler output differs from its signed row and feature dimensions")
                indices = {row.packed_row_index for row in data.rows}
                links = tuple(link for link in data.step.sampler_rows.links if link.packed_row_index in indices)
                if stage is SamplerStage.PACKED_POST_FILTER_EXACT:
                    local = {row.packed_row_index: index for index, row in enumerate(data.rows)}
                    selected = (
                        set(range(len(data.rows)))
                        if mask is None
                        else {int(index.item()) for index in mask.read(data, batch).nonzero().flatten().cpu()}
                    )
                    links = tuple(link for link in links if local[link.packed_row_index] in selected)
                    result = select_rows(result, tuple(local[link.packed_row_index] for link in links))
                observations = (
                    tuple(
                        Observation(
                            "guidance",
                            result[index],
                            data.step.rows.rows[link.packed_row_index],
                            batch.selection.refs[0]
                            if batch.selection.keys == (None,)
                            else batch.selection.ref_for(data.step.rows.rows[link.packed_row_index].identity),
                            data.step.rows.step_id,
                        )
                        for index, link in enumerate(links)
                    )
                    if batch.observing
                    else ()
                )
                return TaskOutput(
                    guidance=PackedSamplerGuidanceBatch(
                        data.step.sampler_rows,
                        result.detach(),
                        tuple(link.sampler_row_index for link in links),
                        beta,
                        stage=stage,
                    ),
                    observations=observations,
                )

            self.operations[prediction.symbol] = replace(operation, function=execute)
        self.sampler(prediction, training, state, output=prediction.outputs[0], stage=stage)

    def export(
        self, call: Call, *, name: str, every: int = 1, max_rows: int = 1, max_cols: int = 16, enabled: bool = True
    ) -> None:
        if min(every, max_rows, max_cols) < 1 or max_rows * max_cols * 8 > MAX_SNAPSHOT_BYTES:
            raise ValueError("tensor export requires positive stride and bounded snapshot dimensions")
        if self.dimensions is not None:
            operation = self.operations[call.symbol]

            def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
                if data.step.rows.step_id % every or not data.rows:
                    return TaskOutput()
                selected = replace(data, rows=data.rows[:max_rows], columns=max_cols)
                operation.function(selected, batch)
                value = (
                    batch.results[call.symbol, call.outputs[0]][:max_rows, :max_cols]
                    .detach()
                    .to(device="cpu", copy=True)
                )
                batch.results[call.symbol, call.outputs[0]] = value
                return TaskOutput(
                    observations=(
                        Observation(
                            name,
                            value,
                            step_id=data.step.rows.step_id,
                            identities=tuple(row.identity for row in selected.rows),
                            export=True,
                        ),
                    )
                )

            if not enabled:

                def execute(data: TensorFrame, batch: TensorBatch) -> TaskOutput:
                    return TaskOutput()

            self.operations[call.symbol] = replace(operation, function=execute)
        self.graph = replace(
            self.graph,
            tasks=tuple(
                replace(
                    task,
                    outputs=tuple(
                        replace(
                            value, kind=SemanticOutputKind.HOST, schema=HostPayloadSchemaId("canonical.hidden_export@1")
                        )
                        for value in task.outputs
                    ),
                )
                if task.symbol == call.symbol
                else task
                for task in self.graph.tasks
            ),
        )
        self.model_effect(call)

    def bind_prepare(self) -> Callable[[TensorStepInput, TensorBatch], TensorFrame]:
        dimensions = self.dimensions
        if dimensions is None:
            raise ValueError("tensor execution requires specialized model dimensions")
        if (
            self.attachment is not None
            and self.attachment.kind is AttachmentKind.MODEL_REPLAY
            and not self.attachment.activation_bindings
        ):

            def prepare_replay(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
                if (
                    value.auxiliary is not None
                    or value.operands
                    or value.hidden is not None
                    or value.rows.plan_epoch != batch.epoch
                ):
                    raise ValueError("tokenwise replay requires only original model rows at its signed epoch")
                return TensorFrame(value, value.rows.rows)

            return prepare_replay
        if self.attachment is not None and (self.attachment.activation_bindings or self.overlay_import):

            def prepare_replacement(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
                if value.rows.plan_epoch != batch.epoch or value.hidden is not None or value.operands:
                    raise ValueError("import input differs from its signed sources or epoch")
                update = value.auxiliary
                data = TensorFrame(value, value.rows.rows)
                if update is None:
                    return data
                if not isinstance(update, StateTensorUpdate):
                    raise ValueError("state imports require original state-field tensors")
                if not update.values:
                    return data
                selection = batch.selection
                if selection.keys == (None,):
                    if update.keys:
                        raise ValueError("instance replacement does not accept request keys")
                    data.replacements = ((0, update.values),)
                else:
                    if (
                        not update.keys
                        or len(set(update.keys)) != len(update.keys)
                        or any(
                            key.fields != (RequestKeyField.PROMPT_IDX,)
                            or key.request_id
                            or key.sample_idx != -1
                            or key.prompt_idx < 0
                            for key in update.keys
                        )
                        or any(tensor.ndim < 1 or tensor.shape[0] != len(update.keys) for tensor in update.values)
                    ):
                        raise ValueError(
                            "request replacement requires distinct original prompt keys and aligned field rows"
                        )
                    indices = {key: index for index, key in enumerate(update.keys)}
                    selected: dict[int, int] = {}
                    for row, index in zip(selection.rows, selection.state_indices, strict=True):
                        if (key := RequestStateKey.project(row.identity, (RequestKeyField.PROMPT_IDX,))) in indices:
                            if selection.refs[index].version_id != 0:
                                continue
                            # Several packed rows of one request share its state; two
                            # different import values for one root would be ambiguous.
                            if index in selected:
                                if selected[index] != indices[key]:
                                    raise ValueError(
                                        "one import batch cannot replace a shared request state with two values"
                                    )
                                continue
                            selected[index] = indices[key]
                    data.replacements = tuple(
                        (index, tuple(tensor[row] for tensor in update.values))
                        for index, row in sorted(selected.items())
                    )
                for index, values in data.replacements:
                    root = batch.roots[index]
                    if len(values) != len(root) or any(
                        source.shape != target.shape or source.dtype != target.dtype or source.device != target.device
                        for source, target in zip(values, root, strict=True)
                    ):
                        raise ValueError("replacement tensors differ from the signed layout")
                return data

            return prepare_replacement
        if self.overlay_targets:
            shapes = dict(zip(self.overlay_targets, dimensions.target_shapes, strict=True))

            def prepare_overlay(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
                expected = {(row.identity, target) for row in batch.selection.rows for target in self.overlay_targets}
                actual = tuple((operand.identity, operand.target) for operand in value.operands)
                if len(actual) != len(expected) or set(actual) != expected:
                    raise ValueError("captured operands must cover each selected request and target exactly once")
                if value.rows.plan_epoch != batch.epoch or value.auxiliary is not None or value.hidden is not None:
                    raise ValueError("overlay input differs from its signed sources or epoch")
                phases = {row.identity: row.phase for row in batch.selection.rows}
                for operand in value.operands:
                    inputs, outputs = shapes[operand.target]
                    if (
                        operand.phase != phases[operand.identity]
                        or operand.hidden.ndim != 2
                        or operand.hidden.shape[0] < 1
                        or operand.hidden.shape[1] != inputs
                        or operand.desired.shape != (operand.hidden.shape[0], outputs)
                        or operand.base_weight.shape != (outputs, inputs)
                        or any(
                            tensor.dtype != tensor_dtype(dimensions.dtype) or tensor.device != batch.roots[0][0].device
                            for tensor in (operand.hidden, operand.desired, operand.base_weight)
                        )
                    ):
                        raise ValueError("operand tensors differ from specialization")
                if self.probe is not None and any(
                    sum(operand.hidden.shape[0] for operand in value.operands if operand.target == target)
                    > self.probe.max_packed_rows
                    for target in self.overlay_targets
                ):
                    raise ValueError("overlay operands exceed the signed packed-row capacity")
                return TensorFrame(value, value.rows.request_rows)

            return prepare_overlay
        expected = {
            (str(source.layer), "index", source.site.value)
            for task in self.graph.tasks
            for port in task.inputs
            if isinstance(source := port.source, HiddenInputSource)
        }

        def rows(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
            return value.rows.decode_rows

        if self.attachment is not None and self.attachment.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD:

            def rows(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
                return tuple(
                    value.rows.rows[link.packed_row_index]
                    for link in value.sampler_rows.links
                    if value.rows.rows[link.packed_row_index].phase is HiddenPhase.DECODE
                )

        row_selection = rows
        if self.sample_limit is not None:
            limit = self.sample_limit
            select_rows = rows

            def row_selection(value: TensorStepInput) -> tuple[PackedPhaseRow, ...]:
                selected = select_rows(value)
                if any(row.identity.sample_idx >= limit for row in selected):
                    raise ValueError("sample index is outside configured iterations")
                return selected

        def prepare(value: TensorStepInput, batch: TensorBatch) -> TensorFrame:
            if value.auxiliary is not None and not self.weighted_training or value.operands:
                raise ValueError("ordinary hidden functions do not consume external state updates")
            if value.rows.plan_epoch != batch.epoch:
                raise ValueError("tensor input differs from the signed epoch")
            snapshots = value.hidden_batch(batch.stream).snapshots
            if self.probe is not None and tuple(snapshot.resource for snapshot in snapshots) != tuple(
                resource.binding for resource in self.probe.hidden_resources
            ):
                raise ValueError("sampler hidden input differs from its signed bindings")
            if {snapshot.resource.locator_key for snapshot in snapshots} != expected:
                raise ValueError("hidden input differs from its declared sources")
            if any(
                (self.probe is not None or batch.roots)
                and snapshot.hidden.shape[1] != dimensions.hidden_size
                or batch.roots
                and (
                    snapshot.hidden.device != batch.roots[0][0].device
                    or self.weighted_training
                    and snapshot.hidden.dtype != batch.roots[0][0].dtype
                )
                for snapshot in snapshots
            ):
                raise ValueError("sampler hidden input differs from parameter specialization")
            return TensorFrame(value, row_selection(value))

        return prepare

    @staticmethod
    def result(data: TensorFrame, invocation: InvocationOutput) -> TensorResult:
        return TensorResult(
            invocation.value, invocation, tuple(data.entries), data.input_gradients, data.parameter_gradient_fields
        )

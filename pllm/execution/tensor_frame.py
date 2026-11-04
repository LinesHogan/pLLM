"""What one task invocation reads: its row index, input ports and differentiable state parameters.

A TensorFrame carries one step's rows; ports read hidden states, sampler feedback, state metadata and
parameters through a RowPlan and its per-invocation device RowIndex. Used by TensorBinding and TensorTask.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, overload

import numpy as np
import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import Projection
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    InputTokensInputSource,
    PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource,
    SemanticInputSource,
    StateResource,
    TaskResultInputSource,
)
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_stream import upload
from pllm.execution.module_graph import ModuleCall
from pllm.execution.ptts.slot_pools import Slots
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.tensor_program import Observation, TensorBatch
from pllm.execution.torch_private import register_gradient_hook

if TYPE_CHECKING:
    from pllm.execution.row_task_binding import WeightedBatch
    from pllm.execution.step_input import StepInput


Tensors = Mapping[str, torch.Tensor]
TensorMath = Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]
TensorReader = Callable[["TensorFrame", TensorBatch], torch.Tensor]


def field_values(
    values: tuple[torch.Tensor, ...], targets: tuple[torch.Tensor, ...], *, resource: str, validate: bool
) -> tuple[torch.Tensor, ...]:
    """Normalize imported or authored fields to storage precision, checking metadata at their input boundary."""
    if validate and (
        len(values) != len(targets)
        or any(
            value.shape != target.shape
            or value.device != target.device
            or (value.dtype != target.dtype and not (value.is_floating_point() and target.is_floating_point()))
            for value, target in zip(values, targets, strict=True)
        )
    ):
        raise ValueError(
            f"write to {resource} needs fields with layout shape/device and floating or matching integer dtypes"
        )
    return tuple(value.to(target.dtype) for value, target in zip(values, targets, strict=True))


def flat_gradient(fields: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return fields[0].flatten() if len(fields) == 1 else torch.cat(tuple(value.flatten() for value in fields))


def gradient_rows(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Give this consumer its own gradient edge without copying the forward value.
    result = value.view_as(value)
    register_gradient_hook(result, lambda gradient: gradient.masked_fill(~mask, 0))
    return result


# An immutable default that torch.compile can trace, unlike a default_factory.
_NO_METRICS: Mapping[str, torch.Tensor] = MappingProxyType({})


@dataclass(frozen=True)
class UpdateOutput:
    """What a training or overlay method returns: its output and 0-d device counters.

    `output` is the loss for `@pllm.train` methods; for `@pllm.train_in_forward` it is the unscaled residual
    the overlay adds, excluding the base projection. `changed` (bool) says whether state changed,
    `update_count` (int64) counts applied updates and `trained_rows` (int64) the rows that trained.
    All three are 0-d tensors on the output's device, never Python scalars, so the step needs no host sync.
    `metrics` are extra 0-d device tensors returned in the step result.
    """

    output: torch.Tensor
    changed: torch.Tensor
    update_count: torch.Tensor
    trained_rows: torch.Tensor
    metrics: Mapping[str, torch.Tensor] = field(default_factory=lambda: _NO_METRICS)

    @classmethod
    def unchanged(cls, output: torch.Tensor) -> "UpdateOutput":
        """A residual-only result with device-valued zero counters."""
        return cls(
            output,
            output.new_zeros((), dtype=torch.bool),
            output.new_zeros((), dtype=torch.int64),
            output.new_zeros((), dtype=torch.int64),
        )


@dataclass(frozen=True, slots=True)
class OverlayTargetResult:
    target: Projection
    update: UpdateOutput


@dataclass(frozen=True, slots=True)
class OverlayEntry:
    identity: RequestIdentity
    before_ref: StateRef
    after_ref: StateRef
    targets: tuple[OverlayTargetResult, ...]

    @property
    def changed(self) -> torch.Tensor:
        return torch.stack(tuple(target.update.changed for target in self.targets)).any()


@dataclass(frozen=True, slots=True)
class TensorPort:
    """A graph input, its tensor reader and the value handed to the declared method."""

    source: SemanticInputSource
    read: TensorReader
    rowwise: bool = True
    field: int | None = None
    row_input: str = ""
    adapt: Callable[[torch.Tensor], torch.Tensor | ModuleCall] = lambda value: value


def take_rows(values: torch.Tensor, rows: Sequence[int]) -> torch.Tensor:
    """`values[rows]` along dim 0: a view when the rows repeat one row or run consecutively, else one gather."""
    first = rows[0] if rows else 0
    if len(set(rows)) == 1:
        return values[first : first + 1].expand(len(rows), *values.shape[1:])
    if tuple(rows) == tuple(range(first, first + len(rows))):
        return values.narrow(0, first, len(rows))
    return values.index_select(0, upload(rows, values.device))


@dataclass(frozen=True, slots=True)
class RowPlan:
    """Host row facts and authorized slots that TensorExecutor stages into its fixed table.

    Columns serialize in sampler/samples/states/versions/packed order; sources are [state, field] slots. Writes pair the
    update-state positions with their issued destination slots.
    """

    packed: Slots
    sampler: tuple[int, ...]
    samples: tuple[int, ...]
    states: tuple[int, ...]
    versions: tuple[int, ...]
    sources: Slots
    writes: tuple[tuple[int, ...], Slots] | None = None

    @classmethod
    def build(cls, data: "TensorFrame", batch: TensorBatch) -> "RowPlan":
        """Resolve operation rows once from the invocation's input and page access."""
        links = {link.packed_row_index: link.sampler_row_index for link in data.step.sampler_rows.links}
        selection = batch.selection
        states = (
            selection.indices(data.rows) if selection.groups and selection.keys != (None,) else (0,) * len(data.rows)
        )
        return cls(
            np.fromiter((row.packed_row_index for row in data.rows), dtype=np.int64, count=len(data.rows)),
            tuple(links.get(row.packed_row_index, -1) for row in data.rows),
            tuple(row.identity.sample_index for row in data.rows),
            states,
            tuple(selection.refs.versions[list(states)].tolist()) if selection.groups else states,
            batch.slot_matrix()[1] if batch.roots else np.empty((0, 0), dtype=np.int64),
        )

    @property
    def columns(self) -> tuple[tuple[int, ...], ...]:
        """Serialize named fields only at the device-table boundary."""
        return self.sampler, self.samples, self.states, self.versions, tuple(self.packed.tolist())

    @property
    def last(self) -> dict[int, int]:
        """Last row position of every state with rows."""
        return {state: position for position, state in enumerate(self.states)}


class RowIndex(NamedTuple):
    """An invocation's fixed device row columns (padding rows point at the sink state)."""

    sampler: torch.Tensor
    samples: torch.Tensor
    states: torch.Tensor
    versions: torch.Tensor
    packed: torch.Tensor


def row_plan(data: "TensorFrame", batch: TensorBatch) -> RowPlan:
    """The invocation's host row facts, built on first use; tasks without row inputs never need them."""
    if data.plan is None:
        data.plan = RowPlan.build(data, batch)
    return data.plan


class FieldRoots(Sequence[tuple[torch.Tensor, ...]]):
    """Lazy per-state views of gathered fields; request ports usually need no root views."""

    def __init__(self, fields: tuple[torch.Tensor, ...]) -> None:
        self.fields = fields

    def __len__(self) -> int:
        return len(self.fields[0]) if self.fields else 0

    @overload
    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[torch.Tensor, ...], ...]: ...

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        return tuple(value[index] for value in self.fields)


class LeafRoots(Sequence[tuple[torch.Tensor, ...]]):
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


class ParameterGradients(MutableMapping[StateRef, tuple[torch.Tensor, ...]]):
    """Stacked parameter derivatives; per-state views are created only when inspected."""

    def __init__(
        self, refs: StateRefs, values: torch.Tensor, previous: Mapping[StateRef, tuple[torch.Tensor, ...]]
    ) -> None:
        if isinstance(previous, ParameterGradients) and previous._fields is None and previous.refs == refs:
            previous = previous.previous
        self.refs, self.stacked = refs, values
        self.previous: Mapping[StateRef, tuple[torch.Tensor, ...]] = previous
        self._fields: dict[StateRef, tuple[torch.Tensor, ...]] | None = None

    def _materialize(self) -> dict[StateRef, tuple[torch.Tensor, ...]]:
        if self._fields is None:
            self._fields = dict(self.previous)
            self._fields.update((ref, (value,)) for ref, value in zip(self.refs, self.stacked.unbind(), strict=True))
        return self._fields

    def __getitem__(self, ref: StateRef) -> tuple[torch.Tensor, ...]:
        return self._materialize()[ref]

    def __setitem__(self, ref: StateRef, values: tuple[torch.Tensor, ...]) -> None:
        self._materialize()[ref] = values

    def __delitem__(self, ref: StateRef) -> None:
        del self._materialize()[ref]

    def __iter__(self) -> Iterator[StateRef]:
        return iter(self._materialize())

    def __len__(self) -> int:
        return len(self._materialize())


@dataclass(frozen=True, slots=True)
class FixedRows:
    """The gathered inputs of one fixed-table body, built by TensorExecutor; every row read goes through it.

    `fields` hold each field's pages for the `states` states and the padding sink at index `states`,
    `index` the device row columns,
    `active` marks real rows (padding rows are False), `hidden` the hidden slots per locator and `row_inputs`
    the indirectly addressed token and sampler rows. `leaf` is the differentiable copy of the flat parameter
    field when the body differentiates parameters; `resources` are the tables of other resources the task reads.
    """

    fields: tuple[torch.Tensor, ...]
    states: int
    index: RowIndex
    active: torch.Tensor
    hidden: Mapping[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]
    row_inputs: Mapping[str, torch.Tensor]
    leaf: torch.Tensor | None = None
    resources: Mapping[StateResource, FixedRows] = field(default_factory=lambda: dict[StateResource, "FixedRows"]())

    def for_resource(self, resource: StateResource) -> FixedRows:
        """The table of `resource`: this one, or that of another resource the task also reads."""
        return self.resources.get(resource, self)


@dataclass(slots=True)
class TensorFrame:
    step: StepInput
    rows: tuple[PackedPhaseRow, ...]
    entries: list[OverlayEntry] = field(default_factory=lambda: list[OverlayEntry]())
    input_gradients: dict[tuple[str, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[str, str], torch.Tensor]()
    )
    parameter_gradient_fields: MutableMapping[StateRef, tuple[torch.Tensor, ...]] = field(
        default_factory=lambda: dict[StateRef, tuple[torch.Tensor, ...]]()
    )
    training: UpdateOutput | None = None
    # A host training step's updated state fields, by name (StepResult.parameters).
    parameters: dict[str, torch.Tensor] = field(default_factory=lambda: dict[str, torch.Tensor]())
    replacements: tuple[tuple[int, tuple[torch.Tensor, ...]], ...] = ()
    plan: RowPlan | None = None
    # Set only on the copy a fixed-table body evaluates.
    fixed: FixedRows | None = None
    # Owned by row_task_binding's weighted batch training.
    weighted: WeightedBatch | None = None

    def producer_rows(self, phase: HiddenPhase | None) -> tuple[PackedPhaseRow, ...]:
        """The producer's eligible rows in this frame's order, for both its math and destinations."""
        if phase is None:
            return self.rows
        eligible = {row.packed_row_index for row in self.step.rows.producer_rows(phase)}
        return tuple(row for row in self.rows if row.packed_row_index in eligible)

    @property
    def table(self) -> FixedRows:
        """The fixed-table inputs; only fixed bodies read rows."""
        assert self.fixed is not None
        return self.fixed


def feedback_port(name: Literal["base_logits", "sampled_tokens"]) -> TensorPort:
    return TensorPort(SamplerFeedbackInputSource(), lambda data, batch: data.table.row_inputs[name], row_input=name)


def training_observations(
    data: TensorFrame, batch: TensorBatch, losses: torch.Tensor, inputs: Tensors
) -> tuple[Observation, ...]:
    if not batch.observing:
        return ()
    return tuple(
        Observation(name, tensor, row, ref, data.step.rows.step_id, gradient)
        for position, row in enumerate(data.rows)
        for index in (batch.selection.row_states[row.identity],)
        for ref in (batch.selection.refs[index],)
        for name, tensor, gradient in (
            *((name, tensor[position], False) for name, tensor in inputs.items()),
            ("loss", losses if losses.ndim == 0 else losses[index], False),
            ("parameters.grad", flat_gradient(data.parameter_gradient_fields[ref]), True),
        )
    )


def hidden_port(source: HiddenInputSource) -> TensorPort:
    """Gather the declared module's slots, promoting model dtypes for FP32 consumer math."""

    def hidden(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        value, slots = data.table.hidden[str(source.layer), "index", source.site.value]
        result = value.index_select(0, slots)
        return result.to(torch.float32) if result.dtype in (torch.float16, torch.bfloat16) else result

    return TensorPort(source, hidden)


def layer_logits_port(source: HiddenInputSource, columns: tuple[int, ...]) -> TensorPort:
    """Gather the step's layer logits, [rows, layers, vocab] in FP32, keeping the marker's `columns` of the probed
    layers; views and a stack keep the read free of host-to-device copies, so captured bodies replay it."""

    def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        value, slots = data.table.hidden[str(source.layer), "index", source.site.value]
        rows = value.index_select(0, slots)
        return rows if columns == tuple(range(rows.shape[1])) else torch.stack([rows[:, c] for c in columns], 1)

    return TensorPort(source, read)


def sample_index_port(source: PackedSamplerRowsInputSource) -> TensorPort:
    """Read packed sample indices for the invocation rows."""

    return TensorPort(source, lambda data, batch: data.table.index.samples)


def input_tokens_port(history: int) -> TensorPort:
    """Read the invocation's current token and lookbacks, in row order."""

    def read(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        return data.table.row_inputs["input_tokens"][:, : history + 1]

    return TensorPort(InputTokensInputSource(history), read, row_input="input_tokens")


def result_port(source: TaskResultInputSource) -> TensorPort:
    """Read a previously evaluated task's named output."""
    return TensorPort(source, lambda data, batch: batch.results[source.task, source.output])

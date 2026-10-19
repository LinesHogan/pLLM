"""What one task invocation reads: its row index, input ports and differentiable state parameters.

A TensorFrame carries one step's rows; ports read hidden states, sampler feedback, state metadata and
parameters for those rows through a single per-invocation RowIndex. Used by TensorBinding and TensorTask.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, overload

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource,
    SemanticInputSource,
    TaskResultInputSource,
)
from pllm.contracts.state_ref import StateRef
from pllm.execution.codec import register_gradient_hook
from pllm.execution.cuda_stream import upload
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.tensor_program import Observation, TensorBatch

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


def flat_gradient(fields: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return fields[0].flatten() if len(fields) == 1 else torch.cat(tuple(value.flatten() for value in fields))


def gradient_rows(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Give this consumer its own gradient edge without copying the forward value.
    result = value.view_as(value)
    register_gradient_hook(result, lambda gradient: gradient.masked_fill(~mask, 0))
    return result


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
    field: int | None = None
    feedback: str = ""


def take_rows(values: torch.Tensor, rows: Sequence[int], device_rows: torch.Tensor | None = None) -> torch.Tensor:
    """`values[rows]` along dim 0: a view when the rows repeat one row or run consecutively, else one gather."""
    first = rows[0] if rows else 0
    if len(set(rows)) == 1:
        return values[first : first + 1].expand(len(rows), *values.shape[1:])
    if tuple(rows) == tuple(range(first, first + len(rows))):
        return values.narrow(0, first, len(rows))
    if device_rows is None:
        device_rows = upload(rows, values.device)
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

    fixed: bool = False

    def column(self, name: str) -> torch.Tensor:
        return self.device[_COLUMNS.index(name)]

    def take(self, values: torch.Tensor, name: str) -> torch.Tensor:
        """Rows of `values` selected by one column, in operation-row order."""
        if self.fixed:
            return values.index_select(0, self.column(name))
        rows: tuple[int, ...] = getattr(self, name)
        if -1 in rows:
            raise ValueError(f"some rows have no {name} row")
        return take_rows(values, rows, self.column(name))

    @classmethod
    def build(cls, data: "TensorFrame", batch: TensorBatch, device: torch.device) -> "RowIndex":
        links = {link.packed_row_index: link.sampler_row_index for link in data.step.sampler_rows.links}
        selection = batch.selection
        states = selection.indices(data.rows) if selection.groups else (0,) * len(data.rows)
        columns = (
            tuple(row.packed_row_index for row in data.rows),
            tuple(links.get(row.packed_row_index, -1) for row in data.rows),
            tuple(row.identity.sample_idx for row in data.rows),
            states,
            tuple(selection.refs.versions[list(states)].tolist()) if selection.groups else states,
        )
        return cls(*columns, device=upload(columns, device))


def row_index(data: "TensorFrame", batch: TensorBatch, device: torch.device) -> RowIndex:
    """The invocation's row tables, built on first use."""
    if data.index is None:
        data.index = RowIndex.build(data, batch, device)
    return data.index


def field_rows(data: "TensorFrame", batch: TensorBatch, field: int, start: int, size: int) -> torch.Tensor:
    """`field[start:start+size]` of each row's state, `[rows, size]`; copies only that slice."""
    if data.fields:
        value = data.fields[field].reshape(len(data.fields[field]), -1).narrow(1, start, size)
        return row_index(data, batch, value.device).take(value, "states")
    pool, slots = batch.field_slots(field)
    view = pool.reshape(len(pool), -1).narrow(1, start, size)
    rows = tuple(slots[state] for state in row_index(data, batch, pool.device).states)
    return take_rows(view, rows)


def state_rows(batch: TensorBatch, field: int) -> torch.Tensor:
    """One row per selected state: `[states, *field_shape]`, a view when there is a single state."""
    pool, slots = batch.field_slots(field)
    if len(slots) == 1:
        return pool[slots[0] : slots[0] + 1]
    return pool.index_select(0, upload(slots, pool.device))


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


def state_means(losses: torch.Tensor, states: Sequence[int], count: int) -> torch.Tensor:
    """Mean per-row loss of each of `count` states (0 for a state without rows).

    Row-to-state membership is known on the host, so this needs no device-to-host read.
    """
    sizes = Counter(states)
    totals = losses.new_zeros(count).index_add(0, upload(states, losses.device), losses)
    return totals / upload([max(sizes[state], 1) for state in range(count)], losses.device, losses.dtype)


def state_leaf(data: "TensorFrame", batch: TensorBatch) -> torch.Tensor:
    """The invocation's differentiable parameters, created on first use."""
    if data.leaf is None:
        data.leaf = state_rows(batch, 0).detach().requires_grad_(True)
    return data.leaf


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
    parameter_gradient_fields: MutableMapping[StateRef, tuple[torch.Tensor, ...]] = field(
        default_factory=lambda: dict[StateRef, tuple[torch.Tensor, ...]]()
    )
    batch_inputs: tuple[torch.Tensor, ...] = ()
    selected: tuple[int, ...] = ()
    trained_rows: int = 0
    replacements: tuple[tuple[int, tuple[torch.Tensor, ...]], ...] = ()
    index: RowIndex | None = None
    # Differentiable [states, parameters] copy of the flat optimizer field, shared by the
    # invocation's differentiable tasks and cleared by the training task that consumes it.
    leaf: torch.Tensor | None = None
    fields: tuple[torch.Tensor, ...] = ()
    weights: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    hidden_inputs: Mapping[tuple[str, str, str], torch.Tensor] | None = None
    feedback_inputs: Mapping[str, torch.Tensor] | None = None


def feedback_port(name: Literal["base_logits", "sampled_tokens"]) -> TensorPort:
    def resolve(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
        if data.feedback_inputs is not None:
            return data.feedback_inputs[name]
        feedback = batch.feedback
        if feedback is None:
            raise RuntimeError("tensor function requires completed sampler feedback")
        values = feedback.base_logits if name == "base_logits" else feedback.sampled_tokens
        return row_index(data, batch, values.device).take(values, "sampler")

    return TensorPort(SamplerFeedbackInputSource(), resolve, feedback=name)


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
            ("parameters.grad", flat_gradient(data.parameter_gradient_fields[ref]), True),
        )
    )


def input_port(source: SemanticInputSource | TensorPort) -> TensorPort:
    match source:
        case TensorPort():
            return source
        case HiddenInputSource(layer=layer, site=site, phase=HiddenPhase.DECODE):

            def hidden(data: TensorFrame, batch: TensorBatch) -> torch.Tensor:
                if data.hidden_inputs is not None:
                    value = data.hidden_inputs[str(layer), "index", site.value]
                    return row_index(data, batch, value.device).take(value, "packed")[:, : data.columns]
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
                if data.index is not None:
                    return data.index.column("samples")
                device = data.step.hidden_batch(batch.stream).snapshots[0].hidden.device
                return row_index(data, batch, device).column("samples")

            return TensorPort(source, samples)
        case SamplerFeedbackInputSource():
            return feedback_port("base_logits")
        case TaskResultInputSource(task=task, output=output):
            return TensorPort(source, lambda data, batch: batch.results[task, output])
        case _:
            raise ValueError("unsupported ordinary tensor input; use an explicit native binding")

"""What one task invocation reads: its row index, input ports and differentiable state parameters.

A TensorFrame carries one step's rows; ports read hidden states, sampler feedback, state metadata and
parameters through a RowPlan and its per-invocation device RowIndex. Used by TensorBinding and TensorTask.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
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
from pllm.execution.ptts.page_bundle import Slots
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
    """Private overlay updates and their unscaled residual, excluding the base projection."""

    output: torch.Tensor
    changed: torch.Tensor
    update_count: torch.Tensor
    trained_rows: torch.Tensor
    metrics: Mapping[str, torch.Tensor] = field(default_factory=lambda: dict[str, torch.Tensor]())

    @classmethod
    def unchanged(cls, output: torch.Tensor) -> "TensorUpdate":
        """A residual-only result with device-valued zero counters."""
        return cls(
            output,
            output.new_zeros((), dtype=torch.bool),
            output.new_zeros((), dtype=torch.int64),
            output.new_zeros((), dtype=torch.int64),
        )


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
    def changed(self) -> torch.Tensor:
        return torch.stack(tuple(target.update.changed for target in self.targets)).any()


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
class RowPlan:
    """Host row facts and authorized slots shared by eager views and fixed tables.

    Columns follow `_COLUMNS`; sources are [state, field] slots. Writes pair the
    update-state positions with their issued destination slots. Selecting rows
    changes only columns; page access stays in the invocation's state order.
    """

    columns: tuple[tuple[int, ...], ...]
    sources: Slots
    writes: tuple[tuple[int, ...], Slots] | None = None

    @classmethod
    def build(cls, data: "TensorFrame", batch: TensorBatch) -> "RowPlan":
        """Resolve operation rows once from the invocation's input and page access."""
        links = {link.packed_row_index: link.sampler_row_index for link in data.step.sampler_rows.links}
        selection = batch.selection
        states = selection.indices(data.rows) if selection.groups else (0,) * len(data.rows)
        return cls(
            (
                tuple(row.packed_row_index for row in data.rows),
                tuple(links.get(row.packed_row_index, -1) for row in data.rows),
                tuple(row.identity.sample_idx for row in data.rows),
                states,
                tuple(selection.refs.versions[list(states)].tolist()) if selection.groups else states,
            ),
            batch.slot_matrix()[1] if batch.roots else np.empty((0, 0), dtype=np.int64),
        )

    def select(self, rows: tuple[int, ...]) -> "RowPlan":
        """Keep operation-row order; capture positions refer to the compact inputs."""
        return replace(self, columns=tuple(tuple(column[row] for row in rows) for column in self.columns))

    @property
    def counts(self) -> Slots:
        """Per-state row counts, followed by the padding sink."""
        return np.bincount(self.columns[3], minlength=len(self.sources) + 1).astype(np.int64, copy=False)

    @property
    def last(self) -> dict[int, int]:
        """Last input position of every state with forward rows."""
        return {state: position for position, state in enumerate(self.columns[3])}


@dataclass(frozen=True, slots=True)
class RowIndex:
    """Device columns with an eager RowPlan, or fixed columns supplied by graph staging."""

    device: torch.Tensor
    plan: RowPlan | None = None

    def column(self, name: str) -> torch.Tensor:
        return self.device[_COLUMNS.index(name)]

    def take(self, values: torch.Tensor, name: str) -> torch.Tensor:
        """Borrow contiguous/repeated eager rows, or gather from fixed device columns."""
        if self.plan is None:
            return values.index_select(0, self.column(name))
        rows = self.plan.columns[_COLUMNS.index(name)]
        if -1 in rows:
            raise ValueError(f"some rows have no {name} row")
        return take_rows(values, rows, self.column(name))

    def select(self, rows: tuple[int, ...]) -> "RowIndex":
        """Select compact inputs without resolving their host facts again."""
        return RowIndex(select_rows(self.device.T, rows).T, None if self.plan is None else self.plan.select(rows))


def row_index(data: "TensorFrame", batch: TensorBatch, device: torch.device) -> RowIndex:
    """The invocation's row tables, built on first use."""
    if data.index is None:
        plan = RowPlan.build(data, batch)
        data.index = RowIndex(upload(plan.columns, device), plan)
    return data.index


def field_rows(data: "TensorFrame", batch: TensorBatch, field: int, start: int, size: int) -> torch.Tensor:
    """`field[start:start+size]` of each row's state, `[rows, size]`; copies only that slice."""
    if data.fields:
        value = data.fields[field].reshape(len(data.fields[field]), -1).narrow(1, start, size)
        return row_index(data, batch, value.device).take(value, "states")
    pool, _ = batch.field_slots(field)
    view = pool.reshape(len(pool), -1).narrow(1, start, size)
    plan = row_index(data, batch, pool.device).plan
    assert plan is not None
    rows = tuple(int(plan.sources[state, field]) for state in plan.columns[3])
    return take_rows(view, rows)


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


def state_means(losses: torch.Tensor, plan: RowPlan) -> torch.Tensor:
    """Average compact row losses using the same host counts as fixed table staging."""
    totals = losses.new_zeros(len(plan.sources)).index_add(0, upload(plan.columns[3], losses.device), losses)
    return totals / upload([max(int(count), 1) for count in plan.counts[:-1]], losses.device, losses.dtype)


def state_leaf(data: "TensorFrame", batch: TensorBatch) -> torch.Tensor:
    """The invocation's differentiable parameters, created on first use."""
    if data.leaf is None:
        pool, slots = batch.field_slots(0)
        data.leaf = (
            (pool[slots[0] : slots[0] + 1] if len(slots) == 1 else pool.index_select(0, upload(slots, pool.device)))
            .detach()
            .requires_grad_(True)
        )
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


def selected_rows(mask: torch.Tensor) -> tuple[int, ...]:
    """Read a host row selection; shape tracing covers every potentially active row."""
    return (
        tuple(range(len(mask)))
        if mask.is_meta
        else tuple(int(index.item()) for index in mask.nonzero().flatten().cpu())
    )


def finite(value: torch.Tensor) -> bool:
    """Validate tensor contents; meta execution can trace the check but cannot inspect its result."""
    result = torch.isfinite(value).all()
    return value.is_meta or bool(result)


def export_tensor(value: torch.Tensor) -> torch.Tensor:
    """Own an export copy; meta has no data to transfer to a host allocation."""
    return value.detach().to(device=value.device if value.is_meta else "cpu", copy=True)

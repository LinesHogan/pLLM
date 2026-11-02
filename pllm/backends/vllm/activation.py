"""pllm.model.ActivationPatch on vLLM: the patched rows of a step, the fields of the versions they read, and the edit
model_sites applies to a layer's MLP output. Patched rows are the first and last prompt tokens per request (and
decode tokens when enabled); versions follow the patch's refresh policy, borrowing admission or invocation pins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from torch.nn import functional as F

from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    DecodeRefresh,
    DecodeWindow,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import StateEffectKind, StateResource
from pllm.contracts.state_ref import StateRef
from pllm.execution.cuda_stream import upload
from pllm.execution.ptts.page_bundle import PagePin
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import DataT, StateTaskContext


def activation_delta(hidden: torch.Tensor, fields: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The activation edit of `hidden` rows: an added vector, or the low-rank affine edit R (W h + b - R^T h).

    Fields are either shared by every row or stacked per row along dim 0; the forward patch, route waves
    and native spans all apply this one function.
    """
    if len(fields) == 1:
        return fields[0].to(hidden.dtype)
    weight, bias, rotation = fields
    if weight.ndim == 2:
        # Shared fields keep F.linear (addmm, bias fused in the epilogue), as before batched fields existed.
        return (F.linear(hidden, weight, bias) - hidden @ rotation) @ rotation.T
    rows = hidden.unsqueeze(-2)
    coefficients = rows @ weight.transpose(-1, -2) + bias.unsqueeze(-2) - rows @ rotation
    return (coefficients @ rotation.transpose(-1, -2)).squeeze(-2)


@dataclass(frozen=True, slots=True)
class _PatchRows:
    """One request's patched rows at one layer and the fields of the version it reads."""

    rows: torch.Tensor
    fields: tuple[torch.Tensor, ...]
    indices: tuple[int, ...]


class VllmActivationPatch:
    """pllm.model.ActivationPatch rows and fields for one step. The patch is a custom residual of each patched layer's
    MLP: model_sites adds the edit to the MLP output before the residual add, through `patch`.

    In a custom forward a layer call covers some packed rows, maybe replayed from a CUDA graph: each patched row's
    fields are also copied into stable per-row buffers, and `presence` tells the custom forward which layers patch a
    row so its waves group rows that patch the same layers.
    """

    def __init__(self, bindings: tuple[ActivationPatchBinding, ...], rows: int, *, in_program: bool) -> None:
        self.bindings, self.in_program = bindings, in_program
        self.prepared: dict[tuple[StateResource, int], tuple[_PatchRows, ...]] = {}
        # Per binding, stable [rows, *field] copies of each packed row's fields (zeros for unpatched rows).
        self._row_fields: dict[tuple[StateResource, int], tuple[torch.Tensor, ...]] = {}
        self._capacity = rows
        self._patched: dict[int, tuple[bool, ...]] = {}

    def prepare(
        self, rows: PackedPhaseRowBatch, context: StateTaskContext[DataT], directory: RequestStateDirectory
    ) -> None:
        held = context.held_input
        assert held is None or (
            held.step_id == rows.step_id
            and {identity for identity, _, _ in held.successful} == {row.identity for row in rows.rows}
        )
        context.input_step_id = rows.step_id
        versions: dict[tuple[StateResource, RequestIdentity, bool], tuple[PagePin, StateRef]] = {}
        self.prepared.clear()
        for binding in self.bindings:
            prepared: list[_PatchRows] = []
            for phase, phase_rows in ((HiddenPhase.PREFILL, rows.prefill_rows), (HiddenPhase.DECODE, rows.decode_rows)):
                if phase is HiddenPhase.DECODE and binding.decode is DecodeWindow.NONE:
                    continue
                refresh = phase is HiddenPhase.DECODE and binding.decode_refresh is DecodeRefresh.STEP
                for identity in dict.fromkeys(row.identity for row in phase_rows):
                    source = tuple(row for row in phase_rows if row.identity == identity)
                    selected = tuple(
                        row.packed_row_index
                        for index, row in enumerate(source)
                        if phase is HiddenPhase.DECODE
                        or index < binding.first_tokens
                        or index >= len(source) - binding.last_tokens
                    )
                    if not selected:
                        continue
                    key = (binding.resource, identity, refresh)
                    if key not in versions:
                        pin, ref = self._version(binding.resource, identity, refresh, context, directory)
                        if self.in_program or context.arena.plan.semantic_graph.input_groups:
                            # Admission pins belong to the request directory. Borrow
                            # the same exact version for this invocation's source view.
                            pin = context.pin_for(StateRefs.of((ref,)))
                        versions[key] = pin, ref
                    pin, ref = versions[key]
                    assert held is None or any(
                        identity == held_identity and ref in refs for held_identity, refs, _ in held.successful
                    )
                    context.injected_refs.setdefault(identity, set()).add(ref)
                    fields = tuple(
                        view.read(context.arena.borrow_field(pin, ref, view.field)) for view in binding.fields
                    )
                    prepared.append(_PatchRows(upload(selected, context.arena.device), fields, selected))
            self.prepared[binding.resource, binding.layer] = tuple(prepared)
        if self.in_program:
            self._stage_rows(rows.total_rows)

    def _stage_rows(self, count: int) -> None:
        """Copy every patched row's fields into the stable per-row buffers a custom forward gathers from."""
        self._patched = {}
        for ordinal, binding in enumerate(self.bindings):
            key = (binding.resource, binding.layer)
            items = self.prepared[key]
            if key not in self._row_fields and items:
                self._row_fields[key] = tuple(
                    field.new_zeros((self._capacity, *field.shape)) for field in items[0].fields
                )
            for buffer in self._row_fields.get(key, ()):
                buffer[:count].zero_()
            for item in items:
                for buffer, field in zip(self._row_fields[key], item.fields, strict=True):
                    buffer.index_copy_(0, item.rows, field.expand(len(item.indices), *field.shape))
                for row in item.indices:
                    marks = self._patched.setdefault(row, (False,) * len(self.bindings))
                    self._patched[row] = (*marks[:ordinal], True, *marks[ordinal + 1 :])

    def presence(self, row: int) -> tuple[bool, ...]:
        """Which bindings patch packed row `row` this step."""
        return self._patched.get(row, (False,) * len(self.bindings))

    def patch(self, resource: StateResource, layer: int, rows: torch.Tensor | None, packed: tuple[int, ...]) -> Patch:
        """The edit of layer `layer`'s MLP output for the current call: every packed row (`rows` None), or the packed
        rows `rows` (host copy `packed`) a custom forward runs."""
        items = self.prepared[resource, layer]
        if rows is None:
            return partial(_patch_step, items)
        patched = {row for item in items for row in item.indices}
        lanes = tuple(lane for lane, row in enumerate(packed) if row in patched)
        return partial(_patch_lanes, self._row_fields.get((resource, layer), ()), rows, lanes, len(packed))

    @staticmethod
    def _version(
        resource: StateResource,
        identity: RequestIdentity,
        refresh: bool,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
    ) -> tuple[PagePin, StateRef]:
        ((pin, ref),) = context.read_versions((identity,), resource, directory, latest=refresh)
        if (
            refresh
            and ref in context.selection.for_resource(resource).refs
            and context.held_input is None
            and any(
                task.timing.after == "request.prepare"
                and any(
                    effect.resource == resource and effect.kind is not StateEffectKind.READ for effect in task.effects
                )
                for task in context.arena.plan.semantic_graph.tasks
            )
        ):
            # No implicit HOLD: its signed no-input branch is a separate lifecycle.
            raise RuntimeError("decode STEP requires a new publication; missing-input handling is not enabled")
        return pin, ref

    def clear(self) -> None:
        self.prepared.clear()
        self._patched = {}


Patch = Callable[[torch.Tensor], torch.Tensor]


def _patch_step(items: tuple[_PatchRows, ...], hidden: torch.Tensor) -> torch.Tensor:
    """Each request's edit on its patched rows of the whole step, with that request's shared fields."""
    for item in items:
        # An added vector needs no gather of the patched rows.
        rows = hidden if len(item.fields) == 1 else hidden.index_select(0, item.rows)
        hidden.index_add_(0, item.rows, activation_delta(rows, item.fields).expand(item.rows.numel(), -1))
    return hidden


def _patch_lanes(
    fields: tuple[torch.Tensor, ...], rows: torch.Tensor, lanes: tuple[int, ...], width: int, hidden: torch.Tensor
) -> torch.Tensor:
    """The edit on the patched lanes of a custom forward's layer call, with each lane's fields gathered from the
    stable per-row buffers (a CUDA-graph wave's lanes all patch the same layers)."""
    if not lanes:
        return hidden
    if len(lanes) == width:
        return hidden + activation_delta(hidden, tuple(field.index_select(0, rows) for field in fields))
    selected = upload(lanes, hidden.device)
    packed = rows.index_select(0, selected)
    delta = activation_delta(hidden.index_select(0, selected), tuple(field.index_select(0, packed) for field in fields))
    return hidden.index_add(0, selected, delta)

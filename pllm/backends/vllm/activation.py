"""Signed phase windows borrowing admission or exact invocation publication pins."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pllm.backends.vllm.architecture import CausalLM, DecoderLayer
from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    ActivationPatchOperator,
    DecodeRefresh,
    DecodeWindow,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import StateEffectKind, StateResource
from pllm.contracts.state_ref import StateRef
from pllm.execution.activation import low_rank_affine_delta
from pllm.execution.cuda_stream import upload
from pllm.execution.ptts.page_bundle import PagePin
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import DataT, StateTaskContext


@dataclass(frozen=True, slots=True)
class _PatchRows:
    rows: torch.Tensor
    fields: tuple[torch.Tensor, ...]
    indices: tuple[int, ...]
    pin: PagePin
    ref: StateRef
    identity: RequestIdentity
    used: dict[RequestIdentity, set[StateRef]]


class _ActivationLayer:
    def __init__(self, layer: DecoderLayer, binding: ActivationPatchBinding) -> None:
        self.layer, self.binding = layer, binding
        self.original = layer.forward
        self.prepared: tuple[_PatchRows, ...] = ()
        self.apply_count = 0

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, residual = self.original(positions, hidden_states, residual)
        if not self.prepared:
            return hidden, residual
        # Decoder layers return a fresh down-projection output, separate from its residual.
        for item in self.prepared:
            if self.binding.operator is ActivationPatchOperator.ADD:
                (delta,) = item.fields
                delta = delta.to(hidden.dtype).expand(item.rows.numel(), -1)
            else:
                weight, bias, rotation = item.fields
                delta = low_rank_affine_delta(hidden.index_select(0, item.rows), weight, bias, rotation)
            hidden.index_add_(0, item.rows, delta)
            item.used.setdefault(item.identity, set()).add(item.ref)
            self.apply_count += item.rows.numel()
        return hidden, residual


class VllmActivationPatch:
    def __init__(
        self,
        model: CausalLM,
        bindings: tuple[ActivationPatchBinding, ...],
        resource: str,
        *,
        install: bool = True,
    ) -> None:
        self.resource = StateResource(resource)
        self.installed = install
        self.row_count = 0
        self.layers = tuple(_ActivationLayer(model.model.layers[binding.layer], binding) for binding in bindings)
        if install:
            for handle in self.layers:
                handle.layer.forward = handle.forward

    @property
    def apply_count(self) -> int:
        return sum(handle.apply_count for handle in self.layers)

    def prepare(
        self, rows: PackedPhaseRowBatch, context: StateTaskContext[DataT], directory: RequestStateDirectory
    ) -> None:
        held = context.held_input
        assert held is None or (
            held.step_id == rows.step_id
            and {identity for identity, _, _ in held.successful} == {row.identity for row in rows.rows}
        )
        context.input_step_id = rows.step_id
        versions: dict[tuple[RequestIdentity, bool], tuple[PagePin, StateRef]] = {}
        self.row_count = rows.total_rows
        for handle in self.layers:
            prepared: list[_PatchRows] = []
            for phase, phase_rows in ((HiddenPhase.PREFILL, rows.prefill_rows), (HiddenPhase.DECODE, rows.decode_rows)):
                if phase is HiddenPhase.DECODE and handle.binding.decode is DecodeWindow.NONE:
                    continue
                refresh = phase is HiddenPhase.DECODE and handle.binding.decode_refresh is DecodeRefresh.STEP
                for identity in dict.fromkeys(row.identity for row in phase_rows):
                    source = tuple(row for row in phase_rows if row.identity == identity)
                    selected = tuple(
                        row.packed_row_index
                        for index, row in enumerate(source)
                        if phase is HiddenPhase.DECODE
                        or index < handle.binding.first_tokens
                        or index >= len(source) - handle.binding.last_tokens
                    )
                    if not selected:
                        continue
                    key = (identity, refresh)
                    if key not in versions:
                        pin, ref = self._version(identity, refresh, context, directory)
                        if not self.installed or context.arena.plan.semantic_graph.input_groups:
                            # Admission pins belong to the request directory. Borrow
                            # the same exact version for this invocation's source view.
                            pin = context.pin_for(StateRefs.of((ref,)))
                        versions[key] = pin, ref
                    pin, ref = versions[key]
                    assert held is None or any(
                        identity == held_identity and ref in refs for held_identity, refs, _ in held.successful
                    )
                    prepared.append(
                        _PatchRows(
                            upload(selected, context.arena.device),
                            tuple(
                                view.read(context.arena.borrow_field(pin, ref, view.field))
                                for view in handle.binding.fields
                            ),
                            selected,
                            pin,
                            ref,
                            identity,
                            context.injected_refs,
                        )
                    )
            handle.prepared = tuple(prepared)

    def prepare_token_rows(self) -> tuple[tuple[tuple[torch.Tensor, ...], ...], ...]:
        """Resolve row/layer fields and account their exact pins once per invocation."""
        rows: list[tuple[tuple[torch.Tensor, ...], ...]] = []
        for row in range(self.row_count):
            selected: list[tuple[torch.Tensor, ...]] = []
            for handle in self.layers:
                item = next((item for item in handle.prepared if row in item.indices), None)
                if item is not None:
                    item.used.setdefault(item.identity, set()).add(item.ref)
                selected.append(() if item is None else item.fields)
            rows.append(tuple(selected))
        return tuple(rows)

    def _version(
        self,
        identity: RequestIdentity,
        refresh: bool,
        context: StateTaskContext[DataT],
        directory: RequestStateDirectory,
    ) -> tuple[PagePin, StateRef]:
        if context.selection.for_resource(self.resource).keys == (None,) and not refresh:
            pin = directory.inference_pin(identity)
            (ref,) = (ref for ref in pin.refs if context.arena.root(ref).resource == self.resource)
            return pin, ref
        if context.selection.for_resource(self.resource).keys == (None,):
            (original,) = context.selection.for_resource(self.resource).refs
        else:
            original = context.selection.ref_for(identity, self.resource)
        ref = context.current_owned_ref(original)
        if (
            refresh
            and ref == original
            and context.held_input is None
            and any(
                task.timing.after == "request.prepare"
                and any(
                    effect.resource == self.resource and effect.kind is not StateEffectKind.READ
                    for effect in task.effects
                )
                for task in context.arena.plan.semantic_graph.tasks
            )
        ):
            # No implicit HOLD: its signed no-input branch is a separate lifecycle.
            raise RuntimeError("decode STEP requires a new publication; missing-input handling is not enabled")
        if ref == original:
            return context.pin, ref
        updated = tuple(
            context.current_owned_ref(candidate)
            for candidate in context.selection.for_resource(self.resource).refs
            if context.current_owned_ref(candidate) != candidate
        )
        return context.borrow_published(StateRefs.of(updated)), ref

    def clear(self) -> None:
        self.row_count = 0
        for handle in self.layers:
            handle.prepared = ()

    def close(self) -> None:
        self.clear()
        if self.installed:
            for handle in self.layers:
                handle.layer.forward = handle.original

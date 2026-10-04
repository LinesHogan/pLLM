"""Signed phase windows borrowing admission or exact invocation publication pins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer

from pllm.execution.activation import low_rank_affine_delta
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.page_bundle import PagePin
from pllm.execution.state_execution import DataT, StateTaskContext
from pllm.contracts.backend_attachment import ActivationPatchBinding, ActivationPatchOperator, DecodeRefresh, DecodeWindow
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import StateResource, StateScope
from pllm.contracts.state_ref import StateRef


_DecoderForward = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class _PatchRows:
    rows: torch.Tensor
    fields: tuple[torch.Tensor, ...]


class _ActivationLayer:
    def __init__(self, layer: Qwen2DecoderLayer, binding: ActivationPatchBinding) -> None:
        self.layer, self.binding = layer, binding
        self.original: _DecoderForward = layer.forward
        self.prepared: tuple[_PatchRows, ...] = ()
        self.apply_count = 0

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, residual = self.original(positions, hidden_states, residual)
        if not self.prepared:
            return hidden, residual
        # Qwen2 returns a fresh down-projection output, separate from its residual.
        for item in self.prepared:
            if self.binding.operator is ActivationPatchOperator.ADD:
                delta, = item.fields
                delta = delta.to(hidden.dtype).expand(item.rows.numel(), -1)
            else:
                weight, bias, rotation = item.fields
                delta = low_rank_affine_delta(hidden.index_select(0, item.rows), weight, bias, rotation)
            hidden.index_add_(0, item.rows, delta)
            self.apply_count += item.rows.numel()
        return hidden, residual


class VllmActivationPatch:
    def __init__(self, model: Qwen2ForCausalLM, bindings: tuple[ActivationPatchBinding, ...], resource: str) -> None:
        self.resource = StateResource(resource)
        self.layers = tuple(_ActivationLayer(model.model.layers[binding.layer], binding) for binding in bindings)
        for handle in self.layers:
            handle.layer.forward = handle.forward

    @property
    def apply_count(self) -> int:
        return sum(handle.apply_count for handle in self.layers)

    def prepare(self, rows: PackedPhaseRowBatch, context: StateTaskContext[DataT],
                directory: RequestStateDirectory) -> None:
        versions: dict[tuple[RequestIdentity, bool], tuple[PagePin, StateRef]] = {}
        for handle in self.layers:
            prepared: list[_PatchRows] = []
            for phase, phase_rows in ((HiddenPhase.PREFILL, rows.prefill_rows), (HiddenPhase.DECODE, rows.decode_rows)):
                if phase is HiddenPhase.DECODE and handle.binding.decode is DecodeWindow.NONE:
                    continue
                refresh = phase is HiddenPhase.DECODE and handle.binding.decode_refresh is DecodeRefresh.STEP
                for identity in dict.fromkeys(row.identity for row in phase_rows):
                    source = tuple(row for row in phase_rows if row.identity == identity)
                    selected = tuple(row.packed_row_index for index, row in enumerate(source)
                        if phase is HiddenPhase.DECODE or index < handle.binding.first_tokens
                        or index >= len(source) - handle.binding.last_tokens)
                    if not selected:
                        continue
                    key = (identity, refresh)
                    if key not in versions:
                        versions[key] = self._version(identity, refresh, context, directory)
                    pin, ref = versions[key]
                    if ref.version_id == 0:
                        if phase is HiddenPhase.DECODE:
                            raise RuntimeError("decode activation requires its first published injection")
                        continue
                    prepared.append(_PatchRows(torch.tensor(selected, device=context.arena.device, dtype=torch.long),
                        tuple(context.arena.borrow_field(pin, ref, field) for field in handle.binding.fields)))
            handle.prepared = tuple(prepared)

    def _version(self, identity: RequestIdentity, refresh: bool, context: StateTaskContext[DataT],
                 directory: RequestStateDirectory) -> tuple[PagePin, StateRef]:
        if directory.scope is StateScope.INSTANCE and not refresh:
            pin = directory.inference_pin(identity, context.pin.stream)
            ref, = (ref for ref in pin.refs if context.arena.root(ref).resource == self.resource)
            return pin, ref
        if directory.scope is StateScope.INSTANCE:
            original, = context.selection.for_resource(self.resource).refs
        else:
            original = context.selection.ref_for(identity, self.resource)
        ref = context.current_owned_ref(original)
        if refresh and ref == original:
            # No implicit HOLD: its signed no-input branch is a separate lifecycle.
            raise RuntimeError("decode STEP requires a new publication; missing-input handling is not enabled")
        if ref == original:
            return context.pin, ref
        updated = tuple(context.current_owned_ref(candidate) for candidate in context.selection.for_resource(self.resource).refs
            if context.current_owned_ref(candidate) != candidate)
        return context.borrow_published(updated), ref

    def clear(self) -> None:
        for handle in self.layers:
            handle.prepared = ()

    def close(self) -> None:
        self.clear()
        for handle in self.layers:
            handle.layer.forward = handle.original

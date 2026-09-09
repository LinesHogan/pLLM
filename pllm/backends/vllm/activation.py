"""Signed prefill activation patches borrowing original instance admission pins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer

from pllm.execution.activation import low_rank_affine_delta
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.state_execution import DataT, StateTaskContext
from pllm.contracts.backend_attachment import ActivationPatchBinding, ActivationPatchOperator
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.contracts.semantic_graph import StateScope


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
    def __init__(self, model: Qwen2ForCausalLM, bindings: tuple[ActivationPatchBinding, ...]) -> None:
        self.layers = tuple(_ActivationLayer(model.model.layers[binding.layer], binding) for binding in bindings)
        for handle in self.layers:
            handle.layer.forward = handle.forward

    @property
    def apply_count(self) -> int:
        return sum(handle.apply_count for handle in self.layers)

    def prepare(self, rows: PackedPhaseRowBatch, context: StateTaskContext[DataT],
                directory: RequestStateDirectory) -> None:
        published = (context.borrow_published(context.publication.after_refs)
            if directory.scope is StateScope.REQUEST and context.publication is not None else None)
        for handle in self.layers:
            prepared: list[_PatchRows] = []
            for identity in dict.fromkeys(row.identity for row in rows.prefill_rows):
                source = tuple(row for row in rows.prefill_rows if row.identity == identity)
                length = len(source)
                selected = tuple(row.packed_row_index for index, row in enumerate(source)
                    if index < handle.binding.first_tokens or index >= length - handle.binding.last_tokens)
                if directory.scope is StateScope.INSTANCE:
                    pin = directory.inference_pin(identity, context.pin.stream)
                    (ref,) = pin.refs
                else:
                    ref, pin = context.selection.ref_for(identity), context.pin
                    if published is not None:
                        replacement = next((candidate for candidate in published.refs if candidate.lineage_id == ref.lineage_id), None)
                        if replacement is not None:
                            ref, pin = replacement, published
                if not selected or ref.version_id == 0:
                    continue
                prepared.append(_PatchRows(torch.tensor(selected, device=context.arena.device, dtype=torch.long),
                    tuple(context.arena.borrow_field(pin, ref, field) for field in handle.binding.fields)))
            handle.prepared = tuple(prepared)

    def clear(self) -> None:
        for handle in self.layers:
            handle.prepared = ()

    def close(self) -> None:
        self.clear()
        for handle in self.layers:
            handle.layer.forward = handle.original

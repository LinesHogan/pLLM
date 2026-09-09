"""Eager Qwen2/FA2 current-slot replay; plans and parameter pins stay upstream."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, get_forward_context, override_forward_context
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata

from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.authoring import DataT, ResultT, TensorStepInput
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.state_execution import StateExecution, StateTaskContext
from pllm.execution.replay import TokenwiseReplayPrimitive
from pllm.backends.vllm.parameter_overlay import VllmParameterOverlay
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.semantic_graph import AnchorId, ConditionalLayerReplay, SemanticTask, TokenwiseLayerReplay
from pllm.contracts.state_ref import StateRef

_DecoderForward = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]


def _select_metadata(context: ForwardContext, layer: Qwen2DecoderLayer, row: PackedPhaseRow,
                     length: int, sequence: torch.Tensor) -> tuple[FlashAttentionMetadata, torch.Tensor, int, int]:
    if (context.dp_metadata is not None or context.ubatch_slices is not None
        or context.cudagraph_runtime_mode != CUDAGraphMode.NONE
        or not isinstance(context.attn_metadata, dict) or not isinstance(context.slot_mapping, dict)):
        raise ValueError("replay requires eager single-batch attention metadata")
    attention = layer.self_attn.attn
    metadata = context.attn_metadata[attention.layer_name]
    if (not isinstance(metadata, FlashAttentionMetadata) or metadata.use_cascade
        or metadata.causal is not True or metadata.max_dcp_context_kv_len not in (None, 0)
        or metadata.sliding_window not in (None, (-1, -1))
        or metadata.mm_prefix_range_tensor is not None or metadata.rswa_prefix_lens is not None):
        raise ValueError("replay requires ordinary causal FlashAttention metadata")
    cache = attention.kv_cache
    if cache.ndim != 4 or cache.shape[1] != attention.num_kv_heads or cache.shape[3] != 2 * attention.head_size:
        raise ValueError("replay requires FA2 packed KV layout")
    slot = context.slot_mapping[attention.layer_name][row.packed_row_index:row.packed_row_index + 1]
    block, offset = divmod(int(slot.item()), int(cache.shape[2]))
    if block < 0 or block >= cache.shape[0]:
        raise ValueError("replay requires the token's owned KV slot")
    selected = replace(metadata, num_actual_tokens=1, max_query_len=1,
        query_start_loc=sequence[:2].to(metadata.query_start_loc.dtype),
        max_seq_len=length, seq_lens=sequence[2:].to(metadata.seq_lens.dtype),
        block_table=metadata.block_table[row.request_ordinal:row.request_ordinal + 1], slot_mapping=slot,
        num_decode_reqs=1, num_prefill_reqs=0, num_decode_tokens=1, num_prefill_tokens=0,
        scheduler_metadata=None, prefix_scheduler_metadata=None)
    return selected, cache, block, offset


@dataclass(frozen=True, slots=True)
class _ReplayInput:
    row: PackedPhaseRow
    positions: torch.Tensor
    carry: torch.Tensor
    metadata: dict[str, FlashAttentionMetadata]
    slots: tuple[tuple[torch.Tensor, int, int], ...]
    original_kv: tuple[torch.Tensor, ...]


class VllmLayerReplay:
    """Re-runs the signed ascending layer range per selected token; the hook sits
    on the range's first layer only, deeper layers are replayed by direct call."""

    def __init__(self, model: Qwen2ForCausalLM, effect: ConditionalLayerReplay,
                 overlay: VllmParameterOverlay) -> None:
        layers = model.model.layers
        if (not effect.layers or tuple(sorted(set(effect.layers))) != effect.layers
                or effect.layers[-1] >= len(layers)):
            raise ValueError("conditional replay requires a sorted unique in-range layer set")
        self.effect, self.overlay = effect, overlay
        self.layers = tuple(layers[index] for index in effect.layers)
        for layer in self.layers:
            attention = layer.self_attn.attn
            if not isinstance(attention.impl, FlashAttentionImpl) or attention.impl.vllm_flash_attn_version != 2:
                raise ValueError("conditional replay requires FlashAttention2")
            if (attention.kv_cache_dtype != "auto" or attention.kv_sharing_target_layer_name is not None
                or attention.num_kv_heads * attention.head_size > model.config.hidden_size):
                raise ValueError("conditional replay requires unquantized nonshared GQA/MHA KV")
        self.originals: tuple[_DecoderForward, ...] = tuple(layer.forward for layer in self.layers)
        self._rows: tuple[PackedPhaseRow, ...] = ()
        self._inputs: tuple[_ReplayInput, ...] = ()
        self._forward_context: ForwardContext | None = None
        self._materialized = False
        self.layers[0].forward = self.forward

    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]:
        return self._rows

    def prepare(self, rows: PackedPhaseRowBatch) -> None:
        self.clear()
        self._rows = tuple(row for index, row in enumerate(rows.rows)
            if index + 1 == len(rows.rows) or rows.rows[index + 1].identity != row.identity)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._rows:
            return self.originals[0](positions, hidden_states, residual)
        context = get_forward_context()
        prepared: list[_ReplayInput] = []
        # Decoder residual addition is in-place; own only the selected token inputs.
        for row in self._rows:
            index = row.packed_row_index
            carry = hidden_states[index:index + 1].detach()
            carry = carry.clone() if residual is None else carry + residual[index:index + 1]
            position = positions[index:index + 1]
            length = int(position.item()) + 1
            sequence = torch.tensor([0, 1, length], dtype=torch.int32, device=carry.device)
            metadata: dict[str, FlashAttentionMetadata] = {}
            slots: list[tuple[torch.Tensor, int, int]] = []
            originals: list[torch.Tensor] = []
            for layer in self.layers:
                selected, cache, block, offset = _select_metadata(context, layer, row, length, sequence)
                metadata[layer.self_attn.attn.layer_name] = selected
                slots.append((cache, block, offset))
                originals.append(cache[block, :, offset, :])
            prepared.append(_ReplayInput(row, position, carry, metadata, tuple(slots), tuple(originals)))
        output = self.originals[0](positions, hidden_states, residual)
        # Views follow the in-place KV writes; deeper layers write their slot only
        # after this hook returns. Clone at the first replay step — after the normal
        # forward finished, before any replay write — so restore returns the
        # successful normal token, not the pre-forward or replayed KV.
        self._inputs = tuple(prepared)
        self._forward_context = context
        return output

    def seed(self, row: PackedPhaseRow) -> torch.Tensor:
        return next(item.carry for item in self._inputs if item.row is row)

    def step(self, row: PackedPhaseRow, carry: torch.Tensor, arena: PageBundleArena,
             pin: PagePin, ref: StateRef) -> torch.Tensor:
        if self._forward_context is None:
            raise RuntimeError("replay requires its original normal forward context")
        if not self._materialized:
            self._inputs = tuple(replace(item,
                original_kv=tuple(original.clone() for original in item.original_kv)) for item in self._inputs)
            self._materialized = True
        item = next(item for item in self._inputs if item.row is row)
        hidden = carry.clone()
        residual: torch.Tensor | None = None
        for index, layer in enumerate(self.layers):
            name = layer.self_attn.attn.layer_name
            metadata = item.metadata[name]
            context = replace(self._forward_context, attn_metadata={name: metadata},
                slot_mapping={name: metadata.slot_mapping}, batch_descriptor=None, is_padding=None)
            self.overlay.prepare_replay(row, self.effect.layers[index], arena, pin, ref)
            try:
                with override_forward_context(context):
                    hidden, residual = self.originals[index](item.positions, hidden, residual)
            finally:
                self.overlay.clear()
        if residual is None:
            raise RuntimeError("replayed layers did not return their residual chain")
        return hidden + residual

    def restore(self) -> None:
        for item in self._inputs:
            for (cache, block, offset), original in zip(item.slots, item.original_kv, strict=True):
                cache[block, :, offset, :].copy_(original)

    def clear(self) -> None:
        self._rows, self._inputs, self._forward_context = (), (), None
        self._materialized = False

    def close(self) -> None:
        self.clear()
        self.layers[0].forward = self.originals[0]


class VllmTokenwiseReplay(TokenwiseReplayPrimitive, Generic[DataT, ResultT]):
    """Model kernels bound to the original signed task and invocation."""

    def __init__(self, model: Qwen2ForCausalLM, task: SemanticTask,
                 execution: StateExecution[TensorStepInput, DataT, ResultT], anchor: AnchorId) -> None:
        if not isinstance(task.replay, TokenwiseLayerReplay) or task.replay.layer_count != len(model.model.layers):
            raise ValueError("tokenwise replay differs from the specialized model")
        for layer in model.model.layers:
            attention = layer.self_attn.attn
            if (not isinstance(attention.impl, FlashAttentionImpl) or attention.impl.vllm_flash_attn_version != 2
                or attention.kv_cache_dtype != "auto" or attention.kv_sharing_target_layer_name is not None
                or attention.num_kv_heads * attention.head_size > model.config.hidden_size):
                raise ValueError("tokenwise replay requires FlashAttention2 and unquantized nonshared GQA/MHA KV")
        self.model, self.effect = model.model, task.replay
        self.execution, self.anchor = execution, anchor
        self.original = self.model.forward
        self._rows: tuple[PackedPhaseRow, ...] = ()
        self._invocation: TaskGraphInvocation[StateTaskContext[DataT]] | None = None
        self._hidden: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._selected_context: ForwardContext | None = None
        self._backups: list[tuple[torch.Tensor, int, int, torch.Tensor]] = []
        self.model.forward = self.forward

    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]:
        return self._rows

    def prepare(self, rows: PackedPhaseRowBatch, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> None:
        self.clear()
        self._rows, self._invocation = rows.rows, invocation

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor | IntermediateTensors:
        if self._invocation is None:
            return self.original(input_ids, positions, intermediate_tensors, inputs_embeds)
        if intermediate_tensors is not None or positions.ndim != 1 or positions.numel() != len(self._rows):
            raise ValueError("tokenwise replay requires one original position per packed row")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("tokenwise replay requires model input IDs or embeddings")
            inputs_embeds = self.model.embed_input_ids(input_ids)
        self._hidden, self._positions = inputs_embeds, positions
        self.execution.execute_anchor(self._invocation, self.anchor)
        if self._output is None:
            raise RuntimeError("signed tokenwise task did not return normal hidden states")
        return self._output

    def normal(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, positions = self._hidden, self._positions
        self._backups = []
        if hidden is None or positions is None or self._output is not None:
            raise RuntimeError("tokenwise replay requires its original model inputs")
        context = get_forward_context()
        index = row.packed_row_index
        position = positions[index:index + 1]
        length = int(position.item()) + 1
        if length != row.token_offset + 1:
            raise ValueError("tokenwise replay position differs from original packed identity")
        # Every layer consumes the same immutable singleton sequence, not a per-layer copy.
        sequence = torch.tensor([0, 1, length], dtype=torch.int32, device=hidden.device)
        metadata_by_layer: dict[str, FlashAttentionMetadata] = {}
        slots_by_layer: dict[str, torch.Tensor] = {}
        for layer in self.model.layers:
            attention = layer.self_attn.attn
            selected, cache, block, offset = _select_metadata(context, layer, row, length, sequence)
            metadata_by_layer[attention.layer_name] = selected
            slots_by_layer[attention.layer_name] = selected.slot_mapping
            self._backups.append((cache, block, offset, cache[block, :, offset, :].clone()))
        self._selected_context = replace(context, attn_metadata=metadata_by_layer,
            slot_mapping=slots_by_layer, batch_descriptor=None, is_padding=None)
        with override_forward_context(self._selected_context):
            current, residual = hidden[index:index + 1], None
            source: torch.Tensor | None = None
            destination: torch.Tensor | None = None
            for layer_index, layer in enumerate(self.model.layers):
                current, residual = layer.forward(position, current, residual)
                if layer_index == self.effect.destination_layer:
                    destination = current + residual
                if layer_index == self.effect.source_layer:
                    source = current + residual
            if residual is None or source is None or destination is None:
                raise RuntimeError("tokenwise normal pass did not reach signed layers")
            normal, _ = self.model.norm.forward(current, residual)
            return normal, source, destination

    def replay(self, row: PackedPhaseRow, hidden: torch.Tensor) -> None:
        if self._selected_context is None or self._positions is None:
            raise RuntimeError("tokenwise replay requires the preceding normal token")
        with override_forward_context(self._selected_context):
            residual = None
            position = self._positions[row.packed_row_index:row.packed_row_index + 1]
            for layer in self.model.layers[self.effect.destination_layer + 1:]:
                hidden, residual = layer.forward(position, hidden, residual)

    def restore(self) -> None:
        for cache, block, offset, saved in self._backups:
            cache[block, :, offset, :].copy_(saved)

    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None:
        self._output = torch.cat(outputs, dim=0)
        self._backups = []

    def clear(self) -> None:
        self._rows, self._invocation = (), None
        self._hidden, self._positions, self._output = None, None, None
        self._selected_context, self._backups = None, []

    def close(self) -> None:
        self.clear()
        self.model.forward = self.original

"""Device-selected compact decoder slices and CALM KV propagation for custom-forward waves."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from pllm.backends.vllm.custom_forward_attention import BatchAttentionBuffers, attention_mode
from pllm.execution.cuda_graph import device_branch
from pllm.execution.token_program import KV, Segment
from pllm.execution.token_program_executor import execute_segment

if TYPE_CHECKING:
    from pllm.authoring import DataT, ResultT
    from pllm.backends.vllm.custom_forward import ProgramWave


def _store(
    attention: Attention, slots: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    assert isinstance(attention.impl, FlashAttentionImpl)
    key = k.view(-1, attention.num_kv_heads, attention.head_size)
    value = v.view(-1, attention.num_kv_heads, attention.head_size)
    attention.impl.do_kv_cache_update(attention, key, value, attention.kv_cache, slots)
    return q


def _omit_projection(hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
    return hidden, None


class AdaptiveSlice:
    """Stable bucket metadata and device predicates; no active counts ever cross to the host."""

    def __init__(self, wave: ProgramWave[DataT, ResultT]) -> None:
        self.wave = wave
        width = len(wave.seed)
        widths = tuple(sorted({width, *(2**power for power in range(width.bit_length()) if 2**power < width)}))
        self.widths = tuple(
            size for index, size in enumerate(widths) if (widths[index - 1] if index else 0) < width - 1
        )
        owner = wave.custom_forward
        self.buckets = {
            size: BatchAttentionBuffers(wave.attention.context, owner.attentions, owner.domain.positions, size)
            for size in self.widths
        }
        self.positions = wave.positions

    @property
    def resident_bytes(self) -> int:
        """Bucket metadata outside the graph allocator, charged against the task's capture budget."""
        return sum(
            tensor.numel() * tensor.element_size()
            for bucket in self.buckets.values()
            for index, attention in enumerate(self.wave.custom_forward.attentions)
            if bucket.group_ids[index] == index
            for metadata in (bucket.metadata[attention.layer_name],)
            for tensor in (metadata.query_start_loc, metadata.seq_lens, metadata.slot_mapping, metadata.block_table)
        )

    def layer(
        self, ordinal: int, hidden: torch.Tensor, residual: torch.Tensor | None, kv: KV
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the native layer with the bucket's positions and attention context."""
        owner = self.wave.custom_forward
        with attention_mode(owner.attentions[ordinal], self.positions, kv):
            hidden, residual = owner.model.layers[ordinal](self.positions, hidden, residual)
        assert residual is not None
        return hidden, residual

    def propagate(self, step: Segment, hidden: torch.Tensor, active: torch.Tensor) -> None:
        """Use native QKV/QK-norm/RoPE and value rewrites, omitting attention and the output projection."""
        owner = self.wave.custom_forward
        mappings: dict[int, torch.Tensor] = {}
        for ordinal in step.propagate:
            layer = owner.model.layers[ordinal]
            attention = layer.self_attn
            group = self.wave.attention.group_ids[ordinal]
            if group not in mappings:
                mappings[group] = self.wave.attention.metadata[attention.attn.layer_name].slot_mapping.masked_fill(
                    active.flatten(), -1
                )
            slots = mappings[group]
            endpoint, projection = attention.attn.forward, attention.o_proj.forward
            try:
                attention.attn.forward = partial(_store, attention.attn, slots)
                attention.o_proj.forward = _omit_projection
                # Propagation normalizes the frozen stream; residual read/write hooks belong to computed rows.
                normal = layer.input_layernorm.forward_cuda(hidden)
                attention(self.wave.positions, normal)
            finally:
                attention.attn.forward, attention.o_proj.forward = endpoint, projection

    def __call__(self, step: Segment, hidden: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = tuple(hidden.clone() for _ in step.taps)
        count = active.sum()
        width = len(hidden)
        owner = self.wave.custom_forward
        if owner.sites is not None:
            owner.sites.frozen(step.start, step.stop, hidden)
        with device_branch(count == width):
            with owner.sites_rows(self.wave.indices[0], self.wave.packed) as stream:
                full = execute_segment(step, hidden, self.wave.layer, stream)
                for destination, value in zip(output, full, strict=True):
                    destination.copy_(value)
        with device_branch(count < width):
            with owner.sites_rows(self.wave.indices[0], self.wave.packed):
                self.propagate(step, hidden, active)
            if self.widths:
                with device_branch(count > 0):
                    order = torch.argsort(~active.flatten(), stable=True)
                    lower = 0
                    for size in self.widths:
                        with device_branch((count > lower) & (count <= size)):
                            self.compact(step, hidden, active, order[:size], output)
                        lower = size
        return output

    def compact(
        self,
        step: Segment,
        hidden: torch.Tensor,
        active: torch.Tensor,
        rows: torch.Tensor,
        output: tuple[torch.Tensor, ...],
    ) -> None:
        """Gather bucket rows and mask padded KV writes, then scatter the computed streams."""
        wave, owner = self.wave, self.wave.custom_forward
        bucket = self.buckets[len(rows)]
        bindings = bucket.bind(wave.attention.context, owner.attentions, wave.positions + 1)
        bucket.refresh(bindings, rows, rows)
        selected = active.index_select(0, rows)
        for index, attention in enumerate(owner.attentions):
            if bucket.group_ids[index] == index:
                metadata = bucket.metadata[attention.layer_name]
                metadata.slot_mapping.masked_fill_(~selected.flatten(), -1)
                metadata.seq_lens.masked_fill_(~selected.flatten(), 1)
        self.positions = wave.positions.index_select(0, rows)
        seed = hidden.index_select(0, rows)
        # A wave already groups equal patch presence. Keep its host presence rows;
        # per-request patch fields are gathered with the compacted device indices.
        with (
            override_forward_context(bucket.context),
            owner.sites_rows(wave.indices[0].index_select(0, rows), wave.packed, selected) as stream,
        ):
            computed = execute_segment(step, seed, self.layer, stream)
            for destination, value in zip(output, computed, strict=True):
                destination.index_copy_(0, rows, torch.where(selected, value, seed))

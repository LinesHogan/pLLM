"""Bind vLLM attention resources without inspecting a model implementation.

The frontend identifies the original Attention endpoints. This adapter retains
vLLM's cache layout and metadata checks, then supplies logical page accesses to
the common planner. It never reconstructs attention computation or owns KV.
"""

from dataclasses import replace

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.execution.paged_access import PagedAccessPlan, PagedLayer, bind_paged_access


def validate_attention_resources(attentions: tuple[Attention, ...], *, hidden_size: int) -> None:
    """Check the backend ABI before attaching a replay resource provider."""
    for attention in attentions:
        if (
            not isinstance(attention.impl, FlashAttentionImpl)
            or attention.impl.vllm_flash_attn_version != 2
            or attention.kv_cache_dtype != "auto"
            or attention.kv_sharing_target_layer_name is not None
            or attention.num_kv_heads * attention.head_size > hidden_size
        ):
            raise ValueError("replay requires FlashAttention2 and unquantized nonshared GQA/MHA KV")


def attention_metadata(
    context: ForwardContext, attention: Attention
) -> tuple[FlashAttentionMetadata, torch.Tensor, torch.Tensor]:
    if (
        context.dp_metadata is not None
        or context.ubatch_slices is not None
        or context.cudagraph_runtime_mode != CUDAGraphMode.NONE
        or not isinstance(context.attn_metadata, dict)
        or not isinstance(context.slot_mapping, dict)
    ):
        raise ValueError("replay requires eager single-batch attention metadata")
    metadata = context.attn_metadata[attention.layer_name]
    if (
        not isinstance(metadata, FlashAttentionMetadata)
        or metadata.use_cascade
        or metadata.causal is not True
        or metadata.max_dcp_context_kv_len not in (None, 0)
        or metadata.sliding_window not in (None, (-1, -1))
        or metadata.mm_prefix_range_tensor is not None
        or metadata.rswa_prefix_lens is not None
    ):
        raise ValueError("replay requires ordinary causal FlashAttention metadata")
    cache = attention.kv_cache
    if cache.ndim != 4 or cache.shape[1] != attention.num_kv_heads or cache.shape[3] != 2 * attention.head_size:
        raise ValueError("replay requires FA2 packed KV layout")
    return metadata, cache, context.slot_mapping[attention.layer_name]


def attention_paged_access(
    context: ForwardContext, attentions: tuple[Attention, ...], rows: tuple[PackedPhaseRow, ...]
) -> PagedAccessPlan:
    return bind_paged_access(
        rows,
        tuple(PagedLayer(item.layer_name, *attention_metadata(context, item)[1:]) for item in attentions),
    )


class BatchAttentionBuffers:
    """Stable width-B FA2 metadata; invocation tables and slots are copied before replay."""

    def __init__(self, context: ForwardContext, attentions: tuple[Attention, ...], capacity: int, width: int) -> None:
        self.metadata: dict[str, FlashAttentionMetadata] = {}
        self.group_ids = self.mapping_groups(context, attentions)
        for index, attention in enumerate(attentions):
            group = self.group_ids[index]
            if group != index:
                self.metadata[attention.layer_name] = self.metadata[attentions[group].layer_name]
                continue
            metadata, cache, slots = attention_metadata(context, attention)
            selected = replace(
                metadata,
                num_actual_tokens=width,
                max_query_len=1,
                query_start_loc=torch.arange(width + 1, dtype=metadata.query_start_loc.dtype, device=cache.device),
                # The launch bound is fixed at capture; the true length is seq_lens. The split count stays
                # vLLM's own choice (its heuristic, or 1 under VLLM_BATCH_INVARIANT).
                max_seq_len=capacity,
                seq_lens=torch.empty(width, dtype=metadata.seq_lens.dtype, device=cache.device),
                block_table=torch.zeros(
                    (width, (capacity + cache.shape[2] - 1) // cache.shape[2]),
                    dtype=metadata.block_table.dtype,
                    device=cache.device,
                ),
                slot_mapping=torch.empty(width, dtype=slots.dtype, device=cache.device),
                num_decode_reqs=width,
                num_prefill_reqs=0,
                num_decode_tokens=width,
                num_prefill_tokens=0,
                scheduler_metadata=None,
                prefix_scheduler_metadata=None,
            )
            self.metadata[attention.layer_name] = selected
        self.context = replace(
            context,
            attn_metadata=self.metadata,
            slot_mapping={name: item.slot_mapping for name, item in self.metadata.items()},
            batch_descriptor=None,
            is_padding=None,
        )

    @staticmethod
    def mapping_groups(context: ForwardContext, attentions: tuple[Attention, ...]) -> tuple[int, ...]:
        """Share staging only while layers borrow identical device mappings and block shapes."""
        keys = tuple(
            (metadata.block_table.data_ptr(), slots.data_ptr(), cache.shape[2])
            for metadata, cache, slots in (attention_metadata(context, item) for item in attentions)
        )
        return tuple(keys.index(key) for key in keys)

    def bind(
        self, context: ForwardContext, attentions: tuple[Attention, ...], lengths: torch.Tensor
    ) -> tuple[tuple[FlashAttentionMetadata, torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Resolve the layer-specific invocation views once before row submission."""
        return tuple(
            (
                self.metadata[item.layer_name],
                attention_metadata(context, item)[0].block_table,
                attention_metadata(context, item)[2],
                lengths.to(self.metadata[item.layer_name].seq_lens.dtype),
            )
            for index, item in enumerate(attentions)
            if self.group_ids[index] == index
        )

    @staticmethod
    def refresh(
        bindings: tuple[tuple[FlashAttentionMetadata, torch.Tensor, torch.Tensor, torch.Tensor], ...],
        rows: torch.Tensor,
        requests: torch.Tensor,
    ) -> None:
        """Queue device copies on the owner's stream without reading device scalars."""
        for metadata, tables, slots, lengths in bindings:
            metadata.slot_mapping.copy_(slots.index_select(0, rows))
            metadata.seq_lens.copy_(lengths.index_select(0, rows))
            width = min(tables.shape[1], metadata.block_table.shape[1])
            metadata.block_table[:, :width].copy_(tables.index_select(0, requests)[:, :width])

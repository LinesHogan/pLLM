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
        if (not isinstance(attention.impl, FlashAttentionImpl) or attention.impl.vllm_flash_attn_version != 2
            or attention.kv_cache_dtype != "auto" or attention.kv_sharing_target_layer_name is not None
            or attention.num_kv_heads * attention.head_size > hidden_size):
            raise ValueError("replay requires FlashAttention2 and unquantized nonshared GQA/MHA KV")


def attention_metadata(context: ForwardContext, attention: Attention) -> tuple[FlashAttentionMetadata, torch.Tensor, torch.Tensor]:
    if (context.dp_metadata is not None or context.ubatch_slices is not None
        or context.cudagraph_runtime_mode != CUDAGraphMode.NONE
        or not isinstance(context.attn_metadata, dict) or not isinstance(context.slot_mapping, dict)):
        raise ValueError("replay requires eager single-batch attention metadata")
    metadata = context.attn_metadata[attention.layer_name]
    if (not isinstance(metadata, FlashAttentionMetadata) or metadata.use_cascade
        or metadata.causal is not True or metadata.max_dcp_context_kv_len not in (None, 0)
        or metadata.sliding_window not in (None, (-1, -1))
        or metadata.mm_prefix_range_tensor is not None or metadata.rswa_prefix_lens is not None):
        raise ValueError("replay requires ordinary causal FlashAttention metadata")
    cache = attention.kv_cache
    if cache.ndim != 4 or cache.shape[1] != attention.num_kv_heads or cache.shape[3] != 2 * attention.head_size:
        raise ValueError("replay requires FA2 packed KV layout")
    return metadata, cache, context.slot_mapping[attention.layer_name]


def attention_paged_access(context: ForwardContext, attentions: tuple[Attention, ...],
                           rows: tuple[PackedPhaseRow, ...]) -> PagedAccessPlan:
    bindings: list[PagedLayer] = []
    count = max((row.packed_row_index for row in rows), default=-1) + 1
    for attention in attentions:
        metadata, cache, slots = attention_metadata(context, attention)
        declared, actual = metadata.slot_mapping[:count], slots[:count]
        if declared.ndim != 1 or declared.dtype not in (torch.int32, torch.int64):
            raise ValueError("attention requires an integer packed slot vector")
        same_view = (declared.device == actual.device and declared.dtype == actual.dtype
            and declared.shape == actual.shape and declared.stride() == actual.stride()
            and declared.data_ptr() == actual.data_ptr())
        if not same_view and not torch.equal(declared, actual):
            raise ValueError("attention and KV update use different slot mappings")
        bindings.append(PagedLayer(attention.layer_name, cache, metadata.block_table, slots))
    return bind_paged_access(rows, tuple(bindings))


def select_attention_metadata(context: ForwardContext, attention: Attention, row: PackedPhaseRow,
                              length: int, sequence: torch.Tensor, access: PagedAccessPlan
                              ) -> tuple[FlashAttentionMetadata, torch.Tensor, int, int]:
    if length != row.token_offset + 1:
        raise ValueError("attention history differs from the verified packed token position")
    metadata, cache, mapping = attention_metadata(context, attention)
    if (metadata.query_start_loc.dtype not in (torch.int32, torch.int64)
        or metadata.seq_lens.dtype not in (torch.int32, torch.int64)):
        raise ValueError("replay sequence metadata requires int32 or int64 storage")
    slot = mapping[row.packed_row_index:row.packed_row_index + 1]
    block, offset = access.slot(attention.layer_name, row)
    selected = replace(metadata, num_actual_tokens=1, max_query_len=1,
        query_start_loc=sequence[:2].to(metadata.query_start_loc.dtype),
        max_seq_len=length, seq_lens=sequence[2:].to(metadata.seq_lens.dtype),
        block_table=metadata.block_table[row.request_ordinal:row.request_ordinal + 1], slot_mapping=slot,
        num_decode_reqs=1, num_prefill_reqs=0, num_decode_tokens=1, num_prefill_tokens=0,
        scheduler_metadata=None, prefix_scheduler_metadata=None)
    return selected, cache, block, offset

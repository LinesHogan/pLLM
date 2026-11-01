"""Bind vLLM attention resources and read-only peek calls for layer-route replay.

vLLM owns the cache and FA2 attention. Peek merges strictly earlier cached
attention with the fresh diagonal without writing any cache slots.
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from vllm.vllm_flash_attn import flash_attn_varlen_func

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.execution.cuda_stream import upload
from pllm.execution.paged_access import PagedAccessPlan, PagedLayer, bind_paged_access
from pllm.execution.token_program import KV


def validate_attention_resources(attentions: tuple[Attention, ...], *, hidden_size: int) -> None:
    """Check the backend ABI before attaching a replay resource provider."""
    for attention in attentions:
        if (
            not isinstance(attention.impl, FlashAttentionImpl)
            or attention.impl.vllm_flash_attn_version != 2
            or attention.impl.sliding_window != (-1, -1)
            or attention.kv_cache_dtype != "auto"
            or attention.kv_sharing_target_layer_name is not None
            or attention.num_kv_heads * attention.head_size > hidden_size
        ):
            raise ValueError("replay requires FlashAttention2 and unquantized nonshared full-attention GQA/MHA KV")


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
        or metadata.causal is not True
        or metadata.max_dcp_context_kv_len not in (None, 0)
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
            )
            self.metadata[attention.layer_name] = selected
        self.context = _replay_context(context, self.metadata)

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


def _replay_context(context: ForwardContext, metadata: dict[str, FlashAttentionMetadata]) -> ForwardContext:
    return replace(
        context,
        attn_metadata=metadata,
        slot_mapping={name: item.slot_mapping for name, item in metadata.items()},
        batch_descriptor=None,
        is_padding=None,
    )


def span_context(
    context: ForwardContext,
    attentions: tuple[Attention, ...],
    groups: tuple[tuple[PackedPhaseRow, ...], ...],
    packed: torch.Tensor,
    requests: torch.Tensor,
    starts: list[int],
) -> ForwardContext:
    """FA2 metadata for one native varlen pass over a subset of the step's packed rows.

    Each request's contiguous rows form one prefill query that ends at its last row's position.
    """
    query_start_loc = upload(starts, packed.device, torch.int32)
    seq_lens = upload(tuple(group[-1].token_offset + 1 for group in groups), packed.device, torch.int32)
    metadata: dict[str, FlashAttentionMetadata] = {}
    for attention in attentions:
        source, _, slots = attention_metadata(context, attention)
        metadata[attention.layer_name] = replace(
            source,
            num_actual_tokens=starts[-1],
            max_query_len=max(len(group) for group in groups),
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=source.block_table.index_select(0, requests),
            slot_mapping=slots.index_select(0, packed),
            num_decode_reqs=0,
            num_decode_tokens=0,
            num_prefill_reqs=len(groups),
            num_prefill_tokens=starts[-1],
            scheduler_metadata=None,
        )
    return _replay_context(context, metadata)


def merge_diagonal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    off: torch.Tensor,
    lse: torch.Tensor,
    positions: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Merge strictly earlier cached attention with each query's fresh diagonal K/V."""
    grouped = q.float().reshape(q.shape[0], k.shape[1], -1, q.shape[-1])
    diagonal = (grouped * k.float().unsqueeze(2)).sum(-1).reshape(q.shape[:2]) * scale
    earlier = lse.T.masked_fill(positions[:, None] == 0, -float("inf"))
    total = torch.logaddexp(earlier, diagonal)
    fresh = v.float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    return (torch.exp(earlier - total)[..., None] * off.float() + torch.exp(diagonal - total)[..., None] * fresh).to(
        q.dtype
    )


def peek_attention(
    attention: Attention,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Read FA2's cache with visible length seq_len - 1, leaving all cache bytes intact."""
    metadata, cache, _ = attention_metadata(get_forward_context(), attention)
    assert isinstance(attention.impl, FlashAttentionImpl)
    q = query.view(-1, attention.num_heads, attention.head_size)
    k, v = (tensor.view(-1, attention.num_kv_heads, attention.head_size) for tensor in (key, value))
    cached_k, cached_v = cache.transpose(1, 2).chunk(2, dim=-1)
    off, lse = flash_attn_varlen_func(
        q=q,
        k=cached_k,
        v=cached_v,
        cu_seqlens_q=metadata.query_start_loc,
        max_seqlen_q=metadata.max_query_len,
        seqused_k=(metadata.seq_lens - 1).clamp_min(0),
        max_seqlen_k=max(0, metadata.max_seq_len - 1),
        softmax_scale=attention.impl.scale,
        causal=True,
        block_table=metadata.block_table,
        return_softmax_lse=True,
        fa_version=2,
    )
    return merge_diagonal(q, k, v, off, lse, positions, attention.impl.scale).reshape(query.shape)


@contextmanager
def attention_mode(attention: Attention, positions: torch.Tensor, kv: KV) -> Generator[None]:
    """Substitute only the Attention endpoint during a peek, preserving native layer math."""
    if kv == "commit":
        yield
        return
    forward = attention.forward
    try:
        attention.forward = partial(peek_attention, attention, positions)
        yield
    finally:
        attention.forward = forward

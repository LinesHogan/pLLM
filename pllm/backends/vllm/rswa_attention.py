"""Read R-SWA output through vLLM's launcher and recover its otherwise discarded LSE.

Custom attention uses FA2/FlashInfer for full and ordinary window reads. vLLM 0.27.1/0.32.0
TritonAttentionImpl.forward calls unified_attention, which returns only output.
Its 2-D branch discards M/L, including all prefill; its 3-D branch exposes only
split-decode scratch. Neither the backend nor reduce_segments returns LSE.
Until upstream exposes LSE, the extra unsplit launch below recovers M/L only;
vLLM's ordinary launcher remains responsible for the returned attention output.
"""

import torch
import triton
from vllm.v1.attention.ops.triton_unified_attention import kernel_unified_attention, unified_attention

from pllm.execution.attention import PagedKV


def rswa_attention(
    kv: PagedKV, q: torch.Tensor, scale: float, output: torch.Tensor, max_query_len: int, max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use TritonAttentionImpl's ordinary R-SWA launcher with a separate LSE recovery."""
    q = q.contiguous() if q.stride(-1) != 1 else q
    rows, heads, dim = q.shape
    padded = triton.next_power_of_2(dim)
    partial = torch.empty((rows, heads, padded), dtype=torch.float32, device=q.device)
    maximum, denominator = (torch.empty((rows, heads), dtype=torch.float32, device=q.device) for _ in range(2))
    unified_attention(
        q=q,
        k=kv.key_cache,
        v=kv.value_cache,
        out=output,
        cu_seqlens_q=kv.query_start,
        max_seqlen_q=max_query_len,
        seqused_k=kv.seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=kv.block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        rswa_prefix_lens=kv.prompt_lens,
        rswa_window=kv.storage_window,
    )
    _lse_statistics(kv, q, scale, partial, maximum, denominator)
    return output, maximum + denominator.log()


def _lse_statistics(
    kv: PagedKV,
    q: torch.Tensor,
    scale: float,
    partial: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
) -> None:
    rows, heads, dim = q.shape
    keys, values = kv.key_cache, kv.value_cache
    grouped = heads // keys.shape[2]
    block_m = max(16, triton.next_power_of_2(grouped))
    block_q = block_m // grouped
    kernel_unified_attention[(rows // block_q + len(kv.seq_lens), keys.shape[2], 1)](
        output_ptr=partial,
        query_ptr=q,
        key_cache_ptr=keys,
        value_cache_ptr=values,
        sink_ptr=None,
        block_tables_ptr=kv.block_table,
        seq_lens_ptr=kv.seq_lens,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        scale=scale,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        out_scale=1.0,
        softcap=0.0,
        num_query_heads=heads,
        num_queries_per_kv=grouped,
        block_table_stride=kv.block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=0,
        output_stride_1=0,
        qq_bias_stride_0=0,
        BLOCK_SIZE=keys.shape[1],
        TILE_SIZE=32,
        HEAD_SIZE=dim,
        HEAD_SIZE_PADDED=triton.next_power_of_2(dim),
        USE_ALIBI_SLOPES=False,
        USE_ALIBI_SQRT=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=False,
        USE_SINKS=False,
        SLIDING_WINDOW=0,
        USE_CAUSAL=True,
        USE_PER_SEQ_CAUSAL=False,
        per_seq_causal_ptr=None,
        USE_MM_PREFIX=False,
        MAX_MM_RANGES=0,
        mm_prefix_range_ptr=None,
        rswa_prefix_lens_ptr=kv.prompt_lens,
        R_SWA_WINDOW=kv.storage_window,
        USE_R_SWA=True,
        stride_k_cache_0=keys.stride(0),
        stride_k_cache_1=keys.stride(1),
        stride_k_cache_2=keys.stride(2),
        stride_k_cache_3=keys.stride(3),
        stride_v_cache_0=values.stride(0),
        stride_v_cache_1=values.stride(1),
        stride_v_cache_2=values.stride(2),
        stride_v_cache_3=values.stride(3),
        query_start_len_ptr=kv.query_start,
        BLOCK_Q=block_q,
        num_seqs=len(kv.seq_lens),
        BLOCK_M=block_m,
        NUM_SEGMENTS_PER_SEQ=1,
        USE_FP8=False,
        IS_3D=True,
        segm_output_ptr=partial,
        segm_max_ptr=maximum,
        segm_expsum_ptr=denominator,
    )

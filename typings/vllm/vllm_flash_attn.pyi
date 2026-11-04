"""FA2 varlen ABI used by read-only route attention."""

import torch

def flash_attn_varlen_func(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    block_table: torch.Tensor,
    alibi_slopes: torch.Tensor | None = None,
    window_size: list[int] | None = None,
    softcap: float = 0.0,
    scheduler_metadata: torch.Tensor | None = None,
    num_splits: int = 0,
    return_softmax_lse: bool,
    fa_version: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...

"""Read-only paged attention views and authored reads, used by backends and CPU checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class AttentionRead(Protocol):
    """The backend's attention kernel, including its float32 log normalizer."""

    def __call__(
        self, q: torch.Tensor, window: int, topk: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class PagedKV:
    """Borrowed paged K/V, already including the current tokens; never changes cache storage.

    Caches have shape [blocks, block_size, kv_heads, head_dim]. ``topk`` contains
    logical token positions [rows, k] or [rows, query_heads, k], padded with -1.
    Different heads and layers may select different budgets. Selection saves reads,
    not allocated KV memory. ``window`` counts tokens including the current token.
    """

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_start: torch.Tensor
    positions: torch.Tensor
    _read: AttentionRead

    def attend(
        self, q: torch.Tensor, *, window: int = 0, topk: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention [rows, heads, dim] and float32 LSE [rows, heads].

        With no selection this uses the backend's native causal kernel. Selections
        intersect causality and the optional window; an empty read returns zero
        output and negative-infinity LSE. Indices must be distinct within a head.
        """
        return self._read(q, window, topk)


def selected_attention(
    kv: PagedKV, q: torch.Tensor, scale: float, window: int, topk: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only selected positions; shared CPU/CUDA reference for dense read selection."""
    rows = torch.arange(len(q), device=q.device)
    requests = (rows[:, None] >= kv.query_start[1:][None, :]).sum(-1)
    chosen = topk[:, None, :].expand(-1, q.shape[1], -1) if topk.ndim == 2 else topk
    valid = (chosen >= 0) & (chosen <= kv.positions[:, None, None])
    if window:
        valid = valid & (chosen > kv.positions[:, None, None] - window)
    size = kv.key_cache.shape[1]
    safe = chosen.clamp(0, kv.block_table.shape[1] * size - 1).long()
    blocks = kv.block_table[requests[:, None, None], safe // size].long()
    heads = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // kv.key_cache.shape[2])
    keys = kv.key_cache[blocks, safe % size, heads[None, :, None]].float()
    values = kv.value_cache[blocks, safe % size, heads[None, :, None]].float()
    values = values.masked_fill(~valid[..., None], 0)
    scores = (q.float()[:, :, None, :] * keys).sum(-1) * scale
    scores = scores.masked_fill(~valid, -float("inf"))
    lse = scores.logsumexp(-1)
    weights = torch.where(valid, (scores - lse[..., None]).exp(), 0)
    return (weights[..., None] * values).sum(-2).to(q.dtype), lse


@dataclass
class DenseAttention:
    """Dense masked-softmax kernel for the seeded CPU decoder and selection oracles."""

    kv: PagedKV
    scale: float

    def __call__(self, q: torch.Tensor, window: int, topk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if topk is None:
            topk = torch.arange(self.kv.block_table.shape[1] * self.kv.key_cache.shape[1], device=q.device)
            topk = topk[None, :].expand(len(q), -1)
        return selected_attention(self.kv, q, self.scale, window, topk)

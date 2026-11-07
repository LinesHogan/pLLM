"""Read-only paged attention views and authored reads, used by backends and CPU checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

# Planning uses FA2 pages; the tiny CPU decoder crosses boundaries with shorter pages.
PAGE_SIZE = 16
CHECK_PAGE_SIZE = 4


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
    query_count: int
    read_attention: AttentionRead
    storage_window: int = 0
    prompt_lens: torch.Tensor | None = None

    @property
    def requests(self) -> torch.Tensor:
        """Request index per query, materialized only by reads that need token positions.

        CUDA-graph padding rows past the last query map to the last request; their outputs are discarded.
        """
        rows = torch.arange(self.query_count, device=self.query_start.device)
        return torch.searchsorted(self.query_start[1:], rows, right=True).clamp(max=len(self.seq_lens) - 1)

    @property
    def positions(self) -> torch.Tensor:
        """Logical query positions derived from vLLM's packed request metadata."""
        requests = self.requests
        rows = torch.arange(self.query_count, device=self.query_start.device)
        return self.seq_lens[requests] - self.query_start[requests + 1] + rows

    @property
    def sink_lens(self) -> torch.Tensor | None:
        """Prompt length per query for authored selection and dense reference reads."""
        return self.prompt_lens[self.requests] if self.prompt_lens is not None else None

    @property
    def window_start(self) -> torch.Tensor:
        """First retained tail position per query; sink_lens also preserves prompt positions."""
        return (
            (self.positions + 1 - self.storage_window).clamp_min(0)
            if self.storage_window
            else torch.zeros_like(self.positions)
        )

    def attend(
        self, q: torch.Tensor, *, window: int = 0, topk: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention [rows, heads, dim] and float32 LSE [rows, heads].

        With no selection this uses the backend's native causal kernel. Selections
        intersect causality and the optional window; an empty read returns zero
        output and negative-infinity LSE. Indices must be distinct within a head.
        Queries keep the packed row count and head_dim; their head count must
        be a positive multiple of the cache's KV heads.
        """
        if (
            q.ndim != 3
            or q.shape[0] != self.query_count
            or q.shape[-1] != self.key_cache.shape[-1]
            or q.shape[1] < 1
            or q.shape[1] % self.key_cache.shape[2] != 0
            or q.device != self.key_cache.device
        ):
            raise ValueError(
                "kv.attend needs queries [packed rows, heads, head_dim] on the cache device; "
                "keep packed rows and head_dim unchanged and use a multiple of the KV heads"
            )
        if self.storage_window and self.prompt_lens is None:
            window = min(window, self.storage_window) if window else self.storage_window
        return self.read_attention(q, window, topk)


def selected_attention(
    kv: PagedKV, q: torch.Tensor, scale: float, window: int, topk: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only selected positions; shared CPU/CUDA reference for dense read selection."""
    requests = kv.requests
    chosen = topk[:, None, :].expand(-1, q.shape[1], -1) if topk.ndim == 2 else topk
    valid = (chosen >= 0) & (chosen <= kv.positions[:, None, None])
    if kv.storage_window:
        retained = chosen >= kv.window_start[:, None, None]
        if kv.sink_lens is not None:
            retained = retained | (chosen < kv.sink_lens[:, None, None])
        valid = valid & retained
    if window:
        valid = valid & (chosen > kv.positions[:, None, None] - window)
    size = kv.key_cache.shape[1]
    # Never dereference reclaimed pages, even for masked lanes of a gather.
    safe = torch.where(valid, chosen.clamp(0, kv.block_table.shape[1] * size - 1), 0).long()
    blocks = torch.where(valid, kv.block_table[requests[:, None, None], safe // size], 0).long()
    heads = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // kv.key_cache.shape[2])
    keys = kv.key_cache[blocks, safe % size, heads[None, :, None]].float()
    values = kv.value_cache[blocks, safe % size, heads[None, :, None]].float()
    keys = keys.masked_fill(~valid[..., None], 0)
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
    max_seq_len: int = 0

    def __call__(self, q: torch.Tensor, window: int, topk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if topk is None:
            length = self.max_seq_len or self.kv.block_table.shape[1] * self.kv.key_cache.shape[1]
            topk = torch.arange(length, device=q.device)
            topk = topk[None, :].expand(len(q), -1)
        return selected_attention(self.kv, q, self.scale, window, topk)

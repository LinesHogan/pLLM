"""Read-only paged attention views and authored reads, used by backends and CPU checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

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

    Dense pages have shape [blocks, block_size, kv_heads, head_dim]. MLA stores
    [blocks, block_size, latent_rank] and [blocks, block_size, rope_dim] views,
    with per-head [nope_dim, latent_rank] and [latent_rank, v_dim] projections.
    ``topk`` contains
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
    storage_kind: Literal["kv", "mla_latent"] = "kv"
    latent_cache: torch.Tensor | None = None
    rope_cache: torch.Tensor | None = None
    w_uk_t: torch.Tensor | None = None
    w_uv: torch.Tensor | None = None

    @classmethod
    def from_latent(
        cls,
        latent: torch.Tensor,
        rope: torch.Tensor,
        w_uk_t: torch.Tensor,
        w_uv: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start: torch.Tensor,
        query_count: int,
        read: AttentionRead,
    ) -> PagedKV:
        """Borrow compressed pages and per-head projections. Dense cache slots are empty."""
        empty = latent.new_empty(0)
        return cls(
            empty,
            empty,
            block_table,
            seq_lens,
            query_start,
            query_count,
            read,
            storage_kind="mla_latent",
            latent_cache=latent,
            rope_cache=rope,
            w_uk_t=w_uk_t,
            w_uv=w_uv,
        )

    @property
    def is_latent(self) -> bool:
        """Whether one compressed vector and one RoPE key are shared by all heads."""
        return self.storage_kind == "mla_latent"

    @property
    def pages(self) -> torch.Tensor:
        """Storage defining the cache device and block size."""
        if self.is_latent:
            assert self.latent_cache is not None
            return self.latent_cache
        return self.key_cache

    @property
    def q_head_dim(self) -> int:
        """Width of one query head before matrix absorption."""
        if self.is_latent:
            assert self.w_uk_t is not None and self.rope_cache is not None
            return self.w_uk_t.shape[1] + self.rope_cache.shape[-1]
        return self.key_cache.shape[-1]

    @property
    def v_head_dim(self) -> int:
        """Width of one output head after attention."""
        if self.is_latent:
            assert self.w_uv is not None
            return self.w_uv.shape[-1]
        return self.value_cache.shape[-1]

    @property
    def kv_heads(self) -> int:
        """Number of independent vectors stored per token."""
        return 1 if self.is_latent else self.key_cache.shape[2]

    @property
    def requests(self) -> torch.Tensor:
        """Request index per query, materialized only by reads that need token positions.

        CUDA-graph padding rows past the last query map to the last request; their outputs are discarded.
        """
        rows = torch.arange(self.query_count, device=self.pages.device)
        return torch.searchsorted(self.query_start[1:], rows, right=True).clamp(max=len(self.seq_lens) - 1)

    @property
    def positions(self) -> torch.Tensor:
        """Logical query positions derived from vLLM's packed request metadata."""
        requests = self.requests
        rows = torch.arange(self.query_count, device=self.pages.device)
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
        match the MLA projection heads or group evenly over dense KV heads.
        """
        if (
            q.ndim != 3
            or q.shape[0] != self.query_count
            or q.shape[-1] != self.q_head_dim
            or q.shape[1] < 1
            or q.shape[1] % self.kv_heads != 0
            or (self.is_latent and (self.w_uk_t is None or q.shape[1] != self.w_uk_t.shape[0]))
            or q.device != self.pages.device
        ):
            raise ValueError(
                "kv.attend needs queries [packed rows, heads, head_dim] on the cache device; "
                "keep packed rows and head_dim unchanged and match the cache head geometry"
            )
        if self.storage_window and self.prompt_lens is None:
            window = min(window, self.storage_window) if window else self.storage_window
        return self.read_attention(q, window, topk)


def _page_indices(kv: PagedKV, chosen: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axes = (-1,) + (1,) * (chosen.ndim - 1)
    requests = kv.requests.reshape(axes)
    position = kv.positions.reshape(axes)
    valid = (chosen >= 0) & (chosen <= position)
    if kv.storage_window:
        retained = chosen > position - kv.storage_window
        if kv.prompt_lens is not None:
            retained |= chosen < kv.prompt_lens[requests]
        valid &= retained
    if window:
        valid &= chosen > position - window
    size = kv.pages.shape[1]
    # Masked lanes may point to reclaimed pages.
    safe = torch.where(valid, chosen.clamp(0, kv.block_table.shape[1] * size - 1), 0).long()
    blocks = torch.where(valid, kv.block_table[requests, safe // size], 0).long()
    return blocks, safe % size, valid


def _softmax(scores: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = scores.masked_fill(~valid, -float("inf"))
    lse = scores.logsumexp(-1)
    return torch.where(valid, (scores - lse[..., None]).exp(), 0), lse


def selected_attention(
    kv: PagedKV, q: torch.Tensor, scale: float, window: int, topk: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only selected positions; shared CPU/CUDA reference for dense read selection."""
    chosen = topk[:, None, :].expand(-1, q.shape[1], -1) if topk.ndim == 2 else topk
    blocks, offsets, valid = _page_indices(kv, chosen, window)
    heads = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // kv.key_cache.shape[2])
    keys = kv.key_cache[blocks, offsets, heads[None, :, None]].float()
    values = kv.value_cache[blocks, offsets, heads[None, :, None]].float()
    keys = keys.masked_fill(~valid[..., None], 0)
    values = values.masked_fill(~valid[..., None], 0)
    scores = (q.float()[:, :, None, :] * keys).sum(-1) * scale
    weights, lse = _softmax(scores, valid)
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


def _absorb_query(kv: PagedKV, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert kv.w_uk_t is not None
    dim = kv.w_uk_t.shape[1]
    return torch.einsum("rhp,hpl->rhl", q[..., :dim].float(), kv.w_uk_t.float()), q[..., dim:].float()


def _latent_read(
    kv: PagedKV, absorbed: torch.Tensor, pe: torch.Tensor, scale: float, window: int, chosen: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kv.latent_cache is not None and kv.rope_cache is not None
    blocks, offsets, valid = _page_indices(kv, chosen, window)
    latent = kv.latent_cache[blocks, offsets].float().masked_fill(~valid[..., None], 0)
    rope = kv.rope_cache[blocks, offsets].float()
    if chosen.ndim == 2:
        scores = torch.einsum("rhl,rkl->rhk", absorbed, latent) + torch.einsum("rhd,rkd->rhk", pe, rope)
        weights, lse = _softmax(scores * scale, valid[:, None, :])
        return torch.einsum("rhk,rkl->rhl", weights, latent), lse
    scores = torch.einsum("rhl,rhkl->rhk", absorbed, latent) + torch.einsum("rhd,rhkd->rhk", pe, rope)
    weights, lse = _softmax(scores * scale, valid)
    return torch.einsum("rhk,rhkl->rhl", weights, latent), lse


def selected_latent_attention(
    kv: PagedKV, q: torch.Tensor, scale: float, window: int, topk: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read selected compressed tokens and project the weighted latent once per query head."""
    assert kv.w_uv is not None
    latent, lse = _latent_read(kv, *_absorb_query(kv, q), scale, window, topk)
    return torch.einsum("rhl,hlv->rhv", latent, kv.w_uv.float()).to(q.dtype), lse


def paged_latent_attention(
    kv: PagedKV, q: torch.Tensor, scale: float, window: int = 0, max_seq_len: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized latent reads across bounded tiles before the value projection."""
    assert kv.w_uv is not None
    absorbed, pe = _absorb_query(kv, q)
    length = max_seq_len or kv.block_table.shape[1] * kv.pages.shape[1]
    if window:
        length = min(length, window)
    start = kv.positions[:, None] + 1 - length if window else q.new_zeros((len(q), 1), dtype=torch.long)
    latent = torch.zeros_like(absorbed)
    lse = q.new_full(q.shape[:2], -float("inf"), dtype=torch.float32)
    for begin in range(0, length, 128):
        chosen = start + torch.arange(begin, min(begin + 128, length), device=q.device)
        part, normalizer = _latent_read(kv, absorbed, pe, scale, window, chosen)
        merged = torch.logaddexp(lse, normalizer)
        # Two empty tiles both have -inf LSE and contribute zero.
        finite = torch.where(merged.isneginf(), 0, merged)
        latent = latent * (lse - finite).exp()[..., None] + part * (normalizer - finite).exp()[..., None]
        lse = merged
    return torch.einsum("rhl,hlv->rhv", latent, kv.w_uv.float()).to(q.dtype), lse


@dataclass
class LatentAttention:
    """Matrix-absorbed reads for MLA pages on CPU and CUDA."""

    kv: PagedKV
    scale: float
    max_seq_len: int = 0

    def __call__(self, q: torch.Tensor, window: int, topk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if topk is None:
            return paged_latent_attention(self.kv, q, self.scale, window, self.max_seq_len)
        return selected_latent_attention(self.kv, q, self.scale, window, topk)

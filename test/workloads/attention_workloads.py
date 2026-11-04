"""Dense attention monitors and scored page reads shared by CPU and GPU contract tests."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

import pllm


@triton.jit
def _page_bound(
    Q,
    K,
    TABLE,
    START,
    POS,
    OUT,
    H: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    B: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    N: tl.constexpr,
    NS: tl.constexpr,
    BD: tl.constexpr,
):
    row, head, page = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    request = tl.sum((row >= tl.load(START + tl.arange(0, N), tl.arange(0, N) < NS, other=2147483647)).to(tl.int32)) - 1
    block = tl.load(TABLE + request * P + page)
    token, dim = tl.arange(0, B), tl.arange(0, BD)
    valid = page * B + token <= tl.load(POS + row)
    key = tl.load(
        K + block * KS0 + token[:, None] * KS1 + (head // (H // KH)) * KS2 + dim[None, :],
        valid[:, None] & (dim[None, :] < D),
        other=0,
    ).to(tl.float32)
    lower = tl.min(tl.where(valid[:, None], key, float("inf")), 0)
    upper = tl.max(tl.where(valid[:, None], key, -float("inf")), 0)
    query = tl.load(Q + row * QS0 + head * QS1 + dim, dim < D, other=0).to(tl.float32)
    bound = tl.sum(tl.where(dim < D, tl.maximum(query * lower, query * upper), 0), 0)
    tl.store(OUT + (row * H + head) * P + page, tl.where(page * B <= tl.load(POS + row), bound, -float("inf")))


@torch.library.custom_op("cattn_test::page_bounds", mutates_args=(), device_types="cuda")
def page_bounds(
    q: torch.Tensor, keys: torch.Tensor, table: torch.Tensor, starts: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Quest-style min/max page bounds; each query sees only its causal prefix."""
    rows, heads, dim = q.shape
    pages, block = table.shape[1], keys.shape[1]
    out = q.new_empty((rows, heads, pages), dtype=torch.float32)
    _page_bound[(rows, heads, pages)](
        q,
        keys,
        table,
        starts,
        positions,
        out,
        heads,
        keys.shape[2],
        dim,
        pages,
        block,
        *keys.stride()[:3],
        *q.stride()[:2],
        triton.next_power_of_2(len(starts)),
        len(starts),
        triton.next_power_of_2(dim),
    )
    return out


@page_bounds.register_fake
def _fake(q, keys, table, starts, positions):
    return q.new_empty((len(q), q.shape[1], table.shape[1]), dtype=torch.float32)


@page_bounds.register_kernel("cpu")
def _cpu(q, keys, table, starts, positions):
    requests = (torch.arange(len(q))[:, None] >= starts[1:][None, :]).sum(-1)
    pages = keys[table[requests].long()].float()
    causal = (
        torch.arange(table.shape[1] * keys.shape[1]).reshape(table.shape[1], keys.shape[1])[None]
        <= positions[:, None, None]
    )
    lower = pages.masked_fill(~causal[..., None, None], float("inf")).amin(2)
    upper = pages.masked_fill(~causal[..., None, None], -float("inf")).amax(2)
    lower, upper = (
        tensor.repeat_interleave(q.shape[1] // keys.shape[2], 2).transpose(1, 2) for tensor in (lower, upper)
    )
    scores = torch.maximum(q.float()[:, :, None] * lower, q.float()[:, :, None] * upper).sum(-1)
    return scores.masked_fill(
        torch.arange(table.shape[1])[None, None] * keys.shape[1] > positions[:, None, None], -float("inf")
    )


@dataclass(frozen=True)
class Monitor(pllm.Workload):
    """Recent attention mass, exported without changing the layer's output."""

    @pllm.custom_attention(layers=(0,))
    def split(self, q, kv):
        out, lse = kv.attend(q)
        _, near = kv.attend(q, window=4)
        return out, (near - lse).exp().mean(-1, keepdim=True)

    @pllm.custom_attention(layers=(2,))
    def carry(self, q, kv, share=pllm.attention(0, "aux"), lse=pllm.attention(1, "lse")):
        return kv.attend(q)[0], share + lse.mean(-1, keepdim=True) * 0

    @pllm.export(name="recent_mass", max_rows=4, max_cols=1)
    def report(self, share=pllm.attention(2, "aux")):
        return share

    @pllm.export(name="lse", max_rows=4, max_cols=32)
    def lse(self, value=pllm.attention(1, "lse")):
        return value


@dataclass(frozen=True)
class Quest(pllm.Workload):
    """Per-layer/per-head page budgets, with fixed-width -1 padding; storage stays dense."""

    @pllm.custom_attention(layers=(0, 1))
    def select(self, q, kv, layer=pllm.depth()):
        scores = page_bounds(q, kv.key_cache, kv.block_table, kv.query_start, kv.positions)
        count = min(2, scores.shape[-1])
        pages = scores.topk(count, dim=-1).indices
        rank = torch.arange(count, device=q.device)[None, None, :]
        budget = 1 + (torch.arange(q.shape[1], device=q.device)[None, :, None] + layer) % 2
        chosen = pages[..., None] * kv.key_cache.shape[1] + torch.arange(kv.key_cache.shape[1], device=q.device)
        chosen = chosen.masked_fill((rank >= budget)[..., None], -1).flatten(2)
        out, lse = kv.attend(q, topk=chosen)
        return out, lse


@dataclass(frozen=True)
class CoupledAttention(pllm.Workload):
    """Deliberately mixes requests; CPU schedule isolation must reject this read."""

    @pllm.custom_attention(layers=(0,))
    def read(self, q, kv):
        return (kv.attend(q)[0].float() + q.float().mean(0, keepdim=True)).to(q.dtype)


@dataclass(frozen=True)
class InstanceAttention(pllm.Workload):
    """An instance field can have the same extent as the planned packed-row capacity."""

    @property
    def pool(self) -> pllm.InstanceState:
        return pllm.InstanceState({"weights": (128,)})

    @pllm.custom_attention(layers=(0,))
    def read(self, q, kv, weights=pllm.parameter("pool", "weights")):
        return (kv.attend(q)[0].float() + weights.reshape(128).sum() * 0).to(q.dtype)


@dataclass(frozen=True)
class CacheLayout(pllm.Workload):
    """Selection exports have fixed width from the paged layout, including during CPU checks."""

    @pllm.custom_attention(layers=(0,))
    def read(self, q, kv):
        pages = (-torch.arange(kv.block_table.shape[1], device=q.device)).topk(2).indices
        chosen = pages[:, None] * kv.key_cache.shape[1] + torch.arange(kv.key_cache.shape[1], device=q.device)
        chosen = chosen.flatten().expand(len(q), -1)
        return kv.attend(q, topk=chosen)[0], chosen

    @pllm.export(max_rows=4, max_cols=64)
    def report(self, chosen: torch.Tensor = pllm.attention(0, "aux")) -> torch.Tensor:
        return chosen


@dataclass(frozen=True)
class WindowAttention(pllm.Workload):
    """vLLM owns and recycles sliding-window blocks."""

    layers: tuple[int, ...] = (0, 1)

    @pllm.custom_attention(layers=pllm.Attr("layers"), keep="window", window=4)
    def read(self, q, kv):
        return kv.attend(q)[0]


@dataclass(frozen=True)
class SinkAttention(pllm.Workload):
    """Keep prompt K/V and a recent generated tail using native R-SWA storage."""

    @pllm.custom_attention(layers=(0, 1), keep="window", window=4, sink=True)
    def read(self, q, kv):
        return kv.attend(q)[0]


@dataclass(frozen=True)
class SharedAttention(pllm.Workload):
    """Each odd layer reads the preceding layer's cache, with no cache of its own."""

    layers: tuple[int, ...] = (1, 3, 5, 7, 9)

    @pllm.custom_attention(layers=pllm.Attr("layers"), keep="none", source=1)
    def read(self, q, kv):
        return kv.attend(q)[0]


@dataclass(frozen=True)
class SharedWindowAttention(WindowAttention):
    """Sharing inherits its owner's window and never reads that owner's freed blocks."""

    layers: tuple[int, ...] = (0,)

    @pllm.custom_attention(layers=(1,), keep="none", source=1)
    def shared(self, q, kv):
        return kv.attend(q)[0]

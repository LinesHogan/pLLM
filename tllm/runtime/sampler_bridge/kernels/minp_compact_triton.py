#!/usr/bin/env python3
"""Fused min-p filtering + survivor compaction for MRv2 intervention steps.

vLLM's MRv2 sampler applies min-p as an in-place fp32 masking kernel, and the
sampler then reads the full masked vocab again to draw. For ESamp intervention
the draw only needs the survivors, so this kernel pair fuses both stages and
works straight off the bf16 logits: a tile-parallel partial-max pass computes
the temperature-scaled row maxima, then a tile-parallel pass compacts
surviving positions and their scaled values into fixed ``[rows, cap]``
buffers via per-row atomic cursors. The fp32 upcast copy, the in-place mask
write, and the full-vocab sampling read all leave the critical path.

The temperature scaling and threshold math mirror
``vllm.v1.worker.gpu.sample.min_p._min_p_kernel`` exactly (fp32 division,
``max + log(min_p)`` threshold, ``< threshold`` masked out), so the survivor
set is bit-identical to the upstream filter. Rows with more survivors than
``cap`` are truncated arbitrarily (tile completion order is not deterministic
across tiles); ``counts`` always exposes the true survivor count so callers
can detect truncation.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import availability depends on runtime env
    triton = None
    tl = None

_BLOCK_SIZE = 4096


if triton is not None:

    @triton.jit
    def _candidate_gumbel_argmax_kernel(
        vals_ptr,
        distiller_ptr,
        ids_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        affected_ptr,
        expanded_idx_mapping_ptr,
        out_ptr,
        beta,
        cap,
        K: tl.constexpr,
    ):
        arow = tl.program_id(0).to(tl.int64)
        row = tl.load(affected_ptr + arow).to(tl.int64)
        req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
        cols = tl.arange(0, K)
        lane = cols < cap
        vals = tl.load(vals_ptr + arow * cap + cols, mask=lane, other=float("-inf"))
        distiller = tl.load(distiller_ptr + arow * cap + cols, mask=lane, other=0.0)
        cand = (1.0 + beta) * vals - beta * distiller
        temperature = tl.load(temp_ptr + req_state_idx).to(tl.float32)
        if temperature >= 1e-5:
            seed = tl.load(seeds_ptr + req_state_idx)
            pos = tl.load(pos_ptr + row)
            gumbel_seed = tl.randint(seed, pos)
            u = tl.rand(gumbel_seed, cols)
            u = tl.maximum(u, 4.6566127342e-10)
            cand = cand - tl.log(-tl.log(u))
        winner = tl.argmax(cand, axis=0)
        token = tl.load(ids_ptr + arow * cap + winner)
        tl.store(out_ptr + row, token.to(tl.int64))

    @triton.jit
    def _minp_partial_max_kernel(
        logits_ptr,
        logits_stride,
        expanded_idx_mapping_ptr,
        temperature_ptr,
        partials_ptr,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
        temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
        # Mirror _temperature_kernel: temp 0.0 (greedy) and 1.0 leave logits
        # unscaled.
        scale = temperature != 0.0 and temperature != 1.0
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row * logits_stride + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        if scale:
            vals = vals / temperature
        tl.store(partials_ptr + row * tl.num_programs(1) + tile, tl.max(vals, axis=0))

    @triton.jit
    def _minp_compact_tiled_kernel(
        logits_ptr,
        logits_stride,
        expanded_idx_mapping_ptr,
        temperature_ptr,
        row_max_ptr,
        min_p_ptr,
        out_ids_ptr,
        out_vals_ptr,
        counts_ptr,
        vocab_size,
        cap,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
        temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
        min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
        row_max = tl.load(row_max_ptr + row).to(tl.float32)
        threshold = row_max + tl.log(tl.maximum(min_p, 1.1754943508222875e-38))
        scale = temperature != 0.0 and temperature != 1.0
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row * logits_stride + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        if scale:
            vals = vals / temperature
        keep = (vals >= threshold) & mask
        keep_i = keep.to(tl.int32)
        n = tl.sum(keep_i, axis=0)
        if n > 0:
            base = tl.atomic_add(counts_ptr + row, n)
            slot = base + tl.cumsum(keep_i, axis=0) - keep_i
            store_mask = keep & (slot < cap)
            tl.store(out_ids_ptr + row * cap + slot, offsets.to(tl.int32), mask=store_mask)
            tl.store(out_vals_ptr + row * cap + slot, vals, mask=store_mask)


def gumbel_argmax_candidates(
    vals: torch.Tensor,
    distiller_vals: torch.Tensor,
    ids: torch.Tensor,
    *,
    beta: float,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    affected: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    row_count: int,
) -> torch.Tensor | None:
    """One fused reweight + gumbel-argmax + scatter over ``[rows, cap]`` lanes.

    Returns int64 token ids for every logits row (rows in ``affected`` are
    drawn from the reweighted candidates with per-request seeded gumbel noise;
    greedy rows take the reweighted argmax). Never syncs the host.
    """
    if triton is None or vals.device.type != "cuda" or vals.ndim != 2:
        return None
    rows, cap = int(vals.shape[0]), int(vals.shape[1])
    out = torch.empty((int(row_count),), device=vals.device, dtype=torch.long)
    k_pow2 = 1 << max(0, (cap - 1)).bit_length()
    _candidate_gumbel_argmax_kernel[(rows,)](
        vals.contiguous(),
        distiller_vals.contiguous(),
        ids.contiguous(),
        temperature,
        seeds,
        pos,
        affected.to(device=vals.device, dtype=torch.int32).contiguous(),
        expanded_idx_mapping.to(device=vals.device).contiguous(),
        out,
        float(beta),
        cap,
        K=k_pow2,
    )
    return out


def minp_compact_candidates(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    min_p: torch.Tensor,
    *,
    cap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return (ids[rows, cap] int32, vals[rows, cap] fp32, counts[rows] int32).

    Never syncs the host. Unused slots are id=-1/val=-inf. Returns None when
    triton or CUDA is unavailable, or the logits are not 2-D CUDA.
    """
    if triton is None or logits.device.type != "cuda" or logits.ndim != 2:
        return None
    rows = int(logits.shape[0])
    vocab = int(logits.shape[1])
    if rows <= 0:
        return None
    cap = max(1, int(cap))
    device = logits.device
    mapping = expanded_idx_mapping.to(device=device).contiguous()
    num_tiles = triton.cdiv(vocab, _BLOCK_SIZE)
    partials = torch.empty((rows, num_tiles), device=device, dtype=torch.float32)
    _minp_partial_max_kernel[(rows, num_tiles)](
        logits,
        int(logits.stride(0)),
        mapping,
        temperature,
        partials,
        vocab,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    row_max = partials.max(dim=1).values
    ids = torch.full((rows, cap), -1, device=device, dtype=torch.int32)
    vals = torch.full((rows, cap), float("-inf"), device=device, dtype=torch.float32)
    counts = torch.zeros((rows,), device=device, dtype=torch.int32)
    _minp_compact_tiled_kernel[(rows, num_tiles)](
        logits,
        int(logits.stride(0)),
        mapping,
        temperature,
        row_max,
        min_p,
        ids,
        vals,
        counts,
        vocab,
        cap,
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return ids, vals, counts

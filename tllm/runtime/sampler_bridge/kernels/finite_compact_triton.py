#!/usr/bin/env python3
"""Sync-free compaction of finite (post-filter survivor) positions.

After vLLM's native min-p / top-k / top-p filtering, surviving candidate
tokens are exactly the finite entries of each logits row. Extracting them
with ``torch.isfinite(x).nonzero()`` forces a device-host sync every decode
step, which stalls the launch pipeline. This kernel instead appends survivor
positions into a fixed-capacity ``[rows, cap]`` buffer using per-row atomic
counters, so the whole selection stays on-device.

Rows with more survivors than ``cap`` are truncated (the overflow is visible
in ``counts`` for stats; callers may treat ``counts > cap`` as an
approximation signal). Ordering within a row is arbitrary.
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
    def _compact_finite_ids_kernel(
        logits_ptr,
        row_ids_ptr,
        out_ids_ptr,
        counts_ptr,
        vocab_size,
        cap,
        stride_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        arow = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        row = tl.load(row_ids_ptr + arow).to(tl.int64)
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size
        vals = tl.load(logits_ptr + row * stride_row + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        finite = (vals == vals) & (vals != float("inf")) & (vals != -float("inf")) & mask
        keep = finite.to(tl.int32)
        count = tl.sum(keep, axis=0)
        base = tl.atomic_add(counts_ptr + arow, count)
        slot = base + tl.cumsum(keep, axis=0) - keep
        tl.store(
            out_ids_ptr + arow * cap + slot,
            offsets.to(tl.int32),
            mask=(keep == 1) & (slot < cap),
        )


def compact_finite_ids(
    logits: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    cap: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (ids[rows, cap] int32, counts[rows] int32) or None if unsupported.

    ``ids[r, :min(counts[r], cap)]`` are surviving vocab positions (unordered);
    unused slots stay -1. Never syncs the host.
    """
    if triton is None or logits.device.type != "cuda":
        return None
    if logits.ndim != 2 or row_ids.ndim != 1 or int(row_ids.numel()) <= 0:
        return None
    rows = int(row_ids.numel())
    vocab = int(logits.shape[1])
    cap = max(1, int(cap))
    ids = torch.full((rows, cap), -1, device=logits.device, dtype=torch.int32)
    counts = torch.zeros((rows,), device=logits.device, dtype=torch.int32)
    grid = (rows, triton.cdiv(vocab, _BLOCK_SIZE))
    _compact_finite_ids_kernel[grid](
        logits,
        row_ids.to(device=logits.device, dtype=torch.int32),
        ids,
        counts,
        vocab,
        cap,
        int(logits.stride(0)),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return ids, counts

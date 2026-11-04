"""Gather MRv2 request token history into the persistent rows consumed by workload methods."""

import torch
import triton
import triton.language as tl


@triton.jit
def input_token_rows(
    history: torch.Tensor,
    requests: torch.Tensor,
    starts: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    stride: tl.constexpr,
    columns: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    """Write current and preceding ids from mapped request histories, one program per scheduled row."""
    row = tl.program_id(0)
    if row < tl.load(requests + 1):
        count = tl.load(requests)
        req = tl.arange(0, R)
        ends = tl.load(starts + req + 1, req < count, other=2147483647)
        batch = tl.sum((row >= ends).to(tl.int32), 0)
        state = tl.load(requests + 2 + batch)
        col = tl.arange(0, C)
        pos = tl.load(positions + row) - col
        ids = tl.load(history + state * stride + pos, (col < columns) & (pos >= 0), other=-1)
        tl.store(output + row * columns + col, ids, col < columns)

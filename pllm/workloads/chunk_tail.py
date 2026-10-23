"""Chunk-tail tensor bookkeeping shared by the causal fast-weight examples."""

import torch

from pllm.kernels.page_copy import prepend_rows


def join_tail(
    tails: tuple[torch.Tensor, ...], rows: tuple[torch.Tensor, ...], length: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, ...]:
    """Prepend the unfinished chunk before processing newly captured rows."""
    return tuple(prepend_rows(list(tails), list(rows), length, chunk - 1))


def save_tail(
    tails: tuple[torch.Tensor, ...], rows: tuple[torch.Tensor, ...], length: torch.Tensor, trained: int
) -> int:
    """Keep only rows that have not participated in a complete update."""
    remaining = len(rows[0]) - trained
    for tail, values in zip(tails, rows, strict=True):
        tail.zero_()
        tail[:remaining].copy_(values[trained:])
    length.fill_(remaining)
    return remaining

"""Counted pending-row math used by overlay execution and chunked trainers."""

import torch

from pllm.kernels.page_copy import prepend_rows


def join_tail(
    tails: tuple[torch.Tensor, ...],
    rows: tuple[torch.Tensor, ...],
    length: torch.Tensor,
    chunk: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Return bounded columns and their device-resident valid row count."""
    return prepend_rows(tails, rows, length, chunk), length + len(rows[0])


def save_tail(
    tails: tuple[torch.Tensor, ...],
    rows: tuple[torch.Tensor, ...],
    length: torch.Tensor,
    valid: torch.Tensor,
    trained: torch.Tensor,
) -> torch.Tensor:
    """Save unfinished valid rows; capacity padding never becomes retained history."""
    remaining = valid - trained
    for tail, values in zip(tails, rows, strict=True):
        positions = torch.arange(len(tail), device=length.device)
        selected = values.index_select(0, (positions + trained).clamp_max(len(values) - 1))
        tail.copy_(selected.masked_fill((positions >= remaining).view(-1, *((1,) * (tail.ndim - 1))), 0))
    length.copy_(remaining)
    return remaining

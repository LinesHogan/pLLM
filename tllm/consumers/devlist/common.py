#!/usr/bin/env python3
"""Common building blocks for devlist-inspired consumers."""

from __future__ import annotations

import math

import torch
from torch import nn


class LowRankAdapter(nn.Module):
    """Small LoRA-style adapter with zero-initialized output projection."""

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        rank: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("LowRankAdapter requires positive in_dim")
        if int(out_dim) <= 0:
            raise ValueError("LowRankAdapter requires positive out_dim")
        if int(rank) <= 0:
            raise ValueError("LowRankAdapter requires positive rank")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.down = nn.Linear(self.in_dim, self.rank, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(self.rank, self.out_dim, bias=False, dtype=dtype, device=device)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(float(self.in_dim)))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

"""Borrowed rotary cache and differentiable static arithmetic."""

from torch import Tensor
from torch.nn import Module

class RotaryEmbedding(Module):
    head_size: int
    rotary_dim: int
    cos_sin_cache: Tensor
    is_neox_style: bool

    @staticmethod
    def forward_static(
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: Tensor,
        is_neox_style: bool,
    ) -> tuple[Tensor, Tensor]: ...

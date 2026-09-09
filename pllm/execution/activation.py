"""Low-rank affine activation mathematics, independent of model hooks and state."""

import torch
from torch.nn import functional as F


def low_rank_affine_delta(rows: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                          rotation: torch.Tensor) -> torch.Tensor:
    return (F.linear(rows, weight, bias) - rows @ rotation) @ rotation.T

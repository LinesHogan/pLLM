"""Head-wise fp32 projection reads shared by overlay execution and examples."""

import torch


def head_correction(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return an fp32 residual without materializing a dense block-diagonal matrix."""
    heads, width, _ = weights.shape
    rows = inputs.float().reshape(-1, heads, width).transpose(0, 1)
    return torch.bmm(rows, weights.transpose(1, 2)).transpose(0, 1).reshape(inputs.shape)

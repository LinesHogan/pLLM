"""Head-wise fp32 projection reads shared by overlay execution and examples."""

import torch


def head_correction(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return an fp32 residual without materializing a dense block-diagonal matrix."""
    *batch, heads, width, _ = weights.shape
    rows = inputs.float().reshape(*batch, -1, heads, width).transpose(-3, -2)
    result = torch.bmm(rows.reshape(-1, rows.shape[-2], width), weights.reshape(-1, width, width).transpose(1, 2))
    return result.reshape(rows.shape).transpose(-3, -2).reshape(inputs.shape)

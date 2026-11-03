"""Head-wise projection reads with at least FP32 accumulation shared by overlay execution and examples."""

import torch


def head_correction(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return a residual with at least FP32 accumulation without materializing a dense block-diagonal matrix."""
    *batch, heads, width, _ = weights.shape
    dtype = torch.promote_types(weights.dtype, torch.float32)
    weights = weights.to(dtype)
    rows = inputs.to(dtype).reshape(*batch, -1, heads, width).transpose(-3, -2)
    result = torch.bmm(rows.reshape(-1, rows.shape[-2], width), weights.reshape(-1, width, width).transpose(1, 2))
    return result.reshape(rows.shape).transpose(-3, -2).reshape(inputs.shape)

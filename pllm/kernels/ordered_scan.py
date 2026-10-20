"""Ordered additive carry accumulation and weight preparation for region scans."""

import torch
import triton
import triton.language as tl


@triton.jit
def _weights(
    increments: torch.Tensor,
    carry: torch.Tensor,
    base: torch.Tensor,
    weights: torch.Tensor,
    final: torch.Tensor,
    rows: tl.constexpr,
    columns: tl.constexpr,
    count: tl.constexpr,
    carry_row: tl.constexpr,
    carry_column: tl.constexpr,
    base_row: tl.constexpr,
    base_column: tl.constexpr,
    scale: tl.constexpr,
) -> None:
    width = rows * columns
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = index < width
    row, column = index // columns, index % columns
    state = tl.load(carry + row * carry_row + column * carry_column, mask, 0).to(tl.float32)
    fixed = tl.load(base + row * base_row + column * base_column, mask, 0).to(tl.float32)
    for chunk in range(count):
        tl.store(weights + chunk * width + index, fixed + state, mask)
        increment = tl.load(increments + chunk * width + index, mask, 0).to(tl.float32) * scale
        increment = increment.to(weights.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        state = (state + increment).to(weights.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    tl.store(final + index, state, mask)


def ordered_weights(
    increments: torch.Tensor, carry: torch.Tensor, base: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve every carry rounding and return pre-update effective weights."""
    weights = torch.empty_like(increments)
    final = torch.empty(carry.shape, device=carry.device, dtype=carry.dtype)
    _weights[((carry.numel() + 255) // 256,)](
        increments,
        carry,
        base,
        weights,
        final,
        carry.shape[0],
        carry.shape[1],
        len(increments),
        carry.stride(0),
        carry.stride(1),
        base.stride(0),
        base.stride(1),
        scale,
        enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
    )
    return weights, final

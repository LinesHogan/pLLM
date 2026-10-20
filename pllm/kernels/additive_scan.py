"""Fixed-capacity additive prefix kernels used by the region executor and graph replay."""

import torch
import triton
import triton.language as tl

from pllm.contracts.region_imports import ScanForm


@triton.jit
def _weights(
    increments: torch.Tensor,
    carry: torch.Tensor,
    base: torch.Tensor,
    weights: torch.Tensor,
    final: torch.Tensor,
    active: torch.Tensor,
    rows: tl.constexpr,
    columns: tl.constexpr,
    capacity: tl.constexpr,
    carry_row: tl.constexpr,
    carry_column: tl.constexpr,
    base_row: tl.constexpr,
    base_column: tl.constexpr,
    scale: tl.constexpr,
    levels: tl.constexpr,
    tree: tl.constexpr,
    block: tl.constexpr,
) -> None:
    width = rows * columns
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < width
    row, column = index // columns, index % columns
    initial = tl.load(carry + row * carry_row + column * carry_column, mask, 0).to(tl.float32)
    fixed = tl.load(base + row * base_row + column * base_column, mask, 0).to(tl.float32)
    count = tl.load(active)
    if tree:
        lane = tl.arange(0, capacity)
        increment = (
            tl.load(increments + lane[:, None] * width + index[None, :], (lane[:, None] < count) & mask[None, :], 0).to(
                tl.float32
            )
            * scale
        )
        prefix = increment.to(weights.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        for level in tl.static_range(levels):
            offset = 1 << level
            previous = tl.gather(prefix, tl.maximum(lane - offset, 0)[:, None] + tl.zeros((1, block), tl.int32), 0)
            prefix = tl.where(lane[:, None] >= offset, previous + prefix, prefix)
        prefix = (prefix + initial[None, :]).to(weights.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        tl.store(weights + index, fixed + initial, mask)
        tl.store(
            weights + (lane[:, None] + 1) * width + index[None, :],
            fixed[None, :] + prefix,
            (lane[:, None] + 1 < capacity) & mask[None, :],
        )
        last = tl.gather(prefix, tl.zeros((1, block), tl.int32) + tl.maximum(count - 1, 0), 0)
        state = tl.where(count > 0, tl.sum(last, 0), initial)
    else:
        state = initial
        for chunk in range(capacity):
            tl.store(weights + chunk * width + index, fixed + state, mask)
            increment = tl.load(increments + chunk * width + index, mask & (chunk < count), 0).to(tl.float32) * scale
            increment = increment.to(weights.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            updated = (state + increment).to(weights.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            state = tl.where(chunk < count, updated, state)
    tl.store(final + index, state, mask)


def launch_prefix(
    increments: torch.Tensor,
    carry: torch.Tensor,
    base: torch.Tensor,
    active: torch.Tensor,
    weights: torch.Tensor,
    final: torch.Tensor,
    scale: float,
    form: ScanForm,
    offsets: tuple[int, ...],
) -> None:
    """Launch into caller-owned, stable buffers; only the device count varies on replay."""
    block = 128 if form is ScanForm.HILLIS_STEELE else 256
    _weights[((carry.numel() + block - 1) // block,)](
        increments,
        carry,
        base,
        weights,
        final,
        active,
        carry.shape[0],
        carry.shape[1],
        len(increments),
        carry.stride(0),
        carry.stride(1),
        base.stride(0),
        base.stride(1),
        scale,
        len(offsets),
        form is ScanForm.HILLIS_STEELE,
        block,
        enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
    )


class AdditiveBuffers:
    """Fixed window storage owned by one invocation or one future graph slot."""

    def __init__(self, carry: torch.Tensor, window: int, chunk: int) -> None:
        self.increments = carry.new_empty((window, *carry.shape))
        self.weights = torch.empty_like(self.increments)
        self.final = torch.empty_like(carry)
        self.output = carry.new_empty((window, chunk, carry.shape[0]))


def launch_window(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    carry: torch.Tensor,
    base: torch.Tensor,
    active: torch.Tensor,
    buffers: AdditiveBuffers,
    scale: float,
    form: ScanForm,
    offsets: tuple[int, ...],
) -> None:
    """Run the whole window into stable buffers; staging and publication stay outside."""
    torch.bmm(targets.transpose(1, 2), inputs, out=buffers.increments)
    launch_prefix(buffers.increments, carry, base, active, buffers.weights, buffers.final, scale, form, offsets)
    torch.bmm(inputs, buffers.weights.transpose(1, 2), out=buffers.output)

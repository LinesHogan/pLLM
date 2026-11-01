"""Fixed-capacity additive prefix kernels used by prepared scans and graph replay."""

import torch
import triton
import triton.language as tl
from torch._functorch.autograd_function import VmapInfo

from pllm.contracts.tensor_policy import ScanForm


@triton.jit
def _weights(
    increments: torch.Tensor,
    carry: torch.Tensor,
    weights: torch.Tensor,
    final: torch.Tensor,
    active: torch.Tensor,
    rows: tl.constexpr,
    columns: tl.constexpr,
    capacity: tl.constexpr,
    carry_row: tl.constexpr,
    carry_column: tl.constexpr,
    increment_batch: tl.constexpr,
    carry_batch: tl.constexpr,
    active_batch: tl.constexpr,
    scale: tl.constexpr,
    levels: tl.constexpr,
    tree: tl.constexpr,
    block: tl.constexpr,
) -> None:
    # Dense deltas can cross 2**31 elements along either the request or prefix axis.
    batch = tl.program_id(1).to(tl.int64)
    width = rows * columns
    increments_ptr = increments + batch * increment_batch
    carry_ptr = carry + batch * carry_batch
    active_ptr = active + batch * active_batch
    weights_ptr = weights + batch * capacity * width
    final_ptr = final + batch * width
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    mask = index < width
    row, column = index // columns, index % columns
    initial = tl.load(carry_ptr + row * carry_row + column * carry_column, mask, 0).to(tl.float32)
    if tree:
        lane = tl.arange(0, capacity)
        increment = (
            tl.load(
                increments_ptr + lane[:, None].to(tl.int64) * width + index[None, :],
                tl.load(active_ptr + lane)[:, None] & mask[None, :],
                0,
            ).to(tl.float32)
            * scale
        )
        prefix = increment.to(weights_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        for level in tl.static_range(levels):
            offset = 1 << level
            previous = tl.gather(prefix, tl.maximum(lane - offset, 0)[:, None] + tl.zeros((1, block), tl.int32), 0)
            prefix = tl.where(lane[:, None] >= offset, previous + prefix, prefix)
        prefix = (prefix + initial[None, :]).to(weights_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        tl.store(weights_ptr + index, initial, mask)
        tl.store(
            weights_ptr + (lane[:, None].to(tl.int64) + 1) * width + index[None, :],
            prefix,
            (lane[:, None] + 1 < capacity) & mask[None, :],
        )
        last = tl.gather(prefix, tl.zeros((1, block), tl.int32) + capacity - 1, 0)
        state = tl.where(tl.sum(tl.load(active_ptr + lane).to(tl.int32), 0) > 0, tl.sum(last, 0), initial)
    else:
        state = initial
        for chunk in range(capacity):
            offset = tl.cast(chunk, tl.int64) * width + index
            tl.store(weights_ptr + offset, state, mask)
            increment = tl.load(increments_ptr + offset, mask & tl.load(active_ptr + chunk), 0).to(tl.float32) * scale
            increment = increment.to(weights_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            updated = (state + increment).to(weights_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            state = tl.where(tl.load(active_ptr + chunk), updated, state)
    tl.store(final_ptr + index, state, mask)


def _launch(
    increments: torch.Tensor,
    carry: torch.Tensor,
    active: torch.Tensor,
    weights: torch.Tensor,
    final: torch.Tensor,
    scale: float,
    form: str,
) -> None:
    block = 128 if form == ScanForm.HILLIS_STEELE.value else 256
    width = carry.shape[1] * carry.shape[2]
    _weights[((width + block - 1) // block, len(carry))](
        increments,
        carry,
        weights,
        final,
        active,
        carry.shape[1],
        carry.shape[2],
        increments.shape[1],
        carry.stride(1),
        carry.stride(2),
        increments.stride(0),
        carry.stride(0),
        active.stride(0),
        scale,
        increments.shape[1].bit_length() - 1,
        form == ScanForm.HILLIS_STEELE.value,
        block,
        enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
    )


@torch.library.custom_op("pllm::additive_prefix", mutates_args=())
def prefix(
    increments: torch.Tensor, carry: torch.Tensor, active: torch.Tensor, scale: float, form: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the selected prefix independently to each request, with identical rounding."""
    weights = torch.empty_like(increments, memory_format=torch.contiguous_format)
    final = torch.empty_like(carry, memory_format=torch.contiguous_format)
    _launch(increments, carry, active, weights, final, scale, form)
    return weights, final


@prefix.register_fake  # pyright: ignore[reportUnknownMemberType]
def prefix_meta(
    increments: torch.Tensor, carry: torch.Tensor, active: torch.Tensor, scale: float, form: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Account for the prefix buffers without launching CUDA during planning."""
    return (
        torch.empty_like(increments, memory_format=torch.contiguous_format),
        torch.empty_like(carry, memory_format=torch.contiguous_format),
    )


@prefix.register_vmap  # pyright: ignore[reportUnknownMemberType]
def prefix_vmap(
    info: VmapInfo,
    in_dims: tuple[int | None, ...],
    increments: torch.Tensor,
    carry: torch.Tensor,
    active: torch.Tensor,
    scale: float,
    form: str,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    """Fold mapped requests into the kernel's grid; keep the prefix axis unchanged."""
    tensors = tuple(
        value.unsqueeze(0).expand(info.batch_size, *value.shape) if dim is None else value.movedim(dim, 0)
        for value, dim in zip((increments, carry, active), in_dims[:3], strict=True)
    )
    weights, final = prefix(*(value.flatten(0, 1).contiguous() for value in tensors), scale, form)
    return (weights.unflatten(0, tensors[0].shape[:2]), final.unflatten(0, tensors[1].shape[:2])), (0, 0)


def launch_prefix(
    increments: torch.Tensor,
    carry: torch.Tensor,
    active: torch.Tensor,
    weights: torch.Tensor,
    final: torch.Tensor,
    scale: float,
    form: ScanForm,
) -> None:
    """Write the selected prefix into caller-owned buffers."""
    _launch(
        increments.unsqueeze(0),
        carry.unsqueeze(0),
        active.unsqueeze(0),
        weights.unsqueeze(0),
        final.unsqueeze(0),
        scale,
        form.value,
    )

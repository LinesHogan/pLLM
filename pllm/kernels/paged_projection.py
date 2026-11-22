"""Paged dense matrix-vector readout for per-request overlay carries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_dense_mv(
    inputs: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    output: torch.Tensor,
    outputs: int,
    features: int,
    input_row: int,
    input_col: int,
    pool_page: int,
    pool_row: int,
    pool_col: int,
    output_row: int,
    output_col: int,
    scale: float,
    cast_input: tl.constexpr,
    scaled: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    row = tl.program_id(1).to(tl.int64)
    ns = tl.program_id(0).to(tl.int64) * block_n + tl.arange(0, block_n)
    n_mask = ns < outputs
    count = tl.load(counts + row)
    slot = tl.load(slots + row)
    out_ptrs = output + row * output_row + ns * output_col
    acc = tl.zeros((block_n,), dtype=tl.float32)
    if (count <= 0) | (slot < 0):
        tl.store(out_ptrs, acc.to(output.dtype.element_ty), mask=n_mask)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    else:
        # Pool strides can exceed 2**31 elements across admitted state pages.
        x_base = inputs + row * input_row
        w_base = pool + slot.to(tl.int64) * pool_page + ns[:, None] * pool_row
        for k0 in range(0, features, block_k):
            ks = k0 + tl.arange(0, block_k).to(tl.int64)
            k_mask = ks < features
            x = tl.load(x_base + ks * input_col, mask=k_mask, other=0.0)
            w = tl.load(w_base + ks[None, :] * pool_col, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            if cast_input:
                w = w.to(x.dtype)
            acc += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
        result = acc.to(output.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        if scaled:
            result = (result.to(tl.float32) * scale).to(output.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        tl.store(out_ptrs, result, mask=n_mask)


def _eager_paged_dense_mv(
    inputs: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    cast_input: bool,
    scale: float,
) -> torch.Tensor:
    output = inputs.new_zeros((len(inputs), pool.shape[1]))
    for index in range(len(inputs)):
        slot = int(slots[index].item())
        if int(counts[index].item()) > 0 and slot >= 0:
            weight = pool[slot].to(inputs.dtype) if cast_input else pool[slot]
            row = torch.mm(inputs[index : index + 1], weight.T)
            output[index : index + 1] = row if scale == 1.0 else row * scale
    return output


@torch.library.custom_op("pllm::paged_dense_mv", mutates_args=())
def paged_dense_mv(
    inputs: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    cast_input: bool,
    scale: float = 1.0,
) -> torch.Tensor:
    """Project active rows against their PTTS carry pages without gathering dense matrices."""
    rows, features = inputs.shape
    outputs = pool.shape[1]
    if rows == 0 or outputs == 0:
        return inputs.new_empty((rows, outputs))
    if features == 0:
        return inputs.new_zeros((rows, outputs))
    if not inputs.is_cuda or inputs.dtype == torch.float64 or pool.dtype == torch.float64:
        return _eager_paged_dense_mv(inputs, pool, slots, counts, cast_input, scale)
    output = inputs.new_empty((rows, outputs))
    block_n = 8
    block_k = 256 if features >= 512 else 64
    warps = 4 if block_k == 256 else 2
    _paged_dense_mv[(triton.cdiv(outputs, block_n), rows)](
        inputs,
        pool,
        slots,
        counts,
        output,
        outputs,
        features,
        inputs.stride(0),
        inputs.stride(1),
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        output.stride(0),
        output.stride(1),
        scale,
        cast_input,
        scale != 1.0,
        block_n,
        block_k,
        num_warps=warps,  # pyright: ignore[reportCallIssue]
    )
    return output


@paged_dense_mv.register_fake  # pyright: ignore[reportUnknownMemberType]
def paged_dense_mv_meta(
    inputs: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    cast_input: bool,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return the projected row shape without reading carry pages."""
    return inputs.new_empty((len(inputs), pool.shape[1]))

"""CUDA-graph-safe gated copy, residual-edit and projection kernels for pllm.model.LayerRoute.

Owned by the kernel layer and called through pllm.execution.layer_route by vLLM's HiddenTaps, VllmActivationPatch and
ParameterOverlay. Inactive candidate layers check row-level int64 routing metadata in L2 cache and exit before reading
or writing HBM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _routed_tap_copy_kernel(
    output_ptr: torch.Tensor,
    src_rows_ptr: torch.Tensor,
    dst_slots_ptr: torch.Tensor,
    row_layers_ptr: torch.Tensor,
    pool_ptr: torch.Tensor,
    slot_layers_ptr: torch.Tensor,
    out_stride0: int,
    out_stride1: int,
    pool_stride: int,
    width: int,
    layer_id: int,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dst = tl.load(dst_slots_ptr + row)
    target_layer = tl.load(row_layers_ptr + row)
    if (dst < 0) | (target_layer != layer_id):
        return
    src = tl.load(src_rows_ptr + row)
    if src < 0:
        return
    block_id = tl.program_id(1)
    if block_id == 0:
        tl.store(slot_layers_ptr + dst.to(tl.int64), tl.full((), layer_id, dtype=tl.int64))
    cols = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    values = tl.load(output_ptr + src.to(tl.int64) * out_stride0 + cols * out_stride1, mask=mask, other=0.0)
    tl.store(pool_ptr + dst.to(tl.int64) * pool_stride + cols, values, mask=mask)


@torch.library.custom_op("pllm::routed_tap_copy", mutates_args=("pool", "slot_layers"))
def routed_tap_copy(
    output: torch.Tensor,
    src_rows: torch.Tensor,
    dst_slots: torch.Tensor,
    row_layers: torch.Tensor,
    pool: torch.Tensor,
    slot_layers: torch.Tensor,
    layer_id: int,
) -> None:
    """Copy `output[src_rows]` into `pool[dst_slots]` and record `layer_id` in `slot_layers[dst_slots]` on match."""
    tap_rows = dst_slots.numel()
    if tap_rows == 0 or output.numel() == 0:
        return
    flat_pool = pool.view(pool.shape[0], -1)
    width = flat_pool.shape[1]
    if not output.is_cuda:
        flat_out = output.reshape(output.shape[0], -1)
        valid = (dst_slots >= 0) & (src_rows >= 0) & (row_layers == layer_id)
        if bool(valid.any()):
            matched_slots = dst_slots[valid]
            flat_pool.index_copy_(0, matched_slots, flat_out.index_select(0, src_rows[valid]).to(pool.dtype))
            slot_layers.index_fill_(0, matched_slots, layer_id)
        return
    out_2d = output if output.ndim == 2 else output.contiguous().view(output.shape[0], -1)
    block = 256
    grid = (tap_rows, triton.cdiv(width, block))
    _routed_tap_copy_kernel[grid](
        out_2d,
        src_rows,
        dst_slots,
        row_layers,
        flat_pool,
        slot_layers,
        out_2d.stride(0),
        out_2d.stride(1),
        flat_pool.stride(0),
        width,
        layer_id,
        BLOCK=block,
    )


def _routed_tap_copy_fake(
    output: torch.Tensor,
    src_rows: torch.Tensor,
    dst_slots: torch.Tensor,
    row_layers: torch.Tensor,
    pool: torch.Tensor,
    slot_layers: torch.Tensor,
    layer_id: int,
) -> None:
    return None


routed_tap_copy.register_fake(_routed_tap_copy_fake)  # pyright: ignore[reportUnknownMemberType]


@triton.jit
def _masked_patch_add_into_kernel(
    target_ptr: torch.Tensor,
    table_ptr: torch.Tensor,
    pool_ptr: torch.Tensor,
    table_stride: int,
    target_stride: int,
    pool_stride: int,
    width: int,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    if tl.load(table_ptr + row) == 0:
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < width
    slot = tl.load(table_ptr + table_stride + row)
    dst_ptrs = target_ptr + row.to(tl.int64) * target_stride + cols
    h = tl.load(dst_ptrs, mask=col_mask, other=0.0)
    raw_delta = tl.load(pool_ptr + slot.to(tl.int64) * pool_stride + cols, mask=col_mask, other=0.0)
    delta = raw_delta.to(target_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    tl.store(dst_ptrs, h + delta, mask=col_mask)


@torch.library.custom_op("pllm::masked_patch_add_into", mutates_args=("target",))
def masked_patch_add_into(target: torch.Tensor, table: torch.Tensor, pool: torch.Tensor) -> None:
    """Add `pool[table[1, r]].to(target.dtype)` into `target[r]` in place where `table[0, r] != 0`."""
    rows, width = target.shape
    if rows == 0:
        return
    if not target.is_cuda or target.dtype == torch.float64 or target.stride(-1) != 1:
        mask = table[0, :rows].bool()
        if not target.is_cuda and not bool(mask.any()):
            return
        slots = table[1, :rows]
        delta = pool.index_select(0, slots).to(target.dtype)
        target.copy_(torch.where(mask[:, None], target + delta, target))
        return
    block = 256
    grid = (rows, triton.cdiv(width, block))
    _masked_patch_add_into_kernel[grid](
        target,
        table,
        pool,
        table.stride(0),
        target.stride(0),
        pool.stride(0),
        width,
        BLOCK=block,
    )


def _masked_patch_add_into_fake(target: torch.Tensor, table: torch.Tensor, pool: torch.Tensor) -> None:
    return None


masked_patch_add_into.register_fake(_masked_patch_add_into_fake)  # pyright: ignore[reportUnknownMemberType]


def masked_patch_add(hidden: torch.Tensor, table: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """Out-of-place `masked_patch_add_into` over a fresh contiguous copy of `hidden`."""
    out = hidden.contiguous().clone()
    masked_patch_add_into(out, table, pool)
    return out


@triton.jit
def _patch_affine_coeff_kernel(
    hidden_ptr: torch.Tensor,
    table_ptr: torch.Tensor,
    weight_ptr: torch.Tensor,
    bias_ptr: torch.Tensor,
    rot_ptr: torch.Tensor,
    coeff_ptr: torch.Tensor,
    table_stride: int,
    hidden_stride: int,
    w_slot_stride: int,
    w_rank_stride: int,
    b_slot_stride: int,
    r_slot_stride: int,
    r_col_stride: int,
    rank: int,
    width: int,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    r = tl.program_id(1)
    if tl.load(table_ptr + row) == 0:
        return
    w_slot = tl.load(table_ptr + table_stride + row).to(tl.int64)
    b_slot = tl.load(table_ptr + 2 * table_stride + row).to(tl.int64)
    r_slot = tl.load(table_ptr + 3 * table_stride + row).to(tl.int64)
    acc_w = tl.zeros((), dtype=tl.float32)
    acc_r = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, width, BLOCK_K):
        cols = k_start + tl.arange(0, BLOCK_K)
        mask = cols < width
        h = tl.load(hidden_ptr + row.to(tl.int64) * hidden_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w_raw = tl.load(weight_ptr + w_slot * w_slot_stride + r * w_rank_stride + cols, mask=mask, other=0.0)
        w = w_raw.to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        rot_raw = tl.load(rot_ptr + r_slot * r_slot_stride + cols * r_col_stride + r, mask=mask, other=0.0)
        rot = rot_raw.to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        acc_w += tl.sum(h * w, axis=0)
        acc_r += tl.sum(h * rot, axis=0)
    b_raw = tl.load(bias_ptr + b_slot * b_slot_stride + r)
    b = b_raw.to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    wh = acc_w.to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    wh_b = (wh + b).to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    hr = acc_r.to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    c = (wh_b - hr).to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    tl.store(coeff_ptr + row * rank + r, c)


@triton.jit
def _patch_affine_out_into_kernel(
    target_ptr: torch.Tensor,
    table_ptr: torch.Tensor,
    rot_ptr: torch.Tensor,
    coeff_ptr: torch.Tensor,
    table_stride: int,
    target_stride: int,
    r_slot_stride: int,
    r_col_stride: int,
    rank: int,
    width: int,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    if tl.load(table_ptr + row) == 0:
        return
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < width
    r_slot = tl.load(table_ptr + 3 * table_stride + row).to(tl.int64)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r in range(0, rank):
        c = tl.load(coeff_ptr + row * rank + r)
        rot_raw = tl.load(rot_ptr + r_slot * r_slot_stride + cols * r_col_stride + r, mask=mask, other=0.0)
        rot = rot_raw.to(target_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        acc += c * rot
    dst_ptrs = target_ptr + row.to(tl.int64) * target_stride + cols
    h = tl.load(dst_ptrs, mask=mask, other=0.0)
    delta = acc.to(target_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    tl.store(dst_ptrs, h + delta, mask=mask)


@torch.library.custom_op("pllm::masked_patch_affine_into", mutates_args=("target", "coeff"))
def masked_patch_affine_into(
    target: torch.Tensor,
    table: torch.Tensor,
    weight_pool: torch.Tensor,
    bias_pool: torch.Tensor,
    rot_pool: torch.Tensor,
    coeff: torch.Tensor,
) -> None:
    """Apply `R (W h + b - R^T h)` into `target` in place where `table[0, r] != 0`, skipping HBM on inactive rows."""
    rows, width = target.shape
    rank = weight_pool.shape[1]
    if rows == 0:
        return
    if not target.is_cuda or target.dtype == torch.float64 or rank > 256 or target.stride(-1) != 1:
        mask = table[0, :rows].bool()
        if not target.is_cuda and not bool(mask.any()):
            return
        w = weight_pool.index_select(0, table[1, :rows]).to(target.dtype)
        b = bias_pool.index_select(0, table[2, :rows]).to(target.dtype)
        r = rot_pool.index_select(0, table[3, :rows]).to(target.dtype)
        h_row = target.unsqueeze(-2)
        c = h_row @ w.transpose(-1, -2) + b.unsqueeze(-2) - h_row @ r
        delta = (c @ r.transpose(-1, -2)).squeeze(-2)
        target.copy_(torch.where(mask[:, None], target + delta, target))
        return
    _patch_affine_coeff_kernel[(rows, rank)](
        target,
        table,
        weight_pool,
        bias_pool,
        rot_pool,
        coeff,
        table.stride(0),
        target.stride(0),
        weight_pool.stride(0),
        weight_pool.stride(1),
        bias_pool.stride(0),
        rot_pool.stride(0),
        rot_pool.stride(1),
        rank,
        width,
        BLOCK_K=256,
    )
    block_n = 256
    _patch_affine_out_into_kernel[(rows, triton.cdiv(width, block_n))](
        target,
        table,
        rot_pool,
        coeff,
        table.stride(0),
        target.stride(0),
        rot_pool.stride(0),
        rot_pool.stride(1),
        rank,
        width,
        BLOCK_N=block_n,
    )


def _masked_patch_affine_into_fake(
    target: torch.Tensor,
    table: torch.Tensor,
    weight_pool: torch.Tensor,
    bias_pool: torch.Tensor,
    rot_pool: torch.Tensor,
    coeff: torch.Tensor,
) -> None:
    return None


masked_patch_affine_into.register_fake(_masked_patch_affine_into_fake)  # pyright: ignore[reportUnknownMemberType]


def masked_patch_affine(
    hidden: torch.Tensor,
    table: torch.Tensor,
    weight_pool: torch.Tensor,
    bias_pool: torch.Tensor,
    rot_pool: torch.Tensor,
) -> torch.Tensor:
    """Out-of-place `masked_patch_affine_into` over a fresh contiguous copy of `hidden`."""
    out = hidden.contiguous().clone()
    coeff = torch.empty((len(out), weight_pool.shape[1]), dtype=torch.float32, device=out.device)
    masked_patch_affine_into(out, table, weight_pool, bias_pool, rot_pool, coeff)
    return out


@triton.jit
def _gate_owners_kernel(
    owners_ptr: torch.Tensor,
    row_layers_ptr: torch.Tensor,
    count: int,
    layer_id: int,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    layers = tl.load(row_layers_ptr + offsets, mask=mask, other=-1)
    keep = layers == layer_id
    owners = tl.load(owners_ptr + offsets, mask=mask, other=-1)
    tl.store(owners_ptr + offsets, tl.where(keep, owners, -1), mask=mask)


def gate_owners_by_layer(owners: torch.Tensor, row_layers: torch.Tensor, count: int, layer_id: int) -> None:
    """Set `owners[:count]` to -1 in place wherever `row_layers[:count] != layer_id`."""
    if count <= 0:
        return
    if not owners.is_cuda:
        owners[:count].masked_fill_(row_layers[:count] != layer_id, -1)
        return
    block = 256
    _gate_owners_kernel[(triton.cdiv(count, block),)](owners, row_layers, count, layer_id, BLOCK=block)


@triton.jit
def _routed_lora_down_kernel(
    hidden_ptr: torch.Tensor,
    slots_ptr: torch.Tensor,
    owners_ptr: torch.Tensor,
    a_pool_ptr: torch.Tensor,
    mid_ptr: torch.Tensor,
    h_stride: int,
    slots_stride: int,
    a_slot_stride: int,
    a_rank_stride: int,
    in_features: int,
    rank: int,
    block_rows: int,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    r = tl.program_id(1)
    owner = tl.load(owners_ptr + row)
    if owner < 0:
        return
    block_idx = owner // block_rows
    a_slot = tl.load(slots_ptr + block_idx * slots_stride).to(tl.int64)
    acc = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, in_features, BLOCK_K):
        cols = k_start + tl.arange(0, BLOCK_K)
        mask = cols < in_features
        x = tl.load(hidden_ptr + row.to(tl.int64) * h_stride + cols, mask=mask, other=0.0).to(tl.float32)
        a = (
            tl.load(a_pool_ptr + a_slot * a_slot_stride + r * a_rank_stride + cols, mask=mask, other=0.0)
            .to(hidden_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            .to(tl.float32)
        )
        acc += tl.sum(x * a, axis=0)
    mid_val = acc.to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    tl.store(mid_ptr + row * rank + r, mid_val)


@triton.jit
def _routed_lora_up_kernel(
    hidden_ptr: torch.Tensor,
    mid_ptr: torch.Tensor,
    slots_ptr: torch.Tensor,
    owners_ptr: torch.Tensor,
    b_pool_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    slots_stride: int,
    b_slot_stride: int,
    b_out_stride: int,
    out_stride0: int,
    out_stride1: int,
    out_offset: int,
    out_features: int,
    rank: int,
    block_rows: int,
    scale: float,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    owner = tl.load(owners_ptr + row)
    if owner < 0:
        return
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < out_features
    block_idx = owner // block_rows
    b_slot = tl.load(slots_ptr + block_idx * slots_stride + 1).to(tl.int64)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r in range(0, rank):
        m = tl.load(mid_ptr + row * rank + r)
        b_raw = tl.load(b_pool_ptr + b_slot * b_slot_stride + cols * b_out_stride + r, mask=mask, other=0.0)
        b = b_raw.to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        acc += m * b
    dst_ptrs = out_ptr + row.to(tl.int64) * out_stride0 + (out_offset + cols).to(tl.int64) * out_stride1
    prev = tl.load(dst_ptrs, mask=mask, other=0.0)
    corr = acc.to(hidden_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    scaled = (corr.to(tl.float32) * scale).to(hidden_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    delta = scaled.to(out_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    tl.store(dst_ptrs, prev + delta, mask=mask)


@triton.jit
def _routed_dense_add_kernel(
    hidden_ptr: torch.Tensor,
    slots_ptr: torch.Tensor,
    owners_ptr: torch.Tensor,
    w_pool_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    h_stride: int,
    slots_stride: int,
    w_slot_stride: int,
    w_out_stride: int,
    out_stride0: int,
    out_stride1: int,
    out_offset: int,
    in_features: int,
    out_features: int,
    block_rows: int,
    scale: float,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    owner = tl.load(owners_ptr + row)
    if owner < 0:
        return
    col = tl.program_id(1)
    block_idx = owner // block_rows
    w_slot = tl.load(slots_ptr + block_idx * slots_stride).to(tl.int64)
    acc = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, in_features, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        mask = ks < in_features
        x = tl.load(hidden_ptr + row.to(tl.int64) * h_stride + ks, mask=mask, other=0.0).to(tl.float32)
        w_raw = tl.load(w_pool_ptr + w_slot * w_slot_stride + col * w_out_stride + ks, mask=mask, other=0.0)
        w = w_raw.to(hidden_ptr.dtype.element_ty).to(tl.float32)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        acc += tl.sum(x * w, axis=0)
    dst_ptr = out_ptr + row.to(tl.int64) * out_stride0 + (out_offset + col) * out_stride1
    prev = tl.load(dst_ptr)
    corr = acc.to(hidden_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    scaled = (corr.to(tl.float32) * scale).to(hidden_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    delta = scaled.to(out_ptr.dtype.element_ty)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    tl.store(dst_ptr, prev + delta)


def gated_low_rank_add(
    hidden: torch.Tensor,
    slots: torch.Tensor,
    owners: torch.Tensor,
    a_pool: torch.Tensor,
    b_pool: torch.Tensor,
    mid: torch.Tensor,
    output: torch.Tensor,
    offset: int,
    width: int,
    block_rows: int,
    scale: float,
) -> None:
    """Add `scale * (hidden @ A^T) @ B^T` into `output[:, offset:offset+width]` only where `owners >= 0`."""
    rows, in_features = hidden.shape
    rank = a_pool.shape[1]
    if rows == 0:
        return
    contiguous = hidden if hidden.stride(-1) == 1 else hidden.contiguous()
    _routed_lora_down_kernel[(rows, rank)](
        contiguous,
        slots,
        owners,
        a_pool,
        mid,
        contiguous.stride(0),
        slots.stride(0),
        a_pool.stride(0),
        a_pool.stride(1),
        in_features,
        rank,
        block_rows,
        BLOCK_K=256,
    )
    block_n = 256
    _routed_lora_up_kernel[(rows, triton.cdiv(width, block_n))](
        contiguous,
        mid,
        slots,
        owners,
        b_pool,
        output,
        slots.stride(0),
        b_pool.stride(0),
        b_pool.stride(1),
        output.stride(0),
        output.stride(1),
        offset,
        width,
        rank,
        block_rows,
        scale,
        BLOCK_N=block_n,
    )


def gated_dense_add(
    hidden: torch.Tensor,
    slots: torch.Tensor,
    owners: torch.Tensor,
    w_pool: torch.Tensor,
    output: torch.Tensor,
    offset: int,
    width: int,
    block_rows: int,
    scale: float,
) -> None:
    """Add `scale * (hidden @ W^T)` into `output[:, offset:offset+width]` only where `owners >= 0`."""
    rows, in_features = hidden.shape
    if rows == 0:
        return
    contiguous = hidden if hidden.stride(-1) == 1 else hidden.contiguous()
    _routed_dense_add_kernel[(rows, width)](
        contiguous,
        slots,
        owners,
        w_pool,
        output,
        contiguous.stride(0),
        slots.stride(0),
        w_pool.stride(0),
        w_pool.stride(1),
        output.stride(0),
        output.stride(1),
        offset,
        in_features,
        width,
        block_rows,
        scale,
        BLOCK_K=256,
    )


__all__ = [
    "gate_owners_by_layer",
    "gated_dense_add",
    "gated_low_rank_add",
    "masked_patch_add",
    "masked_patch_add_into",
    "masked_patch_affine",
    "masked_patch_affine_into",
    "routed_tap_copy",
]

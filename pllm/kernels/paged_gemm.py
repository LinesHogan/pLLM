"""Device-offset grouped FP32 GEMM, including segmented weight gradients."""

import torch
import triton
import triton.language as tl


@triton.jit
def _select_rows(order: torch.Tensor, offsets: torch.Tensor, active: torch.Tensor, mask: torch.Tensor,
                 selected: torch.Tensor, ends: torch.Tensor, enabled: torch.Tensor, blocks: torch.Tensor,
                 rows: int, states: int, mask_stride: tl.constexpr, row_tile: tl.constexpr,
                 state_tile: tl.constexpr, count_only: tl.constexpr) -> None:
    block = tl.program_id(0)
    row = block * row_tile + tl.arange(0, row_tile)
    live = tl.load(offsets + states - 1)
    original = tl.load(order + row, row < live, 0)
    keep = tl.load(mask + original * mask_stride, row < live, 0)
    if count_only:
        tl.store(blocks + block, tl.sum(keep.to(tl.int32), axis=0))
    else:
        prefix = tl.cumsum(keep.to(tl.int32), axis=0) + tl.load(blocks + block - 1, block > 0, 0)
        tl.store(selected + prefix - 1, original, (row < live) & keep)
        logical = tl.load(active + row, row < rows, 0) & tl.load(mask + row * mask_stride, row < rows, 0)
        tl.store(enabled + row, logical, row < rows)
        state = tl.arange(0, state_tile)
        end = tl.load(offsets + state, state < states, 0)
        count = tl.gather(prefix, tl.minimum(tl.maximum(end - 1 - block * row_tile, 0), row_tile - 1), axis=0)
        owned = ((end > block * row_tile) & (end <= (block + 1) * row_tile)) | ((end < 1) & (block < 1))
        tl.store(ends + state, tl.where(end > 0, count, 0), (state < states) & owned)


def select_rows(order: torch.Tensor, offsets: torch.Tensor, active: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected, ends, enabled = torch.empty_like(order), torch.empty_like(offsets), torch.empty_like(active)
    tile = min(16384, triton.next_power_of_2(max(1, order.numel())))
    count = max(1, (order.numel() + tile - 1) // tile)
    blocks = torch.empty(count, dtype=torch.int32, device=order.device) if count > 1 else ends
    if count > 1:
        _select_rows[(count,)](order, offsets, active, mask, selected, ends, enabled, blocks, order.numel(),
            offsets.numel(), mask.stride(0), tile, triton.next_power_of_2(offsets.numel()), True)
        torch.cumsum(blocks, 0, dtype=torch.int32, out=blocks)
    _select_rows[(count,)](order, offsets, active, mask, selected, ends, enabled, blocks, order.numel(),
        offsets.numel(), mask.stride(0), tile, triton.next_power_of_2(offsets.numel()), False)
    return selected, ends, enabled


@triton.jit
def _write_pages(pool: torch.Tensor, slots: torch.Tensor, selected: torch.Tensor, values: torch.Tensor,
                 width: tl.constexpr, stride: tl.constexpr, selected_stride: tl.constexpr, add: tl.constexpr, scale: float) -> None:
    state = tl.program_id(0)
    column = tl.program_id(1) * 256 + tl.arange(0, 256)
    slot = tl.load(slots + tl.load(selected + state * selected_stride))
    valid = (slot >= 0) & (column < width)
    value = tl.load(values + state * width + column, valid, 0) * scale
    if add:
        value = tl.load(pool + slot * stride + column, valid, 0) + value
    tl.store(pool + slot * stride + column, value, valid)


def write_pages(pool: torch.Tensor, slots: torch.Tensor, selected: torch.Tensor, values: torch.Tensor, *, add: bool, scale: float) -> None:
    _write_pages[(selected.numel(), (pool.shape[1] + 255) // 256)](pool, slots, selected, values.contiguous(),
        pool.shape[1], pool.stride(0), selected.stride(0), add, scale)


@triton.jit
def _split_inner(a: torch.Tensor, b: torch.Tensor, slots: torch.Tensor, states: torch.Tensor, active: torch.Tensor,
                 output: torch.Tensor, k: tl.constexpr, n: tl.constexpr,
                 a0: tl.constexpr, a1: tl.constexpr, b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr,
                 n_tile: tl.constexpr) -> None:
    chunk, row = tl.program_id(0), tl.program_id(1)
    valid = tl.load(active + row)
    ni = tl.arange(0, n_tile)
    result = tl.zeros((n_tile,), tl.float32)
    if valid:
        slot = tl.load(slots + tl.load(states + row))
        ki = chunk * 256 + tl.arange(0, 256)
        lhs = tl.load(a + row * a0 + ki * a1, ki < k, 0)
        rhs = tl.load(b + slot * b0 + ki[:, None] * b1 + ni[None, :] * b2,
            (ki[:, None] < k) & (ni[None, :] < n), 0)
        result = tl.sum(lhs[:, None] * rhs, axis=0)
    tl.store(output + (row * tl.cdiv(k, 256) + chunk) * n + ni, result, ni < n)


def split_linear(inputs: torch.Tensor, matrices: torch.Tensor, slots: torch.Tensor, states: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    rows, inner = inputs.shape
    width = matrices.shape[2]
    chunks, tile = (inner + 255) // 256, 1 << (width - 1).bit_length()
    partials = inputs.new_empty((rows, chunks, width))
    _split_inner[(chunks, rows)](inputs, matrices, slots, states, active, partials, inner, width,
        inputs.stride(0), inputs.stride(1), matrices.stride(0), matrices.stride(1), matrices.stride(2), tile)
    return partials.sum(1)


@triton.jit
def _grouped(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor, output: torch.Tensor,
             slots: torch.Tensor, order: torch.Tensor, selected: torch.Tensor, selected_stride: tl.constexpr,
             m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
             a0: tl.constexpr, a1: tl.constexpr, b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr,
             weight_gradient: tl.constexpr, n_tile: tl.constexpr) -> None:
    output_group = tl.program_id(2)
    group = tl.load(selected + output_group * selected_stride) if selected_stride >= 0 else output_group
    begin = tl.load(offsets + group - 1, group > 0, 0)
    end = tl.load(offsets + group)
    mi = tl.program_id(0) * 16 + tl.arange(0, 16)
    ni = tl.program_id(1) * n_tile + tl.arange(0, n_tile)
    ki = tl.arange(0, 32)
    acc = tl.zeros((16, n_tile), tl.float32)
    if weight_gradient:
        if k < 0:
            for row in tl.range(begin, end):
                original = tl.load(order + row)
                lhs = tl.load(a + mi * a1 + original * a0, mi < m, 0)
                rhs = tl.load(b + original * b0 + ni * b1, ni < n, 0)
                acc = tl.fma(lhs[:, None], rhs[None, :], acc)
        else:
            for block in tl.range(tl.cdiv(end - begin, 32)):
                rows = begin + block * 32 + ki
                original = tl.load(order + rows, rows < end, 0)
                lhs = tl.load(a + mi[:, None] * a1 + original[None, :] * a0,
                    (mi[:, None] < m) & (rows[None, :] < end), 0)
                rhs = tl.load(b + original[:, None] * b0 + ni[None, :] * b1,
                    (rows[:, None] < end) & (ni[None, :] < n), 0)
                acc = tl.dot(lhs, rhs, acc, input_precision="ieee")
        tl.store(output + output_group * m * n + mi[:, None] * n + ni[None, :], acc,
            (mi[:, None] < m) & (ni[None, :] < n))
    else:
        if begin + tl.program_id(0) * 16 < end:
            slot = tl.load(slots + group)
            original = tl.load(order + begin + mi, begin + mi < end, 0)
            if k <= 8:
                for inner in range(k):
                    lhs = tl.load(a + original * a0 + inner * a1, begin + mi < end, 0)
                    rhs = tl.load(b + slot * b0 + inner * b1 + ni * b2, ni < n, 0)
                    acc = tl.fma(lhs[:, None], rhs[None, :], acc)
            else:
                for block in range(tl.cdiv(k, 32)):
                    inner = block * 32 + ki
                    lhs = tl.load(a + original[:, None] * a0 + inner[None, :] * a1,
                        (begin + mi[:, None] < end) & (inner[None, :] < k), 0)
                    rhs = tl.load(b + slot * b0 + inner[:, None] * b1 + ni[None, :] * b2,
                        (inner[:, None] < k) & (ni[None, :] < n), 0)
                    acc = tl.dot(lhs, rhs, acc, input_precision="ieee")
            tl.store(output + original[:, None] * n + ni[None, :], acc,
                (begin + mi[:, None] < end) & (ni[None, :] < n))


def grouped_linear(inputs: torch.Tensor, matrices: torch.Tensor, offsets: torch.Tensor,
                   slots: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    rows, inner = inputs.shape
    width = matrices.shape[2]
    tile = 128 if inner <= 8 and width >= 128 else 32
    output = inputs.new_empty((rows, width))
    _grouped[((rows + 15) // 16, (width + tile - 1) // tile, offsets.numel())](inputs, matrices, offsets, output,
        slots, order, offsets, -1, 0, width, inner, inputs.stride(0), inputs.stride(1),
        matrices.stride(0), matrices.stride(1), matrices.stride(2), False, tile)
    return output


def grouped_parameter_gradient(inputs: torch.Tensor, gradients: torch.Tensor,
                               offsets: torch.Tensor, order: torch.Tensor, selected: torch.Tensor | None = None) -> torch.Tensor:
    inner = inputs.shape[1]
    width = gradients.shape[1]
    small = inputs.shape[0] <= 16
    tile = 128 if small and width >= 128 else 32
    output = inputs.new_empty((offsets.numel() if selected is None else selected.numel(), inner, width))
    _grouped[((inner + 15) // 16, (width + tile - 1) // tile, output.shape[0])](inputs, gradients, offsets, output,
        offsets, order, offsets if selected is None else selected, -1 if selected is None else selected.stride(0),
        inner, width, -1 if small else 0, inputs.stride(0), inputs.stride(1),
        gradients.stride(0), gradients.stride(1), 0, True, tile)
    return output

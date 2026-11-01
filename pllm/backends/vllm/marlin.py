"""Transient BF16 inverse of vLLM's Marlin A16 layout, used by frozen_weight.

The tile and scale permutations invert marlin_utils_test.get_weight_perm and
marlin_utils.marlin_permute_scales. vLLM exposes repacking but no inverse.
"""

import torch
import triton
import triton.language as tl
from vllm.scalar_type import ScalarType, scalar_types


@triton.jit
def _unpack(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    output: torch.Tensor,
    inputs: tl.constexpr,
    outputs: tl.constexpr,
    padded_outputs: tl.constexpr,
    bits: tl.constexpr,
    group: tl.constexpr,
    groups: tl.constexpr,
    fp8: tl.constexpr,
) -> None:
    lane = tl.arange(0, 2048)
    row = (tl.program_id(0) // tl.cdiv(outputs, 128) * 16 + lane // 128).to(tl.int64)
    column = tl.program_id(0) % tl.cdiv(outputs, 128) * 128 + lane % 128
    valid = (row < inputs) & (column < outputs)
    # Keep each 16x128 tile together so packed words are reused within the CTA.
    thread = column % 8 * 4 + row % 8 // 2
    slot = column % 16 // 8 * 4 + row % 2 + 2 * (row % 16 // 8)
    if bits == 4:
        slot = slot // 2 + slot % 2 * 4
    else:
        slot = slot // 4 * 4 + slot % 2 * 2 + slot % 4 // 2
    position = row // 16 * padded_outputs * 16 + column // 64 * 1024 + thread * 32 + column % 64 // 16 * 8 + slot
    packed = tl.load(weight + position // (32 // bits), valid, 0)
    quant = (packed >> ((position % (32 // bits)).to(tl.int32) * bits)) & ((1 << bits) - 1)
    scale_column = column // 64 * 64 + column % 8 * 8 + column % 64 // 8
    if groups == 1:
        scale_column = column // 32 * 32 + column % 8 // 2 * 8 + column % 2 + 2 * (column % 32 // 8)
    scale = tl.load(scales + row // group * padded_outputs + scale_column, valid, 0).to(tl.float32)
    if fp8:
        # Marlin embeds FP8 bits in BF16 and folds the exponent bias into scales.
        value = (((quant & 127) << 20) | ((quant & 128) << 24)).to(tl.float32, bitcast=True) * scale
    else:
        zero = 1 << (bits - 1)
        if zeros is not None:
            zero_column = column // 64 * 64 + column % 8 * 8 + column % 64 // 8
            if bits == 4:
                zero_column = zero_column // 8 * 8 + zero_column % 8 // 2 + zero_column % 2 * 4
            else:
                zero_column = zero_column // 4 * 4 + zero_column % 2 * 2 + zero_column % 4 // 2
            packed_zero = tl.load(zeros + (row // group * padded_outputs + zero_column) // (32 // bits), valid, 0)
            zero = (packed_zero >> (zero_column % (32 // bits) * bits)) & ((1 << bits) - 1)
        value = (quant.to(tl.float32) - zero) * scale
    tl.store(output + row * outputs + column, value, valid)


def dequantize(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    shape: tuple[int, int],
    quant_type: ScalarType,
    group_size: int,
) -> torch.Tensor:
    """Return one fresh [output, input] BF16 operand; no tensors survive the caller."""
    inputs, outputs = shape
    result = torch.empty(shape, dtype=torch.bfloat16, device=weight.device)
    _unpack[(triton.cdiv(inputs, 16) * triton.cdiv(outputs, 128),)](
        weight,
        scales,
        zeros,
        result,
        inputs,
        outputs,
        scales.shape[1],
        quant_type.size_bits,
        inputs if group_size == -1 else group_size,
        scales.shape[0],
        quant_type == scalar_types.float8_e4m3fn,
    )
    return result.T

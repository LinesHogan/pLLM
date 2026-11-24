"""Overlay arithmetic shared by training and inference, and pinned projection gathers."""

from collections.abc import Callable
from functools import partial

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.execution.torch_private import disable_current_modes, measure
from pllm.kernels.head_overlay import head_correction

ROWS_PER_BLOCK = 32


def parameter_correction(
    hidden: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    operator: ParameterOverlayOperator,
    *,
    batched: bool = False,
) -> torch.Tensor:
    """Apply one operator while retaining each caller's GEMM or block BMM geometry."""
    if operator is ParameterOverlayOperator.HEAD_ADD:
        return head_correction(hidden, weights[0])
    multiply = torch.bmm if batched else torch.matmul
    if operator is ParameterOverlayOperator.GATED_LOW_RANK_ADD:
        a, b, gate_weight, gate_bias = weights
        dtype = hidden.dtype
        logits = multiply(hidden, gate_weight.to(dtype).transpose(-1, -2)) + gate_bias.to(dtype).unsqueeze(-2)
        gates = torch.softmax(logits.float(), dim=-1).to(dtype)
        experts = a.shape[-3]
        if gates.shape[-1] == experts + 1:
            gates = gates[..., 1:]
        if experts == 1:
            mid = multiply(hidden, a.squeeze(-3).to(dtype).transpose(-1, -2))
            return multiply(mid * gates, b.squeeze(-3).to(dtype).transpose(-1, -2))
        rank = a.shape[-2]
        out_features = b.shape[-2]
        if batched:
            a_flat = a.reshape(a.shape[0], experts * rank, a.shape[-1]).to(dtype)
            b_flat = b.permute(0, 2, 1, 3).reshape(b.shape[0], out_features, experts * rank).to(dtype)
        else:
            a_flat = a.reshape(experts * rank, a.shape[-1]).to(dtype)
            b_flat = b.permute(1, 0, 2).reshape(out_features, experts * rank).to(dtype)
        mid = multiply(hidden, a_flat.transpose(-1, -2))
        mid = (mid.reshape(*hidden.shape[:-1], experts, rank) * gates.unsqueeze(-1)).reshape(
            *hidden.shape[:-1], experts * rank
        )
        return multiply(mid, b_flat.transpose(-1, -2))
    result = multiply(hidden, weights[0].to(hidden.dtype).transpose(-1, -2))
    if operator is ParameterOverlayOperator.LOW_RANK_ADD:
        result = multiply(result, weights[1].to(hidden.dtype).transpose(-1, -2))
    return result


def uses_pinned_projection(*, in_forward: bool, graphs: bool) -> bool:
    """Ordinary overlays use pinned reads except during eager in-forward training."""
    return graphs or not in_forward


def projection_block_rows(rows: int, requests: int) -> int:
    """Uniform decode uses one row per matrix; prefill amortizes each field gather."""
    return 1 if rows <= requests else ROWS_PER_BLOCK


def projection_blocks(rows: int, requests: int) -> int:
    """Bound row blocks including each request's partially filled last block."""
    return min(rows, (rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK + requests - 1)


def projection_read(
    hidden: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    owners: torch.Tensor,
    *pools: torch.Tensor,
    operator: ParameterOverlayOperator,
) -> tuple[torch.Tensor, ...]:
    """Gather bounded blocks; each block reads one pinned matrix or factor pair."""
    blocks, count = rows.shape
    inputs = hidden.index_select(0, rows.flatten()).view(blocks, count, hidden.shape[-1])
    weights = tuple(pool.index_select(0, slots[:, index]) for index, pool in enumerate(pools))
    result = parameter_correction(inputs, weights, operator, batched=True).flatten(0, 1)
    return (result.index_select(0, owners.clamp_min(0)).masked_fill((owners < 0)[:, None], 0),)


def prepare_projection_read(
    examples: tuple[torch.Tensor, ...],
    operator: ParameterOverlayOperator,
    budget: int,
    *,
    blas_workspace_bytes: int,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Fund direct ATen temporaries without charging borrowed field-pool storage."""
    function = partial(projection_read, operator=operator)
    with disable_current_modes(), torch.inference_mode(False), FakeTensorMode():
        samples = tuple(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device) for t in examples)
        usage = measure(lambda: function(*samples))
    # The GPU boundary supplies library scratch; shape tracing itself never queries CUDA.
    if usage.workspace + blas_workspace_bytes > budget:
        raise ValueError("increase the projection execution budget to fund its tensor and BLAS workspace")
    return function

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
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Fund direct ATen temporaries without charging borrowed field-pool storage."""
    function = partial(projection_read, operator=operator)
    with disable_current_modes(), torch.inference_mode(False), FakeTensorMode():
        samples = tuple(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device) for t in examples)
        usage = measure(lambda: function(*samples))
    # BLAS handles own scratch outside the ATen allocation trace. Charge both
    # configured libraries, even though a particular bmm may only use cuBLAS.
    scratch = (
        torch.backends.cuda.cublas_workspace_size() + torch.backends.cuda.cublaslt_workspace_size()
        if examples[0].device.type == "cuda"
        else 0
    )
    if usage.workspace + scratch > budget:
        raise ValueError("increase the projection execution budget to fund its tensor and BLAS workspace")
    return function

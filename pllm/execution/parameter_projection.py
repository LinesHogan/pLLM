"""Pinned projection tensor programs used by the eager and captured model adapters."""

from collections.abc import Callable
from functools import partial

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import _disable_current_modes  # pyright: ignore[reportPrivateUsage]

from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.execution.memory import measure

ROWS_PER_BLOCK = 32


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
    if operator is ParameterOverlayOperator.HEAD_ADD:
        heads, width, _ = pools[0].shape[1:]
        inputs = inputs.float().view(blocks, count, heads, width).transpose(1, 2)
        result = torch.bmm(
            inputs.reshape(blocks * heads, count, width),
            weights[0].reshape(blocks * heads, width, width).transpose(1, 2),
        )
        result = result.view(blocks, heads, count, width).transpose(1, 2).reshape(blocks * count, heads * width)
    else:
        result = torch.bmm(inputs, weights[0].transpose(1, 2))
        if operator is ParameterOverlayOperator.LOW_RANK_ADD:
            result = torch.bmm(result, weights[1].transpose(1, 2))
        result = result.flatten(0, 1)
    return (result.index_select(0, owners.clamp_min(0)).masked_fill((owners < 0)[:, None], 0),)


def prepare_projection_read(
    examples: tuple[torch.Tensor, ...],
    operator: ParameterOverlayOperator,
    budget: int,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Fund direct ATen temporaries without charging borrowed field-pool storage."""
    function = partial(projection_read, operator=operator)
    with _disable_current_modes(), torch.inference_mode(False), FakeTensorMode():
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

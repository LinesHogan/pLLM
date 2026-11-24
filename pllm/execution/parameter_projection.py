"""Overlay arithmetic shared by training and inference, custom operator checks and pinned projection gathers."""

from __future__ import annotations

from collections.abc import Callable, Collection
from functools import partial
from hashlib import sha256
from types import FunctionType, MethodType

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding, ParameterOverlayOperator
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.method_dependencies import check_dependencies
from pllm.execution.torch_private import (
    CustomOpDef,
    CustomOpTrace,
    custom_op,
    disable_current_modes,
    fixed_shapes,
    measure,
)
from pllm.kernels.head_overlay import head_correction

ROWS_PER_BLOCK = 32


def is_static_instance(
    binding: ParameterOverlayFieldBinding,
    *,
    pinned: bool,
    updates_in_forward: bool,
    overlay_phases: tuple[HiddenPhase, ...],
    static_instances: Collection[str],
) -> bool:
    """Whether an overlay binding reads a frozen InstanceState directly without per-step tables or gathers."""
    return (
        pinned
        and not updates_in_forward
        and not overlay_phases
        and binding.route is None
        and binding.resource in static_instances
    )


def custom_operator_id(function: Callable[..., torch.Tensor]) -> str:
    """Stable identifier for a custom projection operator callable, folded into graph cache keys."""
    if isinstance(function, CustomOpDef):
        op = custom_op(function)
        assert op is not None
        payload = f"custom_op:{op.name}:{op.schema}"
    elif isinstance(function, MethodType):
        raw = function.__func__
        payload = f"method:{raw.__module__}.{raw.__qualname__}:{raw.__code__.co_code.hex()}:{raw.__code__.co_consts!r}"
    elif isinstance(function, FunctionType):
        payload = (
            f"function:{function.__module__}.{function.__qualname__}:"
            f"{function.__code__.co_code.hex()}:{function.__code__.co_consts!r}"
        )
    else:
        cls = type(function)
        payload = f"callable:{cls.__module__}.{cls.__qualname__}:{function!r}"
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def validate_custom_projection_operator(
    function: Callable[..., torch.Tensor],
    inputs: int,
    outputs: int,
    hidden_dtype: TensorDType,
    weight_specs: tuple[tuple[tuple[int, ...], TensorDType], ...],
    projection: str,
) -> None:
    """Check dependencies, mutates_args=(), meta shape/dtype/device and 0 host syncs on a custom operator."""
    if isinstance(function, CustomOpDef):
        # Validates mutates_args=() and register_fake.
        custom_op(function)
    else:
        check_dependencies(function)
    rows = 2
    with (
        disable_current_modes(),
        torch.inference_mode(False),
        FakeTensorMode(),
        fixed_shapes(),
        CustomOpTrace() as trace,
    ):
        hidden = torch.empty((rows, inputs), dtype=hidden_dtype.torch, device="meta")
        weights = tuple(torch.empty(shape, dtype=dtype.torch, device="meta") for shape, dtype in weight_specs)
        result = function(hidden, *weights)
    for definition, _, _ in trace.calls:
        custom_op(definition)
    if not isinstance(result, torch.Tensor):
        raise ValueError(
            f"custom projection operator for {projection} must return a torch.Tensor shaped [{rows}, {outputs}]; "
            f"got {type(result).__name__}"
        )
    if tuple(result.shape) != (rows, outputs):
        raise ValueError(
            f"custom projection operator for {projection} must return shape ({rows}, {outputs}); "
            f"got {tuple(result.shape)}"
        )
    if result.dtype not in (hidden_dtype.torch, torch.float32):
        raise ValueError(
            f"custom projection operator for {projection} must return {hidden_dtype.torch} or torch.float32; "
            f"got {result.dtype}"
        )
    if result.device != hidden.device:
        raise ValueError(
            f"custom projection operator for {projection} must return a tensor on the input device; got {result.device}"
        )


def parameter_correction(
    hidden: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    operator: ParameterOverlayOperator,
    *,
    batched: bool = False,
    custom_operator: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply one operator while retaining each caller's GEMM or block BMM geometry."""
    if operator is ParameterOverlayOperator.HEAD_ADD:
        return head_correction(hidden, weights[0])
    if operator is ParameterOverlayOperator.CUSTOM:
        assert custom_operator is not None
        if batched:
            return torch.vmap(custom_operator)(hidden, *weights)
        return custom_operator(hidden, *weights)
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
    custom_operator: Callable[..., torch.Tensor] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Gather bounded blocks; each block reads one pinned matrix or factor pair."""
    blocks, count = rows.shape
    inputs = hidden.index_select(0, rows.flatten()).view(blocks, count, hidden.shape[-1])
    weights = tuple(pool.index_select(0, slots[:, index]) for index, pool in enumerate(pools))
    result = parameter_correction(inputs, weights, operator, batched=True, custom_operator=custom_operator).flatten(
        0, 1
    )
    return (result.index_select(0, owners.clamp_min(0)).masked_fill((owners < 0)[:, None], 0),)


def prepare_projection_read(
    examples: tuple[torch.Tensor, ...],
    operator: ParameterOverlayOperator,
    budget: int,
    *,
    blas_workspace_bytes: int,
    custom_operator: Callable[..., torch.Tensor] | None = None,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Fund direct ATen temporaries without charging borrowed field-pool storage."""
    function = partial(projection_read, operator=operator, custom_operator=custom_operator)
    with disable_current_modes(), torch.inference_mode(False), FakeTensorMode():
        samples = tuple(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device) for t in examples)
        usage = measure(lambda: function(*samples))
    # The GPU boundary supplies library scratch; shape tracing itself never queries CUDA.
    if usage.workspace + blas_workspace_bytes > budget:
        raise ValueError("increase the projection execution budget to fund its tensor and BLAS workspace")
    return function


def prepare_static_projection_read(
    hidden: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    operator: ParameterOverlayOperator,
    budget: int,
    *,
    blas_workspace_bytes: int,
    custom_operator: Callable[..., torch.Tensor] | None = None,
) -> int:
    """Verify FakeTensorMode workspace budget for a direct static-instance projection operator."""
    with disable_current_modes(), torch.inference_mode(False), FakeTensorMode():
        sample_hidden = torch.empty_strided(hidden.shape, hidden.stride(), dtype=hidden.dtype, device=hidden.device)
        sample_weights = tuple(
            torch.empty_strided(w.shape, w.stride(), dtype=w.dtype, device=w.device) for w in weights
        )
        usage = measure(
            lambda: parameter_correction(
                sample_hidden, sample_weights, operator, batched=False, custom_operator=custom_operator
            )
        )
    if usage.workspace + blas_workspace_bytes > budget:
        raise ValueError("increase the projection execution budget to fund its tensor and BLAS workspace")
    return usage.workspace

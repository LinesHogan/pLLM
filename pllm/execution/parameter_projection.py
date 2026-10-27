"""Pinned projection tensor programs used by the eager and captured model adapters."""

from functools import partial

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import _disable_current_modes  # pyright: ignore[reportPrivateUsage]

from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.contracts.region_program import Block, Call, Program, Region, Value
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.tensor_compilation import TensorCompiler
from pllm.contracts.tensor_policy import TensorPolicy
from pllm.execution.exported_cell import ExportedCell, TensorCellModule
from pllm.execution.region_execution import RegionInstructionExecution
from pllm.execution.region_optimization import prepare_region_execution
from pllm.execution.tensor_compilation import DEFAULT_TENSOR_COMPILER

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
    policy: TensorPolicy,
    *,
    compiler: TensorCompiler = DEFAULT_TENSOR_COMPILER,
) -> RegionInstructionExecution:
    """Select a shared readonly-input region before serving; the caller owns its tensors."""
    names = ("hidden", "slots", "rows", "owners", *(f"pool{i}" for i in range(len(examples) - 4)))
    with _disable_current_modes(), torch.inference_mode(False), FakeTensorMode():
        samples = tuple(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device) for t in examples)
        cell = ExportedCell.from_export(
            torch.export.export(TensorCellModule(partial(projection_read, operator=operator)), samples, strict=False),
            inputs=names,
            outputs=("residual",),
        )
    declaration = cell.declaration
    arguments = tuple(Value(name, kind) for name, kind in zip(names, declaration.inputs, strict=True))
    output = Value("residual", declaration.outputs[0])
    program = Program(
        Region(Block(arguments, (Call(names, (output,), declaration.name),), ("residual",))), (declaration,)
    )
    return prepare_region_execution(
        program,
        {declaration.name: cell},
        payload_limits=(RegionPayloadLimit(str(examples[0].device), budget),),
        options=policy,
        compiler=compiler,
    )

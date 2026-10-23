"""Prepare a pinned head read as a pure shared region for model projection adapters."""

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import _disable_current_modes  # pyright: ignore[reportPrivateUsage]

from pllm.contracts.region_program import Block, Call, Program, Region, Value
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelectionPolicy
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_execution import RegionInstructionExecution
from pllm.execution.region_optimization import prepare_region_execution


def head_read(
    hidden: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    owners: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Read pinned matrices through request indices; inactive packed rows contribute no output."""
    requests, count = rows.shape
    heads, width, _ = pool.shape[1:]
    weights = pool.index_select(0, slots.clamp_min(0)).masked_fill((slots < 0)[:, None, None, None], 0)
    inputs = hidden.index_select(0, rows.flatten()).float().view(requests, count, heads, width)
    inputs = inputs.transpose(1, 2).reshape(requests * heads, count, width)
    result = torch.bmm(inputs, weights.reshape(requests * heads, width, width).transpose(1, 2))
    result = result.view(requests, heads, count, width).transpose(1, 2).reshape(requests * count, heads * width)
    return (result.index_select(0, owners.clamp_min(0)).masked_fill((owners < 0)[:, None], 0),)


def prepare_head_read(examples: tuple[torch.Tensor, ...], budget: int) -> RegionInstructionExecution:
    """Export and select a readonly-input region before model capture; runtime owns every tensor."""
    names = ("hidden", "pool", "slots", "rows", "owners")
    with _disable_current_modes(), torch.inference_mode(False), FakeTensorMode():
        samples = tuple(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device) for t in examples)
        cell = ExportedCell.capture(head_read, names, ("residual",), samples)
    declaration = cell.declaration
    arguments = tuple(Value(name, kind) for name, kind in zip(names, declaration.inputs, strict=True))
    output = Value("residual", declaration.outputs[0])
    program = Program(
        Region(Block(arguments, (Call(names, (output,), declaration.name),), ("residual",))), (declaration,)
    )
    execution = prepare_region_execution(
        program,
        {declaration.name: cell},
        payload_limits=(RegionPayloadLimit(str(examples[0].device), budget),),
        options=RegionSelectionPolicy(),
    )
    if not execution.borrows_inputs:
        raise ValueError("Graph projection needs a readonly compiled kernel; use eager for this shape")
    return execution

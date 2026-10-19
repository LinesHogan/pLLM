"""Invocation temporaries are released at their last use."""

import weakref
from dataclasses import replace

import pytest
import torch

from pllm.contracts.region_program import Block, Call, If, Program, Region, ScalarType, Value
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("nested", [False, True])
def test_invocation_temporaries_do_not_outlive_native_last_use(device, nested):
    def create(x):
        return (x + 1,)

    cell = ExportedCell.capture(create, ("x",), ("out",), (torch.ones(4, device=device),))
    (tensor,) = cell.declaration.inputs
    program = Program(
        Region(
            Block(
                (Value("x", tensor),),
                (
                    Call(("x",), (Value("unused", tensor),), cell.digest),
                    Call(("x",), (Value("result", tensor),), cell.digest),
                ),
                ("result",),
            )
        ),
        (cell.declaration,),
    )
    if nested:
        body = program.entry
        program = replace(
            program,
            entry=Region(
                Block(
                    (Value("condition", ScalarType.BOOL), Value("x", tensor)),
                    (If(("condition", "x"), (Value("out", tensor),), body, body),),
                    ("out",),
                )
            ),
        )
    source = analyze_region_program(program)
    seen = []

    def observe(*args):
        if seen:
            assert seen[-1]() is None, "previous output survives into the next tensor call"
        result = cell.function(*args)
        seen.append(weakref.ref(result[0]))
        return result

    execution = RegionExecution(
        program,
        lower_region_schedule(program, source.reference_schedule),
        lower_region_bindings(program, source.reference_schedule),
        {cell.digest: replace(cell, function=observe)},
    )
    supplied = {"x": torch.ones(4, device=device)}
    if nested:
        supplied["condition"] = True
    result = execution.run(supplied)
    assert len(seen) == 2
    assert seen[0]() is None
    assert seen[1]() is result.values[0]

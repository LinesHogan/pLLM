"""Storage origins survive views, banks, loop carry and legal rescheduling."""

from dataclasses import replace
import weakref

import pytest
import torch

from pllm.contracts.region_program import Block, Call, If, Program, Region, ScalarType, SizeParameter, Value
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_frame import lookup
from pllm.execution.region_storage import analyze_region_storage, validate_region_storage
from pllm.execution.tensor_scan import TensorScan
from pllm.execution.codec import encode_boundary
from test.boundary.test_region_tensor import tensor_grid


def table(tables, value):
    return next((t for t in tables if t.input_id == value), None)


def test_call_result_aliases_keep_large_allocation_until_its_last_reader():
    def split(x):
        allocation = x + 1
        return allocation[:1], allocation[1:2]

    def consume(x):
        return (x * 2,)

    split_cell = ExportedCell.capture(split, ("x",), ("a", "b"), (torch.ones(64, 64),))
    use = ExportedCell.capture(consume, ("x",), ("out",), (torch.ones(1, 64),))
    full, = split_cell.declaration.inputs
    small, = use.declaration.inputs
    program = Program(Region(Block((Value("x", full),), (
        Call(("x",), (Value("a", small), Value("b", small)), split_cell.digest),
        Call(("a",), (Value("first", small),), use.digest),
        Call(("b",), (Value("second", small),), use.digest),
    ), ("second",))), (split_cell.declaration, use.declaration))
    source = analyze_region_program(program)
    cells = {split_cell.digest: split_cell, use.digest: use}
    storage = analyze_region_storage(program, source.reference_schedule, cells)
    a, b = source.statements[0].outputs
    assert lookup(table(storage.routes, a), ()) == (a, ())
    assert lookup(table(storage.routes, b), ()) == (a, ())
    assert next(r.bytes for r in storage.roots if r.value_id == a) == 64 * 64 * 4
    assert b not in {r.value_id for r in storage.roots}
    bindings = lower_region_bindings(program, source.reference_schedule)
    expected_last = lookup(table(bindings.times, 2), ())
    assert lookup(table(storage.last_uses, a), ()) == expected_last
    assert lookup(table(storage.retained, a), ()) is None
    validate_region_storage(program, cells, storage)


def test_parametric_view_carry_has_one_storage_origin_even_at_a_million_steps():
    def identity(x):
        return (x.view(x.shape),)

    cell = ExportedCell.capture(identity, ("x",), ("out",), (torch.ones(4),))
    scan = TensorScan(cell, SizeParameter("steps", 0, 1_000_000), carry=1, invariants=0)
    source = analyze_region_program(scan.program)
    storage = analyze_region_storage(scan.program, source.reference_schedule, {cell.digest: cell})
    output, = source.statements[0].outputs
    origin, = source.arguments
    assert {r.value_id for r in storage.roots} == {origin}
    assert not storage.births
    for count in (0, 1, 19, 1_000_000):
        assert lookup(table(storage.retained, origin), (), {"steps": count}) == (origin, ())
        for iteration in (0, count - 1):
            if 0 <= iteration < count:
                assert lookup(table(storage.routes, output), (iteration,), {"steps": count}) == (origin, ())
        assert lookup(table(storage.routes, output), (count,), {"steps": count}) is None
    assert len(encode_boundary(storage)) < 10_000


@pytest.mark.parametrize("chunks,layers", [(0, 3), (3, 0), (1, 1), (3, 4)])
def test_grid_storages_follow_exact_instance_sources_under_wavefront(chunks, layers):
    def step(h, w):
        return h + w, w + 1

    cell = ExportedCell.capture(step, ("h", "w"), ("nh", "nw"), (torch.ones(2), torch.ones(2)))
    program = tensor_grid(chunks, layers, cell)
    source = analyze_region_program(program)
    for candidate in source.candidates:
        storage = analyze_region_storage(program, candidate.schedule_map, {cell.digest: cell})
        bindings = lower_region_bindings(program, candidate.schedule_map)
        for value in source.values:
            if value.name != "old":
                continue
            for c in range(chunks):
                for l in range(layers):
                    assert lookup(table(storage.routes, value.id), (c, l)) == lookup(table(bindings.sources, value.id), (c, l))
        for output in source.statements[0].outputs:
            for c in range(chunks):
                for l in range(layers):
                    assert lookup(table(storage.births, output), (c, l)) == lookup(table(bindings.times, 0), (c, l))


def test_storage_witness_rejects_changed_origins_bytes_and_retention():
    def step(x):
        return (x + 1,)

    cell = ExportedCell.capture(step, ("x",), ("out",), (torch.ones(4),))
    scan = TensorScan(cell, 4, carry=1, invariants=0)
    source = analyze_region_program(scan.program)
    cells = {cell.digest: cell}
    storage = analyze_region_storage(scan.program, source.reference_schedule, cells)
    for forged in (
        replace(storage, roots=(replace(storage.roots[0], bytes=1), *storage.roots[1:])),
        replace(storage, routes=()), replace(storage, retained=()),
        replace(storage, last_uses=()), replace(storage, births=()),
        replace(storage, imports_digest="a" * 64), replace(storage, signature="b" * 64),
    ):
        with pytest.raises(ValueError, match="storage witness"):
            validate_region_storage(scan.program, cells, forged)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("nested", [False, True])
def test_invocation_temporaries_do_not_outlive_native_last_use(device, nested):
    def create(x):
        return (x + 1,)

    cell = ExportedCell.capture(create, ("x",), ("out",), (torch.ones(4, device=device),))
    tensor, = cell.declaration.inputs
    program = Program(Region(Block((Value("x", tensor),), (
        Call(("x",), (Value("unused", tensor),), cell.digest),
        Call(("x",), (Value("result", tensor),), cell.digest),
    ), ("result",))), (cell.declaration,))
    if nested:
        body = program.entry
        program = replace(program, entry=Region(Block((Value("condition", ScalarType.BOOL), Value("x", tensor)), (
            If(("condition", "x"), (Value("out", tensor),), body, body),
        ), ("out",))))
    source = analyze_region_program(program)
    seen = []

    def observe(*args):
        if seen:
            assert seen[-1]() is None, "previous output survives into the next tensor call"
        result = cell.function(*args)
        seen.append(weakref.ref(result[0]))
        return result

    execution = RegionExecution(program, lower_region_schedule(program, source.reference_schedule),
        lower_region_bindings(program, source.reference_schedule), {cell.digest: replace(cell, function=observe)})
    supplied = {"x": torch.ones(4, device=device)}
    if nested:
        supplied["condition"] = True
    result = execution.run(supplied)
    assert len(seen) == 2
    assert seen[0]() is None
    assert seen[1]() is result.values[0]

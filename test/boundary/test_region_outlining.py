"""Exact source extraction keeps tensor graphs, versions, guards and aliases."""

from dataclasses import replace

import pytest
import torch

from pllm import _core
from pllm.contracts.region_outlining import CallOutline, OutlinedProgram, call_outline_sites
from pllm.contracts.region_program import (
    BankType, Block, Call, Constant, Function, If, Map, Program, Region, Resource,
    ScalarType, Scan, TensorType, Value, While,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_outlining import bind_outlined_region, outline_region_calls, outline_region_program
from pllm.execution.region_storage import analyze_region_capacity, validate_region_capacity
from pllm.execution.tensor_rewrites import explore_matmul_reassociations
from test.boundary.test_tensor_access import StorageObserver


def chain(device="cpu", *, escape=False):
    tensors = tuple(torch.ones(s, dtype=torch.float64, device=device) for s in ((4, 3), (3, 128), (128, 2)))
    first = ExportedCell.capture(lambda a, b: (a @ b,), ("a", "b"), ("ab",), tensors[:2])
    second = ExportedCell.capture(lambda ab, c: (ab @ c,), ("ab", "c"), ("out",), (tensors[0] @ tensors[1], tensors[2]))
    a, b = first.declaration.inputs
    c = second.declaration.inputs[1]
    ab, = first.declaration.outputs
    out, = second.declaration.outputs
    program = Program(Region(Block((Value("a", a), Value("b", b), Value("c", c)), (
        Call(("a", "b"), (Value("ab", ab),), first.digest),
        Call(("ab", "c"), (Value("out", out),), second.digest),
    ), ("out", "ab") if escape else ("out",))), (first.declaration, second.declaration))
    return program, {first.digest: first, second.digest: second}, dict(zip(("a", "b", "c"), tensors))


def execution(program, cells):
    schedule = analyze_region_program(program).reference_schedule
    return RegionExecution(program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule), cells)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("escape", (False, True))
def test_exact_composition_preserves_escaping_versions_and_enables_cross_call_math(device, escape):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    program, cells, inputs = chain(device, escape=escape)
    source_graphs = tuple(str(c.graph.graph) for c in cells.values())
    site = CallOutline((), 0, 2)
    result = outline_region_calls(program, cells, site)
    native = decode_boundary(_core.outline_region_program(encode_boundary(program), encode_boundary(site)), OutlinedProgram)
    assert native == result.outlined
    assert result.cell.transformations == ()
    assert result.outlined.body.yields == (("ab", "out") if escape else ("out",))
    assert all(not explore_matmul_reassociations(c) for c in cells.values())
    assert len(explore_matmul_reassociations(result.cell)) == (0 if escape else 1)
    assert not any(n.op == "call_module" for n in result.cell.graph.graph.nodes)
    assert len([n for n in result.cell.graph.graph.nodes if n.op == "call_function"]) == 2
    originals = execution(program, cells).run(inputs)
    combined = execution(result.outlined.program, {**cells, result.cell.digest: result.cell}).run(inputs)
    assert originals.invocations == 2 and combined.invocations == 1
    for before, after in zip(originals.values, combined.values, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(combined.values[0], torch.full_like(combined.values[0], 384), rtol=0, atol=0)
    if escape:
        torch.testing.assert_close(combined.values[1], torch.full_like(combined.values[1], 3), rtol=0, atol=0)
    assert source_graphs == tuple(str(c.graph.graph) for c in cells.values())
    bound = analyze_region_capacity(program, analyze_region_program(program).reference_schedule, cells)
    with pytest.raises(ValueError):
        validate_region_capacity(result.outlined.program, {**cells, result.cell.digest: result.cell}, bound)


def test_both_structural_verifiers_reject_invalid_sites_effects_and_non_tensor_calls():
    program, cells, _ = chain()
    for site in (CallOutline((), 0, 1), CallOutline((), -1, 2), CallOutline((), True, 2),
                 CallOutline((0,), 0, 2), CallOutline((100, 0), 0, 2), CallOutline((0, 1), 0, 2),
                 CallOutline((), 1, 2), CallOutline((), 2**32, 2)):
        with pytest.raises(ValueError):
            outline_region_program(program, site)
        with pytest.raises(ValueError):
            _core.outline_region_program(encode_boundary(program), encode_boundary(site))
    scalar = Function("scalar", (ScalarType.INT,), (ScalarType.INT,))
    invalid = Program(Region(Block((Value("x", ScalarType.INT),), (
        Call(("x",), (Value("a", ScalarType.INT),), scalar.name),
        Call(("a",), (Value("b", ScalarType.INT),), scalar.name),
    ), ("b",))), (scalar,))
    with pytest.raises(ValueError, match="all-tensor"):
        outline_region_program(invalid, CallOutline((), 0, 2))
    assert tuple(call_outline_sites(invalid)) == ()
    tensor = TensorType((4,), "torch.float32")
    effect = Function("effect", (tensor,), (tensor,), ("state",))

    effectful = Program(Region(Block((Value("x", tensor),), (
        Call(("x",), (Value("a", tensor),), effect.name),
        Call(("a",), (Value("b", tensor),), effect.name),
    ), ("b",))), (effect,), (Resource("state", (tensor,)),))
    with pytest.raises(ValueError, match="all-tensor"):
        outline_region_program(effectful, CallOutline((), 0, 2))
    with pytest.raises(ValueError, match="all-tensor"):
        _core.outline_region_program(encode_boundary(effectful), encode_boundary(CallOutline((), 0, 2)))
    # The source range is rederived. Supplied frontier/result/body edits are
    # not accepted merely because the transformed Program itself is valid.
    site = CallOutline((), 0, 2)
    result = outline_region_calls(program, cells, site)
    for altered in (replace(result.outlined, body=replace(result.outlined.body, yields=())),
                    replace(result.outlined, function=replace(result.outlined.function, name="forged")),
                    replace(result.outlined, program=program)):
        with pytest.raises(ValueError, match="original calls"):
            _core.validate_region_outlining(encode_boundary(program), encode_boundary(site), encode_boundary(altered))


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_normalized_alias_inputs_read_old_versions_and_keep_authoring_guard(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.ones(4, dtype=torch.float64, device=device)
    view = ExportedCell.capture(lambda x: (x.view(x.shape),), ("x",), ("view",), (x,))
    def update(x, y):
        x.add_(1)
        return x, y
    update_cell = ExportedCell.capture(update, ("x", "y"), ("new", "old"), (x, x.clone()))
    with pytest.raises(ValueError, match="distinct storage"):
        update_cell(x, x.view(x.shape))
    t, = view.declaration.inputs
    program = Program(Region(Block((Value("x", t),), (
        Call(("x",), (Value("v", t),), view.digest),
        Call(("x", "v"), (Value("new", t), Value("old", t)), update_cell.digest),
    ), ("new", "old", "x"))), (view.declaration, update_cell.declaration))
    cells = {view.digest: view, update_cell.digest: update_cell}
    result = outline_region_calls(program, cells, CallOutline((), 0, 2))
    for p, imports in ((program, cells), (result.outlined.program, {**cells, result.cell.digest: result.cell})):
        new, old, entry = execution(p, imports).run({"x": x}).values
        torch.testing.assert_close(new, torch.full_like(new, 2), rtol=0, atol=0)
        torch.testing.assert_close(old, x, rtol=0, atol=0)
        assert old.untyped_storage().data_ptr() == entry.untyped_storage().data_ptr()
        assert new.untyped_storage().data_ptr() != old.untyped_storage().data_ptr()
        capacity = analyze_region_capacity(p, analyze_region_program(p).reference_schedule, imports)
        assert not capacity.unknown_functions
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)


def test_composition_rechecks_internal_layout_guards_without_executing_callables():
    x = torch.ones(4, 4)
    transpose = ExportedCell.capture(lambda x: (x.T,), ("x",), ("t",), (x,))
    contiguous = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("out",), (x,))
    t = TensorType((4, 4), "torch.float32")
    program = Program(Region(Block((Value("x", t),), (
        Call(("x",), (Value("t", t),), transpose.digest),
        Call(("t",), (Value("out", t),), contiguous.digest),
    ), ("out",))), (transpose.declaration, contiguous.declaration))
    cells = {transpose.digest: transpose, contiguous.digest: contiguous}
    with pytest.raises(ValueError, match="guard"):
        execution(program, cells).run({"x": x})
    with pytest.raises(ValueError, match="original call layout guard"):
        outline_region_calls(program, cells, CallOutline((), 0, 2))
    program, cells, inputs = chain()
    def forbidden(*args):
        raise AssertionError("composition must not execute user mathematics")
    cells = {name: replace(cell, function=forbidden) for name, cell in cells.items()}
    outlined = outline_region_calls(program, cells, CallOutline((), 0, 2))
    proposed = replace(outlined, cell=replace(outlined.cell, function=forbidden))
    rebuilt = bind_outlined_region(program, cells, proposed)
    actual, = rebuilt.cell.invoke(tuple(inputs.values()))
    torch.testing.assert_close(actual, torch.full_like(actual, 384), rtol=0, atol=0)


@pytest.mark.parametrize("kind,child", (("scan", 0), ("map", 0), ("if", 0), ("if", 1), ("while", 0), ("while", 1)))
def test_outlining_stays_inside_nested_control_and_preserves_unused_computation(kind, child):
    x = torch.ones(4)
    first = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("a",), (x,))
    second = ExportedCell.capture(lambda x: (x * 2,), ("x",), ("b",), (x,))
    t, = first.declaration.inputs
    arguments = (Value("x", t),)
    calls = (Call(("x",), (Value("a", t),), first.digest), Call(("a",), (Value("b", t),), second.digest))
    plain = Region(Block(arguments, calls, ("b",)))
    body = Region(replace(plain.block, arguments=(Value("i", ScalarType.INDEX), *arguments)))
    entry_args, supplied = arguments, {"x": x}
    if kind == "scan":
        op = Scan(("x",), (Value("out", t),), 2, body)
        expected = (torch.full_like(x, 10),)
    elif kind == "map":
        op = Map(("x",), (Value("out", BankType(t, 2)),), 2, body)
        expected = (torch.full_like(x, 4),) * 2
    elif kind == "if":
        entry_args = (Value("test", ScalarType.BOOL), *arguments)
        supplied = {"test": child == 0, "x": x}
        op = If(("test", "x"), (Value("out", t),), plain, plain)
        expected = (torch.full_like(x, 4),)
    else:
        # The condition's tensor results are unused, but its operations must
        # remain in the composed graph; outlining does not speculate or DCE.
        condition = Region(Block(arguments, (*calls, Constant((), (Value("test", ScalarType.BOOL),), True)), ("test",)))
        op = While(("x",), (Value("out", t),), 2, condition, plain)
        expected = (torch.full_like(x, 10),)
    program = Program(Region(Block(entry_args, (op,), ("out",))), (first.declaration, second.declaration))
    sources = {first.digest: first, second.digest: second}
    site = CallOutline((0, child), 0, 2)
    assert site in tuple(call_outline_sites(program))
    combined = outline_region_calls(program, sources, site)
    if kind == "while" and child == 0:
        assert combined.cell.outputs == ()
        assert sum(n.op == "call_function" for n in combined.cell.graph.graph.nodes) == 2
    for p, cells in ((program, sources), (combined.outlined.program, {**sources, combined.cell.digest: combined.cell})):
        result, = execution(p, cells).run(supplied).values
        values = result.elements if kind == "map" else (result,)
        for value, oracle in zip(values, expected, strict=True):
            torch.testing.assert_close(value, oracle, rtol=0, atol=0)


def test_outlining_may_hold_frontier_inputs_longer_and_needs_a_new_storage_analysis():
    x, y = torch.ones(4096), torch.ones(4096)
    first = ExportedCell.capture(lambda x: (x.sum(),), ("x",), ("sum",), (x,))
    second = ExportedCell.capture(lambda s, y: (y + s,), ("sum", "y"), ("out",), (x.sum(), y))
    t, = first.declaration.inputs
    scalar, = first.declaration.outputs
    program = Program(Region(Block((Value("x", t), Value("y", t)), (
        Call(("x",), (Value("sum", scalar),), first.digest),
        Call(("sum", "y"), (Value("out", t),), second.digest),
    ), ("out",))), (first.declaration, second.declaration))
    sources = {first.digest: first, second.digest: second}
    combined = outline_region_calls(program, sources, CallOutline((), 0, 2))
    peaks = []
    for p, cells in ((program, sources), (combined.outlined.program, {**sources, combined.cell.digest: combined.cell})):
        bound = analyze_region_capacity(p, analyze_region_program(p).reference_schedule, cells)
        run = execution(p, cells)
        with StorageObserver((x, y)) as observer:
            result = run.run({"x": x, "y": y})
        peak = observer.peak - x.numel() * x.element_size() - y.numel() * y.element_size()
        assert peak <= bound.devices[0].peak_tensor_bytes_upper_bound
        torch.testing.assert_close(result.values[0], torch.full_like(y, 4097), rtol=0, atol=0)
        peaks.append(peak)
    # Exact outlining is a candidate, not a promise to reduce peak storage.
    assert peaks[1] > peaks[0]

"""Call outlining: exact extraction of adjacent tensor calls into one cell.

``outline_region_calls`` composes calls without executing user math, keeps
escaping versions and nested control intact, and refuses calls it cannot prove
equivalent. ``optimize_region`` offers outlined programs as ordinary candidates
that still obey payload limits and the numerical-transform policy.
"""

from dataclasses import replace

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.region_outlining import CallOutline, call_outline_sites
from pllm.contracts.region_program import (
    BankType,
    Call,
    Constant,
    Function,
    If,
    Map,
    Program,
    Region,
    Resource,
    ScalarType,
    Scan,
    SizeParameter,
    TensorType,
    Value,
    While,
)
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_optimization import optimize_region
from pllm.execution.region_outlining import outline_region_calls
from test.region_fixtures import bind, reassociation_candidates, region
from test.state_fixtures import StorageObserver


def chain(device="cpu", *, escape=False):
    """(a @ b) @ c split across two calls; the intermediate may escape."""
    tensors = tuple(torch.ones(s, dtype=torch.float64, device=device) for s in ((4, 3), (3, 128), (128, 2)))
    first = ExportedCell.capture(lambda a, b: (a @ b,), ("a", "b"), ("ab",), tensors[:2])
    second = ExportedCell.capture(lambda ab, c: (ab @ c,), ("ab", "c"), ("out",), (tensors[0] @ tensors[1], tensors[2]))
    a, b = first.declaration.inputs
    c = second.declaration.inputs[1]
    program = Program(
        region(
            (Value("a", a), Value("b", b), Value("c", c)),
            (
                Call(("a", "b"), (Value("ab", first.declaration.outputs[0]),), first.digest),
                Call(("ab", "c"), (Value("out", second.declaration.outputs[0]),), second.digest),
            ),
            ("out", "ab") if escape else ("out",),
        ),
        (first.declaration, second.declaration),
    )
    return program, {first.digest: first, second.digest: second}, dict(zip(("a", "b", "c"), tensors))


def serial(program, cells):
    return bind(program, cells, analyze_region_program(program).reference_schedule)


@pytest.mark.parametrize("device,escape", (("cpu", True), ("cuda:0", False)))
def test_outlining_preserves_escaping_versions_and_exposes_cross_call_math(device, escape):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    program, cells, inputs = chain(device, escape=escape)
    result = outline_region_calls(program, cells, CallOutline((), 0, 2))
    assert result.outlined.body.yields == (("ab", "out") if escape else ("out",))
    # Only the composed cell exposes a reassociation; an escaping intermediate blocks it.
    assert all(not reassociation_candidates(c) for c in cells.values())
    assert len(reassociation_candidates(result.cell)) == (0 if escape else 1)
    before = serial(program, cells).run(inputs)
    after = serial(result.outlined.program, {**cells, result.cell.digest: result.cell}).run(inputs)
    assert before.invocations == 2 and after.invocations == 1
    for a, b in zip(before.values, after.values, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(after.values[0], torch.full_like(after.values[0], 384), rtol=0, atol=0)


def test_outlining_stays_inside_nested_control_and_keeps_unused_computation():
    x = torch.ones(4)
    first = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("a",), (x,))
    second = ExportedCell.capture(lambda x: (x * 2,), ("x",), ("b",), (x,))
    (t,) = first.declaration.inputs
    arguments = (Value("x", t),)
    calls = (Call(("x",), (Value("a", t),), first.digest), Call(("a",), (Value("b", t),), second.digest))
    plain = region(arguments, calls, ("b",))
    body = Region(replace(plain.block, arguments=(Value("i", ScalarType.INDEX), *arguments)))
    # The while condition's tensor results are unused, but outlining neither speculates nor drops them.
    condition = region(arguments, (*calls, Constant((), (Value("test", ScalarType.BOOL),), True)), ("test",))
    test_args = (Value("test", ScalarType.BOOL), *arguments)
    sources = {first.digest: first, second.digest: second}
    for op, child, entry_args, supplied, expected in (
        (Map(("x",), (Value("out", BankType(t, 2)),), 2, body), 0, arguments, {"x": x}, (4, 4)),
        (If(("test", "x"), (Value("out", t),), plain, plain), 1, test_args, {"test": False, "x": x}, (4,)),
        (While(("x",), (Value("out", t),), 2, condition, plain), 0, arguments, {"x": x}, (10,)),
    ):
        program = Program(region(entry_args, (op,), ("out",)), (first.declaration, second.declaration))
        site = CallOutline((0, child), 0, 2)
        assert site in tuple(call_outline_sites(program))
        combined = outline_region_calls(program, sources, site)
        for p, cells in (
            (program, sources),
            (combined.outlined.program, {**sources, combined.cell.digest: combined.cell}),
        ):
            (result,) = serial(p, cells).run(supplied).values
            values = result.elements if isinstance(op, Map) else (result,)
            for value, oracle in zip(values, expected, strict=True):
                torch.testing.assert_close(value, torch.full_like(x, oracle), rtol=0, atol=0)


def test_outlining_refuses_scalar_effectful_and_layout_guarded_calls():
    scalar = Function("scalar", (ScalarType.INT,), (ScalarType.INT,))
    scalars = Program(
        region(
            (Value("x", ScalarType.INT),),
            (
                Call(("x",), (Value("a", ScalarType.INT),), scalar.name),
                Call(("a",), (Value("b", ScalarType.INT),), scalar.name),
            ),
            ("b",),
        ),
        (scalar,),
    )
    tensor = TensorType((4,), "torch.float32")
    effect = Function("effect", (tensor,), (tensor,), ("state",))
    effectful = Program(
        region(
            (Value("x", tensor),),
            (Call(("x",), (Value("a", tensor),), effect.name), Call(("a",), (Value("b", tensor),), effect.name)),
            ("b",),
        ),
        (effect,),
        (Resource("state", (tensor,)),),
    )
    for program in (scalars, effectful):
        assert tuple(call_outline_sites(program)) == ()
    x = torch.ones(4, 4)
    transpose = ExportedCell.capture(lambda x: (x.T,), ("x",), ("t",), (x,))
    increment = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("out",), (x,))
    t = TensorType((4, 4), "torch.float32")
    guarded = Program(
        region(
            (Value("x", t),),
            (Call(("x",), (Value("t", t),), transpose.digest), Call(("t",), (Value("out", t),), increment.digest)),
            ("out",),
        ),
        (transpose.declaration, increment.declaration),
    )
    cells = {transpose.digest: transpose, increment.digest: increment}
    with pytest.raises(ValueError):
        serial(guarded, cells).run({"x": x})
    with pytest.raises(ValueError):
        outline_region_calls(guarded, cells, CallOutline((), 0, 2))


def test_outlined_alias_inputs_still_read_old_versions():
    x = torch.ones(4, dtype=torch.float64)
    view = ExportedCell.capture(lambda x: (x.view(x.shape),), ("x",), ("view",), (x,))

    def update(x, y):
        x.add_(1)
        return x, y

    update_cell = ExportedCell.capture(update, ("x", "y"), ("new", "old"), (x, x.clone()))
    (t,) = view.declaration.inputs
    program = Program(
        region(
            (Value("x", t),),
            (
                Call(("x",), (Value("v", t),), view.digest),
                Call(("x", "v"), (Value("new", t), Value("old", t)), update_cell.digest),
            ),
            ("new", "old", "x"),
        ),
        (view.declaration, update_cell.declaration),
    )
    cells = {view.digest: view, update_cell.digest: update_cell}
    result = outline_region_calls(program, cells, CallOutline((), 0, 2))
    for p, imports in ((program, cells), (result.outlined.program, {**cells, result.cell.digest: result.cell})):
        new, old, entry = serial(p, imports).run({"x": x}).values
        torch.testing.assert_close(new, torch.full_like(new, 2), rtol=0, atol=0)
        torch.testing.assert_close(old, x, rtol=0, atol=0)
        assert old.untyped_storage().data_ptr() == entry.untyped_storage().data_ptr()
        assert new.untyped_storage().data_ptr() != old.untyped_storage().data_ptr()
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)


def test_automatic_outlining_exposes_the_only_math_candidate_that_fits():
    program, sources, inputs = chain()
    before = tuple(t.clone() for t in inputs.values())
    with pytest.warns(NumericalTransformWarning):
        selected = optimize_region(
            program,
            sources,
            inputs,
            payload_limits=(RegionPayloadLimit("cpu", 5328),),
            options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
            warmup_rounds=0,
            measurement_rounds=2,
        )
    # The payload limit leaves a single feasible candidate: the outlined reassociation.
    assert len(selected.request.candidates[selected.selection.selected].outlines) == 1
    with StorageObserver(tuple(inputs.values())) as observed:
        actual = selected.execution.run(inputs)
    assert actual.invocations == 1
    assert observed.peak - sum(t.numel() * t.element_size() for t in inputs.values()) <= 5328
    torch.testing.assert_close(actual.values[0], torch.full_like(actual.values[0], 384), rtol=0, atol=0)
    for saved, tensor in zip(before, inputs.values(), strict=True):
        torch.testing.assert_close(saved, tensor, rtol=0, atol=0)


def test_outlining_in_a_symbolic_scan_runs_the_whole_size_domain():
    a, b, c = (torch.ones(s, dtype=torch.float64) for s in ((2, 2), (2, 16), (16, 2)))
    first = ExportedCell.capture(lambda a, b: (a @ b,), ("a", "b"), ("ab",), (a, b))
    second = ExportedCell.capture(lambda ab, c: (ab @ c,), ("ab", "c"), ("out",), (a @ b, c))
    ta, tb = first.declaration.inputs
    tab, tc = second.declaration.inputs
    body = region(
        (Value("i", ScalarType.INDEX), Value("a", ta), Value("b", tb), Value("c", tc)),
        (Call(("a", "b"), (Value("ab", tab),), first.digest), Call(("ab", "c"), (Value("next_a", ta),), second.digest)),
        ("next_a", "b", "c"),
    )
    outputs = (Value("final", ta), Value("final_b", tb), Value("final_c", tc))
    program = Program(
        region(
            (Value("a", ta), Value("b", tb), Value("c", tc)),
            (Scan(("a", "b", "c"), outputs, "steps", body),),
            ("final",),
        ),
        (first.declaration, second.declaration),
        parameters=(SizeParameter("steps", 0, 1_000_000),),
    )
    # A zero-trip tuning example still yields a working selection.
    selected = optimize_region(
        program,
        {first.digest: first, second.digest: second},
        {"a": a, "b": b, "c": c},
        payload_limits=(RegionPayloadLimit("cpu", 10000),),
        options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
        sizes={"steps": 0},
        warmup_rounds=0,
        measurement_rounds=1,
    )
    for count in (0, 1, 3):
        result = selected.execution.run({"a": a, "b": b, "c": c}, sizes={"steps": count})
        assert result.invocations == count
        torch.testing.assert_close(result.values[0], torch.full_like(a, 32**count), rtol=0, atol=0)

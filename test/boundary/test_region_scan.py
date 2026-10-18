"""Scan results retain their iteration order and exact pre/post-update versions."""

from dataclasses import replace

import pytest
import torch

from pllm.contracts.region_program import (
    AffineIndex, BankType, Block, Call, Constant, Coordinate, If, InvalidProgram, Program, Publish,
    Region, Resource, ScalarType, Scan, SizeParameter, Snapshot, Value, verify,
)
from pllm.execution.codec import encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import (
    analyze_region_program, lower_region_bindings, lower_region_schedule, validate_region_program,
)
from pllm.execution.region_execution import RegionExecution
from test.region_reference import interpret
from pllm.execution.region_values import Bank
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.tensor_scan import TensorScan
from test.boundary.test_region_program import region
from test.boundary.test_tensor_access import StorageObserver


X, I, B = ScalarType.INDEX, ScalarType.INT, ScalarType.BOOL


@pytest.fixture(scope="module")
def cell():
    return ExportedCell.capture(lambda w, scale, x: (w * scale + x, w, w * scale + x),
        ("w", "scale", "x"), ("next", "old", "new"), tuple(torch.ones(3, dtype=torch.float64) for _ in range(3)))


@pytest.mark.parametrize("count", (0, 1, 5))
@pytest.mark.parametrize("dynamic", (False, True))
@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_tensor_scan_collects_exact_old_and_new_versions_under_every_schedule(cell, count, dynamic, device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        cell = ExportedCell.capture(lambda w, scale, x: (w * scale + x, w, w * scale + x),
            ("w", "scale", "x"), ("next", "old", "new"),
            tuple(torch.ones(3, dtype=torch.float64, device=device) for _ in range(3)))
    scan = TensorScan(cell, count, carry=1, invariants=1)
    w = torch.arange(3, dtype=torch.float64, device=device)
    scale = torch.full_like(w, 2.)
    sequence = tuple(torch.full_like(w, i + 1.) for i in range(count))
    inputs = {"arg0": w, "arg1": scale, "arg2": Bank(sequence)}
    program = scan.program
    if dynamic:
        block = program.entry.block
        results = tuple(Value(f"out{i}", type_) for i, type_ in enumerate(verify(program)))
        program = replace(program, entry=region((*block.arguments, Value("enabled", B)),
            (If(("enabled", *(v.name for v in block.arguments)), results, program.entry, program.entry),),
            tuple(v.name for v in results)))
        inputs["enabled"] = True
    expected = interpret(program, inputs, functions={cell.digest: lambda args: cell(*args)})
    old, new = [], []
    current = w
    for x in sequence:
        old.append(current)
        current = current * scale + x
        new.append(current)
    for candidate in analyze_region_program(program).candidates:
        execution = RegionExecution(program, lower_region_schedule(program, candidate.schedule_map),
            lower_region_bindings(program, candidate.schedule_map), {cell.digest: cell})
        capacity = analyze_region_capacity(program, candidate.schedule_map, {cell.digest: cell})
        originals = (w, scale, *sequence)
        with StorageObserver(originals) as storage:
            actual = execution.run(inputs).values
        # The observer sees caller-owned roots (including detach views), while
        # the region budget covers its new storage. Account for the same scope.
        original_bytes = sum(t.untyped_storage().nbytes() for t in originals)
        assert storage.peak - original_bytes <= capacity.devices[0].peak_tensor_bytes_upper_bound
        torch.testing.assert_close(actual[0], current, rtol=0, atol=0)
        for got, oracle, wanted in zip(actual[1:], expected.values[1:], (old, new), strict=True):
            assert len(got.elements) == count
            for a, b, c in zip(got.elements, oracle.elements, wanted, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(a, c, rtol=0, atol=0)
    result = scan((w,), (scale,), (sequence,))
    torch.testing.assert_close(result.carry[0], current, rtol=0, atol=0)
    assert tuple(len(column) for column in result.columns) == (count, count)
    torch.testing.assert_close(w, torch.arange(3, dtype=torch.float64, device=device), rtol=0, atol=0)


@pytest.mark.parametrize("invariants", (-1, True, 2))
def test_scan_rejects_invalid_invariant_counts(invariants):
    body = region((Value("i", X), Value("n", I)), (), ("n",))
    program = Program(region((Value("n", I),),
        (Scan(("n",), (Value("result", I),), 3, body, invariants=invariants),), ("result",)))
    with pytest.raises(ValueError):
        verify(program)


def test_tensor_region_body_keeps_lexical_names_and_can_be_optimized(monkeypatch):
    from pllm.config import ExecutionOptions
    from pllm.contracts.region_selection import RegionPayloadLimit
    import pllm.execution.region_optimization as tuning
    x = torch.ones(3, dtype=torch.float64)
    multiply = ExportedCell.capture(lambda w, s: (w * s,), ("w", "s"), ("out",), (x, x.clone()))
    addition = ExportedCell.capture(lambda w, x: (w + x,), ("w", "x"), ("out",), (x, x.clone()))
    t, = addition.declaration.outputs
    program = Program(Region(Block((Value("scan_index", t), Value("scale", t), Value("scan_sequence0", t)), (
        Call(("scan_index", "scale"), (Value("doubled", t),), multiply.digest),
        Call(("doubled", "scan_sequence0"), (Value("new", t),), addition.digest),
    ), ("new", "scan_index", "new"))), (multiply.declaration, addition.declaration))
    scan = TensorScan(program, SizeParameter("steps", 0, 5), carry=1, invariants=1,
        functions={multiply.digest: multiply, addition.digest: addition})
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args: len(execution._lowered.statements))
    selected, selected_report = scan.optimized((x,), (x * 2,), ((x, x),), payload_limits=(RegionPayloadLimit("cpu", 10000),),
        options=replace(ExecutionOptions(allow_numerical_transforms=False), tuning="measured"), warmup_rounds=0, measurement_rounds=1)
    assert selected_report.request.candidates[selected_report.selection.selected].outlines
    for count in (0, 4):
        rows = tuple(x * (i + 1) for i in range(count))
        expected_old, expected_new, current = [], [], x
        for row in rows:
            expected_old.append(current)
            current = current * 2 + row
            expected_new.append(current)
        result = selected((x,), (x * 2,), (rows,))
        torch.testing.assert_close(result.carry[0], current, rtol=0, atol=0)
        for actual, expected in zip(result.columns, (expected_old, expected_new), strict=True):
            assert len(actual) == count
            for value, want in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, want, rtol=0, atol=0)


@pytest.mark.parametrize("count", (0, 4))
@pytest.mark.parametrize("dynamic", (False, True))
def test_scan_without_carry_collects_indices(count, dynamic):
    body = region((Value("i", X),), (), ("i",))
    branch = region((), (Scan((), (Value("ys", BankType(X, count)),), count, body),), ("ys",))
    program = Program(region((Value("p", B),), (
        If(("p",), (Value("ys", BankType(X, count)),), branch, branch),
    ), ("ys",))) if dynamic else Program(branch)
    schedule = analyze_region_program(program).reference_schedule
    result = RegionExecution(program, lower_region_schedule(program, schedule),
        lower_region_bindings(program, schedule), {}).run({"p": True} if dynamic else {})
    assert result.values == (Bank(tuple(range(count))),)


@pytest.mark.parametrize("count", (0, 3))
@pytest.mark.parametrize("dynamic", (False, True))
def test_scan_collects_snapshots_through_original_page_owner(count, dynamic):
    from test.boundary.test_region_pages import owner_fixture
    execution, task, rows, refs = owner_fixture(rounds=1)
    cell, = task.region._cells.values()
    tensor = cell.declaration.inputs[0]
    args = (Value("marker", I),)
    body = region((Value("i", X), *args), (
        Coordinate((), (Value("zero", X),), AffineIndex(0)),
        Snapshot(("zero",), (Value("old", tensor),), "layers"),
        Call(("old",), (Value("new", tensor),), cell.digest),
        Publish(("new", "zero"), (), "layers"),
    ), ("marker", "old", "new"))
    outputs = (Value("end", I), Value("olds", BankType(tensor, count)), Value("news", BankType(tensor, count)))
    branch = region(args, (Scan(("marker",), outputs, count, body),), tuple(v.name for v in outputs))
    entry = region((*args, Value("p", B)), (
        If(("p", "marker"), outputs, branch, branch),
    ), tuple(v.name for v in outputs)) if dynamic else branch
    task.program = Program(entry, (cell.declaration,), (Resource("layers", (tensor,), (2,)),))
    schedule = analyze_region_program(task.program).candidates[-1].schedule_map
    task.region = RegionExecution(task.program, lower_region_schedule(task.program, schedule),
        lower_region_bindings(task.program, schedule), {cell.digest: cell})
    task.inputs = {"marker": 7, **({"p": True} if dynamic else {})}
    expected = interpret(task.program, task.inputs,
        resources={"layers": (Bank((torch.ones(tensor.shape), torch.full(tensor.shape, 2.))),)},
        functions={cell.digest: lambda args: cell(*args)})
    final = execution.step(rows)
    assert task.result.values[0] == 7
    for got, want in zip(task.result.values[1:], expected.values[1:], strict=True):
        assert len(got.elements) == count
        for a, b in zip(got.elements, want.elements, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert tuple(ref.version_id for ref in final) == (count, 0)
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0


def test_scan_collection_stays_symbolic_at_a_million_iterations(cell):
    sizes = []
    for count in (20, 1_000_000):
        scan = TensorScan(cell, count, carry=1, invariants=1)
        analysis = analyze_region_program(scan.program)
        assert len(analysis.statements) == 1
        bindings = lower_region_bindings(scan.program, analysis.reference_schedule)
        sizes.append(len(encode_boundary(bindings)))
    assert sizes[1] < 2 * sizes[0]


@pytest.mark.parametrize("invalid", ("carry", "extent", "collected_type"))
def test_python_and_rust_reject_invalid_scan_result_signatures(invalid):
    body = region((Value("i", X), Value("n", I)), (), ("n", "i"))
    outputs = (Value("n1", I), Value("ys", BankType(X, 3)))
    if invalid == "carry":
        body = replace(body, block=replace(body.block, yields=("i", "n")))
    elif invalid == "extent":
        outputs = (outputs[0], Value("ys", BankType(X, 4)))
    else:
        outputs = (outputs[0], Value("ys", BankType(I, 3)))
    program = Program(region((Value("n", I),), (
        Scan(("n",), outputs, 3, body),
    ), ("n1", "ys")))
    with pytest.raises(InvalidProgram):
        verify(program)
    with pytest.raises(ValueError):
        validate_region_program(program)

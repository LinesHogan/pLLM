"""Actual exported tensors follow compiler-derived versions under rescheduling."""

from dataclasses import replace

import pytest
import torch

from pllm.contracts.region_bindings import RegionBindings
from pllm.contracts.region_program import (
    AffineIndex, BankType, Call, Constant, Coordinate, IndexTerm, Map, Program, Scalar, ScalarKind, ScalarType,
    Scan, Select, TensorType, Update, Value,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import (
    analyze_region_program, lower_region_bindings, lower_region_schedule, validate_region_bindings,
)
from pllm.execution.region_frame import lookup
from pllm.execution.region_reference import interpret
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_values import Bank
from test.boundary.test_region_program import grid_program, region


X = ScalarType.INDEX


def tensor_grid(chunks, layers, cell):
    tensor = cell.declaration.inputs[0]
    weights, hidden = BankType(tensor, layers), BankType(tensor, chunks)
    layer = region((Value("l", X), Value("h", tensor), Value("w", weights)), (
        Select(("w", "l"), (Value("old", tensor),)),
        Call(("h", "old"), (Value("next_h", tensor), Value("next_w", tensor)), cell.digest),
        Update(("w", "l", "next_w"), (Value("bank", weights),)),
    ), ("next_h", "bank"))
    chunk = region((Value("c", X), Value("w", weights), Value("h", hidden)), (
        Select(("h", "c"), (Value("input", tensor),)),
        Scan(("input", "w"), (Value("output", tensor), Value("next_w", weights)), layers, layer),
        Update(("h", "c", "output"), (Value("next_h", hidden),)),
    ), ("next_w", "next_h"))
    return Program(region((Value("weights", weights), Value("hidden", hidden)), (
        Scan(("weights", "hidden"), (Value("w_final", weights), Value("h_final", hidden)), chunks, chunk),
    ), ("w_final", "h_final")), (cell.declaration,))


@pytest.fixture(scope="module")
def cell():
    def step(h, w):
        old = h.clone()
        h.mul_(0.5).add_(w * 0.25)
        w.add_(old * 0.0625)
        return h, w
    return ExportedCell.capture(step, ("h", "w"), ("next_h", "next_w"),
                                (torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)))


def inputs_for(chunks, layers, device="cpu"):
    return {"weights": Bank(tuple(torch.full((4,), l + 1., dtype=torch.float64, device=device) for l in range(layers))),
            "hidden": Bank(tuple(torch.full((4,), c + 2., dtype=torch.float64, device=device) for c in range(chunks)))}


def execute(program, schedule, cell, inputs):
    lowered = lower_region_schedule(program, schedule)
    bindings = lower_region_bindings(program, schedule)
    assert decode_boundary(encode_boundary(bindings), RegionBindings) == bindings
    return RegionExecution(program, lowered, bindings, {cell.digest: cell}).run(inputs)


@pytest.mark.parametrize("chunks,layers", [(0, 3), (3, 0), (1, 1), (2, 7), (5, 3), (6, 5)])
def test_exported_tensors_read_identical_versions_under_isl_schedules(cell, chunks, layers):
    program = tensor_grid(chunks, layers, cell)
    inputs = inputs_for(chunks, layers)
    expected_calls = []
    def oracle(args):
        expected_calls.append(tuple(x.clone() for x in args))
        return cell(*args)
    expected = interpret(program, inputs, functions={cell.digest: oracle})
    analysis = analyze_region_program(program)
    for candidate in analysis.candidates:
        actual_calls = []
        def observed(*args):
            actual_calls.append(tuple(x.clone() for x in args))
            return cell.function(*args)
        actual = execute(program, candidate.schedule_map, replace(cell, function=observed), inputs)
        from pllm.execution.region_schedule import RegionScheduleExecution
        invocations = tuple(RegionScheduleExecution(program, lower_region_schedule(program, candidate.schedule_map)).invocations())
        for invocation, args in zip(invocations, actual_calls, strict=True):
            c, l = invocation.coordinates
            for got, want in zip(args, expected_calls[c * layers + l], strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        for got, want in zip(actual.values, expected.values, strict=True):
            for a, b in zip(got.elements, want.elements, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert actual.invocations == chunks * layers
    for name, bank in inputs.items():
        for i, tensor in enumerate(bank.elements):
            torch.testing.assert_close(tensor, torch.full_like(tensor, i + (1. if name == "weights" else 2.)), rtol=0, atol=0)


def test_partial_tiles_execute_tensors_and_preserve_final_banks(cell):
    program = tensor_grid(5, 7, cell)
    inputs = inputs_for(5, 7)
    schedule = "{ S0[c,l] -> [floor(c/2),floor(l/3),c,l] : 0<=c<5 and 0<=l<7 }"
    result = execute(program, schedule, cell, inputs)
    expected = interpret(program, inputs, functions={cell.digest: lambda args: cell(*args)})
    for actual_bank, expected_bank in zip(result.values, expected.values, strict=True):
        for a, b in zip(actual_bank.elements, expected_bank.elements, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_last_use_releases_intermediates_instead_of_retaining_the_whole_grid(cell):
    chunks, layers = 80, 12
    program = tensor_grid(chunks, layers, cell)
    candidate = next(c for c in analyze_region_program(program).candidates if c.source == "isl-parallel")
    result = execute(program, candidate.schedule_map, cell, inputs_for(chunks, layers))
    # Input and returned hidden arrays necessarily scale with chunks; the
    # transient frontier does not retain two outputs for all chunks*layers.
    assert result.peak_live_values < 2 * chunks + 6 * layers
    assert result.released_values > chunks * layers


def test_symbolic_sources_route_old_and_new_versions_with_guards():
    program = grid_program(4, 3)
    analysis = analyze_region_program(program)
    bindings = lower_region_bindings(program, analysis.candidates[-1].schedule_map)
    source = {t.input_id: t for t in bindings.sources}
    old = next(v.id for v in bindings.values if v.name == "old")
    weight_input = bindings.arguments[0]
    new_weight = analysis.statements[0].outputs[1]
    for c in range(4):
        for l in range(3):
            assert lookup(source[old], (c, l)) == ((weight_input, (l,)) if c == 0 else (new_weight, (c - 1, l)))
    for point in [(-1, 0), (4, 0), (0, -1), (0, 3)]:
        assert lookup(source[old], point) is None


def test_large_binding_plan_remains_symbolic():
    sizes = []
    for chunks in (20, 1_000_000):
        program = grid_program(chunks, 16)
        candidate = analyze_region_program(program).candidates[-1]
        bindings = lower_region_bindings(program, candidate.schedule_map)
        sizes.append(len(encode_boundary(bindings)))
        assert len(bindings.values) == 18
    assert sizes[1] < sizes[0] * 2


def test_tampered_sources_and_lifetimes_rejected_before_execution(cell):
    program = tensor_grid(3, 3, cell)
    schedule = analyze_region_program(program).candidates[-1].schedule_map
    bindings = lower_region_bindings(program, schedule)
    for tampered in (replace(bindings, sources=()), replace(bindings, last_uses=()),
                     replace(bindings, retained=()), replace(bindings, results=()),
                     replace(bindings, signature="fake")):
        with pytest.raises(ValueError, match="original program"):
            validate_region_bindings(program, tampered)
    serial = lower_region_schedule(program, analyze_region_program(program).reference_schedule)
    with pytest.raises(ValueError, match="same schedule"):
        RegionExecution(program, serial, bindings, {cell.digest: cell})


def test_reverse_map_rematerializes_indices_and_constants():
    body = region((Value("i", X),), (
        Coordinate(("i",), (Value("twice", X),), AffineIndex(0, (IndexTerm(0, 2),))),
    ), ("twice",))
    program = Program(region((), (Map((), (Value("ys", BankType(X, 7)),), 7, body),), ("ys",)))
    analysis = analyze_region_program(program)
    # Everything is affine rematerialization; no statements need to run.
    schedule = analysis.reference_schedule
    result = RegionExecution(program, lower_region_schedule(program, schedule),
                                   lower_region_bindings(program, schedule), {}).run({})
    assert result.values == (Bank(tuple(2 * i for i in range(7))),)
    assert result.invocations == 0


def test_dynamic_bank_index_retains_old_bank_and_reports_bounds():
    tensor = TensorType((2,), "torch.float32", "cpu")
    bank = BankType(tensor, 2)
    program = Program(region((Value("b", bank), Value("i", X), Value("new", tensor)), (
        Update(("b", "i", "new"), (Value("updated", bank),)),
        Select(("b", "i"), (Value("old", tensor),)),
        Select(("updated", "i"), (Value("selected", tensor),)),
    ), ("b", "old", "selected")))
    schedule = analyze_region_program(program).candidates[-1].schedule_map
    execution = RegionExecution(program, lower_region_schedule(program, schedule),
                                      lower_region_bindings(program, schedule), {})
    supplied = {"b": Bank((torch.ones(2), torch.full((2,), 2.))), "i": 1, "new": torch.full((2,), 3.)}
    result = execution.run(supplied)
    torch.testing.assert_close(result.values[1], supplied["b"].elements[1], rtol=0, atol=0)
    torch.testing.assert_close(result.values[2], supplied["new"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="out of bounds"):
        execution.run({**supplied, "i": 2})
    # A failed invocation cannot leave values in the next frame.
    torch.testing.assert_close(execution.run(supplied).values[2], supplied["new"], rtol=0, atol=0)

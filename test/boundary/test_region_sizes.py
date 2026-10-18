"""One invocation binds bank lengths and loop extents to the same shape symbols."""

from dataclasses import replace
import json

import pytest

from pllm import _core
from pllm.contracts.region_program import (
    AffineIndex, BankType, Call, Constant, Coordinate, Function, If, IndexTerm, InvalidProgram, Map, Parameter, ParameterTerm, Program, Scalar, ScalarKind,
    ScalarType, Scan, Select, SizeConstraint, SizeParameter, Value, verify,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_compilation import analyze_region_program, validate_region_program
from test.region_reference import interpret
from pllm.execution.region_values import Bank
from test.boundary.test_region_program import region

X = ScalarType.INDEX
I = ScalarType.INT


def prefix_program():
    bank = BankType(I, "n")
    body = region((Value("i", X), Value("total", I), Value("xs", bank)), (
        Select(("xs", "i"), (Value("x", I),)),
        Scalar(("total", "x"), (Value("next", I),), ScalarKind.ADD),
    ), ("next", "xs", "total", "next"))
    return Program(region((Value("seed", I), Value("xs", bank)), (
        Scan(("seed", "xs"), (Value("total", I), Value("unused", bank),
            Value("before", bank), Value("after", bank)), "n", body),
    ), ("total", "before", "after")), parameters=(SizeParameter("n", 0, 8),))


@pytest.mark.parametrize("count", (0, 1, 3, 8))
def test_symbolic_scan_reference_preserves_prefix_versions_and_zero_trip(count):
    program = prefix_program()
    verify(program)
    validate_region_program(program)
    assert decode_boundary(encode_boundary(program), Program) == program
    xs = tuple(range(1, count + 1))
    result = interpret(program, {"seed": 11, "xs": Bank(xs)}, sizes={"n": count})
    before = tuple(11 + sum(xs[:i]) for i in range(count))
    after = tuple(11 + sum(xs[:i + 1]) for i in range(count))
    assert result.values == (11 + sum(xs), Bank(before), Bank(after))
    assert result.effects == ()


def test_nested_parameterized_map_has_one_shared_result_shape_contract():
    inside = region((Value("j", X),), (), ("j",))
    outside = region((Value("i", X),), (
        Map((), (Value("row", BankType(X, "width")),), "width", inside),
    ), ("row",))
    program = Program(region((), (Map((), (Value("matrix", BankType(BankType(X, "width"), "height")),),
        "height", outside),), ("matrix",)), parameters=(SizeParameter("width", 0, 4), SizeParameter("height", 0, 3)))
    verify(program)
    validate_region_program(program)
    for height, width in ((0, 3), (2, 0), (3, 4)):
        result = interpret(program, {}, sizes={"height": height, "width": width})
        assert result.values == (Bank((Bank(tuple(range(width))),) * height),)


@pytest.mark.parametrize("sizes", ({}, {"n": 2, "extra": 1}, {"n": -1}, {"n": 9}, {"n": True}, {"n": 2.0}))
def test_size_guards_run_before_any_imported_call(sizes):
    program = prefix_program()
    function = Function("observe", (), ())
    program = replace(program, functions=(function,), entry=replace(program.entry,
        block=replace(program.entry.block, operations=(Call((), (), function.name), *program.entry.block.operations))))
    calls = []
    with pytest.raises(ValueError, match="size parameter"):
        interpret(program, {"seed": 1, "xs": Bank((1, 2))}, sizes=sizes,
            functions={"observe": lambda args: calls.append(args) or ()})
    assert not calls


def test_input_bank_cannot_silently_supply_a_different_trip_count():
    program = prefix_program()
    with pytest.raises(ValueError, match="logical extent"):
        interpret(program, {"seed": 1, "xs": Bank((1,))}, sizes={"n": 2})
    with pytest.raises(ValueError, match="logical extent"):
        interpret(program, {"seed": 1, "xs": Bank((1,))}, sizes={"n": 0})


def test_result_extent_must_be_the_same_symbol_even_when_bounds_match():
    program = prefix_program()
    scan, = program.entry.block.operations
    outputs = (*scan.outputs[:2], Value("before", BankType(I, "m")), scan.outputs[3])
    program = replace(program, parameters=(*program.parameters, SizeParameter("m", 0, 8)),
        entry=region(program.entry.block.arguments, (replace(scan, outputs=outputs),), program.entry.block.yields))
    with pytest.raises(InvalidProgram):
        verify(program)
    with pytest.raises(ValueError):
        validate_region_program(program)


def test_closed_size_wire_rejects_boolean_extents_and_unknown_parameter_fields():
    wire = json.loads(encode_boundary(prefix_program()))
    wire["entry"]["block"]["operations"][0]["count"] = True
    with pytest.raises(ValueError):
        decode_boundary(json.dumps(wire), Program)
    with pytest.raises(ValueError):
        _core.validate_region_program(json.dumps(wire))
    wire = json.loads(encode_boundary(prefix_program()))
    wire["parameters"][0]["hint"] = "trust me"
    with pytest.raises(ValueError):
        decode_boundary(json.dumps(wire), Program)
    with pytest.raises(ValueError):
        _core.validate_region_program(json.dumps(wire))


def test_one_native_schedule_handles_all_prefix_lengths_without_recompiling(monkeypatch):
    from pllm.execution.region_compilation import lower_region_schedule, lower_region_bindings
    from pllm.execution.region_execution import RegionExecution
    program = prefix_program()
    analysis = analyze_region_program(program)
    assert analysis.exact_provenance and len(analysis.statements) == 1
    assert "__size_n" in analysis.domain and len(analysis.candidates) == 3
    runners = tuple(RegionExecution(program, lower_region_schedule(program, c.schedule_map),
        lower_region_bindings(program, c.schedule_map), {}) for c in analysis.candidates)
    def forbidden(*args):
        raise AssertionError("bound execution must not compile or revalidate")
    for name in ("analyze_region_program", "lower_region_schedule", "lower_region_bindings"):
        monkeypatch.setattr(_core, name, forbidden)
    for count in (8, 0, 3, 1, 8):
        inputs = {"seed": 11, "xs": Bank(tuple(range(1, count + 1)))}
        expected = interpret(program, inputs, sizes={"n": count})
        for runner in runners:
            actual = runner.run(inputs, sizes={"n": count})
            assert actual.values == expected.values and actual.invocations == count
    for invalid in ({"n": True}, {"n": -1}, {"n": 9}, {}, {"n": 2, "extra": 1}):
        with pytest.raises(ValueError, match="size parameter"):
            runners[0].run({"seed": 11, "xs": Bank(())}, sizes=invalid)


@pytest.mark.parametrize("count", (0, 4))
def test_parameterized_scan_preserves_effect_count_and_snapshot_versions(count):
    from pllm.contracts.region_program import Constant, Publish, Resource, Snapshot

    body = region((Value("i", X),), (
        Snapshot((), (Value("old", I),), "w"),
        Constant((), (Value("one", I),), 1),
        Scalar(("old", "one"), (Value("new", I),), ScalarKind.ADD),
        Publish(("new",), (), "w"),
    ), ("old", "new"))
    program = Program(region((), (
        Scan((), (Value("before", BankType(I, "n")),
            Value("after", BankType(I, "n"))), "n", body),
    ), ("before", "after")), resources=(Resource("w", (I,)),),
        parameters=(SizeParameter("n", 0, 4),))
    verify(program)
    validate_region_program(program)
    result = interpret(program, {}, resources={"w": (3,)}, sizes={"n": count})
    assert result.values == (Bank(tuple(range(3, 3 + count))), Bank(tuple(range(4, 4 + count))))
    assert result.resources == {"w": (3 + count,)}
    assert [(event.kind, event.sequence, event.coordinates) for event in result.effects] == [
        (kind, 2 * i + j + 1, (i,)) for i in range(count) for j, kind in enumerate(("snapshot", "publish"))]


def bind_reference_schedule(program):
    from pllm.execution.region_compilation import lower_region_bindings, lower_region_schedule
    from pllm.execution.region_execution import RegionExecution
    analysis = analyze_region_program(program, optimize=False)
    return analysis, RegionExecution(program, lower_region_schedule(program, analysis.reference_schedule),
        lower_region_bindings(program, analysis.reference_schedule), {})


def test_parameter_offsets_route_bank_elements_in_affine_and_dynamic_regions(monkeypatch):
    bank, output = BankType(I, 32), BankType(I, "n")
    body = region((Value("i", X), Value("xs", bank)), (
        Parameter((), (Value("prefix", X),), "prefix"),
        Coordinate(("prefix", "i"), (Value("position", X),), AffineIndex(0, (IndexTerm(0), IndexTerm(1)))),
        Coordinate(("position",), (Value("doubled", X),), AffineIndex(0, (IndexTerm(0, 2),))),
        Coordinate(("doubled", "prefix"), (Value("index", X),), AffineIndex(0, (IndexTerm(0), IndexTerm(1, -1)))),
        Select(("xs", "index"), (Value("selected", I),)),
    ), ("selected",))
    entry = region((Value("xs", bank),), (
        Map(("xs",), (Value("ys", output),), "n", body),
    ), ("ys",))
    program = Program(entry, parameters=(SizeParameter("prefix", 3, 10), SizeParameter("n", 0, 8)))
    assert decode_boundary(encode_boundary(program), Program) == program
    analysis, affine = bind_reference_schedule(program)
    assert not analysis.statements  # All accesses resolve through symbolic value routing.
    branched = replace(program, entry=region((Value("choose", ScalarType.BOOL), Value("xs", bank)), (
        If(("choose", "xs"), (Value("ys", output),), entry, entry),
    ), ("ys",)))
    _, dynamic = bind_reference_schedule(branched)
    def forbidden(*args):
        raise AssertionError("invocation sizes must not trigger compilation")
    for name in ("analyze_region_program", "lower_region_bindings", "lower_region_schedule"):
        monkeypatch.setattr(_core, name, forbidden)
    xs = Bank(tuple(100 + i for i in range(32)))
    for prefix, count in ((3, 0), (10, 8), (5, 3), (3, 1)):
        sizes = {"prefix": prefix, "n": count}
        expected = (Bank(tuple(100 + prefix + 2 * i for i in range(count))),)
        assert interpret(program, {"xs": xs}, sizes=sizes).values == expected
        assert affine.run({"xs": xs}, sizes=sizes).values == expected
        for choose in (False, True):
            assert dynamic.run({"xs": xs, "choose": choose}, sizes=sizes).values == expected


def test_parameter_products_and_possible_overflow_keep_checked_execution():
    body = region((Value("i", X),), (
        Parameter((), (Value("p", X),), "p"),
        Scalar(("p", "i"), (Value("scaled", X),), ScalarKind.MUL),
        Scalar(("p", "p"), (Value("square", X),), ScalarKind.MUL),
    ), ("scaled", "square"))
    program = Program(region((), (
        Map((), (Value("scaled", BankType(X, 3)), Value("squared", BankType(X, 3))), 3, body),
    ), ("scaled", "squared")), parameters=(SizeParameter("p", 2, 7),))
    analysis, runner = bind_reference_schedule(program)
    assert len(analysis.statements) == 2 and all(s.opaque for s in analysis.statements)
    for p in (2, 7):
        expected = (Bank((0, p, 2 * p)), Bank((p * p,) * 3))
        assert runner.run({}, sizes={"p": p}).values == expected
        assert interpret(program, {}, sizes={"p": p}).values == expected

    overflow = Program(region((), (
        Parameter((), (Value("p", X),), "p"),
        Coordinate((), (Value("one", X),), AffineIndex(1)),
        Scalar(("p", "one"), (Value("unused", X),), ScalarKind.ADD),
    ), ("p",)), parameters=(SizeParameter("p", 0, 2**63 - 1),))
    analysis, runner = bind_reference_schedule(overflow)
    assert analysis.statements[0].opaque
    assert runner.run({}, sizes={"p": 0}).values == (0,)
    for run in (runner.run, lambda inputs, **kwargs: interpret(overflow, inputs, **kwargs)):
        with pytest.raises(ValueError, match="64-bit"):
            run({}, sizes={"p": 2**63 - 1})


def test_relational_sizes_prove_offset_accesses_and_guard_each_invocation():
    bank = BankType(I, 8)
    body = region((Value("i", X), Value("xs", bank)), (
        Parameter((), (Value("prefix", X),), "prefix"),
        Coordinate(("prefix", "i"), (Value("position", X),), AffineIndex(0, (IndexTerm(0), IndexTerm(1)))),
        Select(("xs", "position"), (Value("x", I),)),
    ), ("x",))
    constraint = SizeConstraint((ParameterTerm("prefix"), ParameterTerm("n")), 8)
    program = Program(region((Value("xs", bank),), (
        Map(("xs",), (Value("ys", BankType(I, "n")),), "n", body),
    ), ("ys",)), parameters=(SizeParameter("prefix", 0, 7), SizeParameter("n", 0, 8)), constraints=(constraint,))
    assert decode_boundary(encode_boundary(program), Program) == program
    analysis, runner = bind_reference_schedule(program)
    assert not analysis.statements
    assert analyze_region_program(replace(program, constraints=()), optimize=False).statements[0].opaque
    xs = Bank(tuple(range(10, 18)))
    for prefix, count in ((0, 8), (7, 1), (3, 4), (7, 0)):
        expected = (Bank(xs.elements[prefix:prefix + count]),)
        assert runner.run({"xs": xs}, sizes={"prefix": prefix, "n": count}).values == expected
        assert interpret(program, {"xs": xs}, sizes={"prefix": prefix, "n": count}).values == expected
    with pytest.raises(ValueError, match="declared relation"):
        runner.run({"xs": xs}, sizes={"prefix": 7, "n": 2})
    for invalid in (replace(constraint, terms=(ParameterTerm("absent"),)),
                    replace(constraint, terms=(ParameterTerm("n"), ParameterTerm("n"))),
                    replace(constraint, terms=(ParameterTerm("n", 0),))):
        with pytest.raises(ValueError):
            validate_region_program(replace(program, constraints=(invalid,)))

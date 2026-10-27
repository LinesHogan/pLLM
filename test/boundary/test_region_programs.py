"""Region IR end to end: programs compile, schedule legally and match the reference.

Each representative program (wavefront, tiling, scan, control flow, symbolic
sizes, native imports) is bound under every source-derived and hand-written
legal schedule and executed against ``test.region_reference``. Execution must
stay inside its capacity bound and leave caller inputs untouched. Separate
tests cover what the scheduler and verifiers must reject, runtime guards, and
symbolic scale.
"""

import json
import math
from dataclasses import dataclass, field, replace

import pytest
import torch

from pllm import _core
from pllm.contracts.region_program import (
    AccessMode,
    AccessRange,
    AffineIndex,
    BankType,
    BorrowedResource,
    Call,
    Constant,
    Coordinate,
    Function,
    If,
    IndexTerm,
    InvalidProgram,
    Map,
    Parameter,
    ParameterTerm,
    Program,
    Publish,
    Resource,
    ResourceAccess,
    Scalar,
    ScalarKind,
    ScalarType,
    Scan,
    Select,
    SizeConstraint,
    SizeParameter,
    Snapshot,
    TensorType,
    Update,
    Value,
    While,
    verify,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import (
    analyze_region_program,
    explore_region_tiling,
    lower_region_schedule,
    preview_region_schedule,
    validate_region_program,
    validate_region_schedule,
)
from pllm.execution.region_native import NativePrimitive, tensor_item
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.region_values import Bank, copy_data
from pllm.execution.tensor_scan import TensorScan
from test.region_fixtures import bind, grid_cell, grid_program, inputs_for, region, tensor_grid
from test.region_reference import interpret
from test.state_fixtures import StorageObserver

I, X, B, F = ScalarType.INT, ScalarType.INDEX, ScalarType.BOOL, ScalarType.FLOAT


# ---------------------------------------------------------------------------
# Reference equivalence over representative programs
# ---------------------------------------------------------------------------


@dataclass
class Case:
    program: Program
    invocations: tuple[tuple[dict, dict], ...]  # (inputs, sizes)
    cells: dict = field(default_factory=dict)
    schedules: tuple[str, ...] = ()  # extra hand-written legal schedules
    tiles: bool = False  # also run automatic ISL tile candidates


def tensors_of(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Bank):
        for element in value.elements:
            yield from tensors_of(element)


def assert_same(actual, expected) -> None:
    """Exact value semantics: tensors bitwise, scalars by type and sign of zero."""
    if isinstance(expected, Bank):
        assert isinstance(actual, Bank) and len(actual.elements) == len(expected.elements)
        for a, e in zip(actual.elements, expected.elements, strict=True):
            assert_same(a, e)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert type(actual) is type(expected) and actual == expected
        if isinstance(expected, float):
            assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


def oracle(cell, sizes):
    if isinstance(cell, ExportedCell):
        return lambda args: cell(*args)
    return lambda args: cell.invoke(tuple(copy_data(a) for a in args), None, sizes)


def mutating_grid_cell():
    def step(h, w):
        old = h.clone()
        h.mul_(0.5).add_(w * 0.25)
        w.add_(old * 0.0625)
        return h, w

    zeros, ones = torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)
    return ExportedCell.capture(step, ("h", "w"), ("next_h", "next_w"), (zeros, ones))


def wavefront(chunks, layers):
    cell = mutating_grid_cell()
    domain = f"0<=c<{chunks} and 0<=l<{layers}"
    extra = (f"{{ S0[c,l] -> [c+l,c] : {domain} }}", f"{{ S0[c,l] -> [floor(c/2),floor(l/3),c,l] : {domain} }}")
    return Case(
        tensor_grid(chunks, layers, cell),
        ((inputs_for(chunks, layers), {}),),
        {cell.digest: cell},
        extra,
        tiles=True,
    )


def scan_versions():
    """A tensor scan with a symbolic trip count inside a branch, collecting each carry version."""
    cell = ExportedCell.capture(
        lambda w, scale, x: (w * scale + x, w, w * scale + x),
        ("w", "scale", "x"),
        ("next", "old", "new"),
        tuple(torch.ones(3, dtype=torch.float64) for _ in range(3)),
    )
    program = TensorScan(cell, SizeParameter("n", 0, 5), carry=1, invariants=1).program
    w = torch.arange(3, dtype=torch.float64)

    def inputs(n):
        return {"arg0": w, "arg1": torch.full_like(w, 2.0), "arg2": Bank(tuple(w + i + 1 for i in range(n)))}

    block = program.entry.block
    results = tuple(Value(f"out{i}", type_) for i, type_ in enumerate(verify(program)))
    program = replace(
        program,
        entry=region(
            (*block.arguments, Value("enabled", B)),
            (If(("enabled", *(v.name for v in block.arguments)), results, program.entry, program.entry),),
            tuple(v.name for v in results),
        ),
    )
    invocations = tuple(({**inputs(n), "enabled": e}, {"n": n}) for n in (0, 5) for e in (True, False))
    return Case(program, invocations, {cell.digest: cell})


def doubling_cell():
    return ExportedCell.capture(lambda w: (w * 2,), ("w",), ("next",), (torch.ones(4, dtype=torch.float64),))


def while_program(cell, limit):
    tensor = cell.declaration.inputs[0]
    carried = (Value("n", I), Value("w", tensor), Value("stop", I))
    condition = region(carried, (Scalar(("n", "stop"), (Value("again", B),), ScalarKind.LESS),), ("again",))
    body = region(
        carried,
        (
            Call(("w",), (Value("next_w", tensor),), cell.digest),
            Constant((), (Value("one", I),), 1),
            Scalar(("n", "one"), (Value("next_n", I),), ScalarKind.ADD),
        ),
        ("next_n", "next_w", "stop"),
    )
    results = (Value("final_n", I), Value("final_w", tensor), Value("stop_out", I))
    return Program(
        region(carried, (While(("n", "w", "stop"), results, limit, condition, body),), ("final_n", "final_w")),
        (cell.declaration,),
    )


def pretest_loop():
    """At most seven iterations: stops on the predicate, skips the body, or hits the limit."""
    cell = doubling_cell()
    starts = (-2, 4, -5)
    invocations = tuple(({"n": n, "w": torch.arange(4, dtype=torch.float64), "stop": 3}, {}) for n in starts)
    return Case(while_program(cell, 7), invocations, {cell.digest: cell})


def nested_control(trips, width):
    """A map nested in a scan nested in a branch, run with the branch taken and not."""
    cell = doubling_cell()
    tensor = cell.declaration.inputs[0]
    bank = BankType(tensor, width)
    element = region(
        (Value("j", X), Value("b", bank)),
        (Select(("b", "j"), (Value("w", tensor),)), Call(("w",), (Value("next", tensor),), cell.digest)),
        ("next",),
    )
    scan_body = region(
        (Value("i", X), Value("b", bank)), (Map(("b",), (Value("next", bank),), width, element),), ("next",)
    )
    taken = region((Value("b", bank),), (Scan(("b",), (Value("result", bank),), trips, scan_body),), ("result",))
    identity = region((Value("b", bank),), (), ("b",))
    program = Program(
        region(
            (Value("p", B), Value("b", bank)), (If(("p", "b"), (Value("result", bank),), taken, identity),), ("result",)
        ),
        (cell.declaration,),
    )
    rows = Bank(tuple(torch.full((4,), i + 1.0, dtype=torch.float64) for i in range(width)))
    return Case(program, tuple(({"p": p, "b": rows}, {}) for p in (False, True)), {cell.digest: cell})


def prefix_program():
    """Symbolic prefix sums: one program for every length n in [0, 8]."""
    bank = BankType(I, "n")
    body = region(
        (Value("i", X), Value("total", I), Value("xs", bank)),
        (Select(("xs", "i"), (Value("x", I),)), Scalar(("total", "x"), (Value("next", I),), ScalarKind.ADD)),
        ("next", "xs", "total", "next"),
    )
    outputs = (Value("total", I), Value("unused", bank), Value("before", bank), Value("after", bank))
    return Program(
        region(
            (Value("seed", I), Value("xs", bank)),
            (Scan(("seed", "xs"), outputs, "n", body),),
            ("total", "before", "after"),
        ),
        parameters=(SizeParameter("n", 0, 8),),
    )


def prefix_sums():
    invocations = tuple(({"seed": 11, "xs": Bank(tuple(range(1, n + 1)))}, {"n": n}) for n in (8, 0, 3, 1))
    return Case(prefix_program(), invocations)


def parameter_offsets():
    """Bank elements routed by size parameters through chained affine coordinates."""
    bank, output = BankType(I, 32), BankType(I, "n")
    body = region(
        (Value("i", X), Value("xs", bank)),
        (
            Parameter((), (Value("prefix", X),), "prefix"),
            Coordinate(("prefix", "i"), (Value("position", X),), AffineIndex(0, (IndexTerm(0), IndexTerm(1)))),
            Coordinate(("position",), (Value("doubled", X),), AffineIndex(0, (IndexTerm(0, 2),))),
            Coordinate(("doubled", "prefix"), (Value("index", X),), AffineIndex(0, (IndexTerm(0), IndexTerm(1, -1)))),
            Select(("xs", "index"), (Value("selected", I),)),
        ),
        ("selected",),
    )
    entry = region((Value("xs", bank),), (Map(("xs",), (Value("ys", output),), "n", body),), ("ys",))
    xs = Bank(tuple(100 + i for i in range(32)))
    shapes = ((3, 0), (10, 8), (5, 3))
    entry = region(
        (Value("choose", B), Value("xs", bank)), (If(("choose", "xs"), (Value("ys", output),), entry, entry),), ("ys",)
    )
    invocations = tuple(({"xs": xs, "choose": c}, {"prefix": p, "n": n}) for p, n in shapes for c in (False, True))
    program = Program(entry, parameters=(SizeParameter("prefix", 3, 10), SizeParameter("n", 0, 8)))
    return Case(program, invocations)


def relational_program():
    bank = BankType(I, 8)
    body = region(
        (Value("i", X), Value("xs", bank)),
        (
            Parameter((), (Value("prefix", X),), "prefix"),
            Coordinate(("prefix", "i"), (Value("position", X),), AffineIndex(0, (IndexTerm(0), IndexTerm(1)))),
            Select(("xs", "position"), (Value("x", I),)),
        ),
        ("x",),
    )
    return Program(
        region((Value("xs", bank),), (Map(("xs",), (Value("ys", BankType(I, "n")),), "n", body),), ("ys",)),
        parameters=(SizeParameter("prefix", 0, 7), SizeParameter("n", 0, 8)),
        constraints=(SizeConstraint((ParameterTerm("prefix"), ParameterTerm("n")), 8),),
    )


def literals():
    """Plan literals keep their Python type, including -0.0 inside an empty map."""
    column = BankType(F, "n")
    body = region(
        (Value("i", X), Value("scale", F)),
        (Constant((), (Value("negative_zero", F),), -0.0),),
        ("scale", "negative_zero"),
    )
    program = Program(
        region(
            (),
            (
                Constant((), (Value("yes", B),), True),
                Constant((), (Value("no", B),), False),
                Constant((), (Value("quarter", F),), 0.25),
                Constant((), (Value("zero", F),), -0.0),
                Constant((), (Value("integer", I),), 7),
                Map(("quarter",), (Value("quarters", column), Value("zeros", column)), "n", body),
            ),
            ("yes", "no", "quarter", "zero", "integer", "quarters", "zeros"),
        ),
        parameters=(SizeParameter("n", 0, 3),),
    )
    return Case(program, (({}, {"n": 0}), ({}, {"n": 3})))


def tensor_predicate_loop():
    """A pretest loop whose condition is an explicit tensor-to-scalar read."""
    example = torch.zeros(())
    condition = ExportedCell.capture(lambda x: (x < 3,), ("x",), ("predicate",), (example,))
    update = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("next",), (example,))
    item = tensor_item(condition.declaration.outputs[0])
    t = condition.declaration.inputs[0]
    test = region(
        (Value("x", t),),
        (
            Call(("x",), (Value("tensor_predicate", condition.declaration.outputs[0]),), condition.digest),
            Call(("tensor_predicate",), (Value("predicate", B),), item.declaration.name),
        ),
        ("predicate",),
    )
    body = region((Value("x", t),), (Call(("x",), (Value("next", t),), update.digest),), ("next",))
    program = Program(
        region((Value("x", t),), (While(("x",), (Value("final", t),), 8, test, body),), ("final",)),
        (condition.declaration, update.declaration, item.declaration),
    )
    cells = {condition.digest: condition, update.digest: update, item.declaration.name: item}
    return Case(program, tuple(({"x": torch.full((), s)}, {}) for s in (0.0, 5.0)), cells)


FAMILIES = {
    "wavefront-and-tiling": (lambda: wavefront(3, 3),),
    "scans": (scan_versions, prefix_sums),
    "control-flow": (pretest_loop, lambda: nested_control(3, 4)),
    "symbolic-sizes": (parameter_offsets,),
    "literals-and-native-imports": (literals, tensor_predicate_loop),
}


@pytest.mark.parametrize("family", FAMILIES)
def test_programs_match_reference_under_every_legal_schedule(family):
    for build in FAMILIES[family]:
        check_case(build())


def check_case(case: Case) -> None:
    """Bind every legal schedule, then run each invocation against the reference."""
    program, cells = case.program, case.cells
    analysis = analyze_region_program(program)
    schedules = [candidate.schedule_map for candidate in analysis.candidates]
    schedules.extend(case.schedules)
    if case.tiles:
        schedules.extend(candidate.schedule_map for candidate in explore_region_tiling(program).candidates)
    plans = [(bind(program, cells, schedule), capacity_bound(case, schedule)) for schedule in schedules]
    for inputs, sizes in case.invocations:
        functions = {f.name: oracle(cells[f.name], sizes) for f in program.functions}
        expected = interpret(program, inputs, functions=functions, sizes=sizes)
        external = tuple(t for value in inputs.values() for t in tensors_of(value))
        saved = tuple(t.clone() for t in external)
        external_bytes = sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in external}.values())
        for execution, bound in plans:
            with StorageObserver(external) as observed:
                actual = execution.run(inputs, sizes=sizes)
            for got, want in zip(actual.values, expected.values, strict=True):
                assert_same(got, want)
            if bound is not None:
                assert observed.peak - external_bytes <= bound
        for tensor, before in zip(external, saved, strict=True):
            torch.testing.assert_close(tensor, before, rtol=0, atol=0)


def capacity_bound(case, schedule):
    if not case.cells:  # scalar-only programs allocate no tensor storage
        return None
    devices = analyze_region_capacity(case.program, schedule, case.cells).devices
    return devices[0].peak_tensor_bytes_upper_bound if devices else None


def test_wavefront_candidate_really_reorders_cells():
    """The scheduler finds a real wavefront; it is not a relabelled serial loop."""
    program = tensor_grid(4, 3, grid_cell())
    analysis = analyze_region_program(program)
    serial = [p.coordinates for p in preview_region_schedule(program, analysis.reference_schedule)]
    parallel = next(c for c in analysis.candidates if c.source == "isl-parallel")
    instances = preview_region_schedule(program, parallel.schedule_map)
    assert len({point.time[0] for point in instances}) == 4 + 3 - 1
    assert [p.coordinates for p in instances] != serial


# ---------------------------------------------------------------------------
# Scheduler legality: each program pairs a legal reorder with an illegal one
# ---------------------------------------------------------------------------


def bank_observer():
    """Reading the updated bank must stay after the update."""
    bank = BankType(I, 2)
    return Program(
        region(
            (Value("w", bank),),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Select(("w", "zero"), (Value("old", I),)),
                Call(("old",), (Value("new", I),), "update"),
                Update(("w", "zero", "new"), (Value("next_bank", bank),)),
                Call(("next_bank",), (Value("observed", I),), "observe"),
            ),
            ("next_bank", "observed"),
        ),
        (Function("update", (I,), (I,)), Function("observe", (bank,), (I,))),
    )


def resource_loop(indexed):
    """Per-index state lets phases split; one shared state cell does not."""
    indices = ("i",) if indexed else ()
    body = region(
        (Value("i", X),),
        (
            Snapshot(indices, (Value("w", I),), "state"),
            Call(("w",), (Value("next_w", I),), "cell"),
            Publish(("next_w", *indices), (), "state"),
        ),
        (),
    )
    return Program(
        region((), (Scan((), (), 4, body),), ()),
        (Function("cell", (I,), (I,)),),
        (Resource("state", (I,), (4,) if indexed else ()),),
    )


def branch_between_calls(checked):
    """A branch is movable unless it holds checked arithmetic that may raise."""
    then = region((Value("x", I),), (Call(("x",), (Value("out", I),), "f"),), ("out",))
    if checked:
        then = region(
            (Value("x", I),),
            (Constant((), (Value("one", I),), 1), Scalar(("x", "one"), (Value("unused", I),), ScalarKind.ADD)),
            ("x",),
        )
    other = region((Value("x", I),), (), ("x",))
    return Program(
        region(
            (Value("x", I), Value("p", B)),
            (
                Call(("x",), (Value("a", I),), "f"),
                If(("p", "x"), (Value("b", I),), then, other),
                Call(("x",), (Value("c", I),), "f"),
            ),
            ("a", "b", "c"),
        ),
        (Function("f", (I,), (I,)),),
    )


def indexed_layers():
    """Native read/write/prefix calls over declared layer intervals."""
    tensor = TensorType((2,), "torch.float32")
    cell = AccessRange(AffineIndex(0, (IndexTerm(0),)), AffineIndex(1, (IndexTerm(0),)))
    prefix_range = AccessRange(AffineIndex(), AffineIndex(0, (IndexTerm(0),)))
    read = Function("read", (X,), (tensor,), ("layers",), (ResourceAccess("layers", AccessMode.READ, (cell,)),))
    write = Function("write", (X,), (), ("layers",), (ResourceAccess("layers", AccessMode.WRITE, (cell,)),))
    prefix = Function(
        "prefix", (X,), (tensor,), ("layers",), (ResourceAccess("layers", AccessMode.READ, (prefix_range,)),)
    )
    return Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Coordinate((), (Value("one", X),), AffineIndex(1)),
                Coordinate((), (Value("two", X),), AffineIndex(2)),
                Call(("zero",), (Value("old", tensor),), read.name),
                Call(("one",), (), write.name),
                Call(("zero",), (), write.name),
                Call(("two",), (Value("sum", tensor),), prefix.name),
                Call(("zero",), (), write.name),
            ),
            ("old", "sum"),
        ),
        (read, write, prefix),
        (Resource("layers", (tensor,), (2,)),),
    )


GRID = "0<=c<3 and 0<=l<3"
SWAP = "{ S0[] -> [1]; S1[] -> [0] }"
IN_ORDER = "{ S0[] -> [0]; S1[] -> [1] }"
PHASES = "{ S0[i] -> [0,i] : 0<=i<4; S1[i] -> [1,i] : 0<=i<4; S2[i] -> [2,i] : 0<=i<4 }"
REVERSE = "{ S0[] -> [2]; S1[] -> [1]; S2[] -> [0] }"
LEGALITY = {
    "wavefront": (
        lambda: grid_program(3, 3),
        (f"{{ S0[c,l] -> [c+l,c] : {GRID} }}", f"{{ S0[c,l] -> [floor(c/2),floor(l/2),c,l] : {GRID} }}"),
        (f"{{ S0[c,l] -> [-c,l] : {GRID} }}", "{ S0[c,l] -> [c,l] : 0<=c<2 and 0<=l<3 }"),
    ),
    "bank-versions": (bank_observer, (IN_ORDER,), (SWAP,)),
    "indexed-resource": (lambda: resource_loop(True), (PHASES,), ()),
    "shared-resource": (lambda: resource_loop(False), (), (PHASES,)),
    "resource-intervals": (
        indexed_layers,
        ("{ S0[] -> [1]; S1[] -> [0]; S2[] -> [2]; S3[] -> [3]; S4[] -> [4] }",),
        ("{ S0[] -> [2]; S1[] -> [0]; S2[] -> [1]; S3[] -> [3]; S4[] -> [4] }",),
    ),
    "pure-branch": (lambda: branch_between_calls(False), (REVERSE,), ()),
    "checked-branch": (lambda: branch_between_calls(True), (), (REVERSE,)),
}


def test_scheduler_accepts_independent_reorders_and_rejects_broken_dependences():
    for build, legal, illegal in LEGALITY.values():
        program = build()
        for schedule in legal:
            validate_region_schedule(program, schedule)
        for schedule in illegal:
            with pytest.raises(ValueError):
                validate_region_schedule(program, schedule)


# ---------------------------------------------------------------------------
# Programs the verifiers reject before planning
# ---------------------------------------------------------------------------


def malformed_programs():
    """One representative per verifier family: dataflow, scans, effects, indices, resources, sizes."""
    hidden = region((Value("i", X), Value("carry", I)), (), ("outside",))
    yield Program(region((Value("outside", I),), (Scan(("outside",), (Value("result", I),), 1, hidden),), ("result",)))
    body = region((Value("i", X), Value("n", I)), (), ("n", "i"))
    outputs = (Value("n1", I), Value("ys", BankType(X, 4)))
    yield Program(region((Value("n", I),), (Scan(("n",), outputs, 3, body),), ("n1", "ys")))
    snapshot = region((Value("i", X),), (Snapshot((), (Value("x", I),), "w"),), ())
    yield Program(region((), (Map((), (), 2, snapshot),), ()), resources=(Resource("w", (I,)),))
    two = Coordinate((), (Value("two", X),), AffineIndex(2))
    zero = Coordinate(("two",), (Value("c", X),), AffineIndex(0, (IndexTerm(0, 0),)))
    yield Program(region((), (two, zero), ("two",)))
    read = Function("read", (X,), (TensorType((2,), "torch.float32"),), ("kv",))
    yield Program(
        region((Value("i", X),), (Call(("i",), (Value("v", read.outputs[0]),), read.name),), ("v",)),
        (read,),
        (BorrowedResource("kv", (2,)),),
    )
    program = relational_program()
    (constraint,) = program.constraints
    yield replace(program, constraints=(replace(constraint, terms=(ParameterTerm("absent"),)),))


def test_native_verifier_rejects_malformed_programs():
    for program in malformed_programs():
        with pytest.raises(ValueError):
            validate_region_program(program)


def test_python_frontend_and_wire_decoder_reject_what_the_wire_cannot_carry():
    valid = Program(region((Value("x", I),), (Scalar(("x", "x"), (Value("y", I),), ScalarKind.ADD),), ("y",)))
    (operation,) = valid.entry.block.operations
    with pytest.raises(InvalidProgram):  # A plain string is not a ScalarKind.
        verify(replace(valid, entry=region(valid.entry.block.arguments, (replace(operation, kind="add"),), ("y",))))
    assert verify(valid) == (I,)
    wire = json.loads(encode_boundary(prefix_program()))
    wire["entry"]["block"]["operations"][0]["count"] = True
    with pytest.raises(ValueError):
        _core.validate_region_program(json.dumps(wire))
    with pytest.raises((ValueError, KeyError, InvalidProgram)):
        verify(decode_boundary(json.dumps(wire), Program))


def test_schedules_cannot_introduce_undeclared_size_parameters():
    with pytest.raises(ValueError):
        validate_region_schedule(
            prefix_program(), "[__size_n,alien] -> { S0[i] -> [i+alien] : 0<=__size_n<=8 and 0<=i<__size_n }"
        )


# ---------------------------------------------------------------------------
# Runtime guards and dynamic semantics
# ---------------------------------------------------------------------------


def test_invocation_guards_reject_bad_sizes_and_shapes_before_any_call():
    calls = []
    observe = NativePrimitive(Function("observe", (), ()), lambda inputs, resources: calls.append(1) or ())
    program = replace(prefix_program(), functions=(observe.declaration,))
    block = program.entry.block
    program = replace(
        program, entry=region(block.arguments, (Call((), (), "observe"), *block.operations), block.yields)
    )
    runner = bind(program, {"observe": observe})
    for inputs, sizes in (({"seed": 1, "xs": Bank((1, 2))}, {"n": 9}), ({"seed": 1, "xs": Bank((1,))}, {"n": 2})):
        with pytest.raises(ValueError):
            runner.run(inputs, sizes=sizes)
    assert not calls
    with pytest.raises(ValueError):
        bind(relational_program()).run({"xs": Bank(tuple(range(8)))}, sizes={"prefix": 7, "n": 2})


def test_unselected_branch_is_not_evaluated_and_a_failed_invocation_leaves_no_state():
    bank = BankType(I, 1)
    good = region((Value("b", bank),), (Constant((), (Value("y", I),), 7),), ("y",))
    bad = region(
        (Value("b", bank),), (Constant((), (Value("index", X),), 1), Select(("b", "index"), (Value("y", I),))), ("y",)
    )
    program = Program(region((Value("p", B), Value("b", bank)), (If(("p", "b"), (Value("y", I),), good, bad),), ("y",)))
    execution = bind(program)
    assert execution.run({"p": True, "b": Bank((3,))}).values == (7,)
    with pytest.raises(ValueError):
        execution.run({"p": False, "b": Bank((3,))})
    assert execution.run({"p": True, "b": Bank((3,))}).values == (7,)
    # Checked arithmetic in a taken branch is never eliminated, even when unused.
    checked = branch_between_calls(True).entry.block.operations[1].then_region
    identity = region((Value("x", I),), (), ("x",))
    overflow = Program(
        region((Value("p", B), Value("x", I)), (If(("p", "x"), (Value("out", I),), checked, identity),), ("out",))
    )
    runner = bind(overflow)
    assert runner.run({"p": False, "x": 2**63 - 1}).values == (2**63 - 1,)
    with pytest.raises(ValueError):
        runner.run({"p": True, "x": 2**63 - 1})


def test_million_step_domains_stay_symbolic():
    large = grid_program(1_000_000, 16)
    analysis = analyze_region_program(large)
    assert len(analysis.statements) == 1 and len(analysis.dependencies) < 1000
    schedule = analysis.candidates[-1].schedule_map
    assert len(encode_boundary(lower_region_schedule(large, schedule))) < 5000
    with pytest.raises(ValueError):
        preview_region_schedule(large, schedule, max_instances=100)


# ---------------------------------------------------------------------------
# Native imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_native_import_copies_inputs_and_returns_independent_storage(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    full = TensorType((4, 4), "torch.float32", device)
    small = TensorType((1, 4), "torch.float32", device)

    def update(inputs, resources):
        (x,) = inputs
        x.add_(1)
        return x[:1], x[:1]

    native = NativePrimitive(Function("native.views", (full,), (small, small)), update)
    program = Program(
        region(
            (Value("x", full),),
            (Call(("x",), (Value("a", small), Value("b", small)), native.declaration.name),),
            ("x", "a", "b"),
        ),
        (native.declaration,),
    )
    x = torch.ones((4, 4), device=device)
    same, a, b = bind(program, {native.declaration.name: native}).run({"x": x}).values
    assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    torch.testing.assert_close(a, x[:1] + 1, rtol=0, atol=0)
    torch.testing.assert_close(same, x, rtol=0, atol=0)
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)

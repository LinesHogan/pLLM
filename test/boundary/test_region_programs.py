"""Region IR end to end: programs compile, schedule legally and match the reference.

Each representative program (wavefront, tiling, scan, control flow, symbolic
sizes, native imports) is analyzed once, bound under every source-derived and
hand-written legal schedule, and executed against ``test.region_reference``.
Execution must not recompile, must stay inside its capacity bound and must
leave caller inputs untouched. Separate tests cover what the scheduler and
verifiers must reject, runtime guards, and symbolic scale.
"""

import math
import weakref
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
    ExternalResource,
    Function,
    If,
    IndexTerm,
    InvalidProgram,
    Map,
    Parameter,
    ParameterTerm,
    Program,
    Publish,
    Region,
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
from pllm.execution.codec import encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import (
    analyze_region_program,
    explore_region_tiling,
    lower_region_bindings,
    lower_region_schedule,
    preview_region_schedule,
    validate_region_program,
    validate_region_schedule,
)
from pllm.execution.region_native import NativePrimitive, tensor_item
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.region_values import Bank, copy_data
from pllm.execution.tensor_scan import TensorScan
from test.region_fixtures import (
    StorageObserver,
    assert_grid_closed_form,
    bind,
    grid_cell,
    grid_program,
    inputs_for,
    region,
    tensor_grid,
)
from test.region_reference import interpret

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
    capacity: bool = True  # check real storage against the static capacity bound


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


def forbid_compilation(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("bound execution must not analyze, lower or tile again")

    for name in ("analyze_region_program", "lower_region_schedule", "lower_region_bindings", "explore_region_tiling"):
        monkeypatch.setattr(_core, name, forbidden)


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
        extra if chunks and layers else (),
        tiles=True,
    )


def parametric_wavefront():
    cell = mutating_grid_cell()
    program = replace(
        tensor_grid("chunks", "layers", cell),
        parameters=(SizeParameter("chunks", 0, 80), SizeParameter("layers", 0, 12)),
    )
    prefix = "[__size_chunks,__size_layers] -> "
    guard = "0<=__size_chunks<=80 and 0<=__size_layers<=12 and 0<=c<__size_chunks and 0<=l<__size_layers"
    extra = (
        prefix + "{ S0[c,l] -> [c+l,c] : " + guard + " }",
        prefix + "{ S0[c,l] -> [floor(c/2),floor(l/3),c,l] : " + guard + " }",
    )
    shapes = ((0, 0), (3, 0), (5, 3), (9, 4))
    invocations = tuple((inputs_for(c, l), {"chunks": c, "layers": l}) for c, l in shapes)
    return Case(program, invocations, {cell.digest: cell}, extra)


def scan_versions(dynamic):
    """A tensor scan collecting each pre- and post-update carry version.

    The dynamic variant runs the same scan with a symbolic trip count inside a branch.
    """
    cell = ExportedCell.capture(
        lambda w, scale, x: (w * scale + x, w, w * scale + x),
        ("w", "scale", "x"),
        ("next", "old", "new"),
        tuple(torch.ones(3, dtype=torch.float64) for _ in range(3)),
    )
    count = SizeParameter("n", 0, 5) if dynamic else 5
    program = TensorScan(cell, count, carry=1, invariants=1).program
    w = torch.arange(3, dtype=torch.float64)

    def inputs(n):
        return {"arg0": w, "arg1": torch.full_like(w, 2.0), "arg2": Bank(tuple(w + i + 1 for i in range(n)))}

    if not dynamic:
        return Case(program, ((inputs(5), {}),), {cell.digest: cell})
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


def coordinate_branch():
    """A branch inside an affine map sees each instance's own coordinate."""
    cell = doubling_cell()
    tensor = cell.declaration.inputs[0]
    changed = region((Value("w", tensor),), (Call(("w",), (Value("next", tensor),), cell.digest),), ("next",))
    same = region((Value("w", tensor),), (), ("w",))
    body = region(
        (Value("i", X), Value("w", tensor)),
        (
            Constant((), (Value("target", X),), 2),
            Scalar(("i", "target"), (Value("selected", B),), ScalarKind.EQUAL),
            If(("selected", "w"), (Value("y", tensor),), changed, same),
        ),
        ("y",),
    )
    program = Program(
        region((Value("w", tensor),), (Map(("w",), (Value("ys", BankType(tensor, 5)),), 5, body),), ("ys",)),
        (cell.declaration,),
    )
    return Case(program, (({"w": torch.arange(4, dtype=torch.float64)}, {}),), {cell.digest: cell}, capacity=False)


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


def parameter_offsets(branched):
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
    if branched:
        entry = region(
            (Value("choose", B), Value("xs", bank)),
            (If(("choose", "xs"), (Value("ys", output),), entry, entry),),
            ("ys",),
        )
        invocations = tuple(({"xs": xs, "choose": c}, {"prefix": p, "n": n}) for p, n in shapes for c in (False, True))
    else:
        invocations = tuple(({"xs": xs}, {"prefix": p, "n": n}) for p, n in shapes)
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


def relational_sizes():
    xs = Bank(tuple(range(10, 18)))
    shapes = ((0, 8), (7, 1), (3, 4), (7, 0))
    return Case(relational_program(), tuple(({"xs": xs}, {"prefix": p, "n": n}) for p, n in shapes))


def parameter_products():
    body = region(
        (Value("i", X),),
        (
            Parameter((), (Value("p", X),), "p"),
            Scalar(("p", "i"), (Value("scaled", X),), ScalarKind.MUL),
            Scalar(("p", "p"), (Value("square", X),), ScalarKind.MUL),
        ),
        ("scaled", "square"),
    )
    outputs = (Value("scaled", BankType(X, 3)), Value("squared", BankType(X, 3)))
    program = Program(
        region((), (Map((), outputs, 3, body),), ("scaled", "squared")), parameters=(SizeParameter("p", 2, 7),)
    )
    return Case(program, (({}, {"p": 2}), ({}, {"p": 7})))


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
    return Case(program, (({}, {"n": 0}), ({}, {"n": 3})), capacity=False)


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


def bank_versions():
    """Updating a bank at a dynamic index leaves the old bank readable."""
    tensor = TensorType((2,), "torch.float32", "cpu")
    bank = BankType(tensor, 2)
    program = Program(
        region(
            (Value("b", bank), Value("i", X), Value("new", tensor)),
            (
                Update(("b", "i", "new"), (Value("updated", bank),)),
                Select(("b", "i"), (Value("old", tensor),)),
                Select(("updated", "i"), (Value("selected", tensor),)),
            ),
            ("b", "old", "selected"),
        )
    )
    inputs = {"b": Bank((torch.ones(2), torch.full((2,), 2.0))), "i": 1, "new": torch.full((2,), 3.0)}
    return Case(program, ((inputs, {}),), capacity=False)


def native_bank_in_branch():
    """A native import over a symbolic bank inside a branch uses the bound size."""
    bank = BankType(X, "n")
    signature = Function("identity", (bank,), (bank,))
    primitive = NativePrimitive(signature, lambda inputs, resources: inputs)
    branch = region(
        (),
        (
            Map((), (Value("xs", bank),), "n", region((Value("i", X),), (), ("i",))),
            Call(("xs",), (Value("ys", bank),), signature.name),
        ),
        ("ys",),
    )
    program = Program(
        region((Value("p", B),), (If(("p",), (Value("result", bank),), branch, branch),), ("result",)),
        (signature,),
        parameters=(SizeParameter("n", 0, 7),),
    )
    invocations = tuple(({"p": p}, {"n": n}) for n in (0, 7, 2) for p in (True, False))
    return Case(program, invocations, {signature.name: primitive}, capacity=False)


CASES = {
    "wavefront": lambda: wavefront(5, 7),
    "wavefront-empty": lambda: wavefront(0, 3),
    "parametric-wavefront": parametric_wavefront,
    "scan-versions": lambda: scan_versions(False),
    "scan-versions-symbolic-in-branch": lambda: scan_versions(True),
    "while": pretest_loop,
    "nested-control": lambda: nested_control(3, 4),
    "coordinate-branch": coordinate_branch,
    "prefix-sums": prefix_sums,
    "parameter-offsets": lambda: parameter_offsets(False),
    "parameter-offsets-in-branch": lambda: parameter_offsets(True),
    "relational-sizes": relational_sizes,
    "parameter-products": parameter_products,
    "literals": literals,
    "tensor-predicate-loop": tensor_predicate_loop,
    "bank-versions": bank_versions,
    "native-bank-in-branch": native_bank_in_branch,
}


FAMILIES = {
    "wavefront-and-tiling": ("wavefront", "wavefront-empty", "parametric-wavefront"),
    "scans": ("scan-versions", "scan-versions-symbolic-in-branch", "prefix-sums"),
    "control-flow": ("while", "nested-control", "coordinate-branch", "bank-versions"),
    "symbolic-sizes": ("parameter-offsets", "parameter-offsets-in-branch", "relational-sizes", "parameter-products"),
    "literals-and-native-imports": ("literals", "tensor-predicate-loop", "native-bank-in-branch"),
}


@pytest.mark.parametrize("family", FAMILIES)
def test_programs_match_reference_under_every_legal_schedule(family, monkeypatch):
    for name in FAMILIES[family]:
        check_case(CASES[name](), monkeypatch)


def check_case(case: Case, monkeypatch) -> None:
    """Bind every legal schedule, then run each invocation against the reference."""
    program, cells = case.program, case.cells
    analysis = analyze_region_program(program)
    schedules = [candidate.schedule_map for candidate in analysis.candidates]
    schedules.extend(case.schedules)
    if case.tiles:
        schedules.extend(candidate.schedule_map for candidate in explore_region_tiling(program).candidates)
    plans = [(bind(program, cells, schedule), capacity_bound(case, schedule)) for schedule in schedules]
    with monkeypatch.context() as patch:
        forbid_compilation(patch)
        for inputs, sizes in case.invocations:
            functions = {f.name: oracle(cells[f.name], sizes) for f in program.functions}
            expected = interpret(program, inputs, functions=functions, sizes=sizes)
            external = tuple(t for value in inputs.values() for t in tensors_of(value))
            saved = tuple(t.clone() for t in external)
            external_bytes = sum(
                {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in external}.values()
            )
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
    if not case.capacity:
        return None
    devices = analyze_region_capacity(case.program, schedule, case.cells).devices
    return devices[0].peak_tensor_bytes_upper_bound if devices else None


def test_wavefront_candidates_reorder_cells_and_match_a_closed_form():
    """The scheduler finds a real wavefront; it is not a relabelled serial loop."""
    program = tensor_grid(4, 3, grid_cell())
    analysis = analyze_region_program(program)
    serial = [p.coordinates for p in preview_region_schedule(program, analysis.reference_schedule)]
    parallel = next(c for c in analysis.candidates if c.source == "isl-parallel")
    instances = preview_region_schedule(program, parallel.schedule_map)
    assert len({point.time[0] for point in instances}) == 4 + 3 - 1
    assert [p.coordinates for p in instances] != serial
    cell = grid_cell()
    result = bind(tensor_grid(4, 3, cell), {cell.digest: cell}, parallel.schedule_map).run(inputs_for(4, 3))
    assert_grid_closed_form(result.values, 4, 3)


# ---------------------------------------------------------------------------
# Scheduler legality: each program pairs a legal reorder with an illegal one
# ---------------------------------------------------------------------------


def final_layer_feedback():
    """Chunk c+1 starts from chunk c's last layer: no wavefront is legal."""
    bank = BankType(I, 4)
    layer_body = grid_program(3, 4).entry.block.operations[0].body.block.operations[1].body
    chunk_body = region(
        (Value("c", X), Value("h", I), Value("w", bank)),
        (Scan(("h", "w"), (Value("h_final", I), Value("w_final", bank)), 4, layer_body),),
        ("h_final", "w_final"),
    )
    return Program(
        region(
            (Value("input", I), Value("weights", bank)),
            (Scan(("input", "weights"), (Value("output", I), Value("result", bank)), 3, chunk_body),),
            ("output", "result"),
        ),
        grid_program(3, 4).functions,
    )


def discarded_weights():
    """The same grid, but every layer discards its weight update."""
    program = grid_program(3, 3)
    outer = program.entry.block.operations[0]
    inner = outer.body.block.operations[1]
    inner = replace(inner, body=Region(replace(inner.body.block, yields=("next_h", "w"))))
    body = replace(outer.body.block, operations=(outer.body.block.operations[0], inner, outer.body.block.operations[2]))
    return replace(program, entry=Region(replace(program.entry.block, operations=(replace(outer, body=Region(body)),))))


def bank_observer(read_new):
    bank = BankType(I, 2)
    return Program(
        region(
            (Value("w", bank),),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Select(("w", "zero"), (Value("old", I),)),
                Call(("old",), (Value("new", I),), "update"),
                Update(("w", "zero", "new"), (Value("next_bank", bank),)),
                Call(("next_bank" if read_new else "w",), (Value("observed", I),), "observe"),
            ),
            ("next_bank", "observed"),
        ),
        (Function("update", (I,), (I,)), Function("observe", (bank,), (I,))),
    )


def resource_loop(indexed):
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


def unproven_index():
    return Program(
        region(
            (Value("index", X), Value("new", I)),
            (Snapshot(("index",), (Value("old", I),), "a"), Publish(("new",), (), "b")),
            (),
        ),
        resources=(Resource("a", (I,), (3,)), Resource("b", (I,))),
    )


def overflow_before_effect():
    return Program(
        region(
            (Value("x", I), Value("one", I)),
            (Scalar(("x", "one"), (Value("result", I),), ScalarKind.ADD), Publish(("one",), (), "w")),
            ("result",),
        ),
        resources=(Resource("w", (I,)),),
    )


def external_endpoints():
    first = Function("owner.first", (), (I,), ("first",))
    second = Function("owner.second", (), (I,), ("second",))
    return Program(
        region((), (Call((), (Value("x", I),), first.name), Call((), (Value("y", I),), second.name)), ("x", "y")),
        (first, second),
        (ExternalResource("first"), ExternalResource("second")),
    )


CELL = AccessRange(AffineIndex(0, (IndexTerm(0),)), AffineIndex(1, (IndexTerm(0),)))
PREFIX = AccessRange(AffineIndex(), AffineIndex(0, (IndexTerm(0),)))


def indexed_layers():
    """Native read/write/prefix calls over declared layer intervals."""
    tensor = TensorType((2,), "torch.float32")
    read = Function("read", (X,), (tensor,), ("layers",), (ResourceAccess("layers", AccessMode.READ, (CELL,)),))
    write = Function("write", (X,), (), ("layers",), (ResourceAccess("layers", AccessMode.WRITE, (CELL,)),))
    prefix = Function("prefix", (X,), (tensor,), ("layers",), (ResourceAccess("layers", AccessMode.READ, (PREFIX,)),))
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


def parametric_grid():
    return replace(
        tensor_grid("chunks", "layers", grid_cell()),
        parameters=(SizeParameter("chunks", 0, 80), SizeParameter("layers", 0, 12)),
    )


def affine_scaled_index():
    bank = BankType(I, 8)
    body = region(
        (Value("i", X), Value("w", bank)),
        (
            Coordinate(("i",), (Value("even", X),), AffineIndex(0, (IndexTerm(0, 2),))),
            Select(("w", "even"), (Value("x", I),)),
            Call(("x",), (Value("next", I),), "f"),
            Update(("w", "even", "next"), (Value("out", bank),)),
        ),
        ("out",),
    )
    return Program(
        region((Value("w", bank),), (Scan(("w",), (Value("out", bank),), 4, body),), ("out",)),
        (Function("f", (I,), (I,)),),
    )


GRID = "0<=c<3 and 0<=l<3"
PARAMETRIC = (
    "[__size_chunks,__size_layers] -> { S0[c,l] -> [%s] : 0<=__size_chunks<=80 and 0<=__size_layers<=12"
    " and 0<=c<__size_chunks and 0<=l<__size_layers }"
)
SWAP = "{ S0[] -> [1]; S1[] -> [0] }"
IN_ORDER = "{ S0[] -> [0]; S1[] -> [1] }"


WAVEFRONT = (
    (
        lambda: grid_program(3, 3),
        (f"{{ S0[c,l] -> [c+l,c] : {GRID} }}", f"{{ S0[c,l] -> [floor(c/2),floor(l/2),c,l] : {GRID} }}"),
        (
            f"{{ S0[c,l] -> [-c,l] : {GRID} }}",  # reversed weight carry
            f"{{ S0[c,l] -> [0,0] : {GRID} }}",  # collapsed instances
            "{ S0[c,l] -> [c,l] : 0<=c<2 and 0<=l<3 }",  # missing instances
            f"{{ S0[c,l] -> [t,l] : {GRID} and c<=t<=c+1 }}",  # multivalued
        ),
    ),
    (parametric_grid, (PARAMETRIC % "c+l,c",), (PARAMETRIC % "-c,l",)),
    (affine_scaled_index, ("{ S0[i] -> [-i] : 0<=i<4 }",), ()),
)
VERSIONS = (
    (discarded_weights, (f"{{ S0[c,l] -> [-c,l] : {GRID} }}",), ()),
    (
        final_layer_feedback,
        ("{ S0[c,l] -> [c,l] : 0<=c<3 and 0<=l<4 }",),
        ("{ S0[c,l] -> [c+l,c] : 0<=c<3 and 0<=l<4 }",),
    ),
    (lambda: bank_observer(False), (SWAP,), ()),
    (lambda: bank_observer(True), (IN_ORDER,), (SWAP,)),
)
PHASES = "{ S0[i] -> [0,i] : 0<=i<4; S1[i] -> [1,i] : 0<=i<4; S2[i] -> [2,i] : 0<=i<4 }"
RESOURCES = (
    (lambda: resource_loop(True), (PHASES,), ()),
    (lambda: resource_loop(False), (), (PHASES,)),
    (unproven_index, (IN_ORDER,), (SWAP,)),
    (external_endpoints, (IN_ORDER,), (SWAP,)),
    (
        indexed_layers,
        ("{ S0[] -> [1]; S1[] -> [0]; S2[] -> [2]; S3[] -> [3]; S4[] -> [4] }",),
        ("{ S0[] -> [2]; S1[] -> [0]; S2[] -> [1]; S3[] -> [3]; S4[] -> [4] }",),
    ),
)
REVERSE = "{ S0[] -> [2]; S1[] -> [1]; S2[] -> [0] }"
CHECKED = (
    (lambda: branch_between_calls(False), (REVERSE,), ()),
    (lambda: branch_between_calls(True), (), (REVERSE,)),
    (overflow_before_effect, (IN_ORDER,), (SWAP,)),
)


@pytest.mark.parametrize(
    "cases",
    (
        pytest.param(WAVEFRONT, id="wavefront"),
        pytest.param(VERSIONS, id="value-versions"),
        pytest.param(RESOURCES, id="resource-effects"),
        pytest.param(CHECKED, id="checked-arithmetic"),
    ),
)
def test_scheduler_accepts_independent_reorders_and_rejects_broken_dependences(cases):
    for build, legal, illegal in cases:
        program = build()
        for schedule in legal:
            validate_region_schedule(program, schedule)
        for schedule in illegal:
            with pytest.raises(ValueError):
                validate_region_schedule(program, schedule)


# ---------------------------------------------------------------------------
# Programs the verifiers reject before planning
# ---------------------------------------------------------------------------


def with_entry(program, operations, yields=None):
    block = program.entry.block
    return replace(
        program,
        entry=Region(replace(block, operations=tuple(operations), yields=block.yields if yields is None else yields)),
    )


def malformed_dataflow():
    hidden = region((Value("i", X), Value("carry", I)), (), ("outside",))
    yield Program(region((Value("outside", I),), (Scan(("outside",), (Value("result", I),), 1, hidden),), ("result",)))
    yield Program(region((Value("a", I),), (Constant((), (Value("a", I),), 2),), ("a",)))
    yield replace(grid_program(2, 3), functions=())  # unknown function
    wrong = region((Value("i", X), Value("carry", I)), (), ("i",))
    yield Program(region((Value("a", I),), (Scan(("a",), (Value("b", I),), 2, wrong),), ("b",)))
    left = region((), (Constant((), (Value("x", I),), 1),), ("x",))
    right = region((), (Constant((), (Value("x", B),), True),), ("x",))
    yield Program(region((Value("p", B),), (If(("p",), (Value("r", I),), left, right),), ("r",)))


def malformed_scans():
    body = region((Value("i", X), Value("n", I)), (), ("n", "i"))
    outputs = (Value("n1", I), Value("ys", BankType(X, 3)))
    yield Program(
        region(
            (Value("n", I),),
            (Scan(("n",), outputs, 3, replace(body, block=replace(body.block, yields=("i", "n")))),),
            ("n1", "ys"),
        )
    )
    yield Program(
        region((Value("n", I),), (Scan(("n",), (outputs[0], Value("ys", BankType(X, 4))), 3, body),), ("n1", "ys"))
    )
    yield Program(
        region((Value("n", I),), (Scan(("n",), (outputs[0], Value("ys", BankType(I, 3))), 3, body),), ("n1", "ys"))
    )
    program = prefix_program()
    (scan,) = program.entry.block.operations
    renamed = (*scan.outputs[:2], Value("before", BankType(I, "m")), scan.outputs[3])
    yield with_entry(
        replace(program, parameters=(*program.parameters, SizeParameter("m", 0, 8))), (replace(scan, outputs=renamed),)
    )


def malformed_effects():
    body = region((Value("i", X),), (Snapshot((), (Value("x", I),), "w"),), ())
    yield Program(region((), (Map((), (), 2, body),), ()), resources=(Resource("w", (I,)),))
    cell = region(
        (Value("l", X),),
        (Snapshot(("l",), (Value("w", I),), "layers"), Publish(("w", "l"), (), "layers")),
        (),
    )
    # A cell publication into an atomic bank resource is a different publication scope.
    yield Program(region((), (Scan((), (), 4, cell),), ()), resources=(Resource("layers", (BankType(I, 4),)),))
    payload = (I, I)
    pair = Program(region((Value("a", I),), (Publish(("a",), (), "pair"),), ()), resources=(Resource("pair", payload),))
    yield pair  # missing payload field


def malformed_coordinates():
    base = (Coordinate((), (Value("two", X),), AffineIndex(2)),)

    def coordinate(constant, terms, result=X):
        return Coordinate(("two",), (Value("c", result),), AffineIndex(constant, terms))

    for malformed in (
        coordinate(0, (IndexTerm(1),)),  # term past the input list
        coordinate(0, (IndexTerm(0, 0),)),  # zero coefficient
        coordinate(0, (IndexTerm(0), IndexTerm(0, 2))),  # duplicate argument
        coordinate(0, (IndexTerm(0),), result=I),  # non-index result
    ):
        yield Program(region((), (*base, malformed), ("two",)))
    yield Program(
        region((Value("flag", B),), (Coordinate(("flag",), (Value("c", X),), AffineIndex(0, (IndexTerm(0),))),), ("c",))
    )


def malformed_resources():
    tensor = TensorType((2,), "torch.float32")
    bad_range = AccessRange(AffineIndex(0, (IndexTerm(1),)), AffineIndex(1))
    program = indexed_layers()
    read = replace(program.functions[0], accesses=(ResourceAccess("layers", AccessMode.READ, (bad_range,)),))
    yield replace(program, functions=(read, *program.functions[1:]))
    external = external_endpoints()
    yield with_entry(external, (Snapshot((), (Value("x", I),), "first"), external.entry.block.operations[1]))
    read = Function("read", (X,), (tensor,), ("kv",))  # borrowed storage needs explicit accesses
    yield Program(
        region((Value("i", X),), (Call(("i",), (Value("v", tensor),), read.name),), ("v",)),
        (read,),
        (BorrowedResource("kv", (2,)),),
    )


def malformed_sizes():
    program = relational_program()
    (constraint,) = program.constraints
    for terms in ((ParameterTerm("absent"),), (ParameterTerm("n"), ParameterTerm("n")), (ParameterTerm("n", 0),)):
        yield replace(program, constraints=(replace(constraint, terms=terms),))


@pytest.mark.parametrize(
    "groups",
    (
        pytest.param((malformed_dataflow, malformed_scans), id="dataflow"),
        pytest.param((malformed_effects, malformed_resources), id="effects"),
        pytest.param((malformed_coordinates, malformed_sizes), id="indices"),
    ),
)
def test_native_verifier_rejects_malformed_programs(groups):
    for programs in groups:
        for program in programs():
            with pytest.raises(ValueError):
                validate_region_program(program)


def test_python_frontend_rejects_types_the_wire_cannot_carry():
    class Integer(int):
        pass

    valid = Program(region((Value("x", I),), (Scalar(("x", "x"), (Value("y", I),), ScalarKind.ADD),), ("y",)))
    (operation,) = valid.entry.block.operations
    entry = region((), (), ())
    body = region((Value("i", X), Value("n", I)), (), ("n",))
    read = Function("read", (), (), ("borrow",), (ResourceAccess("borrow", "read"),))
    invalid = (
        with_entry(valid, (replace(operation, kind="add"),)),
        Program(entry, resources=(Resource("state", [I]),)),
        Program(entry, resources=(Resource("state", ScalarType.INT),)),
        Program(entry, resources=(Resource("state", ("int",)),)),
        Program(region((Value("x", TensorType((Integer(2),), "torch.float32")),), (), ("x",))),
        replace(valid, parameters=(SizeParameter("n", Integer(0), 2),)),
        Program(region((), (Call((), (), read.name),), ()), (read,), (BorrowedResource("borrow", ()),)),
        *(
            Program(region((Value("n", I),), (Scan(("n",), (Value("r", I),), 3, body, invariants=n),), ("r",)))
            for n in (-1, True, 2)
        ),
    )
    for program in invalid:
        with pytest.raises(InvalidProgram):
            verify(program)
    assert verify(valid) == (I,)


def test_wire_decoder_and_native_verifier_reject_tampered_programs():
    import json

    from pllm.execution.codec import decode_boundary

    def tampered():
        wire = json.loads(encode_boundary(prefix_program()))
        wire["entry"]["block"]["operations"][0]["count"] = True
        yield wire
        wire = json.loads(encode_boundary(prefix_program()))
        wire["parameters"][0]["hint"] = "trust me"
        yield wire
        resource = Program(
            region((), (Snapshot((), (Value("x", I),), "w"), Publish(("x",), (), "w")), ("x",)),
            resources=(Resource("w", (I,)),),
        )
        for change in ("undefined", "missing_payload", "unknown_resource", "extra_field"):
            wire = json.loads(encode_boundary(resource))
            publish = wire["entry"]["block"]["operations"][1]
            if change == "undefined":
                publish["inputs"][0] = "missing"
            elif change == "missing_payload":
                publish["inputs"] = []
            elif change == "unknown_resource":
                publish["resource"] = "not_w"
            else:
                publish["trust_me"] = True
            yield wire
        for key, value in (("payload", []), ("undeclared", []), ("kind", "unknown")):
            wire = json.loads(encode_boundary(external_endpoints()))
            wire["resources"][0][key] = value
            yield wire

    for wire in tampered():
        payload = json.dumps(wire)
        with pytest.raises(ValueError):
            _core.validate_region_program(payload)
        with pytest.raises((ValueError, KeyError, InvalidProgram)):
            verify(decode_boundary(payload, Program))
    wire = json.loads(encode_boundary(grid_program(3, 4)))
    inner = wire["entry"]["block"]["operations"][0]["body"]["block"]["operations"][1]["body"]["block"]
    inner["operations"][1]["inputs"][0] = "hidden"  # Exists only at the program entry.
    with pytest.raises(ValueError, match="captures must be explicit"):
        _core.validate_region_program(json.dumps(wire))


def test_analysis_rejects_dynamic_coordinates_and_undeclared_schedule_parameters():
    dynamic = Program(
        region(
            (Value("w", BankType(X, 4)), Value("at", X)),
            (
                Select(("w", "at"), (Value("v", X),)),
                Coordinate(("v",), (Value("c", X),), AffineIndex(0, (IndexTerm(0),))),
            ),
            ("c",),
        )
    )
    with pytest.raises(ValueError, match="affine"):
        analyze_region_program(dynamic, optimize=False)
    schedule = "[__size_n,alien] -> { S0[i] -> [i+alien] : 0<=__size_n<=8 and 0<=i<__size_n }"
    for check in (validate_region_schedule, lower_region_schedule, lower_region_bindings):
        with pytest.raises(ValueError, match="undeclared size parameters"):
            check(prefix_program(), schedule)


# ---------------------------------------------------------------------------
# Runtime guards and dynamic semantics
# ---------------------------------------------------------------------------


def observed_prefix_program():
    observe = Function("observe", (), ())
    program = prefix_program()
    program = replace(program, functions=(observe,))
    return with_entry(program, (Call((), (), observe.name), *program.entry.block.operations))


def test_invocation_guards_reject_bad_sizes_and_shapes_before_any_call():
    calls = []
    observe = NativePrimitive(Function("observe", (), ()), lambda inputs, resources: calls.append(1) or ())
    runner = bind(observed_prefix_program(), {"observe": observe})
    for sizes in ({}, {"n": 2, "extra": 1}, {"n": -1}, {"n": 9}, {"n": True}, {"n": 2.0}):
        with pytest.raises(ValueError, match="size parameter"):
            runner.run({"seed": 1, "xs": Bank((1, 2))}, sizes=sizes)
    for sizes in ({"n": 2}, {"n": 0}):
        with pytest.raises(ValueError, match="logical extent"):
            runner.run({"seed": 1, "xs": Bank((1,))}, sizes=sizes)
    assert not calls
    relational = bind(relational_program())
    with pytest.raises(ValueError, match="declared relation"):
        relational.run({"xs": Bank(tuple(range(8)))}, sizes={"prefix": 7, "n": 2})


def test_unselected_branch_is_not_evaluated_and_a_failed_invocation_leaves_no_state():
    bank = BankType(I, 1)
    good = region((Value("b", bank),), (Constant((), (Value("y", I),), 7),), ("y",))
    bad = region(
        (Value("b", bank),), (Constant((), (Value("index", X),), 1), Select(("b", "index"), (Value("y", I),))), ("y",)
    )
    program = Program(region((Value("p", B), Value("b", bank)), (If(("p", "b"), (Value("y", I),), good, bad),), ("y",)))
    execution = bind(program)
    assert execution.run({"p": True, "b": Bank((3,))}).values == (7,)
    with pytest.raises(ValueError, match="out of bounds"):
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
    with pytest.raises(ValueError, match="64-bit"):
        runner.run({"p": True, "x": 2**63 - 1})
    # A zero-limit loop never evaluates its (trapping) condition.
    trap = region(
        (Value("n", I),),
        (
            Constant((), (Value("one", I),), 1),
            Scalar(("n", "one"), (Value("trap", I),), ScalarKind.ADD),
            Constant((), (Value("yes", B),), True),
        ),
        ("yes",),
    )
    loop = Program(region((Value("n", I),), (While(("n",), (Value("out", I),), 0, trap, identity),), ("out",)))
    assert bind(loop).run({"n": 2**63 - 1}).values == (2**63 - 1,)


def test_reordered_tensor_branch_runs_only_the_selected_body():
    cell = doubling_cell()
    tensor = cell.declaration.inputs[0]
    then = region((Value("w", tensor),), (Call(("w",), (Value("next", tensor),), cell.digest),), ("next",))
    other = region((Value("w", tensor),), (), ("w",))
    program = Program(
        region(
            (Value("p", B), Value("x", tensor), Value("y", tensor), Value("z", tensor)),
            (
                Call(("x",), (Value("a", tensor),), cell.digest),
                If(("p", "y"), (Value("b", tensor),), then, other),
                Call(("z",), (Value("c", tensor),), cell.digest),
            ),
            ("a", "b", "c"),
        ),
        (cell.declaration,),
    )
    calls = []

    def observed(value):
        calls.append(int(value[0]))
        return cell.function(value)

    runner = bind(program, {cell.digest: replace(cell, function=observed)}, "{ S0[] -> [2]; S1[] -> [1]; S2[] -> [0] }")
    for predicate in (True, False):
        calls.clear()
        inputs = {"p": predicate, **{n: torch.full((4,), v, dtype=torch.float64) for n, v in zip("xyz", (1, 2, 3))}}
        result = runner.run(inputs)
        assert calls == ([3, 2, 1] if predicate else [3, 1])
        for actual, value in zip(result.values, (2, 4 if predicate else 2, 6), strict=True):
            torch.testing.assert_close(actual, torch.full((4,), value, dtype=torch.float64), rtol=0, atol=0)


def test_values_are_released_at_their_last_use():
    cell = doubling_cell()
    refs, peaks = [], []

    def observed(*args):
        result = cell.function(*args)
        refs.append(weakref.ref(result[0]))
        peaks.append(sum(ref() is not None for ref in refs))
        return result

    tracked = replace(cell, function=observed)
    result = bind(while_program(tracked, 200), {tracked.digest: tracked}).run(
        {"n": 0, "w": torch.ones(4, dtype=torch.float64), "stop": 200}
    )
    assert result.values[0] == 200
    assert max(peaks) <= 4 and sum(ref() is not None for ref in refs) == 1
    # The wavefront frontier, not the whole grid, stays live.
    chunks, layers = 80, 12
    step = mutating_grid_cell()
    grid = tensor_grid(chunks, layers, step)
    parallel = next(c for c in analyze_region_program(grid).candidates if c.source == "isl-parallel")
    run = bind(grid, {step.digest: step}, parallel.schedule_map).run(inputs_for(chunks, layers))
    assert run.peak_live_values < 2 * chunks + 6 * layers
    assert run.released_values > chunks * layers


def test_million_step_domains_stay_symbolic():
    small, large = grid_program(20, 16), grid_program(1_000_000, 16)
    analysis = analyze_region_program(large)
    assert analysis.exact_provenance and len(analysis.statements) == 1
    assert len(analysis.values) == len(analyze_region_program(small).values)
    assert len(analysis.dependencies) < 1000
    assert len(encode_boundary(large)) < 5000
    schedule = analysis.candidates[-1].schedule_map
    sizes = [
        len(encode_boundary(lower_region_bindings(p, analyze_region_program(p).candidates[-1].schedule_map)))
        for p in (small, large)
    ]
    assert sizes[1] < 2 * sizes[0]
    assert len(encode_boundary(lower_region_schedule(large, schedule))) < 5000
    with pytest.raises(ValueError, match="preview instance budget"):
        preview_region_schedule(large, schedule, max_instances=100)
    cell = ExportedCell.capture(lambda h, w: (h + w, w + 1), ("h", "w"), ("nh", "nw"), (torch.ones(4), torch.ones(4)))
    rows = tensor_grid(1_000_000, 16, cell)
    bound = analyze_region_capacity(
        rows, "{ S0[c,l] -> [c+l,c] : 0 <= c < 1000000 and 0 <= l < 16 }", {cell.digest: cell}
    )
    assert bound.devices[0].peak_tensor_bytes_upper_bound <= (2 * 1_000_000 + 20 * 16) * 16
    assert len(encode_boundary(bound)) < 4000
    loop = while_program(doubling_cell(), 1_000_000)
    assert len(encode_boundary(lower_region_schedule(loop, analyze_region_program(loop).reference_schedule))) < 5000
    step = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("out",), (torch.ones(4),))
    scan = TensorScan(step, SizeParameter("steps", 0, 1_000_000), carry=1, invariants=0)
    facts = analyze_region_capacity(
        scan.program, analyze_region_program(scan.program).reference_schedule, {step.digest: step}
    ).devices[0]
    assert facts.peak_tensor_bytes_upper_bound == 48  # Iterations do not accumulate allocations.
    for count in (0, 19):
        inputs = (torch.ones(4),)
        with StorageObserver(inputs) as observed:
            (carry,) = scan(inputs, (), (), count=count).carry
        assert observed.peak - 16 <= facts.peak_tensor_bytes_upper_bound
        torch.testing.assert_close(carry, inputs[0] + count, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Native imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_native_import_copies_inputs_and_returns_independent_storage(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    full = TensorType((4, 4), "torch.float32", device)
    small = TensorType((1, 4), "torch.float32", device)
    saved = []

    def update(inputs, resources):
        (x,) = inputs
        x.add_(1)
        saved.append(resources)
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
    schedule = analyze_region_program(program).reference_schedule
    cells = {native.declaration.name: native}
    bound = analyze_region_capacity(program, schedule, cells).devices[0]
    x = torch.ones((4, 4), device=device)
    with StorageObserver((x,)) as observed:
        same, a, b = bind(program, cells, schedule).run({"x": x}).values
    assert observed.peak - 64 <= bound.peak_tensor_bytes_upper_bound
    assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    torch.testing.assert_close(a, x[:1] + 1, rtol=0, atol=0)
    torch.testing.assert_close(same, x, rtol=0, atol=0)
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="expired"):
        saved[0].snapshot("undeclared")


def test_native_import_must_match_its_source_declaration():
    declaration = Function("native.test", (I,), (I,))
    program = Program(
        region((Value("n", I),), (Call(("n",), (Value("out", I),), declaration.name),), ("out",)), (declaration,)
    )
    with pytest.raises(ValueError, match="source declaration"):
        bind(program, {declaration.name: NativePrimitive(replace(declaration, inputs=(B,)), lambda x, r: x)})

    class UndeclaredCallable:
        declaration = Function("native.test", (I,), (I,))

        def invoke(self, inputs, pages):
            raise AssertionError("unrecognized callable was executed")

    with pytest.raises(ValueError, match="explicit native import"):
        bind(program, {declaration.name: UndeclaredCallable()})


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_adamw_regions_match_the_existing_kernel_bitwise(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from pllm.contracts.tensor_layout import LayoutFieldId
    from pllm.execution.region_adamw import AdamWRegion, adamw_cell
    from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw

    fields = AdamWFields(*(LayoutFieldId(i) for i in range(5)))
    generator = torch.Generator(device=device).manual_seed(37)

    def root(shape, dtype):
        return (
            torch.randn(shape, dtype=dtype, device=device, generator=generator),
            torch.zeros(shape, dtype=dtype, device=device),
            torch.tensor(11.0, device=device),
            torch.randn(shape, dtype=dtype, device=device, generator=generator),
            torch.rand(shape, dtype=dtype, device=device, generator=generator),
        )

    # A five-step scan over one bf16 parameter.
    config = AdamWConfig(0.013, betas=(0.81, 0.97), eps=1e-4, weight_decay=0.07)
    scan = TensorScan(adamw_cell(TensorType((3, 7), "torch.bfloat16", device), config), 5, carry=5, invariants=0)
    initial = root((3, 7), torch.bfloat16)
    gradients = tuple(torch.randn((3, 7), dtype=torch.bfloat16, device=device, generator=generator) for _ in range(5))
    expected, saved = tuple(t.clone() for t in initial), tuple(t.clone() for t in initial)
    for gradient in gradients:
        apply_adamw((expected,), fields, config, (gradient,))
    result = scan(initial, (), (gradients,))
    for actual, want, value, before in zip(result.carry, expected, initial, saved, strict=True):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    # Heterogeneous parameters stay one foreach batch, and never recompile.
    import pllm.execution.region_adamw as imported

    calls = []
    original = imported.apply_adamw
    monkeypatch.setattr(imported, "apply_adamw", lambda *args: calls.append(1) or original(*args))
    optimizer = AdamWRegion(
        (TensorType((2, 7), "torch.bfloat16", device), TensorType((3,), "torch.float32", device)), config
    )
    roots = (root((2, 7), torch.bfloat16), root((3,), torch.float32))
    expected = tuple(tuple(v.clone() for v in r) for r in roots)
    forbid_compilation(monkeypatch)
    for _ in range(3):
        gradients = tuple(torch.randn(r[0].shape, dtype=r[0].dtype, device=device, generator=generator) for r in roots)
        apply_adamw(expected, fields, config, gradients)
        roots = optimizer(roots, gradients)
        for updated, want in zip(roots, expected, strict=True):
            for value, reference in zip(updated, want, strict=True):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert len(calls) == 3

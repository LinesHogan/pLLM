"""Architecture acceptance: reference semantics and rejected illegal programs."""

from dataclasses import replace

import pytest

from pllm.contracts.region_program import (
    BankType,
    Block,
    Call,
    Constant,
    Function,
    If,
    InvalidProgram,
    Map,
    Program,
    Publish,
    Region,
    Resource,
    Scalar,
    ScalarKind,
    ScalarType,
    Scan,
    Select,
    Snapshot,
    Update,
    Value,
    While,
    verify,
)
from pllm.execution.region_compilation import validate_region_program
from test.region_reference import Bank, interpret

I = ScalarType.INT
X = ScalarType.INDEX
B = ScalarType.BOOL


def region(arguments, operations, yields):
    return Region(Block(tuple(arguments), tuple(operations), tuple(yields)))


def grid_program(chunks, layers):
    """The author's chunk-major loop; no dependencies or schedule supplied."""
    weights = BankType(I, layers)
    hidden = BankType(I, chunks)
    layer = region(
        (Value("l", X), Value("h", I), Value("w", weights)),
        (
            Select(("w", "l"), (Value("old", I),)),
            Call(("h", "old"), (Value("next_h", I), Value("next_w", I)), "cell"),
            Update(("w", "l", "next_w"), (Value("bank", weights),)),
        ),
        ("next_h", "bank"),
    )
    chunk = region(
        (Value("c", X), Value("w", weights), Value("h", hidden)),
        (
            Select(("h", "c"), (Value("input", I),)),
            Scan(("input", "w"), (Value("output", I), Value("next_w", weights)), layers, layer),
            Update(("h", "c", "output"), (Value("next_h", hidden),)),
        ),
        ("next_w", "next_h"),
    )
    return Program(
        region(
            (Value("weights", weights), Value("hidden", hidden)),
            (Scan(("weights", "hidden"), (Value("w_final", weights), Value("h_final", hidden)), chunks, chunk),),
            ("w_final", "h_final"),
        ),
        (Function("cell", (I, I), (I, I)),),
    )


@pytest.mark.parametrize("chunks", range(1, 7))
@pytest.mark.parametrize("layers", range(1, 6))
def test_structured_grid_matches_the_original_loop(chunks, layers):
    initial_w = tuple(range(1, layers + 1))
    initial_h = tuple(range(2, chunks + 2))
    weights, hidden = list(initial_w), list(initial_h)
    for c in range(chunks):
        for l in range(layers):
            old_h = hidden[c]
            hidden[c] = old_h * 3 + weights[l]
            weights[l] += old_h
    result = interpret(
        grid_program(chunks, layers),
        {"weights": Bank(initial_w), "hidden": Bank(initial_h)},
        functions={"cell": lambda args: (args[0] * 3 + args[1], args[1] + args[0])},
    )
    assert result.values == (Bank(tuple(weights)), Bank(tuple(hidden)))
    assert result.calls == tuple(("cell", (c, l)) for c in range(chunks) for l in range(layers))


def test_two_states_training_sampling_and_read_only_observation():
    program = Program(
        region(
            (),
            (
                Snapshot((), (Value("wa", I),), "a"),
                Snapshot((), (Value("wb", I),), "b"),
                Call(("wa",), (Value("sample", I),), "sample"),
                Call(("wa",), (Value("ua", I),), "train"),
                Call(("wb",), (Value("ub", I),), "train"),
                Publish(("ua",), (), "a"),
                Publish(("ub",), (), "b"),
            ),
            ("wa", "sample"),
        ),
        (Function("sample", (I,), (I,), ("rng",)), Function("train", (I,), (I,))),
        (Resource("a", (I,)), Resource("b", (I,)), Resource("rng", (I,))),
    )
    result = interpret(
        program,
        {},
        resources={"a": (2,), "b": (5,), "rng": (0,)},
        functions={"sample": lambda args: (args[0] * 10, args[1] + 1), "train": lambda args: (args[0] + 1,)},
    )
    assert result.values == (2, 20)  # The old immutable snapshot remains observable.
    assert result.resources == {"a": (3,), "b": (6,), "rng": (1,)}
    assert [(e.kind, e.resource) for e in result.effects] == [
        ("snapshot", "a"),
        ("snapshot", "b"),
        ("sample", "rng"),
        ("publish", "a"),
        ("publish", "b"),
    ]


@pytest.mark.parametrize("bound, expected", [(0, 0), (2, 2), (10, 3)])
def test_bounded_replay_updates_kv_and_stops_on_predicate(bound, expected):
    condition = region(
        (Value("n", I),),
        (
            Constant((), (Value("threshold", I),), 3),
            Scalar(("n", "threshold"), (Value("again", B),), ScalarKind.LESS),
        ),
        ("again",),
    )
    body = region(
        (Value("n", I),),
        (
            Constant((), (Value("one", I),), 1),
            Scalar(("n", "one"), (Value("next", I),), ScalarKind.ADD),
            Snapshot((), (Value("old", I),), "kv"),
            Scalar(("old", "one"), (Value("new", I),), ScalarKind.ADD),
            Publish(("new",), (), "kv"),
        ),
        ("next",),
    )
    program = Program(
        region(
            (),
            (
                Constant((), (Value("zero", I),), 0),
                While(("zero",), (Value("steps", I),), bound, condition, body),
            ),
            ("steps",),
        ),
        resources=(Resource("kv", (I,)),),
    )
    result = interpret(program, {}, resources={"kv": (17,)})
    assert result.values == (expected,)
    assert result.resources["kv"] == (17 + expected,)
    assert len(result.effects) == expected * 2


@pytest.mark.parametrize("take_update", [False, True])
def test_branch_preserves_state_and_route_continuation(take_update):
    then = region(
        (),
        (
            Snapshot((), (Value("w", I),), "route"),
            Constant((), (Value("one", I),), 1),
            Scalar(("w", "one"), (Value("next", I),), ScalarKind.ADD),
            Publish(("next",), (), "route"),
        ),
        (),
    )
    otherwise = region((), (), ())
    program = Program(
        region((Value("take", B),), (If(("take",), (), then, otherwise),), ()), resources=(Resource("route", (I,)),)
    )
    first = interpret(program, {"take": take_update}, resources={"route": (10,)})
    second = interpret(program, {"take": True}, resources=first.resources)
    assert second.resources["route"] == (11 + int(take_update),)
    assert len(first.effects) == 2 * int(take_update)


def test_unselected_branch_is_not_speculated():
    bank = BankType(I, 1)
    bad = region(
        (Value("w", bank),),
        (
            Constant((), (Value("out_of_bounds", X),), 4),
            Select(("w", "out_of_bounds"), (Value("result", I),)),
        ),
        ("result",),
    )
    good = region((Value("w", bank),), (Constant((), (Value("result", I),), 9),), ("result",))
    program = Program(
        region(
            (Value("take", B), Value("w", bank)), (If(("take", "w"), (Value("result", I),), bad, good),), ("result",)
        )
    )
    assert interpret(program, {"take": False, "w": Bank((1,))}).values == (9,)
    with pytest.raises(ValueError, match="out of bounds"):
        interpret(program, {"take": True, "w": Bank((1,))})


@pytest.mark.parametrize("count", [0, 1, 7])
def test_map_collects_independent_results_without_loop_unrolling(count):
    body = region((Value("i", X),), (Scalar(("i", "i"), (Value("square", X),), ScalarKind.MUL),), ("square",))
    program = Program(region((), (Map((), (Value("squares", BankType(X, count)),), count, body),), ("squares",)))
    assert interpret(program, {}).values == (Bank(tuple(i * i for i in range(count))),)
    assert len(program.entry.block.operations) == 1


def test_hidden_capture_and_duplicate_definitions_are_rejected():
    hidden = region((Value("i", X), Value("carry", I)), (), ("outside",))
    program = Program(
        region((Value("outside", I),), (Scan(("outside",), (Value("result", I),), 1, hidden),), ("result",))
    )
    with pytest.raises(InvalidProgram, match="captures must be explicit"):
        verify(program)
    duplicate = Program(region((Value("a", I),), (Constant((), (Value("a", I),), 2),), ("a",)))
    with pytest.raises(InvalidProgram, match="duplicate"):
        verify(duplicate)


def test_loop_carry_and_branch_result_types_must_match():
    wrong = region((Value("i", X), Value("carry", I)), (), ("i",))
    program = Program(region((Value("a", I),), (Scan(("a",), (Value("b", I),), 2, wrong),), ("b",)))
    with pytest.raises(InvalidProgram, match="signature mismatch"):
        verify(program)
    left = region((), (Constant((), (Value("x", I),), 1),), ("x",))
    right = region((), (Constant((), (Value("x", B),), True),), ("x",))
    program = Program(region((Value("p", B),), (If(("p",), (Value("r", I),), left, right),), ("r",)))
    with pytest.raises(InvalidProgram, match="signature mismatch"):
        verify(program)


def test_effects_cannot_be_hidden_in_map():
    body = region((Value("i", X),), (Snapshot((), (Value("x", I),), "w"),), ())
    program = Program(region((), (Map((), (), 2, body),), ()), resources=(Resource("w", (I,)),))
    with pytest.raises(InvalidProgram, match="pure region"):
        verify(program)


def test_reference_guards_reject_wrong_inputs_before_calls():
    program = grid_program(2, 3)
    called = []
    with pytest.raises(ValueError, match="logical extent"):
        interpret(
            program,
            {"weights": Bank((1,)), "hidden": Bank((1, 2))},
            functions={"cell": lambda args: called.append(args) or args},
        )
    assert not called
    with pytest.raises(InvalidProgram, match="unknown function"):
        verify(replace(program, functions=()))


def test_indexed_resource_family_preserves_each_cells_publication_scope():
    body = region(
        (Value("l", X),),
        (
            Snapshot(("l",), (Value("w", I),), "layers"),
            Constant((), (Value("one", I),), 1),
            Scalar(("w", "one"), (Value("next_w", I),), ScalarKind.ADD),
            Publish(("next_w", "l"), (), "layers"),
        ),
        (),
    )
    program = Program(region((), (Scan((), (), 4, body),), ()), resources=(Resource("layers", (I,), (4,)),))
    validate_region_program(program)
    result = interpret(program, {}, resources={"layers": (Bank((2, 4, 8, 16)),)})
    assert result.resources["layers"] == (Bank((3, 5, 9, 17)),)
    assert [e.indices for e in result.effects if e.kind == "publish"] == [(0,), (1,), (2,), (3,)]
    # An atomic bank is a different algorithmic publication scope, even with
    # identical physical contents. It cannot silently accept cell publications.
    atomic = replace(program, resources=(Resource("layers", (BankType(I, 4),)),))
    with pytest.raises(InvalidProgram, match="signature mismatch"):
        verify(atomic)
    with pytest.raises(ValueError, match="signature mismatch"):
        validate_region_program(atomic)


def test_resource_family_bounds_are_checked_before_publication():
    program = Program(
        region((Value("i", X), Value("j", X), Value("x", I)), (Publish(("x", "i", "j"), (), "grid"),), ()),
        resources=(Resource("grid", (I,), (2, 2)),),
    )
    state = {"grid": (Bank((Bank((1, 2)), Bank((3, 4)))),)}
    result = interpret(program, {"i": 1, "j": 0, "x": 9}, resources=state)
    assert result.resources["grid"] == (Bank((Bank((1, 2)), Bank((9, 4)))),)
    with pytest.raises(ValueError, match="out of bounds"):
        interpret(program, {"i": 1, "j": 3, "x": 9}, resources=state)
    assert state["grid"] == (Bank((Bank((1, 2)), Bank((3, 4)))),)


def test_external_effect_contents_are_observable_by_a_later_snapshot():
    program = Program(
        region(
            (),
            (
                Call((), (Value("sample", I),), "sample"),
                Snapshot((), (Value("after", I),), "rng"),
            ),
            ("sample", "after"),
        ),
        (Function("sample", (), (I,), ("rng",)),),
        (Resource("rng", (I,)),),
    )
    result = interpret(
        program, {}, resources={"rng": (7,)}, functions={"sample": lambda args: (args[0] * 3, args[0] + 1)}
    )
    assert result.values == (21, 8)
    assert result.resources["rng"] == (8,)
    with pytest.raises(ValueError, match="arity"):
        interpret(program, {}, resources={"rng": (7,)}, functions={"sample": lambda args: (21,)})


def test_coordinates_declare_affine_indices_and_reject_malformed_terms():
    import json

    from pllm.contracts.codec import _converter
    from pllm.contracts.region_program import AffineIndex, Coordinate, IndexTerm, Op

    bank = BankType(I, 8)
    declared = Program(
        region(
            (Value("w", bank),),
            (
                Coordinate((), (Value("two", X),), AffineIndex(2)),
                Coordinate(("two",), (Value("twice", X),), AffineIndex(0, (IndexTerm(0, 2),))),
                Select(("w", "twice"), (Value("x", I),)),
            ),
            ("x",),
        )
    )
    verify(declared)
    op = declared.entry.block.operations[1]
    wire = json.loads(json.dumps(_converter.unstructure(op, Op)))
    assert _converter.structure(wire, Op) == op
    assert interpret(declared, {"w": Bank(tuple(range(8)))}).values == (4,)

    def coordinate(constant, terms, inputs=("two",), result=X):
        return Coordinate(inputs, (Value("c", result),), AffineIndex(constant, terms))

    base = (Coordinate((), (Value("two", X),), AffineIndex(2)),)
    for malformed in (
        coordinate(0, (IndexTerm(1),)),  # term past the input list
        coordinate(0, (IndexTerm(0, 0),)),  # zero coefficient
        coordinate(0, (IndexTerm(0), IndexTerm(0, 2))),  # duplicate argument
        coordinate(0, (IndexTerm(0),), result=I),  # non-index result
    ):
        with pytest.raises(InvalidProgram):
            verify(Program(region((), (*base, malformed), ("two",))))
    # Non-index operands are rejected as well.
    boolean = Program(
        region((Value("flag", B),), (Coordinate(("flag",), (Value("c", X),), AffineIndex(0, (IndexTerm(0),))),), ("c",))
    )
    with pytest.raises(InvalidProgram):
        verify(boolean)


def test_coordinate_operands_must_stay_affine_at_analysis():
    from pllm.contracts.region_program import AffineIndex, Coordinate, IndexTerm
    from pllm.execution.region_compilation import analyze_region_program

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

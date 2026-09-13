"""Dependences come from source SSA, including versions and resource addresses."""

from dataclasses import replace

import pytest

from pllm.contracts.region_program import (
    AffineIndex, BankType, Block, Call, Constant, Coordinate, Function, If, IndexTerm, Program, Publish,
    Region, Resource, Scalar, ScalarKind, ScalarType, Scan, Select, Snapshot, Update, Value,
)
from pllm.execution.region_compilation import (
    analyze_region_program, preview_region_schedule, validate_region_schedule,
)
from test.boundary.test_region_program import grid_program, region


I, X, B = ScalarType.INT, ScalarType.INDEX, ScalarType.BOOL


def evaluate_grid(instances, chunks, layers):
    weights = list(range(1, layers + 1))
    hidden = [[c + 2] + [None] * layers for c in range(chunks)]
    observed = {}
    for instance in instances:
        c, l = instance.coordinates
        h, w = hidden[c][l], weights[l]
        assert h is not None, "scheduler ran a consumer before its hidden producer"
        observed[c, l] = (h, w)
        hidden[c][l + 1] = h * 3 + w
        weights[l] = w + h
    return hidden, weights, observed


@pytest.mark.parametrize("chunks", range(1, 7))
@pytest.mark.parametrize("layers", range(1, 6))
def test_source_derived_wavefront_preserves_every_cell_input(chunks, layers):
    program = grid_program(chunks, layers)
    analysis = analyze_region_program(program)
    assert analysis.exact_provenance
    assert len(analysis.statements) == 1
    assert not analysis.diagnostics
    serial = preview_region_schedule(program, analysis.reference_schedule)
    reference = evaluate_grid(serial, chunks, layers)
    for candidate in analysis.candidates:
        instances = preview_region_schedule(program, candidate.schedule_map)
        assert len(instances) == chunks * layers
        assert evaluate_grid(instances, chunks, layers) == reference
        if candidate.source == "isl-parallel" and chunks > 1 and layers > 1:
            assert len({instance.time[0] for instance in instances}) == chunks + layers - 1
            if layers > 2:
                assert [x.coordinates for x in instances] != [x.coordinates for x in serial]


def test_large_iteration_domain_stays_symbolic():
    program = grid_program(1_000_000, 16)
    analysis = analyze_region_program(program)
    assert len(analysis.statements) == 1
    assert len(analysis.values) == len(analyze_region_program(grid_program(2, 2)).values)
    assert analysis.exact_provenance
    assert len(analysis.dependencies) < 1000
    assert len(analysis.candidates) == 3
    with pytest.raises(ValueError, match="preview instance budget"):
        preview_region_schedule(program, analysis.candidates[-1].schedule_map, max_instances=100)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_zero_trip_loops_have_no_scheduled_instances(shape):
    program = grid_program(*shape)
    analysis = analyze_region_program(program)
    assert preview_region_schedule(program, analysis.reference_schedule, max_instances=0) == ()


@pytest.mark.parametrize("schedule", [
    "{ S0[c,l] -> [-c,l] : 0<=c<3 and 0<=l<3 }",
    "{ S0[c,l] -> [0,0] : 0<=c<3 and 0<=l<3 }",
    "{ S0[c,l] -> [c,l] : 0<=c<2 and 0<=l<3 }",
    "{ S0[c,l] -> [t,l] : 0<=c<3 and 0<=l<3 and c<=t<=c+1 }",
])
def test_missing_collapsed_reversed_and_multivalued_witnesses_are_rejected(schedule):
    with pytest.raises(ValueError):
        validate_region_schedule(grid_program(3, 3), schedule)


def test_old_bank_reads_do_not_accidentally_become_reads_of_the_new_version():
    bank = BankType(I, 2)
    program = Program(region((Value("w", bank),), (
        Coordinate((), (Value("zero", X),), AffineIndex(0)),
        Select(("w", "zero"), (Value("old", I),)),
        Call(("old",), (Value("new", I),), "update"),
        Update(("w", "zero", "new"), (Value("next_bank", bank),)),
        Call(("w",), (Value("observed", I),), "observe"),
    ), ("next_bank", "observed")), (Function("update", (I,), (I,)), Function("observe", (bank,), (I,))))
    reversed_order = "{ S0[] -> [1]; S1[] -> [0] }"
    validate_region_schedule(program, reversed_order)  # Observing the old bank is independent.
    block = program.entry.block
    reader = replace(block.operations[-1], inputs=("next_bank",))
    new_read = replace(program, entry=Region(replace(block, operations=(*block.operations[:-1], reader))))
    with pytest.raises(ValueError, match="dependency"):
        validate_region_schedule(new_read, reversed_order)


def test_discarding_state_updates_changes_the_derived_dependencies():
    program = grid_program(3, 3)
    outer = program.entry.block.operations[0]
    inner = outer.body.block.operations[1]
    inner = replace(inner, body=Region(replace(inner.body.block, yields=("next_h", "w"))))
    body = replace(outer.body.block, operations=(outer.body.block.operations[0], inner, outer.body.block.operations[2]))
    independent = replace(program, entry=Region(replace(program.entry.block,
        operations=(replace(outer, body=Region(body)),))))
    reverse_chunks = "{ S0[c,l] -> [-c,l] : 0<=c<3 and 0<=l<3 }"
    validate_region_schedule(independent, reverse_chunks)
    with pytest.raises(ValueError, match="dependency"):
        validate_region_schedule(program, reverse_chunks)


def test_final_layer_feedback_prevents_the_grid_wavefront():
    bank = BankType(I, 4)
    base = grid_program(3, 4)
    layer_body = base.entry.block.operations[0].body.block.operations[1].body
    chunk_body = region((Value("c", X), Value("h", I), Value("w", bank)), (
        Scan(("h", "w"), (Value("h_final", I), Value("w_final", bank)), 4, layer_body),
    ), ("h_final", "w_final"))
    program = Program(region((Value("input", I), Value("weights", bank)), (
        Scan(("input", "weights"), (Value("output", I), Value("result", bank)), 3, chunk_body),
    ), ("output", "result")), base.functions)
    analysis = analyze_region_program(program)
    for candidate in analysis.candidates:
        validate_region_schedule(program, candidate.schedule_map)
    with pytest.raises(ValueError, match="dependency"):
        validate_region_schedule(program, "{ S0[c,l] -> [c+l,c] : 0<=c<3 and 0<=l<4 }")


def resource_loop(indexed):
    indices = ("i",) if indexed else ()
    body = region((Value("i", X),), (
        Snapshot(indices, (Value("w", I),), "state"),
        Call(("w",), (Value("next_w", I),), "cell"),
        Publish(("next_w", *indices), (), "state"),
    ), ())
    return Program(region((), (
        Scan((), (), 4, body),
    ), ()), (Function("cell", (I,), (I,)),), (Resource("state", (I,), (4,) if indexed else ()),))


def test_resource_indices_allow_independent_publications_without_dropping_atomicity():
    phase_order = "{ S0[i] -> [0,i] : 0<=i<4; S1[i] -> [1,i] : 0<=i<4; S2[i] -> [2,i] : 0<=i<4 }"
    validate_region_schedule(resource_loop(True), phase_order)
    with pytest.raises(ValueError, match="dependency"):
        validate_region_schedule(resource_loop(False), phase_order)


def test_proven_branches_can_move_but_checked_arithmetic_keeps_a_boundary():
    then = region((Value("x", I),), (Call(("x",), (Value("out", I),), "f"),), ("out",))
    other = region((Value("x", I),), (), ("x",))
    program = Program(region((Value("x", I), Value("p", B)), (
        Call(("x",), (Value("a", I),), "f"),
        If(("p", "x"), (Value("b", I),), then, other),
        Call(("x",), (Value("c", I),), "f"),
    ), ("a", "b", "c")), (Function("f", (I,), (I,)),))
    analysis = analyze_region_program(program)
    assert not any(s.opaque for s in analysis.statements)
    assert analysis.statements[1].path == (1,)
    reverse = "{ S0[] -> [2]; S1[] -> [1]; S2[] -> [0] }"
    validate_region_schedule(program, reverse)
    checked = region((Value("x", I),), (
        Constant((), (Value("one", I),), 1),
        Scalar(("x", "one"), (Value("unused", I),), ScalarKind.ADD),
    ), ("x",))
    block = program.entry.block
    guarded = replace(program, entry=replace(program.entry, block=replace(block,
        operations=(block.operations[0], replace(block.operations[1], then_region=checked), block.operations[2]))))
    assert [s.opaque for s in analyze_region_program(guarded).statements] == [False, True, False]
    with pytest.raises(ValueError, match="dependency"):
        validate_region_schedule(guarded, reverse)


def test_affine_scaled_indices_are_analyzed_but_unknown_indices_stay_opaque():
    bank = BankType(I, 8)
    body = region((Value("i", X), Value("w", bank)), (
        Coordinate(("i",), (Value("even", X),), AffineIndex(0, (IndexTerm(0, 2),))),
        Select(("w", "even"), (Value("x", I),)),
        Call(("x",), (Value("next", I),), "f"),
        Update(("w", "even", "next"), (Value("out", bank),)),
    ), ("out",))
    program = Program(region((Value("w", bank),), (Scan(("w",), (Value("out", bank),), 4, body),), ("out",)),
                      (Function("f", (I,), (I,)),))
    analysis = analyze_region_program(program)
    assert len(analysis.statements) == 1 and not analysis.statements[0].opaque
    validate_region_schedule(program, "{ S0[i] -> [-i] : 0<=i<4 }")
    unknown = Program(region((Value("w", bank), Value("index", X)), (
        Select(("w", "index"), (Value("x", I),)),
    ), ("x",)))
    assert analyze_region_program(unknown).statements[0].opaque


def test_unproven_resource_index_cannot_move_after_an_unrelated_publication():
    program = Program(region((Value("index", X), Value("new", I)), (
        Snapshot(("index",), (Value("old", I),), "a"),
        Publish(("new",), (), "b"),
    ), ()), resources=(Resource("a", (I,), (3,)), Resource("b", (I,))))
    assert analyze_region_program(program).statements[0].opaque
    with pytest.raises(ValueError, match="dependency"):
        validate_region_schedule(program, "{ S0[] -> [1]; S1[] -> [0] }")


def test_potential_integer_overflow_preserves_its_order_relative_to_effects():
    program = Program(region((Value("x", I), Value("one", I)), (
        Scalar(("x", "one"), (Value("result", I),), ScalarKind.ADD),
        Publish(("one",), (), "w"),
    ), ("result",)), resources=(Resource("w", (I,)),))
    assert analyze_region_program(program).statements[0].opaque
    with pytest.raises(ValueError, match="dependency"):
        validate_region_schedule(program, "{ S0[] -> [1]; S1[] -> [0] }")

"""Native-lowered dynamic regions preserve branch, carry and exception semantics."""

import weakref
from dataclasses import replace
from math import copysign

import pytest
import torch

from pllm.contracts.control_instructions import ControlWhile
from pllm.contracts.region_program import (
    BankType,
    Call,
    Constant,
    If,
    Map,
    Program,
    Scalar,
    ScalarKind,
    ScalarType,
    Scan,
    Select,
    SizeParameter,
    Value,
    While,
)
from pllm.contracts.schedule_instructions import LoweredSchedule
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_values import Bank
from test.boundary.test_region_program import region
from test.region_reference import interpret

I, X, B = ScalarType.INT, ScalarType.INDEX, ScalarType.BOOL


def bind(program, cells=None):
    schedule = analyze_region_program(program).candidates[-1].schedule_map
    lowered = lower_region_schedule(program, schedule)
    assert decode_boundary(encode_boundary(lowered), LoweredSchedule) == lowered
    return RegionExecution(program, lowered, lower_region_bindings(program, schedule), cells or {})


def test_literals_use_plan_values_including_empty_maps_and_negative_zero():
    floating = ScalarType.FLOAT
    column = BankType(floating, "n")
    body = region(
        (Value("i", X), Value("scale", floating)),
        (Constant((), (Value("negative_zero", floating),), -0.0),),
        ("scale", "negative_zero"),
    )
    program = Program(
        region(
            (),
            (
                Constant((), (Value("yes", B),), True),
                Constant((), (Value("no", B),), False),
                Constant((), (Value("quarter", floating),), 0.25),
                Constant((), (Value("zero", floating),), -0.0),
                Constant((), (Value("integer", I),), 7),
                Map(("quarter",), (Value("quarters", column), Value("zeros", column)), "n", body),
            ),
            ("yes", "no", "quarter", "zero", "integer", "quarters", "zeros"),
        ),
        parameters=(SizeParameter("n", 0, 3),),
    )
    execution = bind(program)
    for count in (0, 3):
        actual = execution.run({}, sizes={"n": count}).values
        assert actual == (True, False, 0.25, -0.0, 7, Bank((0.25,) * count), Bank((-0.0,) * count))
        assert tuple(type(value) for value in actual[:5]) == (bool, bool, float, float, int)
        assert copysign(1.0, actual[3]) == -1.0
        assert all(copysign(1.0, value) == -1.0 for value in actual[-1].elements)


@pytest.fixture(scope="module")
def cell():
    return ExportedCell.capture(lambda w: (w * 2,), ("w",), ("next",), (torch.ones(4, dtype=torch.float64),))


def test_unselected_branch_does_not_read_out_of_bounds():
    bank = BankType(I, 1)
    good = region((Value("b", bank),), (Constant((), (Value("y", I),), 7),), ("y",))
    bad = region(
        (Value("b", bank),),
        (
            Constant((), (Value("index", X),), 1),
            Select(("b", "index"), (Value("y", I),)),
        ),
        ("y",),
    )
    program = Program(region((Value("p", B), Value("b", bank)), (If(("p", "b"), (Value("y", I),), good, bad),), ("y",)))
    execution = bind(program)
    assert execution.run({"p": True, "b": Bank((3,))}).values == (7,)
    with pytest.raises(ValueError, match="out of bounds"):
        execution.run({"p": False, "b": Bank((3,))})
    assert execution.run({"p": True, "b": Bank((3,))}).values == (7,)


def test_reordered_tensor_branch_still_executes_only_the_selected_body(cell):
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

    tracked = replace(cell, function=observed)
    schedule = "{ S0[] -> [2]; S1[] -> [1]; S2[] -> [0] }"
    runner = RegionExecution(
        program,
        lower_region_schedule(program, schedule),
        lower_region_bindings(program, schedule),
        {cell.digest: tracked},
    )
    for predicate in (True, False):
        calls.clear()
        inputs = {
            "p": predicate,
            **{name: torch.full((4,), value, dtype=torch.float64) for name, value in zip(("x", "y", "z"), (1, 2, 3))},
        }
        result = runner.run(inputs)
        assert calls == ([3, 2, 1] if predicate else [3, 1])
        for actual, value in zip(result.values, (2, 4 if predicate else 2, 6), strict=True):
            torch.testing.assert_close(actual, torch.full((4,), value, dtype=torch.float64), rtol=0, atol=0)


def while_program(cell, limit):
    tensor = cell.declaration.inputs[0]
    condition = region(
        (Value("n", I), Value("w", tensor), Value("stop", I)),
        (Scalar(("n", "stop"), (Value("again", B),), ScalarKind.LESS),),
        ("again",),
    )
    body = region(
        (Value("n", I), Value("w", tensor), Value("stop", I)),
        (
            Call(("w",), (Value("next_w", tensor),), cell.digest),
            Constant((), (Value("one", I),), 1),
            Scalar(("n", "one"), (Value("next_n", I),), ScalarKind.ADD),
        ),
        ("next_n", "next_w", "stop"),
    )
    return Program(
        region(
            (Value("n", I), Value("w", tensor), Value("stop", I)),
            (
                While(
                    ("n", "w", "stop"),
                    (Value("final_n", I), Value("final_w", tensor), Value("stop_out", I)),
                    limit,
                    condition,
                    body,
                ),
            ),
            ("final_n", "final_w"),
        ),
        (cell.declaration,),
    )


@pytest.mark.parametrize("limit,start,stop", [(0, 0, 3), (1, 0, 3), (7, 0, 3), (7, 4, 3), (5, -2, 3)])
def test_pretest_loop_preserves_count_carry_and_exported_tensors(cell, limit, start, stop):
    program = while_program(cell, limit)
    inputs = {"n": start, "w": torch.arange(4, dtype=torch.float64), "stop": stop}
    expected = interpret(program, inputs, functions={cell.digest: lambda args: cell(*args)})
    result = bind(program, {cell.digest: cell}).run(inputs)
    assert result.values[0] == expected.values[0]
    torch.testing.assert_close(result.values[1], expected.values[1], rtol=0, atol=0)
    torch.testing.assert_close(inputs["w"], torch.arange(4, dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("trips,width", [(0, 3), (3, 0), (3, 4)])
def test_scan_and_map_nested_inside_branch_keep_structural_bodies(cell, trips, width):
    tensor = cell.declaration.inputs[0]
    bank = BankType(tensor, width)
    element = region(
        (Value("j", X), Value("b", bank)),
        (
            Select(("b", "j"), (Value("w", tensor),)),
            Call(("w",), (Value("next", tensor),), cell.digest),
        ),
        ("next",),
    )
    scan_body = region(
        (Value("i", X), Value("b", bank)), (Map(("b",), (Value("next", bank),), width, element),), ("next",)
    )
    branch = region((Value("b", bank),), (Scan(("b",), (Value("result", bank),), trips, scan_body),), ("result",))
    identity = region((Value("b", bank),), (), ("b",))
    program = Program(
        region(
            (Value("p", B), Value("b", bank)),
            (If(("p", "b"), (Value("result", bank),), branch, identity),),
            ("result",),
        ),
        (cell.declaration,),
    )
    execution = bind(program, {cell.digest: cell})
    for enabled in (False, True):
        original = Bank(tuple(torch.full((4,), i + 1.0, dtype=torch.float64) for i in range(width)))
        result = execution.run({"p": enabled, "b": original})
        for got, want in zip(result.values[0].elements, original.elements, strict=True):
            torch.testing.assert_close(got, want * (2**trips if enabled else 1), rtol=0, atol=0)


def test_checked_unused_arithmetic_is_not_eliminated_from_a_dynamic_region():
    branch = region(
        (Value("n", I),),
        (
            Constant((), (Value("one", I),), 1),
            Scalar(("n", "one"), (Value("unused", I),), ScalarKind.ADD),
        ),
        ("n",),
    )
    identity = region((Value("n", I),), (), ("n",))
    program = Program(
        region((Value("p", B), Value("n", I)), (If(("p", "n"), (Value("n_out", I),), branch, identity),), ("n_out",))
    )
    execution = bind(program)
    assert execution.run({"p": False, "n": 2**63 - 1}).values == (2**63 - 1,)
    with pytest.raises(ValueError, match="64-bit"):
        execution.run({"p": True, "n": 2**63 - 1})


def test_zero_limit_does_not_evaluate_the_condition():
    condition = region(
        (Value("n", I),),
        (
            Constant((), (Value("one", I),), 1),
            Scalar(("n", "one"), (Value("trap", I),), ScalarKind.ADD),
            Constant((), (Value("yes", B),), True),
        ),
        ("yes",),
    )
    body = region((Value("n", I),), (), ("n",))
    program = Program(region((Value("n", I),), (While(("n",), (Value("out", I),), 0, condition, body),), ("out",)))
    assert bind(program).run({"n": 2**63 - 1}).values == (2**63 - 1,)


def test_large_dynamic_bound_is_compiled_once_and_stops_on_first_false(cell):
    program = while_program(cell, 1_000_000)
    lowered = lower_region_schedule(program, analyze_region_program(program).reference_schedule)
    assert len(lowered.actions) == 1
    assert isinstance(lowered.actions[0], ControlWhile)
    assert len(encode_boundary(lowered)) < 5000
    result = bind(program, {cell.digest: cell}).run({"n": 0, "w": torch.ones(4, dtype=torch.float64), "stop": 0})
    assert result.values[0] == 0


def test_dynamic_loop_does_not_retain_previous_iteration_tensors(cell):
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
    assert max(peaks) <= 4
    assert sum(ref() is not None for ref in refs) == 1


def dynamic_page_program(cell, limit):
    from pllm.contracts.region_program import Publish, Resource, Snapshot

    tensor = cell.declaration.inputs[0]
    args = (Value("n", I), Value("stop", I))
    condition = region(
        (Value("n", I), Value("stop", I)), (Scalar(("n", "stop"), (Value("again", B),), ScalarKind.LESS),), ("again",)
    )
    body = region(
        args,
        (
            Constant((), (Value("zero", X),), 0),
            Constant((), (Value("one", I),), 1),
            Snapshot(("zero",), (Value("w", tensor),), "layers"),
            Call(("w",), (Value("next_w", tensor),), cell.digest),
            Publish(("next_w", "zero"), (), "layers"),
            Scalar(("n", "one"), (Value("next_n", I),), ScalarKind.ADD),
        ),
        ("next_n", "stop"),
    )
    branch = region(
        args,
        (While(("n", "stop"), (Value("end_n", I), Value("end_stop", I)), limit, condition, body),),
        ("end_n", "end_stop"),
    )
    identity = region(args, (), ("n", "stop"))
    return Program(
        region(
            (*args, Value("enabled", B)),
            (
                Constant((), (Value("zero", X),), 0),
                Snapshot(("zero",), (Value("old", tensor),), "layers"),
                If(("enabled", "n", "stop"), (Value("end_n", I), Value("end_stop", I)), branch, identity),
                Snapshot(("zero",), (Value("final", tensor),), "layers"),
            ),
            ("end_n", "old", "final"),
        ),
        (cell.declaration,),
        (Resource("layers", (tensor,), (2,)),),
    )


def rebind_dynamic_task(task, *, limit, inputs):
    (cell,) = task.region._cells.values()
    program = dynamic_page_program(cell, limit)
    task.region, task.program, task.inputs = bind(program, {cell.digest: cell}), program, inputs
    return cell


@pytest.mark.parametrize(
    "limit,start,stop,enabled", [(0, 0, 4, True), (2, 0, 4, True), (8, 0, 4, True), (8, 0, 4, False), (8, 5, 4, True)]
)
def test_branch_and_while_use_original_ptts_owner_with_data_carry(limit, start, stop, enabled):
    from test.boundary.test_region_pages import owner_fixture

    execution, task, rows, refs = owner_fixture(rounds=1)
    inputs = {"n": start, "stop": stop, "enabled": enabled}
    cell = rebind_dynamic_task(task, limit=limit, inputs=inputs)
    shape = cell.input_signatures[0].shape
    expected = interpret(
        task.program,
        inputs,
        resources={"layers": (Bank((torch.ones(shape), torch.full(shape, 2.0))),)},
        functions={cell.digest: lambda args: cell(*args)},
    )
    final = execution.step(rows)
    assert task.result.values[0] == expected.values[0]
    for got, want in zip(task.result.values[1:], expected.values[1:], strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    steps = min(limit, max(0, stop - start)) if enabled else 0
    assert tuple(ref.version_id for ref in final) == (steps, 0)
    assert len(task.context.page_publications) == steps
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0


def test_dynamic_cancellation_keeps_first_commit_and_stops_later_iterations():
    from pllm.execution.state_execution import StateSettlementError
    from test.boundary.test_region_pages import owner_fixture

    execution, task, rows, _ = owner_fixture(rounds=1)
    (original,) = task.region._cells.values()
    calls = []

    def cancel(*args):
        calls.append(1)
        if len(calls) == 2:
            task.context.cancel_requested.set()
        return original.function(*args)

    tracked = replace(original, function=cancel)
    task.program = dynamic_page_program(tracked, 8)
    task.region = bind(task.program, {tracked.digest: tracked})
    task.inputs = {"n": 0, "stop": 6, "enabled": True}
    invocation = execution.start_step(rows)
    with pytest.raises(StateSettlementError) as failure:
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert failure.value.receipt is invocation.context.publication
    assert len(calls) == 2 and len(invocation.context.page_publications) == 1
    assert not invocation.context.page_writes
    execution.cancel_step(invocation)
    assert all(release is not None for release in invocation.context.page_pins.values())
    execution.retire(tuple(row.identity for row in rows))
    assert execution.arena.active_lineages == 0


def test_dynamic_branch_inside_affine_map_receives_each_instance_coordinates(cell):
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
    inputs = {"w": torch.arange(4, dtype=torch.float64)}
    for candidate in analyze_region_program(program).candidates:
        result = RegionExecution(
            program,
            lower_region_schedule(program, candidate.schedule_map),
            lower_region_bindings(program, candidate.schedule_map),
            {cell.digest: cell},
        ).run(inputs)
        for i, tensor in enumerate(result.values[0].elements):
            torch.testing.assert_close(tensor, inputs["w"] * (2 if i == 2 else 1), rtol=0, atol=0)

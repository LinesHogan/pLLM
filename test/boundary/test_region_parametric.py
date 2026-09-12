"""Parameterized ISL schedules preserve exact tensor sources for every length."""

from dataclasses import replace

import pytest
import torch

from pllm import _core
from pllm.contracts.region_program import (
    BankType, Call, Function, If, Map, Program, ScalarType, Scan, SizeParameter, Value,
)
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import (
    analyze_region_program, lower_region_bindings, lower_region_schedule, validate_region_schedule,
)
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import NativePrimitive
from pllm.execution.region_reference import interpret
from pllm.execution.region_schedule import RegionScheduleExecution
from pllm.execution.region_values import Bank
from pllm.execution.tensor_scan import TensorScan
from test.boundary.test_region_program import region
from test.boundary.test_region_tensor import tensor_grid, inputs_for


def lower(program, schedule, functions):
    return RegionExecution(program, lower_region_schedule(program, schedule),
        lower_region_bindings(program, schedule), functions)


def test_parametric_wavefront_and_tiles_preserve_every_cell_source(monkeypatch):
    cell = ExportedCell.capture(lambda h, w: (h * .5 + w * .25, w + h * .0625),
        ("h", "w"), ("next_h", "next_w"), (torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)))
    program = replace(tensor_grid("chunks", "layers", cell),
        parameters=(SizeParameter("chunks", 0, 80), SizeParameter("layers", 0, 12)))
    analysis = analyze_region_program(program)
    assert analysis.exact_provenance and len(analysis.statements) == 1
    assert not analysis.statements[0].opaque
    prefix = "[__size_chunks,__size_layers] -> "
    guard = "0<=__size_chunks<=80 and 0<=__size_layers<=12 and 0<=c<__size_chunks and 0<=l<__size_layers"
    wave = prefix + "{ S0[c,l] -> [c+l,c] : " + guard + " }"
    tiled = prefix + "{ S0[c,l] -> [floor(c/2),floor(l/3),c,l] : " + guard + " }"
    with pytest.raises(ValueError):
        validate_region_schedule(program, prefix + "{ S0[c,l] -> [-c,l] : " + guard + " }")
    schedules = (*tuple(c.schedule_map for c in analysis.candidates), wave, tiled)
    observed = []
    def step(*args):
        observed.append(tuple(v.clone() for v in args))
        return cell.function(*args)
    runners = tuple(lower(program, s, {cell.digest: replace(cell, function=step)}) for s in schedules)
    cursors = tuple(RegionScheduleExecution(program, lower_region_schedule(program, s)) for s in schedules)
    # Source/AST sizes stay fixed even when the permitted iteration space grows.
    large = replace(program, parameters=(SizeParameter("chunks", 0, 1_000_000), SizeParameter("layers", 0, 12)))
    large_analysis = analyze_region_program(large)
    assert len(large_analysis.values) == len(analysis.values) and len(large_analysis.statements) == 1
    assert large_analysis.exact_provenance
    def forbidden(*args):
        raise AssertionError("parametric execution must not recompile")
    for name in ("analyze_region_program", "lower_region_schedule", "lower_region_bindings"):
        monkeypatch.setattr(_core, name, forbidden)
    for chunks, layers in ((0, 0), (0, 5), (3, 0), (1, 1), (5, 3), (7, 5), (80, 12)):
        sizes = {"chunks": chunks, "layers": layers}
        inputs = inputs_for(chunks, layers)
        oracle_calls = []
        def oracle(args):
            oracle_calls.append(tuple(v.clone() for v in args))
            return cell(*args)
        expected = interpret(program, inputs, sizes=sizes, functions={cell.digest: oracle})
        for runner, cursor in zip(runners, cursors, strict=True):
            observed.clear()
            actual = runner.run(inputs, sizes=sizes)
            assert actual.invocations == chunks * layers
            for point, args in zip(cursor.invocations(sizes), observed, strict=True):
                chunk, layer = point.coordinates
                for value, want in zip(args, oracle_calls[chunk * layers + layer], strict=True):
                    torch.testing.assert_close(value, want, rtol=0, atol=0)
            for bank, wanted in zip(actual.values, expected.values, strict=True):
                for value, want in zip(bank.elements, wanted.elements, strict=True):
                    torch.testing.assert_close(value, want, rtol=0, atol=0)
            if chunks and layers:
                assert actual.peak_live_values < 2 * chunks + 6 * layers


def test_nested_dynamic_action_and_native_bank_signature_use_bound_sizes():
    x = ScalarType.INDEX
    bank = BankType(x, "n")
    signature = Function("identity", (bank,), (bank,))
    primitive = NativePrimitive(signature, lambda inputs, resources: inputs)
    body = region((Value("i", x),), (), ("i",))
    branch = region((), (
        Map((), (Value("xs", bank),), "n", body),
        Call(("xs",), (Value("ys", bank),), signature.name),
    ), ("ys",))
    program = Program(region((Value("p", ScalarType.BOOL),), (
        If(("p",), (Value("result", bank),), branch, branch),
    ), ("result",)), (signature,), parameters=(SizeParameter("n", 0, 7),))
    analysis = analyze_region_program(program)
    # The runtime branch stays one action. Its nested pure identity has no
    # resource effects, so it does not fence unrelated surrounding calls.
    assert len(analysis.statements) == 1 and not analysis.statements[0].opaque
    runner = lower(program, analysis.reference_schedule, {signature.name: primitive})
    for count in (0, 7, 2):
        for predicate in (True, False):
            assert runner.run({"p": predicate}, sizes={"n": count}).values == (Bank(tuple(range(count))),)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_tensor_scan_reuses_one_plan_for_runtime_extents_and_zero_trip(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cell = ExportedCell.capture(lambda w, x: (w + x, w), ("w", "x"), ("next", "old"),
        (torch.ones(3, device=device), torch.zeros(3, device=device)))
    scan = TensorScan(cell, SizeParameter("n", 0, 9), carry=1, invariants=0)
    def forbidden(*args):
        raise AssertionError("runtime scan compilation")
    monkeypatch.setattr(_core, "analyze_region_program", forbidden)
    for count in (9, 0, 2):
        inputs = tuple(torch.full((3,), i + 1., device=device) for i in range(count))
        result = scan((torch.ones(3, device=device),), (), (inputs,))
        torch.testing.assert_close(result.carry[0], torch.full((3,), 1. + sum(range(1, count + 1)), device=device), rtol=0, atol=0)
        assert len(result.columns[0]) == count
    with pytest.raises(ValueError, match="bounds"):
        scan((torch.ones(3, device=device),), (), ((torch.ones(3, device=device),) * 10,))
    with pytest.raises(ValueError, match="signature"):
        scan((torch.ones(3, device=device),), (), ((torch.ones(3, device=device),),), count=True)


def test_parametric_publications_use_original_owner_and_matching_size_binding(monkeypatch):
    from pllm.contracts.tensor_layout import LayoutFieldId
    from pllm.execution.region_resources import RegionResources, ResourcePageCell
    from test.boundary.test_region_pages import owner_fixture, page_program
    execution, task, rows, original = owner_fixture(rounds=1)
    cell, = task.region._cells.values()
    program = replace(page_program(cell, rounds="rounds"), parameters=(SizeParameter("rounds", 0, 5),))
    analysis = analyze_region_program(program)
    runner = lower(program, analysis.candidates[-1].schedule_map, {cell.digest: cell})
    active_sizes = {"rounds": 0}
    contexts, results = [], []
    def run(context):
        contexts.append(context)
        pages = RegionResources(program, context, tuple(ResourcePageCell("layers", (i,), ref,
            (LayoutFieldId(0),)) for i, ref in enumerate(context.pin.refs)), sizes=active_sizes)
        results.append(runner.run({}, pages=pages, sizes=active_sizes))
    monkeypatch.setattr(task, "execute", run)
    total = 0
    for count in (0, 5, 2):
        active_sizes["rounds"] = count
        final = execution.step(rows)
        context = contexts[-1]
        assert len(context.page_publications) == count * 2
        assert all(context.page_publications.values()) and not context.page_writes
        total += count
        assert tuple(ref.version_id for ref in final) == (total, total)
        for result, expected in zip(results[-1].values[1:], (total + 1., total + 2.), strict=True):
            torch.testing.assert_close(result, torch.full_like(result, expected), rtol=0, atol=0)
    invocation = execution.begin(rows)
    pages = RegionResources(program, invocation.context, tuple(ResourcePageCell("layers", (i,), ref,
        (LayoutFieldId(0),)) for i, ref in enumerate(invocation.context.pin.refs)), sizes={"rounds": 1})
    with pytest.raises(ValueError, match="different size"):
        runner.run({}, pages=pages, sizes={"rounds": 2})
    assert not invocation.context.page_publications and not invocation.context.page_writes
    execution.cancel_step(invocation)
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == execution.arena.usage().pinned_versions == 0
    assert execution.arena.usage().private_versions == execution.arena.usage().pending_receipts == 0


def test_runtime_size_mapping_is_snapshotted_at_invocation_entry():
    x = ScalarType.INDEX
    sizes = {"p0": 3}
    signature = Function("change_caller_mapping", (x,), (x,))
    def change(inputs, resources):
        sizes["p0"] = 0
        return inputs
    primitive = NativePrimitive(signature, change)
    body = region((Value("i", x),), (Call(("i",), (Value("j", x),), signature.name),), ("j",))
    program = Program(region((), (Map((), (Value("result", BankType(x, "p0")),), "p0", body),), ("result",)),
        (signature,), parameters=(SizeParameter("p0", 0, 3),))
    analysis = analyze_region_program(program)
    runner = lower(program, analysis.reference_schedule, {signature.name: primitive})
    assert runner.run({}, sizes=sizes).values == (Bank((0, 1, 2)),)
    assert sizes == {"p0": 0}
    assert runner.run({}, sizes=sizes).values == (Bank(()),)


def test_parameterized_input_shapes_are_checked_before_copying_any_payload(monkeypatch):
    from test.boundary.test_region_sizes import prefix_program
    import pllm.execution.region_frame as frames
    program = prefix_program()
    analysis = analyze_region_program(program)
    runner = lower(program, analysis.reference_schedule, {})
    def copied_too_early(value):
        raise AssertionError("payload copied before all shape guards passed")
    monkeypatch.setattr(frames, "copy_data", copied_too_early)
    with pytest.raises(ValueError, match="logical extent"):
        runner.run({"seed": 1, "xs": Bank((1,))}, sizes={"n": 2})


def test_schedule_cannot_add_a_runtime_parameter_outside_the_source_signature():
    from test.boundary.test_region_sizes import prefix_program
    program = prefix_program()
    schedule = "[__size_n,alien] -> { S0[i] -> [i+alien] : 0<=__size_n<=8 and 0<=i<__size_n }"
    for check in (validate_region_schedule, lower_region_schedule, lower_region_bindings):
        with pytest.raises(ValueError, match="undeclared size parameters"):
            check(program, schedule)

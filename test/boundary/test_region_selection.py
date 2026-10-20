"""Region tuning and selection: measured latency inside hard legality and memory limits.

``select_region_candidate`` ranks legal candidates by measured latency, but only
among those whose symbolic capacity (tensors plus retained recovery storage)
fits the payload limits and whose numerical changes are permitted.
``optimize_region`` and ``TensorScan.optimized`` measure on private inputs and
return an execution that serves every declared size without replanning.
"""

import warnings
from dataclasses import replace

import pytest
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_program import (
    Call,
    Program,
    SizeParameter,
    TensorType,
    Value,
)
from pllm.contracts.region_selection import (
    RegionCandidate,
    RegionPayloadLimit,
    RegionSelectionPolicy,
    RegionSelectionRequest,
    RegionTuningSize,
)
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import (
    analyze_region_program,
    analyze_region_retention,
    explore_region_schedules,
    lower_region_schedule,
)
from pllm.execution.region_optimization import (
    assess_region_candidates,
    optimize_region,
    select_region_candidate,
)
from pllm.execution.region_schedule import RegionScheduleExecution
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.replay import Recurrence
from pllm.execution.tensor_scan import TensorScan
from pllm.execution.tokenwise_region import TokenwiseRegion
from test.region_fixtures import (
    StorageObserver,
    assert_grid_closed_form,
    grid_cell,
    inputs_for,
    reassociation_candidates,
    region,
    single_call,
    tensor_grid,
)


def measured(**changes):
    return replace(ExecutionOptions(**changes), tuning="measured")


def chain_request(shapes=((4, 1), (1, 4), (4, 32))):
    """A source call and its reassociated variant with fixed timing evidence."""
    tensors = tuple(torch.ones(shape, dtype=torch.float64) for shape in shapes)
    cell = ExportedCell.capture(lambda a, b, c: ((a @ b) @ c,), ("a", "b", "c"), ("out",), tensors)
    program = single_call(cell)
    (variant,) = reassociation_candidates(cell)
    schedule = analyze_region_program(program).reference_schedule
    source = import_region_functions(program, {cell.digest: cell})
    rewritten = import_region_functions(program, {cell.digest: variant.cell})
    request = RegionSelectionRequest(
        source,
        (RegionCandidate(schedule, source, (100, 101, 900)), RegionCandidate(schedule, rewritten, (50, 51, 600))),
        (RegionPayloadLimit("cpu", 1 << 20),),
        True,
        "deterministic timing oracle",
    )
    return cell, program, variant, request


def with_times(request, *times):
    return replace(
        request, candidates=tuple(replace(c, elapsed_ns=t) for c, t in zip(request.candidates, times, strict=True))
    )


def test_latency_ranks_only_feasible_permitted_candidates():
    cell, program, variant, request = chain_request()
    selected = select_region_candidate(program, request)
    assert selected.selected == 1  # The faster rewrite wins although it needs more storage.
    assert select_region_candidate(program, with_times(request, (100,), (200, 201, 202))).selected == 0
    source_bytes = selected.candidates[0].capacity.devices[0].peak_tensor_bytes_upper_bound
    limited = select_region_candidate(
        program, replace(request, payload_limits=(RegionPayloadLimit("cpu", source_bytes),))
    )
    assert limited.selected == 0
    assert select_region_candidate(program, replace(request, allow_numerical_transforms=False)).selected == 0
    with pytest.raises(ValueError):
        select_region_candidate(program, replace(request, payload_limits=(RegionPayloadLimit("cpu", 0),)))


def test_fast_measurement_cannot_authorize_an_illegal_schedule():
    cell = grid_cell()
    program = tensor_grid(3, 2, cell)
    imports = import_region_functions(program, {cell.digest: cell})
    request = RegionSelectionRequest(
        imports,
        (
            RegionCandidate(analyze_region_program(program).reference_schedule, imports, (100,)),
            RegionCandidate("{ S0[c,l] -> [-c,-l] : 0 <= c < 3 and 0 <= l < 2 }", imports, (1,)),
        ),
        (RegionPayloadLimit("cpu", 100000),),
        True,
        "illegal schedule oracle",
    )
    with pytest.raises(ValueError):
        select_region_candidate(program, request)


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_cold_tuning_measures_candidates_on_private_inputs(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cell = grid_cell(device)
    inputs = inputs_for(3, 2, device)
    saved = tuple(t.clone() for bank in inputs.values() for t in bank.elements)
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
    cpu_rng = torch.get_rng_state()
    result = optimize_region(
        tensor_grid(3, 2, cell),
        {cell.digest: cell},
        inputs,
        payload_limits=(RegionPayloadLimit(device, 1 << 20),),
        measurement_rounds=2,
        options=RegionSelectionPolicy(tuning="measured"),
    )
    assert all(len(c.elapsed_ns) == 2 and all(t > 0 for t in c.elapsed_ns) for c in result.request.candidates)
    actual = result.execution.run(inputs)
    assert actual.invocations == 6
    assert_grid_closed_form(actual.values, 3, 2)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for before, after in zip(saved, (t for bank in inputs.values() for t in bank.elements), strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_tuning_one_example_size_serves_every_declared_size():
    cell = grid_cell()
    program = replace(
        tensor_grid("chunks", "layers", cell), parameters=(SizeParameter("layers", 0, 5), SizeParameter("chunks", 0, 8))
    )
    selected = optimize_region(
        program,
        {cell.digest: cell},
        inputs_for(2, 3),
        sizes={"layers": 3, "chunks": 2},
        payload_limits=(RegionPayloadLimit("cpu", 10000),),
        warmup_rounds=0,
        measurement_rounds=1,
        max_candidates=3,
        options=RegionSelectionPolicy(tuning="measured"),
    )
    for chunks, layers in ((0, 0), (3, 0), (8, 5), (2, 3), (8, 5), (0, 0)):
        result = selected.execution.run(inputs_for(chunks, layers), sizes={"chunks": chunks, "layers": layers})
        assert result.invocations == chunks * layers
        assert_grid_closed_form(result.values, chunks, layers)
    with pytest.raises(ValueError):
        selected.execution.run(inputs_for(2, 3), sizes={"layers": 3, "chunks": 9})


def test_capacity_is_checked_for_the_full_declared_range_not_the_timing_example():
    cell = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("y",), (torch.ones(4),))
    scan = TensorScan(cell, SizeParameter("steps", 0, 1000), carry=0, invariants=1)
    imports = import_region_functions(scan.program, {cell.digest: cell})
    request = RegionSelectionRequest(
        imports,
        (RegionCandidate(analyze_region_program(scan.program).reference_schedule, imports, (100,)),),
        (RegionPayloadLimit("cpu", 1 << 30),),
        True,
        "parameter timing oracle",
        (RegionTuningSize("steps", 2),),
    )
    (assessment,) = assess_region_candidates(scan.program, request)
    other = replace(request, tuning_sizes=(RegionTuningSize("steps", 999),))
    assert assess_region_candidates(scan.program, other)[0].capacity == assessment.capacity
    with pytest.raises(ValueError):  # 100 bytes fit two steps, not the declared thousand.
        select_region_candidate(scan.program, replace(request, payload_limits=(RegionPayloadLimit("cpu", 100),)))


def test_tuned_tensor_scan_keeps_one_symbolic_plan_and_its_guards():
    device = "cpu"
    cell = ExportedCell.capture(
        lambda w, x: (w + x, w),
        ("w", "x"),
        ("next", "old"),
        (torch.ones(3, device=device), torch.zeros(3, device=device)),
    )
    scan = TensorScan(cell, SizeParameter("n", 0, 9), carry=1, invariants=0)
    w = torch.ones(3, device=device)
    rows = tuple(torch.full((3,), i + 1.0, device=device) for i in range(9))
    tuned, report = scan.optimized(
        (w,),
        (),
        (rows[:2],),
        payload_limits=(RegionPayloadLimit(device, 10000),),
        options=measured(allow_numerical_transforms=False),
        measurement_rounds=1,
    )
    assert report.selection is not None and tuned.count == scan.count
    for count in (9, 0, 2):
        for run in (scan, tuned):
            result = run((w,), (), (rows[:count],))
            torch.testing.assert_close(result.carry[0], w + sum(range(1, count + 1)), rtol=0, atol=0)
            assert len(result.columns[0]) == count
            for i, before in enumerate(result.columns[0]):
                torch.testing.assert_close(before, w + sum(range(1, i + 1)), rtol=0, atol=0)
    with pytest.raises(ValueError):  # Beyond the declared maximum trip count.
        tuned((w,), (), (rows + rows[:1],))
    assert torch.equal(w, torch.ones_like(w))


def test_tuner_without_numerical_permission_never_warns_or_offers_rewrites():
    cell, program, _, request = chain_request()
    inputs = dict(
        zip(
            ("in0", "in1", "in2"),
            (
                torch.ones(4, 1, dtype=torch.float64),
                torch.ones(1, 4, dtype=torch.float64),
                torch.ones(4, 32, dtype=torch.float64),
            ),
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = optimize_region(
            program,
            {cell.digest: cell},
            inputs,
            payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=measured(allow_numerical_transforms=False, schedule_policy="serial"),
            warmup_rounds=1,
            measurement_rounds=3,
        )
    assert result.selection.selected == 0 and not result.rewrites and len(result.request.candidates) == 1
    assert not any(isinstance(w.message, NumericalTransformWarning) for w in caught)


def two_chains(device="cpu"):
    tensors = tuple(torch.ones(shape, dtype=torch.float64, device=device) for shape in ((4, 3), (3, 128), (128, 2)))
    cells = tuple(
        ExportedCell.capture(f, ("a", "b", "c"), ("out",), tensors)
        for f in (lambda a, b, c: ((a @ b) @ c,), lambda a, b, c: (torch.matmul(torch.matmul(a, b), c),))
    )
    arguments = tuple(Value(name, t) for name, t in zip(("a", "b", "c"), cells[0].declaration.inputs))
    calls = tuple(
        Call(("a", "b", "c"), (Value(f"out{i}", cell.declaration.outputs[0]),), cell.digest)
        for i, cell in enumerate(cells)
    )
    program = Program(region(arguments, calls, ("out0", "out1")), tuple(c.declaration for c in cells))
    return program, {c.digest: c for c in cells}, dict(zip(("a", "b", "c"), tensors))


def test_only_a_joint_rewrite_fits_and_budgets_never_authorize_an_over_budget_one():
    # Each source call needs a 4*128*8-byte product; only rewriting both calls fits 5392 bytes.
    program, cells, inputs = two_chains()
    with pytest.warns(NumericalTransformWarning):
        selected = optimize_region(
            program,
            cells,
            inputs,
            payload_limits=(RegionPayloadLimit("cpu", 5392),),
            options=measured(schedule_policy="serial"),
            warmup_rounds=0,
            measurement_rounds=2,
        )
    with StorageObserver(tuple(inputs.values())) as observed:
        result = selected.execution.run(inputs)
    assert observed.peak - sum(t.numel() * t.element_size() for t in inputs.values()) <= 5392
    for value in result.values:
        torch.testing.assert_close(value, torch.full_like(value, 384), rtol=0, atol=0)
    for limit in ({"max_rewrite_depth": 1}, {"options": measured(allow_numerical_transforms=False)}):
        with pytest.raises(ValueError):
            optimize_region(
                program,
                cells,
                inputs,
                payload_limits=(RegionPayloadLimit("cpu", 5392),),
                warmup_rounds=0,
                measurement_rounds=1,
                **{"options": measured(schedule_policy="serial"), **limit},
            )


def recovery_region():
    return TokenwiseRegion(
        TensorType((1, 4), "torch.float32"),
        8,
        CausalModelDomain(1, 3, 8),
        Recurrence().effect(3),
    )


def test_bounded_window_is_selected_when_the_faster_candidate_exceeds_the_budget():
    tokenwise = recovery_region()
    analysis = analyze_region_program(tokenwise.program)
    window = next(c for c in explore_region_schedules(tokenwise.program).candidates if "-window-2-" in c.source)
    bound = analyze_region_retention(tokenwise.program, window.schedule_map, tokenwise.retained)
    assert bound.allocations[0].peak_instances_upper_bound == 2
    windowed = RegionScheduleExecution(tokenwise.program, lower_region_schedule(tokenwise.program, window.schedule_map))
    serial = RegionScheduleExecution(
        tokenwise.program, lower_region_schedule(tokenwise.program, analysis.reference_schedule)
    )
    for count in (0, 1, 3, 8):
        sizes = {"tokens0": count, "prefix0": 0}
        actual, expected = tuple(windowed.invocations(sizes)), tuple(serial.invocations(sizes))
        assert set(actual) == set(expected) and len(actual) == len(expected)
        windows_seen = tuple(point.coordinates[0] // 2 for point in actual)
        assert windows_seen == tuple(sorted(windows_seen))
    imports = import_region_functions(tokenwise.program, tokenwise.functions)
    payload = analyze_region_payload(tokenwise.program, window.schedule_map, tokenwise.functions, tokenwise.retained)
    limit = payload.devices[0].peak_bytes_upper_bound
    # Fixed scores isolate the capacity constraint: the fastest candidate retains too much.
    request = RegionSelectionRequest(
        imports,
        (
            RegionCandidate(analysis.reference_schedule, imports, (100,)),
            RegionCandidate(window.schedule_map, imports, (10,)),
            RegionCandidate(analysis.candidates[-1].schedule_map, imports, (1,)),
        ),
        (RegionPayloadLimit("cpu", limit),),
        True,
        "window capacity oracle",
        (RegionTuningSize("prefix0", 0), RegionTuningSize("tokens0", 3)),
        tokenwise.retained,
    )
    selection = select_region_candidate(tokenwise.program, request)
    assert selection.selected == 1
    assert [c.feasible for c in selection.candidates] == [True, True, False]
    assert selection.candidates[1].payload.retention == bound

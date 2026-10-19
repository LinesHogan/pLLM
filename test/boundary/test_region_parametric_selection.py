"""Example-size timing cannot narrow symbolic legality or payload bounds."""

import warnings
from dataclasses import replace

import pytest
import torch

from pllm import _core
from pllm.config import ExecutionOptions
from pllm.contracts.region_program import SizeParameter
from pllm.contracts.region_selection import (
    RegionCandidate,
    RegionPayloadLimit,
    RegionSelectionPolicy,
    RegionSelectionRequest,
    RegionTuningSize,
)
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_optimization import (
    assess_region_candidates,
    optimize_region,
    select_region_candidate,
)
from pllm.execution.region_storage import import_region_functions
from pllm.execution.tensor_scan import TensorScan
from test.boundary.test_region_tensor import inputs_for, tensor_grid


def request_for(scan, cell, size):
    imports = import_region_functions(scan.program, {cell.digest: cell})
    schedule = analyze_region_program(scan.program).reference_schedule
    return RegionSelectionRequest(
        imports,
        (RegionCandidate(schedule, imports, (100,)),),
        (RegionPayloadLimit("cpu", 1 << 30),),
        True,
        "parameter timing oracle",
        (RegionTuningSize("steps", size),),
    )


def test_timing_a_short_scan_still_checks_capacity_for_the_full_range():
    cell = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("y",), (torch.ones(4),))
    scan = TensorScan(cell, SizeParameter("steps", 0, 1000), carry=0, invariants=1)
    request = request_for(scan, cell, 2)
    (assessment,) = assess_region_candidates(scan.program, request)
    assert assessment.capacity.devices[0].peak_tensor_bytes_upper_bound == 1001 * 16
    assert max(root.live_instances_upper_bound for root in assessment.capacity.roots) == 1000
    other = replace(request, tuning_sizes=(RegionTuningSize("steps", 999),))
    assert assess_region_candidates(scan.program, other)[0].capacity == assessment.capacity
    # The two-step example fits, but the declared retained results do not.
    with pytest.raises(ValueError, match="no region candidate"):
        select_region_candidate(scan.program, replace(request, payload_limits=(RegionPayloadLimit("cpu", 100),)))


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_optimized_scan_reuses_one_symbolic_plan_and_preserves_original_callable(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cell = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("y",), (torch.ones(4, device=device),))
    scan = TensorScan(cell, SizeParameter("steps", 0, 1_000_000), carry=1, invariants=0)
    x = torch.ones(4, device=device)
    tuned, tuned_report = scan.optimized(
        (x,),
        (),
        (),
        count=3,
        payload_limits=(RegionPayloadLimit(device, 48),),
        options=replace(ExecutionOptions(allow_numerical_transforms=False), tuning="measured"),
        measurement_rounds=2,
    )
    assert tuned_report.selection is not None
    assert tuned.count == scan.count
    assert (
        tuned_report.selection.candidates[tuned_report.selection.selected]
        .capacity.devices[0]
        .peak_tensor_bytes_upper_bound
        == 48
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("a new execution length must not trigger planning or tuning")

    for name in (
        "select_region_candidate",
        "assess_region_candidates",
        "analyze_region_program",
        "lower_region_schedule",
        "lower_region_bindings",
        "analyze_region_capacity",
        "analyze_region_storage",
    ):
        monkeypatch.setattr(_core, name, forbidden)
    for count in (0, 1, 7, 37):
        (actual,) = tuned((x,), (), (), count=count).carry
        (original,) = scan((x,), (), (), count=count).carry
        torch.testing.assert_close(actual, x + count, rtol=0, atol=0)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    for count in (-1, 1_000_001, True):
        with pytest.raises(ValueError):
            tuned((x,), (), (), count=count)
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_selected_math_rewrite_runs_through_the_scan_authoring_interface(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import pllm.execution.region_optimization as tuning

    tensors = tuple(torch.ones(s, dtype=torch.float64, device=device) for s in ((4, 2), (2, 8), (8, 2)))
    cell = ExportedCell.capture(lambda x, b, c: ((x @ b) @ c,), ("x", "b", "c"), ("y",), tensors)
    scan = TensorScan(cell, SizeParameter("steps", 0, 100), carry=1, invariants=2)
    times = iter((100, 10))
    monkeypatch.setattr(tuning, "_measure", lambda *args: next(times))
    with pytest.warns(NumericalTransformWarning):
        tuned, tuned_report = scan.optimized(
            tensors[:1],
            tensors[1:],
            (),
            count=2,
            payload_limits=(RegionPayloadLimit(device, 1 << 20),),
            options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
            warmup_rounds=0,
            measurement_rounds=1,
        )
    assert tuned_report.selection.selected == 1
    for count in (0, 1, 3, 5):
        (actual,) = tuned(tensors[:1], tensors[1:], (), count=count).carry
        # These powers of two are exactly representable; no tolerance widening.
        torch.testing.assert_close(actual, tensors[0] * 16**count, rtol=0, atol=0)
        (source,) = scan(tensors[:1], tensors[1:], (), count=count).carry
        torch.testing.assert_close(actual, source, rtol=0, atol=0)


def test_tuned_sequence_inputs_keep_length_guards_and_collection_order():
    cell = ExportedCell.capture(
        lambda x, row: (x + row, x), ("x", "row"), ("next", "old"), (torch.ones(4), torch.ones(4))
    )
    scan = TensorScan(cell, SizeParameter("steps", 0, 8), carry=1, invariants=0)
    x = torch.zeros(4)
    rows = tuple(torch.full((4,), float(i + 1)) for i in range(5))
    tuned, tuned_report = scan.optimized(
        (x,),
        (),
        (rows[:2],),
        payload_limits=(RegionPayloadLimit("cpu", 10000),),
        measurement_rounds=1,
        options=RegionSelectionPolicy(tuning="measured"),
    )
    for count in (0, 1, 5):
        result = tuned((x,), (), (rows[:count],))
        torch.testing.assert_close(result.carry[0], torch.full_like(x, count * (count + 1) / 2), rtol=0, atol=0)
        assert len(result.columns[0]) == count
        for i, before in enumerate(result.columns[0]):
            torch.testing.assert_close(before, torch.full_like(x, i * (i + 1) / 2), rtol=0, atol=0)
    with pytest.raises(ValueError, match="operands"):
        tuned((x,), (), (rows[:1],), count=2)


def test_two_parameter_tuning_signs_the_example_without_specializing_the_program():
    cell = ExportedCell.capture(
        lambda h, w: (h + w, w + 1),
        ("h", "w"),
        ("nh", "nw"),
        (torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)),
    )
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
    for chunks, layers in ((0, 0), (0, 5), (3, 0), (1, 4), (8, 5)):
        result = selected.execution.run(inputs_for(chunks, layers), sizes={"chunks": chunks, "layers": layers})
        assert result.invocations == chunks * layers
        weights, hidden = result.values
        for l, value in enumerate(weights.elements):
            torch.testing.assert_close(value, torch.full_like(value, l + 1 + chunks), rtol=0, atol=0)
        for c, value in enumerate(hidden.elements):
            torch.testing.assert_close(
                value, torch.full_like(value, c + 2 + layers * c + layers * (layers + 1) / 2), rtol=0, atol=0
            )
    for sizes in ({}, {"layers": 3}, {"layers": 3, "chunks": 9}, {"layers": 3, "chunks": 2, "extra": 0}):
        with pytest.raises(ValueError, match="size parameter"):
            selected.execution.run(inputs_for(2, 3), sizes=sizes)


@pytest.mark.parametrize("maximum", (0, 9))
def test_rounding_warning_uses_the_source_domain_not_the_zero_trip_timing_example(monkeypatch, maximum):
    import pllm.execution.region_optimization as tuning

    tensors = tuple(torch.ones(s, dtype=torch.float64) for s in ((4, 2), (2, 8), (8, 2)))
    cell = ExportedCell.capture(lambda x, b, c: ((x @ b) @ c,), ("x", "b", "c"), ("y",), tensors)
    scan = TensorScan(cell, SizeParameter("steps", 0, maximum), carry=1, invariants=2)
    times = iter((100, 10))
    monkeypatch.setattr(tuning, "_measure", lambda *args: next(times))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuned, tuned_report = scan.optimized(
            tensors[:1],
            tensors[1:],
            (),
            count=0,
            payload_limits=(RegionPayloadLimit("cpu", 10000),),
            options=replace(ExecutionOptions(schedule_policy="serial"), tuning="measured"),
            warmup_rounds=0,
            measurement_rounds=1,
        )
    numerical = [w for w in caught if isinstance(w.message, NumericalTransformWarning)]
    assert len(numerical) == (1 if maximum else 0)
    assert tuned_report.selection.candidates[1].changed_functions == ((cell.digest,) if maximum else ())
    (result,) = tuned(tensors[:1], tensors[1:], (), count=0).carry
    torch.testing.assert_close(result, tensors[0], rtol=0, atol=0)

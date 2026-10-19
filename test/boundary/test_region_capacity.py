"""Parameter-range tensor bounds checked against live real execution storage."""

import pytest
import torch

from pllm.contracts.region_program import SizeParameter
from pllm.execution.codec import encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_storage import analyze_region_capacity
from pllm.execution.tensor_scan import TensorScan
from test.boundary.test_region_tensor import inputs_for, tensor_grid
from test.boundary.test_tensor_access import StorageObserver


def increment(x):
    return (x + 1,)


def make_cell(function=increment, device="cpu", outputs=("out",)):
    return ExportedCell.capture(function, ("x",), outputs, (torch.ones(4, device=device),))


def capacity(scan, cell, schedule=None):
    source = analyze_region_program(scan.program)
    return analyze_region_capacity(scan.program, schedule or source.reference_schedule, {cell.digest: cell})


def test_million_step_carry_bound_does_not_accumulate_iteration_allocations():
    cell = make_cell()
    scan = TensorScan(cell, SizeParameter("steps", 0, 1_000_000), carry=1, invariants=0)
    bound = capacity(scan, cell)
    (facts,) = bound.devices
    assert facts.entry_copy_bytes_upper_bound == 16
    assert facts.peak_tensor_bytes_upper_bound == 48  # Sum of per-root maxima, conservative.
    assert max(root.live_instances_upper_bound for root in bound.roots) == 2
    assert facts.logical_read_bytes_upper_bound == facts.logical_write_bytes_upper_bound == 16 * 1_000_001
    assert len(encode_boundary(bound)) < 3000
    for count in (0, 1, 5, 19):
        inputs = (torch.ones(4),)
        with StorageObserver(inputs) as observed:
            result = scan(inputs, (), (), count=count)
        assert observed.peak - 16 <= facts.peak_tensor_bytes_upper_bound
        torch.testing.assert_close(result.carry[0], inputs[0] + count, rtol=0, atol=0)


def test_view_carry_needs_one_private_copy_even_for_a_million_steps():
    def view(x):
        return (x.view(x.shape),)

    cell = make_cell(view)
    scan = TensorScan(cell, SizeParameter("steps", 0, 1_000_000), carry=1, invariants=0)
    bound = capacity(scan, cell)
    (facts,) = bound.devices
    assert len(bound.roots) == 1
    assert facts.peak_tensor_bytes_upper_bound == 16
    assert facts.logical_read_bytes_upper_bound == facts.logical_write_bytes_upper_bound == 16


def test_retained_columns_charge_shared_output_storage_once():
    def update(x):
        next_x = x + 1
        return next_x, next_x

    cell = make_cell(update, outputs=("next", "collected"))
    scan = TensorScan(cell, 17, carry=1, invariants=0)
    bound = capacity(scan, cell)
    assert len(bound.roots) == 2  # One entry root and one allocation template.
    assert max(root.live_instances_upper_bound for root in bound.roots) == 17
    assert bound.devices[0].peak_tensor_bytes_upper_bound == 18 * 16
    inputs = (torch.ones(4),)
    with StorageObserver(inputs) as observed:
        result = scan(inputs, (), ())
    assert observed.peak - 16 <= bound.devices[0].peak_tensor_bytes_upper_bound
    assert result.carry[0] is result.columns[0][-1]


def test_equal_time_independent_instances_are_not_collapsed_to_one_allocation():
    cell = make_cell()
    scan = TensorScan(cell, 8, carry=0, invariants=1)
    bound = capacity(scan, cell, "{ S0[i] -> [0] : 0 <= i < 8 }")
    assert max(root.live_instances_upper_bound for root in bound.roots) == 8
    assert bound.devices[0].peak_tensor_bytes_upper_bound == 9 * 16


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("chunks,layers", [(0, 3), (1, 1), (5, 6)])
def test_schedule_bounds_cover_real_grid_storage(chunks, layers, device):
    def step(h, w):
        return h + w, w + 1

    cell = ExportedCell.capture(
        step,
        ("h", "w"),
        ("nh", "nw"),
        (torch.ones(4, dtype=torch.float64, device=device), torch.ones(4, dtype=torch.float64, device=device)),
    )
    program = tensor_grid(chunks, layers, cell)
    source = analyze_region_program(program)
    supplied = inputs_for(chunks, layers, device)
    flat = tuple(t for bank in supplied.values() for t in bank.elements)
    external_bytes = sum(t.untyped_storage().nbytes() for t in flat)
    schedules = [candidate.schedule_map for candidate in source.candidates]
    if chunks and layers:
        schedules.extend(
            (
                f"{{ S0[c,l] -> [c+l,c] : 0 <= c < {chunks} and 0 <= l < {layers} }}",
                f"{{ S0[c,l] -> [floor(c/2),floor(l/3),c,l] : 0 <= c < {chunks} and 0 <= l < {layers} }}",
            )
        )
    for schedule in schedules:
        bound = analyze_region_capacity(program, schedule, {cell.digest: cell})
        (facts,) = bound.devices
        execution = RegionExecution(
            program,
            lower_region_schedule(program, schedule),
            lower_region_bindings(program, schedule),
            {cell.digest: cell},
        )
        with StorageObserver(flat) as observed:
            result = execution.run(supplied)
        assert observed.peak - external_bytes <= facts.peak_tensor_bytes_upper_bound
        assert facts.entry_copy_bytes_upper_bound == (chunks + layers) * 32
        assert not bound.unknown_functions
        assert result.invocations == chunks * layers


def test_large_wavefront_bound_scales_with_frontier_and_results_not_whole_grid():
    def step(h, w):
        return h + w, w + 1

    cell = ExportedCell.capture(step, ("h", "w"), ("nh", "nw"), (torch.ones(4), torch.ones(4)))
    rows, layers = 1_000_000, 16
    program = tensor_grid(rows, layers, cell)
    schedule = f"{{ S0[c,l] -> [c+l,c] : 0 <= c < {rows} and 0 <= l < {layers} }}"
    bound = analyze_region_capacity(program, schedule, {cell.digest: cell})
    assert bound.devices[0].peak_tensor_bytes_upper_bound <= (2 * rows + 20 * layers) * 16
    assert len(encode_boundary(bound)) < 4000


def test_cell_internal_scratch_is_combined_with_outer_live_storage():
    def step(x):
        return ((x + 1) * 2,)

    cell = make_cell(step)
    scan = TensorScan(cell, 3, carry=1, invariants=0)
    bound = capacity(scan, cell)
    (facts,) = bound.devices
    assert facts.call_scratch_bytes_upper_bound == 16
    assert facts.peak_tensor_bytes_upper_bound == facts.live_tensor_bytes_upper_bound + 16


def test_zero_trip_does_not_charge_call_scratch_or_unknown_accesses():
    def step(x):
        return (x.sin(),)

    cell = make_cell(step)
    empty = capacity(TensorScan(cell, 0, carry=1, invariants=0), cell)
    assert not empty.unknown_functions
    assert empty.devices[0].peak_tensor_bytes_upper_bound == 16
    used = capacity(TensorScan(cell, 2, carry=1, invariants=0), cell)
    assert used.unknown_functions == (cell.digest,)

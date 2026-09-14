"""Automatic ISL band tiling keeps source versions, tails and storage bounds."""

from dataclasses import replace
from itertools import groupby

import pytest
import torch

from pllm import _core
from pllm.config import ExecutionOptions
from pllm.contracts.region_program import Call, Function, Program, Publish, Resource, ScalarType, Scan, SizeParameter, Snapshot, Value
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import (
    analyze_region_program, explore_region_schedules, explore_region_tiling, explore_region_windows, lower_region_bindings, lower_region_schedule, preview_region_schedule,
)
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_optimization import optimize_region
from pllm.execution.region_storage import analyze_region_capacity
from test.boundary.test_region_program import grid_program, region
from test.boundary.test_region_schedule import evaluate_grid
from test.boundary.test_region_tensor import tensor_grid, inputs_for
from test.boundary.test_tensor_access import StorageObserver


@pytest.mark.parametrize("limit", (2, 4, 8))
def test_lazy_catalog_matches_complete_family_merge_including_ties(limit):
    program = grid_program(9, 7)
    source = analyze_region_program(program)
    streams = [iter(source.candidates[1:]),
        iter(explore_region_windows(program, sizes=(2, 4), depth=1, max_candidates=limit - 1).candidates),
        iter(explore_region_tiling(program, max_candidates=limit - 1).candidates)]
    def order(candidate):
        instances = preview_region_schedule(program, candidate.schedule_map)
        # Equal-time groups retain simultaneity; arbitrary ordering inside a
        # tie must not make two different partial orders look equivalent.
        return tuple(frozenset((point.statement_id, point.coordinates) for point in group)
                     for _, group in groupby(instances, key=lambda point: point.time))
    expected = [source.candidates[0]]
    orders = [order(expected[0])]
    # Each family turn pulls until one distinct order arrives; duplicates do
    # not consume the turn. This mirrors the shared catalog merge exactly.
    live = [True] * len(streams)
    while any(live) and len(expected) < limit:
        advanced = False
        for index, stream in enumerate(streams):
            if not live[index]:
                continue
            for candidate in stream:
                relation = order(candidate)
                if relation in orders:
                    continue
                expected.append(candidate)
                orders.append(relation)
                advanced = True
                break
            else:
                live[index] = False
            if len(expected) == limit:
                break
        if not advanced:
            break
    assert explore_region_schedules(program, max_candidates=limit).candidates == tuple(expected)


def test_shared_catalog_keeps_distinct_global_window_and_tiled_orders():
    program = grid_program(9, 7)
    catalog = explore_region_schedules(program)
    assert not catalog.diagnostics
    assert catalog.candidates[0].source == "reference"
    assert any(candidate.source in ("isl-locality", "isl-parallel") for candidate in catalog.candidates)
    assert any("-window-" in candidate.source for candidate in catalog.candidates)
    assert any("-tile-" in candidate.source for candidate in catalog.candidates)
    orders = []
    expected = None
    for candidate in catalog.candidates:
        instances = preview_region_schedule(program, candidate.schedule_map)
        actual = evaluate_grid(instances, 9, 7)
        if expected is None:
            expected = actual
        assert actual == expected
        orders.append(tuple((point.statement_id, point.coordinates) for point in instances))
    assert len(set(orders)) == len(orders) <= 8
    assert explore_region_schedules(program, max_candidates=1).candidates == catalog.candidates[:1]


@pytest.mark.parametrize("chunks,layers", ((3, 5), (5, 7), (9, 11)))
def test_automatic_tiles_preserve_every_read_version_and_partial_tile(chunks, layers):
    program = grid_program(chunks, layers)
    source = analyze_region_program(program)
    tiled = explore_region_tiling(program)
    assert tiled.program_digest == source.program_digest
    assert not tiled.diagnostics
    assert tiled.candidates
    reference_instances = preview_region_schedule(program, source.reference_schedule)
    expected = evaluate_grid(reference_instances, chunks, layers)
    different = False
    for candidate in tiled.candidates:
        assert "-tile-" in candidate.source and candidate.schedule_tree
        actual = preview_region_schedule(program, candidate.schedule_map)
        assert len(actual) == chunks * layers
        assert len({p.coordinates for p in actual}) == chunks * layers
        assert evaluate_grid(actual, chunks, layers) == expected
        different |= [p.coordinates for p in actual] != [p.coordinates for p in reference_instances]
    assert different
    assert len({c.schedule_map for c in tiled.candidates}) == len(tiled.candidates)
    assert any("-tile-2x4-" in c.source or "-tile-4x2-" in c.source for c in tiled.candidates)


@pytest.mark.parametrize("shape", ((0, 7), (5, 0), (0, 0), (1, 9), (9, 1)))
def test_empty_or_single_dimensional_bands_do_not_invent_tile_candidates(shape):
    assert not explore_region_tiling(grid_program(*shape)).candidates


def test_tiling_search_is_bounded_and_rejects_invalid_sizes():
    program = grid_program(9, 7)
    assert len(explore_region_tiling(program, max_candidates=1).candidates) == 1
    assert not explore_region_tiling(program, max_candidates=0).candidates
    assert not explore_region_tiling(program, tile_sizes=()).candidates
    rectangular = explore_region_tiling(program, tile_sizes=(2, 3))
    assert any("-tile-2x3-" in c.source for c in rectangular.candidates)
    assert any("-tile-3x2-" in c.source for c in rectangular.candidates)
    for sizes in ((0,), (1,), (-1,), (True,), (2, 2), (2**63,)):
        with pytest.raises(ValueError):
            explore_region_tiling(program, tile_sizes=sizes)


def test_tiles_keep_each_state_address_snapshot_update_publish_order():
    integer, index = ScalarType.INT, ScalarType.INDEX
    inner = region((Value("l", index),), (
        Snapshot(("l",), (Value("w", integer),), "state"),
        Call(("w",), (Value("next", integer),), "step"),
        Publish(("next", "l"), (), "state"),
    ), ())
    outer = region((Value("c", index),), (
        Scan((), (), 4, inner),
    ), ())
    program = Program(region((), (
        Scan((), (), 3, outer),
    ), ()), (Function("step", (integer,), (integer,)),), (Resource("state", (integer,), (4,)),))
    candidates = explore_region_tiling(program, tile_sizes=(2, 3)).candidates
    assert candidates
    for candidate in candidates:
        next_action = [(0, 0)] * 4
        instances = preview_region_schedule(program, candidate.schedule_map)
        for invocation in instances:
            chunk, address = invocation.coordinates
            phase = invocation.statement_id  # snapshot, pure update, publication
            assert (chunk, phase) == next_action[address]
            next_action[address] = (chunk, phase + 1) if phase < 2 else (chunk + 1, 0)
        assert len(instances) == 3 * 4 * 3
        assert next_action == [(3, 0)] * 4


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_generated_tiles_execute_real_tensors_inside_source_capacity(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cell = ExportedCell.capture(lambda h, w: (h + w, w + 1), ("h", "w"), ("nh", "nw"),
        (torch.ones(4, dtype=torch.float64, device=device), torch.ones(4, dtype=torch.float64, device=device)))
    chunks, layers = 5, 7
    program = tensor_grid(chunks, layers, cell)
    inputs = inputs_for(chunks, layers, device)
    external = tuple(t for bank in inputs.values() for t in bank.elements)
    external_bytes = sum(t.untyped_storage().nbytes() for t in external)
    cells = {cell.digest: cell}
    for candidate in explore_region_tiling(program).candidates:
        capacity = analyze_region_capacity(program, candidate.schedule_map, cells)
        execution = RegionExecution(program, lower_region_schedule(program, candidate.schedule_map),
            lower_region_bindings(program, candidate.schedule_map), cells)
        with StorageObserver(external) as observed:
            result = execution.run(inputs)
        assert observed.peak - external_bytes <= capacity.devices[0].peak_tensor_bytes_upper_bound
        assert result.invocations == chunks * layers
        weights, hidden = result.values
        for l, value in enumerate(weights.elements):
            torch.testing.assert_close(value, torch.full_like(value, l + 1 + chunks), rtol=0, atol=0)
        for c, value in enumerate(hidden.elements):
            expected = c + 2 + layers * c + layers * (layers + 1) / 2
            torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)


def test_million_row_parameterized_tiles_are_symbolic_and_reach_the_tuner(monkeypatch):
    import pllm.execution.region_optimization as tuning
    cell = ExportedCell.capture(lambda h, w: (h + w, w + 1), ("h", "w"), ("nh", "nw"),
        (torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)))
    program = replace(tensor_grid("chunks", "layers", cell),
        parameters=(SizeParameter("chunks", 0, 1_000_000), SizeParameter("layers", 0, 16)))
    tiled = explore_region_tiling(program)
    assert tiled.candidates and len(tiled.candidates) <= 8
    assert all(len(c.schedule_map) < 3000 and len(c.schedule_tree) < 4000 for c in tiled.candidates)
    target = tiled.candidates[0].schedule_map
    lowered_target = lower_region_schedule(program, target).schedule_map
    monkeypatch.setattr(tuning, "_measure", lambda execution, *args: 1 if execution._lowered.schedule_map == lowered_target else 100)
    selected = optimize_region(program, {cell.digest: cell}, inputs_for(2, 3), sizes={"chunks": 2, "layers": 3},
        payload_limits=(RegionPayloadLimit("cpu", 1 << 30),), options=ExecutionOptions(allow_numerical_transforms=False),
        warmup_rounds=0, measurement_rounds=1)
    assert selected.request.candidates[selected.selection.selected].schedule_map == target
    def forbidden(*args):
        raise AssertionError("execution must not generate tiles or recompile")
    for name in ("explore_region_tiling", "analyze_region_program", "select_region_candidate", "lower_region_schedule", "lower_region_bindings"):
        monkeypatch.setattr(_core, name, forbidden)
    for chunks, layers in ((0, 0), (0, 7), (3, 0), (1, 1), (5, 7)):
        result = selected.run(inputs_for(chunks, layers), sizes={"chunks": chunks, "layers": layers})
        assert result.invocations == chunks * layers
        for c, value in enumerate(result.values[1].elements):
            expected = c + 2 + layers * c + layers * (layers + 1) / 2
            torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)

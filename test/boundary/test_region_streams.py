"""Concurrent physical phases preserve versions, completion and original owners."""

from dataclasses import replace

import pytest
import torch

from pllm import _core
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.region_streams import RegionStreamPlan
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution, RegionStreamExecution
from pllm.execution.region_frame import lookup
from pllm.execution.region_schedule import RegionScheduleExecution
from pllm.execution.region_streams import RegionStreamBindings, RegionStreamFrame, explore_region_streams, lower_region_streams, validate_region_streams
from test.boundary.test_region_tensor import inputs_for, tensor_grid


def grid(device="cpu", chunks=3, layers=5):
    cell = ExportedCell.capture(lambda h, w: ((h + w) * 0.5, w + h * 0.25), ("h", "w"), ("nh", "nw"),
        (torch.ones(4, dtype=torch.float64, device=device), torch.ones(4, dtype=torch.float64, device=device)))
    program = tensor_grid(chunks, layers, cell)
    source = analyze_region_program(program)
    cells = {cell.digest: cell}
    plan, = explore_region_streams(program, cells, lanes=2, max_candidates=1).candidates
    return program, cell, source, plan


def limits(plan):
    return tuple(RegionPayloadLimit(d.device, d.peak_bytes_upper_bound) for d in plan.devices)


def test_phase_legality_capacity_and_transport_are_rederived_from_source():
    program, cell, source, plan = grid()
    cells = {cell.digest: cell}
    assert decode_boundary(encode_boundary(plan), RegionStreamPlan) == plan
    validate_region_streams(program, cells, plan)
    phase_capacity = plan.payload.capacity.devices[0]
    assert phase_capacity.call_scratch_bytes_upper_bound > 0
    assert plan.devices[0].concurrent_scratch_bytes_upper_bound == 2 * phase_capacity.call_scratch_bytes_upper_bound
    assert plan.devices[0].peak_bytes_upper_bound == plan.payload.devices[0].peak_bytes_upper_bound + phase_capacity.call_scratch_bytes_upper_bound
    times = {t.input_id: t for t in lower_region_bindings(program, plan.phase_schedule).times}
    placements = {t.input_id: t for t in plan.placements}
    groups = {}
    for point in RegionScheduleExecution(program, lower_region_schedule(program, plan.phase_schedule)).invocations():
        phase = lookup(times[point.statement_id], point.coordinates)[1]
        lane, = lookup(placements[point.statement_id], point.coordinates)[1]
        groups.setdefault(phase, []).append((point.coordinates, lane))
    assert any(len(points) == 2 and {lane for _, lane in points} == {0, 1} for points in groups.values())
    assert all(len({sum(coordinates) for coordinates, _ in points}) == 1 for points in groups.values())
    # A lexical chunk loop carries updated weights. Compressing its chunk
    # coordinate would put a producer and consumer in the same physical phase.
    with pytest.raises(ValueError, match="dependency"):
        lower_region_streams(program, source.reference_schedule, cells, axis=1, lanes=2)
    with pytest.raises(ValueError, match="budget"):
        RegionStreamExecution(program, cells, plan, payload_limits=(RegionPayloadLimit("cpu", 1),))
    for tampered in (
        replace(plan, phase_schedule=plan.schedule_map),
        replace(plan, placements=()),
        replace(plan, devices=(replace(plan.devices[0], peak_bytes_upper_bound=0),)),
    ):
        with pytest.raises(ValueError, match="stream plan differs"):
            validate_region_streams(program, cells, tampered)


def test_pipeline_frontiers_cover_each_grid_predecessor_and_fund_pending_values():
    program, cell, _, base = grid()
    cells = {cell.digest: cell}
    plan = lower_region_streams(program, base.schedule_map, cells, axis=base.axis, lanes=2, pipeline_depth=3)
    assert decode_boundary(encode_boundary(plan), RegionStreamPlan) == plan
    assert not plan.host_fences
    assert plan.devices[0].peak_bytes_upper_bound == 3 * base.devices[0].peak_bytes_upper_bound
    assert plan.devices[0].pending_phase_bytes_upper_bound == 2 * base.devices[0].peak_bytes_upper_bound
    bindings = lower_region_bindings(program, plan.phase_schedule)
    times = {t.input_id: t for t in bindings.times}
    placements = {t.input_id: t for t in plan.placements}
    waits = {w.lane: {t.input_id: t for t in w.predecessors} for w in plan.waits}
    for i in range(3):
        for j in range(5):
            expected = {}
            for predecessor in ((i - 1, j), (i, j - 1)):
                if min(predecessor) < 0:
                    continue
                phase = lookup(times[0], predecessor)[1]
                lane, = lookup(placements[0], predecessor)[1]
                expected[lane] = max(expected.get(lane, phase), phase)
            actual = {lane: result[1] for lane, tables in waits.items()
                if (result := lookup(tables.get(0), (i, j))) is not None}
            assert actual == expected
    with pytest.raises(ValueError, match="budget"):
        RegionStreamExecution(program, cells, plan, payload_limits=limits(base))
    for bad in (replace(plan, waits=()), replace(plan, pipeline_depth=1),
                replace(plan, host_fences=(0,)), replace(plan, devices=base.devices)):
        with pytest.raises(ValueError, match="stream plan differs"):
            validate_region_streams(program, cells, bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("depth", (1, 3))
def test_wavefront_uses_borrowed_streams_drains_completion_and_keeps_exact_versions(monkeypatch, depth):
    program, cell, source, plan = grid("cuda:0")
    plan = lower_region_streams(program, plan.schedule_map, {cell.digest: cell}, axis=plan.axis, lanes=2, pipeline_depth=depth)
    owner, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    streams = RegionStreamBindings(owner, (auxiliary,))
    inputs = inputs_for(3, 5, "cuda:0")
    owner.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(owner):
        reference = RegionExecution(program, lower_region_schedule(program, source.reference_schedule),
            lower_region_bindings(program, source.reference_schedule), {cell.digest: cell}).run(inputs)
    launches, completions = [], []
    def observed(*args):
        launches.append(torch.cuda.current_stream().cuda_stream)
        result = cell.function(*args)
        event = torch.cuda.Event()
        event.record()
        completions.append(event)
        return result
    execution = RegionStreamExecution(program, {cell.digest: replace(cell, function=observed)}, plan, payload_limits=limits(plan))
    def forbidden(*args):
        raise AssertionError("runtime cannot plan, prove or allocate streams")
    for name in ("lower_region_streams", "validate_region_streams", "lower_region_schedule", "lower_region_bindings"):
        monkeypatch.setattr(_core, name, forbidden)
    # Reject invalid lane bindings before touching any mathematical input.
    with pytest.raises(ValueError, match="distinct streams"):
        execution.run(inputs, streams=RegionStreamBindings(owner, (owner,)))
    assert not launches
    result = execution.run(inputs, streams=streams)
    assert set(launches) == {owner.cuda_stream, auxiliary.cuda_stream}
    assert all(event.query() for event in completions)
    assert result.invocations == 15
    with torch.cuda.stream(owner):
        for got, want in zip(result.values, reference.values, strict=True):
            for a, b in zip(got.elements, want.elements, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    owner.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fail", (False, True))
def test_pipeline_issues_later_phases_before_completion_and_drains_on_failure(monkeypatch, fail):
    from test.boundary.test_native_kernel_streams import native_scan
    from pllm.execution.tensor_scan import TensorScan
    from pllm.execution.region_values import Bank
    _, native, _ = native_scan("cuda:0")
    scan = TensorScan(native, 6, carry=0, invariants=0)
    cells = {native.declaration.name: native}
    plans = explore_region_streams(scan.program, cells, lanes=2, max_candidates=3, max_pipeline_depth=3).candidates
    assert tuple(plan.pipeline_depth for plan in plans) == (1, 2, 3)
    plan = plans[-1]
    streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
    events, issued_early, overlapped, windows = [], [], [], []
    calls = 0
    recording = False
    def kernel(inputs, resources):
        nonlocal calls
        if not recording:
            return native.function(inputs, resources)
        calls += 1
        if calls == 3:
            issued_early.append(not events[0].query())
        if calls == 5:
            # The independent auxiliary lane can finish its second phase
            # while the owner's first phase is still running on the GPU.
            events[3].synchronize()
            overlapped.append(not events[0].query())
        if calls == 1:
            torch.cuda._sleep(50_000_000)
        result = native.function(inputs, resources)
        event = torch.cuda.Event()
        event.record()
        events.append(event)
        if fail and calls == 5:
            raise RuntimeError("failure with earlier phases pending")
        return result
    retire = RegionStreamFrame._retire
    def retiring(frame):
        windows.append(len(frame._pending))
        assert frame._pending[0].values
        retire(frame)
    monkeypatch.setattr(RegionStreamFrame, "_retire", retiring)
    cells = {native.declaration.name: replace(native, function=kernel)}
    runner = RegionStreamExecution(scan.program, cells, plan, payload_limits=limits(plan))
    values = tuple(torch.full((4,), float(i), device="cuda:0") for i in range(6))
    streams.owner.wait_stream(torch.cuda.current_stream())
    # CUDA module loading/allocator initialization may synchronize the device.
    # Warm the same execution before observing overlap between issued kernels.
    with torch.cuda.stream(streams.owner):
        torch.cuda._sleep(0)
    runner.run({"arg0": Bank(values)}, streams=streams)
    windows.clear()
    recording = True
    if fail:
        with pytest.raises(RuntimeError, match="earlier phases pending"):
            runner.run({"arg0": Bank(values)}, streams=streams)
    else:
        result, = runner.run({"arg0": Bank(values)}, streams=streams).values
        for i, output in enumerate(result.elements):
            torch.testing.assert_close(output, torch.full_like(output, i + 2), rtol=0, atol=0)
    assert issued_early == [True]
    assert overlapped == [True]
    assert windows and max(windows) == 3
    assert all(event.query() for event in events)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_exception_joins_already_launched_auxiliary_work_before_frame_release():
    program, cell, _, plan = grid("cuda:0")
    owner, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    completions = []
    def failing(*args):
        result = cell.function(*args)
        if torch.cuda.current_stream() == auxiliary:
            torch.cuda._sleep(2_000_000)
            event = torch.cuda.Event()
            event.record()
            completions.append(event)
            raise RuntimeError("injected after auxiliary launch")
        return result
    runner = RegionStreamExecution(program, {cell.digest: replace(cell, function=failing)}, plan, payload_limits=limits(plan))
    inputs = inputs_for(3, 5, "cuda:0")
    owner.wait_stream(torch.cuda.current_stream())
    with pytest.raises(RuntimeError, match="injected"):
        runner.run(inputs, streams=RegionStreamBindings(owner, (auxiliary,)))
    assert completions and all(event.query() for event in completions)
    owner.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("depth", (1, 3))
def test_page_effects_remain_on_original_pin_stream(depth):
    from pllm.execution.region_resources import RegionResources, ResourcePageCell
    from test.boundary.test_region_pages import owner_fixture
    owner, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    execution, task, rows, refs = owner_fixture(rounds=2, stream=owner)
    original = task.region
    program = task.program
    source = analyze_region_program(program)
    # ISL puts the independent layer coordinate first in this page program.
    # Keep the later stage coordinates that separate snapshots/calls/publishes.
    candidate = next(c for c in source.candidates if c.source == "isl-parallel")
    plan = lower_region_streams(program, candidate.schedule_map, original._cells, axis=0, lanes=2, pipeline_depth=depth)
    runner = RegionStreamExecution(program, original._cells, plan, payload_limits=limits(plan))
    invocation = execution.start_step(rows, owner)
    pages = RegionResources(program, invocation.context, tuple(ResourcePageCell("layers", (i,), ref, task.fields)
        for i, ref in enumerate(invocation.context.pin.refs)))
    try:
        with pytest.raises(ValueError, match="original PTTS pin stream"):
            runner.run({}, pages=pages, streams=RegionStreamBindings(auxiliary, (owner,)))
        result = runner.run({}, pages=pages, streams=RegionStreamBindings(owner, (auxiliary,)))
        assert len(invocation.context.page_publications) == 4
        old, first, second = result.values
        with torch.cuda.stream(owner):
            torch.testing.assert_close(old, torch.ones_like(old), rtol=0, atol=0)
            torch.testing.assert_close(first, torch.full_like(first, 3), rtol=0, atol=0)
            torch.testing.assert_close(second, torch.full_like(second, 4), rtol=0, atol=0)
    finally:
        execution.cancel_step(invocation)

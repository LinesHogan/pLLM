"""Side-stream execution: concurrent phases keep versions, completion and owners.

A stream plan compresses one independent schedule axis into lanes on borrowed
CUDA streams. Planning must respect dependences and concurrent memory; the
executor must keep exact results, launch on the borrowed streams, drain issued
work on failure and leave page effects on the PTTS pin stream.
"""

from dataclasses import replace

import pytest
import torch

from pllm import _core
from pllm.config import ExecutionOptions
from pllm.contracts.region_imports import KernelAllocation, NativeKernelImport
from pllm.contracts.region_program import (
    AccessMode,
    AffineIndex,
    BorrowedResource,
    Call,
    Coordinate,
    Function,
    Program,
    ResourceAccess,
    ScalarType,
    TensorType,
    Value,
)
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_selection import RegionCandidate, RegionPayloadLimit, RegionSelectionRequest
from pllm.execution.decoder_region import index_cell
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program, lower_region_schedule
from pllm.execution.region_execution import RegionExecutionPlan
from pllm.execution.region_native import NativePrimitive, external_primitive
from pllm.execution.region_optimization import optimize_region, prepare_region_execution, select_region_candidate
from pllm.execution.region_resources import ExternalFunction, ReadonlyResourceBinding, RegionResources
from pllm.execution.region_storage import import_region_functions
from pllm.execution.region_streams import (
    RegionStreamBindings,
    explore_region_streams,
    lower_region_streams,
    plan_region_schedule,
)
from pllm.execution.region_values import Bank
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.tensor_scan import TensorScan
from test.region_fixtures import bind, inputs_for, limits, owner_fixture, region, stream_execution, tensor_grid

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def grid(device="cpu", chunks=3, layers=5, depth=2):
    cell = ExportedCell.capture(
        lambda h, w: ((h + w) * 0.5, w + h * 0.25),
        ("h", "w"),
        ("nh", "nw"),
        (torch.ones(4, dtype=torch.float64, device=device), torch.ones(4, dtype=torch.float64, device=device)),
    )
    program = tensor_grid(chunks, layers, cell)
    cells = {cell.digest: cell}
    parallel = analyze_region_program(program, parallel_only=True).candidates[-1].schedule_map
    (plan,) = explore_region_streams(program, cells, schedule=parallel).candidates
    plan = lower_region_streams(program, plan.schedule_map, cells, axis=plan.axis, lanes=2, pipeline_depth=depth)
    return program, cell, plan


def native_scan(device):
    """A native kernel with an explicit current-stream ABI and private backing storage."""
    tensor = TensorType((4,), "torch.float32", device)
    declaration = Function("test.native.sliced_buffer", (tensor,), (tensor,))
    implementation = NativeKernelImport(
        declaration, "test.sliced_buffer.current_stream@1", device, (KernelAllocation("backing", device, 64),)
    )
    streams = []

    def kernel(inputs, resources):
        (value,) = inputs
        if device.startswith("cuda"):
            streams.append(torch.cuda.current_stream().cuda_stream)
        backing = torch.empty(16, device=device)
        backing[:4].copy_(value).add_(2)
        return (backing[:4],)

    cell = NativePrimitive(declaration, kernel, implementation)
    return TensorScan(cell, 3, carry=0, invariants=0), cell, streams


def parallel_plans(program, cells):
    schedule = analyze_region_program(program, parallel_only=True).candidates[-1].schedule_map
    return explore_region_streams(program, cells, schedule=schedule).candidates


def test_stream_plans_respect_dependences_budgets_and_kernel_abis():
    program, cell, plan = grid()
    cells = {cell.digest: cell}
    # The chunk loop carries weights: compressing it would put producer and consumer in one phase.
    with pytest.raises(ValueError, match="dependency"):
        lower_region_streams(program, analyze_region_program(program).reference_schedule, cells, axis=1, lanes=2)
    with pytest.raises(ValueError, match="budget|reservation"):
        stream_execution(program, cells, plan, payload_limits=(RegionPayloadLimit("cpu", 1),))
    # A generic callback stays opaque even when its code equals a kernel with a declared ABI.
    scan, native, _ = native_scan("cpu")
    assert parallel_plans(scan.program, {native.declaration.name: native})
    assert not parallel_plans(scan.program, {native.declaration.name: replace(native, implementation=None)})


def test_selection_charges_concurrent_scratch_and_the_default_mapping_honors_budgets(monkeypatch):
    program, cell, plan = grid()
    cells = {cell.digest: cell}
    source = analyze_region_program(program)
    imports = import_region_functions(program, cells)
    reference = RegionCandidate(source.reference_schedule, imports, (100,))
    parallel = RegionCandidate(plan.phase_schedule, imports, (1,), stream_plan=plan)
    request = RegionSelectionRequest(imports, (reference, parallel), limits(plan), False, "deterministic ranking")
    selected = select_region_candidate(program, request)
    assert selected.selected == 1 and selected.candidates[1].peak_bytes == limits(plan)
    RegionExecutionPlan(program, imports, RegionScheduleProposal(plan.payload, plan), limits(plan))
    # The ordinary phase payload fits, but its concurrent scratch does not.
    phase_budget = (RegionPayloadLimit("cpu", plan.payload.devices[0].peak_bytes_upper_bound),)
    constrained = select_region_candidate(program, replace(request, payload_limits=phase_budget))
    assert constrained.selected == 0 and not constrained.candidates[1].feasible
    # Without measurements the fixed default mapping picks lanes the budget can fund.
    monkeypatch.setattr(_core, "explore_region_schedules", lambda *a, **k: pytest.fail("regenerated a catalog"))
    tight = plan_region_schedule(program, cells, parallel=True, payload_limits=(RegionPayloadLimit("cpu", 768),))
    assert tight is not None and tight.stream_plan is None
    assert tight.payload.capacity.schedule_map == lower_region_schedule(program, source.reference_schedule).schedule_map
    funded = plan_region_schedule(program, cells, parallel=True, payload_limits=(RegionPayloadLimit("cpu", 2112),))
    assert funded.stream_plan.lanes == funded.stream_plan.pipeline_depth == 2
    assert plan_region_schedule(program, cells, parallel=True, payload_limits=(RegionPayloadLimit("cpu", 1),)) is None


@requires_cuda
def test_wavefront_launches_on_borrowed_streams_and_keeps_exact_versions(monkeypatch):
    program, cell, plan = grid("cuda:0", depth=3)
    owner, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    inputs = inputs_for(3, 5, "cuda:0")
    owner.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(owner):
        reference = bind(program, {cell.digest: cell}, analyze_region_program(program).reference_schedule).run(inputs)
    launches, completions = [], []

    def observed(*args):
        launches.append(torch.cuda.current_stream().cuda_stream)
        result = cell.function(*args)
        event = torch.cuda.Event()
        event.record()
        completions.append(event)
        return result

    execution = stream_execution(
        program, {cell.digest: replace(cell, function=observed)}, plan, payload_limits=limits(plan)
    )
    for name in ("lower_region_streams", "lower_region_schedule", "lower_region_bindings"):
        monkeypatch.setattr(_core, name, lambda *args: pytest.fail("runtime planned streams"))
    with pytest.raises(ValueError, match="distinct streams"):
        execution.run(inputs, streams=RegionStreamBindings(owner, (owner,)))
    assert not launches  # Lane bindings are checked before any math.
    result = execution.run(inputs, streams=RegionStreamBindings(owner, (auxiliary,)))
    assert set(launches) == {owner.cuda_stream, auxiliary.cuda_stream}
    assert all(event.query() for event in completions)
    assert result.invocations == 15
    with torch.cuda.stream(owner):
        for got, want in zip(result.values, reference.values, strict=True):
            for a, b in zip(got.elements, want.elements, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    owner.synchronize()


@requires_cuda
def test_pipeline_issues_later_phases_before_earlier_ones_complete():
    _, native, _ = native_scan("cuda:0")
    scan = TensorScan(native, 6, carry=0, invariants=0)
    cells = {native.declaration.name: native}
    (base,) = parallel_plans(scan.program, cells)
    plan = lower_region_streams(scan.program, base.schedule_map, cells, axis=base.axis, lanes=2, pipeline_depth=3)
    streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
    events, issued_early, overlapped, calls = [], [], [], []
    recording = False

    def kernel(inputs, resources):
        if not recording:
            return native.function(inputs, resources)
        calls.append(1)
        if len(calls) == 3:
            issued_early.append(not events[0].query())
        if len(calls) == 5:
            # The auxiliary lane finishes its second phase while the owner's first still runs.
            events[3].synchronize()
            overlapped.append(not events[0].query())
        if len(calls) == 1:
            torch.cuda._sleep(50_000_000)
        result = native.function(inputs, resources)
        event = torch.cuda.Event()
        event.record()
        events.append(event)
        return result

    runner = stream_execution(
        scan.program, {native.declaration.name: replace(native, function=kernel)}, plan, payload_limits=limits(plan)
    )
    values = tuple(torch.full((4,), float(i), device="cuda:0") for i in range(6))
    streams.owner.wait_stream(torch.cuda.current_stream())
    runner.run({"arg0": Bank(values)}, streams=streams)  # Warm module loading, which may synchronize.
    recording = True
    (result,) = runner.run({"arg0": Bank(values)}, streams=streams).values
    for i, output in enumerate(result.elements):
        torch.testing.assert_close(output, torch.full_like(output, i + 2), rtol=0, atol=0)
    assert issued_early == [True] and overlapped == [True]
    assert all(event.query() for event in events)


@requires_cuda
def test_failure_joins_already_launched_auxiliary_work_before_release():
    program, cell, plan = grid("cuda:0")
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

    runner = stream_execution(
        program, {cell.digest: replace(cell, function=failing)}, plan, payload_limits=limits(plan)
    )
    owner.wait_stream(torch.cuda.current_stream())
    with pytest.raises(RuntimeError, match="injected"):
        runner.run(inputs_for(3, 5, "cuda:0"), streams=RegionStreamBindings(owner, (auxiliary,)))
    assert completions and all(event.query() for event in completions)
    owner.synchronize()


@requires_cuda
def test_page_effects_stay_on_the_original_pin_stream():
    owner, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    execution, task, rows, _ = owner_fixture(rounds=2, stream=owner)
    program, cells = task.program, task.region.cells
    candidate = next(c for c in analyze_region_program(program).candidates if c.source == "isl-parallel")
    plan = lower_region_streams(program, candidate.schedule_map, cells, axis=0, lanes=2, pipeline_depth=3)
    runner = stream_execution(program, cells, plan, payload_limits=limits(plan))
    invocation = execution.start_step(rows, owner)
    pages = task.pages(invocation.context)
    try:
        with pytest.raises(ValueError, match="original PTTS pin stream"):
            runner.run({}, pages=pages, streams=RegionStreamBindings(auxiliary, (owner,)))
        old, first, second = runner.run({}, pages=pages, streams=RegionStreamBindings(owner, (auxiliary,))).values
        assert len(invocation.context.page_publications) == 4
        with torch.cuda.stream(owner):
            for value, expected in ((old, 1), (first, 3), (second, 4)):
                torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    finally:
        execution.cancel_step(invocation)


@requires_cuda
def test_tuner_measures_parallel_candidates_and_rebinds_without_runtime_search(monkeypatch):
    import pllm.execution.region_optimization as tuning

    program, cell, _ = grid("cuda:0")
    streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
    inputs = inputs_for(3, 5, "cuda:0")
    streams.owner.wait_stream(torch.cuda.current_stream())
    measured, measure = [], tuning._measure

    def ranked(execution, *args):
        assert measure(execution, *args) > 0
        measured.append(execution.stream_plan is not None)
        return 1 if execution.stream_plan is not None else 100  # Exercise the parallel winner.

    monkeypatch.setattr(tuning, "_measure", ranked)
    options = ExecutionOptions(allow_numerical_transforms=False)
    selected = optimize_region(
        program,
        {cell.digest: cell},
        inputs,
        streams=streams,
        options=replace(options, tuning="measured"),
        payload_limits=(RegionPayloadLimit("cuda:0", 1 << 20),),
        warmup_rounds=0,
        measurement_rounds=1,
        max_candidates=3,
    )
    assert any(measured) and not all(measured)
    assert selected.execution.stream_plan is not None
    serial = prepare_region_execution(
        program,
        {cell.digest: cell},
        payload_limits=selected.request.payload_limits,
        options=replace(options, schedule_policy="serial"),
    )
    assert serial.stream_plan is None
    for name in ("explore_region_streams", "select_region_candidate", "lower_region_schedule"):
        monkeypatch.setattr(_core, name, lambda *args: pytest.fail("selected execution searched again"))
    with pytest.raises(ValueError, match="must be supplied together"):
        selected.execution.run(inputs)
    got, want = selected.execution.run(inputs, streams=streams), serial.run(inputs, streams=streams)
    with torch.cuda.stream(streams.owner):
        for left, right in zip(got.values, want.values, strict=True):
            for a, b in zip(left.elements, right.elements, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    streams.owner.synchronize()


@requires_cuda
def test_native_kernels_run_on_the_auxiliary_lane_with_private_backing_storage():
    scan, cell, launches = native_scan("cuda:0")
    cells = {cell.declaration.name: cell}
    (plan,) = parallel_plans(scan.program, cells)
    runner = stream_execution(
        scan.program,
        cells,
        plan,
        payload_limits=(RegionPayloadLimit("cuda:0", plan.devices[0].peak_bytes_upper_bound),),
    )
    bindings = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
    values = tuple(torch.full((4,), float(i), device="cuda:0") for i in range(3))
    bindings.owner.wait_stream(torch.cuda.current_stream())
    (outputs,) = runner.run({"arg0": Bank(values)}, streams=bindings).values
    assert set(launches) == {bindings.owner.cuda_stream, bindings.auxiliary[0].cuda_stream}
    with torch.cuda.stream(bindings.owner):
        for i, (value, source) in enumerate(zip(outputs.elements, values, strict=True)):
            assert value.untyped_storage().nbytes() == 16  # The 64-byte backing stays kernel scratch.
            torch.testing.assert_close(value, torch.full_like(value, i + 2), rtol=0, atol=0)
            torch.testing.assert_close(source, torch.full_like(source, i), rtol=0, atol=0)
    bindings.owner.synchronize()


@requires_cuda
def test_external_kernel_borrows_original_weights_only_with_the_matching_local_abi():
    from pllm.backends.vllm.replay import VllmTokenwiseReplay

    owner = torch.cuda.Stream()
    execution, _, rows, _ = owner_fixture(rounds=1, stream=owner)
    invocation = execution.start_step(rows, owner)
    # The actual adapter's stream loan, without loading model weights.
    backend = VllmTokenwiseReplay.__new__(VllmTokenwiseReplay)
    backend._invocation = invocation
    backend._auxiliary_streams = ()
    streams = backend.borrow_streams(invocation.context, 2)
    assert backend.borrow_streams(invocation.context, 2) == streams and streams.owner is owner
    weights = BorrowedResource("weights", (2,))
    index = ScalarType.INDEX
    tensor = TensorType((2,), "torch.float32", "cuda:0")
    function = Function(
        "test.readonly.kernel",
        (index,),
        (tensor,),
        (weights.name,),
        (ResourceAccess(weights.name, AccessMode.READ, (index_cell(0),)),),
    )
    program = Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", index),), AffineIndex(0)),
                Coordinate((), (Value("one", index),), AffineIndex(1)),
                Call(("zero",), (Value("a", tensor),), function.name),
                Call(("one",), (Value("b", tensor),), function.name),
            ),
            ("a", "b"),
        ),
        (function,),
        (weights,),
    )
    abi = NativeKernelImport(function, "test.borrowed_weights.current_stream@1", "cuda:0", ())
    cells = {function.name: external_primitive(function, abi)}
    plan = lower_region_streams(program, "{ S0[] -> [0]; S1[] -> [1] }", cells, axis=0, lanes=2)
    runner = stream_execution(program, cells, plan, payload_limits=(RegionPayloadLimit("cuda:0", 1024),))
    value = torch.tensor([3.0, 7.0], device="cuda:0")
    view = ReadonlyTensorView(weights, tuple(ReadonlyTensorCell.bind((i,), (value,)) for i in range(2)))
    launches = []

    def endpoint(inputs):
        launches.append(torch.cuda.current_stream().cuda_stream)
        return (value,)

    def resources(local_abi):
        return RegionResources(
            program,
            invocation.context,
            (),
            (ExternalFunction(function, endpoint, local_abi),),
            borrows=(ReadonlyResourceBinding(view, invocation.context),),
        )

    owner.wait_stream(torch.cuda.current_stream())
    try:
        for different in (None, replace(abi, implementation="another.kernel@1")):
            with pytest.raises(ValueError, match="execution contract"):
                runner.run({}, pages=resources(different), streams=streams)
        assert not launches
        result = runner.run({}, pages=resources(abi), streams=streams)
        assert set(launches) == {owner.cuda_stream, streams.auxiliary[0].cuda_stream}
        with torch.cuda.stream(owner):
            for output in result.values:
                torch.testing.assert_close(output, value, rtol=0, atol=0)
                assert output.data_ptr() != value.data_ptr()
        assert not invocation.context.page_publications
    finally:
        execution.cancel_step(invocation)
        execution.retire(tuple(row.identity for row in rows))

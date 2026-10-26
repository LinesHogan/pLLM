"""Side-stream execution: concurrent phases keep versions, completion and owners.

A stream plan compresses one independent schedule axis into lanes on borrowed
CUDA streams. Planning must respect dependences and concurrent memory; the
executor must keep exact results, launch on the borrowed streams, join issued
work asynchronously on failure and leave page effects on the PTTS pin stream.
"""

import json
import warnings
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

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
from pllm.contracts.region_selection import RegionCandidate, RegionPayloadLimit, RegionSelectionRequest
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import analyze_region_program
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
from pllm.execution.region_window import WindowBackpressure
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.tensor_scan import TensorScan
from test.region_fixtures import (
    bind,
    index_cell,
    inputs_for,
    limits,
    owner_fixture,
    region,
    stream_execution,
    tensor_grid,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@contextmanager
def no_host_waits(case):
    """Audit production issue/settlement; explicit observation waits stay outside."""
    torch.cuda.set_sync_debug_mode("warn")
    torch.cuda.set_sync_debug_mode("default")
    counts = {"event_synchronize": 0, "stream_synchronize": 0, "device_synchronize": 0}
    event_sync, stream_sync, device_sync = (
        torch.cuda.Event.synchronize,
        torch.cuda.Stream.synchronize,
        torch.cuda.synchronize,
    )

    def event_wait(event):
        counts["event_synchronize"] += 1
        return event_sync(event)

    def stream_wait(stream):
        counts["stream_synchronize"] += 1
        return stream_sync(stream)

    def device_wait(device=None):
        counts["device_synchronize"] += 1
        return device_sync(device)

    with (
        warnings.catch_warnings(record=True) as messages,
        patch.object(torch.cuda.Event, "synchronize", event_wait),
        patch.object(torch.cuda.Stream, "synchronize", stream_wait),
        patch.object(torch.cuda, "synchronize", device_wait),
    ):
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            yield counts
        finally:
            torch.cuda.set_sync_debug_mode("default")
    counts["sync_warnings"] = sum("synchroniz" in str(message.message) for message in messages)
    print(json.dumps({"case": case, **counts}))
    assert not any(counts.values()), counts


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
        declaration,
        "test.sliced_buffer.current_stream@1",
        device,
        (KernelAllocation("backing", device, 64),),
        host_reads=False,
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
    with pytest.raises(ValueError):
        lower_region_streams(program, analyze_region_program(program).reference_schedule, cells, axis=1, lanes=2)
    with pytest.raises(ValueError):  # One byte cannot fund the concurrent lanes.
        stream_execution(program, cells, plan, payload_limits=(RegionPayloadLimit("cpu", 1),))
    # A generic callback stays opaque even when its code equals a kernel with a declared ABI.
    scan, native, _ = native_scan("cpu")
    assert parallel_plans(scan.program, {native.declaration.name: native})
    assert not parallel_plans(scan.program, {native.declaration.name: replace(native, implementation=None)})
    assert not parallel_plans(
        scan.program,
        {native.declaration.name: replace(native, implementation=replace(native.implementation, host_reads=True))},
    )
    request = RegionSelectionRequest(
        import_region_functions(program, cells),
        (
            RegionCandidate(
                analyze_region_program(program).reference_schedule, import_region_functions(program, cells), (100,)
            ),
            RegionCandidate(
                plan.phase_schedule, import_region_functions(program, cells), (1,), stream_plan=replace(plan, slots=())
            ),
        ),
        limits(plan),
        False,
        "tampered window",
    )
    with pytest.raises(ValueError):
        select_region_candidate(program, request)


def test_selection_charges_concurrent_scratch_and_the_default_mapping_honors_budgets():
    program, cell, plan = grid()
    cells = {cell.digest: cell}
    source = analyze_region_program(program)
    imports = import_region_functions(program, cells)
    reference = RegionCandidate(source.reference_schedule, imports, (100,))
    parallel = RegionCandidate(plan.phase_schedule, imports, (1,), stream_plan=plan)
    request = RegionSelectionRequest(imports, (reference, parallel), limits(plan), False, "deterministic ranking")
    assert select_region_candidate(program, request).selected == 1
    # The ordinary phase payload fits, but its concurrent scratch does not.
    phase_budget = (RegionPayloadLimit("cpu", plan.payload.devices[0].peak_bytes_upper_bound),)
    constrained = select_region_candidate(program, replace(request, payload_limits=phase_budget))
    assert constrained.selected == 0 and not constrained.candidates[1].feasible
    # Without measurements the default mapping uses lanes only when the budget can fund them.
    tight = plan_region_schedule(program, cells, parallel=True, payload_limits=phase_budget)
    assert tight is not None and tight.stream_plan is None
    funded = plan_region_schedule(program, cells, parallel=True, payload_limits=limits(plan))
    assert funded.stream_plan is not None
    assert plan_region_schedule(program, cells, parallel=True, payload_limits=(RegionPayloadLimit("cpu", 1),)) is None


@requires_cuda
def test_wavefront_launches_on_borrowed_streams_and_keeps_exact_versions():
    program, cell, plan = grid("cuda:0", chunks=12, depth=2)
    owner, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    inputs = inputs_for(12, 5, "cuda:0")
    owner.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(owner):
        reference = bind(program, {cell.digest: cell}, analyze_region_program(program).reference_schedule).run(inputs)
    launches, intervals = [], []
    origin = torch.cuda.Event(enable_timing=True)
    origin.record(owner)
    with torch.cuda.stream(owner):
        torch.cuda._sleep(60_000_000)

    def observed(*args):
        lane = torch.cuda.current_stream().cuda_stream
        launches.append(lane)
        if len(launches) == 1:
            with pytest.raises(WindowBackpressure):
                execution.run(inputs, streams=RegionStreamBindings(owner, (auxiliary,)))
            assert len(launches) == 1
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(4_000_000)
        result = cell.function(*args)
        end.record()
        intervals.append((lane, start, end))
        return result

    execution = stream_execution(
        program, {cell.digest: replace(cell, function=observed)}, plan, payload_limits=limits(plan)
    )
    with no_host_waits("two_stream_window"):
        with pytest.raises(ValueError):
            execution.run(inputs, streams=RegionStreamBindings(owner, (owner,)))
        assert not launches
        result = execution.run(inputs, streams=RegionStreamBindings(owner, (auxiliary,)))
    inputs.clear()  # The caller may drop its inputs as soon as submission returns.
    poison = [torch.full((4,), 999.0, dtype=torch.float64, device="cuda:0") for _ in range(17)]
    assert len(poison) == 17
    assert result.completion is not None
    result.completion.order_on(torch.cuda.current_stream())
    owner.synchronize()
    spans = [(lane, origin.elapsed_time(start), origin.elapsed_time(end)) for lane, start, end in intervals]
    overlap = any(
        a != b and max(start, other_start) < min(end, other_end)
        for a, start, end in spans
        for b, other_start, other_end in spans
    )
    assert overlap and set(launches) == {owner.cuda_stream, auxiliary.cuda_stream}
    assert result.invocations == 60 and result.window_reuses > 0
    for got, want in zip(result.values, reference.values, strict=True):
        for a, b in zip(got.elements, want.elements, strict=True):
            assert torch.equal(a.contiguous().view(torch.int64), b.contiguous().view(torch.int64))
    print(
        json.dumps(
            {"case": "two_stream_window", "concurrent_intervals": overlap, "window_reuses": result.window_reuses}
        )
    )


@requires_cuda
def test_failure_orders_already_launched_auxiliary_work_before_reuse():
    program, cell, plan = grid("cuda:0")
    owner, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    completions, inject = [], [True]

    def failing(*args):
        torch.cuda._sleep(40_000_000)
        result = cell.function(*args)
        if torch.cuda.current_stream() == auxiliary:
            event = torch.cuda.Event()
            event.record()
            completions.append(event)
            if inject[0]:
                raise RuntimeError("injected after auxiliary launch")
        return result

    runner = stream_execution(
        program, {cell.digest: replace(cell, function=failing)}, plan, payload_limits=limits(plan)
    )
    inputs = inputs_for(3, 5, "cuda:0")
    owner.wait_stream(torch.cuda.current_stream())
    with no_host_waits("pending_cancellation_and_reuse"):
        with pytest.raises(RuntimeError, match="injected"):
            runner.run(inputs, streams=RegionStreamBindings(owner, (auxiliary,)))
        assert completions and not all(event.query() for event in completions)
        inject[0] = False
        result = runner.run(inputs, streams=RegionStreamBindings(owner, (auxiliary,)))
    owner.synchronize()
    reference = bind(program, {cell.digest: cell}, analyze_region_program(program).reference_schedule).run(inputs)
    for got, want in zip(result.values, reference.values, strict=True):
        for a, b in zip(got.elements, want.elements, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert result.window_reuses > 0


@requires_cuda
def test_page_effects_stay_on_the_original_pin_stream():
    owner, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    execution, task, rows, refs = owner_fixture(rounds=12, stream=owner)
    program, cells = task.program, task.region.cells
    candidate = next(c for c in analyze_region_program(program).candidates if c.source == "isl-parallel")
    plan = lower_region_streams(program, candidate.schedule_map, cells, axis=0, lanes=2, pipeline_depth=3)
    runner = stream_execution(program, cells, plan, payload_limits=limits(plan))
    task.region, task.streams = runner, RegionStreamBindings(owner, (auxiliary,))
    anchor = execution.arena.plan.schedule_physical.tasks[0].anchor_id
    invocation = execution.start_step(rows, owner)
    pages = task.pages(invocation.context)
    try:
        with pytest.raises(ValueError):  # Page effects may not move to another owner stream.
            runner.run({}, pages=pages, streams=RegionStreamBindings(auxiliary, (owner,)))
        context, arena = invocation.context, execution.arena
        field = LayoutFieldId(0)
        cancelled = arena.prepare_transition((refs[0],), auxiliary, fields=((field,),), copy_fields=())
        with torch.cuda.stream(auxiliary):
            torch.cuda._sleep(60_000_000)
            arena.write_field(cancelled, 0, field).fill_(1000)
        with no_host_waits("private_write_abort_and_reuse"):
            with pytest.raises(WindowBackpressure, match="pending writer"):
                execution.execute_anchor(invocation, anchor)
            assert task.result is None and not context.cancel_requested.is_set()
            arena.abort(cancelled)
            assert arena.latest(refs[0].lineage_id) == refs[0]
            assert arena.usage().private_versions == 0 and arena.usage().pending_receipts > 0
            token = context.prepare_owned_write((refs[0],), ((field,),))
            with torch.cuda.stream(owner):
                arena.write_field(token, 0, field).fill_(1)
        receipt = context.publish_owned_write(token)
        context.settle_owned_publication(receipt)
        held = context.borrow_published(tuple(receipt.after_refs))
        before = tuple(arena.latest(ref.lineage_id) for ref in refs)
        with no_host_waits("ptts_window_backpressure"):
            with pytest.raises(WindowBackpressure, match="complete window"):
                execution.execute_anchor(invocation, anchor)
            assert context.release is None and not context.cancel_requested.is_set()
            assert task.result is None
            assert tuple(arena.latest(ref.lineage_id) for ref in refs) == before
            context.release_owned_pin(held)
        with torch.cuda.stream(owner):
            torch.cuda._sleep(300_000_000)
        with no_host_waits("ptts_capacity_and_publication"):
            execution.execute_anchor(invocation, anchor)
            old, first, second = task.result.values
            assert tuple(arena.latest(ref.lineage_id).version_id for ref in refs) == (13, 12)
            with pytest.raises(WindowBackpressure, match="previous page window"):
                with arena.reserve_window((), (), (), owner):
                    pytest.fail("pending receipts must backpressure another page window")
            execution.cancel_step(invocation)
        with torch.cuda.stream(owner):
            for value, expected in ((old, 1), (first, 13), (second, 14)):
                torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    finally:
        with no_host_waits("retirement"):
            if invocation.context.release is None:
                execution.cancel_step(invocation)
            execution.retire(tuple(row.identity for row in rows))
        execution.arena.collect_completed(wait=True)
    assert execution.arena.usage().pending_receipts == execution.arena.active_lineages == 0


@requires_cuda
def test_tuned_parallel_region_matches_the_serial_execution():
    program, cell, _ = grid("cuda:0")
    streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
    inputs = inputs_for(3, 5, "cuda:0")
    streams.owner.wait_stream(torch.cuda.current_stream())
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
    serial = prepare_region_execution(
        program,
        {cell.digest: cell},
        payload_limits=selected.request.payload_limits,
        options=replace(options, schedule_policy="serial"),
    )
    assert serial.stream_plan is None
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
    owner = torch.cuda.Stream()
    execution, _, rows, _ = owner_fixture(rounds=1, stream=owner)
    invocation = execution.start_step(rows, owner)
    streams = RegionStreamBindings(owner, (torch.cuda.Stream(),))
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
    abi = NativeKernelImport(function, "test.borrowed_weights.current_stream@1", "cuda:0", (), host_reads=False)
    cells = {function.name: external_primitive(function, abi)}
    plan = lower_region_streams(program, "{ S0[] -> [0]; S1[] -> [1] }", cells, axis=0, lanes=2)
    runner = stream_execution(program, cells, plan, payload_limits=limits(plan))
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
        with pytest.raises(ValueError):  # A different local kernel is not the planned one.
            runner.run({}, pages=resources(replace(abi, implementation="another.kernel@1")), streams=streams)
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

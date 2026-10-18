"""Physical concurrency uses the same AER, numerical policy and replay boundary."""

from dataclasses import replace

import pytest
import torch

from pllm import _core
from pllm.config import ExecutionOptions
from pllm.contracts.region_program import SizeParameter
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_selection import (
    RegionCandidate,
    RegionPayloadLimit,
    RegionSelectionRequest,
    RegionTuningSize,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.exported_cell import ExportedCell, NumericalTransformWarning
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_execution import RegionExecutionPlan
from pllm.execution.region_optimization import (
    PreparedRegionCandidate,
    bind_region_candidate,
    optimize_region,
    prepare_region_execution,
    select_region_candidate,
    warn_region_transforms,
)
from pllm.execution.region_storage import import_region_functions
from pllm.execution.region_streams import RegionStreamBindings, explore_region_streams, lower_region_streams
from pllm.execution.tensor_compilation import compile_tensor_cell
from pllm.execution.tensor_scan import TensorScan
from test.boundary.test_region_streams import grid, limits
from test.boundary.test_region_tensor import inputs_for
from test.boundary.test_tensor_rewrites import reassociation_candidates


def test_aer_checks_concurrent_peak_before_ranking_and_rejects_changed_physical_evidence():
    program, cell, source, plan = grid()
    imports = import_region_functions(program, {cell.digest: cell})
    reference = RegionCandidate(source.reference_schedule, imports, (100,))
    parallel = RegionCandidate(plan.phase_schedule, imports, (1,), stream_plan=plan)
    request = RegionSelectionRequest(imports, (reference, parallel), limits(plan), False, "deterministic ranking")
    assert decode_boundary(encode_boundary(request), RegionSelectionRequest) == request
    selected = select_region_candidate(program, request)
    assert selected.selected == 1
    assert selected.candidates[1].payload == plan.payload
    RegionExecutionPlan(program, imports, RegionScheduleProposal(plan.payload, plan), limits(plan))
    assert selected.candidates[1].peak_bytes == limits(plan)
    assert not selected.candidates[1].changed_functions
    # The ordinary phase payload fits, but its concurrent scratch does not.
    phase_budget = (RegionPayloadLimit("cpu", plan.payload.devices[0].peak_bytes_upper_bound),)
    constrained = select_region_candidate(program, replace(request, payload_limits=phase_budget))
    assert constrained.selected == 0 and not constrained.candidates[1].feasible
    slower = replace(request, candidates=(reference, replace(parallel, elapsed_ns=(200,))))
    assert select_region_candidate(program, slower).selected == 0
    bad_plan = replace(plan, devices=(replace(plan.devices[0], peak_bytes_upper_bound=0),))
    for bad in (replace(parallel, stream_plan=bad_plan), replace(parallel, schedule_map=source.reference_schedule)):
        with pytest.raises(ValueError, match="stream plan|phase lifetimes"):
            select_region_candidate(program, replace(request, candidates=(reference, bad)))
    # Matching outer and nested claims must not bypass source reconstruction.
    # The stream verifier owns capacity, retention and combined device bounds.
    for payload in (
        replace(plan.payload, capacity=replace(plan.payload.capacity, roots=())),
        replace(plan.payload, retention=replace(plan.payload.retention, signature="0" * 64)),
        replace(plan.payload, devices=()),
    ):
        forged = replace(plan, payload=payload)
        with pytest.raises(ValueError, match="stream plan differs"):
            select_region_candidate(
                program, replace(request, candidates=(reference, replace(parallel, stream_plan=forged)))
            )
    with pytest.raises(ValueError, match="same payload"):
        RegionExecutionPlan(
            program, imports, RegionScheduleProposal(replace(plan.payload, devices=()), plan), limits(plan)
        )
    with pytest.raises(ValueError, match="original reference"):
        select_region_candidate(program, replace(request, candidates=(parallel,)))
    for invalid in (
        replace(request, source_imports=()),
        replace(request, source_imports=(), candidates=(replace(reference, imports=()), parallel)),
    ):
        with pytest.raises(ValueError):
            select_region_candidate(program, invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tuner_measures_parallel_candidates_and_rebinds_without_runtime_search(monkeypatch):
    import pllm.execution.region_optimization as tuning

    program, cell, _, _ = grid("cuda:0")
    streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
    inputs = inputs_for(3, 5, "cuda:0")
    streams.owner.wait_stream(torch.cuda.current_stream())
    measured = []
    measure = tuning._measure

    def measured_rank(execution, *args):
        elapsed = measure(execution, *args)
        assert elapsed > 0
        measured.append(execution.stream_plan is not None)
        # Deterministic ranking exercises the parallel winner without making
        # a performance claim about this tiny launch-bound example.
        return 1 if execution.stream_plan is not None else 100

    monkeypatch.setattr(tuning, "_measure", measured_rank)
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
    assert selected.request.candidates[selected.selection.selected].stream_plan is not None
    sequential = prepare_region_execution(
        program,
        {cell.digest: cell},
        payload_limits=selected.request.payload_limits,
        options=ExecutionOptions(allow_numerical_transforms=False, schedule_policy="serial"),
    )
    assert sequential.stream_plan is None
    # Make serial win using the same measured candidate evidence. Both bindings
    # execute on the original owner when given the same available streams.
    serial_request = replace(
        selected.request,
        candidates=tuple(
            replace(c, elapsed_ns=(1 if i == 0 else 100,)) for i, c in enumerate(selected.request.candidates)
        ),
    )
    serial_choice = select_region_candidate(program, serial_request)
    serial = bind_region_candidate(
        PreparedRegionCandidate(program, serial_request.candidates[serial_choice.selected], (), {cell.digest: cell}),
        serial_request.payload_limits,
        serial_choice.candidates[serial_choice.selected],
    )

    def forbidden(*args):
        raise AssertionError("selected execution must not search or recompile")

    for name in ("explore_region_streams", "select_region_candidate", "lower_region_schedule"):
        monkeypatch.setattr(_core, name, forbidden)
    with pytest.raises(ValueError, match="must be supplied together"):
        selected.execution.run(inputs)
    got, want = selected.execution.run(inputs, streams=streams), serial.run(inputs, streams=streams)
    with torch.cuda.stream(streams.owner):
        for left, right in zip(got.values, want.values, strict=True):
            for a, b in zip(left.elements, right.elements, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    streams.owner.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_reassociation_and_parallel_placement_share_numerical_replay():
    tensors = tuple(torch.ones(shape, device="cuda:0") for shape in ((4, 3), (3, 16), (16, 2)))
    cell = ExportedCell.capture(lambda a, b, c: ((a @ b) @ c,), ("a", "b", "c"), ("out",), tensors)
    scan = TensorScan(cell, SizeParameter("steps", 0, 5), carry=0, invariants=3)
    program = scan.program
    (variant,) = reassociation_candidates(cell)
    compiled = compile_tensor_cell(variant.cell)
    originals = {cell.digest: cell}
    changed = {cell.digest: compiled}
    source = analyze_region_program(program, optimize=False)
    (plan,) = explore_region_streams(
        program, changed, schedule=analyze_region_program(program, parallel_only=True).candidates[-1].schedule_map
    ).candidates
    plan = lower_region_streams(program, plan.schedule_map, changed, axis=plan.axis, lanes=2, pipeline_depth=2)
    imports = import_region_functions(program, originals)
    candidate = RegionCandidate(plan.phase_schedule, import_region_functions(program, changed), (1,), stream_plan=plan)
    request = RegionSelectionRequest(
        imports,
        (RegionCandidate(source.reference_schedule, imports, (100,)), candidate),
        (RegionPayloadLimit("cuda:0", 1 << 20),),
        True,
        "combined mathematical and physical ranking",
        (RegionTuningSize("steps", 3),),
    )
    assert select_region_candidate(program, replace(request, allow_numerical_transforms=False)).selected == 0
    selection = select_region_candidate(program, request)
    with pytest.warns(NumericalTransformWarning):
        selected = bind_region_candidate(
            PreparedRegionCandidate(program, candidate, (variant.rewrite,), changed),
            request.payload_limits,
            selection.candidates[selection.selected],
        )
        warn_region_transforms(selected, selection.candidates[selection.selected], selection.signature)
    assert selection.selected == 1
    streams = RegionStreamBindings(torch.cuda.Stream(), (torch.cuda.Stream(),))
    streams.owner.wait_stream(torch.cuda.current_stream())
    inputs = dict(zip(("arg0", "arg1", "arg2"), tensors, strict=True))
    for count in (0, 3, 5):
        result = selected.run(inputs, sizes={"steps": count}, streams=streams)
        assert result.invocations == count
        (bank,) = result.values
        assert len(bank.elements) == count
        with torch.cuda.stream(streams.owner):
            for value in bank.elements:
                torch.testing.assert_close(value, torch.full_like(value, 48), rtol=0, atol=0)
    streams.owner.synchronize()

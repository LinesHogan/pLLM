"""Source windows bound recovery retention before measured AER selection."""

from dataclasses import replace

import pytest

from pllm import _core
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.region_program import TensorType
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_selection import (
    RegionCandidate,
    RegionPayloadLimit,
    RegionSelectionRequest,
    RegionTuningSize,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_compilation import (
    analyze_region_program,
    analyze_region_retention,
    explore_region_schedules,
    lower_region_schedule,
)
from pllm.execution.region_optimization import PreparedRegionCandidate, bind_region_candidate, select_region_candidate
from pllm.execution.region_schedule import RegionScheduleExecution
from pllm.execution.region_storage import analyze_region_payload, import_region_functions
from pllm.execution.replay import Recurrence
from pllm.execution.tokenwise_region import TokenwiseRegion


def test_bounded_model_window_is_eligible_when_faster_unbounded_candidate_exceeds_budget(monkeypatch):
    region = TokenwiseRegion(
        TensorType((1, 4), "torch.float32"),
        8,
        CausalModelDomain(1, 3, 8),
        Recurrence().effect(3),
        lambda a, b, i: a + b,
    )
    analysis = analyze_region_program(region.program)
    # Window families enumerate constructor-faithfully; the shared catalog's
    # merge holds semantic dedup and keeps the genuinely windowed candidate.
    catalog = explore_region_schedules(region.program)
    candidate = next(candidate for candidate in catalog.candidates if "-window-2-" in candidate.source)
    bound = analyze_region_retention(region.program, candidate.schedule_map, region.retained)
    assert bound.allocations[0].peak_instances_upper_bound == 2
    runner = RegionScheduleExecution(region.program, lower_region_schedule(region.program, candidate.schedule_map))
    original = RegionScheduleExecution(
        region.program, lower_region_schedule(region.program, analysis.reference_schedule)
    )
    for count in (0, 1, 3, 7, 8):
        sizes = {"tokens0": count, "prefix0": 0}
        actual, expected = tuple(runner.invocations(sizes)), tuple(original.invocations(sizes))
        assert set(actual) == set(expected) and len(actual) == len(expected)
        windows_seen = tuple(point.coordinates[0] // 2 for point in actual)
        assert windows_seen == tuple(sorted(windows_seen))
        if count > 2:
            assert actual != expected
    cells = region.functions
    imports = import_region_functions(region.program, cells)
    payload = analyze_region_payload(region.program, candidate.schedule_map, cells, region.retained)
    limit = payload.devices[0].peak_bytes_upper_bound
    proposals = decode_boundary(
        encode_boundary(tuple(RegionScheduleProposal(p) for p in (region.payload, payload))),
        tuple[RegionScheduleProposal, ...],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("worker validation must not regenerate a schedule catalog")

    with monkeypatch.context() as worker:
        for name in ("explore_region_schedules", "explore_region_windows", "explore_region_tiling"):
            worker.setattr(_core, name, forbidden)
        rebuilt = region.with_catalog(proposals, limit)
    assert rebuilt.candidate_proposals == proposals and region.candidate_proposals == (
        RegionScheduleProposal(region.payload),
    )
    with pytest.raises(ValueError, match="payload reservation"):
        region.with_catalog(proposals, limit - 1)
    # Fixed scores isolate the hard capacity constraint. This is not a timing
    # claim: production must supply completed measurements from its backend.
    request = RegionSelectionRequest(
        imports,
        (
            RegionCandidate(analysis.reference_schedule, imports, (100,)),
            RegionCandidate(candidate.schedule_map, imports, (10,)),
            RegionCandidate(analysis.candidates[-1].schedule_map, imports, (1,)),
        ),
        (RegionPayloadLimit("cpu", limit),),
        True,
        "window capacity oracle",
        (RegionTuningSize("prefix0", 0), RegionTuningSize("tokens0", 3)),
        region.retained,
    )
    selection = select_region_candidate(region.program, request)
    assert selection.selected == 1
    assert selection.candidates[0].feasible and selection.candidates[1].feasible
    assert not selection.candidates[2].feasible
    assert selection.candidates[1].payload.retention == bound
    # Costs still include retention when the tensor subtotal alone fits.
    assert (
        selection.candidates[2].payload.devices[0].retained_bytes_upper_bound
        > payload.devices[0].retained_bytes_upper_bound
    )
    # A valid AER decision with a caller-provided larger budget is not authority
    # to exceed the original model attachment's reservation.
    inflated = replace(request, payload_limits=(RegionPayloadLimit("cpu", limit + 1),))
    with pytest.raises(ValueError, match="original attachment payload reservation"):
        region.with_execution(
            bind_region_candidate(
                PreparedRegionCandidate(region.program, inflated.candidates[selection.selected], (), cells),
                inflated.payload_limits,
                selection.candidates[selection.selected],
            )
        )
    funded = replace(
        request,
        payload_limits=tuple(RegionPayloadLimit(d.device, d.peak_bytes_upper_bound) for d in region.payload.devices),
    )
    funded_selection = select_region_candidate(region.program, funded)
    rebound = region.with_execution(
        bind_region_candidate(
            PreparedRegionCandidate(region.program, funded.candidates[funded_selection.selected], (), cells),
            funded.payload_limits,
            funded_selection.candidates[funded_selection.selected],
        )
    )
    assert rebound is not region and rebound.execution is not region.execution
    assert rebound.program is region.program and rebound.functions is region.functions

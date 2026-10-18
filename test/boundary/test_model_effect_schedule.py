"""Actual ISL candidates move model work while preserving the trace sequence."""

from dataclasses import replace

import pytest

from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.control_instructions import ControlIf, ControlPrimitive
from pllm.contracts.region_program import BorrowedResource, Call, TensorType
from pllm.contracts.region_retention import RetainedAllocation, RetentionEndpoint
from pllm.execution.host_borrow import TraceResourceView, validate_host_views
from pllm.execution.region_compilation import analyze_region_program, analyze_region_retention, lower_region_schedule
from pllm.execution.region_schedule import RegionScheduleExecution
from pllm.execution.region_storage import analyze_region_payload
from pllm.execution.replay import Recurrence
from pllm.execution.tokenwise_region import TokenwiseRegion
from test.boundary.test_region_pages import owner_fixture


def test_model_candidates_preserve_trace_but_require_more_recovery_capacity():
    region = TokenwiseRegion(
        TensorType((1, 4), "torch.float32"),
        8,
        CausalModelDomain(2, 3, 8),
        Recurrence().effect(3),
        lambda a, b, i: a + b,
    )
    analysis = analyze_region_program(region.program)
    assert not any(s.opaque for s in analysis.statements)
    functions = {f.name: f for f in region.program.functions}

    def traces(action):
        if isinstance(action, ControlPrimitive) and isinstance(action.operation, Call):
            return "trace" in functions[action.operation.function].resources
        if isinstance(action, ControlIf):
            return any(
                traces(step.action) for branch in (action.then_block, action.else_block) for step in branch.steps
            )
        return False

    reference_plan = lower_region_schedule(region.program, analysis.reference_schedule)
    trace_sites = {i for i, action in enumerate(reference_plan.actions) if traces(action)}
    assert trace_sites
    sizes = {"tokens0": 3, "tokens1": 1, "prefix0": 0, "prefix1": 0}
    imports = (
        RetainedAllocation(
            "recovery", RetentionEndpoint(region.begin.name, 0), RetentionEndpoint(region.complete.name, 0), "cpu", 1
        ),
    )
    reference, changed = None, False
    for candidate in analysis.candidates:
        bound = analyze_region_retention(region.program, candidate.schedule_map, imports)
        order = tuple(
            RegionScheduleExecution(
                region.program, lower_region_schedule(region.program, candidate.schedule_map)
            ).invocations(sizes)
        )
        if reference is None:
            reference = order
            assert bound.allocations[0].peak_instances_upper_bound == 1
        else:
            assert tuple(p for p in order if p.statement_id in trace_sites) == tuple(
                p for p in reference if p.statement_id in trace_sites
            )
            changed |= order != reference and bound.allocations[0].peak_instances_upper_bound > 1
    assert changed  # Source-derived candidates, not a hand-written row order.
    # Compose both kinds of storage under the same source and schedule. Native
    # kernel internals remain explicitly unknown rather than charged as zero.
    payload = analyze_region_payload(region.program, analysis.candidates[-1].schedule_map, region.functions, imports)
    (device,) = payload.devices
    assert device.retained_bytes_upper_bound > 1
    assert device.peak_bytes_upper_bound == device.tensor_bytes_upper_bound + device.retained_bytes_upper_bound
    assert region.normal.first.name in payload.capacity.unknown_functions


def test_two_host_names_cannot_hide_the_same_original_trace():
    execution, _, _, _ = owner_fixture(rounds=1)
    view = TraceResourceView(BorrowedResource("trace", ()), execution.trace)
    validate_host_views((view,))
    with pytest.raises(ValueError, match="same original effect namespace"):
        validate_host_views((view, replace(view, resource=BorrowedResource("renamed", ()))))

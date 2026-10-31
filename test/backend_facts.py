"""The planner budget check workload tests share; facts come from pllm.testing.backend_facts."""

from dataclasses import replace

import pytest

from pllm import ExecutionOptions


def assert_finite_resources(workload, facts, *, main_fallback=False):
    """Rust admits the plan at its exact peak budget and rejects one byte less; returns the exact deployment."""
    bounded, declaration = workload.specialize(facts, ExecutionOptions(memory_budget_bytes=1 << 30))
    attachment = workload.plan_backend(bounded, declaration)
    peak = attachment.plan.schedule.peak_memory_bytes
    assert peak > 0
    if main_fallback and any(task.lane == "side" for task in attachment.plan.schedule.tasks):
        inline = workload.plan_backend(
            replace(bounded, timeline=replace(bounded.timeline, memory_budget_bytes=peak - 1)), declaration
        )
        assert all(task.lane == "main" for task in inline.plan.schedule.tasks)
        peak = inline.plan.schedule.peak_memory_bytes
    exact = replace(bounded, timeline=replace(bounded.timeline, memory_budget_bytes=peak))
    deployment = workload.bind_backend(exact, workload.plan_backend(exact, declaration), declaration)
    with pytest.raises(ValueError):
        workload.plan_backend(
            replace(exact, timeline=replace(exact.timeline, memory_budget_bytes=peak - 1)), declaration
        )
    return deployment

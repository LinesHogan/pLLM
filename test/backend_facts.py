"""The planner budget check workload tests share; facts come from pllm.testing.backend_facts."""

from dataclasses import replace

import pytest

from pllm import ExecutionOptions


def assert_finite_resources(workload, facts, *, main_fallback=False):
    """Rust admits the plan at its exact peak budget and rejects one byte less."""
    bounded = workload.specialize(facts, ExecutionOptions(memory_budget_bytes=1 << 30))
    attachment, _ = workload.plan_backend(bounded)
    peak = attachment.plan.schedule.peak_memory_bytes
    assert peak > 0
    if main_fallback and any(task.lane == "side" for task in attachment.plan.schedule.tasks):
        inline, _ = workload.plan_backend(
            replace(bounded, timeline=replace(bounded.timeline, memory_budget_bytes=peak - 1))
        )
        assert all(task.lane == "main" for task in inline.plan.schedule.tasks)
        peak = inline.plan.schedule.peak_memory_bytes
    exact = replace(bounded, timeline=replace(bounded.timeline, memory_budget_bytes=peak))
    workload.bind_backend(exact, *workload.plan_backend(exact))
    with pytest.raises(ValueError):
        workload.plan_backend(replace(exact, timeline=replace(exact.timeline, memory_budget_bytes=peak - 1)))
    return exact

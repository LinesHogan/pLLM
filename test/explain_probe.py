"""Compare CPU explanation with actual model geometry and load-time planning facts in GPU tests."""

from dataclasses import replace

from pllm.backends.vllm.model_shape import ModelShape
from pllm.backends.vllm.runtime import VllmLlm, probe_backend
from pllm.config import ModelConfig
from pllm.execution.tensor_binding import TensorOutput
from pllm.execution.tensor_frame import TensorFrame
from pllm.explanation import explain
from pllm.usage import Workload


def assert_explanation(workload: Workload, config: ModelConfig, handle: VllmLlm[TensorFrame, TensorOutput]) -> None:
    """Model facts and funded bytes agree; only explicitly excluded GPU BLAS scratch is normalized."""
    report = explain(workload, config)
    requirements = workload.backend_requirements()
    runner = handle.runner
    assert runner is not None
    assert report.model == ModelShape.of_model(runner.decoder, requirements.parameter_targets, runner.dtype)
    assert report.eager_reason == handle.diagnostics().eager_reason
    facts = probe_backend(requirements, runner, publish_lag=config.execution.publish_lag)
    facts = replace(facts, probe=replace(facts.probe, blas_workspace_bytes=0, device_index=0))
    facts, declaration = workload.specialize(facts, config.execution)
    plan = workload.plan_backend(facts, declaration).plan
    assert tuple(anchor.reserved_bytes for anchor in report.anchors) == plan.timeline.reserved_bytes
    assert report.peak_memory_bytes == plan.schedule.peak_memory_bytes
    assert sum(state.reserved_bytes for state in report.states) == handle.attachment.plan.states.reserved_bytes
    assert {
        (task.name, task.anchor, task.lane, task.read_origin, task.read_distance)
        for member in report.members
        for task in member.tasks
    } | {
        (task.name, task.anchor, task.lane, task.read_origin, task.read_distance) for task in report.framework_tasks
    } == {
        (
            plan.semantic_graph.tasks[task.task_id].symbol,
            plan.timeline.anchors[task.anchor_id].symbol,
            task.lane,
            task.read_origin,
            task.read_distance,
        )
        for task in plan.schedule.tasks
    }
    if not requirements.parameter_targets:
        assert report.peak_memory_bytes == handle.attachment.plan.schedule.peak_memory_bytes

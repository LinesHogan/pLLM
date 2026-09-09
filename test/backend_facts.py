"""Original backend facts for native execution and page-arena oracles."""

from pllm.authoring import BackendCompileFacts
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode, AttachmentHiddenResource, AttachmentTopology, BackendPlacementProbe,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.semantic_graph import SchedulerTimeline
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts


def backend_facts(hidden_size: int, hidden: tuple[PackedHiddenResourceBinding, ...],
                  timeline: SchedulerTimeline, provider: PageBundleProviderFacts,
                  *, max_lineages: int = 2, memory_bytes: int = 10000) -> BackendCompileFacts:
    return BackendCompileFacts(
        BackendPlacementProbe("test.native@1", "a" * 64, "b" * 64, "c" * 64,
            AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER,
            provider.device_kind, provider.device_ordinal, 1, 1, TensorDType.FLOAT32,
            hidden_size, max_lineages * 2,
            tuple(AttachmentHiddenResource(binding, HiddenPhase.DECODE)
                for binding in sorted(hidden, key=lambda item: item.locator_key)), ()),
        timeline, (provider,), max_lineages, memory_bytes, 7)


def assert_capture_preflight(workload, facts, monkeypatch) -> None:
    """Real CUDA plan/binding, no CUDA allocation: reject a one-byte capture entitlement."""
    from dataclasses import replace
    from types import SimpleNamespace
    import pytest
    import torch
    from pllm import Workload
    from pllm.config import ExecutionOptions
    from pllm.contracts.state_representation import StateDeviceKind
    from pllm.execution.runtime_facts import RuntimeFacts

    facts = workload.specialize(replace(facts,
        probe=replace(facts.probe, device_kind=StateDeviceKind.CUDA),
        timeline=RuntimeFacts(torch.device("cuda", 0)).timeline(("state/dense_page_cuda@1",)),
        state_memory_budget_bytes=32 * 1024**2,
        providers=tuple(replace(provider, device_kind=StateDeviceKind.CUDA) for provider in facts.providers)),
        ExecutionOptions(capture=True, capture_memory_bytes=1, training_rows=4))
    attachment = workload.plan_backend(facts)
    with monkeypatch.context() as guarded:
        # Initialization never runs here; do not initialize a CUDA RNG on the CPU oracle.
        functions = Workload._functions
        guarded.setattr(Workload, "_functions", lambda self, binding, device, seed:
            functions(self, binding, torch.device("cpu"), seed))
        for name in ("zeros", "ones", "arange"):
            original = getattr(torch, name)
            def allocate(*args, _original=original, **kwargs):
                assert torch.device(kwargs["device"]).type == "meta", "allocation before capture preflight"
                return _original(*args, **kwargs)
            guarded.setattr(torch, name, allocate)
        math = workload.bind_math(facts, attachment.inputs.plan)
        assert math.captures
        # Only the original plan is needed: rejection must precede any arena operation.
        with pytest.raises(ValueError, match="capture inputs and field scratch"):
            math.bind(SimpleNamespace(plan=attachment.inputs.plan), attachment.inputs.request.sampler_anchor_id)


def assert_finite_resources(workload, facts):
    """Original Rust admission and worker binding, including re-signed underfunded claims."""
    from dataclasses import replace
    import pytest
    from pllm import ExecutionOptions

    bounded = workload.specialize(facts, ExecutionOptions(memory_budget_bytes=1 << 30))
    attachment = workload.plan_backend(bounded)
    plan = attachment.inputs.plan
    assert len(plan.timeline.task_resources) == len(plan.semantic_graph.tasks)
    assert len(plan.timeline.value_resources) == sum(len(task.outputs) for task in plan.semantic_graph.tasks)
    peak = plan.schedule_decision.peak_memory_bytes
    assert peak > 0
    exact = replace(bounded, timeline=replace(bounded.timeline, memory_budget_bytes=peak))
    workload.bind_backend(exact, workload.plan_backend(exact))
    with pytest.raises(ValueError):
        workload.plan_backend(replace(exact, timeline=replace(exact.timeline, memory_budget_bytes=peak - 1)))
    largest = max(bounded.timeline.task_resources, key=lambda item: item.workspace_bytes)
    lowered = replace(bounded, timeline=replace(bounded.timeline, task_resources=tuple(
        replace(item, workspace_bytes=item.workspace_bytes - 1) if item is largest else item
        for item in bounded.timeline.task_resources)))
    with pytest.raises(ValueError, match="workspace|envelope"):
        workload.bind_backend(lowered, workload.plan_backend(lowered))
    for value in bounded.timeline.value_resources:
        if value.bytes:
            lowered = replace(bounded, timeline=replace(bounded.timeline, value_resources=tuple(
                replace(item, bytes=item.bytes - 1) if item is value else item for item in bounded.timeline.value_resources)))
            with pytest.raises(ValueError, match="output storage"):
                workload.bind_backend(lowered, workload.plan_backend(lowered))
            break
    return exact


def assert_auxiliary_preflight(attachment, auxiliary) -> None:
    """Exercise public submission and closed STAGE rejection before any storage transfer."""
    import os
    from types import SimpleNamespace
    from unittest.mock import patch
    import pytest
    from vllm import SamplingParams
    from pllm.backends.vllm.runtime import BackendCommand, BackendStage, VllmLlm
    from pllm.contracts.auxiliary import map_auxiliary
    from pllm.execution.codec import encode_boundary
    from pllm.execution.shared_tensor import CpuFileStorage, SharedTensor
    from pllm.worker import GPUWorker

    handle = SimpleNamespace(_closed=False, _runner=None, _next_prompt_index=0, attachment=attachment)
    session = SimpleNamespace(attachment=attachment)
    worker = SimpleNamespace(workload_runner=SimpleNamespace(require_session=lambda: session))
    wire = map_auxiliary(auxiliary, lambda tensor: SharedTensor(os.getpid(), SharedTensor.describe(tensor), CpuFileStorage(-1, 0)))
    with patch.object(SharedTensor, "export", side_effect=AssertionError("export before preflight")), \
         patch.object(SharedTensor, "map", side_effect=AssertionError("mapping before preflight")), \
         patch("torch.cuda.current_stream", side_effect=AssertionError("CUDA before preflight")):
        with pytest.raises(ValueError):
            VllmLlm.generate(handle, ["test"], SamplingParams(max_tokens=1), auxiliary=auxiliary)
        with pytest.raises(ValueError):
            GPUWorker.workload_command(worker, BackendCommand.STAGE.value, encode_boundary(BackendStage((), wire, None)))

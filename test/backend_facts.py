"""Backend compile facts for CPU/CUDA workload tests, and the planner budget check they share."""

from dataclasses import replace

import pytest

from pllm import ExecutionOptions
from pllm.authoring import BackendCompileFacts
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentTopology,
    BackendPlacementProbe,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.semantic_graph import SchedulerTimeline
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts


def backend_facts(
    hidden_size: int,
    hidden: tuple[PackedHiddenResourceBinding, ...],
    timeline: SchedulerTimeline,
    provider: PageBundleProviderFacts | None,
    *,
    max_lineages: int = 2,
    memory_bytes: int = 10000,
    max_rows: int | None = None,
    dtype: TensorDType = TensorDType.FLOAT32,
    **extra,
) -> BackendCompileFacts:
    """In-process eager facts; ``extra`` sets optional fields such as ``vocab_size`` or ``targets``."""
    return BackendCompileFacts(
        BackendPlacementProbe(
            "test.native@1",
            "a" * 64,
            "b" * 64,
            "c" * 64,
            AttachmentTopology.IN_PROCESS,
            AttachmentGraphMode.EAGER,
            StateDeviceKind.CPU if provider is None else provider.device_kind,
            0 if provider is None else provider.device_ordinal,
            1,
            1,
            dtype,
            hidden_size,
            max_lineages * 2 if max_rows is None else max_rows,
            tuple(
                AttachmentHiddenResource(binding, HiddenPhase.DECODE)
                for binding in sorted(hidden, key=lambda item: item.locator_key)
            ),
            (),
        ),
        timeline,
        () if provider is None else (provider,),
        max_lineages,
        memory_bytes,
        7,
        **extra,
    )


def assert_finite_resources(workload, facts):
    """Rust admits the plan at its exact peak budget and rejects one byte less."""
    bounded = workload.specialize(facts, ExecutionOptions(memory_budget_bytes=1 << 30))
    attachment = workload.plan_backend(bounded)
    peak = attachment.inputs.plan.schedule_decision.peak_memory_bytes
    assert peak > 0
    exact = replace(bounded, timeline=replace(bounded.timeline, memory_budget_bytes=peak))
    workload.bind_backend(exact, workload.plan_backend(exact))
    with pytest.raises(ValueError):
        workload.plan_backend(replace(exact, timeline=replace(exact.timeline, memory_budget_bytes=peak - 1)))
    return exact

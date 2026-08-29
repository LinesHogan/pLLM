"""Multiple native lifecycle services compose under one signed plan."""

from __future__ import annotations

from typing import cast
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.fttt import (
    FTTTConfig,
    FTTTFeedbackService,
    FTTTJudgeResult,
    FTTTTrainingBatch,
    FTTTWorkload,
)
from tllm.consumers.gtta import (
    GTTAConfig,
    GTTAEnvironmentService,
    GTTAWorkload,
    TrajectoryFeedback,
)
from tllm.runtime import residual_runtime
from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState


def test_two_stateful_workloads_execute_and_account_independently() -> None:
    fttt = FTTTWorkload(
        FTTTConfig(hidden_size=4, rank=2, feedback_feature_size=3)
    )
    gtta = GTTAWorkload(GTTAConfig())
    runtime = TllmRuntime(
        workloads=(fttt, gtta), workload_memory_budget_bytes=1 << 20
    )
    runtime.cold_specialize_workloads(
        RuntimeSpecializationSpec(1, 4, torch.float32, torch.device("cpu"))
    )
    compilation = runtime.compile_workloads(
        export_main_workload_timeline(
            BackendCapabilities(
                backend_id="vllm-0.27-mrv2",
                runner_family="mrv2",
                sampler_stages=frozenset(),
                native_min_p=False,
                tuple_sampler_return=False,
                split_execute_and_sample=True,
                graph_safety=GraphSafety.EAGER_ONLY,
            ),
            graph_mode="eager",
            static_headroom_bytes=1 << 20,
        )
    )
    feedback = FTTTFeedbackService(
        fttt.config,
        judge=lambda entry: FTTTJudgeResult(
            is_correct=False, feedback_text="wrong arithmetic"
        ),
    )
    feedback.enqueue(
        GenerationTraceBatch(
            (
                CompletionTraceEntry(
                    "req", 0, 0, 0, (1,), "wrong", "stop", "group"
                ),
            )
        ),
        torch.randn(1, 4),
    )
    batch = cast(FTTTTrainingBatch, feedback.next_batch())
    environment = GTTAEnvironmentService(gtta.config)
    environment.enqueue_feedback(
        TrajectoryFeedback("task", "act", reward=1.0)
    )
    runtime.stage_pair(
        FTTTWorkload.hidden_rows,
        batch.hidden_rows,
        FTTTWorkload.plans,
        batch.plan,
    )
    runtime.stage(GTTAWorkload.trajectories, environment.drain())
    state = ResidualRuntimeState()
    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation, bindings=runtime.workload_execution_bindings
        )
        residual_runtime.prepare_physical_workgraph_step(1)
        for region in compilation.decision.execution_regions:
            residual_runtime.advance_workload_anchor(region.placement_anchor)
        execution = state.workload_execution
        assert execution is not None
        assert execution.state_versions.observed_version(
            "fttt/edit_adapter", None
        ) == 1
        assert execution.state_versions.observed_version("gtta/dynamics", None) == 1
        assert execution.read_and_reset_stats(sync=False).loss_count == 1

"""Contracts and numerical oracle for the native FTTT workload."""

from __future__ import annotations

import unittest
from unittest import mock
from typing import cast

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.fttt import (
    FTTTConfig,
    FTTTFeedbackService,
    FTTTInstanceState,
    FTTTJudgeResult,
    FTTTTrainingBatch,
    FTTTWorkload,
)
from tllm.consumers.fttt.feedback import edit_inputs, target_delta
from tllm.runtime import residual_runtime
from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_bindings import LocalRuntimeServices, WorkerRuntimeBinding
from tllm.schedule import MainWorkloadTimeline, WorkloadCompilation
from tllm.workload import EffectKind, declare_workload


_HIDDEN = 4


def _spec() -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=2,
        hidden_size=_HIDDEN,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _timeline() -> MainWorkloadTimeline:
    return export_main_workload_timeline(
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


def _trace(*, correct: bool = False) -> GenerationTraceBatch:
    return GenerationTraceBatch(
        entries=(
            CompletionTraceEntry(
                request_id="req-0",
                prompt_idx=3,
                sample_idx=0,
                output_index=0,
                token_ids=(1, 2),
                text="correct" if correct else "wrong arithmetic",
                finish_reason="stop",
                group_id="round-0",
            ),
        )
    )


def _build(
    config: FTTTConfig,
) -> tuple[FTTTWorkload, TllmRuntime[FTTTWorkload], WorkloadCompilation]:
    workload = FTTTWorkload(config)
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=1 << 20
    )
    runtime.cold_specialize_workloads(_spec())
    return workload, runtime, runtime.compile_workloads(_timeline())


def _drive(
    runtime: TllmRuntime[FTTTWorkload],
    compilation: WorkloadCompilation,
    step_id: int = 9,
) -> int:
    state = ResidualRuntimeState()
    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation, bindings=runtime.workload_execution_bindings
        )
        residual_runtime.prepare_physical_workgraph_step(step_id)
        for region in compilation.decision.execution_regions:
            residual_runtime.advance_workload_anchor(region.placement_anchor)
        execution = state.workload_execution
        assert execution is not None
        return execution.state_versions.observed_version("edit_adapter", None)


class FTTTWorkloadUnitTest(unittest.TestCase):
    def test_config_constraints_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "constraint"):
            FTTTConfig(rank=0)
        with self.assertRaisesRegex(ValueError, "constraint"):
            FTTTConfig(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "constraint"):
            FTTTConfig(max_retries=-1)

    def test_service_queues_without_tensor_compute_or_early_state_mutation(self) -> None:
        service = FTTTFeedbackService(
            FTTTConfig(max_retries=2, feedback_feature_size=3),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                feedback_text="wrong arithmetic carry",
                judge="toy",
            ),
        )
        hidden = torch.randn(1, _HIDDEN)
        self.assertTrue(service.enqueue(_trace(), hidden))
        self.assertEqual(service.queued_feedback, 1)
        self.assertEqual(service.instance_state(3, request_id="req-0").retries, 0)

        batch = cast(FTTTTrainingBatch, service.next_batch())
        self.assertIs(batch.hidden_rows, hidden)
        self.assertEqual(batch.plan.selected_rows, (0,))
        self.assertEqual(batch.plan.intent.retry_controls.entries[0].action, "retry")
        self.assertEqual(batch.plan.intent.update_barriers.entries[0].state, "ready")

    def test_declaration_contains_only_semantic_dependencies_and_effects(self) -> None:
        declaration = declare_workload(
            FTTTWorkload(FTTTConfig(hidden_size=_HIDDEN, rank=2))
        ).ir
        self.assertEqual(
            set(declaration.schedule),
            {
                "prepare",
                "train",
                "train__optimize_edit_adapter",
                "predict",
                "publish",
            },
        )
        optimizer = next(
            task
            for task in declaration.tasks
            if task.name == "train__optimize_edit_adapter"
        )
        self.assertEqual(optimizer.effects[0].kind, EffectKind.TRANSITION_MAYBE)
        self.assertEqual(
            FTTTConfig.field_names(),
            (
                "enabled",
                "hidden_size",
                "rank",
                "learning_rate",
                "max_queue_size",
                "feedback_feature_size",
                "positive_reward_threshold",
                "max_retries",
                "edit_scale",
            ),
        )

    def test_signed_graph_matches_one_step_adam_oracle_and_publishes_controls(self) -> None:
        config = FTTTConfig(
            hidden_size=_HIDDEN,
            rank=2,
            learning_rate=0.05,
            feedback_feature_size=3,
        )
        workload, runtime, compilation = _build(config)
        service = FTTTFeedbackService(
            config,
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                feedback_text="wrong arithmetic carry",
                judge="toy",
            ),
        )
        hidden = torch.randn(1, _HIDDEN)
        self.assertTrue(service.enqueue(_trace(), hidden))
        batch = cast(FTTTTrainingBatch, service.next_batch())
        selected = hidden.index_select(0, torch.tensor(batch.plan.selected_rows))
        expected_input = edit_inputs(
            selected,
            batch.plan.intent.feedback_records,
            feature_size=config.feedback_feature_size,
        )
        expected_target = target_delta(
            selected,
            batch.plan.intent.feedback_records,
            feature_size=config.feedback_feature_size,
            edit_scale=config.edit_scale,
        )
        with torch.no_grad():
            down = workload.edit_adapter.down.weight
            down.copy_(torch.arange(down.numel(), dtype=down.dtype).reshape_as(down) / 10)
            workload.edit_adapter.up.weight.zero_()
        expected_down = down.detach().clone()
        rank_activation = expected_input @ expected_down.transpose(0, 1)
        prediction_gradient = -2.0 * expected_target / expected_target.numel()
        up_gradient = prediction_gradient.transpose(0, 1) @ rank_activation
        expected_up = -config.learning_rate * up_gradient / (
            up_gradient.abs() + 1e-8
        )
        expected_delta = rank_activation @ expected_up.transpose(0, 1)

        runtime.stage_pair(
            FTTTWorkload.hidden_rows,
            batch.hidden_rows,
            FTTTWorkload.plans,
            batch.plan,
        )
        self.assertEqual(_drive(runtime, compilation), 1)

        torch.testing.assert_close(workload.edit_adapter.down.weight, expected_down)
        torch.testing.assert_close(workload.edit_adapter.up.weight, expected_up)
        (actual_delta,) = runtime.drain(FTTTWorkload.edit_delta)
        torch.testing.assert_close(actual_delta, expected_delta)
        (published,) = runtime.drain(FTTTWorkload.closed_loop)
        self.assertEqual(published.adapter_version, 1)
        service.acknowledge(published)
        self.assertEqual(
            service.instance_state(3, request_id="req-0"),
            FTTTInstanceState(retries=1, adapter_version=1),
        )
        self.assertEqual(
            runtime.drain(FTTTWorkload.retry_controls)[0].entries[0].action,
            "retry",
        )
        services = cast(
            LocalRuntimeServices, workload.native_workload_resources().services
        )
        self.assertEqual(services.stats(False).loss_count, 1)

    def test_abort_rolls_back_projected_lifecycle_and_pending_queue(self) -> None:
        service = FTTTFeedbackService(FTTTConfig(max_retries=2))
        hidden = torch.randn(1, _HIDDEN)
        service.enqueue(_trace(), hidden)
        first = cast(FTTTTrainingBatch, service.next_batch())
        service.enqueue(_trace(), hidden)

        service.abort(first.plan.intent.transaction_id)

        self.assertEqual(service.instance_state(3, request_id="req-0").retries, 0)
        self.assertEqual(service.queued_feedback, 0)
        self.assertIsNone(service.next_batch())

    def test_positive_feedback_publishes_without_optimizer_transition(self) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2)
        workload, runtime, compilation = _build(config)
        service = FTTTFeedbackService(config)
        self.assertFalse(service.enqueue(_trace(correct=True), torch.randn(1, _HIDDEN)))
        batch = cast(FTTTTrainingBatch, service.next_batch())
        runtime.stage_pair(
            FTTTWorkload.hidden_rows,
            batch.hidden_rows,
            FTTTWorkload.plans,
            batch.plan,
        )

        self.assertEqual(_drive(runtime, compilation), 0)
        (empty_delta,) = runtime.drain(FTTTWorkload.edit_delta)
        self.assertEqual(empty_delta.shape, (0, _HIDDEN))
        published = runtime.drain(FTTTWorkload.closed_loop)[0]
        self.assertEqual(published.adapter_version, 0)
        service.acknowledge(published)
        services = cast(
            LocalRuntimeServices, workload.native_workload_resources().services
        )
        self.assertEqual(services.stats(False).loss_count, 0)

    def test_worker_reconstruction_has_one_atomic_input(self) -> None:
        workload = FTTTWorkload(FTTTConfig(hidden_size=_HIDDEN, rank=2))
        resources = workload.native_workload_resources()
        worker = cast(WorkerRuntimeBinding, resources.worker)
        rebuilt = FTTTWorkload.reconstruct(worker.spec.config)
        self.assertEqual(
            set(rebuilt.ports.inputs),
            {FTTTWorkload.hidden_rows.key, FTTTWorkload.plans.key},
        )
        self.assertEqual(
            set(rebuilt.ports.outputs),
            {
                FTTTWorkload.edit_delta.key,
                FTTTWorkload.reward_outcomes.key,
                FTTTWorkload.retry_controls.key,
                FTTTWorkload.update_barriers.key,
                FTTTWorkload.closed_loop.key,
            },
        )
if __name__ == "__main__":
    unittest.main()

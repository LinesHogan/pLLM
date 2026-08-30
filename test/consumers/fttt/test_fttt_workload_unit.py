"""Contracts and numerical oracle for the native FTTT workload."""

from __future__ import annotations

import copy
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
        config = FTTTConfig(
            hidden_size=_HIDDEN,
            rank=2,
            max_retries=2,
            feedback_feature_size=3,
        )
        _workload, runtime, _compilation = _build(config)
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
        self.assertEqual(service.queued_feedback, 1)
        self.assertEqual(service.instance_state(3, request_id="req-0").retries, 0)

        transaction = service.stage_next(runtime)
        assert transaction is not None
        batch = transaction.batch
        self.assertIs(batch.hidden_rows, hidden)
        self.assertEqual(batch.plan.selected_rows, (0,))
        self.assertEqual(batch.plan.intent.retry_controls.entries[0].action, "retry")
        self.assertEqual(batch.plan.intent.update_barriers.entries[0].state, "ready")
        transaction.abort()

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
        transaction = service.stage_next(runtime)
        assert transaction is not None
        batch = transaction.batch
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

        with transaction:
            self.assertEqual(_drive(runtime, compilation), 1)
            published = transaction.commit()

        torch.testing.assert_close(workload.edit_adapter.down.weight, expected_down)
        torch.testing.assert_close(workload.edit_adapter.up.weight, expected_up)
        (actual_delta,) = runtime.drain(FTTTWorkload.edit_delta)
        torch.testing.assert_close(actual_delta, expected_delta)
        self.assertEqual(published.adapter_version, 1)
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

    def test_abort_rolls_back_projected_lifecycle_and_discards_dependent_queue(
        self,
    ) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2, max_retries=2)
        _workload, runtime, _compilation = _build(config)
        service = FTTTFeedbackService(config)
        hidden = torch.randn(1, _HIDDEN)
        service.enqueue(_trace(), hidden)
        first = service.stage_next(runtime)
        assert first is not None
        service.enqueue(_trace(), hidden)

        with self.assertRaisesRegex(RuntimeError, "unresolved staged ports"):
            runtime.cold_specialize_workloads(_spec())
        with (
            mock.patch(
                "tllm.runtime.mp.control.set_active_control_plane"
            ) as reset_plane,
            self.assertRaisesRegex(RuntimeError, "unresolved staged ports"),
        ):
            runtime.make_llm(model_name="must-not-construct")
        reset_plane.assert_not_called()
        first.abort()

        self.assertEqual(service.instance_state(3, request_id="req-0").retries, 0)
        self.assertEqual(service.queued_feedback, 0)
        self.assertIsNone(service.stage_next(runtime))
        self.assertTrue(service.enqueue(_trace(), hidden))
        self.assertEqual(service.instance_state(3, request_id="req-0").retries, 0)
        replacement = service.stage_next(runtime)
        assert replacement is not None
        transition = replacement.batch.plan.intent.transitions[0]
        self.assertEqual(transition.previous, FTTTInstanceState())
        self.assertEqual(transition.next.retries, 1)
        replacement.abort()

    def test_staging_failure_aborts_active_and_dependent_feedback(self) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2, max_retries=2)
        _workload, runtime, _compilation = _build(config)
        service = FTTTFeedbackService(config)
        hidden = torch.randn(1, _HIDDEN)
        service.enqueue(_trace(), hidden)
        service.enqueue(_trace(), hidden)

        with (
            mock.patch.object(
                runtime,
                "stage_pair",
                side_effect=RuntimeError("injected staging failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "staging failure"),
        ):
            service.stage_next(runtime)

        self.assertEqual(service.queued_feedback, 0)
        self.assertEqual(
            service.instance_state(3, request_id="req-0"), FTTTInstanceState()
        )
        self.assertIsNone(service.stage_next(runtime))

    def test_exception_before_signed_consumption_cancels_staged_ports(self) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2)
        _workload, runtime, _compilation = _build(config)
        service = FTTTFeedbackService(config)
        hidden = torch.randn(1, _HIDDEN)
        service.enqueue(_trace(), hidden)
        transaction = service.stage_next(runtime)
        assert transaction is not None

        with self.assertRaisesRegex(RuntimeError, "host failure"):
            with transaction:
                raise RuntimeError("host failure before generate")

        self.assertEqual(
            service.instance_state(3, request_id="req-0"), FTTTInstanceState()
        )
        self.assertTrue(service.enqueue(_trace(), hidden))
        replacement = service.stage_next(runtime)
        assert replacement is not None
        replacement.abort()

    def test_retry_limit_stops_lifecycle_after_final_signed_update(self) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2, max_retries=2)
        _workload, runtime, compilation = _build(config)
        service = FTTTFeedbackService(config)

        for step_id in (1, 2):
            self.assertTrue(service.enqueue(_trace(), torch.randn(1, _HIDDEN)))
            transaction = service.stage_next(runtime)
            assert transaction is not None
            with transaction:
                self.assertEqual(_drive(runtime, compilation, step_id), step_id)
                intent = transaction.commit()

        state = service.instance_state(3, request_id="req-0")
        self.assertEqual(state, FTTTInstanceState(2, True, 2))
        self.assertEqual(intent.retry_controls.entries[0].action, "stop")
        self.assertEqual(intent.retry_controls.entries[0].reason, "retry-limit")
        self.assertFalse(intent.retry_controls.entries[0].retry_after_update)
        self.assertTrue(intent.transitions[0].trained)

        self.assertFalse(service.enqueue(_trace(), torch.randn(1, _HIDDEN)))
        stopped = service.stage_next(runtime)
        assert stopped is not None
        with stopped:
            self.assertEqual(_drive(runtime, compilation, 3), 2)
            stopped_intent = stopped.commit()
        self.assertEqual(stopped_intent.retry_controls.entries[0].action, "stop")
        self.assertEqual(
            stopped_intent.retry_controls.entries[0].reason, "instance-stopped"
        )
        self.assertFalse(stopped_intent.transitions[0].trained)
        self.assertEqual(
            service.instance_state(3, request_id="req-0"),
            FTTTInstanceState(2, True, 2),
        )

    def test_queue_overflow_preserves_controls_without_training_dropped_row(
        self,
    ) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2, max_queue_size=1)
        _workload, runtime, _compilation = _build(config)
        service = FTTTFeedbackService(config)
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-0",
                    prompt_idx=3,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(1,),
                    text="wrong zero",
                    finish_reason="stop",
                    group_id="round-0",
                ),
                CompletionTraceEntry(
                    request_id="req-1",
                    prompt_idx=4,
                    sample_idx=0,
                    output_index=1,
                    token_ids=(2,),
                    text="wrong one",
                    finish_reason="stop",
                    group_id="round-0",
                ),
            )
        )

        self.assertTrue(service.enqueue(trace, torch.randn(2, _HIDDEN)))
        transaction = service.stage_next(runtime)
        assert transaction is not None
        batch = transaction.batch
        intent = batch.plan.intent
        self.assertEqual(batch.plan.selected_rows, (0,))
        self.assertEqual(len(intent.feedback_records), 1)
        self.assertEqual(service.dropped_feedback, 1)
        self.assertEqual(
            tuple(entry.action for entry in intent.retry_controls.entries),
            ("retry", "stop"),
        )
        self.assertEqual(
            tuple(entry.reason for entry in intent.retry_controls.entries),
            ("feedback-update", "feedback-queue-full"),
        )
        self.assertEqual(
            tuple(entry.state for entry in intent.update_barriers.entries),
            ("ready", "blocked"),
        )
        self.assertEqual(
            tuple(transition.trained for transition in intent.transitions),
            (True, False),
        )
        transaction.abort()

    def test_two_transactions_match_adam_parameters_and_moments(self) -> None:
        config = FTTTConfig(
            hidden_size=_HIDDEN,
            rank=2,
            learning_rate=0.01,
            feedback_feature_size=3,
        )
        workload, runtime, compilation = _build(config)
        reference = copy.deepcopy(workload.edit_adapter)
        reference_optimizer = torch.optim.Adam(
            reference.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=False,
        )
        expected_down = reference.down.weight.detach().clone()
        expected_up = reference.up.weight.detach().clone()
        down_exp_avg = torch.zeros_like(expected_down)
        up_exp_avg = torch.zeros_like(expected_up)
        down_exp_avg_sq = torch.zeros_like(expected_down)
        up_exp_avg_sq = torch.zeros_like(expected_up)
        service = FTTTFeedbackService(
            config,
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                feedback_text="wrong arithmetic carry",
                judge="toy",
            ),
        )
        versions: list[int] = []
        for step_id, hidden in enumerate(
            (
                torch.tensor([[0.2, -0.5, 1.0, 0.7]]),
                torch.tensor([[1.2, 0.3, -0.4, 0.1]]),
            ),
            start=1,
        ):
            self.assertTrue(service.enqueue(_trace(), hidden))
            transaction = service.stage_next(runtime)
            assert transaction is not None
            batch = transaction.batch
            selected = hidden.index_select(
                0, torch.tensor(batch.plan.selected_rows)
            )
            reference_input = edit_inputs(
                selected,
                batch.plan.intent.feedback_records,
                feature_size=config.feedback_feature_size,
            )
            reference_target = target_delta(
                selected,
                batch.plan.intent.feedback_records,
                feature_size=config.feedback_feature_size,
                edit_scale=config.edit_scale,
            )
            rank_activation = reference_input @ expected_down.transpose(0, 1)
            reference_prediction = rank_activation @ expected_up.transpose(0, 1)
            prediction_gradient = (
                2.0
                * (reference_prediction - reference_target)
                / reference_target.numel()
            )
            up_gradient = prediction_gradient.transpose(0, 1) @ rank_activation
            down_gradient = (
                prediction_gradient @ expected_up
            ).transpose(0, 1) @ reference_input
            down_exp_avg.mul_(0.9).add_(down_gradient, alpha=0.1)
            up_exp_avg.mul_(0.9).add_(up_gradient, alpha=0.1)
            down_exp_avg_sq.mul_(0.999).addcmul_(
                down_gradient, down_gradient, value=0.001
            )
            up_exp_avg_sq.mul_(0.999).addcmul_(
                up_gradient, up_gradient, value=0.001
            )
            step_size = config.learning_rate / (1.0 - 0.9**step_id)
            bias_correction = (1.0 - 0.999**step_id) ** 0.5
            with torch.no_grad():
                expected_down.addcdiv_(
                    down_exp_avg,
                    down_exp_avg_sq.sqrt() / bias_correction + 1e-8,
                    value=-step_size,
                )
                expected_up.addcdiv_(
                    up_exp_avg,
                    up_exp_avg_sq.sqrt() / bias_correction + 1e-8,
                    value=-step_size,
                )

            with transaction:
                versions.append(_drive(runtime, compilation, step_id))
                intent = transaction.commit()
            self.assertEqual(intent.adapter_version, step_id)
            torch.testing.assert_close(
                workload.edit_adapter.down.weight, expected_down
            )
            torch.testing.assert_close(
                workload.edit_adapter.up.weight, expected_up
            )

        self.assertEqual(versions, [1, 2])
        expected_optimizer_state = reference_optimizer.state_dict()
        expected_optimizer_state["state"] = {
            0: {
                "step": torch.tensor(2.0),
                "exp_avg": down_exp_avg,
                "exp_avg_sq": down_exp_avg_sq,
            },
            1: {
                "step": torch.tensor(2.0),
                "exp_avg": up_exp_avg,
                "exp_avg_sq": up_exp_avg_sq,
            },
        }
        torch.testing.assert_close(
            workload.optimizer.state_dict(),
            expected_optimizer_state,
        )
        services = cast(
            LocalRuntimeServices, workload.native_workload_resources().services
        )
        self.assertEqual(services.stats(False).loss_count, 2)

    def test_signed_failure_after_port_consumption_fail_stops_transaction(
        self,
    ) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2)
        workload, runtime, compilation = _build(config)
        service = FTTTFeedbackService(config)
        service.enqueue(_trace(), torch.randn(1, _HIDDEN))
        transaction = service.stage_next(runtime)
        assert transaction is not None

        with (
            mock.patch.object(
                workload.optimizer,
                "step",
                side_effect=RuntimeError("injected optimizer failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "optimizer failure"),
        ):
            with transaction:
                _drive(runtime, compilation, 1)

        self.assertEqual(
            service.instance_state(3, request_id="req-0"), FTTTInstanceState()
        )
        self.assertEqual(runtime.drain(FTTTWorkload.closed_loop), ())
        with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
            runtime.stage(FTTTWorkload.hidden_rows, torch.randn(1, _HIDDEN))
        with self.assertRaisesRegex(RuntimeError, "active FTTT transaction"):
            service.stage_next(runtime)

    def test_exception_after_signed_publication_settles_irreversible_update(
        self,
    ) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2)
        _workload, runtime, compilation = _build(config)
        service = FTTTFeedbackService(config)
        service.enqueue(_trace(), torch.randn(1, _HIDDEN))
        transaction = service.stage_next(runtime)
        assert transaction is not None

        with self.assertRaisesRegex(RuntimeError, "host failure"):
            with transaction:
                self.assertEqual(_drive(runtime, compilation, 1), 1)
                raise RuntimeError("host failure after signed publication")

        self.assertEqual(
            service.instance_state(3, request_id="req-0"),
            FTTTInstanceState(retries=1, adapter_version=1),
        )
        self.assertEqual(runtime.drain(FTTTWorkload.closed_loop), ())
        self.assertTrue(service.enqueue(_trace(), torch.randn(1, _HIDDEN)))
        replacement = service.stage_next(runtime)
        assert replacement is not None
        with replacement:
            self.assertEqual(_drive(runtime, compilation, 2), 2)
            intent = replacement.commit()
        self.assertEqual(intent.adapter_version, 2)

    def test_closed_loop_drain_failure_fail_stops_without_second_resolution(
        self,
    ) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2)
        _workload, runtime, _compilation = _build(config)
        service = FTTTFeedbackService(config)
        service.enqueue(_trace(), torch.randn(1, _HIDDEN))
        transaction = service.stage_next(runtime)
        assert transaction is not None

        with (
            mock.patch.object(
                runtime,
                "drain",
                side_effect=RuntimeError("injected drain quorum failure"),
            ) as drain,
            self.assertRaisesRegex(RuntimeError, "drain quorum failure"),
        ):
            with transaction:
                transaction.commit()

        self.assertEqual(drain.call_count, 1)
        with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
            runtime.stage(FTTTWorkload.hidden_rows, torch.randn(1, _HIDDEN))

    def test_service_settlement_failure_after_publication_fail_stops_once(
        self,
    ) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2)
        _workload, runtime, compilation = _build(config)
        service = FTTTFeedbackService(config)
        service.enqueue(_trace(), torch.randn(1, _HIDDEN))
        transaction = service.stage_next(runtime)
        assert transaction is not None

        with (
            mock.patch.object(
                service,
                "settle_transaction",
                side_effect=RuntimeError("injected settlement failure"),
            ) as settle,
            self.assertRaisesRegex(RuntimeError, "settlement failure"),
        ):
            with transaction:
                self.assertEqual(_drive(runtime, compilation, 1), 1)
                transaction.commit()

        settle.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
            runtime.stage(FTTTWorkload.hidden_rows, torch.randn(1, _HIDDEN))

    def test_positive_feedback_publishes_without_optimizer_transition(self) -> None:
        config = FTTTConfig(hidden_size=_HIDDEN, rank=2)
        workload, runtime, compilation = _build(config)
        service = FTTTFeedbackService(config)
        self.assertFalse(service.enqueue(_trace(correct=True), torch.randn(1, _HIDDEN)))
        transaction = service.stage_next(runtime)
        assert transaction is not None

        with transaction:
            self.assertEqual(_drive(runtime, compilation), 0)
            published = transaction.commit()
        (empty_delta,) = runtime.drain(FTTTWorkload.edit_delta)
        self.assertEqual(empty_delta.shape, (0, _HIDDEN))
        self.assertEqual(published.adapter_version, 0)
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

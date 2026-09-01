"""One Python-Rust/host-service lifecycle proof for canonical FTTT."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt

import pytest
import torch

from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.workloads.fttt import (
    FTTTBarrierState,
    FTTTCompileOptions,
    FTTTCompletion,
    FTTTCompletionBatch,
    FTTTConfig,
    FTTTFeedbackService,
    FTTTFinishReason,
    FTTTExecution,
    FTTTInstanceKey,
    FTTTInstanceState,
    FTTTExecuted,
    FTTTJudgeResult,
    FTTTJudgeService,
    FTTTRetryAction,
    FTTTSession,
    FTTTSpecialization,
    FTTTWorkload,
)
from tllm.canonical.workloads.fttt.math import edit_inputs, target_delta
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.semantic_graph import (
    AnchorId,
    StateScope,
)
from tllm.contracts.tensor_layout import TensorDType


class _NegativeJudge(FTTTJudgeService):
    def judge(self, completion: FTTTCompletion) -> FTTTJudgeResult:
        return FTTTJudgeResult(
            False,
            0.0,
            "wrong arithmetic carry",
            (11, 12),
            "incorrect",
            "toy_exact",
            "",
        )


class _SecondJudgeFails(FTTTJudgeService):
    def __init__(self) -> None:
        self._calls = 0

    def judge(self, completion: FTTTCompletion) -> FTTTJudgeResult:
        self._calls += 1
        if self._calls == 2:
            raise RuntimeError("injected judge failure")
        return _NegativeJudge().judge(completion)


def test_fttt_runs_signed_edit_and_settles_typed_retry_lifecycle() -> None:
    config = FTTTConfig(
        rank=2,
        learning_rate=0.05,
        feedback_feature_size=3,
        max_retries=2,
    )
    workload = FTTTWorkload(config)
    graph = workload.semantic_graph()
    assert graph.states[0].scope is StateScope.INSTANCE
    specialization = FTTTSpecialization(
        hidden_size=4,
        tensor_dtype=TensorDType.FLOAT32,
        device=torch.device("cpu"),
        initialization_seed=7,
    )
    compiled = workload.compile(
        specialization,
        FTTTCompileOptions(
            CanonicalHostRuntimeFacts(torch.device("cpu")).timeline()
        ),
    )
    execution = compiled.execution
    assert execution.plan.fallback_count == 0
    assert len(execution.plan.tasks) == 5
    scheduled_anchors = {
        task.task_symbol: task.anchor_symbol
        for task in compiled.schedule_bundle.schedule_decision.tasks
    }
    assert scheduled_anchors["predict"] == "host.compute.end"
    assert scheduled_anchors["publish"] == "host.output.publish"
    with pytest.raises(ValueError, match="differ from placement"):
        workload.compile(
            specialization,
            FTTTCompileOptions(
                replace(
                    compiled.schedule_bundle.timeline,
                    capabilities=("compute/cuda@1", "host/service@1"),
                )
            ),
        )
    tampered_physical = replace(
        compiled.schedule_bundle.schedule_physical,
        tasks=(
            replace(
                compiled.schedule_bundle.schedule_physical.tasks[0],
                anchor_id=AnchorId(2),
            ),
            *compiled.schedule_bundle.schedule_physical.tasks[1:],
        ),
    )
    with pytest.raises(ValueError):
        FTTTExecution(
            config,
            specialization,
            replace(
                compiled.schedule_bundle,
                schedule_physical=tampered_physical,
            ),
        )

    identity = RequestIdentity("feedback-0", 3, 0)
    instance_key = FTTTInstanceKey("feedback-0", 3)
    completion = FTTTCompletion(
        identity,
        0,
        "round-0",
        (1, 2),
        "wrong arithmetic",
        FTTTFinishReason.STOP,
    )
    hidden = torch.tensor(((0.2, -0.1, 0.4, 0.3),), dtype=torch.float32)
    second_identity = RequestIdentity("feedback-1", 4, 0)
    failed_service = FTTTFeedbackService(config, _SecondJudgeFails())
    with pytest.raises(RuntimeError, match="injected judge failure"):
        failed_service.enqueue(
            FTTTCompletionBatch(
                (
                    completion,
                    replace(
                        completion,
                        identity=second_identity,
                        output_index=1,
                    ),
                )
            ),
            torch.cat((hidden, hidden), dim=0),
        )
    assert failed_service.queued_feedback == 0
    assert failed_service.queued_transactions == 0
    assert failed_service.dropped_feedback == 0
    assert failed_service.instance_state(instance_key) == FTTTInstanceState()
    assert failed_service.instance_state(
        FTTTInstanceKey("feedback-1", 4)
    ) == FTTTInstanceState()
    service = FTTTFeedbackService(config, _NegativeJudge())
    session = FTTTSession(service, execution)
    initial_down, initial_up = execution.parameter_snapshot()
    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(7)
    expected_down = torch.empty_like(initial_down)
    expected_down.normal_(
        mean=0.0,
        std=1.0 / sqrt(float(4 + config.feedback_feature_size + 1)),
        generator=expected_generator,
    )
    torch.testing.assert_close(initial_down, expected_down)
    torch.testing.assert_close(initial_up, torch.zeros_like(initial_up))
    submitted_hidden = hidden.clone()
    assert service.enqueue(FTTTCompletionBatch((completion,)), submitted_hidden)
    submitted_hidden.zero_()
    first_polled = session.run_next()
    match first_polled:
        case FTTTExecuted(result=first_result):
            pass
        case _:
            raise AssertionError("FTTT host transaction did not execute")
    selected = hidden
    expected_input = edit_inputs(
        selected,
        first_result.intent.feedback_records,
        feature_size=config.feedback_feature_size,
    )
    expected_target = target_delta(
        selected,
        first_result.intent.feedback_records,
        feature_size=config.feedback_feature_size,
        edit_scale=config.edit_scale,
    )
    rank_activation = expected_input @ initial_down.transpose(0, 1)
    prediction_gradient = -2.0 * expected_target / expected_target.numel()
    up_gradient = prediction_gradient.transpose(0, 1) @ rank_activation
    expected_up = -config.learning_rate * up_gradient / (
        up_gradient.abs() + config.eps
    )
    expected_delta = rank_activation @ expected_up.transpose(0, 1)
    reference_down = initial_down.detach().clone().requires_grad_(True)
    reference_up = initial_up.detach().clone().requires_grad_(True)
    reference_optimizer = torch.optim.Adam(
        (reference_down, reference_up),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )
    for _ in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = (
            (expected_input @ reference_down.transpose(0, 1))
            @ reference_up.transpose(0, 1)
            - expected_target
        ).square().mean()
        reference_loss.backward()
        reference_optimizer.step()

    torch.testing.assert_close(first_result.losses[0], expected_target.square().mean())
    torch.testing.assert_close(first_result.edit_delta, expected_delta)
    actual_down, actual_up = execution.parameter_snapshot()
    torch.testing.assert_close(actual_down, initial_down)
    torch.testing.assert_close(actual_up, expected_up)
    assert service.instance_state(instance_key).adapter_version == 1
    assert first_result.intent.retry_controls[0].action is FTTTRetryAction.RETRY
    assert first_result.intent.update_barriers[0].state is FTTTBarrierState.READY

    assert service.enqueue(FTTTCompletionBatch((completion,)), hidden)
    second_polled = session.run_next()
    match second_polled:
        case FTTTExecuted(result=second_result):
            pass
        case _:
            raise AssertionError("FTTT retry transaction did not execute")
    assert second_result.intent.retry_controls[0].action is FTTTRetryAction.STOP
    assert not second_result.intent.retry_controls[0].retry_after_update
    assert service.instance_state(instance_key).adapter_version == 2
    assert service.instance_state(instance_key).stopped
    second_down, second_up = execution.parameter_snapshot()
    torch.testing.assert_close(second_down, reference_down.detach())
    torch.testing.assert_close(second_up, reference_up.detach())
    diagnostics = execution.diagnostics()
    assert diagnostics.transaction_count == 2
    assert diagnostics.loss_count == 2
    assert diagnostics.update_count == 2
    assert diagnostics.state_version == 2
    assert diagnostics.fallback_count == 0

    shared_service = FTTTFeedbackService(config, _NegativeJudge())
    shared_session = FTTTSession(shared_service, execution)
    shared_sample = replace(
        completion,
        identity=RequestIdentity("feedback-0", 3, 1),
        output_index=1,
    )
    assert shared_service.enqueue(
        FTTTCompletionBatch((completion, shared_sample)),
        torch.cat((hidden, hidden), dim=0),
    )
    shared_polled = shared_session.run_next()
    match shared_polled:
        case FTTTExecuted(result=shared_result):
            pass
        case _:
            raise AssertionError("same-instance FTTT batch did not execute")
    assert len(shared_result.intent.transitions) == 2
    assert shared_result.update_count == 1
    assert shared_service.instance_state(instance_key) == FTTTInstanceState(
        retries=2,
        stopped=True,
        adapter_version=3,
    )

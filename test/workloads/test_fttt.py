"""Evidence 21: independent Adam/edit oracle, rollback and judge retry settlement."""

from hashlib import blake2b
import re

import torch

from tllm.canonical.host_runtime import (
    CanonicalHostRuntimeFacts, HostTransactionCoordinator, NO_HOST_REQUEST,
)
from tllm.canonical.workloads.fttt import (
    FTTTCompletion, FTTTCompletionBatch, FTTTConfig,
    FTTTFeedbackService, FTTTFinishReason, FTTTInstanceKey, FTTTJudgeResult,
    FTTTJudgeService, FTTTRetryAction, FTTTSpecialization, FTTTWorkload,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.tensor_layout import TensorDType


class ArithmeticJudge(FTTTJudgeService):
    def judge(self, completion: FTTTCompletion) -> FTTTJudgeResult:
        correct = completion.text == "4"
        return FTTTJudgeResult(correct, float(correct), "correct" if correct else "wrong carry",
                              (11, 12), "arithmetic", "exact", "")


def test_fttt_matches_edit_oracle_and_settles_judge_retry() -> None:
    config = FTTTConfig(rank=2, learning_rate=0.05, feedback_feature_size=3, max_retries=2)
    execution = FTTTWorkload(config).compile(
        FTTTSpecialization(4, TensorDType.FLOAT32, torch.device("cpu"), 7),
        CanonicalHostRuntimeFacts(torch.device("cpu")).timeline(),
    )
    service = FTTTFeedbackService(config, ArithmeticJudge())
    coordinator = HostTransactionCoordinator(service, execution)
    identity = RequestIdentity("arithmetic", 0, 0)
    hidden = torch.tensor([[-1.0, -0.25, 0.5, 1.0]])
    wrong = FTTTCompletion(identity, 0, "round", (1, 2), "3", FTTTFinishReason.STOP)
    batch = FTTTCompletionBatch((wrong,))
    service.enqueue(batch, hidden)
    transaction = service.begin(NO_HOST_REQUEST)
    assert transaction is not None
    interrupted = execution.execute(transaction)
    assert interrupted.result.update_count == 1
    execution.abort(interrupted)
    service.abort(transaction)
    assert execution.diagnostics().state_version == 0

    # Spell out the feature projection and use PyTorch Adam, not production helpers.
    feature = torch.zeros(3)
    text = (
        "feedback: wrong carry\nfailed_completion: 3\n"
        "failed_tokens: failed_tok_1 failed_tok_2\nfeedback_tokens: feedback_tok_11 feedback_tok_12"
    )
    for token in re.findall(r"[a-z0-9_]+", text):
        digest = blake2b(token.encode(), digest_size=8).digest()
        feature[int.from_bytes(digest[:4], "little") % 3] += (
            1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
        )
    feature /= feature.norm().clamp_min(1.0)
    inputs = torch.cat((hidden, feature[None], torch.zeros(1, 1)), dim=1)
    target = torch.cat((feature, torch.zeros(1)))[None]
    generator = torch.Generator().manual_seed(7)
    down = (torch.randn(2, 8, generator=generator) / (8 ** 0.5)).requires_grad_()
    up = torch.zeros(4, 2, requires_grad=True)
    optimizer = torch.optim.Adam((down, up), lr=config.learning_rate, betas=config.betas, eps=config.eps)
    for version, action in ((1, FTTTRetryAction.RETRY), (2, FTTTRetryAction.STOP)):
        optimizer.zero_grad()
        loss = ((inputs @ down.T @ up.T - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        service.enqueue(batch, hidden)
        result = coordinator.run(NO_HOST_REQUEST)
        assert result is not None and result.update_count == 1
        torch.testing.assert_close(result.losses[0], loss.detach())
        torch.testing.assert_close(result.edit_delta, (inputs @ down.T @ up.T).detach())
        assert result.state_version == version
        assert result.intent.retry_controls[0].action is action
        state = service.instance_state(FTTTInstanceKey("arithmetic", 0))
        assert state.retries == state.adapter_version == version
    accepted = FTTTCompletion(identity, 0, "round", (4,), "4", FTTTFinishReason.STOP)
    assert not service.enqueue(FTTTCompletionBatch((accepted,)), hidden)
    result = coordinator.run(NO_HOST_REQUEST)
    assert result is not None and result.update_count == 0 and result.losses == ()
    assert result.state_version == 2
    assert result.intent.retry_controls[0].action is FTTTRetryAction.ACCEPT
    assert service.queued_transactions == service.queued_feedback == 0
    assert execution.diagnostics().fallback_count == 0

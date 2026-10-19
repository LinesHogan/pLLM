"""FTTT edit training against an independent Adam oracle, plus the judge-driven retry loop.

The retry loop is the vLLM handle's ``generate_feedback`` driven by a scripted
generator, so retry, stop, update barriers and TTRL vote quorums run without a model.
"""

import re
from dataclasses import replace
from hashlib import blake2b
from importlib.util import find_spec

import pytest
import torch

from pllm import make_llm
from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason, RetryAction
from pllm.contracts.request_identity import RequestIdentity
from pllm.execution.host_runtime import NO_HOST_REQUEST, CompletionInput, HostRuntimeFacts
from pllm.workloads.fttt import FTTTConfig, FTTTFeedbackService, FTTTJudgeResult, FTTTJudgeService, FTTTWorkload


class ArithmeticJudge(FTTTJudgeService):
    def judge(self, completion: Completion) -> FTTTJudgeResult:
        correct = completion.text == "4"
        return FTTTJudgeResult(
            correct, float(correct), "correct" if correct else "wrong carry", (11, 12), "arithmetic", "exact", ""
        )


def test_fttt_matches_edit_oracle_and_settles_judge_retry() -> None:
    with pytest.raises(ValueError):
        FTTTConfig(edit_scale=float("nan"))
    config = FTTTConfig(rank=2, learning_rate=0.05, feedback_feature_size=3, max_retries=2)
    service = FTTTFeedbackService(config, ArithmeticJudge())
    workload = FTTTWorkload(service)
    facts = HostRuntimeFacts(
        torch.device("cpu"), hidden_size=4, initialization_seed=7, max_input_rows=1, memory_budget_bytes=1 << 20
    )
    coordinator = make_llm(workload=workload, config=facts)
    peak = coordinator.execution.schedule_bundle.schedule_decision.peak_memory_bytes
    with pytest.raises(ValueError):
        make_llm(workload=workload, config=replace(facts, memory_budget_bytes=peak - 1))
    with pytest.raises(ValueError):  # host workloads need a finite input row bound
        workload.declare_host(replace(facts, max_input_rows=None))
    coordinator = make_llm(workload=workload, config=replace(facts, memory_budget_bytes=peak))
    execution = coordinator.execution
    identity = RequestIdentity("arithmetic", 0, 0)
    hidden = torch.tensor([[-1.0, -0.25, 0.5, 1.0]])
    wrong = Completion(identity, 0, "round", (1, 2), "3", FinishReason.STOP)
    batch = CompletionBatch((wrong,))
    service.enqueue(CompletionInput.capture(batch, hidden))
    # An aborted transaction leaves the edit state at version 0.
    transaction = service.begin(NO_HOST_REQUEST)
    execution.abort(execution.execute(transaction))
    service.abort(transaction)
    assert execution.diagnostics().state_version == 0
    overflow = FTTTFeedbackService(config, ArithmeticJudge())
    extra = replace(wrong, identity=RequestIdentity("excess", 1, 0))
    overflow.enqueue(CompletionInput.capture(CompletionBatch((wrong, extra)), hidden.expand(2, -1)))
    oversized = overflow.begin(NO_HOST_REQUEST)
    with pytest.raises(ValueError):  # two completions exceed max_input_rows=1
        execution.execute(oversized)
    overflow.abort(oversized)
    assert execution.diagnostics().state_version == 0 and overflow.queued_feedback == 0

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
    down = (torch.randn(2, 8, generator=generator) / (8**0.5)).requires_grad_()
    up = torch.zeros(4, 2, requires_grad=True)
    optimizer = torch.optim.Adam((down, up), lr=config.learning_rate, betas=config.betas, eps=config.eps)
    for version, action in ((1, RetryAction.RETRY), (2, RetryAction.STOP)):
        optimizer.zero_grad()
        loss = ((inputs @ down.T @ up.T - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        retry_batch = CompletionBatch((replace(wrong, identity=RequestIdentity(f"attempt-{version}", 0, version)),))
        result = coordinator.run(CompletionInput.capture(retry_batch, hidden))
        assert result.update_count == 1
        torch.testing.assert_close(result.losses[0], loss.detach())
        torch.testing.assert_close(result.edit_delta, (inputs @ down.T @ up.T).detach())
        assert result.state_version == version
        assert result.intent.retry_controls[0].action is action
        state = service.instance_state("round", 0)
        assert state.retries == state.adapter_version == version
    accepted = Completion(identity, 0, "round", (4,), "4", FinishReason.STOP)
    assert not service.enqueue(CompletionInput.capture(CompletionBatch((accepted,)), hidden))
    result = coordinator.run(NO_HOST_REQUEST)
    assert result.update_count == 0 and result.losses == () and result.state_version == 2
    assert result.intent.retry_controls[0].action is RetryAction.ACCEPT
    assert service.queued_transactions == service.queued_feedback == 0
    if find_spec("vllm") is not None:
        _automatic_retry_uses_original_control_and_quorum()


def _automatic_retry_uses_original_control_and_quorum() -> None:
    from vllm import SamplingParams

    from pllm.backends.vllm.runtime import VllmLlm
    from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload

    class Generation:
        """Scripted generation: prompt 7 answers correctly; with votes, two of three agree."""

        generate_feedback = VllmLlm.generate_feedback

        def __init__(self, votes=False):
            self.calls = []
            self.votes = votes

        def generate(
            self,
            prompts,
            sampling_params,
            *,
            use_tqdm,
            feedback_group,
            prompt_indices,
            sample_index,
            auxiliary=None,
            completion_device=None,
            routes=None,
        ):
            assert routes is None
            self.calls.append(prompt_indices)
            entries = tuple(
                Completion(
                    RequestIdentity(f"request-{sample_index}-{row}", index, sample_index),
                    0,
                    feedback_group,
                    (1,),
                    (r"\boxed{1}" if row < 2 else r"\boxed{2}") if self.votes else "4" if index == 7 else "3",
                    FinishReason.STOP,
                )
                for row, index in enumerate(prompt_indices)
            )
            return CompletionInput.capture(CompletionBatch(entries), torch.ones(len(entries), 4))

    service = FTTTFeedbackService(FTTTConfig(rank=2, max_retries=2), ArithmeticJudge())
    processor = make_llm(workload=FTTTWorkload(service), config=HostRuntimeFacts(torch.device("cpu"), hidden_size=4))
    generator = Generation()
    rounds = list(
        generator.generate_feedback(
            ["accepted", "retry"],
            SamplingParams(max_tokens=1),
            feedback=processor.run,
            feedback_group="retry",
            max_rounds=5,
            prompt_indices=(7, 9),
        )
    )
    assert generator.calls == [(7, 9), (9,)]
    assert service.instance_state("retry", 9).retries == 2
    assert rounds[-1][1].retry_controls[0].action is RetryAction.STOP
    assert service.queued_transactions == service.queued_feedback == 0
    update = rounds[0][1].state_update()
    assert update.values[0] is rounds[0][1].edit_delta and update.keys[0].prompt_idx == 9

    # Retries that wait for an update stop when nothing was published.
    def no_published_update(batch):
        result = processor.run(batch)
        return replace(result, update_count=0)

    blocked = Generation()
    assert (
        len(
            list(
                blocked.generate_feedback(
                    ["retry"],
                    SamplingParams(max_tokens=1),
                    feedback=no_published_update,
                    feedback_group="blocked",
                    max_rounds=5,
                    prompt_indices=(9,),
                )
            )
        )
        == 1
    )

    def copied_completion(batch):
        result = processor.run(batch)
        controls = tuple(replace(control, completion=replace(control.completion)) for control in result.retry_controls)
        return replace(result, intent=replace(result.intent, retry_controls=controls))

    with pytest.raises(ValueError):  # controls must reference the original completions
        list(
            Generation().generate_feedback(
                ["retry"],
                SamplingParams(max_tokens=1),
                feedback=copied_completion,
                feedback_group="copied",
                max_rounds=2,
                prompt_indices=(9,),
            )
        )

    reward = make_llm(
        workload=TTRLWorkload(TTRLConfig(rank=2, votes_per_prompt=3, samples_per_prompt=3)),
        config=HostRuntimeFacts(torch.device("cpu"), hidden_size=4),
    )
    reward.admit(())
    votes = Generation(votes=True)
    # A vote group retries as a whole after its quorum update.
    rounds = list(
        votes.generate_feedback(
            ["vote"] * 3,
            SamplingParams(max_tokens=1),
            feedback=reward.step,
            feedback_group="votes",
            max_rounds=2,
            prompt_indices=(11, 11, 11),
        )
    )
    assert votes.calls == [(11, 11, 11), (11, 11, 11)]
    assert len(rounds) == 2 and reward.diagnostics().update_count == 2
    reward.retire(())

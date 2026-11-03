"""TTRL majority-vote rewards and head updates against an independent Adam oracle.

Also covers settlement retry after a post-publication failure, blocked vote groups,
the disabled workload and a zero-loss update.
"""

from dataclasses import replace

import pytest
import torch

from pllm.authoring import FunctionMath, StepInput, WeightedRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, HostServiceInputSource
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.state_execution import PartialCommitError
from pllm.host import (
    BarrierState,
    Completion,
    CompletionBatch,
    CompletionInput,
    FinishReason,
    HostConfig,
    RetryAction,
    make_host,
)
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload, boxed_label, ttrl_vote


def fail_output(self, context):
    raise RuntimeError("injected output failure")


def test_ttrl_matches_reward_update_retry_and_control_quorum(monkeypatch: pytest.MonkeyPatch, device_name) -> None:
    assert boxed_label(r"old \boxed{0}, final \boxed{ \frac{ A }{ b } }") == r"\frac{a}{b}"
    assert boxed_label(r"\boxed{unfinished") == ""
    assert boxed_label(r"\fbox{ TWO Words }") == "twowords"
    completions = CompletionBatch(
        tuple(
            Completion(RequestIdentity(str(i), i // 3, i % 3), 0, f"group-{i // 3}", (1,), text, FinishReason.STOP)
            for i, text in enumerate(
                (r"\boxed{ B }", r"\boxed{A}", "missing", r"\boxed{2}", r"\boxed{1}", r"\boxed{1}")
            )
        )
    )
    config = TTRLConfig(rank=2, learning_rate=0.02, votes_per_prompt=3, samples_per_prompt=2)
    device = torch.device(device_name)
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    workload = TTRLWorkload(config)
    facts = HostConfig(
        device,
        hidden_size=4,
        seed=11,
        state_memory_bytes=65536,
        max_input_rows=6,
        memory_budget_bytes=1 << 20,
    )
    execution = make_host(facts, workload)
    plan = execution.arena.plan
    assert {
        port.source.schema
        for task in plan.semantic_graph.tasks
        for port in task.inputs
        if isinstance(port.source, HostServiceInputSource)
    } == {WeightedRowBatch.schema}
    peak = plan.schedule.peak_memory_bytes
    with pytest.raises(ValueError):
        make_host(replace(facts, memory_budget_bytes=peak - 1), workload)
    execution = make_host(replace(facts, memory_budget_bytes=peak), workload)
    (initial,) = execution.admit((), stream)
    old_pin = execution.arena.pin_exact(StateRefs.of((initial,)), stream)
    original = execution.snapshot(initial, stream)
    source = torch.linspace(-1, 1, 24, device=device).reshape(6, 4)
    producer = torch.cuda.current_stream(device) if stream is not None else None
    batch = CompletionInput.copy_of(completions, source, producer)
    source.zero_()
    assert batch.completions is completions and torch.count_nonzero(batch.hidden) > 0
    rewards = ttrl_vote(batch, config)
    if stream is not None:
        stream.synchronize()
    down = torch.nn.Parameter(original[0][:8].reshape(2, 4).clone())
    up = torch.nn.Parameter(original[0][8:].reshape(1, 2).clone())
    optimizer = torch.optim.Adam((down, up), lr=config.learning_rate, foreach=False)
    selected = [0, 1, 3, 4]
    expected_rewards = [1.0, 0.0, None, 0.0, 1.0, 1.0]
    latest = initial
    ones = torch.ones(7, device=device)
    excess = StepInput.of_batch(WeightedRowBatch(torch.ones(7, 4, device=device), ones, ones))
    with pytest.raises(ValueError):  # seven rows exceed max_input_rows
        execution.step(excess, stream)
    assert execution.arena.latest(initial.lineage_id) == initial
    for _ in range(2):
        result = execution.step(rewards.step, stream)
        if stream is not None:
            stream.synchronize()
        assert result.publication is not None and tuple(result.publication.before_refs) == (latest,)
        (latest,) = result.publication.after_refs
        assert [item.reward for item in rewards.reward_outcomes] == expected_rewards
        assert [item.action for item in rewards.retry_controls] == [
            RetryAction.ACCEPT,
            RetryAction.RETRY,
            RetryAction.RETRY,
            RetryAction.RETRY,
            RetryAction.ACCEPT,
            RetryAction.ACCEPT,
        ]
        assert [item.retry_after_update for item in rewards.retry_controls] == [
            False,
            True,
            False,
            True,
            False,
            False,
        ]
        assert all(item.state is BarrierState.READY and item.ready_count == 2 for item in rewards.update_barriers)
        for completion, reward, control in zip(
            completions.entries, rewards.reward_outcomes, rewards.retry_controls, strict=True
        ):
            assert reward.completion is control.completion is completion
        optimizer.zero_grad()
        prediction = batch.hidden[selected] @ down.T @ up.T
        target = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device).reshape(-1, 1)
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()
        torch.testing.assert_close(result.loss, loss.detach())
        actual = execution.snapshot(latest, stream)
        if stream is not None:
            stream.synchronize()
        expected = (
            torch.cat((down.flatten(), up.flatten())),
            torch.cat((down.grad.flatten(), up.grad.flatten())),
            optimizer.state[down]["step"].to(device),
            torch.cat((optimizer.state[down]["exp_avg"].flatten(), optimizer.state[up]["exp_avg"].flatten())),
            torch.cat((optimizer.state[down]["exp_avg_sq"].flatten(), optimizer.state[up]["exp_avg_sq"].flatten())),
        )
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference)
    if stream is not None:
        # The snapshot must be ready for comparison on the caller's default stream.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
    for value, reference in zip(execution.arena.snapshot(old_pin, initial), original, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    execution.arena.wait_release(execution.arena.release_pin(old_pin))
    pending = execution.start_step(rewards.step, stream)
    execution.execute_anchor(pending, AnchorId(0))
    execution.cancel_step(pending)
    assert execution.arena.latest(initial.lineage_id) == latest

    with monkeypatch.context() as fault:
        # Fail result conversion after the update is published (workload code is signed, so inject here).
        fault.setattr(FunctionMath, "output", fail_output)
        with pytest.raises(PartialCommitError) as failure:
            execution.step(rewards.step, stream)
    # Settlement can be retried: it returns the publication that already happened.
    retry = execution.finish_step(failure.value.context)
    assert retry.publication is failure.value.receipt
    (latest,) = retry.publication.after_refs
    assert execution.diagnostics().update_count == 3
    with pytest.raises(RuntimeError):
        execution.finish_step(failure.value.context)
    missing = replace(
        batch, completions=CompletionBatch(tuple(replace(c, text="missing") for c in completions.entries))
    )
    blocked = ttrl_vote(missing, config)
    assert all(item.reward is None for item in blocked.reward_outcomes)
    assert all(item.state is BarrierState.BLOCKED and item.ready_count == 0 for item in blocked.update_barriers)
    # Nothing selected: the step publishes the same values and leaves every optimizer field untouched.
    before = execution.snapshot(latest, stream)
    result = execution.step(blocked.step, stream)
    (published,) = result.publication.after_refs
    for value, reference in zip(execution.snapshot(published, stream), before, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    with pytest.raises(ValueError):  # an incomplete vote group
        ttrl_vote(
            replace(batch, completions=CompletionBatch((completions.entries[0],)), hidden=batch.hidden[:1]), config
        )
    execution.retire((), wait=True)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_states == diagnostics.pending_state_releases == 0

    disabled = ttrl_vote(CompletionInput.copy_of(completions, torch.zeros(6, 4)), replace(config, enabled=False))
    assert all(item.state is BarrierState.BLOCKED for item in disabled.update_barriers)

    facts = HostConfig(torch.device("cpu"), hidden_size=4, seed=11)
    execution = make_host(facts, TTRLWorkload(config))
    (initial,) = execution.admit(())
    before = execution.snapshot(initial)
    zeros = torch.zeros(6, 4)
    result = execution.step(StepInput.of_batch(WeightedRowBatch(zeros, zeros[:, 0], torch.ones(6))))
    # A zero loss still publishes: the step count advances with zero gradients and unchanged weights.
    assert result.loss.item() == 0 and result.publication is not None
    after = execution.snapshot(result.publication.after_refs[0])
    torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
    assert after[2].item() == 1 and torch.count_nonzero(after[1]) == 0
    execution.retire(())

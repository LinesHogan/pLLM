"""Evidence 29: independent reward/Adam oracle and original publication settlement."""

from dataclasses import replace
from collections.abc import Mapping

import pytest
import torch
from pllm import make_llm
import pllm

from pllm.execution.host_runtime import HostRuntimeFacts, CompletionInput
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.state_execution import StateSettlementError
from pllm.workloads.ttrl import TTRLConfig, TTRLData, TTRLWorkload, boxed_label
from pllm.authoring import FunctionMath
from pllm.contracts.feedback import BarrierState, Completion, CompletionBatch, FinishReason, RetryAction
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId


def test_ttrl_matches_reward_update_retry_and_control_quorum(monkeypatch: pytest.MonkeyPatch) -> None:
    assert boxed_label(r"old \boxed{0}, final \boxed{ \frac{ A }{ b } }") == r"\frac{a}{b}"
    assert boxed_label(r"\boxed{unfinished") == ""
    assert boxed_label(r"\fbox{ TWO Words }") == "twowords"
    completions = CompletionBatch(tuple(Completion(RequestIdentity(str(i), i // 3, i % 3), 0,
        f"group-{i // 3}", (1,), text, FinishReason.STOP) for i, text in enumerate((
            r"\boxed{ B }", r"\boxed{A}", "missing", r"\boxed{2}", r"\boxed{1}", r"\boxed{1}"))))
    config = TTRLConfig(rank=2, learning_rate=0.02, votes_per_prompt=3, samples_per_prompt=2)
    for device in (torch.device("cpu"), *((torch.device("cuda", 0),) if torch.cuda.is_available() else ())):
        stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        workload = TTRLWorkload(config)
        facts = HostRuntimeFacts(device, hidden_size=4, initialization_seed=11, state_memory_budget_bytes=65536,
            max_input_rows=6, memory_budget_bytes=1 << 20)
        execution = make_llm(workload=workload, config=facts)
        plan = execution.arena.plan
        peak = plan.schedule_decision.peak_memory_bytes
        assert len(plan.timeline.task_resources) == len(plan.timeline.value_resources) == 2
        assert plan.timeline.reserved_bytes == (6 * 4 * 4,) * 4
        with pytest.raises(ValueError):
            make_llm(workload=workload, config=replace(facts, memory_budget_bytes=peak - 1))
        facts = replace(facts, memory_budget_bytes=peak)
        binding = workload.declare_host(facts)
        assert binding.facts is facts
        execution = binding.compile()
        assert execution.arena.plan.timeline.memory_budget_bytes == peak
        initial, = execution.admit((), stream)
        old_pin = execution.arena.pin_exact((initial,), stream)
        original = execution.snapshot(initial, stream)
        source = torch.linspace(-1, 1, 24, device=device).reshape(6, 4)
        producer = torch.cuda.current_stream(device) if stream is not None else None
        batch = CompletionInput.capture(completions, source, producer)
        source.zero_()
        assert batch.completions is completions and torch.count_nonzero(batch.hidden) > 0
        if stream is not None:
            stream.synchronize()
        down = torch.nn.Parameter(original[0][:8].reshape(2, 4).clone())
        up = torch.nn.Parameter(original[0][8:].reshape(1, 2).clone())
        optimizer = torch.optim.Adam((down, up), lr=config.learning_rate, foreach=False)
        selected = [0, 1, 3, 4]
        expected_rewards = [1.0, 0.0, None, 0.0, 1.0, 1.0]
        latest = initial
        extra = replace(completions.entries[0], identity=RequestIdentity("excess", 2, 0))
        excess = CompletionInput.capture(CompletionBatch((*completions.entries, extra)),
            torch.ones(7, 4, device=device), producer)
        with pytest.raises(ValueError, match="signed input capacity"):
            execution.step(excess, stream)
        assert execution.arena.latest(initial.lineage_id) == initial
        for _ in range(2):
            result = execution.step(batch, stream)
            if stream is not None:
                stream.synchronize()
            assert result.publication is not None and result.publication.before_refs == (latest,)
            latest, = result.publication.after_refs
            assert result.state_ref is latest
            assert [item.reward for item in result.reward_outcomes] == expected_rewards
            assert [item.action for item in result.retry_controls] == [
                RetryAction.ACCEPT, RetryAction.RETRY, RetryAction.RETRY,
                RetryAction.RETRY, RetryAction.ACCEPT, RetryAction.ACCEPT]
            assert [item.retry_after_update for item in result.retry_controls] == [False, True, False, True, False, False]
            assert all(item.state is BarrierState.READY and item.ready_count == 2 for item in result.update_barriers)
            for completion, reward, control in zip(completions.entries, result.reward_outcomes, result.retry_controls, strict=True):
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
            expected = (torch.cat((down.flatten(), up.flatten())), torch.cat((down.grad.flatten(), up.grad.flatten())),
                optimizer.state[down]["step"].to(device),
                torch.cat((optimizer.state[down]["exp_avg"].flatten(), optimizer.state[up]["exp_avg"].flatten())),
                torch.cat((optimizer.state[down]["exp_avg_sq"].flatten(), optimizer.state[up]["exp_avg_sq"].flatten())))
            for value, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, reference)
        for value, reference in zip(execution.arena.snapshot(old_pin, initial), original, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        execution.arena.wait_release(execution.arena.release_pin(old_pin))
        pending = execution.start_step(batch, stream)
        execution.execute_anchor(pending, AnchorId(0))
        execution.cancel_step(pending)
        assert execution.arena.latest(initial.lineage_id) == latest
        def fail_publish(self, write):
            raise RuntimeError("publication failed")
        with monkeypatch.context() as fault:
            fault.setattr(PageBundleArena, "publish", fail_publish)
            with pytest.raises(RuntimeError, match="publication failed"):
                execution.step(batch, stream)
        assert execution.arena.latest(initial.lineage_id) == latest
        def fail_output(self, context):
            raise RuntimeError("output failed")
        with monkeypatch.context() as fault:
            fault.setattr(FunctionMath, "output", fail_output)
            with pytest.raises(StateSettlementError) as failure:
                execution.step(batch, stream)
        retry = execution.finish_step(failure.value.invocation)
        assert retry.publication is failure.value.receipt
        assert execution.diagnostics().update_count == 3
        with pytest.raises(RuntimeError, match="not registered"):
            execution.finish_step(failure.value.invocation)
        missing = replace(batch, completions=CompletionBatch(tuple(replace(c, text="missing") for c in completions.entries)))
        result = execution.step(missing, stream)
        assert result.loss is result.publication is None
        assert all(item.reward is None for item in result.reward_outcomes)
        assert all(item.state is BarrierState.BLOCKED and item.ready_count == 0 for item in result.update_barriers)
        with pytest.raises(ValueError, match="vote group"):
            execution.step(replace(batch, completions=CompletionBatch((completions.entries[0],)), hidden=batch.hidden[:1]), stream)
        execution.retire(())
        diagnostics = execution.diagnostics()
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0

    disabled = make_llm(workload=TTRLWorkload(replace(config, enabled=False)), config=HostRuntimeFacts(
        torch.device("cpu"), hidden_size=4, state_memory_budget_bytes=65536))
    disabled.admit(())
    result = disabled.step(CompletionInput.capture(completions, torch.zeros(6, 4)))
    assert result.loss is result.publication is None
    assert all(item.state is BarrierState.BLOCKED for item in result.update_barriers)
    disabled.retire(())

    class ZeroLoss(TTRLWorkload):
        scale = 0.0

        def loss(self, data: TTRLData, parameters: Mapping[str, torch.Tensor]) -> torch.Tensor:
            loss = parameters["head"].square().sum() * self.scale
            data.loss = loss.detach()
            return loss

    facts = pllm.HostRuntimeFacts(torch.device("cpu"), hidden_size=4, initialization_seed=11)
    zero = ZeroLoss(config)
    with pytest.raises(ValueError, match="resource_facts"):
        zero.declare_host(replace(facts, max_input_rows=6, memory_budget_bytes=1 << 20))
    with pytest.raises(ValueError, match="max_input_rows"):
        TTRLWorkload(config).declare_host(replace(facts, memory_budget_bytes=1 << 20))
    native_prepare = TTRLWorkload.prepare
    with monkeypatch.context() as changed:
        changed.setattr(TTRLWorkload, "prepare", lambda self, value: native_prepare(self, value))
        with pytest.raises(ValueError, match="resource_facts"):
            TTRLWorkload(config).declare_host(replace(facts, max_input_rows=6, memory_budget_bytes=1 << 20))

    class LooseBudget(TTRLWorkload):
        def resource_facts(self, facts):
            return replace(TTRLWorkload.resource_facts(self, facts), memory_budget_bytes=None)

    with pytest.raises(ValueError):
        make_llm(workload=LooseBudget(config), config=replace(facts, max_input_rows=6, memory_budget_bytes=1))
    binding = zero.declare_host(facts)
    execution = binding.compile()
    initial, = execution.admit(())
    before = execution.snapshot(initial)
    with monkeypatch.context() as changed:
        changed.setattr(ZeroLoss, "scale", 1.0)
        assert zero.declare_host(facts).graph.execution_parameters_digest != binding.graph.execution_parameters_digest
        result = execution.step(CompletionInput.capture(completions, torch.ones(6, 4)))
    assert result.loss.item() == 0 and result.publication is not None and result.update_count == 1
    after = execution.snapshot(result.state_ref)
    torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
    assert after[2].item() == 1 and torch.count_nonzero(after[1]) == 0
    execution.retire(())

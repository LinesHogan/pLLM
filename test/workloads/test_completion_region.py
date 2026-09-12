"""Completion authoring lowers AdamW while retaining its original page owner."""

from dataclasses import replace

import pytest
import torch
import pllm

from pllm import _core
from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.execution.host_runtime import CompletionInput, HostRuntimeFacts
from pllm.execution.region_adamw import AdamWRegion
from pllm.execution.tensor_program import InvocationOutput
import pllm.execution.declaration as declaration


class TwoGroups(pllm.CompletionWorkload[CompletionInput, InvocationOutput]):
    name = "test.completion.two_groups"

    @property
    def parameters(self):
        def initialize(parameters, generator):
            for value in parameters.values():
                value.normal_(generator=generator)
        return pllm.InstanceState({"matrix": (2, 3), "bias": (3,)},
            optimizer=pllm.AdamW(lr=0.017, betas=(0.83, 0.96), eps=1e-5, weight_decay=0.09),
            initialize=initialize)

    def prepare(self, value):
        return value

    def loss(self, data, parameters):
        if data.completions.entries[0].text == "skip":
            return None
        return (data.hidden @ parameters["matrix"] + parameters["bias"]).square().mean()

    def trained_rows(self, data):
        return len(data.hidden)

    def output(self, data, result):
        return result

    def resource_facts(self, facts):
        return replace(facts.timeline(), task_resources=(TaskResourceFact(TaskSymbol("publish"), 0),
            TaskResourceFact(TaskSymbol("train"), 4096)), value_resources=(
                ValueResourceFact(TaskSymbol("publish"), "control", 0),
                ValueResourceFact(TaskSymbol("train"), "control", 4)))


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_default_completion_region_keeps_multigroup_adamw_and_publication(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device(device)
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    facts = HostRuntimeFacts(device, hidden_size=2, initialization_seed=23,
        state_memory_budget_bytes=65536, memory_budget_bytes=1 << 20)
    workload = TwoGroups()
    binding = workload.declare_host(facts)
    planned = next(fact for fact in binding.timeline.task_resources if fact.task == "train")
    assert planned.workspace_bytes == 4096 + 13 * (9 * 4) + 2 * 12
    execution = binding.compile()
    peak = execution.arena.plan.schedule_decision.peak_memory_bytes
    with pytest.raises(ValueError):
        workload.compile_host(replace(facts, memory_budget_bytes=peak - 1))
    assert any(task.implementation_id.startswith("region.completion_adamw@1:") for task in execution.plan.tasks)
    initial, = execution.admit((), stream)
    original = execution.snapshot(initial, stream)
    if stream is not None:
        stream.synchronize()
    parameters = [torch.nn.Parameter(original[i].clone()) for i in (0, 5)]
    config = workload.parameters.optimizer
    oracle = torch.optim.AdamW(parameters, lr=config.lr, betas=config.betas, eps=config.eps,
        weight_decay=config.weight_decay, foreach=device.type == "cuda", capturable=device.type == "cuda")
    entry = Completion(RequestIdentity("a", 0, 0), 0, "g", (1,), "train", FinishReason.STOP)
    batch = CompletionInput.capture(CompletionBatch((entry,)), torch.tensor([[0.4, -0.7]], device=device),
        torch.cuda.current_stream(device) if stream is not None else None)
    calls = []
    original_region = AdamWRegion.__call__
    def execute(self, roots, gradients):
        calls.append(len(roots))
        return original_region(self, roots, gradients)
    monkeypatch.setattr(AdamWRegion, "__call__", execute)
    def forbidden(*args, **kwargs):
        raise AssertionError("completion update used legacy mathematics or compiled on the request path")
    monkeypatch.setattr(declaration, "apply_adamw", forbidden)
    for name in ("analyze_region_program", "lower_region_schedule", "lower_region_bindings",
                 "analyze_region_capacity", "analyze_region_storage"):
        monkeypatch.setattr(_core, name, forbidden)
    latest = initial
    for round_ in range(1, 4):
        oracle.zero_grad(set_to_none=True)
        loss = (batch.hidden @ parameters[0] + parameters[1]).square().mean()
        loss.backward()
        oracle.step()
        result = execution.step(batch, stream)
        if stream is not None:
            stream.synchronize()
        assert result.publication.before_refs == (latest,)
        latest, = result.publication.after_refs
        assert latest.version_id == round_
        actual = execution.snapshot(latest, stream)
        if stream is not None:
            stream.synchronize()
        for index, parameter in enumerate(parameters):
            state = oracle.state[parameter]
            expected = (parameter, parameter.grad, state["step"], state["exp_avg"], state["exp_avg_sq"])
            for got, want in zip(actual[5 * index:5 * index + 5], expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
    skipped = replace(batch, completions=CompletionBatch((replace(entry, text="skip"),)))
    result = execution.step(skipped, stream)
    assert result.publication is None
    assert execution.arena.latest(initial.lineage_id) == latest
    assert calls == [2, 2, 2]
    assert execution.diagnostics().update_count == 3
    execution.retire(())
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == execution.arena.usage().private_versions == 0
    assert execution.arena.usage().pinned_versions == execution.arena.usage().pending_receipts == 0

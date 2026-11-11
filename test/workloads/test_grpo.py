"""Weighted causal loss and complete AdamW state against independent CPU torch arithmetic."""

import os
from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch

from pllm import AdamW
from pllm.contracts.host_batch import CausalTrainBatch, map_host_batch, validate_host_batch
from pllm.contracts.semantic_graph import HostServiceInputSource
from pllm.execution.shared_tensor import SharedTensor
from test.workloads.grpo_workload import GRPO
from test.workloads.test_tlm import TinyCausalPort, batch_input, causal_facts


def test_weighted_causal_loss_and_adamw_match_torch() -> None:
    model = TinyCausalPort()
    workload = GRPO(optimizer=AdamW(0.01, (0.8, 0.95), 1e-7, 0.1))
    execution = workload.compile_backend(causal_facts(model), causal=model.bind).execution
    (initial,) = execution.admit(())
    plan = execution.arena.plan
    ports = [
        port.source
        for task in plan.semantic_graph.tasks
        for port in task.inputs
        if isinstance(port.source, HostServiceInputSource)
    ]
    assert {port.schema for port in ports} == {"causal_tokens@2"}
    batch = CausalTrainBatch(
        torch.tensor([[0, 1, 2, 3, 4], [2, 4, 1, 0, 0], [1, 2, 0, 0, 0]]),
        torch.tensor([[-100, -100, 2, 3, 4], [-100, 4, 1, -100, -100], [-100, -100, -100, -100, -100]]),
        torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]]),
        torch.tensor([1.25, -0.75, 0.0]),
    )
    with ExitStack() as lifetime:
        wire = map_host_batch(batch, lambda tensor: SharedTensor.export(tensor, lifetime))
        delivered = map_host_batch(wire, lambda tensor: tensor.map(os.getpid()))
        assert isinstance(delivered, CausalTrainBatch)
    for weights in (None, torch.ones(3, 5), torch.ones(3, dtype=torch.int64)):
        with pytest.raises(ValueError, match="weights"):
            validate_host_batch(
                replace(batch, weights=weights),
                plan.semantic_graph,
                plan.tensor_layout_request,
                plan.timeline,
                SharedTensor.describe,
                hidden_size=3,
                vocab_size=5,
            )
    parameters = tuple(
        torch.nn.Parameter(value.reshape(2, 3) if index % 2 == 0 else value.reshape(3, 2))
        for index, value in enumerate(execution.snapshot(initial)[0].clone().split(6))
    )
    optimizer = torch.optim.AdamW(parameters, lr=0.01, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.1, foreach=False)
    for version in (1, 2):
        # No cross_entropy or workload helper: gather target log probabilities and average answer tokens by row.
        hidden = model.embedding[batch.input_ids]
        q, v = (model.base[index] + parameters[2 * index + 1] @ parameters[2 * index] for index in range(2))
        logits = model.head(torch.tanh(hidden @ q.T + hidden @ v.T))
        targets = batch.labels[:, 1:]
        mask = targets != -100
        logp = logits[:, :-1].log_softmax(-1).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        assert batch.weights is not None
        loss = -(batch.weights * (logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result = execution.step(batch_input(delivered))
        assert result.publication is not None
        (current,) = result.publication.after_refs
        assert current.version_id == version
        torch.testing.assert_close(result.loss, loss.detach())
        actual = execution.snapshot(current)
        expected = (
            torch.cat([p.flatten() for p in parameters]),
            optimizer.state[parameters[0]]["step"].to(actual[1]),
            torch.cat([optimizer.state[p]["exp_avg"].flatten() for p in parameters]),
            torch.cat([optimizer.state[p]["exp_avg_sq"].flatten() for p in parameters]),
        )
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference)
    execution.retire((), wait=True)
    assert execution.diagnostics().update_count == 2
    torch.testing.assert_close(
        workload.advantages(torch.tensor([1.0, 1.0, 0.0])), torch.tensor([1.0, 1.0, -2.0]) / (2**0.5 + 3e-4)
    )
    assert not workload.advantages(torch.ones(3)).any()

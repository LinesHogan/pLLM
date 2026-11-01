"""An ordinary host workload with two parameter views matches torch AdamW: bitwise on CPU, to rounding on CUDA."""

from contextlib import nullcontext
from dataclasses import dataclass, replace

import pytest
import torch

import pllm
from pllm.authoring import StepInput, WeightedRowBatch
from pllm.host import HostConfig, make_host


@dataclass(frozen=True)
class TwoGroups(pllm.Workload):
    @property
    def parameters(self) -> pllm.InstanceState:
        return pllm.InstanceState(
            {"matrix": (2, 3), "bias": (3,)},
            optimizer=pllm.AdamW(lr=0.017, betas=(0.83, 0.96), eps=1e-5, weight_decay=0.09),
            initialize=self.initialize,
        )

    def initialize(self, parameters, generator):
        for value in parameters.values():
            value.normal_(generator=generator)

    @pllm.train(state="parameters")
    def update(
        self,
        hidden: torch.Tensor = pllm.batch("hidden"),
        matrix: torch.Tensor = pllm.parameter("parameters", "matrix"),
        bias: torch.Tensor = pllm.parameter("parameters", "bias"),
    ) -> torch.Tensor | None:
        if hidden.shape[0] > 1:  # a batch of several rows is this test's way to skip the update
            return None
        return (hidden @ matrix + bias).square().mean()


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_default_completion_region_keeps_multigroup_adamw_and_publication(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device(device)
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    facts = HostConfig(
        device,
        hidden_size=2,
        seed=23,
        state_memory_bytes=65536,
        memory_budget_bytes=1 << 20,
        max_input_rows=4,
    )
    workload = TwoGroups()
    execution = make_host(facts, workload)
    peak = execution.arena.plan.schedule.peak_memory_bytes
    with pytest.raises(ValueError):
        make_host(replace(facts, memory_budget_bytes=peak - 1), workload)
    (initial,) = execution.admit((), stream)
    original = execution.snapshot(initial, stream)
    if stream is not None:
        stream.synchronize()
    # The two views share one packed AdamW group; AdamW is elementwise, so each matches its own torch parameter.
    parameters = [
        torch.nn.Parameter(original[0][:6].reshape(2, 3).clone()),
        torch.nn.Parameter(original[0][6:].clone()),
    ]
    config = workload.parameters.optimizer
    context = nullcontext() if stream is None else torch.cuda.stream(stream)
    oracle = torch.optim.AdamW(
        parameters,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=device.type == "cuda",
        capturable=device.type == "cuda",
    )
    with context:
        hidden = torch.tensor([[0.4, -0.7]], device=device)
        batch = StepInput.of_batch(WeightedRowBatch(hidden, hidden[:, 0], hidden[:, 0]))
    latest = initial
    for round_ in range(1, 4):
        oracle.zero_grad(set_to_none=True)
        loss = (hidden @ parameters[0] + parameters[1]).square().mean()
        loss.backward()
        oracle.step()
        result = execution.step(batch, stream)
        if stream is not None:
            stream.synchronize()
        assert tuple(result.publication.before_refs) == (latest,)
        (latest,) = result.publication.after_refs
        assert latest.version_id == round_
        actual = execution.snapshot(latest, stream)
        if stream is not None:
            stream.synchronize()
        states = [oracle.state[parameter] for parameter in parameters]

        expected = (
            torch.cat([parameter.flatten() for parameter in parameters]),
            torch.cat([parameter.grad.flatten() for parameter in parameters]),
            states[0]["step"],
            torch.cat([state["exp_avg"].flatten() for state in states]),
            torch.cat([state["exp_avg_sq"].flatten() for state in states]),
        )
        for got, want in zip(actual, expected, strict=True):
            # The CUDA page kernel and torch's foreach AdamW round the last bit differently.
            torch.testing.assert_close(got, want, **({"rtol": 0, "atol": 0} if device.type == "cpu" else {}))
    # A None loss skips the update: no publication, the latest version stays.
    with context:
        pair = hidden.expand(2, 2)
        skipped = StepInput.of_batch(WeightedRowBatch(pair, pair[:, 0], pair[:, 0]))
    result = execution.step(skipped, stream)
    assert result.publication is None
    assert execution.arena.latest(initial.lineage_id) == latest
    assert execution.diagnostics().update_count == 3
    execution.retire((), wait=True)
    assert execution.diagnostics().active_states == execution.diagnostics().pending_state_releases == 0

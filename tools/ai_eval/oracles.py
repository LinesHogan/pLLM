"""Independent acceptance math and CPU state checks; only the trusted scorer imports this module."""

from collections import Counter
from dataclasses import dataclass
from types import ModuleType

import torch

import pllm
from pllm import testing
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.testing_contracts import contract_facts
from tools.ai_eval import reference


def close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare owned values without hiding dtype, shape or infinity-mask mistakes."""
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def declarations(task: str, workload: pllm.Workload, config: pllm.ModelConfig) -> None:
    """Require the requested effects, state scopes, inputs, optimizer and model sites."""
    expected = reference.WORKLOADS[task]()
    assert workload.backend_requirements() == expected.backend_requirements(), "wrong model inputs/effects/targets"
    _, facts = contract_facts(expected, config)
    actual, wanted = workload.build(facts), expected.build(facts)
    assert actual.graph.states == wanted.graph.states, "wrong state scope, keys or mutability"
    actual_fields = {layout.resource: layout.layout.fields for layout in actual.layouts.layouts}
    wanted_fields = {layout.resource: layout.layout.fields for layout in wanted.layouts.layouts}
    assert actual_fields == wanted_fields, "wrong state field shapes or dtypes"
    initial_state(actual, wanted)
    assert actual.imports == wanted.imports, "wrong imported state fields"
    assert actual.effects.overlay_bindings == wanted.effects.overlay_bindings, "wrong adapter math/scale/fields"
    # Model property names are author choices; their effects and bindings must still match.
    assert Counter((member.effect, member.tokens) for member in actual.members.values()) == Counter(
        (member.effect, member.tokens) for member in wanted.members.values()
    ), "missing, extra or incorrect declared effects"
    actual_timing = {task.symbol: task.timing for task in actual.graph.tasks}
    wanted_timing = {task.symbol: task.timing for task in wanted.graph.tasks}
    assert actual_timing == wanted_timing, "wrong effective task timing"
    for name, member in wanted.members.items():
        if member.effect in ("Adapter", "ActivationPatch"):
            continue
        observed = actual.members[name]
        assert observed.effect == member.effect, f"{name}: wrong effect"
        if member.tokens is not None:
            assert observed.tokens == member.tokens, f"{name}: wrong token selection"

    for name, method in wanted.tensor_tasks.items():
        observed = actual.tensor_tasks[name]
        assert observed.sources == method.sources, f"{name}: wrong input/state bindings"
        assert observed.phase == method.phase, f"{name}: wrong token phase"
        if method.writer is not None:
            assert observed.writer is not None, f"{name}: missing state writer"
            assert observed.writer.fields == method.writer.fields, f"{name}: wrong write field order"
            assert observed.writer.resource == method.writer.resource, f"{name}: wrong write state"
        assert observed.sampler_beta == method.sampler_beta, f"{name}: wrong guidance strength"
        assert observed.policy.reduction == method.policy.reduction, f"{name}: wrong loss reduction"
        if method.training is not None:
            assert observed.training is not None, f"{name}: missing optimizer"
            assert observed.training.optimizer == method.training.optimizer, f"{name}: wrong optimizer"
    for observed, wanted_site in zip(actual.sites.rewrites, wanted.sites.rewrites, strict=True):
        assert (observed.site, observed.places) == (wanted_site.site, wanted_site.places), "wrong residual site/layer"
        assert observed.inputs == wanted_site.inputs, "wrong residual state field binding"
        if wanted_site.memory is not None:
            assert observed.memory is not None and observed.memory.chunk == 2, "memory needs request chunk=2"


def initial_state(actual: TensorDeclaration, expected: TensorDeclaration) -> None:
    """Compare declared initialization, including memory zeros and optimizer slots."""
    for layout in actual.layouts.layouts:
        observed = tuple(torch.zeros(field.shape, dtype=field.dtype.torch) for field in layout.layout.fields)
        wanted = tuple(torch.zeros_like(value) for value in observed)
        resource = str(layout.resource)
        actual.initializer(resource, torch.device("cpu"), 7)((observed,), (None,))
        expected.initializer(resource, torch.device("cpu"), 7)((wanted,), (None,))
        for value, reference_value in zip(observed, wanted, strict=True):
            close(value, reference_value)


def steering(task: str, submission: ModuleType, device: str) -> None:
    """Check the population statistic, projected head sum and neutral residual behavior."""
    x, y = torch.randn(3, 8, device=device), torch.randn(3, 8, device=device)
    directions = torch.randn(2, 8, device=device)
    if task == "caa":
        positive, negative = torch.randn(7, 8, device=device), torch.randn(5, 8, device=device)
        # Different population sizes prohibit pairwise subtraction as a substitute for two means.
        expected = torch.stack(positive.unbind()).sum(0) / 7 - torch.stack(negative.unbind()).sum(0) / 5
        close(submission.direction(positive, negative), expected)
        directions = expected
        alpha = 0.7
    else:
        alpha = 0.6
    workload = submission.Workload()
    shift = directions if task == "caa" else directions[0] + directions[1]
    close(workload.intervene(x, y, directions), x + y + alpha * shift)
    close(submission.Workload(alpha=0).intervene(x, y, directions), x + y)
    half_x, half_y = x.to(torch.bfloat16), y.to(torch.bfloat16)
    assert workload.intervene(half_x, half_y, directions).dtype == torch.bfloat16


def dola(submission: ModuleType, device: str) -> None:
    """Enumerate rows/candidates in float64 instead of sharing the candidate's vectorized selection."""
    logits = torch.randn(5, 3, 16, device=device) * 2
    logits[0, 0] = logits[0, 1]  # First candidate wins a tie.
    output: list[torch.Tensor] = []
    for row in logits.double():
        probabilities = [layer.softmax(-1) for layer in row]
        final = probabilities[-1]
        divergences: list[float] = []
        for early in probabilities[:-1]:
            mixture = (final + early) / 2
            divergences.append(
                float(
                    ((final * (final.log() - mixture.log())).sum() + (early * (early.log() - mixture.log())).sum()) / 2
                )
            )
        chosen = max(range(2), key=lambda index: float(divergences[index]))
        desired = final.log() - probabilities[chosen].log()
        desired[final < 0.1 * final.max()] = -torch.inf
        output.append(desired.float())
    close(submission.Workload().contrast(logits) + logits[:, -1], torch.stack(output))
    uniform = torch.zeros(2, 3, 16, device=device)
    close(submission.Workload().contrast(uniform), torch.zeros(2, 16, device=device))


def prompt_lora(submission: ModuleType, device: str) -> None:
    """Compare row-specific factors and the resulting projection against scalar construction."""
    h = torch.randn(3, 8, device=device)
    a, b = torch.zeros(3, 2, 8, device=device), torch.zeros(3, 8, 2, device=device)
    actual_a, actual_b = submission.Workload().produce(h, a, b)
    expected_a = torch.stack([torch.stack([row.tanh(), (2 * row).tanh()]) for row in h])
    expected_b = torch.stack([torch.stack([row / 8, -row / 8], -1) for row in h])
    close(actual_a, expected_a)
    close(actual_b, expected_b)
    close(actual_b @ actual_a, expected_b @ expected_a)
    assert not torch.count_nonzero(a) and not torch.count_nonzero(b), "borrowed templates mutated"


def esamp_math(submission: ModuleType, device: str) -> None:
    """Compare prediction, per-row loss and both factor gradients to per-request linear layers."""
    x, target = torch.randn(3, 8, device=device), torch.randn(3, 8, device=device)
    a = torch.randn(3, 2, 8, device=device, requires_grad=True)
    b = torch.randn(3, 8, 2, device=device, requires_grad=True)
    workload = submission.Workload()
    expected = torch.stack(
        [
            torch.nn.functional.linear(torch.nn.functional.linear(row, down), up)
            for row, down, up in zip(x, a, b, strict=True)
        ]
    )
    close(workload.predict(x, a, b), expected)
    loss = workload.loss(x, target, a, b)
    wanted = torch.stack(
        [torch.nn.functional.mse_loss(row, label) for row, label in zip(expected, target, strict=True)]
    )
    close(loss, wanted)
    gradients = torch.autograd.grad(loss.sum(), (a, b), retain_graph=True)
    wanted_gradients = torch.autograd.grad(wanted.sum(), (a, b))
    for actual, expected_gradient in zip(gradients, wanted_gradients, strict=True):
        close(actual, expected_gradient)


def esamp_state(submission: ModuleType) -> None:
    """Execute one real request SGD-free Adam step and compare every optimizer field."""
    bindings = tuple(testing.hidden_binding(layer, 3) for layer in (0, 2))
    facts = testing.backend_facts(8, bindings, layer_count=3, vocab_size=16, memory_bytes=1 << 20)
    execution = testing.compile_execution(submission.Workload(), facts)
    identities = (testing.RequestIdentity("a", 0, 0), testing.RequestIdentity("b", 1, 0))
    refs = execution.admit(identities)
    x, target = torch.randn(2, 8), torch.randn(2, 8)
    before = [execution.snapshot(ref) for ref in refs]
    try:
        for values in before:
            close(values[0], torch.cat((torch.full((16,), 0.05), torch.zeros(16))))
        testing.run_step(execution, testing.step_input(identities, dict(zip(bindings, (x, target), strict=True))))
        for index, ref in enumerate(refs):
            down = torch.nn.Parameter(torch.full((2, 8), 0.05))
            up = torch.nn.Parameter(torch.zeros(8, 2))
            optimizer = torch.optim.AdamW([down, up], lr=0.01, weight_decay=0)
            loss = torch.nn.functional.mse_loss(x[index] @ down.T @ up.T, target[index])
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            actual = execution.snapshot(execution.arena.latest(ref.lineage_id))
            assert down.grad is not None and up.grad is not None
            expected = (
                torch.cat((down.flatten(), up.flatten())),
                torch.cat((down.grad.flatten(), up.grad.flatten())),
                optimizer.state[down]["step"],
                torch.cat((optimizer.state[down]["exp_avg"].flatten(), optimizer.state[up]["exp_avg"].flatten())),
                torch.cat((optimizer.state[down]["exp_avg_sq"].flatten(), optimizer.state[up]["exp_avg_sq"].flatten())),
            )
            for value, wanted in zip(actual, expected, strict=True):
                close(value, wanted)
    finally:
        execution.retire(identities, wait=True)
        execution.retire((), wait=True)


def memory(submission: ModuleType, device: str) -> None:
    """Use a gradient-derived memory transition, including zero-norm keys and partial chunks."""
    workload = submission.Workload()
    for count in (1, 2):
        x, y = torch.randn(count, 8, device=device), torch.randn(count, 8, device=device)
        x[0] = 0
        matrix = torch.randn(8, 8, device=device, requires_grad=True)
        keys = x / x.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        read = torch.stack([matrix @ key for key in keys])
        gradient = torch.autograd.grad(0.5 * (read - y).square().sum() / count, matrix)[0]
        matrix = matrix.detach()
        before = matrix.clone()
        stream, updated = workload.remember(x, y, matrix)
        close(stream, x + y + 0.5 * read.detach())
        close(updated, matrix - 0.3 * gradient)
        close(matrix, before)


@dataclass
class FakeLayers:
    """Independent positionless decoder for checking algorithm-specific masks and recurrence."""

    embeddings: torch.Tensor
    layer_count: int = 3

    def __post_init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(
        self,
        start: int = 0,
        stop: int | None = None,
        hidden: torch.Tensor | None = None,
        *,
        kv: str = "commit",
        active: torch.Tensor | None = None,
    ) -> pllm.model.LayerOutputs:
        value = self.embeddings if hidden is None else hidden
        stop = self.layer_count if stop is None else stop
        assert kv == "commit", "task requires ordinary KV commit semantics"
        result: list[torch.Tensor] = []
        if active is not None:
            self.calls.append(active.clone())
        for layer in range(start, stop):
            transformed = (value * (0.5 + layer * 0.1) + 0.07).tanh()
            value = transformed if active is None else torch.where(active, transformed, value)
            result.append(value)
        return pllm.model.LayerOutputs(tuple(result), start, self.layer_count)


def looped_depth(submission: ModuleType, device: str) -> None:
    """Compare the requested bounded recurrence, not merely any framework-valid adaptive program."""
    inputs = torch.cat((torch.zeros(1, 8), torch.ones(1, 8), torch.full((1, 8), 4.0))).to(device)
    for threshold in (0.0, 0.4, 10.0):
        decoder = FakeLayers(inputs)
        output = submission.Workload(threshold=threshold).forward(decoder)
        value = (inputs * 0.5 + 0.07).tanh()
        live = torch.ones(3, 1, dtype=torch.bool, device=device)
        masks: list[torch.Tensor] = []
        for _ in range(3):
            live = live & (value.square().mean(-1, keepdim=True) > threshold)
            masks.append(live.clone())
            for layer in (1, 2):
                value = torch.where(live, (value * (0.5 + layer * 0.1) + 0.07).tanh(), value)
        close(output, value)
        assert len(decoder.calls) == 3, "must use three bounded active= calls"
        for actual, wanted in zip(decoder.calls, masks, strict=True):
            close(actual, wanted)


def ttrl(submission: ModuleType, device: str) -> None:
    """Verify grouped voting and real repeated host updates, including empty update suppression."""
    answers = [" B ", "a", "", "2", "1", "1", " b ", "  "]
    groups = [4, 4, 4, 9, 9, 9, 4, 11]
    assert submission.vote(answers, groups) == (
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    )
    assert submission.vote([" A", "b"], [0, 0]) == ([1.0, 0.0], [1.0, 1.0])
    if device != "cpu":
        hidden = torch.randn(8, 8, device=device)
        weight = torch.randn(8, device=device, requires_grad=True)
        targets = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], device=device)
        weights = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0], device=device)
        actual = submission.Workload().loss(hidden, targets, weights, weight)
        expected = (weights * (hidden @ weight - targets).square()).sum() / weights.sum()
        close(actual.output, expected)
        close(torch.autograd.grad(actual.output, weight)[0], torch.autograd.grad(expected, weight)[0])
        return  # Host readiness/stream protocols are outside the optional tensor-only CUDA oracle.
    hidden = torch.randn(8, 8)
    config = pllm.host.HostConfig(torch.device("cpu"), hidden_size=8, max_input_rows=8, state_page_budget_bytes=1 << 20)
    execution = pllm.host.make_host(config, submission.Workload())
    refs = execution.admit(())
    weight = torch.nn.Parameter(torch.zeros(8))
    optimizer = torch.optim.AdamW([weight], lr=0.01, weight_decay=0)
    targets = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    weights = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    actual = execution.snapshot(refs[0])
    try:
        for _ in range(2):
            optimizer.zero_grad()
            loss = (weights * (hidden @ weight - targets).square()).sum() / weights.sum()
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            assert weight.grad is not None
            result = submission.feedback(execution, hidden, answers, groups)
            close(result.loss, loss.detach())
            assert result.publication is not None
            actual = execution.snapshot(result.publication.after_refs[0])
            for value, expected in zip(
                actual,
                (
                    weight,
                    weight.grad,
                    optimizer.state[weight]["step"],
                    optimizer.state[weight]["exp_avg"],
                    optimizer.state[weight]["exp_avg_sq"],
                ),
                strict=True,
            ):
                close(value, expected)
        before = [value.clone() for value in actual]
        result = submission.feedback(execution, hidden, [""] * 8, groups)
        assert result.publication is not None
        for value, expected in zip(execution.snapshot(result.publication.after_refs[0]), before, strict=True):
            close(value, expected)
        assert execution.diagnostics().update_count == 2
        assert execution.diagnostics().loss_count == 3, "feedback must step exactly once, including empty selection"
    finally:
        execution.retire((), wait=True)
    assert refs and execution.diagnostics().active_states == 0


def check_math(task: str, submission: ModuleType, device: str = "cpu") -> None:
    """Run fixed-seed hidden acceptance cases on one submitted module."""
    torch.manual_seed(619)  # pyright: ignore[reportUnknownMemberType]
    if task in ("caa", "iti"):
        steering(task, submission, device)
    elif task == "dola":
        dola(submission, device)
    elif task == "prompt_lora":
        prompt_lora(submission, device)
    elif task == "esamp":
        esamp_math(submission, device)
        if device == "cpu":
            esamp_state(submission)
    elif task == "memory":
        memory(submission, device)
    elif task == "looped_depth":
        looped_depth(submission, device)
    elif task == "ttrl":
        ttrl(submission, device)
    else:
        raise ValueError(f"choose a task in run.TASKS, got {task}")

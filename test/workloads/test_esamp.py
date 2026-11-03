"""ESamp request distillation against an independent gated-residual/AdamW oracle.

Covers the step lifecycle (prediction before training, sampler-completion ordering,
cancellation, failed publication and admission), per-request version isolation, the
CPU/CUDA and captured implementations, and an ordinary authored distiller with memory.
"""

from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import pllm
from pllm import testing
from pllm.config import ExecutionOptions
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, TaskSymbol
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.declaration import Dimensions
from pllm.execution.module_state import initial_values
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import TaskOutput, TensorBatch
from pllm.execution.tensor_task import TensorTask
from pllm.testing import backend_facts
from pllm.workloads.esamp import ESampConfig, ESampWorkload

BINDINGS = tuple(
    PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]")
    for layer in (0, -1)
)


def facts_for(device, hidden_size, pages, lineages=2, memory_bytes=10000):
    provider = PageBundleProviderFacts("test.esamp@1", pages, StateDeviceKind(device.type))
    timeline = model_timeline(device)
    return backend_facts(hidden_size, BINDINGS, timeline, provider, max_lineages=lineages, memory_bytes=memory_bytes)


def step_input(identities, source, target, step_id, order=None, stream=None):
    """Packed decode rows with the source (layer 0) and target (layer -1) hidden states."""
    count = len(identities)
    hidden = {BINDINGS[0]: source[:count], BINDINGS[1]: target[:count]}
    return testing.step_input(
        identities, hidden, step_id=step_id, token_offset=step_id, sampler_order=order, stream=stream
    )


def effective(guidance):
    """Rows the guidance actually changes (nonzero weight) and their predictions."""
    if guidance.row_weights is None:
        return guidance.affected_row_ids, guidance.values
    keep = [index for index, weight in enumerate(guidance.row_weights.tolist()) if weight]
    return tuple(guidance.affected_row_ids[index] for index in keep), guidance.values[keep]


def assert_roots(actual, expected, **tolerance):
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, **tolerance)


def adamw_roots(optimizer, parameter):
    """The published root layout: parameter, gradient, step, exp_avg, exp_avg_sq."""
    state = optimizer.state[parameter]
    return parameter, parameter.grad, state["step"].to(parameter.device), state["exp_avg"], state["exp_avg_sq"]


def reference(source, parameters, hidden=4, rank=2, normalized=True):
    """Gated residual with optional layer norm over one flattened parameter row per input row."""
    up, gate, down, bias = parameters[:, : 3 * hidden * rank + rank].split((hidden * rank,) * 3 + (rank,), dim=1)
    u = torch.bmm(source.unsqueeze(1), up.view(-1, hidden, rank)).squeeze(1)
    z = torch.bmm(source.unsqueeze(1), gate.view(-1, hidden, rank)).squeeze(1) + bias
    y = source + torch.bmm((u * F.silu(z)).unsqueeze(1), down.view(-1, rank, hidden)).squeeze(1)
    if not normalized:
        return y
    scale, shift = parameters[:, -2 * hidden :].split((hidden, hidden), dim=1)
    return F.layer_norm(y, (hidden,)) * scale + shift


def reject_publication(arena, write):
    raise RuntimeError("injected publication failure")


def test_esamp_predicts_before_training_and_isolates_request_versions(monkeypatch, device_name) -> None:
    if device_name == "cpu":
        _request_lifecycle(monkeypatch)
    for normalized in (False, True):
        _esamp_native_oracle(torch.device(device_name), normalized)
    _ordinary_request_distillation(device_name)
    _module_request_distillation(device_name)


def _request_lifecycle(monkeypatch):
    refs = StateRefs(np.array([0, 0, 1]), np.array([1, 2, 3]), np.array([9, 5, 0]))
    history = (
        StateRefs(refs.layouts[:1], refs.lineages[:1], np.array([3])),
        StateRefs(refs.layouts[:2], refs.lineages[:2], np.array([6, 5])),
    )
    for lag, expected in enumerate(([9, 5, 0], [6, 5, 0], [3, 5, 0])):
        np.testing.assert_array_equal(
            RequestStateDirectory._rewind(refs, history[-lag:] if lag else ()).versions, expected
        )
    with pytest.raises(ValueError):
        ExecutionOptions(publish_lag=-1)
    with pytest.raises(ValueError):
        ESampConfig(learning_rate=float("nan"))
    config = ESampConfig(rank=2, intervention_beta=0.2, learning_rate=0.03, weight_decay=0.1)
    facts = facts_for(torch.device("cpu"), 4, 6)
    execution = ESampWorkload(config).compile_backend(facts).execution
    identities = (RequestIdentity("first", 0, 0), RequestIdentity("second", 0, 1))
    latest = list(execution.admit(identities))
    parameters = [execution.snapshot(ref)[0].requires_grad_() for ref in latest]
    optimizers = [
        torch.optim.AdamW((p,), lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=0.1)
        for p in parameters
    ]
    generator = torch.Generator().manual_seed(facts.initialization_seed)
    normal = (torch.empty(8).normal_(0.0, 0.02, generator=generator) for _ in range(3))
    expected = torch.cat((*normal, torch.ones(2), torch.ones(4), torch.zeros(4)))
    for parameter in parameters:  # every request starts from the module's initialization
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    source = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2], [0.1, 0.3, -0.2, 0.8]])
    target = torch.tensor([[0.6, 0.0, -0.3, 0.2], [0.4, -0.2, 0.8, 0.1], [-0.2, 0.5, 0.1, 0.4]])
    sampler = execution.sampler_anchor_id
    for step_id, owners in enumerate(((0, 1, 0), (0,))):
        step = step_input(tuple(identities[owner] for owner in owners), source, target, step_id)
        before = [execution.snapshot(ref) for ref in latest]
        cancelled = execution.start_step(step)
        execution.execute_anchor(cancelled, AnchorId(0))
        execution.cancel_step(cancelled)
        with monkeypatch.context() as fault:
            # A failed arena publication must leave every request at its exact version.
            fault.setattr(PageBundleArena, "publish", reject_publication)
            with pytest.raises(RuntimeError):
                execution.step(step)
        for ref, root in zip(latest, before, strict=True):
            assert_roots(execution.snapshot(ref), root, rtol=0, atol=0)
        handle = execution.start_step(step)
        for anchor in range(sampler + 1):
            execution.execute_anchor(handle, AnchorId(anchor))
        guidance = execution.sampler_guidance(handle)
        execution.complete_sampler(handle)
        for anchor in execution.arena.plan.timeline.anchors[sampler + 1 :]:
            execution.execute_anchor(handle, AnchorId(anchor.ordinal))
        result = execution.finish_step(handle)
        # Only a trained (version > 0) distiller guides sampling.
        rows, values = effective(guidance[0])
        if step_id == 0:
            assert rows == ()
        else:
            assert rows == (0,)
            torch.testing.assert_close(values, reference(source[:1], parameters[0][None]).detach())
        for index, owner in enumerate(sorted(set(owners))):
            selected = [i for i, selected_owner in enumerate(owners) if selected_owner == owner]
            optimizers[owner].zero_grad()
            loss = F.mse_loss(
                reference(source[selected], parameters[owner].expand(len(selected), -1)), target[selected]
            )
            loss.backward()
            torch.testing.assert_close(result.loss[index], loss.detach())
            optimizers[owner].step()
            assert result.publication.before_refs[index] == latest[owner]
            latest[owner] = result.publication.after_refs[index]
            assert latest[owner].version_id == step_id + 1
            assert_roots(execution.snapshot(latest[owner]), adamw_roots(optimizers[owner], parameters[owner]))
        for owner in set(range(2)) - set(owners):
            assert_roots(execution.snapshot(latest[owner]), before[owner], rtol=0, atol=0)

    execution.retire(identities, wait=True)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_states == diagnostics.pending_state_releases == 0
    assert diagnostics.loss_count == diagnostics.update_count == 3 and diagnostics.trained_rows == 4
    with monkeypatch.context() as fault:
        # Admission publishes initial versions; a failure must not leak a lineage.
        fault.setattr(PageBundleArena, "publish", reject_publication)
        with pytest.raises(RuntimeError):
            execution.admit(identities)
    assert execution.diagnostics().active_states == 0
    execution.admit(identities)
    execution.retire(identities)


def _esamp_native_oracle(device, normalized):
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    config = ESampConfig(
        rank=4, output_layernorm=normalized, intervention_beta=0.2, learning_rate=0.03, weight_decay=0.1
    )
    workload = ESampWorkload(config)
    facts = facts_for(device, 8, 8, lineages=3, memory_bytes=100000)
    facts, declaration = workload.specialize(facts, ExecutionOptions(capture_memory_bytes=16 * 1024**2))
    deployment = workload.bind_backend(facts, workload.plan_backend(facts, declaration), declaration)
    execution = deployment.execution
    plan = execution.arena.plan
    assert plan.timeline.visibility == VisibilitySource.post_sampler(1)
    assert any(task.lane == "side" for task in plan.schedule.tasks) == (stream is not None)
    if stream is not None and normalized:
        for lag in (0, 1, 2):
            attachment = workload.plan_backend(
                replace(facts, timeline=replace(facts.timeline, visibility=VisibilitySource.post_sampler(lag))),
                declaration,
            )
            assert any(task.lane == "side" for task in attachment.plan.schedule.tasks) == bool(lag)
    identities = tuple(RequestIdentity(str(i), i, 0) for i in range(3))
    initial = execution.admit(identities, stream)
    held = execution.arena.pin_exact(StateRefs.of(initial), stream)
    with nullcontext() if stream is None else torch.cuda.stream(stream):
        parameters = [execution.snapshot(ref, stream)[0].requires_grad_() for ref in initial]
        originals = [parameter.detach().clone() for parameter in parameters]
        optimizers = [
            torch.optim.AdamW((p,), lr=0.03, weight_decay=0.1, capturable=stream is not None, foreach=True)
            for p in parameters
        ]
        # A strided source view: the workload must not assume contiguous hidden rows.
        source = torch.linspace(-0.7, 0.9, 48, device=device).view(3, 16)[:, ::2]
        target = source.flip(1) * 0.7
        for step_id, owners in enumerate(((1, 0, 1), (0,))):
            value = step_input(tuple(identities[o] for o in owners), source, target, step_id, stream=stream)
            for optimizer in optimizers:
                optimizer.zero_grad()
            stacked = torch.stack([parameters[owner] for owner in owners])
            prediction = reference(source[: len(owners)], stacked, hidden=8, rank=4, normalized=normalized)
            row_losses = (prediction - target[: len(owners)]).square().mean(1)
            expected_losses = {
                owner: row_losses[[i for i, key in enumerate(owners) if key == owner]].mean() for owner in set(owners)
            }
            sum(expected_losses.values()).backward()
            result = execution.step(value, stream)
            if step_id:
                torch.testing.assert_close(result.guidance[0].values, prediction, rtol=1e-5, atol=1e-6)
            for ref, loss in zip(result.publication.after_refs, result.loss, strict=True):
                owner = next(i for i, first in enumerate(initial) if first.lineage_id == ref.lineage_id)
                torch.testing.assert_close(loss, expected_losses[owner], rtol=1e-5, atol=1e-6)
                optimizers[owner].step()
                expected = adamw_roots(optimizers[owner], parameters[owner])
                assert_roots(execution.snapshot(ref, stream), expected, rtol=1e-4, atol=1e-6)
        for ref, expected in zip(initial, originals, strict=True):
            torch.testing.assert_close(execution.arena.snapshot(held, ref)[0], expected, rtol=0, atol=0)
    execution.arena.wait_release(execution.arena.release_pin(held))
    execution.retire(identities)
    assert execution.diagnostics().active_states == 0


class Distillation(pllm.Workload):
    """An authored request distiller keyed by prompt and sample, with a captured carry."""

    heads = pllm.RequestState(
        {"weight": lambda m: (m.hidden_size, m.hidden_size)},
        optimizer=pllm.AdamW(lr=0.03, weight_decay=0.1),
        key_fields=("prompt_index", "sample_index"),
        carry={"carry": lambda m: (m.hidden_size,)},
    )

    def ready(self, versions: torch.Tensor = pllm.version("heads")) -> torch.Tensor:
        return versions > 0

    @pllm.guide_sampling(beta=0.2, rows="ready")
    def predict(
        self,
        source: torch.Tensor = pllm.hidden(0),
        weight: torch.Tensor = pllm.parameter("heads", "weight"),
    ) -> torch.Tensor:
        return torch.bmm(source.unsqueeze(1), weight).squeeze(1)

    @pllm.train(state="heads", reduction="state_mean")
    def train(
        self,
        source: torch.Tensor = pllm.carry(pllm.hidden(0), state="heads", field="carry"),
        target: torch.Tensor = pllm.hidden(-1),
        weight: torch.Tensor = pllm.parameter("heads", "weight"),
        carry: torch.Tensor = pllm.parameter("heads", "carry"),
    ) -> torch.Tensor:
        prediction = torch.bmm(source.unsqueeze(1), weight).squeeze(1) + carry
        return (prediction - target).square().mean(1)


def _ordinary_request_distillation(device_name) -> None:
    device = torch.device(device_name)
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    workload = Distillation()
    options = ExecutionOptions(capture_memory_bytes=16 * 1024**2)
    execution = workload.compile_backend(facts_for(device, 4, 9, lineages=3), options)
    execution = execution.execution
    # Same request id, different samples: the keys keep three separate lineages.
    identities = tuple(RequestIdentity("same-id", 0, index) for index in range(3))
    latest = list(execution.admit(identities, stream))
    weights = [torch.nn.Parameter(torch.zeros((4, 4), device=device)) for _ in identities]
    optimizers = [torch.optim.AdamW((weight,), lr=0.03, weight_decay=0.1) for weight in weights]
    source = torch.linspace(-0.6, 0.8, 12, device=device).reshape(3, 4)
    target = torch.linspace(0.7, -0.3, 12, device=device).reshape(3, 4)
    initial = tuple(latest)
    held = execution.arena.pin_exact(StateRefs.of(initial), stream)
    # Cross-step memory: carries hold each lineage's last published capture; step 0 reads zeros.
    carries = [torch.zeros(4, device=device) for _ in identities]
    for step_id, owners in enumerate(((0, 1, 0), (2, 0, 0))):
        value = step_input(tuple(identities[o] for o in owners), source, target, step_id, (2, 0, 1), stream)
        links = value.sampler_rows.links
        active = tuple(link for link in links if latest[owners[link.packed_row_index]].version_id > 0)
        weight_of = lambda link: weights[owners[link.packed_row_index]].detach()
        expected = [source[link.packed_row_index] @ weight_of(link) for link in active]
        result = execution.step(value, stream)
        rows, values = effective(result.guidance[0])
        assert rows == tuple(link.sampler_row_index for link in active) and result.guidance[0].beta == 0.2
        torch.testing.assert_close(values, torch.stack(expected) if expected else source.new_empty((0, 4)))
        for before, after, loss in zip(
            result.publication.before_refs, result.publication.after_refs, result.loss, strict=True
        ):
            owner = latest.index(before)
            selected = [index for index, selected_owner in enumerate(owners) if selected_owner == owner]
            optimizers[owner].zero_grad()
            oracle = F.mse_loss(source[selected] @ weights[owner] + carries[owner], target[selected])
            oracle.backward()
            optimizers[owner].step()
            torch.testing.assert_close(loss, oracle.detach())
            # Each lineage's memory receives its last selected row's captured source.
            carries[owner] = source[selected[-1]].clone()
            weight, gradient, step, exp_avg, exp_avg_sq = adamw_roots(optimizers[owner], weights[owner])
            expected_roots = (weight, gradient, step, exp_avg, exp_avg_sq)
            expected_roots = (*(t.flatten() if t.dim() else t for t in expected_roots), carries[owner])
            assert_roots(execution.snapshot(after, stream), expected_roots)
            latest[owner] = after
    for ref in initial:
        assert torch.count_nonzero(execution.arena.snapshot(held, ref)[0]) == 0
    execution.arena.wait_release(execution.arena.release_pin(held))
    execution.retire(identities, wait=True)
    diagnostics = execution.diagnostics()
    assert diagnostics.loss_count == diagnostics.update_count == 4 and diagnostics.trained_rows == 6
    assert diagnostics.active_states == diagnostics.pending_state_releases == 0


def test_sampler_adapts_step_results_in_memory_trace_and_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed steps never re-enter prediction evaluation, including the meta resource trace."""
    evaluate = TensorTask.evaluate
    adapt = TensorTask.sampler_output
    devices: list[str] = []
    calls: list[TaskSymbol] = []

    def dependency(task: TensorTask, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        calls.append(task.symbol)
        return evaluate(task, data, batch)

    def output(
        task: TensorTask,
        data: TensorFrame,
        batch: TensorBatch,
        weights: torch.Tensor | None,
    ) -> TaskOutput:
        result = adapt(task, data, batch, weights)
        assert result.guidance is not None
        devices.append(result.guidance.values.device.type)
        return result

    monkeypatch.setattr(TensorTask, "evaluate", dependency)
    monkeypatch.setattr(TensorTask, "sampler_output", output)
    workload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.2))
    execution = workload.compile_backend(facts_for(torch.device("cpu"), 4, 8), ExecutionOptions()).execution
    assert "meta" in devices
    identities = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    latest = execution.admit(identities)
    source = torch.linspace(-0.3, 0.7, 8).reshape(2, 4)
    devices.clear()
    calls.clear()
    for step in range(2):
        expected = reference(source, torch.stack([execution.snapshot(ref)[0] for ref in latest]))
        result = execution.step(step_input(identities, source, source.flip(1), step, (1, 0)))
        latest = result.publication.after_refs
        assert result.guidance[0].affected_row_ids == (0, 1)
        torch.testing.assert_close(result.guidance[0].row_weights, torch.full((2,), float(step)))
        torch.testing.assert_close(result.guidance[0].values, expected.flip(0) if step else torch.zeros_like(expected))
    assert devices == ["cpu", "cpu"] and calls.count(TaskSymbol("predict")) == 2
    execution.retire(identities)


class Head(torch.nn.Module):
    """An ordinary PyTorch module, used unchanged as per-request state."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.up, self.down = torch.nn.Linear(width, 3), torch.nn.Linear(3, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.down(torch.tanh(self.up(hidden)))


class ModuleDistillation(pllm.Workload):
    head = pllm.RequestState(module=lambda m: Head(m.hidden_size), optimizer=pllm.AdamW(lr=0.03))

    def ready(self, versions: torch.Tensor = pllm.version("head")) -> torch.Tensor:
        return versions > 0

    @pllm.guide_sampling(beta=0.2, rows="ready")
    def predict(self, source: torch.Tensor = pllm.hidden(0), head: Head = pllm.module("head")) -> torch.Tensor:
        return head(source)

    @pllm.train(state="head", reduction="state_mean")
    def train(
        self,
        source: torch.Tensor = pllm.hidden(0),
        target: torch.Tensor = pllm.hidden(-1),
        head: Head = pllm.module("head"),
    ) -> torch.Tensor:
        return (head(source) - target).square().mean(1)


def _module_request_distillation(device_name) -> None:
    """Module state matches one torch copy of the module per request, trained by torch AdamW."""
    device = torch.device(device_name)
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    options = ExecutionOptions(capture_memory_bytes=16 * 1024**2)
    facts = facts_for(device, 4, 9, lineages=2)
    execution = ModuleDistillation().compile_backend(facts, options).execution
    identities = tuple(RequestIdentity(f"r{index}", 0, 0) for index in range(2))
    latest = list(execution.admit(identities, stream))
    start = initial_values(lambda m: Head(m.hidden_size), Dimensions(4, 0), facts.initialization_seed, torch.float32)
    copies = [Head(4).to(device) for _ in identities]
    for copy in copies:
        torch.nn.utils.vector_to_parameters(start.to(device, copy=True), copy.parameters())
    optimizers = [torch.optim.AdamW(copy.parameters(), lr=0.03, weight_decay=0.0) for copy in copies]
    source = torch.linspace(-0.6, 0.8, 8, device=device).reshape(2, 4)
    target = torch.linspace(0.7, -0.3, 8, device=device).reshape(2, 4)
    for step_id in range(3):
        result = execution.step(step_input(identities, source, target, step_id, stream=stream), stream)
        rows, values = effective(result.guidance[0])
        if step_id:  # predictions use each request's own parameters once it has trained
            assert rows == (0, 1)
            with torch.no_grad():
                torch.testing.assert_close(values, torch.cat([copies[i](source[i : i + 1]) for i in (0, 1)]))
        for index, (copy, optimizer) in enumerate(zip(copies, optimizers, strict=True)):
            optimizer.zero_grad()
            F.mse_loss(copy(source[index : index + 1]), target[index : index + 1]).backward()
            optimizer.step()
            after = result.publication.after_refs[result.publication.before_refs.index(latest[index])]
            flat = torch.nn.utils.parameters_to_vector(copy.parameters()).detach()
            torch.testing.assert_close(execution.snapshot(after, stream)[0], flat)
            latest[index] = after
    execution.retire(identities)

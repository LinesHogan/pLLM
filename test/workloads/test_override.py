"""OverRIDE and ordinary sampler-feedback heads match an independent PyTorch Adam oracle.

Each scenario compiles a workload, drives it through the public step lifecycle with
hand-built packed rows and sampler feedback, and compares guidance, loss, gradients
and the published optimizer state with ``torch.optim``.
"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

import pllm
from pllm import testing
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.runtime_facts import model_timeline
from pllm.workloads.override import OverRIDEConfig, OverRIDEWorkload
from test.backend_facts import assert_finite_resources, backend_facts

BINDING = testing.hidden_binding(-1, HiddenSite.MODEL_OUTPUT, 1)
HIDDEN = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2], [0.1, 0.3, -0.2, 0.8]])
BASE = torch.linspace(-1, 1, 21).reshape(3, 7)
TOKENS = torch.tensor([2, 4, 1])


def facts_for(device, lineages=3, pages=3):
    provider = PageBundleProviderFacts("test.page@1", pages, StateDeviceKind(device.type))
    timeline = model_timeline(device)
    return backend_facts(
        4, (BINDING,), timeline, provider, max_lineages=lineages, memory_bytes=65536, max_rows=8, vocab_size=7
    )


def step_input(identities, hidden, step_id, order=None, prefill=False, stream=None):
    phases = tuple(HiddenPhase.PREFILL if prefill and i == 0 else HiddenPhase.DECODE for i in range(len(identities)))
    return testing.step_input(
        identities,
        {BINDING: hidden[: len(identities)]},
        step_id=step_id,
        token_offset=step_id,
        phases=phases,
        sampler_order=order,
        stream=stream,
    )


def samples(indices, step_id, **kwargs):
    """One request per row; the sample index selects the OverRIDE iteration."""
    return step_input(tuple(RequestIdentity(str(i), i, s) for i, s in enumerate(indices)), HIDDEN, step_id, **kwargs)


def run(execution, value, base=BASE, tokens=TOKENS, stream=None):
    """One step that delivers sampler feedback at the sampler anchor."""
    return testing.run_step(execution, value, feedback=SamplerFeedback(value.sampler_rows, base, tokens), stream=stream)


def adam_roots(optimizer, *parameters):
    """The published root layout: parameters, gradients, step, exp_avg, exp_avg_sq."""
    states = [optimizer.state[parameter] for parameter in parameters]
    flat = lambda tensors: torch.cat([tensor.flatten() for tensor in tensors])
    step = states[0]["step"].to(parameters[0].device)
    return (
        flat(parameters),
        flat(p.grad for p in parameters),
        step,
        flat(s["exp_avg"] for s in states),
        flat(s["exp_avg_sq"] for s in states),
    )


def assert_roots(actual, expected, exact=False):
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, **({"rtol": 0, "atol": 0} if exact else {}))


def head_logits(down, up, row, sample):
    return HIDDEN[row] @ down[sample].T @ up[sample].T


def test_override_matches_reweight_adam_and_final_noop() -> None:
    facts = facts_for(torch.device("cpu"))
    config = OverRIDEConfig(rank=2, num_iterations=3, learning_rate=0.03)
    execution = OverRIDEWorkload(config).compile_backend(assert_finite_resources(OverRIDEWorkload(config), facts))
    execution = execution.execution
    (initial,) = execution.admit(())
    original = execution.snapshot(initial)
    held = execution.arena.pin_exact(StateRefs.of((initial,)))
    down = torch.nn.Parameter(original[0][:16].reshape(2, 2, 4).clone())
    up = torch.nn.Parameter(original[0][16:].reshape(2, 7, 2).clone())
    optimizer = torch.optim.Adam((down, up), lr=config.learning_rate, foreach=False)
    latest = initial
    # Sample i is guided by head i - 1 and trains head i; the last iteration only guides.
    for step_id, indices in enumerate(((0, 1, 2), (1, 0, 2), (2, 2, 2))):
        value = samples(indices, step_id, order=(2, 0, 1) if step_id == 1 else None, prefill=step_id == 1)
        rows = value.rows.rows
        decode = [link for link in value.sampler_rows.links if rows[link.packed_row_index].phase is HiddenPhase.DECODE]
        sample = lambda link: indices[link.packed_row_index]
        expected = torch.stack(
            [
                config.lambd * head_logits(down, up, link.packed_row_index, sample(link) - 1)
                if sample(link)
                else torch.zeros(7)
                for link in decode
            ]
        ).detach()
        before = execution.snapshot(latest)
        result = run(execution, value)
        torch.testing.assert_close(result.guidance.values, expected)
        assert result.guidance.affected_row_ids == tuple(link.sampler_row_index for link in decode)
        selected = [link for link in decode if sample(link) < 2]
        if not selected:
            assert result.loss is result.publication is None
            assert_roots(execution.snapshot(latest), before, exact=True)
            continue
        optimizer.zero_grad()
        logits = [
            BASE[link.sampler_row_index] + head_logits(down, up, link.packed_row_index, sample(link))
            for link in selected
        ]
        loss = F.cross_entropy(torch.stack(logits), TOKENS[[link.sampler_row_index for link in selected]])
        loss.backward()
        optimizer.step()
        assert tuple(result.publication.before_refs) == (latest,)
        (latest,) = result.publication.after_refs
        torch.testing.assert_close(result.loss, loss.detach())
        assert_roots(execution.snapshot(latest), adam_roots(optimizer, down, up))
    assert_roots(execution.arena.snapshot(held, initial), original, exact=True)
    execution.arena.wait_release(execution.arena.release_pin(held))
    execution.retire(())
    assert execution.diagnostics().loss_count == execution.diagnostics().update_count == 2
    assert execution.diagnostics().active_lineages == 0

    single = OverRIDEWorkload(replace(config, num_iterations=1)).compile_backend(facts).execution
    single.admit(())
    result = run(single, samples((0,), 6), BASE[:1], TOKENS[:1])
    assert not result.guidance.values.any() and result.publication is result.loss is None
    single.retire(())

    def zero_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), labels.long()) * 0

    modified = replace(OverRIDEWorkload(config), objective=zero_loss).compile_backend(facts).execution
    modified.admit(())
    result = run(modified, samples((0, 1, 2), 7))
    assert result.loss.item() == 0 and result.publication is not None
    modified.retire(())

    _request_heads()
    _chained_heads(facts)
    _independent_domains()


def _request_heads() -> None:
    class RequestHeads(pllm.Workload):
        name = "request_heads"
        heads = pllm.RequestState({"weight": (4, 7)}, optimizer=pllm.AdamW(lr=0.03))

        @pllm.add_logits()
        def reweight(
            self,
            hidden: torch.Tensor = pllm.hidden(-1, HiddenSite.MODEL_OUTPUT),
            samples: torch.Tensor = pllm.samples(),
            weight: torch.Tensor = pllm.parameter("heads", "weight"),
        ) -> torch.Tensor:
            return torch.bmm(hidden.unsqueeze(1), weight).squeeze(1)

        def eligible(self, samples: torch.Tensor = pllm.samples()) -> torch.Tensor:
            return samples == 0

        @pllm.train(state="heads", rows="eligible")
        def train(
            self,
            hidden: torch.Tensor = pllm.hidden(-1, HiddenSite.MODEL_OUTPUT),
            samples: torch.Tensor = pllm.samples(),
            weight: torch.Tensor = pllm.parameter("heads", "weight"),
            base: torch.Tensor = pllm.base_logits(),
            tokens: torch.Tensor = pllm.sampled_tokens(),
        ) -> torch.Tensor:
            return F.cross_entropy(base + torch.bmm(hidden.unsqueeze(1), weight).squeeze(1), tokens)

    requests = RequestHeads().compile_backend(facts_for(torch.device("cpu"), pages=9)).execution
    value = samples((0, 2, 0), 8)
    identities = tuple(row.identity for row in value.rows.rows)
    initial = requests.admit(identities)
    untouched = requests.snapshot(initial[1])
    result = run(requests, value)
    # Only the eligible requests publish; the ineligible one keeps its exact version.
    assert tuple(result.publication.before_refs) == (initial[0], initial[2])
    assert requests.arena.latest(initial[1].lineage_id) == initial[1]
    assert_roots(requests.snapshot(initial[1]), untouched, exact=True)
    weights = torch.nn.Parameter(torch.zeros((2, 4, 7)))
    optimizer = torch.optim.Adam((weights,), lr=0.03, foreach=False)
    loss = F.cross_entropy(BASE[[0, 2]] + torch.bmm(HIDDEN[[0, 2]].unsqueeze(1), weights).squeeze(1), TOKENS[[0, 2]])
    loss.backward()
    optimizer.step()
    torch.testing.assert_close(result.loss, loss.detach())
    for index, ref in enumerate(result.publication.after_refs):
        torch.testing.assert_close(requests.snapshot(ref)[0].reshape(4, 7), weights[index])
    requests.retire(identities)
    assert requests.diagnostics().update_count == 2 and requests.diagnostics().active_lineages == 0


def _chained_heads(facts) -> None:
    def initialize_chain(parameters, generator):
        parameters["weight"].copy_(torch.linspace(-0.2, 0.3, 28).reshape(4, 7))

    class ChainedHeads(pllm.Workload):
        heads = pllm.InstanceState({"weight": (4, 7)}, optimizer=pllm.AdamW(lr=0.03), initialize=initialize_chain)

        @pllm.add_logits()
        @pllm.differentiate(inputs=("hidden",), parameters=True)
        def reweight(
            self,
            hidden: torch.Tensor = pllm.hidden(-1, HiddenSite.MODEL_OUTPUT),
            samples: torch.Tensor = pllm.samples(),
            weight: torch.Tensor = pllm.parameter("heads", "weight"),
        ) -> torch.Tensor:
            return hidden @ weight

        @pllm.on(pllm.Phase.AFTER_SAMPLE)
        @pllm.differentiate(inputs=("features",))
        def frozen(
            self,
            features: torch.Tensor = pllm.result("reweight"),
            weight: torch.Tensor = pllm.parameter("heads", "weight"),
        ) -> torch.Tensor:
            return features @ weight.T @ weight

        def eligible(self, samples: torch.Tensor = pllm.samples()) -> torch.Tensor:
            return samples == 0

        @pllm.train(state="heads", rows="eligible")
        @pllm.differentiate(inputs=("features",))
        def train(
            self,
            features: torch.Tensor = pllm.result("frozen"),
            samples: torch.Tensor = pllm.samples(),
            base: torch.Tensor = pllm.base_logits(),
            tokens: torch.Tensor = pllm.sampled_tokens(),
        ) -> torch.Tensor:
            return F.cross_entropy(base + features, tokens)

    chain = ChainedHeads().compile_backend(facts).execution
    (initial,) = chain.admit(())
    weight = torch.nn.Parameter(chain.snapshot(initial)[0].reshape(4, 7).clone())
    oracle_input = HIDDEN.clone().requires_grad_(True)
    optimizer = torch.optim.Adam((weight,), lr=0.03, foreach=False)
    predicted = oracle_input @ weight
    # ``frozen`` differentiates only its features, so the weight it reads is a constant.
    frozen = predicted @ weight.detach().T @ weight.detach()
    loss = F.cross_entropy(BASE[[0, 2]] + frozen[[0, 2]], TOKENS[[0, 2]])
    loss.backward()
    optimizer.step()
    result = run(chain, samples((0, 2, 0), 9))
    torch.testing.assert_close(result.guidance.values, predicted.detach())
    torch.testing.assert_close(result.loss, loss.detach())
    torch.testing.assert_close(result.input_gradients["reweight", "hidden"], oracle_input.grad)
    torch.testing.assert_close(chain.snapshot(result.publication.after_refs[0])[0].reshape(4, 7), weight)
    latest = result.publication.after_refs[0]
    before = chain.snapshot(latest)
    cancelled = chain.start_step(samples((0, 2, 0), 10))
    chain.execute_anchor(cancelled, AnchorId(0))
    chain.cancel_step(cancelled)
    assert chain.arena.latest(latest.lineage_id) == latest
    # No eligible row: no loss, no gradients and no publication.
    result = run(chain, samples((2, 2, 2), 11))
    assert result.publication is result.loss is None and not result.input_gradients
    assert_roots(chain.snapshot(latest), before, exact=True)
    assert chain.diagnostics().update_count == 1
    chain.retire(())


class Domains(pllm.Workload):
    """Sample-index bits pick forward rows (1), input gradients (2), parameter gradients (4), updates (8)."""

    def __init__(self, request):
        def initialize(parameters, generator):
            parameters["weight"].copy_(torch.linspace(-0.2, 0.3, 28).reshape(4, 7))

        state = pllm.RequestState if request else pllm.InstanceState
        self.heads = state(
            {"weight": (4, 7)},
            optimizer=pllm.AdamW(lr=0.03, weight_decay=0.1),
            initialize=initialize,
            memory={"carry": (7,)},
        )

    def forward_rows(self, samples: torch.Tensor = pllm.samples()) -> torch.Tensor:
        return samples.bitwise_and(1) != 0

    def input_rows(self, samples: torch.Tensor = pllm.samples()) -> torch.Tensor:
        return samples.bitwise_and(2) != 0

    def parameter_rows(self, samples: torch.Tensor = pllm.samples()) -> torch.Tensor:
        return samples.bitwise_and(4) != 0

    def updates(
        self,
        samples: torch.Tensor = pllm.samples(),
        tokens: torch.Tensor = pllm.sampled_tokens(),
    ) -> torch.Tensor:
        return (samples.bitwise_and(8) != 0) & (tokens == 0)

    @pllm.add_logits(rows="forward_rows")
    @pllm.differentiate(inputs=("hidden",), parameters=True, input_rows="input_rows", parameter_rows="parameter_rows")
    def reweight(
        self,
        hidden: torch.Tensor = pllm.hidden(-1, HiddenSite.MODEL_OUTPUT),
        samples: torch.Tensor = pllm.samples(),
        weight: torch.Tensor = pllm.parameter("heads", "weight", per_row=True),
    ) -> torch.Tensor:
        return torch.bmm(hidden.unsqueeze(1), weight).squeeze(1)

    @pllm.train(state="heads", update="updates", rows="forward_rows")
    @pllm.differentiate(inputs=("features",))
    def train(
        self,
        features: torch.Tensor = pllm.capture(pllm.result("reweight"), state="heads", field="carry"),
        samples: torch.Tensor = pllm.samples(),
        tokens: torch.Tensor = pllm.sampled_tokens(),
    ) -> torch.Tensor:
        return features.square().mean()


class EarlyFeedback(Domains):
    @pllm.on(pllm.Phase.BEFORE_SAMPLE)
    def early(self, tokens: torch.Tensor = pllm.sampled_tokens()) -> torch.Tensor:
        return tokens == 0


class IntegerRows(Domains):
    def forward_rows(self, samples: torch.Tensor = pllm.samples()) -> torch.Tensor:
        return samples


def _independent_domains() -> None:
    cpu = facts_for(torch.device("cpu"), lineages=4, pages=16)
    with pytest.raises(ValueError):  # sampled tokens do not exist before sampling
        EarlyFeedback(False).compile_backend(cpu)
    identities = tuple(RequestIdentity(str(i), i, 0) for i in range(4))
    rejected = IntegerRows(False).compile_backend(cpu).execution
    (ref,) = rejected.admit(())
    with pytest.raises(ValueError):  # a row selector must return a boolean mask
        run(rejected, step_input(identities, torch.zeros((4, 4)), 0), torch.zeros(4, 7), torch.zeros(4).long())
    assert rejected.arena.latest(ref.lineage_id) == ref
    rejected.retire(())
    for device_name in ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",):
        for request in (False, True):
            _domains_oracle(torch.device(device_name), request, identities)


def _domains_oracle(device, request, identities) -> None:
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    execution = Domains(request).compile_backend(facts_for(device, lineages=4, pages=16)).execution
    initial = execution.admit(identities if request else (), stream)
    original = tuple(execution.snapshot(ref, stream) for ref in initial)
    held = execution.arena.pin_exact(StateRefs.of(initial), stream)
    parameters = [torch.nn.Parameter(root[0].reshape(4, 7).clone()) for root in original]
    optimizers = [torch.optim.AdamW((p,), lr=0.03, weight_decay=0.1, foreach=False) for p in parameters]
    hidden = torch.linspace(-0.5, 0.8, 16, device=device).reshape(4, 4)
    carries = [root[-1].clone() for root in original]
    latest = list(initial)
    patterns = (
        (15,) * 4,
        (11, 5, 7, 8),
        (3,) * 4,
        (5,) * 4,
        (8,) * 4,
        (0,) * 4,
        (8, 11, 5, 8),
        (11, 8, 5, 8),
        (11, 8, 5, 11),
        (15,) * 4,
    )
    for step_id, bits in enumerate(patterns):
        before = tuple(execution.snapshot(ref, stream) for ref in latest)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        source = hidden.clone().requires_grad_(True)
        forward = [row for row, flags in enumerate(bits) if flags & 1]
        # The last step samples token 1, so its update mask is empty.
        no_feedback_update = step_id == len(patterns) - 1
        active = sorted(
            {row if request else 0 for row, flags in enumerate(bits) if flags & 8 and not no_feedback_update}
        )
        expected_output = torch.zeros((4, 7), device=device)
        loss = None
        if forward:
            predictions = []
            for row in forward:
                parameter = parameters[row if request else 0]
                x = source[row] if bits[row] & 2 else source[row].detach()
                predictions.append(x @ (parameter if bits[row] & 4 else parameter.detach()))
                expected_output[row] = predictions[-1].detach()
            loss = torch.stack(predictions).square().mean()
            loss.backward()
        gradients = tuple(torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in parameters)
        for index in active:
            parameters[index].grad = gradients[index]
            optimizers[index].step()
        sampled = tuple(replace(identity, sample_idx=flags) for identity, flags in zip(identities, bits, strict=True))
        value = step_input(sampled, hidden, step_id, stream=stream)
        tokens = torch.full((4,), int(no_feedback_update), device=device, dtype=torch.long)
        result = run(execution, value, torch.zeros((4, 7), device=device), tokens, stream)
        torch.testing.assert_close(result.guidance.values, expected_output)
        if loss is None:
            assert result.loss is None and not result.input_gradients
        else:
            torch.testing.assert_close(result.loss, loss.detach())
            expected_input = torch.zeros_like(source) if source.grad is None else source.grad
            torch.testing.assert_close(result.input_gradients["reweight", "hidden"], expected_input)
            for ref, gradient in zip(latest, gradients, strict=True):
                torch.testing.assert_close(result.parameter_gradients[ref].reshape(4, 7), gradient)
        if active:
            assert tuple(result.publication.before_refs) == tuple(latest[index] for index in active)
            for index, ref in zip(active, result.publication.after_refs, strict=True):
                latest[index] = ref
                captured = [row for row in forward if not request or row == index]
                if captured:
                    carries[index] = expected_output[captured[-1]].detach().clone()
                assert_roots(
                    execution.snapshot(ref, stream), (*adam_roots(optimizers[index], parameters[index]), carries[index])
                )
        else:
            assert result.publication is None
        for index in set(range(len(latest))) - set(active):
            assert_roots(execution.snapshot(latest[index], stream), before[index], exact=True)
    for ref, root in zip(initial, original, strict=True):
        assert_roots(execution.arena.snapshot(held, ref), root, exact=True)
    execution.arena.wait_release(execution.arena.release_pin(held))
    execution.retire(identities if request else (), wait=True)
    assert execution.diagnostics().active_lineages == execution.diagnostics().pending_state_releases == 0

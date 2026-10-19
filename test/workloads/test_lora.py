"""Evidence 43: archived shared weighted CE/SGD, original state and sampler boundaries."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

import pllm
from pllm.authoring import TensorStepInput, WeightedCEBatch
from pllm.config import ExecutionOptions
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import AnchorId, StateScope
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.lora import OutputLoRAConfig, OutputLoRAWorkload
from test.backend_facts import backend_facts

BINDING = PackedHiddenResourceBinding("-1", HiddenLayerKind.INDEX, HiddenSite.MODEL_OUTPUT, "model.norm")


def compile_lora(config, device=torch.device("cpu"), *, lineages=3, pages=3):
    cuda = device.type == "cuda"
    provider = PageBundleProviderFacts(
        "canonical.cuda_dense_page@1" if cuda else "test.page@1",
        pages,
        StateDeviceKind.CUDA if cuda else StateDeviceKind.CPU,
        0,
    )
    facts = backend_facts(
        4, (BINDING,), RuntimeFacts(device).timeline(), provider, max_lineages=lineages, memory_bytes=32 * 1024**2
    )
    workload = OutputLoRAWorkload(config)
    return workload.compile_backend(
        workload.specialize(replace(facts, vocab_size=7), ExecutionOptions(training_rows=4))
    ).execution


def step_input(identities, hidden, auxiliary, readiness=None):
    rows = PackedPhaseRowBatch(
        1,
        0,
        len(identities),
        tuple(PackedPhaseRow(i, i, 0, HiddenPhase.DECODE, identity) for i, identity in enumerate(identities)),
    )
    return TensorStepInput(
        rows,
        PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(len(identities)))),
        PackedHiddenSnapshotBatch(rows, (PackedHiddenSnapshot(BINDING, hidden),)),
        auxiliary=auxiliary,
        hidden_readiness=readiness,
    )


def run_with_guidance(execution, value, stream=None):
    """Run one step and return the sampler guidance it exposed."""
    invocation = execution.start_step(value, stream)
    for anchor in execution.arena.plan.timeline.anchors:
        execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
        if anchor.ordinal == execution.sampler_anchor_id:
            guidance = execution.sampler_guidance(invocation, stream)
            execution.complete_sampler(invocation, stream)
    return guidance, execution.finish_step(invocation)


def test_weighted_ce_shared_update_matches_sgd_and_preserves_original_pins() -> None:
    config = OutputLoRAConfig(rank=2, alpha=3.0, learning_rate=0.3, steps=3, seed=7)
    execution = compile_lora(config)
    (initial,) = execution.admit(())
    original = execution.snapshot(initial)
    old_pin = execution.arena.pin_exact((initial,))
    hidden = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2], [0.1, 0.3, -0.2, 0.8]])
    batch = WeightedCEBatch(
        torch.cat((hidden, torch.full((1, 4), float("nan")))),
        torch.linspace(-1, 1, 28).reshape(4, 7),
        torch.tensor([2, 4, 1, -100]),
        torch.tensor([1.0, 3.0, 0.0, 1.0]),
    )
    value = step_input(tuple(RequestIdentity(str(i), i, 0) for i in range(3)), hidden, batch)
    # Unequal row weights are normalized together, not averaged per request.
    a = torch.nn.Parameter(torch.randn((4, 2), generator=torch.Generator().manual_seed(7)) * 0.01)
    b = torch.nn.Parameter(torch.zeros(2, 7))
    optimizer = torch.optim.SGD((a, b), lr=config.learning_rate)
    for _ in range(config.steps):
        optimizer.zero_grad()
        losses = F.cross_entropy(batch.base_logits[:2] + hidden[:2] @ a @ b * 1.5, batch.labels[:2], reduction="none")
        ((losses[0] + 3 * losses[1]) / 4).backward()
        optimizer.step()
    result = execution.step(value)
    torch.testing.assert_close(result.loss[0], (losses[0] + 3 * losses[1]).detach() / 4)
    assert result.publication.before_refs == (initial,)
    for actual, expected in zip(execution.snapshot(result.publication.after_refs[0]), (a, b), strict=True):
        torch.testing.assert_close(actual, expected)
    # Another batch restarts from the initial factors; no input keeps the head and only guides sampling.
    (latest,) = execution.step(value).publication.after_refs
    for actual, expected in zip(execution.snapshot(latest), (a, b), strict=True):
        torch.testing.assert_close(actual, expected)
    guidance, idle = run_with_guidance(execution, replace(value, auxiliary=None))
    torch.testing.assert_close(guidance.values, hidden @ a @ b * 1.5)
    assert idle.publication is None and idle.loss is None
    with pytest.raises(ValueError):  # At least one row must carry weight.
        execution.step(replace(value, auxiliary=replace(batch, weights=torch.zeros(4))))
    for actual, expected in zip(execution.arena.snapshot(old_pin, initial), original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    execution.arena.wait_release(execution.arena.release_pin(old_pin))
    execution.retire(())
    assert execution.diagnostics().active_lineages == 0


def sgd_reference(config, seed, hidden, logits, labels, weights, groups):
    """Per-adapter weighted CE, averaged over the adapters of the batch, one SGD step per iteration."""
    a, b = torch.nn.Parameter(seed.clone()), torch.nn.Parameter(torch.zeros((2, 7), device=seed.device))
    optimizer = torch.optim.SGD((a, b), lr=config.learning_rate)
    active = (labels >= 0) & (weights > 0)
    for _ in range(config.steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(logits[active] + hidden[active] @ a @ b * 1.5, labels[active], reduction="none")
        ((loss * weights[active]).sum() / weights[active].sum() / groups).backward()
        optimizer.step()
    return a.detach(), b.detach()


def test_independent_lora_bmm_preserves_keyed_mean_and_private_publication() -> None:
    identities = tuple(RequestIdentity(str(index), index, 0) for index in range(3))
    keys = tuple(
        RequestStateKey.project(identity, (RequestKeyField.PROMPT_IDX, RequestKeyField.SAMPLE_IDX))
        for identity in identities
    )
    config = OutputLoRAConfig(rank=2, alpha=3.0, learning_rate=0.2, steps=3, seed=11, scope=StateScope.REQUEST)
    for device in (torch.device("cpu"), *([torch.device("cuda", 0)] if torch.cuda.is_available() else [])):
        stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        execution = compile_lora(config, device, pages=8)
        initial = execution.admit(identities, stream)
        old_pin = execution.arena.pin_exact(initial, stream)
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            original = tuple(execution.arena.snapshot(old_pin, ref) for ref in initial)
            hidden = torch.linspace(-0.7, 0.9, 32, device=device).reshape(2, 4, 4)
            logits = torch.linspace(-1, 1, 56, device=device).reshape(2, 4, 7)
            labels = torch.tensor([[2, 4, -100, 1], [0, 6, 3, 2]], device=device)
            weights = torch.tensor([[1.0, 3.0, 1.0, 0.0], [2.0, 1.0, 0.0, 4.0]], device=device)
            hidden[0, 2].fill_(float("nan"))
            batch = pllm.WeightedCEBatch.for_prompts(hidden, logits, labels, weights, (1, 0))
            assert batch.keys == (keys[1], keys[0])
            readiness = CudaStreamReadiness(stream) if stream is not None else None
            value = step_input(identities, torch.ones((3, 4), device=device), batch, readiness)
            generator = torch.Generator(device=device).manual_seed(config.seed)
            seed = torch.randn((2, 4, 2), device=device, generator=generator) * 0.01
            references = tuple(
                sgd_reference(config, seed[i], hidden[i], logits[i], labels[i], weights[i], 2) for i in range(2)
            )
            publication = execution.step(value, stream).publication
            assert publication.before_refs == (initial[1], initial[0])
            for ref, expected in zip(publication.after_refs, references, strict=True):
                for actual, target in zip(execution.snapshot(ref, stream), expected, strict=True):
                    torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-7)
            assert execution.arena.latest(initial[2].lineage_id) == initial[2]
            # Each row is guided by its own adapter; the untrained request keeps its initial factors.
            guidance, _ = run_with_guidance(execution, replace(value, auxiliary=None), stream)
            expected = torch.cat(
                [
                    torch.ones(1, 4, device=device) @ factors[0] @ factors[1] * 1.5
                    for factors in (references[1], references[0], original[2])
                ]
            )
            torch.testing.assert_close(guidance.values, expected, rtol=1e-5, atol=1e-7)
            for ref, expected in zip(initial, original, strict=True):
                for actual, target in zip(execution.arena.snapshot(old_pin, ref), expected, strict=True):
                    torch.testing.assert_close(actual, target, rtol=0, atol=0)
        execution.arena.wait_release(execution.arena.release_pin(old_pin))
        execution.retire(identities)
        assert execution.diagnostics().active_lineages == 0
    route_keyed_lineages_until_dormant_retirement()


def route_keyed_lineages_until_dormant_retirement() -> None:
    """Same-route requests share one lineage; it persists dormant until dormant retirement."""
    config = OutputLoRAConfig(
        rank=2,
        alpha=3.0,
        learning_rate=0.2,
        steps=1,
        seed=11,
        scope=StateScope.REQUEST,
        key_fields=(RequestKeyField.ROUTE,),
    )
    execution = compile_lora(config, lineages=4, pages=8)
    math_a, math_b = RequestIdentity("r0", 0, 0, "math"), RequestIdentity("r1", 1, 0, "math")
    code = RequestIdentity("r2", 2, 0, "code")
    with pytest.raises(ValueError):  # Route-keyed state needs a route on every request.
        execution.admit((RequestIdentity("r9", 9, 0),))
    (first,) = execution.admit((math_a,))
    joined_math, joined_code = execution.admit((math_b, code))
    assert joined_math.lineage_id == first.lineage_id != joined_code.lineage_id
    math_initial, code_initial = execution.snapshot(first), execution.snapshot(joined_code)
    hidden = torch.linspace(-0.4, 0.6, 8).reshape(1, 2, 4)
    logits = torch.linspace(-1, 1, 14).reshape(1, 2, 7)
    batch = pllm.WeightedCEBatch.for_prompts(hidden, logits, torch.tensor([[1, 3]]), torch.tensor([[1.0, 2.0]]), (1,))
    publication = execution.step(step_input((math_b, code), torch.ones((2, 4)), batch)).publication
    latest_math = execution.arena.latest(first.lineage_id)
    assert publication.after_refs == (latest_math,)
    # One request's update is visible to its whole route; the other route is untouched.
    for actual, initial in zip(execution.snapshot(latest_math), math_initial, strict=True):
        assert not torch.equal(actual, initial)
    for actual, initial in zip(execution.snapshot(joined_code), code_initial, strict=True):
        torch.testing.assert_close(actual, initial)
    execution.retire((math_a, math_b, code))
    assert execution.diagnostics().active_lineages == 2, "dormant route lineages persist"
    execution.retire_dormant()
    assert execution.diagnostics().active_lineages == 0
    # A later admission of the same route starts a fresh initialized lineage.
    (fresh,) = execution.admit((RequestIdentity("r3", 3, 0, "math"),))
    assert fresh.version_id == 0 and fresh.lineage_id != first.lineage_id
    for actual, initial in zip(execution.snapshot(fresh), math_initial, strict=True):
        torch.testing.assert_close(actual, initial)

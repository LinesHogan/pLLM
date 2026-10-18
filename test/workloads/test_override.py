"""Evidence 28: independent iterative-head/Adam oracle over real sampler feedback."""

from contextlib import nullcontext
from dataclasses import replace
from typing import Annotated

import pytest
import torch
from torch.nn import functional as F

import pllm
from pllm import authoring as a
from pllm.authoring import BackendCompileFacts, TensorStepInput
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentTopology,
    BackendPlacementProbe,
)
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import SamplerFeedback, SamplerStage
from pllm.contracts.semantic_graph import AnchorId, SemanticEffect, StateEffectKind, TaskSymbol
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.scheduler import compile_schedule, validate_schedule
from pllm.workloads.override import OverRIDEConfig, OverRIDEWorkload, cross_entropy
from test.backend_facts import assert_finite_resources
from test.revalidation import validate_backend_attachment


def test_override_matches_reweight_adam_and_final_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = PackedHiddenResourceBinding("-1", HiddenLayerKind.INDEX, HiddenSite.MODEL_OUTPUT, "model.norm")
    device = torch.device("cpu")
    facts = BackendCompileFacts(
        BackendPlacementProbe(
            "test.sampler@1",
            "a" * 64,
            "b" * 64,
            "c" * 64,
            AttachmentTopology.IN_PROCESS,
            AttachmentGraphMode.EAGER,
            StateDeviceKind.CPU,
            0,
            1,
            1,
            TensorDType.FLOAT32,
            4,
            8,
            (AttachmentHiddenResource(binding, HiddenPhase.DECODE),),
            (),
        ),
        RuntimeFacts(device).timeline(),
        (PageBundleProviderFacts("test.page@1", 3),),
        3,
        65536,
        7,
        vocab_size=7,
    )
    config = OverRIDEConfig(rank=2, num_iterations=3, learning_rate=0.03)
    assert_finite_resources(OverRIDEWorkload(config), facts)
    observed_versions = []

    class ReadPublished(OverRIDEWorkload):
        def _build(self, b):
            super()._build(b)

            def observe(data, batch):
                observed_versions.append((batch.after_refs, batch.roots[0][0].clone()))
                return a.TaskOutput()

            b.compute(
                "read_published",
                inputs={},
                function=observe,
                when=a.When.AFTER_SAMPLE,
                reads=(SemanticEffect(b.graph.states[0].resource, StateEffectKind.READ, TaskSymbol("train")),),
            )

    deployment = ReadPublished(config).compile_backend(facts)
    execution, signed = deployment.execution, deployment.attachment
    graph = execution.arena.plan.semantic_graph
    training = next(task for task in graph.tasks if task.symbol == "train")
    policy = training.math
    assert policy is not None and policy.reduction == "mean"
    for invalid in (
        replace(policy, detached_inputs=("missing",)),
        replace(policy, input_gradients=("hidden",)),
        replace(policy, update_states="missing"),
    ):
        with pytest.raises(ValueError, match="mathematical contract"):
            compile_schedule(
                replace(
                    graph,
                    tasks=tuple(replace(task, math=invalid) if task is training else task for task in graph.tasks),
                ),
                facts.timeline,
                state=execution.state_plan,
            )
    with pytest.raises(ValueError):
        validate_schedule(
            replace(
                graph,
                tasks=tuple(
                    replace(task, math=replace(policy, reduction="sum")) if task is training else task
                    for task in graph.tasks
                ),
            ),
            facts.timeline,
            signed.inputs.plan.schedule_decision,
            state=execution.state_plan,
        )
    (initial,) = execution.admit(())
    original = execution.snapshot(initial)
    down = torch.nn.Parameter(original[0][:16].reshape(2, 2, 4).clone())
    up = torch.nn.Parameter(original[0][16:].reshape(2, 7, 2).clone())
    optimizer = torch.optim.Adam((down, up), lr=config.learning_rate, foreach=False)
    old_pin = execution.arena.pin_exact((initial,))
    hidden = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2], [0.1, 0.3, -0.2, 0.8]])
    base = torch.linspace(-1, 1, 21).reshape(3, 7)
    tokens = torch.tensor([2, 4, 1])

    def step(indices: tuple[int, ...], step_id: int) -> TensorStepInput:
        rows = PackedPhaseRowBatch(
            1,
            step_id,
            len(indices),
            tuple(
                PackedPhaseRow(
                    i,
                    i,
                    step_id,
                    HiddenPhase.PREFILL if step_id == 1 and i == 0 else HiddenPhase.DECODE,
                    RequestIdentity(str(i), i, sample),
                )
                for i, sample in enumerate(indices)
            ),
        )
        order = (2, 0, 1) if step_id == 1 else tuple(range(len(indices)))
        sampler_rows = PackedSamplerRowBatch(
            rows, tuple(PackedSamplerRowLink(i, packed) for i, packed in enumerate(order))
        )
        return TensorStepInput(
            rows,
            sampler_rows,
            PackedHiddenSnapshotBatch(rows, (PackedHiddenSnapshot(binding, hidden[: len(indices)]),)),
        )

    latest = initial
    for step_id, indices in enumerate(((0, 1, 2), (1, 0, 2), (2, 2, 2))):
        value = step(indices, step_id)
        before = execution.snapshot(latest)
        invocation = execution.start_step(value)
        for anchor in range(4):
            execution.execute_anchor(invocation, AnchorId(anchor))
        guidance = execution.sampler_guidance(invocation)
        decode = tuple(
            link
            for link in value.sampler_rows.links
            if value.rows.rows[link.packed_row_index].phase is HiddenPhase.DECODE
        )
        expected_delta = torch.stack(
            [
                torch.zeros(7)
                if indices[link.packed_row_index] == 0
                else config.lambd
                * (
                    hidden[link.packed_row_index]
                    @ down[indices[link.packed_row_index] - 1].T
                    @ up[indices[link.packed_row_index] - 1].T
                )
                for link in decode
            ]
        ).detach()
        torch.testing.assert_close(guidance.values, expected_delta)
        assert guidance.affected_row_ids == tuple(link.sampler_row_index for link in decode)
        assert guidance.stage is SamplerStage.PACKED_PRE_FILTER_ADD
        with pytest.raises(RuntimeError, match="sampler completion"):
            execution.execute_anchor(invocation, AnchorId(5))
        with pytest.raises(ValueError, match="original"):
            execution.complete_sampler(invocation, feedback=SamplerFeedback(replace(value.sampler_rows), base, tokens))
        execution.complete_sampler(invocation, feedback=SamplerFeedback(value.sampler_rows, base, tokens))
        for anchor in range(4, 8):
            execution.execute_anchor(invocation, AnchorId(anchor))
        result = execution.finish_step(invocation)
        assert invocation.context.guidance is None and invocation.context.sampler_feedback is None
        selected = [link for link in decode if indices[link.packed_row_index] < 2]
        if selected:
            optimizer.zero_grad()
            predicted = torch.stack(
                [
                    base[link.sampler_row_index]
                    + hidden[link.packed_row_index]
                    @ down[indices[link.packed_row_index]].T
                    @ up[indices[link.packed_row_index]].T
                    for link in selected
                ]
            )
            loss = F.cross_entropy(predicted, tokens[[link.sampler_row_index for link in selected]])
            loss.backward()
            optimizer.step()
            assert result.publication is not None and result.publication.before_refs == (latest,)
            (latest,) = result.publication.after_refs
            torch.testing.assert_close(result.loss, loss.detach())
            root = execution.snapshot(latest)
            for actual, expected in zip(
                root,
                (
                    torch.cat((down.flatten(), up.flatten())),
                    torch.cat((down.grad.flatten(), up.grad.flatten())),
                    optimizer.state[down]["step"],
                    torch.cat((optimizer.state[down]["exp_avg"].flatten(), optimizer.state[up]["exp_avg"].flatten())),
                    torch.cat(
                        (optimizer.state[down]["exp_avg_sq"].flatten(), optimizer.state[up]["exp_avg_sq"].flatten())
                    ),
                ),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected)
        else:
            assert result.loss is result.publication is None
            assert torch.count_nonzero(result.guidance.values) > 0
            for actual, expected in zip(execution.snapshot(latest), before, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        refs, observed = observed_versions[-1]
        assert refs == (latest,)
        torch.testing.assert_close(observed, execution.snapshot(latest)[0])
    for actual, expected in zip(execution.arena.snapshot(old_pin, initial), original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    execution.arena.wait_release(execution.arena.release_pin(old_pin))

    def fail_publish(self, write):
        raise RuntimeError("publication failed")

    with monkeypatch.context() as fault:
        fault.setattr(PageBundleArena, "publish", fail_publish)
        value = step((0, 1, 2), 4)
        invocation = execution.start_step(value)
        for anchor in range(4):
            execution.execute_anchor(invocation, AnchorId(anchor))
        execution.complete_sampler(invocation, feedback=SamplerFeedback(value.sampler_rows, base, tokens))
        with pytest.raises(RuntimeError, match="publication failed"):
            execution.execute_anchor(invocation, AnchorId(5))
    assert execution.arena.latest(initial.lineage_id) == latest
    with pytest.raises(ValueError):
        validate_backend_attachment(
            replace(
                signed.inputs,
                request=replace(signed.inputs.request, sampler_stage=SamplerStage.PACKED_POST_FILTER_EXACT),
            ),
            signed.decision,
        )
    with pytest.raises(ValueError, match="iterations"):
        execution.start_step(step((3,), 5))
    execution.retire(())
    assert execution.diagnostics().loss_count == execution.diagnostics().update_count == 2
    assert execution.diagnostics().active_lineages == execution.diagnostics().fallback_count == 0

    single = OverRIDEWorkload(replace(config, num_iterations=1)).compile_backend(facts).execution
    (ref,) = single.admit(())
    assert single.snapshot(ref)[0].numel() == 0
    value = step((0,), 6)
    invocation = single.start_step(value)
    for anchor in range(4):
        single.execute_anchor(invocation, AnchorId(anchor))
    single.complete_sampler(invocation, feedback=SamplerFeedback(value.sampler_rows, base[:1], tokens[:1]))
    for anchor in range(4, 8):
        single.execute_anchor(invocation, AnchorId(anchor))
    result = single.finish_step(invocation)
    assert torch.count_nonzero(result.guidance.values) == 0 and result.publication is result.loss is None
    single.retire(())
    assert single.diagnostics().update_count == single.diagnostics().active_lineages == 0

    def zero_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), labels.long()) * 0

    original_workload = OverRIDEWorkload(config)
    variant = replace(original_workload, objective=zero_loss)
    assert original_workload.objective is cross_entropy
    with pytest.raises(ValueError, match="signed declaration"):
        variant.bind_backend(facts, original_workload.plan_backend(facts))
    modified = variant.compile_backend(facts).execution
    with pytest.raises(ValueError, match="resource_facts"):
        variant.specialize(facts, pllm.ExecutionOptions(memory_budget_bytes=1 << 30))
    (ref,) = modified.admit(())
    value = step((0, 1, 2), 7)
    invocation = modified.start_step(value)
    for anchor in range(4):
        modified.execute_anchor(invocation, AnchorId(anchor))
    modified.complete_sampler(invocation, feedback=SamplerFeedback(value.sampler_rows, base, tokens))
    for anchor in range(4, 8):
        modified.execute_anchor(invocation, AnchorId(anchor))
    result = modified.finish_step(invocation)
    assert result.loss.item() == 0 and result.publication is not None
    assert modified.diagnostics().update_count == 1
    modified.retire(())

    class RequestHeads(pllm.Workload):
        name = "request_heads"
        heads = pllm.RequestState({"weight": (4, 7)}, optimizer=a.AdamW(lr=0.03))

        @pllm.on(pllm.Phase.BEFORE_SAMPLE)
        @pllm.add_logits
        def reweight(
            self,
            hidden: Annotated[torch.Tensor, pllm.Hidden(-1, HiddenSite.MODEL_OUTPUT)],
            samples: Annotated[torch.Tensor, pllm.Samples()],
            weight: Annotated[torch.Tensor, pllm.Parameter("heads", "weight")],
        ) -> torch.Tensor:
            return torch.bmm(hidden.unsqueeze(1), weight).squeeze(1)

        def eligible(self, samples: Annotated[torch.Tensor, pllm.Samples()]) -> torch.Tensor:
            return samples == 0

        @pllm.on(pllm.Phase.AFTER_SAMPLE, rows="eligible")
        @pllm.train(state="heads")
        def train(
            self,
            hidden: Annotated[torch.Tensor, pllm.Hidden(-1, HiddenSite.MODEL_OUTPUT)],
            samples: Annotated[torch.Tensor, pllm.Samples()],
            weight: Annotated[torch.Tensor, pllm.Parameter("heads", "weight")],
            base: Annotated[torch.Tensor, pllm.BaseLogits()],
            tokens: Annotated[torch.Tensor, pllm.SampledTokens()],
        ) -> torch.Tensor:
            return F.cross_entropy(base + torch.bmm(hidden.unsqueeze(1), weight).squeeze(1), tokens)

    requests = (
        RequestHeads().compile_backend(replace(facts, providers=(PageBundleProviderFacts("test.page@1", 9),))).execution
    )
    value = step((0, 2, 0), 8)
    identities = tuple(row.identity for row in value.rows.rows)
    initial = requests.admit(identities)
    original = tuple(requests.snapshot(ref) for ref in initial)
    invocation = requests.start_step(value)
    for anchor in range(4):
        requests.execute_anchor(invocation, AnchorId(anchor))
    requests.complete_sampler(invocation, feedback=SamplerFeedback(value.sampler_rows, base, tokens))
    for anchor in range(4, 8):
        requests.execute_anchor(invocation, AnchorId(anchor))
    result = requests.finish_step(invocation)
    assert result.publication.before_refs == (initial[0], initial[2])
    assert requests.arena.latest(initial[1].lineage_id) == initial[1]
    for actual, expected in zip(requests.snapshot(initial[1]), original[1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.nn.Parameter(torch.zeros((2, 4, 7)))
    optimizer = torch.optim.Adam((weights,), lr=0.03, foreach=False)
    loss = F.cross_entropy(base[[0, 2]] + torch.bmm(hidden[[0, 2]].unsqueeze(1), weights).squeeze(1), tokens[[0, 2]])
    loss.backward()
    optimizer.step()
    torch.testing.assert_close(result.loss, loss.detach())
    for index, ref in enumerate(result.publication.after_refs):
        torch.testing.assert_close(requests.snapshot(ref)[0].reshape(4, 7), weights[index])
    requests.retire(identities)
    assert requests.diagnostics().update_count == 2
    assert requests.diagnostics().active_lineages == requests.diagnostics().fallback_count == 0

    def initialize_chain(parameters, generator):
        parameters["weight"].copy_(torch.linspace(-0.2, 0.3, 28).reshape(4, 7))

    class ChainedHeads(pllm.Workload):
        heads = pllm.InstanceState({"weight": (4, 7)}, optimizer=pllm.AdamW(lr=0.03), initialize=initialize_chain)

        @pllm.on(pllm.Phase.BEFORE_SAMPLE)
        @pllm.add_logits
        @pllm.differentiate(inputs=("hidden",), parameters=True)
        def reweight(
            self,
            hidden: Annotated[torch.Tensor, pllm.Hidden(-1, HiddenSite.MODEL_OUTPUT)],
            samples: Annotated[torch.Tensor, pllm.Samples()],
            weight: Annotated[torch.Tensor, pllm.Parameter("heads", "weight")],
        ) -> torch.Tensor:
            return hidden @ weight

        @pllm.on(pllm.Phase.AFTER_SAMPLE)
        @pllm.differentiate(inputs=("features",))
        def frozen(
            self,
            features: Annotated[torch.Tensor, pllm.Result("reweight")],
            weight: Annotated[torch.Tensor, pllm.Parameter("heads", "weight")],
        ) -> torch.Tensor:
            return features @ weight.T @ weight

        def eligible(self, samples: Annotated[torch.Tensor, pllm.Samples()]) -> torch.Tensor:
            return samples == 0

        @pllm.on(pllm.Phase.AFTER_SAMPLE, rows="eligible")
        @pllm.train(state="heads")
        @pllm.differentiate(inputs=("features",))
        def train(
            self,
            features: Annotated[torch.Tensor, pllm.Result("frozen")],
            samples: Annotated[torch.Tensor, pllm.Samples()],
            base: Annotated[torch.Tensor, pllm.BaseLogits()],
            tokens: Annotated[torch.Tensor, pllm.SampledTokens()],
        ) -> torch.Tensor:
            return F.cross_entropy(base + features, tokens)

    chain = ChainedHeads().compile_backend(facts).execution
    (initial,) = chain.admit(())
    weight = torch.nn.Parameter(chain.snapshot(initial)[0].reshape(4, 7).clone())
    oracle_input = hidden.clone().requires_grad_(True)
    optimizer = torch.optim.Adam((weight,), lr=0.03, foreach=False)
    predicted = oracle_input @ weight
    frozen = predicted @ weight.detach().T @ weight.detach()
    loss = F.cross_entropy(base[[0, 2]] + frozen[[0, 2]], tokens[[0, 2]])
    loss.backward()
    optimizer.step()
    value = step((0, 2, 0), 9)
    invocation = chain.start_step(value)
    for anchor in range(4):
        chain.execute_anchor(invocation, AnchorId(anchor))
    chain.complete_sampler(invocation, feedback=SamplerFeedback(value.sampler_rows, base, tokens))
    for anchor in range(4, 8):
        chain.execute_anchor(invocation, AnchorId(anchor))
    assert set(invocation.context.results) == {(TaskSymbol("reweight"), "value"), (TaskSymbol("train"), "value")}
    assert all(value.grad_fn is None for value in invocation.context.results.values())
    assert not invocation.context.data().derivative_sources and not invocation.context.data().parameters
    result = chain.finish_step(invocation)
    torch.testing.assert_close(result.guidance.values, predicted.detach())
    torch.testing.assert_close(result.loss, loss.detach())
    torch.testing.assert_close(result.input_gradients["reweight", "hidden"], oracle_input.grad)
    assert torch.count_nonzero(result.input_gradients["reweight", "hidden"][1]) == 0
    torch.testing.assert_close(chain.snapshot(result.publication.after_refs[0])[0].reshape(4, 7), weight)
    assert invocation.context.value is None
    latest = result.publication.after_refs[0]
    before = chain.snapshot(latest)
    cancelled = chain.start_step(step((0, 2, 0), 10))
    for anchor in range(4):
        chain.execute_anchor(cancelled, AnchorId(anchor))
    assert cancelled.context.results[TaskSymbol("reweight"), "value"].grad_fn is not None
    chain.cancel_step(cancelled)
    assert cancelled.context.value is None and not cancelled.context.results
    assert chain.arena.latest(latest.lineage_id) == latest
    skipped_value = step((2, 2, 2), 11)
    skipped = chain.start_step(skipped_value)
    for anchor in range(4):
        chain.execute_anchor(skipped, AnchorId(anchor))
    chain.complete_sampler(skipped, feedback=SamplerFeedback(skipped_value.sampler_rows, base, tokens))
    for anchor in range(4, 8):
        chain.execute_anchor(skipped, AnchorId(anchor))
    assert all(value.grad_fn is None for value in skipped.context.results.values())
    assert not skipped.context.data().derivative_sources and not skipped.context.data().parameters
    result = chain.finish_step(skipped)
    assert result.publication is result.loss is None and not result.input_gradients
    assert skipped.context.value is None and chain.diagnostics().update_count == 1
    for actual, expected in zip(chain.snapshot(latest), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    chain.retire(())

    _independent_domains(facts, binding, monkeypatch)


def _independent_domains(facts, binding, monkeypatch) -> None:
    def initialize(parameters, generator):
        parameters["weight"].copy_(torch.linspace(-0.2, 0.3, 28).reshape(4, 7))

    class Domains(pllm.Workload):
        def __init__(self, request):
            state = pllm.RequestState if request else pllm.InstanceState
            self.heads = state(
                {"weight": (4, 7)}, optimizer=pllm.AdamW(lr=0.03, weight_decay=0.1), initialize=initialize
            )

        def forward_rows(self, samples: Annotated[torch.Tensor, pllm.Samples()]) -> torch.Tensor:
            return samples.bitwise_and(1) != 0

        def input_rows(self, samples: Annotated[torch.Tensor, pllm.Samples()]) -> torch.Tensor:
            return samples.bitwise_and(2) != 0

        def parameter_rows(self, samples: Annotated[torch.Tensor, pllm.Samples()]) -> torch.Tensor:
            return samples.bitwise_and(4) != 0

        def updates(
            self,
            samples: Annotated[torch.Tensor, pllm.Samples()],
            tokens: Annotated[torch.Tensor, pllm.SampledTokens()],
        ) -> torch.Tensor:
            return (samples.bitwise_and(8) != 0) & (tokens == 0)

        @pllm.on(pllm.Phase.BEFORE_SAMPLE, rows="forward_rows")
        @pllm.add_logits
        @pllm.differentiate(
            inputs=("hidden",), parameters=True, input_rows="input_rows", parameter_rows="parameter_rows"
        )
        def reweight(
            self,
            hidden: Annotated[torch.Tensor, pllm.Hidden(-1, HiddenSite.MODEL_OUTPUT)],
            samples: Annotated[torch.Tensor, pllm.Samples()],
            weight: Annotated[torch.Tensor, pllm.Parameter("heads", "weight", per_row=True)],
        ) -> torch.Tensor:
            return torch.bmm(hidden.unsqueeze(1), weight).squeeze(1)

        @pllm.on(pllm.Phase.AFTER_SAMPLE, rows="forward_rows")
        @pllm.train(state="heads", update="updates")
        @pllm.differentiate(inputs=("features",))
        def train(
            self,
            features: Annotated[torch.Tensor, pllm.Result("reweight")],
            samples: Annotated[torch.Tensor, pllm.Samples()],
            tokens: Annotated[torch.Tensor, pllm.SampledTokens()],
        ) -> torch.Tensor:
            return features.square().mean()

    class EarlyFeedback(Domains):
        @pllm.on(pllm.Phase.BEFORE_SAMPLE)
        def early(self, tokens: Annotated[torch.Tensor, pllm.SampledTokens()]) -> torch.Tensor:
            return tokens == 0

    class LateFeedback(Domains):
        @pllm.on(pllm.Phase.AFTER_SAMPLE)
        def late(
            self,
            loss: Annotated[torch.Tensor, pllm.Result("train")],
            tokens: Annotated[torch.Tensor, pllm.SampledTokens()],
        ) -> torch.Tensor:
            return tokens + loss

    with pytest.raises(ValueError, match="sampler feedback"):
        EarlyFeedback(False).compile_backend(facts)
    with pytest.raises(ValueError, match="sampler feedback"):
        LateFeedback(False).compile_backend(facts)
    with monkeypatch.context() as invalid:
        invalid.setattr(
            Domains.reweight,
            "__annotations__",
            dict(Domains.reweight.__annotations__, weight=Annotated[torch.Tensor, pllm.Parameter("heads", "weight")]),
        )
        with pytest.raises(ValueError, match="per_row=True"):
            Domains(False).compile_backend(facts)

    for device_name in ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",):
        device = torch.device(device_name)
        stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        for request in (False, True):
            selected_facts = replace(
                facts,
                probe=replace(
                    facts.probe, device_kind=StateDeviceKind.CUDA if device.type == "cuda" else StateDeviceKind.CPU
                ),
                timeline=RuntimeFacts(device).timeline(),
                max_admitted_lineages=4,
                providers=(
                    PageBundleProviderFacts(
                        "test.page@1", 16, StateDeviceKind.CUDA if device.type == "cuda" else StateDeviceKind.CPU
                    ),
                ),
            )
            workload = Domains(request)
            execution = workload.compile_backend(selected_facts).execution
            graph = execution.arena.plan.semantic_graph
            prediction = next(task for task in graph.tasks if task.symbol == "reweight")
            training = next(task for task in graph.tasks if task.symbol == "train")
            assert (
                prediction.math.forward_rows,
                prediction.math.input_gradient_rows,
                prediction.math.parameter_gradient_rows,
                training.math.update_states,
            ) == ("__forward_rows", "__input_gradient_rows", "__parameter_gradient_rows", "__update_states")
            identities = tuple(RequestIdentity(str(i), i, 0) for i in range(4))
            initial = execution.admit(identities if request else (), stream)
            original = tuple(execution.snapshot(ref, stream) for ref in initial)
            held = execution.arena.pin_exact(initial, stream)
            parameters = [torch.nn.Parameter(root[0].reshape(4, 7).clone()) for root in original]
            optimizers = [
                torch.optim.AdamW((parameter,), lr=0.03, weight_decay=0.1, foreach=False) for parameter in parameters
            ]
            hidden = torch.linspace(-0.5, 0.8, 16, device=device).reshape(4, 4)
            latest = list(initial)
            loss_count = update_count = 0
            patterns = (
                (15, 15, 15, 15),
                (11, 5, 7, 8),
                (3, 3, 3, 3),
                (5, 5, 5, 5),
                (8, 8, 8, 8),
                (0, 0, 0, 0),
                (8, 11, 5, 8),
                (11, 8, 5, 8),
                (15, 15, 15, 15),
            )
            for step_id, bits in enumerate(patterns):
                before = tuple(execution.snapshot(ref, stream) for ref in latest)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                source = hidden.clone().requires_grad_(True)
                forward = [row for row, flags in enumerate(bits) if flags & 1]
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
                        prediction = (source[row] if bits[row] & 2 else source[row].detach()) @ (
                            parameter if bits[row] & 4 else parameter.detach()
                        )
                        predictions.append(prediction)
                        expected_output[row] = prediction.detach()
                    loss = torch.stack(predictions).square().mean()
                    loss.backward()
                expected_gradients = tuple(
                    torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.clone()
                    for parameter in parameters
                )
                expected_input = torch.zeros_like(source) if source.grad is None else source.grad
                for index in active:
                    parameters[index].grad = expected_gradients[index]
                    optimizers[index].step()
                rows = PackedPhaseRowBatch(
                    1,
                    step_id,
                    4,
                    tuple(
                        PackedPhaseRow(i, i, step_id, HiddenPhase.DECODE, replace(identities[i], sample_idx=flags))
                        for i, flags in enumerate(bits)
                    ),
                )
                value = TensorStepInput(
                    rows,
                    PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(4))),
                    PackedHiddenSnapshotBatch(rows, (PackedHiddenSnapshot(binding, hidden),)),
                    hidden_readiness=None if stream is None else CudaStreamReadiness(stream),
                )
                invocation = execution.start_step(value, stream)
                saved_weights = []
                saved_hidden = []

                def save(tensor):
                    if tensor.shape == (len(forward), 4, 7):
                        saved_weights.append((tensor.stride(), tensor.untyped_storage().data_ptr()))
                    if tensor.shape == (len(forward), 1, 4):
                        saved_hidden.append((tensor.untyped_storage().data_ptr(), tensor.storage_offset()))
                    return tensor.detach()

                with (
                    torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True)
                    if device.type == "cpu" and request and step_id == 1
                    else nullcontext() as packing
                ):
                    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
                        for anchor in range(4):
                            execution.execute_anchor(invocation, AnchorId(anchor))
                if forward and not request:
                    parameter = invocation.context.value.leaf
                    assert saved_weights and all(
                        stride[0] == 0 and pointer == parameter.untyped_storage().data_ptr()
                        for stride, pointer in saved_weights
                    )
                if forward and forward == list(range(forward[0], forward[0] + len(forward))):
                    assert saved_hidden and all(
                        pointer == hidden.untyped_storage().data_ptr() and offset == forward[0] * hidden.stride(0)
                        for pointer, offset in saved_hidden
                    )
                elif forward:
                    assert saved_hidden and all(
                        pointer != hidden.untyped_storage().data_ptr() for pointer, _ in saved_hidden
                    )
                if packing is not None:
                    assert (
                        sum(
                            event.cpu_memory_usage
                            for event in packing.key_averages()
                            if event.key == "aten::index_select"
                        )
                        == (len(parameters) + len(forward))
                        * 4
                        * 7
                        * hidden.element_size()  # [states, field] leaf + packed rows
                    )
                with pytest.raises(RuntimeError, match="sampler completion"):
                    execution.execute_anchor(invocation, AnchorId(5))
                execution.complete_sampler(
                    invocation,
                    stream,
                    feedback=SamplerFeedback(
                        value.sampler_rows,
                        torch.zeros((4, 7), device=device),
                        torch.full((4,), int(no_feedback_update), device=device, dtype=torch.long),
                    ),
                )
                for anchor in range(4, 8):
                    execution.execute_anchor(invocation, AnchorId(anchor))
                result = execution.finish_step(invocation)
                assert invocation.context.value is None
                torch.testing.assert_close(result.guidance.values, expected_output)
                if loss is None:
                    assert result.loss is None and not result.input_gradients
                else:
                    loss_count += 1
                    torch.testing.assert_close(result.loss, loss.detach())
                    torch.testing.assert_close(result.input_gradients["reweight", "hidden"], expected_input)
                    for ref, gradient in zip(latest, expected_gradients, strict=True):
                        torch.testing.assert_close(result.parameter_gradients[ref].reshape(4, 7), gradient)
                if active:
                    assert result.publication.before_refs == tuple(latest[index] for index in active)
                    for index, ref in zip(active, result.publication.after_refs, strict=True):
                        latest[index] = ref
                        state = optimizers[index].state[parameters[index]]
                        for actual, expected in zip(
                            execution.snapshot(ref, stream),
                            (
                                parameters[index].flatten(),
                                expected_gradients[index].flatten(),
                                state["step"].to(device),
                                state["exp_avg"].flatten(),
                                state["exp_avg_sq"].flatten(),
                            ),
                            strict=True,
                        ):
                            torch.testing.assert_close(actual, expected)
                    update_count += len(active)
                else:
                    assert result.publication is None
                for index in set(range(len(latest))) - set(active):
                    for actual, expected in zip(execution.snapshot(latest[index], stream), before[index], strict=True):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for ref, root in zip(initial, original, strict=True):
                for actual, expected in zip(execution.arena.snapshot(held, ref), root, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            execution.arena.wait_release(execution.arena.release_pin(held))
            assert execution.diagnostics().loss_count == loss_count
            assert execution.diagnostics().update_count == update_count

            def invalid_rows(samples: Annotated[torch.Tensor, pllm.Samples()]) -> torch.Tensor:
                return samples

            with monkeypatch.context() as invalid:
                invalid.setattr(workload, "forward_rows", invalid_rows)
                rejected = workload.compile_backend(selected_facts).execution
                refs = rejected.admit(identities if request else (), stream)
                with pytest.raises(ValueError, match="boolean mask"):
                    rejected.step(value, stream)
                assert all(rejected.arena.latest(ref.lineage_id) == ref for ref in refs)
                rejected.retire(identities if request else ())
                assert rejected.diagnostics().active_lineages == 0
            execution.retire(identities if request else ())
            assert execution.diagnostics().active_lineages == execution.diagnostics().pending_state_releases == 0

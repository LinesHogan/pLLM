"""Evidence 25: independent gated-residual/AdamW oracle and exact request views."""

from contextlib import nullcontext
from dataclasses import replace
from typing import Annotated

import pytest
import torch
import torch.nn.functional as F

import pllm
from pllm.authoring import FunctionMath, TensorResult, TensorStepInput
from pllm.config import ExecutionOptions
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from pllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
)
from pllm.contracts.packed_rows import (
    PackedPhaseRow,
    PackedPhaseRowBatch,
    PackedSamplerRowBatch,
    PackedSamplerRowLink,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    SchedulerTimeline,
    SemanticOutputKind,
    TimelineAnchor,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePublication, PageWrite
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.state_execution import StateSettlementError, StateTaskContext
from pllm.execution.tensor_frame import TensorFrame
from pllm.workloads.esamp_request import (
    ESampRequestConfig,
    ESampRequestWorkload,
)
from test.backend_facts import assert_capture_preflight, backend_facts


def effective(guidance):
    """Rows the guidance actually changes (nonzero weight) and their predictions."""
    if guidance.row_weights is None:
        return guidance.affected_row_ids, guidance.values
    keep = [index for index, weight in enumerate(guidance.row_weights.tolist()) if weight]
    return tuple(guidance.affected_row_ids[index] for index in keep), guidance.values[keep]


def _reference(source: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    up, gate, down, bias, scale, shift = parameters.split((8, 8, 8, 2, 4, 4))
    residual = (
        (source @ up.reshape(4, 2))
        * F.silu(
            source @ gate.reshape(4, 2) + bias,
        )
    ) @ down.reshape(2, 4)
    return F.layer_norm(source + residual, (4,)) * scale + shift


def test_esamp_predicts_before_training_and_isolates_request_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ESampRequestConfig(intervention_beta=value)
        with pytest.raises(ValueError):
            ESampRequestConfig(learning_rate=value)
    bindings = tuple(
        PackedHiddenResourceBinding(
            str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]"
        )
        for layer in (0, -1)
    )
    config = ESampRequestConfig(
        rank=2,
        intervention=True,
        intervention_beta=0.2,
        learning_rate=0.03,
        weight_decay=0.1,
        key_fields=(RequestKeyField.REQUEST_ID, RequestKeyField.PROMPT_IDX, RequestKeyField.SAMPLE_IDX),
    )
    timeline = SchedulerTimeline(
        tuple(
            TimelineAnchor(AnchorSymbol(name), index)
            for index, name in enumerate(
                (
                    "model.execute",
                    "sampler.pre_draw",
                    "token.commit",
                    "next.prepare",
                )
            )
        ),
        ("compute/cpu@1", "state/dense_page_cpu@1"),
    )
    workload = ESampRequestWorkload(config)
    facts = backend_facts(4, bindings, timeline, PageBundleProviderFacts("test.esamp@1", 6))
    bounded = ESampRequestWorkload(replace(config, key_fields=(RequestKeyField.REQUEST_ID,)))
    for paged in (False, True):
        assert_capture_preflight(ESampRequestWorkload(replace(bounded.config, paged=paged)), facts, monkeypatch)
    specialized = bounded.specialize(facts, ExecutionOptions())

    def shifted_source(self, source, up, gate, down, bias, scale, shift):
        return source + shift

    with monkeypatch.context() as changed:
        changed.setattr(ESampRequestWorkload, "forward", shifted_source)
        with pytest.raises(ValueError, match="resource_facts declaration"):
            bounded.specialize(facts, ExecutionOptions())
        with pytest.raises(ValueError, match="resource_facts declaration"):
            bounded.compile_backend(specialized)
        changed.setattr(ESampRequestWorkload, "resource_facts", lambda self, facts, options: facts.timeline)
        assert bounded.compile_backend(specialized).execution.arena.active_lineages == 0
    (state_shape,) = next(item for item in specialized.timeline.task_resources if item.task == "train").input_shapes
    resources = tuple(
        replace(
            item,
            resident_bytes=32,
            input_shapes=(item.input_shapes[0], item.input_shapes[0], state_shape, (item.input_shapes[0][0],)),
        )
        if item.task == "loss"
        else item
        for item in specialized.timeline.task_resources
    )
    # Loss and optimizer share one capture entitlement; signed loss residency cannot fund an unreserved optimizer.
    with pytest.raises(ValueError, match="train workspace/residency"):
        bounded.compile_backend(replace(specialized, timeline=replace(specialized.timeline, task_resources=resources)))
    declaration = workload.declaration(facts)
    plan = compile_page_bundle_plan(
        (declaration.graph, declaration.layouts),
        timeline,
        facts.providers,
        facts.max_admitted_lineages,
        facts.state_memory_budget_bytes,
        facts.device,
    )
    binding = workload.bind_math(facts, plan)
    with pytest.raises(ValueError, match="request-id"):
        workload.declare(facts)
    with pytest.raises(ValueError, match="signed declaration"):
        ESampRequestWorkload(replace(config, learning_rate=0.9)).bind_math(facts, plan)
    with monkeypatch.context() as changed:
        changed.setattr(ESampRequestWorkload, "optimizer", property(lambda self: pllm.AdamW(lr=0.9)))
        with pytest.raises(ValueError, match="signed declaration"):
            workload.bind_math(facts, plan)
    with pytest.raises(ValueError, match="bindings"):
        workload.bind_math(
            replace(facts, probe=replace(facts.probe, hidden_resources=facts.probe.hidden_resources[:1])), plan
        )
    # Frozen dataclasses block plain rebinding; drive the drift behind the freeze to
    # prove an already bound execution keeps its own signed mathematics.
    object.__setattr__(workload, "config", replace(config, intervention_beta=0.9, learning_rate=0.9))
    execution = replace(binding, tasks=tuple(reversed(binding.tasks))).bind(
        PageBundleArena(plan),
        next(AnchorId(item.ordinal) for item in plan.timeline.anchors if item.symbol == "sampler.pre_draw"),
    )
    assert execution.device.type == "cpu"
    identities = (RequestIdentity("shared-id", 0, 0), RequestIdentity("shared-id", 0, 1))
    latest = list(execution.admit(identities))
    parameters = [execution.snapshot(ref)[0].requires_grad_() for ref in latest]
    optimizers = [
        torch.optim.AdamW(
            (parameter,), lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay
        )
        for parameter in parameters
    ]
    generator = torch.Generator().manual_seed(7)
    for parameter in parameters:
        expected = torch.cat(
            (
                *(torch.empty(8).normal_(0.0, 0.02, generator=generator) for _ in range(3)),
                torch.ones(2),
                torch.ones(4),
                torch.zeros(4),
            )
        )
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    source = torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.7, 0.1, -0.2], [0.1, 0.3, -0.2, 0.8]])
    target = torch.tensor([[0.6, 0.0, -0.3, 0.2], [0.4, -0.2, 0.8, 0.1], [-0.2, 0.5, 0.1, 0.4]])
    for step_id, owners in enumerate(((0, 1, 0), (0,))):
        count = len(owners)
        rows = PackedPhaseRowBatch(
            1,
            step_id,
            count,
            tuple(
                PackedPhaseRow(index, index, step_id, HiddenPhase.DECODE, identities[owner])
                for index, owner in enumerate(owners)
            ),
        )
        tensors = {bindings[0]: source[:count], bindings[1]: target[:count]}
        step = TensorStepInput(
            rows,
            PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(count))),
            PackedHiddenSnapshotBatch(
                rows,
                tuple(
                    PackedHiddenSnapshot(resource, tensors[resource])
                    for resource in sorted(bindings, key=lambda item: item.locator_key)
                ),
            ),
        )
        indexed = execution.math.rows(step)
        selected = execution.directory.select(indexed)
        assert tuple(selected.ref_for(identities[owner]) for owner in owners) == tuple(
            latest[owner] for owner in owners
        )
        before = [execution.snapshot(ref) for ref in latest]
        for last_anchor in (0, 1):
            cancelled = execution.start_step(step)
            with pytest.raises(RuntimeError, match="overlaps"):
                execution.start_step(step)
            for anchor in range(last_anchor + 1):
                execution.execute_anchor(cancelled, AnchorId(anchor))
            execution.cancel_step(cancelled)
            assert cancelled.context.value is None and cancelled.context.loss is None
            assert cancelled.context.guidance is None and cancelled.context.sampler_feedback is None
            with pytest.raises(RuntimeError, match="active"):
                execution.cancel_step(cancelled)

        def reject_publication(
            self: PageBundleArena,
            batch: PageWrite,
        ) -> PagePublication:
            raise RuntimeError("injected publication failure")

        with monkeypatch.context() as fault:
            fault.setattr(PageBundleArena, "publish", reject_publication)
            failed = execution.start_step(step)
            execution.execute_anchor(failed, AnchorId(0))
            execution.execute_anchor(failed, AnchorId(1))
            execution.complete_sampler(failed)
            with pytest.raises(RuntimeError, match="injected publication failure"):
                execution.execute_anchor(failed, AnchorId(2))
        assert execution.arena.free_pages == 4
        for ref, root in zip(latest, before, strict=True):
            for actual, expected in zip(execution.snapshot(ref), root, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        handle = execution.start_step(step)
        execution.execute_anchor(handle, AnchorId(0))
        guidance = execution.sampler_guidance(handle)
        expected_prediction = torch.cat(
            tuple(_reference(source[i : i + 1], parameters[owner]).detach() for i, owner in enumerate(owners))
        )
        rows, values = effective(guidance)
        if step_id == 0:
            assert rows == ()
        else:
            assert rows == (0,)
            torch.testing.assert_close(values, expected_prediction)
        # No update or version change is allowed before sampler completion.
        for ref, root in zip(latest, before, strict=True):
            for actual, expected in zip(execution.snapshot(ref), root, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="sampler completion"):
            execution.execute_anchor(handle, AnchorId(2))
        execution.execute_anchor(handle, AnchorId(1))
        execution.complete_sampler(handle)
        execution.execute_anchor(handle, AnchorId(2))
        publication = handle.context.publication
        retained_loss = handle.context.loss

        def reject_output(self: FunctionMath, context: StateTaskContext[TensorFrame]) -> TensorResult:
            raise RuntimeError("injected postpublication output failure")

        with monkeypatch.context() as fault:
            fault.setattr(FunctionMath, "output", reject_output)
            with pytest.raises(StateSettlementError) as failed:
                execution.finish_step(handle)
        assert failed.value.invocation is handle and failed.value.receipt is publication
        assert handle.context.loss is retained_loss
        result = execution.finish_step(handle)
        assert result.guidance is guidance and result.publication is publication
        assert result.loss is retained_loss and result.loss is not None
        assert handle.context.loss is None and handle.context.value is None
        assert handle.context.guidance is None and handle.context.sampler_feedback is None
        assert publication is not None
        for index, owner in enumerate(sorted(set(owners))):
            selected = [i for i, selected_owner in enumerate(owners) if selected_owner == owner]
            optimizer = optimizers[owner]
            optimizer.zero_grad()
            loss = F.mse_loss(_reference(source[selected], parameters[owner]), target[selected])
            loss.backward()
            torch.testing.assert_close(result.loss[index], loss.detach())
            optimizer.step()
            assert publication.before_refs[index] == latest[owner]
            latest[owner] = publication.after_refs[index]
            assert latest[owner].version_id == step_id + 1
            root = execution.snapshot(latest[owner])
            state = optimizer.state[parameters[owner]]
            for actual, expected in zip(
                root,
                (parameters[owner], parameters[owner].grad, state["step"], state["exp_avg"], state["exp_avg_sq"]),
                strict=True,
            ):
                torch.testing.assert_close(actual, expected)
        if step_id == 1:
            for actual, expected in zip(execution.snapshot(latest[1]), before[1], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(guidance.values, expected_prediction)
        with pytest.raises(RuntimeError):
            execution.finish_step(handle)

    execution.retire(identities)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
    assert diagnostics.loss_count == diagnostics.update_count == 3
    assert diagnostics.trained_rows == 4
    assert diagnostics.completed_sampler_pins == 4
    assert execution.arena.free_pages == 6
    assert execution.plan.fallback_count == 0

    def reject_admission(self: PageBundleArena, write: PageWrite) -> PagePublication:
        raise RuntimeError("injected admission publication failure")

    with monkeypatch.context() as fault:
        fault.setattr(PageBundleArena, "publish", reject_admission)
        with pytest.raises(RuntimeError, match="admission publication failure"):
            execution.admit(identities)
    assert execution.diagnostics().active_lineages == 0
    assert execution.arena.free_pages == 6
    execution.admit(identities)
    execution.retire(identities)
    assert execution.arena.free_pages == 6
    _esamp_native_oracle(monkeypatch, torch.device("cpu"), explicit=True)
    if torch.cuda.is_available():
        _esamp_native_oracle(monkeypatch, torch.device("cuda", 0))
        _esamp_native_oracle(monkeypatch, torch.device("cuda", 0), explicit=True)
    _ordinary_request_distillation()


def _ordinary_request_distillation() -> None:
    from pllm.execution.cuda_stream import CudaStreamReadiness
    from pllm.execution.runtime_facts import RuntimeFacts

    class Distillation(pllm.Workload):
        heads = pllm.RequestState(
            {"weight": lambda m: (m.hidden_size, m.hidden_size)},
            optimizer=pllm.AdamW(lr=0.03, weight_decay=0.1),
            keys=(RequestKeyField.PROMPT_IDX, RequestKeyField.SAMPLE_IDX),
            memory={"carry": lambda m: (m.hidden_size,)},
        )

        def ready(self, versions: Annotated[torch.Tensor, pllm.StateVersion("heads")]) -> torch.Tensor:
            return versions > 0

        @pllm.on(pllm.Phase.BEFORE_SAMPLE, rows="ready")
        @pllm.guide_sampling(beta=0.2)
        def predict(
            self,
            source: Annotated[torch.Tensor, pllm.Hidden(0)],
            weight: Annotated[torch.Tensor, pllm.Parameter("heads", "weight")],
            versions: Annotated[torch.Tensor, pllm.StateVersion("heads")],
        ) -> torch.Tensor:
            return torch.bmm(source.unsqueeze(1), weight).squeeze(1)

        @pllm.on(pllm.Phase.AFTER_SAMPLE)
        @pllm.train(state="heads", reduction="state_mean")
        def train(
            self,
            source: Annotated[torch.Tensor, pllm.Hidden(0), pllm.Capture("heads", "carry")],
            target: Annotated[torch.Tensor, pllm.Hidden(-1)],
            weight: Annotated[torch.Tensor, pllm.Parameter("heads", "weight")],
            carry: Annotated[torch.Tensor, pllm.Parameter("heads", "carry")],
        ) -> torch.Tensor:
            prediction = torch.bmm(source.unsqueeze(1), weight).squeeze(1) + carry
            return (prediction - target).square().mean(1)

    bindings = tuple(
        PackedHiddenResourceBinding(
            str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]"
        )
        for layer in (0, -1)
    )
    for device_name in ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",):
        device = torch.device(device_name)
        stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        facts = backend_facts(
            4,
            bindings,
            RuntimeFacts(device).timeline(),
            PageBundleProviderFacts("test.plain@1", 9, StateDeviceKind(device.type)),
            max_lineages=3,
        )
        workload = Distillation()
        deployment = workload.compile_backend(facts)
        execution = deployment.execution
        identities = tuple(RequestIdentity("same-id", 0, index) for index in range(3))
        latest = list(execution.admit(identities, stream))
        weights = [torch.nn.Parameter(torch.zeros((4, 4), device=device)) for _ in identities]
        optimizers = [torch.optim.AdamW((weight,), lr=0.03, weight_decay=0.1) for weight in weights]
        source = torch.linspace(-0.6, 0.8, 12, device=device).reshape(3, 4)
        target = torch.linspace(0.7, -0.3, 12, device=device).reshape(3, 4)
        initial = tuple(latest)
        held = execution.arena.pin_exact(initial, stream)
        # Cross-step memory: carries hold each lineage's last published capture; step 0 reads zeros.
        carries = [torch.zeros(4, device=device) for _ in identities]
        for step_id, owners in enumerate(((0, 1, 0), (2, 0, 0))):
            rows = PackedPhaseRowBatch(
                1,
                step_id,
                3,
                tuple(
                    PackedPhaseRow(i, i, step_id, HiddenPhase.DECODE, identities[owner])
                    for i, owner in enumerate(owners)
                ),
            )
            order = (2, 0, 1)
            sampler = PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, row) for i, row in enumerate(order)))
            value = TensorStepInput(
                rows,
                sampler,
                PackedHiddenSnapshotBatch(
                    rows,
                    tuple(
                        sorted(
                            (PackedHiddenSnapshot(bindings[0], source), PackedHiddenSnapshot(bindings[1], target)),
                            key=lambda item: item.resource.locator_key,
                        )
                    ),
                ),
                hidden_readiness=None if stream is None else CudaStreamReadiness(stream),
            )
            active_links = tuple(link for link in sampler.links if latest[owners[link.packed_row_index]].version_id > 0)
            expected = (
                torch.stack(
                    tuple(
                        source[link.packed_row_index] @ weights[owners[link.packed_row_index]].detach()
                        for link in active_links
                    )
                )
                if active_links
                else source.new_empty((0, 4))
            )
            result = execution.step(value, stream)
            rows, values = effective(result.guidance)
            assert rows == tuple(link.sampler_row_index for link in active_links)
            assert result.guidance.beta == 0.2
            torch.testing.assert_close(values, expected)
            for before, after, loss in zip(
                result.publication.before_refs, result.publication.after_refs, result.loss, strict=True
            ):
                owner = latest.index(before)
                selected = [index for index, selected_owner in enumerate(owners) if selected_owner == owner]
                optimizer = optimizers[owner]
                optimizer.zero_grad()
                oracle = F.mse_loss(source[selected] @ weights[owner] + carries[owner], target[selected])
                oracle.backward()
                optimizer.step()
                torch.testing.assert_close(loss, oracle.detach())
                state = optimizer.state[weights[owner]]
                # Each lineage's memory receives its last selected row's captured source.
                carries[owner] = source[selected[-1]].clone()
                for actual, expected_field in zip(
                    execution.snapshot(after, stream),
                    (
                        weights[owner].flatten(),
                        weights[owner].grad.flatten(),
                        state["step"].to(device),
                        state["exp_avg"].flatten(),
                        state["exp_avg_sq"].flatten(),
                        carries[owner],
                    ),
                    strict=True,
                ):
                    torch.testing.assert_close(actual, expected_field)
                latest[owner] = after
        for ref in initial:
            assert torch.count_nonzero(execution.arena.snapshot(held, ref)[0]) == 0
        execution.arena.wait_release(execution.arena.release_pin(held))
        execution.retire(identities)
        diagnostics = execution.diagnostics()
        assert diagnostics.loss_count == diagnostics.update_count == 4 and diagnostics.trained_rows == 6
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0


def _esamp_native_oracle(monkeypatch, device, explicit=False):
    from pllm.execution.cuda_stream import CudaStreamReadiness

    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    bindings = tuple(
        PackedHiddenResourceBinding(
            str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]"
        )
        for layer in (0, -1)
    )
    for paged, normalized, captured in (
        (False, False, False),
        (False, True, False),
        *(
            ((True, False, False), (True, True, False), (True, True, True))
            if device.type == "cuda" and not explicit
            else ()
        ),
    ):
        config = ESampRequestConfig(
            rank=4,
            paged=paged,
            output_layernorm=normalized,
            intervention=True,
            intervention_beta=0.2,
            learning_rate=0.03,
            weight_decay=0.1,
        )
        workload = ESampRequestWorkload(config)
        timeline = SchedulerTimeline(
            tuple(
                TimelineAnchor(AnchorSymbol(name), index)
                for index, name in enumerate(("model.execute", "sampler.pre_draw", "token.commit", "next.prepare"))
            ),
            (f"compute/{device.type}@1", f"state/dense_page_{device.type}@1"),
        )
        facts = backend_facts(
            8,
            bindings,
            timeline,
            PageBundleProviderFacts("test.paged@1", 8, StateDeviceKind(device.type), 0),
            max_lineages=3,
            memory_bytes=100000,
        )
        facts = workload.specialize(
            facts, ExecutionOptions(capture=captured, capture_memory_bytes=16 * 1024**2, explicit_math=explicit)
        )
        timeline = facts.timeline
        with pytest.raises(ValueError, match="workspace"):
            workload.compile_backend(
                replace(
                    facts,
                    timeline=replace(
                        timeline,
                        task_resources=tuple(
                            replace(item, workspace_bytes=0) if item.task == "loss" else item
                            for item in timeline.task_resources
                        ),
                    ),
                )
            )
        with pytest.raises(ValueError, match="output storage"):
            workload.compile_backend(
                replace(
                    facts,
                    timeline=replace(
                        timeline, value_resources=tuple(replace(item, bytes=0) for item in timeline.value_resources)
                    ),
                )
            )
        if (paged or explicit) and not captured:

            def modified_forward(self, source, up, gate, down, bias, scale, shift):
                return source + shift

            with monkeypatch.context() as changed:
                changed.setattr(ESampRequestWorkload, "forward", modified_forward)
                with pytest.raises(ValueError, match="customized mathematics"):
                    workload.compile_backend(facts)
            with monkeypatch.context() as changed:
                changed.setitem(
                    ESampRequestWorkload.predict.__annotations__, "source", Annotated[torch.Tensor, pllm.Hidden(-1)]
                )
                with pytest.raises(ValueError, match="customized mathematics"):
                    workload.compile_backend(facts)
        declaration = workload.declaration(facts)
        plan = compile_page_bundle_plan(
            (declaration.graph, declaration.layouts),
            timeline,
            facts.providers,
            facts.max_admitted_lineages,
            facts.state_memory_budget_bytes,
            device,
        )
        assert (
            next(task for task in plan.semantic_graph.tasks if task.symbol == "loss").outputs[0].kind
            is SemanticOutputKind.HOST
        )
        workload.plan_backend(
            replace(facts, timeline=replace(timeline, memory_budget_bytes=plan.schedule_decision.peak_memory_bytes))
        )
        with pytest.raises(ValueError):
            workload.plan_backend(
                replace(facts, timeline=replace(timeline, memory_budget_bytes=plan.state_physical.reserved_bytes))
            )
        execution = workload.bind_math(facts, plan).bind(PageBundleArena(plan), AnchorId(1))
        identities = tuple(RequestIdentity(str(i), i, 0) for i in range(3))
        initial = execution.admit(identities, stream)
        old_pin = execution.arena.pin_exact(initial, stream)
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            parameters = [execution.snapshot(ref, stream)[0].requires_grad_() for ref in initial]
            originals = [parameter.detach().clone() for parameter in parameters]
            optimizers = [
                torch.optim.AdamW(
                    (parameter,),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    capturable=stream is not None,
                    foreach=True,
                )
                for parameter in parameters
            ]
            source = torch.linspace(-0.7, 0.9, 48, device=device).view(3, 16)[:, ::2]
            target = source.flip(1) * 0.7

            def reference(x, p):
                a, g, b, bias = p[:, :100].split((32, 32, 32, 4), dim=1)
                u = torch.bmm(x.unsqueeze(1), a.view(-1, 8, 4)).squeeze(1)
                z = torch.bmm(x.unsqueeze(1), g.view(-1, 8, 4)).squeeze(1) + bias
                y = x + torch.bmm((u * F.silu(z)).unsqueeze(1), b.view(-1, 4, 8)).squeeze(1)
                return F.layer_norm(y, (8,)) * p[:, 100:108] + p[:, 108:] if normalized else y

            for step_id, owners in enumerate(((1, 0, 1), (0,))):
                rows = PackedPhaseRowBatch(
                    1,
                    step_id,
                    len(owners),
                    tuple(
                        PackedPhaseRow(i, i, step_id, HiddenPhase.DECODE, identities[owner])
                        for i, owner in enumerate(owners)
                    ),
                )
                values = {bindings[0]: source[: len(owners)], bindings[1]: target[: len(owners)]}
                value = TensorStepInput(
                    rows,
                    PackedSamplerRowBatch(rows, tuple(PackedSamplerRowLink(i, i) for i in range(len(owners)))),
                    PackedHiddenSnapshotBatch(
                        rows,
                        tuple(
                            PackedHiddenSnapshot(key, values[key])
                            for key in sorted(bindings, key=lambda item: item.locator_key)
                        ),
                    ),
                    hidden_readiness=None if stream is None else CudaStreamReadiness(stream),
                )
                if step_id == 0:
                    excess = replace(rows, total_rows=4, rows=(*rows.rows, replace(rows.rows[0], packed_row_index=3)))
                    with pytest.raises(ValueError, match="signed workspace envelope"):
                        execution.start_step(
                            replace(
                                value,
                                rows=excess,
                                sampler_rows=PackedSamplerRowBatch(
                                    excess, tuple(PackedSamplerRowLink(i, i) for i in range(4))
                                ),
                                hidden=PackedHiddenSnapshotBatch(
                                    excess,
                                    tuple(
                                        PackedHiddenSnapshot(
                                            snapshot.resource, torch.cat((snapshot.hidden, snapshot.hidden[:1]))
                                        )
                                        for snapshot in value.hidden.snapshots
                                    ),
                                ),
                            ),
                            stream,
                        )
                for optimizer in optimizers:
                    optimizer.zero_grad()
                expected_prediction = reference(
                    source[: len(owners)], torch.stack([parameters[owner] for owner in owners])
                )
                row_losses = (expected_prediction - target[: len(owners)]).square().mean(1)
                expected_losses = {
                    owner: row_losses[[i for i, key in enumerate(owners) if key == owner]].mean()
                    for owner in set(owners)
                }
                sum(expected_losses.values()).backward()
                invocation = execution.start_step(value, stream)
                assert not invocation.context.results
                execution.cancel_step(invocation)
                saved_sources = []
                saved_shapes = []

                def save(tensor):
                    saved_shapes.append(tuple(tensor.shape))
                    if tensor.shape == (len(owners), 1, 8):
                        saved_sources.append(tensor.untyped_storage().data_ptr())
                    return tensor

                with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
                    result = execution.step(value, stream)
                if not paged and not captured and not explicit:
                    assert saved_sources and set(saved_sources) == {source.untyped_storage().data_ptr()}
                if paged or explicit:
                    assert not saved_shapes, "explicit derivatives must not build an autograd tape"
                assert result.guidance is not None and result.loss is not None and result.publication is not None
                if step_id:
                    torch.testing.assert_close(result.guidance.values, expected_prediction, rtol=1e-5, atol=1e-6)
                for ref, loss in zip(result.publication.after_refs, result.loss, strict=True):
                    owner = next(i for i, original in enumerate(initial) if original.lineage_id == ref.lineage_id)
                    optimizer = optimizers[owner]
                    expected = expected_losses[owner]
                    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
                    optimizer.step()
                    state = optimizer.state[parameters[owner]]
                    actual = execution.snapshot(ref, stream)
                    for tensor, target_value in zip(
                        actual,
                        (
                            parameters[owner],
                            parameters[owner].grad,
                            state["step"].to(device),
                            state["exp_avg"],
                            state["exp_avg_sq"],
                        ),
                        strict=True,
                    ):
                        torch.testing.assert_close(
                            tensor,
                            target_value,
                            rtol=1e-4,
                            atol=1e-6,
                            msg=lambda message: (
                                f"paged={paged}, normalized={normalized}, captured={captured}, step={step_id}: {message}"
                            ),
                        )
            for ref, expected in zip(initial, originals, strict=True):
                torch.testing.assert_close(execution.arena.snapshot(old_pin, ref)[0], expected, rtol=0, atol=0)
        execution.arena.wait_release(execution.arena.release_pin(old_pin))
        execution.retire(identities)
        assert execution.diagnostics().fallback_count == execution.diagnostics().active_lineages == 0

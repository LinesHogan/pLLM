"""Independent producer arithmetic, atomic publication and resource/request isolation."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

import pllm
from pllm import testing
from pllm.contracts.backend_attachment import AttachmentHiddenResource
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import ParameterTargetShape
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, StateResource
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.workloads.state_producers import TARGETS, DecodeFactors, ProjectionState, PromptFactors, PromptMixture
from test.backend_facts import assert_finite_resources, backend_facts


@pytest.fixture(params=("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",))
def device(request):
    return torch.device(request.param)


def deployment(workload, device=torch.device("cpu")):
    facts = backend_facts(
        4,
        (),
        model_timeline(device, ("transition/full_root@1", "optimizer/adamw@1")),
        PageBundleProviderFacts("test.producers@1", 16, StateDeviceKind(device.type)),
        memory_bytes=1 << 20,
        max_rows=8,
        layer_count=10,
        targets=tuple(ParameterTargetShape(t, 4, o) for t, o in zip(TARGETS, (4, 2), strict=True)),
    )
    facts = replace(
        facts,
        probe=replace(
            facts.probe,
            hidden_resources=tuple(
                sorted(
                    (
                        AttachmentHiddenResource(testing.hidden_binding(h.layer, h.site, 10), h.phase)
                        for h in workload.backend_requirements().hidden_inputs
                    ),
                    key=lambda r: r.resource_key,
                )
            ),
        ),
    )
    return workload.compile_backend(assert_finite_resources(workload, facts, main_fallback=True))


def packed(identities, hidden, phases, lengths=(1, 1), *, offset=0, stream=None):
    value = testing.step_input(
        identities, hidden, phases=phases, sampler_order=(1, 0), token_offset=offset, stream=stream
    )
    rows = replace(
        value.rows,
        rows=tuple(replace(row, prompt_length=length) for row, length in zip(value.rows.rows, lengths, strict=True)),
    )
    return replace(
        value,
        rows=rows,
        sampler_rows=replace(value.sampler_rows, row_batch=rows),
        hidden=replace(value.hidden, row_batch=rows),
    )


@pytest.mark.parametrize("workload", [PromptFactors(), PromptMixture()])
@pytest.mark.parametrize("length", [1, 75, 127, 128, 129, 2047, 2048, 2049])
def test_prompt_producers_publish_only_completed_requests(workload, device, length):
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    deployed = deployment(workload, device=device)
    network = next(p for p in deployed.attachment.inputs.plan.state_physical.representations if p.resource == "network")
    assert network.page_capacity == 1
    execution = deployed.execution
    identities = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    execution.admit((), stream=stream)
    initial = execution.admit(identities, stream=stream)
    context = torch.tensor([[0.2, -0.4, 0.1, 0.7], [-0.6, 0.3, 0.8, -0.2]], device=device)
    tap = testing.hidden_binding(8, pllm.model.HiddenSite.BLOCK_OUTPUT, layer_count=10)
    for offset in range(0, length - 1, 128):
        testing.run_step(
            execution,
            packed(
                identities,
                {tap: context},
                (HiddenPhase.PREFILL, HiddenPhase.PREFILL),
                (length, length + 1),
                offset=offset,
                stream=stream,
            ),
            stream=stream,
        )
        assert tuple(execution.arena.latest(ref.lineage_id) for ref in initial) == tuple(initial)
    value = packed(
        identities,
        {tap: context},
        (HiddenPhase.PREFILL, HiddenPhase.PREFILL),
        (length, length + 1),
        offset=length - 1,
        stream=stream,
    )
    invocation = execution.start_step(replace(value, hidden=None, hidden_readiness=None), stream=stream)
    selected = invocation.selection.for_resource(StateResource("factors"))
    network_ref = invocation.selection.for_resource(StateResource("network")).refs[0]
    weights = execution.snapshot(network_ref, stream=stream)[0].double()
    refs = tuple(selected.refs)
    execution.execute_anchor(invocation, AnchorId(0))
    assert all(execution.arena.latest(ref.lineage_id) == ref for ref in refs)
    execution.submit_input(invocation, value)
    for anchor in execution.arena.plan.timeline.anchors[1:]:
        execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
        if anchor.ordinal == execution.sampler_anchor_id:
            execution.complete_sampler(invocation, stream=stream)
    execution.finish_step(invocation)
    changed = execution.arena.latest(refs[0].lineage_id)
    assert changed.version_id == 1 and execution.arena.latest(refs[1].lineage_id) == refs[1]
    actual = execution.snapshot(changed, stream=stream)
    h = context[:1].double()
    if isinstance(workload, PromptMixture):
        k, r = workload.components, workload.rank
        gate, bias, aq, av, bq, bv = weights.split((4 * k, k, k * r * 4, k * r * 4, k * 4 * r, k * 2 * r))
        probabilities = F.linear(h, gate.reshape(k, 4), bias).softmax(-1)[0]
        for a, b, bank_a, bank_b, width in ((actual[0], actual[1], aq, bq, 4), (actual[2], actual[3], av, bv, 2)):
            dense = sum(
                probabilities[j] * (bank_b.reshape(k, width, r)[j] @ bank_a.reshape(k, r, 4)[j]) for j in range(k)
            )
            torch.testing.assert_close(b.double() @ a.double(), dense, rtol=2e-6, atol=1e-9)
    else:
        count = 2 * (4 + 4 + 4 + 2)
        w1, b1, w2, b2 = weights.split((16, 4, count * 4, count))
        expected = F.linear(F.linear(h, w1.reshape(4, 4), b1).tanh(), w2.reshape(count, 4), b2)[0]
        torch.testing.assert_close(
            torch.cat([field.flatten() for field in actual]).double(), expected, rtol=2e-6, atol=1e-7
        )
    # Reordering and decode cannot publish another prompt update; a partial prompt cannot publish early.
    testing.run_step(
        execution,
        packed(identities[::-1], {tap: context.flip(0)}, (HiddenPhase.DECODE, HiddenPhase.DECODE), stream=stream),
        stream=stream,
    )
    assert execution.arena.latest(refs[0].lineage_id) == changed
    torch.testing.assert_close(execution.snapshot(network_ref, stream=stream)[0].double(), weights, rtol=0, atol=0)
    execution.retire(identities, wait=True)
    execution.retire((), wait=True)
    assert execution.diagnostics().active_lineages == 0
    assert initial


def test_decode_adam_matches_independent_optimizer_and_keeps_entry_pin(device):
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    execution = deployment(DecodeFactors(), device=device).execution
    ids = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    refs = execution.admit(ids, stream=stream)
    flat = execution.snapshot(refs[0], stream=stream)[0].clone().requires_grad_()
    optimizer = torch.optim.AdamW([flat], lr=1e-3, weight_decay=0.0)
    x, target = (
        torch.tensor([[0.2, -0.4, 0.1, 0.7]], device=device),
        torch.tensor([[0.1, -0.2, 0.3, -0.4]], device=device),
    )
    aq, bq, av, bv = (
        t.reshape(shape) for t, shape in zip(flat.split((8, 8, 8, 4)), ((2, 4), (4, 2), (2, 4), (2, 2)), strict=True)
    )
    loss = ((x @ aq.T @ bq.T - target) ** 2).mean() + ((x @ av.T @ bv.T - target[:, :2]) ** 2).mean()
    loss.backward()
    optimizer.step()
    hidden = {
        testing.hidden_binding(0, pllm.model.HiddenSite.BLOCK_OUTPUT, layer_count=10): x.expand(2, -1),
        testing.hidden_binding(9, pllm.model.HiddenSite.BLOCK_OUTPUT, layer_count=10): target.expand(2, -1),
    }
    result = testing.run_step(
        execution, packed(ids, hidden, (HiddenPhase.DECODE, HiddenPhase.PREFILL), stream=stream), stream=stream
    )
    assert tuple(result.publication.before_refs) == (refs[0],)
    latest = execution.arena.latest(refs[0].lineage_id)
    actual = execution.snapshot(latest, stream=stream)
    for tensor, expected in zip(
        (actual[0], actual[2], actual[3], actual[4]),
        (flat, optimizer.state[flat]["step"], optimizer.state[flat]["exp_avg"], optimizer.state[flat]["exp_avg_sq"]),
        strict=True,
    ):
        torch.testing.assert_close(tensor, expected.to(tensor), rtol=2e-6, atol=1e-8)
    assert execution.arena.latest(refs[1].lineage_id) == refs[1]
    assert not x.requires_grad and not target.requires_grad
    execution.retire(ids, wait=True)
    assert execution.diagnostics().active_lineages == 0
    assert PromptFactors.consume is PromptMixture.consume is DecodeFactors.consume is ProjectionState.consume


def test_import_and_write_share_views_and_cancel_private_publication(monkeypatch, device):
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    from pllm.contracts.request_identity import RequestKeyField
    from pllm.contracts.request_state import RequestStateKey
    from pllm.execution.backend_attachment import compile_backend_attachment
    from pllm.execution.state_execution import PublishStateTask

    class Imported(ProjectionState):
        @property
        def factors(self) -> pllm.RequestState:
            return replace(super().factors, imported=True)

    class Written(ProjectionState):
        @pllm.write(state="factors", fields=("a_0", "b_0", "a_1", "b_1"))
        def produce(
            self,
            aq=pllm.parameter("factors", "a_0"),
            bq=pllm.parameter("factors", "b_0"),
            av=pllm.parameter("factors", "a_1"),
            bv=pllm.parameter("factors", "b_1"),
        ):
            return tuple(torch.full_like(value, index + 1) for index, value in enumerate((aq, bq, av, bv)))

    imported, written = deployment(Imported(), device=device), deployment(Written(), device=device)
    assert imported.attachment.inputs.request.overlay_bindings == written.attachment.inputs.request.overlay_bindings
    identity = RequestIdentity("swap", 0, 0)
    key = RequestStateKey.project(identity, (RequestKeyField.PROMPT_IDX,))
    values = tuple(
        torch.full((1, *shape), index + 1.0, device=device)
        for index, shape in enumerate(((2, 4), (4, 2), (2, 4), (2, 2)))
    )
    for deployed, payload in ((imported, pllm.state.StateTensorUpdate(values, (key,))), (written, None)):
        execution = deployed.execution
        execution.admit((identity,), stream=stream)
        result = testing.run_step(execution, testing.step_input((identity,), auxiliary=payload), stream=stream)
        torch.testing.assert_close(
            execution.snapshot(result.publication.after_refs[0], stream=stream), tuple(value[0] for value in values)
        )
        execution.retire((identity,), wait=True)
        assert execution.diagnostics().active_lineages == 0
        execution.admit((identity,), stream=stream)
        payload = None if payload is None else replace(payload, values=tuple(value.clone() * 2 for value in values))
        result = testing.run_step(execution, testing.step_input((identity,), auxiliary=payload), stream=stream)
        torch.testing.assert_close(
            execution.snapshot(result.publication.after_refs[0], stream=stream),
            tuple(value[0] * (2 if payload is not None else 1) for value in values),
        )
        execution.retire((identity,), wait=True)
        assert execution.diagnostics().active_lineages == execution.diagnostics().pending_state_releases == 0
        pair = (identity, replace(identity, request_id="another", prompt_idx=1))
        execution.admit(pair, stream=stream)
        payload = (
            None
            if payload is None
            else replace(
                payload,
                values=tuple(torch.cat((value, value * 3)) for value in values),
                keys=tuple(RequestStateKey.project(item, (RequestKeyField.PROMPT_IDX,)) for item in pair),
            )
        )
        result = testing.run_step(execution, testing.step_input(pair, auxiliary=payload), stream=stream)
        for index, item in enumerate(pair):
            (ref,) = execution.directory.latest_refs((item,))
            torch.testing.assert_close(
                execution.snapshot(ref, stream=stream),
                tuple(value[0] * (3 if index and payload is not None else 1) for value in values),
            )
        execution.retire(pair, wait=True)
        assert execution.diagnostics().active_lineages == execution.diagnostics().pending_state_releases == 0
    execution = written.execution
    (entry,) = execution.admit((identity,), stream=stream)
    before = execution.snapshot(entry, stream=stream)

    def fail_before_publish(self, context):
        assert len(context.page_writes) == 1
        raise RuntimeError("cancel between private write and publication")

    monkeypatch.setattr(PublishStateTask, "execute", fail_before_publish)
    with pytest.raises(RuntimeError, match="cancel between"):
        testing.run_step(execution, testing.step_input((identity,)), stream=stream)
    assert execution.arena.latest(entry.lineage_id) == entry
    torch.testing.assert_close(execution.snapshot(entry, stream=stream), before)
    execution.retire((identity,), wait=True)
    assert execution.diagnostics().active_lineages == execution.diagnostics().pending_state_releases == 0
    inputs = written.attachment.inputs
    binding = inputs.request.overlay_bindings[0]
    broken = replace(binding, fields=(replace(binding.fields[0], offset=10000), binding.fields[1]))
    with pytest.raises(ValueError, match="view"):
        compile_backend_attachment(
            replace(
                inputs, request=replace(inputs.request, overlay_bindings=(broken, *inputs.request.overlay_bindings[1:]))
            )
        )


def test_initialized_reader_and_hidden_before_model_boundary():
    execution = deployment(ProjectionState()).execution
    identity = RequestIdentity("readonly", 0, 0)
    (ref,) = execution.admit((identity,))
    assert testing.run_step(execution, testing.step_input((identity,))).publication is None
    assert execution.arena.latest(ref.lineage_id) == ref
    execution.retire((identity,))

    class Cycle(PromptFactors):
        @pllm.on(pllm.Phase.BEFORE_MODEL)
        @pllm.write(state="factors", fields=("a_0",))
        def produce(self, h=pllm.hidden(8, phase="prefill")):
            return (h,)

    with pytest.raises(ValueError, match="after model forward"):
        Cycle().declaration(
            backend_facts(
                4,
                (),
                model_timeline(torch.device("cpu")),
                None,
                targets=tuple(ParameterTargetShape(target, 4, 4) for target in TARGETS),
            )
        )


def test_backend_points_and_compile_rejects_unknown_anchor():
    from pllm.contracts.semantic_graph import AnchorSymbol, TimingConstraint
    from pllm.execution.scheduler import compile_schedule

    plan = deployment(PromptFactors()).attachment.inputs.plan
    point = next(a.ordinal for a in plan.timeline.anchors if a.symbol == "state.publish")
    assert all(t.earliest == t.anchor_id == point for t in plan.schedule_physical.tasks)
    constrained = replace(
        plan.semantic_graph,
        tasks=tuple(
            replace(task, timing=TimingConstraint(AnchorSymbol("state.publish"), AnchorSymbol("sampler.pre_draw")))
            for task in plan.semantic_graph.tasks
        ),
    )
    decision = compile_schedule(constrained, plan.timeline, state=plan.state_physical)
    assert all(plan.timeline.anchors[task.anchor_id].symbol == "state.publish" for task in decision.tasks)
    undeclared = replace(
        constrained,
        tasks=tuple(
            replace(task, timing=TimingConstraint(AnchorSymbol("model.layer.8"), AnchorSymbol("sampler.pre_draw")))
            for task in constrained.tasks
        ),
    )
    with pytest.raises(ValueError, match="unknown anchor"):
        compile_schedule(undeclared, plan.timeline, state=plan.state_physical)

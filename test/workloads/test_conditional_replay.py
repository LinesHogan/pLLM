"""Evidence 40: real publication, independent recurrence, skip and committed failure."""


from dataclasses import replace

import pytest
import torch
from pllm.execution.trace import TracePhase

from pllm.authoring import ParameterTargetShape, TensorStepInput, BackendCompileFacts
from pllm.contracts.backend_attachment import BackendPlacementProbe, AttachmentTopology, AttachmentGraphMode
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.state_execution import StateSettlementError
from pllm.execution.tensor_program import _read
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.parameter_overlay import CapturedParameterOperand, ParameterModuleKind, ParameterRole, ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, ConditionalLayerReplay
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType


def test_learning_publication_conditional_replay_matches_reference():
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    config = InPlaceTTTConfig((target,), chunk_size=2, learning_rate=0.2, replay=ConditionalLayerReplay((0,)))
    timeline = RuntimeFacts(torch.device("cpu")).timeline(("model/layer_replay@1", "transition/full_root@1"))
    providers = (PageBundleProviderFacts("oracle.cpu@1", 8),)
    workload = InPlaceTTTWorkload(config)
    facts = BackendCompileFacts(BackendPlacementProbe("test.replay@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CPU, 0, 1, 1,
        TensorDType.FLOAT64, 2, 8, (), ()), timeline, providers, 2, 8192, 0, (ParameterTargetShape(target, 2, 2),))
    execution = workload.compile_backend(facts).execution
    assert decode_boundary(encode_boundary(execution.arena.plan), PlanBundle) == execution.arena.plan
    identities = (RequestIdentity("train", 0, 0), RequestIdentity("skip", 1, 0))
    initial = execution.admit(identities)
    old_pin = execution.arena.pin_exact(initial)
    base = torch.tensor([[0.4, -0.1], [0.2, 0.3]], dtype=torch.float64)
    x = torch.tensor([[0.2, 0.3]], dtype=torch.float64)
    y = torch.tensor([[0.4, 0.1]], dtype=torch.float64)
    rows = PackedPhaseRowBatch(1, 0, 2, tuple(PackedPhaseRow(index, index, 2,
        HiddenPhase.PREFILL if index == 0 else HiddenPhase.DECODE, identity)
        for index, identity in enumerate(identities)))
    value = TensorStepInput(rows, PackedSamplerRowBatch(rows, ()), operands=tuple(
        CapturedParameterOperand(row.identity, target, row.phase, x, y, base) for row in rows.rows))

    class ToyLayer:
        def __init__(self):
            self.rows = rows.rows
            self.kv = torch.tensor([[[0.1, 0.2], [0.3, -0.1], [0.5, 0.1]],
                [[0.9, 0.2], [0.1, 0.7], [0.2, 0.2]]], dtype=torch.float64)
            self.normal = self.kv.clone()
            self.calls = 0
            self.fail_at = 0
            self.cancel_at = 0
            self.context = None
            self.pins = []

        def seed(self, row):
            return self.normal[row.request_ordinal, -1:].clone()

        def step(self, row, carry, arena, pin, ref):
            self.calls += 1
            self.pins.append(pin)
            weights = base + arena.borrow_field(pin, ref, LayoutFieldId(0))
            result = torch.tanh(carry @ weights.T + self.kv[row.request_ordinal, :-1].mean(0))
            self.kv[row.request_ordinal, -1:].copy_(result)
            if self.calls == self.cancel_at:
                self.context.cancel_requested.set()
            if self.calls == self.fail_at:
                raise RuntimeError("injected replay failure")
            return result

        def restore(self):
            self.kv.copy_(self.normal)

    def advance(layer):
        invocation = execution.start_step(value)
        context = invocation.context
        context.replay = layer
        layer.context = context
        for anchor in execution.arena.plan.timeline.anchors:
            execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        if context.updated_refs:
            assert context.write is None and context.publication is not None
            batch = _read(context, "private")
            versions = dict(zip(context.publication.before_refs, context.publication.after_refs, strict=True))
            assert batch.after_refs == tuple(versions.get(ref, ref) for ref in context.pin.refs)
            for ref, root in zip(batch.after_refs, batch.roots, strict=True):
                for actual, expected in zip(root, execution.snapshot(ref), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        result = execution.finish_step(invocation)
        return context, result

    tail = ToyLayer()
    context, result = advance(tail)
    assert context.publication is not None and tail.calls == 0 and context.published_pin is None
    assert not result.entries[0].changed
    layer = ToyLayer()
    context, result = advance(layer)
    trace = execution.trace.snapshot()
    replay = tuple(event for event in trace.events
        if event.invocation_id == context.invocation_id and event.phase is TracePhase.REPLAY)
    assert layer.calls == 2 and len(replay) == 2
    assert all(pin is context.published_pin for pin in layer.pins)
    trained, = (entry for entry in result.entries if entry.identity == identities[0])
    assert context.published_pin.refs == (trained.after_ref,)
    assert context.published_release.pin is context.published_pin
    assert all(execution.arena.plan.semantic_graph.tasks[event.task_id] is execution.arena.plan.semantic_graph.tasks[1]
        and event.refs == context.published_pin.refs and event.pin_id == context.published_pin.pin_id
        and event.operation_id == context.publication.operation_id for event in replay)
    published, = (event for event in trace.events if event.invocation_id == context.invocation_id
        and event.phase is TracePhase.PUBLISH)
    settled, = (event for event in trace.events if event.invocation_id == context.invocation_id
        and event.phase is TracePhase.SETTLED)
    assert published.refs is context.publication.after_refs and published.sequence < replay[0].sequence < settled.sequence
    assert settled.release_id == context.release.release_id and settled.operation_id == published.operation_id
    # Independent scalar outer products and recurrence; no production math helper.
    weights = base.clone()
    for output in range(2):
        for feature in range(2):
            weights[output, feature] += 0.2 * 2 * y[0, output] * x[0, feature]
    expected = layer.normal[0, -1].clone()
    for _ in range(2):
        expected = torch.tensor([torch.tanh(sum(expected[j] * weights[i, j] for j in range(2))
            + sum(layer.normal[0, k, i] for k in range(2)) / 2) for i in range(2)])
    torch.testing.assert_close(layer.kv[0, -1], expected)
    torch.testing.assert_close(layer.kv[0, :-1], layer.normal[0, :-1])
    torch.testing.assert_close(layer.kv[1], layer.normal[1])
    for ref in initial:
        assert torch.count_nonzero(execution.arena.borrow_field(old_pin, ref, LayoutFieldId(0))) == 0
    execution.arena.wait_release(execution.arena.release_pin(old_pin))

    # Tail-only publication still skips. Failure after the next update restores KV,
    # but does not roll back or repeat the committed parameter publication.
    advance(ToyLayer())
    for cancellation in (False, True):
        failed_layer = ToyLayer()
        if cancellation:
            failed_layer.cancel_at = 1
        else:
            failed_layer.fail_at = 2
        with pytest.raises(StateSettlementError) as failure:
            advance(failed_layer)
        failed = failure.value.invocation
        assert failure.value.receipt is failed.context.publication
        torch.testing.assert_close(failed_layer.kv, failed_layer.normal)
        execution.cancel_step(failed)
        assert failed.context.published_release is not None
        failed_events = tuple(event for event in execution.trace.snapshot().events if event.invocation_id == failed.invocation_id)
        assert sum(event.phase is TracePhase.PUBLISH for event in failed_events) == 1
        assert any(event.phase is TracePhase.TASK_FAILED for event in failed_events)
        assert any(event.phase is TracePhase.FAILED and event.operation_id == failure.value.receipt.operation_id
            for event in failed_events)
        assert [event.phase for event in failed_events[-2:]] == [TracePhase.ABORT, TracePhase.SETTLED]
        advance(ToyLayer())
    altered = InPlaceTTTWorkload(replace(config, replay=ConditionalLayerReplay((0,), max_iterations=1))).plan_backend(facts).inputs.plan
    assert altered.schedule_physical.schedule_signature != execution.plan.schedule_signature
    with pytest.raises(ValueError, match="unsupported conditional replay"):
        InPlaceTTTWorkload(replace(config, replay=ConditionalLayerReplay((0,), max_iterations=3))).plan_backend(facts)
    # A replay range is a sorted unique set, and every replayed layer must be
    # covered by a signed overlay target.
    ranged_targets = (target, ParameterTarget(1, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT))
    ranged_facts = replace(facts, targets=tuple(ParameterTargetShape(item, 2, 2) for item in ranged_targets))
    ranged = InPlaceTTTWorkload(InPlaceTTTConfig(ranged_targets, chunk_size=2, learning_rate=0.2,
        replay=ConditionalLayerReplay((0, 1)))).plan_backend(ranged_facts)
    replay_task, = (task for task in ranged.inputs.plan.semantic_graph.tasks if task.replay is not None)
    assert replay_task.replay == ConditionalLayerReplay((0, 1))
    for bad in ((1, 0), (0, 0), ()):
        with pytest.raises(ValueError, match="sorted|unique|requires"):
            InPlaceTTTWorkload(InPlaceTTTConfig(ranged_targets, chunk_size=2, learning_rate=0.2,
                replay=ConditionalLayerReplay(bad))).plan_backend(ranged_facts)
    with pytest.raises(ValueError, match="published parameter overlay"):
        InPlaceTTTWorkload(InPlaceTTTConfig(ranged_targets, chunk_size=2, learning_rate=0.2,
            replay=ConditionalLayerReplay((0, 2)))).plan_backend(ranged_facts)
    execution.retire(identities)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0

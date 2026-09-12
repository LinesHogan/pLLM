"""Evidence 27: serial recurrence oracle, row selection and stateless settlement."""

from dataclasses import replace
from time import perf_counter_ns

import pytest
import torch
import pllm
from pllm.execution.trace import TracePhase

from pllm.authoring import BackendCompileFacts, TensorStepInput
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.execution.replay import TokenwiseReplayPrimitive, TokenwiseReplayTask, TOKENWISE_REPLAY_IMPLEMENTATION, mix_layer_outputs
from pllm.execution.tokenwise_tuning import TokenwiseTuning
from pllm.execution.tokenwise_region import compile_tokenwise_region
from pllm.contracts.region_program import TensorType
from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
from pllm.execution.region_resources import HostResourceBinding, PagedResourceBinding, ReadonlyResourceBinding
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.recirculation import RecirculationConfig, RecirculationWorkload
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe, CausalModelDomain
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.config import ExecutionOptions


def test_recirculation_selects_rows_and_replays_typed_effect() -> None:
    probe = BackendPlacementProbe("test.replay@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CPU, 0, 1, 1,
        TensorDType.FLOAT32, 4, 8, (), ("model/tokenwise_replay@1",), causal_domain=CausalModelDomain(2, 3, 8))
    facts = BackendCompileFacts(probe, RuntimeFacts(torch.device("cpu")).timeline(("model/tokenwise_replay@1",)),
        (), 2, 0, 0, layer_count=3)
    config = RecirculationConfig(alpha=0.7, beta=None, ramp_tokens=2)
    assert RecirculationConfig is pllm.Recurrence
    class PlainRecurrence(pllm.Workload):
        @property
        def recurrence(self) -> pllm.Recurrence:
            return config

    plain = PlainRecurrence().compile_backend(facts).execution
    execution = RecirculationWorkload(config).compile_backend(facts).execution
    assert plain.arena.plan.semantic_graph.tasks == execution.arena.plan.semantic_graph.tasks
    # A budget that fits only the reference must still install an auto-policy
    # model; expensive windows must not make the whole attachment inadmissible.
    workload = RecirculationWorkload(config)
    serial_facts = workload.specialize(facts, ExecutionOptions(schedule_policy="serial"))
    reference_budget = (max(serial_facts.timeline.reserved_bytes)
        + sum(f.workspace_bytes + f.resident_bytes for f in serial_facts.timeline.task_resources)
        + sum(f.bytes for f in serial_facts.timeline.value_resources))
    tight_facts = workload.specialize(facts, ExecutionOptions(memory_budget_bytes=reference_budget))
    tight = workload.compile_backend(tight_facts)
    task = TokenwiseReplayTask.from_implementation(tight.execution.implementation(TOKENWISE_REPLAY_IMPLEMENTATION))
    assert task.policy.schedule_policy == "auto" and len(task.region.candidate_proposals) == 1
    assert task.region.candidate_proposals[0].payload == task.region.payload
    assert tight.attachment.inputs.plan.timeline.memory_budget_bytes == reference_budget
    with pytest.raises(ValueError, match="reference payload exceeds"):
        workload.specialize(facts, ExecutionOptions(memory_budget_bytes=reference_budget - 1))
    from pllm.workload_transport import decode_workload, encode_workload
    restored = decode_workload(encode_workload(RecirculationWorkload(config)), device=torch.device("cpu"), seed=0)
    assert isinstance(restored, RecirculationWorkload)
    assert restored.declare(facts) == RecirculationWorkload(config).declare(facts)
    assert decode_boundary(encode_boundary(execution.arena.plan), PlanBundle) == execution.arena.plan
    identities = (RequestIdentity("prefill", 0, 0), RequestIdentity("decode", 1, 0))
    rows = PackedPhaseRowBatch(1, 1, 4, tuple(PackedPhaseRow(i, int(i == 3),
        i if i < 3 else 0, HiddenPhase.PREFILL if i < 3 else HiddenPhase.DECODE, identities[int(i == 3)])
        for i in range(4)))
    value = TensorStepInput(rows, PackedSamplerRowBatch(rows, (PackedSamplerRowLink(0, 2), PackedSamplerRowLink(1, 3))))
    wrong_epoch = replace(rows, plan_epoch=2)
    with pytest.raises(ValueError, match="signed epoch"):
        execution.start_step(replace(value, rows=wrong_epoch, sampler_rows=replace(value.sampler_rows, row_batch=wrong_epoch)))
    inputs = torch.tensor([[0.4, -0.1, 0.3, 0.7], [0.2, 0.9, -0.3, 0.8], [-0.2, 0.1, 0.8, 0.4], [0.7, 0.2, -0.1, 0.3]])

    class Decoder(TokenwiseReplayPrimitive):
        def __init__(self):
            self.batch = rows
            self.cache = torch.zeros(2, 3, 3, 4)
            self.saved = {}
            self.normalized = set()
            self.replaying = set()
            self.seen_layers = set()
            self.output = None
            self.fail = False
            self.cancel = None
            self.calls = []

        @property
        def rows(self):
            return self.batch.rows

        def bind_kv(self, resource, owner):
            table = torch.arange(self.cache.shape[0]).reshape(-1, 1)
            slots = torch.tensor([row.request_ordinal * self.cache.shape[2] + row.token_offset for row in self.rows])
            layers = tuple(PagedLayer(str(i), self.cache[:, i].unsqueeze(1), table, slots) for i in range(self.cache.shape[1]))
            plan = bind_paged_access(self.rows, layers)
            return PagedResourceBinding(PagedResourceView(resource, plan, tuple((i, str(i)) for i in range(len(layers)))), owner)

        def bind_weights(self, resource, owner):
            # This oracle has scalar formulae rather than learned parameters.
            cells = tuple(ReadonlyTensorCell.bind((i,), ()) for i in range(4))
            return ReadonlyResourceBinding(ReadonlyTensorView(resource, cells), owner)

        def bind_context(self, resource, owner):
            decoder = self
            class ContextView:
                namespace = id(decoder)
                def validate(self):
                    assert len(decoder.rows) <= self.resource.domain[0]
                def check(self, mode, ranges):
                    self.validate()
                    assert len(ranges) == 1 and 0 <= ranges[0][0] <= ranges[0][1] <= len(decoder.rows)
            view = ContextView()
            view.resource = resource
            return HostResourceBinding(view, owner)

        def layer(self, row, ordinal, positions, hidden, residual):
            # A read-only final normalization can legally move after replay.
            # Observe the second call to a source layer, not normalization's
            # position in the reference program.
            if (row, ordinal) in self.seen_layers and row not in self.replaying:
                self.calls.append((row.packed_row_index, "replay"))
                self.replaying.add(row)
            self.seen_layers.add((row, ordinal))
            slot = self.cache[row.request_ordinal, ordinal]
            previous = slot[row.token_offset - 1] if row.token_offset else torch.zeros_like(hidden)
            slot[row.token_offset] = hidden
            output = torch.tanh(hidden * (ordinal + 1) * 0.8 + previous * 0.6)
            return output, torch.zeros_like(output)

        def begin(self, row):
            self.calls.append((row.packed_row_index, "normal"))
            assert row not in self.saved
            self.saved[row] = self.cache[row.request_ordinal, :, row.token_offset].clone()
            return torch.tensor([row.token_offset]), inputs[row.packed_row_index:row.packed_row_index + 1].clone()

        def normalize(self, row, hidden, residual):
            self.normalized.add(row)
            return hidden + residual

        def commit(self, row, record):
            if row in self.replaying and self.fail:
                raise RuntimeError("replay kernel failed")
            if row in self.replaying and self.cancel is not None:
                self.cancel.set()
            record()
            del self.saved[row]
            self.normalized.discard(row)
            self.replaying.discard(row)
            self.seen_layers.difference_update((row, layer) for layer in range(self.cache.shape[1]))

        def restore(self):
            for row, saved in reversed(tuple(self.saved.items())):
                self.cache[row.request_ordinal, :, row.token_offset].copy_(saved)
            self.saved.clear()
            self.normalized.clear()
            self.replaying.clear()
            self.seen_layers.clear()

        def finish(self, outputs):
            self.output = torch.cat(outputs)

    expected_cache = torch.zeros(2, 3, 3, 4)
    expected_output = []
    for index in range(4):
        request, position = int(index == 3), index if index < 3 else 0
        hidden = inputs[index].clone()
        for layer in range(3):
            previous = expected_cache[request, layer, position - 1] if position else torch.zeros(4)
            expected_cache[request, layer, position] = hidden
            hidden = (hidden * (layer + 1) * 0.8 + previous * 0.6).tanh()
            if layer == 0:
                destination = hidden.clone()
        expected_output.append(hidden.clone())
        if index < 3:
            alpha = 0.7 * min(position, 2) / 2
            hidden = destination * (1 - alpha) + hidden * alpha * (destination.square().sum() / hidden.square().sum()).sqrt()
            for layer in (1, 2):
                previous = expected_cache[request, layer, position - 1] if position else torch.zeros(4)
                expected_cache[request, layer, position] = hidden
                hidden = (hidden * (layer + 1) * 0.8 + previous * 0.6).tanh()

    assert execution.admit(identities) == ()
    assert execution.arena.pool_addresses == () and execution.arena.active_lineages == 0
    public_trace, rng = execution.trace.snapshot(), torch.random.get_rng_state().clone()
    proposals = compile_tokenwise_region(TensorType((1, 4), "torch.float32"), 8,
        probe.causal_domain, config.effect(3)).candidate_proposals
    tuning = TokenwiseTuning(execution, TOKENWISE_REPLAY_IMPLEMENTATION, proposals)
    samples = []
    changed = False
    try:
        assert tuning.count > 1
        for candidate in range(tuning.count):
            tuning.candidate(candidate)
            decoder = Decoder()
            decoder.batch = replace(rows, total_rows=3, rows=rows.rows[:3])
            sample = TensorStepInput(decoder.batch, PackedSamplerRowBatch(decoder.batch, (PackedSamplerRowLink(0, 2),)))
            tuning.record_rows(decoder.rows)
            started = perf_counter_ns()
            invocation = execution.start_step(sample)
            invocation.context.replay = decoder
            for anchor in range(8):
                execution.execute_anchor(invocation, AnchorId(anchor))
                if anchor == 3:
                    execution.complete_sampler(invocation)
            execution.finish_step(invocation)
            samples.append((perf_counter_ns() - started,))
            torch.testing.assert_close(decoder.output, torch.stack(expected_output[:3]))
            torch.testing.assert_close(decoder.cache[0], expected_cache[0])
            assert torch.count_nonzero(decoder.cache[1]) == 0 and not decoder.saved
            changed |= decoder.calls != [(i, phase) for i in range(3) for phase in ("normal", "replay")]
        tuning.finish(tuple(samples))
    finally:
        tuning.abort()
    assert changed and execution.trace.snapshot() == public_trace
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    decoder = Decoder()
    invocation = execution.start_step(value)
    invocation.context.replay = decoder
    for anchor in range(8):
        execution.execute_anchor(invocation, AnchorId(anchor))
        if anchor == 3:
            execution.complete_sampler(invocation)
    assert execution.finish_step(invocation) is None
    torch.testing.assert_close(decoder.output, torch.stack(expected_output))
    torch.testing.assert_close(decoder.cache, expected_cache)
    assert sorted(decoder.calls) == sorted([(i, phase) for i in range(3) for phase in ("normal", "replay")] + [(3, "normal")])
    assert all(decoder.calls.index((i, "normal")) < decoder.calls.index((i, "replay")) for i in range(3))
    replay = tuple(event for event in execution.trace.snapshot().events
        if event.invocation_id == invocation.invocation_id and event.phase is TracePhase.REPLAY)
    assert [event.row for event in replay] == list(rows.rows[:3])
    assert all(not event.refs for event in replay)
    assert invocation.context.publication is None
    for fail in (True, False):
        decoder = Decoder()
        invocation = execution.start_step(value)
        invocation.context.replay = decoder
        decoder.fail = fail
        decoder.cancel = None if fail else invocation.context.cancel_requested
        execution.execute_anchor(invocation, AnchorId(0))
        with pytest.raises(RuntimeError, match="failed" if fail else "cancelled"):
            execution.execute_anchor(invocation, AnchorId(1))
        assert not execution._active
        if fail:
            assert torch.count_nonzero(decoder.cache) == 0
        else:
            torch.testing.assert_close(decoder.cache[0, :, 0], expected_cache[0, :, 0])
            assert torch.count_nonzero(decoder.cache[:, :, 1:]) == 0
    assert execution.retire(identities) == ()
    assert execution.diagnostics().pending_state_releases == execution.diagnostics().fallback_count == 0
    disabled = RecirculationWorkload(replace(config, enabled=False)).compile_backend(facts).execution
    decoder = Decoder()
    invocation = disabled.start_step(value)
    invocation.context.replay = decoder
    for anchor in range(8):
        disabled.execute_anchor(invocation, AnchorId(anchor))
        if anchor == 3:
            disabled.complete_sampler(invocation)
    disabled.finish_step(invocation)
    assert decoder.calls == [(index, "normal") for index in range(4)]
    assert not any(event.phase is TracePhase.REPLAY for event in disabled.trace.snapshot().events)
    # Request zero has no rows this time. The same cold source must carry the
    # original effects through its empty scan and start request one's row at 0.
    decoder = Decoder()
    decoder.batch = replace(rows, total_rows=1, rows=(replace(rows.rows[-1], packed_row_index=0),))
    single = TensorStepInput(decoder.batch, PackedSamplerRowBatch(decoder.batch, (PackedSamplerRowLink(0, 0),)))
    invocation = disabled.start_step(single)
    invocation.context.replay = decoder
    for anchor in range(8):
        disabled.execute_anchor(invocation, AnchorId(anchor))
        if anchor == 3:
            disabled.complete_sampler(invocation)
    disabled.finish_step(invocation)
    expected = inputs[:1].clone()
    for ordinal in range(3):
        expected = torch.tanh(expected * (ordinal + 1) * 0.8)
    torch.testing.assert_close(decoder.output, expected)
    assert decoder.calls == [(0, "normal")] and torch.count_nonzero(decoder.cache[0]) == 0
    destination = torch.tensor([[3.0, 4.0]])
    torch.testing.assert_close(mix_layer_outputs(RecirculationConfig().effect(3), torch.zeros_like(destination), destination, 0),
        destination * 0.85)
    for invalid in (replace(config, source_layer=0), replace(config, destination_layer=3)):
        with pytest.raises(ValueError):
            RecirculationWorkload(invalid).compile_backend(facts)

"""Evidence 27: serial recurrence oracle, row selection and stateless settlement."""

from dataclasses import replace

import pytest
import torch
import pllm
from pllm.execution.trace import TracePhase

from pllm.authoring import BackendCompileFacts, TensorStepInput
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.contracts.compilation import PlanBundle
from pllm.execution.replay import TokenwiseReplayPrimitive, mix_layer_outputs
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.recirculation import RecirculationConfig, RecirculationWorkload
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType


def test_recirculation_selects_rows_and_replays_typed_effect() -> None:
    probe = BackendPlacementProbe("test.replay@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CPU, 0, 1, 1,
        TensorDType.FLOAT32, 4, 8, (), ("model/tokenwise_replay@1",))
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
    from pllm.worker import encode_workload
    restored = decode_boundary(encode_workload(RecirculationWorkload(config)), RecirculationConfig)
    assert RecirculationWorkload(restored).declare(facts) == RecirculationWorkload(config).declare(facts)
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
            self.cache = torch.zeros(2, 3, 3, 4)
            self.saved = None
            self.output = None
            self.fail = False
            self.cancel = None
            self.calls = []

        @property
        def rows(self):
            return rows.rows

        def layer(self, row, layer, hidden):
            slot = self.cache[row.request_ordinal, layer]
            previous = slot[row.token_offset - 1] if row.token_offset else torch.zeros_like(hidden)
            slot[row.token_offset] = hidden
            return torch.tanh(hidden * (layer + 1) * 0.8 + previous * 0.6)

        def normal(self, row):
            self.calls.append((row.packed_row_index, "normal"))
            self.saved = self.cache.clone()
            hidden = inputs[row.packed_row_index].clone()
            for layer in range(3):
                hidden = self.layer(row, layer, hidden)
                if layer == 0:
                    destination = hidden.clone()
            return hidden.clone(), hidden.clone(), destination

        def replay(self, row, hidden):
            self.calls.append((row.packed_row_index, "replay"))
            for layer in (1, 2):
                hidden = self.layer(row, layer, hidden)
            if self.fail:
                raise RuntimeError("replay kernel failed")
            if self.cancel is not None:
                self.cancel.set()

        def restore(self):
            self.cache.copy_(self.saved)

        def finish(self, outputs):
            self.output = torch.stack(outputs)

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
    assert decoder.calls == [(0, "normal"), (0, "replay"), (1, "normal"), (1, "replay"), (2, "normal"), (2, "replay"), (3, "normal")]
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
    destination = torch.tensor([[3.0, 4.0]])
    torch.testing.assert_close(mix_layer_outputs(RecirculationConfig().effect(3), torch.zeros_like(destination), destination, 0),
        destination * 0.85)
    for invalid in (replace(config, source_layer=0), replace(config, destination_layer=3)):
        with pytest.raises(ValueError):
            RecirculationWorkload(invalid).compile_backend(facts)

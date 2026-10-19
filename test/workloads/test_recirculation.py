"""Evidence 27: serial recurrence oracle, row selection and stateless settlement."""

from dataclasses import replace
from functools import cache
from time import perf_counter_ns
from types import SimpleNamespace

import pytest
import torch

import pllm
from pllm.authoring import BackendCompileFacts, TensorStepInput
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import (
    ActivationPatchBinding,
    ActivationPatchOperator,
    AttachmentGraphMode,
    AttachmentTopology,
    BackendPlacementProbe,
    CausalModelDomain,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.region_program import TensorType
from pllm.contracts.region_selection import RegionSelectionPolicy
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import (
    AnchorId,
    StateEffectKind,
    TaskResourceFact,
    TaskSymbol,
    ValueResourceFact,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType
from pllm.execution.backend_attachment import (
    attachment_buffer_bytes,
    bind_backend_attachment,
    derive_attachment_resources,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.declaration import Dimensions, Field, When
from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.region_resources import HostResourceBinding, PagedResourceBinding, ReadonlyResourceBinding
from pllm.execution.replay import (
    TOKENWISE_REPLAY_IMPLEMENTATION,
    TokenwiseReplayPrimitive,
    TokenwiseReplayTask,
    mix_layer_outputs,
)
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.state_execution import MathBinding
from pllm.execution.tensor_binding import TensorBinding
from pllm.execution.tensor_borrow import ReadonlyTensorCell, ReadonlyTensorView
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import FunctionMath, TaskOutput
from pllm.execution.tokenwise_region import compile_tokenwise_region
from pllm.execution.tokenwise_tuning import TokenwiseTuning
from pllm.execution.trace import TracePhase
from pllm.workloads.recirculation import RecirculationConfig, RecirculationWorkload

PROBE = BackendPlacementProbe(
    "test.replay@1",
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
    (),
    ("model/tokenwise_replay@1",),
    causal_domain=CausalModelDomain(2, 3, 8),
)
FACTS = BackendCompileFacts(
    PROBE, RuntimeFacts(torch.device("cpu")).timeline(("model/tokenwise_replay@1",)), (), 2, 0, 0, layer_count=3
)
CONFIG = RecirculationConfig(alpha=0.7, beta=None, ramp_tokens=2)
IDENTITIES = (RequestIdentity("prefill", 0, 0), RequestIdentity("decode", 1, 0))
ROWS = PackedPhaseRowBatch(
    1,
    1,
    4,
    tuple(
        PackedPhaseRow(
            i,
            int(i == 3),
            i if i < 3 else 0,
            HiddenPhase.PREFILL if i < 3 else HiddenPhase.DECODE,
            IDENTITIES[int(i == 3)],
        )
        for i in range(4)
    ),
)
VALUE = TensorStepInput(ROWS, PackedSamplerRowBatch(ROWS, (PackedSamplerRowLink(0, 2), PackedSamplerRowLink(1, 3))))
INPUTS = torch.tensor([[0.4, -0.1, 0.3, 0.7], [0.2, 0.9, -0.3, 0.8], [-0.2, 0.1, 0.8, 0.4], [0.7, 0.2, -0.1, 0.3]])


class Decoder(TokenwiseReplayPrimitive):
    def __init__(self):
        self.batch = ROWS
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
        layers = tuple(
            PagedLayer(str(i), self.cache[:, i].unsqueeze(1), table, slots) for i in range(self.cache.shape[1])
        )
        plan = bind_paged_access(self.rows, layers)
        return PagedResourceBinding(
            PagedResourceView(resource, plan, tuple((i, str(i)) for i in range(len(layers)))), owner
        )

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
        return torch.tensor([row.token_offset]), INPUTS[row.packed_row_index : row.packed_row_index + 1].clone()

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


def serial_oracle():
    """Independent serial recurrence over the four packed rows."""
    expected_cache = torch.zeros(2, 3, 3, 4)
    expected_output = []
    for index in range(4):
        request, position = int(index == 3), index if index < 3 else 0
        hidden = INPUTS[index].clone()
        for layer in range(3):
            previous = expected_cache[request, layer, position - 1] if position else torch.zeros(4)
            expected_cache[request, layer, position] = hidden
            hidden = (hidden * (layer + 1) * 0.8 + previous * 0.6).tanh()
            if layer == 0:
                destination = hidden.clone()
        expected_output.append(hidden.clone())
        if index < 3:
            alpha = 0.7 * min(position, 2) / 2
            hidden = (
                destination * (1 - alpha) + hidden * alpha * (destination.square().sum() / hidden.square().sum()).sqrt()
            )
            for layer in (1, 2):
                previous = expected_cache[request, layer, position - 1] if position else torch.zeros(4)
                expected_cache[request, layer, position] = hidden
                hidden = (hidden * (layer + 1) * 0.8 + previous * 0.6).tanh()
    return expected_cache, expected_output


def run_step(execution, value, decoder):
    invocation = execution.start_step(value)
    invocation.context.replay = decoder
    for anchor in range(8):
        execution.execute_anchor(invocation, AnchorId(anchor))
        if anchor == 3:
            execution.complete_sampler(invocation)
    return invocation, execution.finish_step(invocation)


@cache
def measured_plan_execution():
    """Shared by read-only plan checks; stateful tests compile their own."""
    return measured_execution()


def measured_execution():
    workload = RecirculationWorkload(CONFIG)
    return workload.compile_backend(workload.specialize(FACTS, ExecutionOptions(tuning="measured"))).execution


def test_recirculation_declaration_fits_reference_budget_and_crosses_the_wire() -> None:
    assert RecirculationConfig is pllm.Recurrence

    class PlainRecurrence(pllm.Workload):
        @property
        def recurrence(self) -> pllm.Recurrence:
            return CONFIG

    plain = PlainRecurrence().compile_backend(FACTS).execution
    execution = measured_plan_execution()
    assert plain.arena.plan.semantic_graph.tasks == execution.arena.plan.semantic_graph.tasks
    # A budget that fits only the reference must still install an auto-policy
    # model; expensive windows must not make the whole attachment inadmissible.
    workload = RecirculationWorkload(CONFIG)
    serial_facts = workload.specialize(FACTS, ExecutionOptions(schedule_policy="serial"))
    reference_budget = (
        max(serial_facts.timeline.reserved_bytes)
        + sum(f.workspace_bytes + f.resident_bytes for f in serial_facts.timeline.task_resources)
        + sum(f.bytes for f in serial_facts.timeline.value_resources)
    )
    tight_facts = workload.specialize(FACTS, ExecutionOptions(memory_budget_bytes=reference_budget, tuning="measured"))
    tight = workload.compile_backend(tight_facts)
    task = TokenwiseReplayTask.from_implementation(tight.execution.implementation(TOKENWISE_REPLAY_IMPLEMENTATION))
    assert task.policy.schedule_policy == "auto"
    assert all(
        all(d.bytes <= task.region.payload_limits[0].bytes for d in p.peak_bytes)
        for p in task.region.candidate_proposals
    )
    assert tight.attachment.inputs.plan.timeline.memory_budget_bytes == reference_budget
    with pytest.raises(ValueError, match="reference payload exceeds"):
        workload.specialize(FACTS, ExecutionOptions(memory_budget_bytes=reference_budget - 1))
    from pllm.workload_transport import decode_workload, encode_workload

    restored = decode_workload(encode_workload(workload), device=torch.device("cpu"), seed=0)
    assert isinstance(restored, RecirculationWorkload)
    assert restored.declare(FACTS) == workload.declare(FACTS)
    assert decode_boundary(encode_boundary(execution.arena.plan), PlanBundle) == execution.arena.plan
    for invalid in (replace(CONFIG, source_layer=0), replace(CONFIG, destination_layer=3)):
        with pytest.raises(ValueError):
            RecirculationWorkload(invalid).compile_backend(FACTS)
    destination = torch.tensor([[3.0, 4.0]])
    torch.testing.assert_close(
        mix_layer_outputs(RecirculationConfig().effect(3), torch.zeros_like(destination), destination, 0),
        destination * 0.85,
    )


def test_recirculation_replay_matches_serial_oracle_under_every_tuned_candidate() -> None:
    execution = measured_execution()
    expected_cache, expected_output = serial_oracle()
    assert execution.admit(IDENTITIES) == ()
    assert execution.arena.active_lineages == 0
    public_trace, rng = execution.trace.snapshot(), torch.random.get_rng_state().clone()
    proposals = compile_tokenwise_region(
        TensorType((1, 4), "torch.float32"),
        8,
        PROBE.causal_domain,
        CONFIG.effect(3),
        RegionSelectionPolicy(tuning="measured"),
    ).candidate_proposals
    tuning = TokenwiseTuning(execution, TOKENWISE_REPLAY_IMPLEMENTATION, proposals)
    samples = []
    changed = False
    try:
        assert tuning.count > 1
        for candidate in range(tuning.count):
            tuning.candidate(candidate)
            decoder = Decoder()
            decoder.batch = replace(ROWS, total_rows=3, rows=ROWS.rows[:3])
            sample = TensorStepInput(decoder.batch, PackedSamplerRowBatch(decoder.batch, (PackedSamplerRowLink(0, 2),)))
            tuning.record_rows(decoder.rows)
            started = perf_counter_ns()
            run_step(execution, sample, decoder)
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
    invocation, published = run_step(execution, VALUE, decoder)
    assert published is None
    torch.testing.assert_close(decoder.output, torch.stack(expected_output))
    torch.testing.assert_close(decoder.cache, expected_cache)
    assert sorted(decoder.calls) == sorted(
        [(i, phase) for i in range(3) for phase in ("normal", "replay")] + [(3, "normal")]
    )
    assert all(decoder.calls.index((i, "normal")) < decoder.calls.index((i, "replay")) for i in range(3))
    replay = tuple(
        event
        for event in execution.trace.snapshot().events
        if event.invocation_id == invocation.invocation_id and event.phase is TracePhase.REPLAY
    )
    assert [event.row for event in replay] == list(ROWS.rows[:3])
    assert all(not event.refs for event in replay)
    for fail in (True, False):
        decoder = Decoder()
        invocation = execution.start_step(VALUE)
        invocation.context.replay = decoder
        decoder.fail = fail
        decoder.cancel = None if fail else invocation.context.cancel_requested
        execution.execute_anchor(invocation, AnchorId(0))
        with pytest.raises(RuntimeError, match="failed" if fail else "cancelled"):
            execution.execute_anchor(invocation, AnchorId(1))
        if fail:
            assert torch.count_nonzero(decoder.cache) == 0
        else:
            torch.testing.assert_close(decoder.cache[0, :, 0], expected_cache[0, :, 0])
            assert torch.count_nonzero(decoder.cache[:, :, 1:]) == 0
    assert execution.retire(IDENTITIES) == ()
    assert execution.diagnostics().pending_state_releases == execution.diagnostics().fallback_count == 0


def test_disabled_recirculation_runs_only_the_normal_pass() -> None:
    disabled = RecirculationWorkload(replace(CONFIG, enabled=False)).compile_backend(FACTS).execution
    decoder = Decoder()
    run_step(disabled, VALUE, decoder)
    assert decoder.calls == [(index, "normal") for index in range(4)]
    assert not any(event.phase is TracePhase.REPLAY for event in disabled.trace.snapshot().events)
    # Request zero has no rows this time. The same cold source must carry the
    # original effects through its empty scan and start request one's row at 0.
    decoder = Decoder()
    decoder.batch = replace(ROWS, total_rows=1, rows=(replace(ROWS.rows[-1], packed_row_index=0),))
    single = TensorStepInput(decoder.batch, PackedSamplerRowBatch(decoder.batch, (PackedSamplerRowLink(0, 0),)))
    run_step(disabled, single, decoder)
    expected = INPUTS[:1].clone()
    for ordinal in range(3):
        expected = torch.tanh(expected * (ordinal + 1) * 0.8)
    torch.testing.assert_close(decoder.output, expected)
    assert decoder.calls == [(0, "normal")] and torch.count_nonzero(decoder.cache[0]) == 0


def test_activation_source_matches_serial_oracle() -> None:
    _activation_source_matches_serial_oracle(Decoder, VALUE, INPUTS, CONFIG, PROBE)


def _activation_source_matches_serial_oracle(decoder_type, value, inputs, config, probe):
    from pllm.backends.vllm.activation import VllmActivationPatch

    delta = torch.tensor([0.2, -0.3, 0.4, 0.1])
    weight = torch.tensor([[0.1, 0.2, -0.1, 0.3], [-0.2, 0.1, 0.4, 0.2]])
    bias = torch.tensor([0.07, -0.04])
    rotation = torch.tensor([[0.2, 0.0], [0.1, 0.3], [-0.1, 0.2], [0.4, -0.1]])
    tensors = ((delta,), (weight, bias, rotation))
    bindings = tuple(
        ActivationPatchBinding(
            layer,
            tuple(LayoutFieldId(i + offset) for i in range(len(fields))),
            1,
            1,
            operator,
            DecodeWindow.ALL,
            DecodeRefresh.STEP,
            MissingPolicy.HOLD,
        )
        for layer, operator, offset, fields in (
            (0, ActivationPatchOperator.ADD, 0, tensors[0]),
            (2, ActivationPatchOperator.LOW_RANK_AFFINE, 1, tensors[1]),
        )
    )
    b = TensorBinding("owned.activation.replay", "a" * 64, Dimensions(4, 0, layer_count=3))
    _, ports = b.parameters(
        "features", {name: (2,) for name in ("qp", "residual", "mv", "segmentation")}, optimizer=None
    )
    encoded = b.tensor("encode", inputs=ports, when=When.BEFORE_MODEL)(lambda values: sum(values.values()))
    fields = (delta, weight, bias, rotation)
    inject = b.state("inject", tuple(Field(str(i), tuple(tensor.shape)) for i, tensor in enumerate(fields)))

    def write(data, batch):
        return TaskOutput(
            field_writes=tuple((LayoutFieldId(i), tensor.unsqueeze(0)) for i, tensor in enumerate(fields)),
            changed=(0,),
            update_count=1,
        )

    update = b.compute(
        "write",
        inputs={"encoded": encoded["value"]},
        outputs=("updated",),
        changes=inject,
        effect=StateEffectKind.TRANSITION,
        when=When.BEFORE_MODEL,
        function=write,
    )
    publication = b.publish(update, inject, when=When.BEFORE_MODEL)
    b.overlay(inject, update, publication, activations=bindings)
    b.recurrence(config)
    b.conditional_input(("features",))
    graph, layouts, spec = b.finish()
    reserve = attachment_buffer_bytes(probe, graph, layouts, spec)
    timeline = replace(
        RuntimeFacts(torch.device("cpu")).timeline(("model/tokenwise_replay@1", "transition/full_root@1")),
        region_policy=RegionSelectionPolicy(schedule_policy="serial"),
        reserved_bytes=(reserve,) * 8,
        task_resources=(
            TaskResourceFact(TaskSymbol("encode"), 32),
            TaskResourceFact(TaskSymbol("publish"), 0),
            TaskResourceFact(TaskSymbol("write"), 512),
        ),
        value_resources=(
            ValueResourceFact(TaskSymbol("encode"), "value", 8),
            ValueResourceFact(TaskSymbol("publish"), "publication", 0),
            ValueResourceFact(TaskSymbol("write"), "updated", 0),
        ),
    )
    timeline = derive_attachment_resources(probe, graph, timeline, layouts=layouts, attachment=spec)
    plan = compile_page_bundle_plan(
        (graph, layouts), timeline, (PageBundleProviderFacts("test.owned@1", 3),), 1, 65536, torch.device("cpu")
    )
    signed = bind_backend_attachment(plan, probe, spec)
    math = FunctionMath(
        {"inject": lambda roots: None},
        lambda value: (),
        lambda value, batch: TensorFrame(value, value.rows.rows),
        lambda data, result: result.publication,
    )
    execution = MathBinding(math, b.bind_all(plan, model_probe=probe), borrowed_resources=("features",)).bind(
        PageBundleArena(plan, _owned_resources=("inject",)), signed.inputs.request.sampler_anchor_id
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(forward=lambda *args: None) for _ in range(3)])
    )
    selector = VllmActivationPatch(model, bindings, "inject", install=False)
    assert all(handle.layer.forward is handle.original for handle in selector.layers)

    class BoundDecoder(decoder_type):
        def bind_activation(self, resource, layer, owner):
            binding = selector.bind_pinned(resource, layer, owner)
            assert {proof.ref for proof in binding.view.proofs} == set(
                owner.pin.refs if owner.held_input else owner.publication.after_refs
            )
            return binding

        def activation_selected(self, row, layer):
            return selector.selected(row, layer)

    decoder = BoundDecoder()
    rows = decoder.rows
    expected_cache = torch.zeros_like(decoder.cache)
    expected = []
    for row in rows:
        index, request, position = row.packed_row_index, row.request_ordinal, row.token_offset

        def layer(ordinal, hidden):
            previous = expected_cache[request, ordinal, position - 1] if position else torch.zeros(4)
            expected_cache[request, ordinal, position] = hidden
            hidden = torch.tanh(hidden * (ordinal + 1) * 0.8 + previous * 0.6)
            if index != 1:
                if ordinal == 0:
                    hidden = hidden + delta
                elif ordinal == 2:
                    hidden = hidden + (hidden @ weight.T + bias - hidden @ rotation) @ rotation.T
            return hidden

        hidden = inputs[index].clone()
        for ordinal in range(3):
            hidden = layer(ordinal, hidden)
            if ordinal == 0:
                destination = hidden.clone()  # The tap must include this layer's injection.
        expected.append(hidden.clone())
        if row.phase is HiddenPhase.PREFILL:
            alpha = 0.7 * min(position, 2) / 2
            hidden = (
                destination * (1 - alpha) + hidden * alpha * (destination.square().sum() / hidden.square().sum()).sqrt()
            )
            for ordinal in (1, 2):
                hidden = layer(ordinal, hidden)  # New hidden, same affine parameters; no repeat of layer zero.
    execution.admit(())
    from test.workloads.test_video import received_features

    received = received_features(plan)
    root, peers = received.__enter__()
    execution.stage_readonly((root,))
    invocation = execution.start_step(value)
    execution.execute_anchor(invocation, AnchorId(0))
    identities = tuple(dict.fromkeys(row.identity for row in rows))
    execution.admit(identities)
    selector.prepare(value.rows, invocation.context, execution.directory)
    invocation.context.replay = decoder
    try:
        for anchor in range(1, 8):
            execution.execute_anchor(invocation, AnchorId(anchor))
            if anchor == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        selector.clear()
        execution.finish_step(invocation)
        torch.testing.assert_close(decoder.output, torch.stack(expected))
        torch.testing.assert_close(decoder.cache, expected_cache)
        assert execution.diagnostics().update_count == 1 and execution.diagnostics().fallback_count == 0
        assert all(
            peer.poll() and peer.recv_bytes() == encode_boundary(page.descriptor).encode()
            for peer, page in zip(peers, root.pages, strict=True)
        )
        # Re-run the same kernel oracle with a separately signed no-input invocation.
        # Normal/replay retain the exact version, but recompute affine delta from new hidden.
        decoder = BoundDecoder()
        execution.stage_hold(identities, "absent", step_id=1)
        held_rows = replace(value.rows, step_id=1)
        held_value = replace(value, rows=held_rows, sampler_rows=replace(value.sampler_rows, row_batch=held_rows))
        invocation = execution.start_step(held_value)
        execution.execute_anchor(invocation, AnchorId(0))
        assert invocation.context.publication is None and not invocation.context.results
        selector.prepare(held_rows, invocation.context, execution.directory)
        invocation.context.replay = decoder
        for anchor in range(1, 8):
            execution.execute_anchor(invocation, AnchorId(anchor))
            if anchor == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        selector.clear()
        execution.finish_step(invocation)
        torch.testing.assert_close(decoder.output, torch.stack(expected))
        torch.testing.assert_close(decoder.cache, expected_cache)
        assert execution.diagnostics().update_count == execution.diagnostics().hold_count == 1
        assert not any(peer.poll() for peer in peers)
    finally:
        selector.close()
        if execution._active:
            execution.cancel_step(invocation)
        execution.retire(identities)
        execution.retire(())
        received.__exit__(None, None, None)

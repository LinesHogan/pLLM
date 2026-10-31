"""Evidence 26: independent LoReFT math, explicit publication and no-update path."""

from dataclasses import dataclass, replace
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

import pllm
from pllm.authoring import StateTensorUpdate
from pllm.backends.vllm.activation import activation_delta
from pllm.contracts.backend_attachment import (
    ActivationPatchOperator,
    AttachmentGraphMode,
    AttachmentTopology,
    BackendPlacementProbe,
    DecodeRefresh,
    DecodeWindow,
    MissingPolicy,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import AnchorId, StateScope
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.step_input import TensorStepInput
from pllm.workloads.reft import ActivationConfig, ActivationWorkload, load_loreft_interventions


def test_reft_matches_loreft_math_and_publish_skip_semantics(tmp_path) -> None:
    device = torch.device("cpu")
    facts = BackendCompileFacts(
        BackendPlacementProbe(
            "test.activation@1",
            "a" * 64,
            "b" * 64,
            "c" * 64,
            AttachmentTopology.IN_PROCESS,
            AttachmentGraphMode.EAGER,
            StateDeviceKind.CPU,
            0,
            1,
            1,
            TensorDType.FLOAT64,
            4,
            8,
            (),
            (),
        ),
        model_timeline(device, ("transition/full_root@1",)),
        (PageBundleProviderFacts("test.page@1", 4),),
        2,
        65536,
        0,
    )
    workload = ActivationWorkload(ActivationConfig((0,), 2, first_tokens=1, last_tokens=1))
    deployment = workload.compile_backend(facts)
    execution = deployment.execution
    _decode_declarations_are_checked(facts)
    rows = PackedPhaseRowBatch(0, 0, ())
    value = TensorStepInput(rows, PackedSamplerRowBatch(rows, ()))
    (initial,) = execution.admit(())
    assert execution.step(value).publication is None
    identity = RequestIdentity("old", 0, 0)
    execution.admit((identity,))
    old = execution.directory.inference_pin(identity)
    generator = torch.Generator().manual_seed(23)
    fields = tuple(torch.randn(shape, dtype=torch.float64, generator=generator) for shape in ((2, 4), (2,), (4, 2)))
    checkpoint = tmp_path / "intkey_layer.0.comp.block_output.unit.pos.nunit.1#0.bin"
    torch.save(dict(weight=fields[0], bias=fields[1], rotate_layer=fields[2]), checkpoint)
    update = load_loreft_interventions(tmp_path, (0,), device, torch.float64)
    publication = execution.step(replace(value, auxiliary=update)).publication
    assert publication is not None and publication.after_refs[0].version_id == 1
    for actual, expected in zip(execution.snapshot(publication.after_refs[0]), fields, strict=True):
        torch.testing.assert_close(actual, expected)
    assert all(torch.count_nonzero(field) == 0 for field in execution.arena.snapshot(old, initial))
    assert execution.step(value).publication is None
    with pytest.raises(ValueError):  # A missing field does not match the declared layout.
        execution.step(replace(value, auxiliary=StateTensorUpdate(fields[:2])))
    assert execution.arena.latest(initial.lineage_id) == publication.after_refs[0]
    hidden = torch.randn((5, 4), dtype=torch.float64, generator=generator)
    weight, bias, rotation = fields
    expected = torch.stack(
        tuple(
            sum(
                (
                    sum(row[j] * weight[k, j] for j in range(4))
                    + bias[k]
                    - sum(row[j] * rotation[j, k] for j in range(4))
                )
                * rotation[:, k]
                for k in range(2)
            )
            for row in hidden
        )
    )
    torch.testing.assert_close(activation_delta(hidden, fields), expected)
    if find_spec("vllm") is not None:
        _decode_low_rank_keeps_parameters_and_recomputes_delta(
            deployment, value, update, identity, hidden, expected, bias, rotation
        )
    execution.retire((identity,))
    execution.retire(())
    assert execution.diagnostics().active_lineages == execution.diagnostics().inference_pins == 0
    if find_spec("vllm") is not None:
        _request_edits_share_publication_and_exact_prefill_rows(facts)


def _decode_low_rank_keeps_parameters_and_recomputes_delta(
    deployment, value, update, identity, hidden, expected, bias, rotation
):
    from pllm.backends.vllm.activation import VllmActivationPatch

    execution = deployment.execution
    (binding,) = deployment.attachment.spec.activation_bindings
    layer = SimpleNamespace(forward=lambda positions, hidden, residual: (hidden.clone(), residual))
    adapter = VllmActivationPatch(
        SimpleNamespace(model=SimpleNamespace(layers=[layer])),
        (replace(binding, decode=DecodeWindow.ALL, decode_refresh=DecodeRefresh.STEP),),
        deployment.attachment.spec.state_resource,
    )
    rows = PackedPhaseRowBatch(
        0, len(hidden), tuple(PackedPhaseRow(i, 0, i, HiddenPhase.DECODE, identity) for i in range(len(hidden)))
    )
    invocation = execution.start_step(replace(value, auxiliary=update))
    execution.execute_anchor(invocation, AnchorId(0))
    try:
        adapter.prepare(rows, invocation, execution.directory)
        actual, _ = layer.forward(torch.arange(len(hidden)), hidden, None)
        torch.testing.assert_close(actual, hidden + expected)
        # The same exact parameters must operate on each forward's current hidden.
        actual, _ = layer.forward(torch.arange(len(hidden)), hidden * 2, None)
        constant = sum(bias[k] * rotation[:, k] for k in range(len(bias)))
        torch.testing.assert_close(actual, 2 * hidden + 2 * expected - constant)
        adapter.clear()
        for anchor in execution.arena.plan.timeline.anchors[1:]:
            execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        execution.finish_step(invocation)
        missing = execution.start_step(value)
        execution.execute_anchor(missing, AnchorId(0))
        with pytest.raises(RuntimeError):  # Decode refresh needs a publication in this step.
            adapter.prepare(rows, missing, execution.directory)
        execution.cancel_step(missing)
    finally:
        adapter.close()


def _decode_declarations_are_checked(facts):
    legacy = pllm.model.Activation("intervention", (0,), 1, 1, ActivationPatchOperator.ADD)
    assert (legacy.decode, legacy.decode_refresh, legacy.on_missing) == (
        DecodeWindow.NONE,
        DecodeRefresh.INHERIT,
        MissingPolicy.STOP,
    )
    with pytest.raises(ValueError):  # Refresh is meaningless while the decode window is off.
        replace(legacy, decode_refresh=DecodeRefresh.STEP)

    @dataclass(frozen=True)
    class DeclaredActivation(pllm.Workload):
        patch: pllm.model.Activation

        @property
        def intervention(self) -> pllm.InstanceState:
            return pllm.InstanceState({"delta": pllm.Buffer((4,), dtype=torch.float32)}, imported=True)

    workload = DeclaredActivation(replace(legacy, decode=DecodeWindow.ALL, on_missing=MissingPolicy.HOLD))
    workload.plan_backend(*workload.specialize(facts, pllm.ExecutionOptions()))


def _request_edits_share_publication_and_exact_prefill_rows(facts) -> None:
    from pllm.backends.vllm.activation import VllmActivationPatch

    config = ActivationConfig(
        (0,), first_tokens=0, last_tokens=1, operator=ActivationPatchOperator.ADD, scope=StateScope.REQUEST
    )
    deployment = ActivationWorkload(config).compile_backend(facts)
    execution = deployment.execution
    identities = (RequestIdentity("a", 7, 1), RequestIdentity("b", 9, 1))
    initial = execution.admit(identities)
    old_pin = execution.arena.pin_exact(StateRefs.of(initial))
    rows = PackedPhaseRowBatch(
        0,
        3,
        tuple(
            PackedPhaseRow(index, 0 if index < 2 else 1, index if index < 2 else 0, HiddenPhase.PREFILL, identity)
            for index, identity in enumerate((identities[0], identities[0], identities[1]))
        ),
    )
    delta = torch.tensor([[-0.25, 0.5, -1.0, 0.75], [1.0, -2.0, 3.0, -4.0]])
    keys = tuple(RequestStateKey.project(identity, (RequestKeyField.PROMPT_IDX,)) for identity in reversed(identities))
    value = TensorStepInput(rows, PackedSamplerRowBatch(rows, ()), auxiliary=StateTensorUpdate((delta,), keys))
    invocation = execution.start_step(value)
    execution.execute_anchor(invocation, AnchorId(0))
    context = invocation

    class Layer(torch.nn.Module):
        def forward(self, positions, hidden_states, residual=None):
            # Like Qwen2's down projection, the layer owns a fresh output.
            self.output = hidden_states.clone()
            return self.output, residual

    layer = Layer()
    adapter = VllmActivationPatch(
        SimpleNamespace(model=SimpleNamespace(layers=[layer])),
        deployment.attachment.spec.activation_bindings,
        deployment.attachment.spec.state_resource,
    )
    try:
        adapter.prepare(rows, context, execution.directory)
        hidden = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        residual = torch.ones_like(hidden)
        expected = hidden.clone()
        expected[1] += delta[1].double()
        expected[2] += delta[0].double()
        patched, actual_residual = layer(torch.arange(3), hidden, residual)
        assert patched is layer.output
        torch.testing.assert_close(patched, expected, rtol=0, atol=0)
        assert actual_residual is residual
        assert torch.equal(hidden, torch.arange(12, dtype=torch.float64).reshape(3, 4))
        for ref in initial:
            assert torch.count_nonzero(execution.arena.snapshot(old_pin, ref)[0]) == 0
        adapter.clear()
        for anchor in range(1, 8):
            execution.execute_anchor(invocation, AnchorId(anchor))
            if anchor == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        assert execution.finish_step(invocation).publication is not None
        assert execution.step(value).publication is None, (
            "a request imports its edit once, even if the input batch remains staged"
        )
    finally:
        adapter.close()
        execution.arena.wait_release(execution.arena.release_pin(old_pin))
        execution.retire(identities)
    assert execution.diagnostics().active_lineages == 0

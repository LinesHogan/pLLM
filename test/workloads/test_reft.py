"""Evidence 26: independent LoReFT math, explicit publication and no-update path."""

from dataclasses import dataclass, replace
from importlib.util import find_spec
import json
from types import SimpleNamespace

import pytest
import torch
import pllm

from pllm.execution.activation import low_rank_affine_delta
from pllm.execution.backend_attachment import compile_backend_attachment
from pllm.execution.codec import encode_boundary
from test.revalidation import validate_backend_attachment
from pllm.authoring import BackendCompileFacts, StateTensorUpdate, TensorStepInput
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.reft import ActivationConfig, ActivationWorkload, load_loreft_interventions
from pllm.contracts.backend_attachment import ActivationPatchOperator, AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe, DecodeWindow, DecodeRefresh, MissingPolicy
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import AnchorId, StateScope
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from test.backend_facts import assert_auxiliary_preflight, assert_finite_resources


def test_reft_matches_loreft_math_and_publish_skip_semantics(tmp_path) -> None:
    device = torch.device("cpu")
    facts = BackendCompileFacts(BackendPlacementProbe("test.activation@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CPU, 0, 1, 1,
        TensorDType.FLOAT64, 4, 8, (), ()), RuntimeFacts(device).timeline(("transition/full_root@1",)),
        (PageBundleProviderFacts("test.page@1", 4),), 2, 65536, 0)
    workload = ActivationWorkload(ActivationConfig((0,), 2, first_tokens=1, last_tokens=1))
    deployment = workload.compile_backend(assert_finite_resources(workload, facts))
    execution, signed = deployment.execution, deployment.attachment
    _decode_declarations_require_resigning(facts, signed)
    assert_auxiliary_preflight(signed, StateTensorUpdate(tuple(torch.empty(shape, device="meta", dtype=torch.float64)
        for shape in ((2, 5), (2,), (4, 2)))))
    rows = PackedPhaseRowBatch(1, 0, 0, ())
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
    validate_auxiliary(update, signed.inputs.plan, SharedTensor.describe)
    assert_auxiliary_preflight(signed, replace(update, values=tuple(tensor.float() for tensor in update.values)))
    publication = execution.step(replace(value, auxiliary=update)).publication
    assert publication is not None and publication.after_refs[0].version_id == 1
    for actual, expected in zip(execution.snapshot(publication.after_refs[0]), fields, strict=True):
        torch.testing.assert_close(actual, expected)
    assert all(torch.count_nonzero(field) == 0 for field in execution.arena.snapshot(old, initial))
    assert execution.step(value).publication is None
    with pytest.raises(ValueError, match="layout"):
        execution.step(replace(value, auxiliary=StateTensorUpdate(fields[:2])))
    assert execution.arena.latest(initial.lineage_id) == publication.after_refs[0]
    hidden = torch.randn((5, 4), dtype=torch.float64, generator=generator)
    weight, bias, rotation = fields
    expected = torch.stack(tuple(sum((sum(row[j] * weight[k, j] for j in range(4)) + bias[k]
        - sum(row[j] * rotation[j, k] for j in range(4))) * rotation[:, k] for k in range(2)) for row in hidden))
    torch.testing.assert_close(low_rank_affine_delta(hidden, *fields), expected)
    binding, = signed.inputs.request.activation_bindings
    changed = replace(signed.inputs, request=replace(signed.inputs.request,
        activation_bindings=(replace(binding, last_tokens=2),)))
    assert compile_backend_attachment(changed) != signed.decision
    with pytest.raises(ValueError):
        validate_backend_attachment(changed, signed.decision)
    if find_spec("vllm") is not None:
        _decode_low_rank_keeps_parameters_and_recomputes_delta(deployment, value, update, identity, hidden, expected, bias, rotation)
    execution.retire((identity,))
    execution.retire(())
    diagnostics = execution.diagnostics()
    assert diagnostics.update_count == (2 if find_spec("vllm") is not None else 1)
    assert diagnostics.active_lineages == diagnostics.inference_pins == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
    if find_spec("vllm") is not None:
        _request_edits_share_publication_and_exact_prefill_rows(facts)


def _decode_low_rank_keeps_parameters_and_recomputes_delta(deployment, value, update, identity, hidden, expected, bias, rotation):
    from pllm.backends.vllm.activation import VllmActivationPatch
    execution = deployment.execution
    binding, = deployment.attachment.inputs.request.activation_bindings
    layer = SimpleNamespace(forward=lambda positions, hidden, residual: (hidden.clone(), residual))
    adapter = VllmActivationPatch(SimpleNamespace(model=SimpleNamespace(layers=[layer])),
        (replace(binding, decode=DecodeWindow.ALL, decode_refresh=DecodeRefresh.STEP),),
        deployment.attachment.inputs.request.state_resource)
    rows = PackedPhaseRowBatch(1, 0, len(hidden), tuple(PackedPhaseRow(i, 0, i, HiddenPhase.DECODE, identity)
        for i in range(len(hidden))))
    invocation = execution.start_step(replace(value, auxiliary=update))
    execution.execute_anchor(invocation, AnchorId(0))
    try:
        adapter.prepare(rows, invocation.context, execution.directory)
        actual, _ = layer.forward(torch.arange(len(hidden)), hidden, None)
        torch.testing.assert_close(actual, hidden + expected)
        # The same exact parameters must operate on each forward's current hidden.
        actual, _ = layer.forward(torch.arange(len(hidden)), hidden * 2, None)
        constant = sum(bias[k] * rotation[:, k] for k in range(len(bias)))
        torch.testing.assert_close(actual, 2 * hidden + 2 * expected - constant)
        assert invocation.context.published_pin.refs == invocation.context.publication.after_refs
        adapter.clear()
        for anchor in execution.arena.plan.timeline.anchors[1:]:
            execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        execution.finish_step(invocation)
        missing = execution.start_step(value)
        execution.execute_anchor(missing, AnchorId(0))
        with pytest.raises(RuntimeError, match="requires a new publication"):
            adapter.prepare(rows, missing.context, execution.directory)
        execution.cancel_step(missing)
    finally:
        adapter.close()


def _decode_declarations_require_resigning(facts, signed):
    legacy = pllm.Activation("intervention", (0,), 1, 1, ActivationPatchOperator.ADD)
    with pytest.raises(TypeError):
        pllm.Activation("intervention", (0,), 1, 1, ActivationPatchOperator.ADD, DecodeWindow.ALL)
    assert (legacy.decode, legacy.decode_refresh, legacy.on_missing) == (
        DecodeWindow.NONE, DecodeRefresh.INHERIT, MissingPolicy.STOP)
    for change in (dict(decode_refresh=DecodeRefresh.STEP), dict(on_missing=MissingPolicy.HOLD)):
        with pytest.raises(ValueError, match="decode window is off"):
            replace(legacy, **change)
        # Direct compiler callers cannot bypass the author-side check.
        binding, = signed.inputs.request.activation_bindings
        invalid = replace(signed.inputs, request=replace(signed.inputs.request,
            activation_bindings=(replace(binding, **change),)))
        with pytest.raises(ValueError, match="decode window is off"):
            compile_backend_attachment(invalid)

    @dataclass(frozen=True)
    class DeclaredActivation(pllm.Workload):
        patch: pllm.Activation

        @property
        def intervention(self) -> pllm.InstanceState:
            return pllm.InstanceState({"delta": pllm.Buffer((4,), dtype=torch.float32)})

    declarations = [legacy, replace(legacy, decode=DecodeWindow.ALL)]
    declarations.append(replace(declarations[-1], decode_refresh=DecodeRefresh.STEP))
    declarations.append(replace(declarations[-1], on_missing=MissingPolicy.HOLD))
    plans = tuple(DeclaredActivation(declaration).plan_backend(facts) for declaration in declarations)
    assert len({plan.inputs.plan.semantic_graph.execution_parameters_digest for plan in plans}) == 4
    assert len({plan.decision.signature for plan in plans}) == 4
    for declaration, plan in zip(declarations, plans, strict=True):
        binding, = plan.inputs.request.activation_bindings
        assert (binding.decode, binding.decode_refresh, binding.on_missing) == (
            declaration.decode, declaration.decode_refresh, declaration.on_missing)
    for previous, current in zip(plans, plans[1:]):
        # Changing only the attachment contract also requires a new signature.
        changed = replace(previous.inputs, request=current.inputs.request)
        assert compile_backend_attachment(changed) != previous.decision
        with pytest.raises(ValueError):
            validate_backend_attachment(changed, previous.decision)

    for field in ("decode", "decode_refresh", "on_missing"):
        payload = json.loads(encode_boundary(plans[-1].inputs))
        del payload["request"]["activation_bindings"][0][field]
        with pytest.raises(ValueError, match="missing field"):
            pllm._core.compile_backend_attachment(json.dumps(payload))
    payload = json.loads(encode_boundary(plans[-1].inputs))
    payload["request"]["activation_bindings"][0]["unknown_policy"] = "hold"
    with pytest.raises(ValueError, match="unknown field"):
        pllm._core.compile_backend_attachment(json.dumps(payload))
    # HOLD on the author face does not synthesize a no-input graph; that graph
    # is separately signed, and even an empty group list is mandatory on wire.
    assert plans[-1].inputs.plan.semantic_graph.input_groups == ()
    payload = json.loads(encode_boundary(plans[-1].inputs))
    del payload["plan"]["semantic_graph"]["input_groups"]
    with pytest.raises(ValueError, match="missing field.*input_groups"):
        pllm._core.compile_backend_attachment(json.dumps(payload))


def _request_edits_share_publication_and_exact_prefill_rows(facts) -> None:
    from pllm.backends.vllm.activation import VllmActivationPatch
    config = ActivationConfig((0,), first_tokens=0, last_tokens=1, operator=ActivationPatchOperator.ADD, scope=StateScope.REQUEST)
    deployment = ActivationWorkload(config).compile_backend(facts)
    execution = deployment.execution
    identities = (RequestIdentity("a", 7, 1), RequestIdentity("b", 9, 1))
    initial = execution.admit(identities)
    old_pin = execution.arena.pin_exact(initial)
    rows = PackedPhaseRowBatch(1, 0, 3, tuple(PackedPhaseRow(index, 0 if index < 2 else 1,
        index if index < 2 else 0, HiddenPhase.PREFILL, identity)
        for index, identity in enumerate((identities[0], identities[0], identities[1]))))
    delta = torch.tensor([[-0.25, 0.5, -1., 0.75], [1., -2., 3., -4.]])
    keys = tuple(RequestStateKey.project(identity, (RequestKeyField.PROMPT_IDX,)) for identity in reversed(identities))
    value = TensorStepInput(rows, PackedSamplerRowBatch(rows, ()), auxiliary=StateTensorUpdate((delta,), keys))
    # Queued checkpoint rows are not simultaneous arena admissions.
    serial = ActivationWorkload(config).compile_backend(replace(facts, max_admitted_lineages=1))
    validate_auxiliary(value.auxiliary, serial.attachment.inputs.plan, SharedTensor.describe)
    for index, identity in enumerate(identities):
        serial.execution.admit((identity,))
        one = PackedPhaseRowBatch(1, index, 1, (PackedPhaseRow(0, 0, 0, HiddenPhase.PREFILL, identity),))
        publication = serial.execution.step(TensorStepInput(one, PackedSamplerRowBatch(one, ()),
            auxiliary=value.auxiliary)).publication
        assert publication is not None and len(publication.after_refs) == 1
        torch.testing.assert_close(serial.execution.snapshot(publication.after_refs[0])[0], delta[1 - index])
        serial.execution.retire((identity,))
    assert serial.execution.diagnostics().update_count == 2 and serial.execution.arena.active_lineages == 0
    bad = delta.clone()
    bad[0, 0] = float("nan")
    capacity = execution.arena.free_pages
    with pytest.raises(ValueError, match="layout"):
        execution.step(replace(value, auxiliary=StateTensorUpdate((bad,), keys)))
    assert execution.arena.free_pages == capacity
    assert all(execution.arena.latest(ref.lineage_id) == ref for ref in initial)
    invocation = execution.start_step(value)
    execution.execute_anchor(invocation, AnchorId(0))
    context = invocation.context
    assert context.publication is not None and context.update_count == 2
    class Layer(torch.nn.Module):
        def forward(self, positions, hidden_states, residual=None):
            # Like Qwen2's down projection, the layer owns a fresh output.
            self.output = hidden_states.clone()
            return self.output, residual
    layer = Layer()
    adapter = VllmActivationPatch(SimpleNamespace(model=SimpleNamespace(layers=[layer])),
        deployment.attachment.inputs.request.activation_bindings, deployment.attachment.inputs.request.state_resource)
    try:
        adapter.prepare(rows, context, execution.directory)
        assert context.published_pin is not None and context.published_pin.refs == context.publication.after_refs
        hidden = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        residual = torch.ones_like(hidden)
        expected = hidden.clone()
        expected[1] += delta[1].double()
        expected[2] += delta[0].double()
        patched, actual_residual = layer(torch.arange(3), hidden, residual)
        assert patched is layer.output
        torch.testing.assert_close(patched, expected, rtol=0, atol=0)
        assert actual_residual is residual and adapter.apply_count == 2
        assert torch.equal(hidden, torch.arange(12, dtype=torch.float64).reshape(3, 4))
        for ref in initial:
            assert torch.count_nonzero(execution.arena.snapshot(old_pin, ref)[0]) == 0
        adapter.clear()
        for anchor in range(1, 8):
            execution.execute_anchor(invocation, AnchorId(anchor))
            if anchor == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        assert execution.finish_step(invocation).publication is context.publication
        assert context.published_release.pin is context.published_pin
        assert execution.step(value).publication is None, "a request imports its edit once, even if the input batch remains staged"
    finally:
        adapter.close()
        execution.arena.wait_release(execution.arena.release_pin(old_pin))
        execution.retire(identities)
    assert execution.diagnostics().active_lineages == execution.diagnostics().pending_state_releases == 0

"""Evidence 15: exact native binding to shared plans, layouts and backend facts."""

import json
from dataclasses import replace

import pytest
import torch

from tllm import _core
from tllm.canonical.backend_attachment import (
    _inputs_json, compile_backend_attachment, compile_parameter_overlay_attachment,
    compile_request_sampler_attachment, lower_backend_attachment,
    validate_backend_attachment, validate_backend_attachment_physical,
    validate_backend_attachment_reconstruction,
)
from tllm.canonical.ptts.dense_page_planning import DensePageCpuProviderFacts
from tllm.canonical.runtime_facts import CanonicalRuntimeFacts
from tllm.canonical.tensor_layout import intern_tensor_layouts
from tllm.canonical.workloads.esamp_request import ESampRequestWorkload
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions, ESampRequestConfig, ESampRequestPackedHiddenBindings,
    ESampRequestSpecialization,
)
from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTCompileOptions, InPlaceTTTConfig, InPlaceTTTSpecialization,
    InPlaceTTTTargetSpecialization, InPlaceTTTWorkload,
)
from tllm.contracts.backend_attachment import (
    AttachmentGraphMode, AttachmentHiddenResource, AttachmentTopology, BackendPlacementProbe,
)
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from tllm.contracts.state_representation import StateDeviceKind
from tllm.contracts.tensor_layout import TensorDType
from tllm.contracts.semantic_graph import (
    AnchorId, AnchorSymbol, ImplementationId, SchedulerTimeline, StateResource, TaskId, TimelineAnchor,
)


def test_signed_attachment_binds_exact_tasks_fields_stages_and_rejects_tamper() -> None:
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2))
    compiled = workload.compile(
        InPlaceTTTSpecialization(
            (InPlaceTTTTargetSpecialization(target, 3, 2),),
            TensorDType.FLOAT32, torch.device("cpu"),
        ),
        InPlaceTTTCompileOptions(
            CanonicalRuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",)),
            (DensePageCpuProviderFacts("test.attachment@1", 2),), 2, 4096,
        ),
    )
    probe = BackendPlacementProbe(
        "test.backend@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER,
        StateDeviceKind.CPU, 0, 1, 1, TensorDType.FLOAT32, 3, 8, (), (),
    )
    overlay = compile_parameter_overlay_attachment(
        compiled.plan_bundle, probe, compiled.state_layout.overlay_bindings,
        StateResource("dense_fast_weights"),
    )
    assert overlay.inputs.plan is compiled.plan_bundle
    assert overlay.inputs.probe is probe
    assert overlay.inputs.request.overlay_field_ids == (0,)
    hidden = ESampRequestPackedHiddenBindings(*(
        PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX,
            HiddenSite.BLOCK_OUTPUT, f"model.layers[{layer}]") for layer in (0, 1)
    ))
    esamp = ESampRequestWorkload(ESampRequestConfig(rank=2, source_layer=0, target_layer=1))
    esamp_compiled = esamp.compile(
        ESampRequestSpecialization(3, TensorDType.FLOAT32, 7, hidden, StateDeviceKind.CPU, 0),
        ESampRequestCompileOptions(SchedulerTimeline(tuple(
            TimelineAnchor(AnchorSymbol(name), index) for index, name in enumerate((
                "model.execute", "sampler.pre_draw", "token.commit", "next.prepare",
            ))), ("compute/cpu@1", "state/dense_page_cpu@1")),
            (DensePageCpuProviderFacts("test.attachment@1", 2),), 2, 4096),
    )
    sampler_probe = replace(probe, hidden_resources=tuple(
        AttachmentHiddenResource(binding, HiddenPhase.DECODE) for binding in hidden.resources
    ))
    sampler = compile_request_sampler_attachment(
        esamp_compiled.plan_bundle, sampler_probe, esamp.request_sampler_attachment_spec(),
    )
    assert sampler.inputs.plan is esamp_compiled.plan_bundle
    assert sampler.inputs.probe is sampler_probe

    for signed in (overlay, sampler):
        inputs, decision, physical = signed.inputs, signed.decision, signed.physical
        plan, request = inputs.plan, inputs.request
        assert compile_backend_attachment(inputs) == decision
        assert lower_backend_attachment(inputs, decision) == physical
        assert physical.schedule_signature == plan.schedule_decision.signature
        assert physical.state_representation_signature == plan.state_decision.signature
        validate_backend_attachment_physical(inputs, decision, physical)
        validate_backend_attachment_reconstruction(
            inputs, decision, physical, replace(inputs, plan=replace(plan)), replace(physical),
        )
        first, second = request.task_bindings
        invalid_requests = (
            replace(request, task_bindings=(replace(first, task_id=TaskId(999)), second)),
            replace(request, task_bindings=(replace(first, implementation_id=ImplementationId("other@1")), second)),
            replace(request, task_bindings=(replace(first, anchor_id=AnchorId(999)), second)),
            replace(request, sampler_anchor_id=AnchorId(999)),
            replace(request, state_resource="missing"),
            replace(request, schema_version=1),
        )
        for invalid in invalid_requests:
            with pytest.raises(ValueError):
                compile_backend_attachment(replace(inputs, request=invalid))
        for invalid_probe in (
            replace(inputs.probe, device_kind=StateDeviceKind.CUDA),
            replace(inputs.probe, max_packed_rows=0),
            replace(inputs.probe, tensor_parallel_size=2),
        ):
            with pytest.raises(ValueError):
                compile_backend_attachment(replace(inputs, probe=invalid_probe))
        # These are legal facts, but cannot replace the original signed placement.
        for changed_probe in (
            replace(inputs.probe, model_fingerprint="d" * 64),
            replace(inputs.probe, runner_fingerprint="e" * 64),
            replace(inputs.probe, backend_abi_digest="f" * 64),
            replace(inputs.probe, max_packed_rows=9),
            replace(inputs.probe, tensor_dtype=TensorDType.BFLOAT16),
        ):
            local = replace(inputs, probe=changed_probe)
            local_decision = compile_backend_attachment(local)
            assert local_decision != decision
            with pytest.raises(ValueError):
                validate_backend_attachment(local, decision)
            with pytest.raises(ValueError):
                validate_backend_attachment_reconstruction(inputs, decision, physical, local,
                    lower_backend_attachment(local, local_decision))
        for field, value in (("signature", "0" * 64), ("request_digest", "0" * 64), ("schema_version", 1)):
            with pytest.raises(ValueError):
                validate_backend_attachment(inputs, replace(decision, **{field: value}))
        for field, value in (("bundle_signature", "0" * 64), ("plan_epoch", 9), ("schema_version", 1)):
            with pytest.raises(ValueError):
                validate_backend_attachment_physical(inputs, decision, replace(physical, **{field: value}))
        for field in ("schedule_decision", "state_decision", "tensor_layout_decision"):
            altered = replace(getattr(plan, field), signature="0" * 64)
            with pytest.raises(ValueError):
                compile_backend_attachment(replace(inputs, plan=replace(plan, **{field: altered})))
        original_layout, = plan.tensor_layout_request.layouts
        other_layouts = replace(plan.tensor_layout_request, layouts=(replace(original_layout,
            layout=replace(original_layout.layout, namespace="other.layout@1")),))
        with pytest.raises(ValueError, match="state representation differs from signed tensor layout"):
            compile_backend_attachment(replace(inputs, plan=replace(plan,
                tensor_layout_request=other_layouts, tensor_layout_decision=intern_tensor_layouts(other_layouts))))
        wire = _inputs_json(inputs)
        for path, key in (((), "plan"), (("request",), "state_resource"),
                          (("probe",), "model_fingerprint"), (("plan",), "tensor_layout_request")):
            for missing in (True, False):
                payload = json.loads(wire)
                node = payload
                for part in path:
                    node = node[part]
                if missing:
                    del node[key]
                else:
                    node["unknown"] = 0
                with pytest.raises(ValueError):
                    _core.compile_backend_attachment(json.dumps(payload))
        with pytest.raises(ValueError):
            _core.compile_backend_attachment(wire.replace('"schema_version":2', '"schema_version":2,"schema_version":2', 1))

    request = overlay.inputs.request
    stages = request.overlay_stage_anchors
    for invalid in (
        replace(request, overlay_field_ids=(999,)),
        replace(request, overlay_field_ids=(0, 0)),
        replace(request, overlay_stage_anchors=stages[:-1]),
        replace(request, overlay_stage_anchors=(replace(stages[0], anchor_id=stages[-1].anchor_id,
            anchor_symbol=stages[-1].anchor_symbol), *stages[1:])),
        replace(request, overlay_stage_anchors=(*stages[:3], replace(stages[3],
            anchor_id=stages[4].anchor_id, anchor_symbol=stages[4].anchor_symbol), *stages[4:])),
    ):
        with pytest.raises(ValueError):
            compile_backend_attachment(replace(overlay.inputs, request=invalid))
    # Another in-layout field is a different signed choice, not an out-of-range field.
    changed_field = replace(overlay.inputs, request=replace(request, overlay_field_ids=(1,)))
    assert compile_backend_attachment(changed_field) != overlay.decision
    with pytest.raises(ValueError):
        validate_backend_attachment(changed_field, overlay.decision)
    resource, other = sampler_probe.hidden_resources
    for altered in (replace(resource, phase=HiddenPhase.PREFILL),
                    replace(resource, binding=replace(resource.binding, layer="-1"))):
        with pytest.raises(ValueError):
            compile_backend_attachment(replace(sampler.inputs,
                probe=replace(sampler_probe, hidden_resources=(altered, other))))
    changed_module = replace(sampler.inputs, probe=replace(sampler_probe, hidden_resources=(
        replace(resource, binding=replace(resource.binding, module_path="other.layer")), other,
    )))
    assert compile_backend_attachment(changed_module) != sampler.decision
    with pytest.raises(ValueError):
        validate_backend_attachment(changed_module, sampler.decision)
    old_wire = json.loads(_inputs_json(sampler.inputs))
    first_hidden = old_wire["probe"]["hidden_resources"][0]
    old_wire["probe"]["hidden_resources"][0] = {**first_hidden["binding"], "phase": first_hidden["phase"]}
    with pytest.raises(ValueError):
        _core.compile_backend_attachment(json.dumps(old_wire))

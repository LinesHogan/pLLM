"""Evidence 15: exact native binding to shared plans, layouts and backend facts."""

import json
from dataclasses import replace
from unittest.mock import patch
from typing import Annotated

import pytest
import torch
from pllm.authoring import BackendCompileFacts, ParameterTargetShape
from pllm.execution.tensor_binding import TensorBinding

from pllm import _core, Parameter
from pllm.execution.backend_attachment import (
    compile_backend_attachment,
    lower_backend_attachment,
    validate_backend_attachment_physical,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.page_bundle import PageBundleArena
from test.revalidation import validate_backend_attachment, validate_backend_attachment_reconstruction
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.scheduler import intern_tensor_layouts
from pllm.workloads.esamp_request import ESampRequestWorkload
from pllm.workloads.esamp_request import ESampRequestConfig
from pllm.workloads.inplace_ttt import (
    InPlaceTTTConfig, InPlaceTTTWorkload,
)
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode, AttachmentHiddenResource, AttachmentTopology, BackendPlacementProbe, SignedBackendAttachment,
)
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget, ParameterOverlayOperator
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.contracts.semantic_graph import (
    AnchorId, ImplementationId, TaskId, SemanticGraph, SemanticOutputKind, HostPayloadSchemaId,
)


def test_signed_attachment_binds_exact_tasks_fields_stages_and_rejects_tamper() -> None:
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2))
    probe = BackendPlacementProbe(
        "test.backend@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER,
        StateDeviceKind.CPU, 0, 1, 1, TensorDType.FLOAT32, 3, 8, (), (),
    )
    facts = BackendCompileFacts(probe,
        RuntimeFacts(torch.device("cpu")).timeline(("state/dense_page_cpu@1", "transition/full_root@1")),
        (PageBundleProviderFacts("test.attachment@1", 4),), 2, 4096, 7,
        (ParameterTargetShape(target, 3, 2),))
    with patch.object(PageBundleArena, "__init__", side_effect=AssertionError("planning allocated state")):
        planned = workload.plan_backend(facts)
    with patch.object(_core, "compile_state_schedule", side_effect=AssertionError("worker selected a plan")):
        overlay_deployment = workload.bind_backend(facts, planned)
    with pytest.raises(ValueError, match="mathematics"):
        InPlaceTTTWorkload(replace(workload.config, learning_rate=0.5)).bind_backend(facts, planned)
    annotations = dict(InPlaceTTTWorkload.transition.__annotations__,
        delta=Annotated[torch.Tensor, Parameter("dense_fast_weights", "tail_input")])
    with patch.object(InPlaceTTTWorkload.transition, "__annotations__", annotations):
        with pytest.raises(ValueError, match="mathematics"):
            workload.bind_backend(facts, planned)
    with pytest.raises(ValueError, match="worker backend facts"):
        workload.bind_backend(replace(facts, probe=replace(probe, topology=AttachmentTopology.ENGINE_CORE)), planned)
    overlay = overlay_deployment.attachment
    assert overlay.inputs.plan is overlay_deployment.execution.arena.plan
    assert overlay.inputs.probe is facts.probe
    binding, = overlay.inputs.request.overlay_bindings
    changed_target = replace(overlay.inputs, request=replace(overlay.inputs.request,
        overlay_bindings=(replace(binding, target=replace(binding.target, layer=1)),)))
    assert compile_backend_attachment(changed_target) != overlay.decision
    with pytest.raises(ValueError):
        validate_backend_attachment(changed_target, overlay.decision)
    esamp = ESampRequestWorkload(ESampRequestConfig(rank=2, source_layer=0, target_layer=1))
    sampler_probe = replace(probe, hidden_resources=tuple(
        AttachmentHiddenResource(PackedHiddenResourceBinding(str(source.layer), HiddenLayerKind.INDEX,
            source.site, f"model.layers[{source.layer}]"), source.phase)
        for source in esamp.backend_requirements().hidden_inputs))
    sampler_deployment = esamp.compile_backend(replace(facts, probe=sampler_probe, targets=()))
    sampler = sampler_deployment.attachment
    assert sampler.inputs.plan is sampler_deployment.execution.arena.plan
    assert sampler.inputs.probe is sampler_probe
    with pytest.raises(ValueError, match="composition"):
        compile_backend_attachment(replace(overlay.inputs, probe=sampler_probe))
    class UnconsumedSamplerWorkload(InPlaceTTTWorkload):
        def _build(self, b: TensorBinding) -> None:
            super()._build(b)
            graph = b.graph
            task, other = graph.tasks
            output, = task.outputs
            b.graph = replace(graph, tasks=(replace(task,
                outputs=(replace(output, kind=SemanticOutputKind.SAMPLER, schema=HostPayloadSchemaId("")),)), other))
    with pytest.raises(ValueError, match="composition"):
        UnconsumedSamplerWorkload(workload.config).compile_backend(facts)

    for signed in (overlay, sampler):
        reconstructed = decode_boundary(encode_boundary(signed), SignedBackendAttachment)
        assert reconstructed == signed
        validate_backend_attachment_physical(reconstructed.inputs, reconstructed.decision, reconstructed.physical)
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
        wire = encode_boundary(inputs)
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
            _core.compile_backend_attachment(wire.replace('"schema_version":7', '"schema_version":7,"schema_version":7', 1))

    request = overlay.inputs.request
    for invalid in (
        replace(request, overlay_bindings=(replace(binding, fields=(999,)),)),
        replace(request, overlay_bindings=(replace(binding, fields=(0, 0)),)),
        replace(request, overlay_bindings=(binding, binding)),
        replace(request, overlay_bindings=(replace(binding, operator=ParameterOverlayOperator.LOW_RANK_ADD),)),
    ):
        with pytest.raises(ValueError):
            compile_backend_attachment(replace(overlay.inputs, request=invalid))
    # Another in-layout field, operator arity or scale is a different signed choice, not tampering.
    for altered in (
        replace(request, overlay_bindings=(replace(binding, fields=(1,)),)),
        replace(request, overlay_bindings=(replace(binding, scale=2.0),)),
        replace(request, overlay_bindings=(replace(binding, operator=ParameterOverlayOperator.LOW_RANK_ADD,
            fields=(binding.fields[0], binding.fields[0] + 1)),)),
    ):
        changed = replace(overlay.inputs, request=altered)
        assert compile_backend_attachment(changed) != overlay.decision
        with pytest.raises(ValueError):
            validate_backend_attachment(changed, overlay.decision)
    resource, other = sampler_probe.hidden_resources
    for altered in (replace(resource, phase=HiddenPhase.PREFILL),
                    *(replace(resource, binding=replace(resource.binding, layer=layer))
                        for layer in ("-1", "01", str(1 << 63))),
                    replace(resource, binding=replace(resource.binding, module_path=""))):
        with pytest.raises(ValueError):
            compile_backend_attachment(replace(sampler.inputs,
                probe=replace(sampler_probe, hidden_resources=(altered, other))))
    changed_module = replace(sampler.inputs, probe=replace(sampler_probe, hidden_resources=(
        replace(resource, binding=replace(resource.binding, module_path="other.layer")), other,
    )))
    assert compile_backend_attachment(changed_module) != sampler.decision
    with pytest.raises(ValueError):
        validate_backend_attachment(changed_module, sampler.decision)
    old_wire = json.loads(encode_boundary(sampler.inputs))
    first_hidden = old_wire["probe"]["hidden_resources"][0]
    old_wire["probe"]["hidden_resources"][0] = {**first_hidden["binding"], "phase": first_hidden["phase"]}
    with pytest.raises(ValueError):
        _core.compile_backend_attachment(json.dumps(old_wire))

    # Factory failure must release an already-created engine, before a handle owns it.
    from importlib.util import find_spec
    if find_spec("vllm") is not None:
        import io
        import os
        from types import SimpleNamespace
        import pllm
        from pllm.backends.vllm import runtime
        for multiprocessing, message in ((False, "in-process client"), (True, "single-worker quorum")):
            with io.BytesIO(b"owned engine resource") as resource:
                engine = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=resource.close)),
                    collective_rpc=lambda *args, **kwargs: ())
                environment = {"VLLM_ENABLE_V1_MULTIPROCESSING": "1" if multiprocessing else "0",
                    "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
                with patch.dict(os.environ, environment), patch.object(runtime, "LLM", return_value=engine):
                    with pytest.raises(RuntimeError, match=message):
                        pllm.make_llm(pllm.ModelConfig("unused-local-model", multiprocessing=multiprocessing),
                            workload=ESampRequestWorkload())
                    assert resource.closed

        # A failed shutdown retains the real shared-input FD until close can retry.
        from weakref import ref
        from vllm import SamplingParams
        from pllm.execution.shared_tensor import CpuFileStorage
        staged = []
        shutdown_failure = [True]
        def rpc(*args, **kwargs):
            command, payload = kwargs["args"]
            if command == runtime.BackendCommand.STAGE.value:
                staged.append(decode_boundary(payload, runtime.BackendStage))
            return ("",)
        def fail(*args, **kwargs):
            raise RuntimeError("synthetic engine failure")
        def shutdown():
            if shutdown_failure[0]:
                raise RuntimeError("synthetic shutdown failure")
        engine = SimpleNamespace(owned=torch.empty(128), collective_rpc=rpc,
            llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown), abort_request=fail,
                output_processor=SimpleNamespace(request_states={"0": SimpleNamespace(external_req_id="0")})),
            sleep=lambda **kwargs: None, enqueue=lambda *args, **kwargs: ["0"],
            wake_up=lambda **kwargs: None, wait_for_completion=fail)
        owned = ref(engine.owned)
        handle = runtime.VllmLlm(engine, None, None, overlay, facts)
        del engine
        with pytest.raises(RuntimeError, match="synthetic engine failure"):
            handle.generate(["test"], SamplingParams(max_tokens=1), feedback_group="ownership")
        shared = staged[0].completion.storage
        assert isinstance(shared, CpuFileStorage)
        assert os.fstat(shared.descriptor).st_size == facts.probe.hidden_size * 4
        assert owned() is not None and handle._pending_input is not None
        with pytest.raises(RuntimeError, match="closed"):
            handle.generate(["test"], SamplingParams(max_tokens=1))
        shutdown_failure[0] = False
        handle.close()
        assert handle._llm is handle._runner is handle._deployment is handle._pending_input is None
        assert owned() is None
        with pytest.raises(OSError):
            os.fstat(shared.descriptor)
        handle.close()

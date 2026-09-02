"""Dense parameter overlays share the Rust-signed backend attachment boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tllm.canonical.backend_attachment import (
    compile_backend_attachment,
    lower_backend_attachment,
    validate_backend_attachment_physical,
    validate_backend_attachment_reconstruction,
)
from tllm.canonical.ptts import DensePageCpuProviderFacts
from tllm.canonical.runtime_facts import CanonicalRuntimeFacts
from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTSpecialization,
    InPlaceTTTTargetSpecialization,
    InPlaceTTTWorkload,
)
from tllm.contracts.backend_attachment import (
    AttachmentCleanupStage,
    AttachmentGraphMode,
    AttachmentKind,
    AttachmentOverlayStage,
    AttachmentOverlayStageAnchor,
    AttachmentSamplerStage,
    AttachmentTaskBinding,
    AttachmentTaskRole,
    AttachmentTopology,
    BackendAttachmentCompilerInputs,
    BackendAttachmentRequest,
    BackendPlacementProbe,
    StreamDependency,
)
from tllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)
from tllm.contracts.request_identity import RequestKeyField
from tllm.contracts.semantic_graph import AnchorId, AnchorSymbol, ImplementationId
from tllm.contracts.state_representation import StateDeviceKind
from tllm.contracts.tensor_layout import TensorDType


def _overlay_inputs() -> BackendAttachmentCompilerInputs:
    target = ParameterTarget(
        0,
        ParameterModuleKind.MLP,
        ParameterRole.DOWN_PROJECTION_WEIGHT,
    )
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2))
    timeline = CanonicalRuntimeFacts(torch.device("cpu")).timeline(
        ("transition/full_root@1",)
    )
    compiled = workload.compile(
        InPlaceTTTSpecialization(
            (InPlaceTTTTargetSpecialization(target, 3, 2),),
            TensorDType.FLOAT32,
            torch.device("cpu"),
        ),
        InPlaceTTTCompileOptions(
            timeline,
            (DensePageCpuProviderFacts("test.full_root@1", 2),),
            1,
            4096,
        ),
    )
    plan = compiled.plan_bundle
    anchors = {anchor.symbol: anchor for anchor in plan.timeline.anchors}
    scheduled = {
        str(task.task_symbol): task for task in plan.schedule_decision.tasks
    }
    physical = {task.task_id: task for task in plan.schedule_physical.tasks}
    task_bindings = tuple(
        AttachmentTaskBinding(
            role,
            scheduled[symbol].task_id,
            physical[scheduled[symbol].task_id].implementation_id,
            physical[scheduled[symbol].task_id].anchor_id,
        )
        for role, symbol in (
            (AttachmentTaskRole.TRANSITION, "transition"),
            (AttachmentTaskRole.PUBLISH, "publish"),
        )
    )
    stage_symbols = (
        (AttachmentOverlayStage.PIN, AnchorSymbol("request.prepare")),
        (AttachmentOverlayStage.APPLY, AnchorSymbol("model.execute")),
        (AttachmentOverlayStage.CAPTURE, AnchorSymbol("model.execute")),
        (AttachmentOverlayStage.TRANSITION, AnchorSymbol("model.execute")),
        (AttachmentOverlayStage.PUBLISH, AnchorSymbol("state.publish")),
        (AttachmentOverlayStage.RELEASE, AnchorSymbol("sampler.complete")),
        (AttachmentOverlayStage.FINISH, AnchorSymbol("request.finish")),
    )
    stage_anchors = tuple(
        AttachmentOverlayStageAnchor(
            stage,
            AnchorId(anchors[symbol].ordinal),
            symbol,
        )
        for stage, symbol in stage_symbols
    )
    fields = tuple(
        sorted(
            int(field)
            for binding in compiled.state_layout.overlay_bindings
            for field in binding.fields
        )
    )
    capabilities = (
        "backend/vllm_mrv2@1",
        "operand_capture/mlp_down_projection_pair@1",
        "overlay/dense_add@1",
    )
    request = BackendAttachmentRequest(
        adapter_id="canonical.vllm_mrv2@1",
        attachment_kind=AttachmentKind.DENSE_PARAMETER_OVERLAY,
        backend_abi_digest="a" * 64,
        runner_fingerprint="b" * 64,
        model_fingerprint="c" * 64,
        topology=AttachmentTopology.IN_PROCESS,
        graph_mode=AttachmentGraphMode.EAGER,
        device_kind=StateDeviceKind.CPU,
        device_ordinal=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_dtype=TensorDType.FLOAT32,
        hidden_size=3,
        max_packed_rows=4,
        packed_row_schema_version=1,
        hidden_resources=(),
        task_bindings=task_bindings,
        sampler_anchor_id=stage_anchors[5].anchor_id,
        sampler_anchor_symbol=stage_anchors[5].anchor_symbol,
        stream_dependencies=(StreamDependency.TRANSITION_TO_RELEASE,),
        sampler_stage=AttachmentSamplerStage.PACKED_POST_FILTER_EXACT,
        cleanup_stage=AttachmentCleanupStage.REQUEST_FINISH,
        request_key_fields=(
            RequestKeyField.REQUEST_ID,
            RequestKeyField.PROMPT_IDX,
            RequestKeyField.SAMPLE_IDX,
        ),
        state_resource="dense_fast_weights",
        plan_epoch=plan.schedule_physical.plan_epoch,
        schedule_signature=plan.schedule_physical.schedule_signature,
        state_request_digest=plan.state_physical.request_digest,
        state_representation_signature=(
            plan.state_physical.representation_signature
        ),
        execution_parameters_digest=compiled.execution_parameters_digest,
        capabilities=capabilities,
        overlay_field_ids=fields,
        overlay_stage_anchors=stage_anchors,
    )
    probe = BackendPlacementProbe(
        adapter_id=request.adapter_id,
        backend_abi_digest=request.backend_abi_digest,
        runner_fingerprint=request.runner_fingerprint,
        model_fingerprint=request.model_fingerprint,
        topology=request.topology,
        graph_mode=request.graph_mode,
        device_kind=request.device_kind,
        device_ordinal=request.device_ordinal,
        tensor_parallel_size=request.tensor_parallel_size,
        pipeline_parallel_size=request.pipeline_parallel_size,
        tensor_dtype=request.tensor_dtype,
        hidden_size=request.hidden_size,
        max_packed_rows=request.max_packed_rows,
        hidden_resources=request.hidden_resources,
        capabilities=request.capabilities,
    )
    return BackendAttachmentCompilerInputs(
        request,
        probe,
        plan.semantic_graph,
        plan.timeline,
        plan.schedule_decision,
        plan.schedule_physical,
        plan.state_request,
        plan.state_decision,
        plan.state_physical,
    )


def test_dense_overlay_rejects_field_stage_and_task_tampering() -> None:
    inputs = _overlay_inputs()
    decision = compile_backend_attachment(inputs)
    physical = lower_backend_attachment(inputs, decision)
    validate_backend_attachment_physical(inputs, decision, physical)
    validate_backend_attachment_reconstruction(
        inputs,
        decision,
        physical,
        inputs,
        physical,
    )

    with pytest.raises(ValueError, match="attachment"):
        tampered = replace(
            inputs,
            request=replace(inputs.request, overlay_field_ids=(999,)),
        )
        validate_backend_attachment_reconstruction(
            inputs,
            decision,
            physical,
            tampered,
            physical,
        )

    bad_stage = replace(
        inputs.request.overlay_stage_anchors[1],
        anchor_symbol="state.publish",
    )
    with pytest.raises(ValueError, match="stage anchor"):
        compile_backend_attachment(
            replace(
                inputs,
                request=replace(
                    inputs.request,
                    overlay_stage_anchors=(
                        inputs.request.overlay_stage_anchors[0],
                        bad_stage,
                        *inputs.request.overlay_stage_anchors[2:],
                    ),
                ),
            )
        )

    bad_task = replace(
        inputs.request.task_bindings[0],
        implementation_id=ImplementationId("tampered:transition@1"),
    )
    with pytest.raises(ValueError, match="task binding"):
        compile_backend_attachment(
            replace(
                inputs,
                request=replace(
                    inputs.request,
                    task_bindings=(bad_task, inputs.request.task_bindings[1]),
                ),
            )
        )

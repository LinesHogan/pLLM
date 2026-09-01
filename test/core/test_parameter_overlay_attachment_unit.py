"""One Python-Rust tamper boundary for signed parameter-overlay attachment."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tllm.canonical.parameter_overlay_attachment import (
    compile_parameter_overlay_attachment,
    lower_parameter_overlay_attachment,
    validate_parameter_overlay_attachment_reconstruction,
)
from tllm.canonical.parameter_overlay_runtime import (
    CanonicalParameterOverlayRuntimeFacts,
)
from tllm.canonical.ptts import DensePageCpuProviderFacts
from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTSpecialization,
    InPlaceTTTTargetSpecialization,
    InPlaceTTTWorkload,
)
from tllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)
from tllm.contracts.parameter_overlay_attachment import (
    OperandCaptureKind,
    ParameterOverlayAttachmentBinding,
    ParameterOverlayAttachmentCandidate,
    ParameterOverlayAttachmentInputs,
    ParameterOverlayAttachmentRequest,
    ParameterOverlayStage,
    ParameterOverlayStageAnchor,
    ParameterOverlayScope,
    ParameterOverlayTaskBinding,
    ParameterOverlayTaskRole,
)
from tllm.contracts.request_identity import RequestKeyField
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    ImplementationId,
)
from tllm.contracts.tensor_layout import LayoutFieldId, TensorDType


def test_parameter_overlay_attachment_rejects_field_and_task_tampering() -> None:
    target = ParameterTarget(
        0,
        ParameterModuleKind.MLP,
        ParameterRole.DOWN_PROJECTION_WEIGHT,
    )
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2))
    timeline = CanonicalParameterOverlayRuntimeFacts(
        torch.device("cpu")
    ).timeline(("transition/full_root@1",))
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
    bindings = tuple(
        ParameterOverlayAttachmentBinding(
            binding.target,
            binding.operator,
            binding.fields,
        )
        for binding in compiled.state_layout.overlay_bindings
    )
    anchor_by_symbol = {
        anchor.symbol: anchor for anchor in compiled.plan_bundle.timeline.anchors
    }
    stage_anchors = tuple(
        ParameterOverlayStageAnchor(
            stage,
            AnchorId(anchor_by_symbol[symbol].ordinal),
            symbol,
        )
        for stage, symbol in (
            (ParameterOverlayStage.PIN, AnchorSymbol("request.prepare")),
            (ParameterOverlayStage.APPLY, AnchorSymbol("model.execute")),
            (ParameterOverlayStage.CAPTURE, AnchorSymbol("model.execute")),
            (ParameterOverlayStage.TRANSITION, AnchorSymbol("model.execute")),
            (ParameterOverlayStage.PUBLISH, AnchorSymbol("state.publish")),
            (ParameterOverlayStage.RELEASE, AnchorSymbol("sampler.complete")),
            (ParameterOverlayStage.FINISH, AnchorSymbol("request.finish")),
        )
    )
    scheduled = {
        str(task.task_symbol): task
        for task in compiled.plan_bundle.schedule_decision.tasks
    }
    physical = {
        task.task_id: task for task in compiled.plan_bundle.schedule_physical.tasks
    }
    task_bindings = tuple(
        ParameterOverlayTaskBinding(
            role,
            scheduled[symbol].task_id,
            physical[scheduled[symbol].task_id].implementation_id,
            physical[scheduled[symbol].task_id].anchor_id,
        )
        for role, symbol in (
            (ParameterOverlayTaskRole.TRANSITION, "transition"),
            (ParameterOverlayTaskRole.PUBLISH, "publish"),
        )
    )
    digest = "0" * 64
    capabilities = ("backend/test@1", "overlay/dense_add@1")
    request = ParameterOverlayAttachmentRequest(
        "test.adapter@1",
        digest,
        digest,
        digest,
        "dense_fast_weights",
        ParameterOverlayScope.REQUEST,
        (
            RequestKeyField.REQUEST_ID,
            RequestKeyField.PROMPT_IDX,
            RequestKeyField.SAMPLE_IDX,
        ),
        bindings,
        OperandCaptureKind.MLP_DOWN_PROJECTION_PAIR,
        stage_anchors,
        task_bindings,
        compiled.execution.plan.plan_epoch,
        compiled.execution_parameters_digest,
        compiled.execution.plan.schedule_signature,
        compiled.execution.state_plan.representation_signature,
        capabilities,
    )
    candidate = ParameterOverlayAttachmentCandidate(
        "test.candidate@1",
        "test.adapter@1",
        digest,
        digest,
        digest,
        ParameterOverlayScope.REQUEST,
        bindings,
        OperandCaptureKind.MLP_DOWN_PROJECTION_PAIR,
        stage_anchors,
        task_bindings,
        capabilities,
    )
    inputs = ParameterOverlayAttachmentInputs(
        request,
        (candidate,),
        compiled.plan_bundle.semantic_graph,
        compiled.plan_bundle.timeline,
        compiled.plan_bundle.schedule_decision,
        compiled.plan_bundle.schedule_physical,
        compiled.plan_bundle.state_request,
        compiled.plan_bundle.state_decision,
        compiled.plan_bundle.state_physical,
        compiled.tensor_layout_request,
        compiled.tensor_layout_decision,
    )
    decision = compile_parameter_overlay_attachment(inputs)
    attachment = lower_parameter_overlay_attachment(inputs, decision)
    validate_parameter_overlay_attachment_reconstruction(
        inputs,
        decision,
        attachment,
        inputs,
        attachment,
    )

    invalid_binding = replace(bindings[0], fields=(LayoutFieldId(999),))
    invalid_fields = replace(
        inputs,
        request=replace(request, bindings=(invalid_binding,)),
        candidates=(replace(candidate, bindings=(invalid_binding,)),),
    )
    with pytest.raises(ValueError, match="frozen tensor layout"):
        compile_parameter_overlay_attachment(invalid_fields)

    invalid_task = replace(
        task_bindings[0],
        implementation_id=ImplementationId("tampered:transition@1"),
    )
    invalid_tasks = replace(
        inputs,
        request=replace(
            request,
            task_bindings=(invalid_task, task_bindings[1]),
        ),
        candidates=(
            replace(
                candidate,
                task_bindings=(invalid_task, task_bindings[1]),
            ),
        ),
    )
    with pytest.raises(ValueError, match="physical plan"):
        compile_parameter_overlay_attachment(invalid_tasks)

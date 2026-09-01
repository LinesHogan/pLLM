"""Cold signed deployment of a dense-fast-weight workload onto vLLM MRv2."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch

from tllm.canonical.parameter_overlay_attachment import (
    compile_parameter_overlay_attachment,
    lower_parameter_overlay_attachment,
    validate_parameter_overlay_attachment,
    validate_parameter_overlay_attachment_reconstruction,
)
from tllm.canonical.parameter_overlay_runtime import (
    CanonicalParameterOverlayRuntimeFacts,
)
from tllm.canonical.ptts import DensePageCudaProviderFacts
from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTCompilation,
    InPlaceTTTCompileOptions,
    InPlaceTTTCudaExecution,
    InPlaceTTTSpecialization,
    InPlaceTTTTargetSpecialization,
    InPlaceTTTWorkload,
)
from tllm.canonical.parameter_overlay_plan import (
    ParameterOverlayRuntimePlan,
)
from tllm.canonical_vllm.dense_overlay import (
    VllmDenseAddTargetSpec,
    inspect_dense_add_targets,
)
from tllm.contracts.parameter_overlay_attachment import (
    OperandCaptureKind,
    ParameterOverlayAttachmentBinding,
    ParameterOverlayAttachmentCandidate,
    ParameterOverlayAttachmentDecision,
    ParameterOverlayAttachmentInputs,
    ParameterOverlayAttachmentPhysicalPlan,
    ParameterOverlayAttachmentRequest,
    ParameterOverlayStage,
    ParameterOverlayStageAnchor,
    ParameterOverlayScope,
    ParameterOverlayTaskBinding,
    ParameterOverlayTaskRole,
)
from tllm.contracts.request_identity import RequestKeyField
from tllm.contracts.semantic_graph import AnchorId, AnchorSymbol
from tllm.contracts.tensor_layout import TensorDType

from vllm import __version__ as _VLLM_VERSION
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


_ADAPTER_ID = "canonical.vllm_mrv2.parameter_overlay@1"
_CANDIDATE_ID = "canonical.vllm_mrv2.dense_add@1"
_SUPPORTED_VLLM_VERSION = "0.27.1"
_CAPABILITIES = (
    "backend/vllm_mrv2@1",
    "operand_capture/mlp_down_projection_pair@1",
    "overlay/dense_add@1",
)


def _digest(fields: tuple[str, ...]) -> str:
    return sha256("\x00".join(fields).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SignedParameterOverlayAttachment:
    inputs: ParameterOverlayAttachmentInputs
    decision: ParameterOverlayAttachmentDecision
    physical: ParameterOverlayAttachmentPhysicalPlan


@dataclass(frozen=True, slots=True)
class CanonicalParameterOverlayDeployment:
    workload: InPlaceTTTWorkload
    compilation: InPlaceTTTCompilation
    execution: InPlaceTTTCudaExecution
    attachment: SignedParameterOverlayAttachment
    runtime_plan: ParameterOverlayRuntimePlan
    specialization: InPlaceTTTSpecialization
    targets: tuple[VllmDenseAddTargetSpec, ...]
    max_packed_rows: int
    model_dtype: torch.dtype

    def __post_init__(self) -> None:
        plan = self.compilation.plan_bundle
        physical = self.attachment.physical
        if (
            self.compilation.execution is not self.execution
            or self.execution.plan is not plan.schedule_physical
            or self.execution.state_plan is not plan.state_physical
            or self.execution.specialization != self.specialization
            or self.compilation.execution_parameters_digest
            != self.specialization.execution_parameters_digest(self.workload.config)
            or self.attachment.inputs.graph is not plan.semantic_graph
            or self.attachment.inputs.schedule_physical is not plan.schedule_physical
            or self.attachment.inputs.state_physical is not plan.state_physical
            or self.runtime_plan.physical is not physical
            or physical.plan_epoch != self.execution.plan.plan_epoch
            or physical.schedule_signature != self.execution.plan.schedule_signature
            or physical.state_representation_signature
            != self.execution.state_plan.representation_signature
            or self.max_packed_rows < 1
        ):
            raise ValueError("canonical parameter-overlay deployment is not closed")


def compile_parameter_overlay_vllm_deployment(
    workload: InPlaceTTTWorkload,
    runner: GPUModelRunner,
) -> CanonicalParameterOverlayDeployment:
    """Compile from loaded model facts and sign every adapter binding."""

    if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError("canonical parameter overlays require vLLM 0.27.1")
    if (
        not runner.model_config.enforce_eager
        or runner.parallel_config.tensor_parallel_size != 1
        or runner.parallel_config.pipeline_parallel_size != 1
        or runner.dp_size != 1
        or runner.dcp_size != 1
        or runner.num_speculative_steps != 0
        or not runner.is_first_pp_rank
        or not runner.is_last_pp_rank
        or runner.lora_config is not None
        or runner.device.type != "cuda"
        or runner.max_num_reqs < 1
    ):
        raise RuntimeError(
            "canonical dense-add requires eager in-process TP1/PP1/DP1 MRv2"
        )
    match runner.model:
        case Qwen2ForCausalLM() as model:
            pass
        case _:
            raise RuntimeError("canonical dense-add currently requires Qwen2")
    if model.quant_config is not None:
        raise RuntimeError("canonical dense-add requires an unquantized model")
    targets = inspect_dense_add_targets(model, workload.config.targets)
    if any(target.dtype is not runner.dtype for target in targets):
        raise RuntimeError("dense-add target dtype differs from the runner")
    match runner.dtype:
        case torch.float16:
            tensor_dtype = TensorDType.FLOAT16
        case torch.bfloat16:
            tensor_dtype = TensorDType.BFLOAT16
        case torch.float32:
            tensor_dtype = TensorDType.FLOAT32
        case _:
            raise RuntimeError("canonical dense-add runner dtype is unsupported")
    device_ordinal = runner.device.index
    max_packed_rows = runner.max_num_tokens
    free_device_bytes, _total_device_bytes = torch.cuda.mem_get_info(runner.device)
    specialization = InPlaceTTTSpecialization(
        tuple(
            InPlaceTTTTargetSpecialization(
                target.target,
                target.input_size,
                target.output_size,
            )
            for target in targets
        ),
        tensor_dtype,
        runner.device,
    )
    timeline = CanonicalParameterOverlayRuntimeFacts(runner.device).timeline(
        (
            "state/fenced_reclaim@1",
            "transition/full_root@1",
        )
    )
    compilation = workload.compile(
        specialization,
        InPlaceTTTCompileOptions(
            timeline,
            (
                DensePageCudaProviderFacts(
                    "canonical.cuda.full_root@1",
                    device_ordinal,
                    3,
                ),
            ),
            max_admitted_lineages=runner.max_num_reqs,
            state_memory_budget_bytes=free_device_bytes,
        ),
    )
    match compilation.execution:
        case InPlaceTTTCudaExecution() as execution:
            pass
        case _:
            raise RuntimeError("vLLM facts did not select CUDA dense-fast-weight")

    plan = compilation.plan_bundle
    anchor_by_symbol = {
        anchor.symbol: anchor for anchor in plan.timeline.anchors
    }
    stage_symbols = (
        (ParameterOverlayStage.PIN, AnchorSymbol("request.prepare")),
        (ParameterOverlayStage.APPLY, AnchorSymbol("model.execute")),
        (ParameterOverlayStage.CAPTURE, AnchorSymbol("model.execute")),
        (ParameterOverlayStage.TRANSITION, AnchorSymbol("model.execute")),
        (ParameterOverlayStage.PUBLISH, AnchorSymbol("state.publish")),
        (ParameterOverlayStage.RELEASE, AnchorSymbol("sampler.complete")),
        (ParameterOverlayStage.FINISH, AnchorSymbol("request.finish")),
    )
    stage_anchors = tuple(
        ParameterOverlayStageAnchor(
            stage,
            AnchorId(anchor_by_symbol[symbol].ordinal),
            symbol,
        )
        for stage, symbol in stage_symbols
    )
    scheduled_by_symbol = {
        str(task.task_symbol): task for task in plan.schedule_decision.tasks
    }
    physical_by_id = {
        task.task_id: task for task in plan.schedule_physical.tasks
    }
    task_bindings = tuple(
        ParameterOverlayTaskBinding(
            role,
            scheduled_by_symbol[symbol].task_id,
            physical_by_id[
                scheduled_by_symbol[symbol].task_id
            ].implementation_id,
            physical_by_id[scheduled_by_symbol[symbol].task_id].anchor_id,
        )
        for role, symbol in (
            (ParameterOverlayTaskRole.TRANSITION, "transition"),
            (ParameterOverlayTaskRole.PUBLISH, "publish"),
        )
    )
    bindings = tuple(
        sorted(
            (
                ParameterOverlayAttachmentBinding(
                    binding.target,
                    binding.operator,
                    binding.fields,
                )
                for binding in compilation.state_layout.overlay_bindings
            )
        )
    )
    backend_abi_digest = _digest(
        (
            "vllm=0.27.1",
            "runner=vllm.v1.worker.gpu.model_runner.GPUModelRunner",
            "model=qwen2",
            "mode=in_process/eager/tp1/pp1/dp1/non_spec",
            "overlay=dense_add",
            "capture=mlp_down_projection_pair",
        )
    )
    runner_fingerprint = _digest(
        (
            _ADAPTER_ID,
            f"device={runner.device}",
            f"dtype={runner.dtype}",
            f"max_num_tokens={max_packed_rows}",
            f"max_num_reqs={runner.max_num_reqs}",
        )
    )
    model_fingerprint = _digest(
        (
            f"model={runner.model_config.model}",
            f"revision={runner.model_config.revision or ''}",
            "class=vllm.model_executor.models.qwen2.Qwen2ForCausalLM",
            *(
                "target="
                + str(target.target.layer)
                + ":"
                + target.target.module.value
                + ":"
                + target.target.role.value
                + f":{target.input_size}:{target.output_size}"
                for target in targets
            ),
        )
    )
    request = ParameterOverlayAttachmentRequest(
        _ADAPTER_ID,
        backend_abi_digest,
        runner_fingerprint,
        model_fingerprint,
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
        execution.plan.plan_epoch,
        compilation.execution_parameters_digest,
        execution.plan.schedule_signature,
        execution.state_plan.representation_signature,
        _CAPABILITIES,
    )
    candidate = ParameterOverlayAttachmentCandidate(
        _CANDIDATE_ID,
        _ADAPTER_ID,
        backend_abi_digest,
        runner_fingerprint,
        model_fingerprint,
        ParameterOverlayScope.REQUEST,
        bindings,
        OperandCaptureKind.MLP_DOWN_PROJECTION_PAIR,
        stage_anchors,
        task_bindings,
        _CAPABILITIES,
    )
    inputs = ParameterOverlayAttachmentInputs(
        request,
        (candidate,),
        plan.semantic_graph,
        plan.timeline,
        plan.schedule_decision,
        plan.schedule_physical,
        plan.state_request,
        plan.state_decision,
        plan.state_physical,
        compilation.tensor_layout_request,
        compilation.tensor_layout_decision,
    )
    decision = compile_parameter_overlay_attachment(inputs)
    validate_parameter_overlay_attachment(inputs, decision)
    physical = lower_parameter_overlay_attachment(inputs, decision)
    validate_parameter_overlay_attachment_reconstruction(
        inputs,
        decision,
        physical,
        inputs,
        physical,
    )
    return CanonicalParameterOverlayDeployment(
        workload,
        compilation,
        execution,
        SignedParameterOverlayAttachment(inputs, decision, physical),
        ParameterOverlayRuntimePlan(physical),
        specialization,
        targets,
        max_packed_rows,
        runner.dtype,
    )


__all__ = [
    "CanonicalParameterOverlayDeployment",
    "SignedParameterOverlayAttachment",
    "compile_parameter_overlay_vllm_deployment",
]

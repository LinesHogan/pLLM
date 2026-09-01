"""Cold compilation of one canonical ESamp workload for vLLM MRv2."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch

from tllm.canonical.ptts import DensePageCudaProviderFacts
from tllm.canonical.request_sampler_attachment import (
    SignedBackendAttachment,
    compile_request_sampler_attachment,
)
from tllm.canonical.workloads.esamp_request import (
    ESampRequestCompilation,
    ESampRequestCudaExecution,
    ESampRequestWorkload,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions,
    ESampRequestPackedHiddenBindings,
    ESampRequestSpecialization,
)
from tllm.contracts.backend_attachment import (
    AttachmentCleanupStage,
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentTaskRole,
    AttachmentTopology,
    BackendPlacementProbe,
)
from tllm.contracts.hidden_locator import (
    HiddenLayerKind,
    HiddenPhase,
    HiddenSite,
)
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.request_identity import RequestKeyField
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    SchedulerTimeline,
    TimelineAnchor,
)
from tllm.contracts.state_representation import StateDeviceKind
from tllm.contracts.tensor_layout import TensorDType

from vllm import __version__ as _VLLM_VERSION
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler


_ADAPTER_ID = "canonical.vllm_mrv2@1"
_SUPPORTED_VLLM_VERSION = "0.27.1"
_BACKEND_ABI_FIELDS = (
    "vllm=0.27.1",
    "runner=vllm.v1.worker.gpu.model_runner.GPUModelRunner",
    "input=vllm.v1.worker.gpu.input_batch.InputBatch",
    "sampler=vllm.v1.worker.gpu.sample.sampler.Sampler",
    "mode=in_process/eager/tp1/pp1/dp1/non_spec",
    "rows=one_logit_per_request",
    "sampler_stage=packed_post_filter_exact",
)


def _digest(fields: tuple[str, ...]) -> str:
    return sha256("\x00".join(fields).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalEsampVllmDeployment:
    """Executable CUDA workload plus the exact backend attachment that admits it."""

    workload: ESampRequestWorkload
    compilation: ESampRequestCompilation
    execution: ESampRequestCudaExecution
    attachment: SignedBackendAttachment
    specialization: ESampRequestSpecialization
    source: PackedHiddenResourceBinding
    target: PackedHiddenResourceBinding
    hidden_size: int
    max_packed_rows: int
    model_dtype: torch.dtype
    sampler_beta: float

    def __post_init__(self) -> None:
        plan_bundle = self.compilation.plan_bundle
        attachment_inputs = self.attachment.inputs
        attachment_request = attachment_inputs.request
        attachment_physical = self.attachment.physical
        signed_bindings = tuple(
            resource.binding for resource in attachment_request.hidden_resources
        )
        signed_phases = tuple(
            resource.phase for resource in attachment_request.hidden_resources
        )
        expected_bindings = tuple(
            sorted((self.source, self.target), key=lambda binding: binding.locator_key)
        )
        if (
            type(self.workload) is not ESampRequestWorkload
            or type(self.compilation) is not ESampRequestCompilation
            or type(self.execution) is not ESampRequestCudaExecution
            or self.compilation.execution is not self.execution
            or type(self.attachment) is not SignedBackendAttachment
            or type(self.specialization) is not ESampRequestSpecialization
            or type(self.source) is not PackedHiddenResourceBinding
            or type(self.target) is not PackedHiddenResourceBinding
            or type(self.hidden_size) is not int
            or self.hidden_size < 1
            or type(self.max_packed_rows) is not int
            or self.max_packed_rows < 1
            or type(self.model_dtype) is not torch.dtype
            or type(self.sampler_beta) is not float
            or self.sampler_beta < 0.0
            or self.sampler_beta != self.workload.config.intervention_beta
            or self.execution.config != self.workload.config
            or self.execution.specialization != self.specialization
            or self.specialization.hidden_bindings.source != self.source
            or self.specialization.hidden_bindings.target != self.target
            or self.compilation.execution_parameters_digest
            != self.specialization.execution_parameters_digest(self.workload.config)
            or attachment_inputs.graph is not plan_bundle.semantic_graph
            or attachment_inputs.timeline is not plan_bundle.timeline
            or attachment_inputs.schedule_decision is not plan_bundle.schedule_decision
            or attachment_inputs.schedule_physical is not plan_bundle.schedule_physical
            or attachment_inputs.state_request is not plan_bundle.state_request
            or attachment_inputs.state_decision is not plan_bundle.state_decision
            or attachment_inputs.state_physical is not plan_bundle.state_physical
            or plan_bundle.schedule_physical is not self.execution.plan
            or plan_bundle.state_physical is not self.execution.state_plan
            or signed_bindings != expected_bindings
            or signed_phases != tuple(HiddenPhase.DECODE for _ in signed_bindings)
            or attachment_request.cleanup_stage
            is not AttachmentCleanupStage.REQUEST_FINISH
            or attachment_request.execution_parameters_digest
            != self.compilation.execution_parameters_digest
            or tuple(binding.role for binding in attachment_request.task_bindings)
            != (AttachmentTaskRole.COMPUTE, AttachmentTaskRole.TRANSITION)
            or attachment_request.plan_epoch != self.execution.plan.plan_epoch
            or attachment_physical.plan_epoch != self.execution.plan.plan_epoch
            or attachment_physical.schedule_signature
            != self.execution.plan.schedule_signature
            or attachment_physical.state_representation_signature
            != self.execution.state_plan.representation_signature
            or attachment_physical.device_kind is not StateDeviceKind.CUDA
            or attachment_physical.device_ordinal != self.execution.device.index
        ):
            raise ValueError("canonical ESamp vLLM deployment is not closed")

    @property
    def compute_anchor_id(self) -> AnchorId:
        return self.attachment.inputs.request.task_bindings[0].anchor_id

    @property
    def sampler_anchor_id(self) -> AnchorId:
        return self.attachment.inputs.request.sampler_anchor_id

    @property
    def transition_anchor_id(self) -> AnchorId:
        return self.attachment.inputs.request.task_bindings[1].anchor_id

    @property
    def cleanup_stage(self) -> AttachmentCleanupStage:
        return self.attachment.inputs.request.cleanup_stage


def compile_esamp_vllm_deployment(
    workload: ESampRequestWorkload,
    runner: GPUModelRunner,
) -> CanonicalEsampVllmDeployment:
    """Compile from concrete loaded-runner facts, then sign the full attachment."""

    if type(workload) is not ESampRequestWorkload:
        raise TypeError("canonical vLLM currently accepts one exact ESamp workload")
    if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError("canonical vLLM requires vLLM 0.27.1 exactly")
    if workload.config.key_fields != (RequestKeyField.REQUEST_ID,):
        raise RuntimeError(
            "canonical vLLM n=1 slice currently requires request-id state keys"
        )
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
        or type(runner.sampler) is not Sampler
    ):
        raise RuntimeError(
            "canonical vLLM requires eager in-process TP1/PP1/DP1 non-spec MRv2"
        )
    match runner.model:
        case Qwen2ForCausalLM() as model:
            pass
        case _:
            raise RuntimeError("canonical ESamp vLLM currently requires Qwen2")
    if model.quant_config is not None:
        raise RuntimeError("canonical ESamp vLLM does not support quantized LM heads")

    layer_count = len(model.model.layers)
    source_layer = workload.config.source_layer
    target_layer = workload.config.target_layer
    source_module_index = (
        source_layer if source_layer >= 0 else layer_count + source_layer
    )
    target_module_index = (
        target_layer if target_layer >= 0 else layer_count + target_layer
    )
    if (
        source_module_index < 0
        or source_module_index >= layer_count
        or target_module_index < 0
        or target_module_index >= layer_count
        or source_module_index == target_module_index
    ):
        raise ValueError("ESamp hidden layers do not identify two Qwen2 blocks")
    source = PackedHiddenResourceBinding(
        str(source_layer),
        HiddenLayerKind.INDEX,
        HiddenSite.BLOCK_OUTPUT,
        f"model.layers.{source_module_index}",
    )
    target = PackedHiddenResourceBinding(
        str(target_layer),
        HiddenLayerKind.INDEX,
        HiddenSite.BLOCK_OUTPUT,
        f"model.layers.{target_module_index}",
    )
    hidden_bindings = ESampRequestPackedHiddenBindings(source, target)
    hidden_size = model.config.hidden_size
    if type(hidden_size) is not int or hidden_size < 1:
        raise RuntimeError("Qwen2 hidden size is not a closed positive integer")
    device_ordinal = runner.device.index
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise RuntimeError("canonical vLLM requires an ordinal CUDA device")
    max_packed_rows = runner.max_num_tokens
    max_admitted_lineages = runner.max_num_reqs
    if (
        type(max_packed_rows) is not int
        or max_packed_rows < 1
        or type(max_admitted_lineages) is not int
        or max_admitted_lineages < 1
    ):
        raise RuntimeError("vLLM runner capacities are not closed positive integers")
    free_device_bytes, _total_device_bytes = torch.cuda.mem_get_info(runner.device)
    if type(free_device_bytes) is not int or free_device_bytes < 1:
        raise RuntimeError("CUDA did not report an attachment memory budget")

    timeline = SchedulerTimeline(
        anchors=(
            TimelineAnchor(AnchorSymbol("model.execute"), 0),
            TimelineAnchor(AnchorSymbol("sampler.pre_draw"), 1),
            TimelineAnchor(AnchorSymbol("token.commit"), 2),
            TimelineAnchor(AnchorSymbol("next.prepare"), 3),
        ),
        capabilities=(
            "compute/cuda@1",
            "state/dense_page_cuda@1",
        ),
    )
    specialization = ESampRequestSpecialization(
        hidden_size=hidden_size,
        tensor_dtype=TensorDType.FLOAT32,
        initialization_seed=runner.model_config.seed,
        hidden_bindings=hidden_bindings,
        device_kind=StateDeviceKind.CUDA,
        device_ordinal=device_ordinal,
    )
    compilation = workload.compile(
        specialization,
        ESampRequestCompileOptions(
            timeline=timeline,
            providers=(
                DensePageCudaProviderFacts(
                    "canonical.cuda_dense_page@1",
                    device_ordinal,
                    max_live_versions_per_lineage=3,
                ),
            ),
            max_admitted_lineages=max_admitted_lineages,
            state_memory_budget_bytes=free_device_bytes,
        ),
    )
    match compilation.execution:
        case ESampRequestCudaExecution() as execution:
            pass
        case _:
            raise RuntimeError("vLLM facts did not select CUDA ESamp execution")

    hidden_resources = tuple(
        sorted(
            (
                AttachmentHiddenResource(source, HiddenPhase.DECODE),
                AttachmentHiddenResource(target, HiddenPhase.DECODE),
            ),
            key=lambda resource: resource.resource_key,
        )
    )
    capabilities = (
        "backend/vllm_mrv2@1",
        "producer/packed_hidden@1",
        "sampler/post_filter_exact@1",
    )
    backend_abi_digest = _digest(_BACKEND_ABI_FIELDS)
    runner_fingerprint = _digest(
        (
            _ADAPTER_ID,
            f"device=cuda:{device_ordinal}",
            f"max_num_tokens={max_packed_rows}",
            f"max_num_reqs={max_admitted_lineages}",
            f"dtype={runner.dtype}",
            f"seed={runner.model_config.seed}",
        )
    )
    revision = runner.model_config.revision
    model_fingerprint = _digest(
        (
            f"model={runner.model_config.model}",
            f"revision={revision if revision is not None else ''}",
            "class=vllm.model_executor.models.qwen2.Qwen2ForCausalLM",
            f"hidden_size={hidden_size}",
            f"layers={layer_count}",
            f"source={source.module_path}",
            f"target={target.module_path}",
        )
    )
    probe = BackendPlacementProbe(
        adapter_id=_ADAPTER_ID,
        backend_abi_digest=backend_abi_digest,
        runner_fingerprint=runner_fingerprint,
        model_fingerprint=model_fingerprint,
        topology=AttachmentTopology.IN_PROCESS,
        graph_mode=AttachmentGraphMode.EAGER,
        device_kind=StateDeviceKind.CUDA,
        device_ordinal=device_ordinal,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_dtype=TensorDType.FLOAT32,
        hidden_size=hidden_size,
        max_packed_rows=max_packed_rows,
        hidden_resources=hidden_resources,
        capabilities=capabilities,
    )
    attachment = compile_request_sampler_attachment(
        compilation.plan_bundle,
        probe,
        workload.request_sampler_attachment_spec(
            compilation.execution_parameters_digest
        ),
    )
    return CanonicalEsampVllmDeployment(
        workload=workload,
        compilation=compilation,
        execution=execution,
        attachment=attachment,
        specialization=specialization,
        source=source,
        target=target,
        hidden_size=hidden_size,
        max_packed_rows=max_packed_rows,
        model_dtype=runner.dtype,
        sampler_beta=workload.config.intervention_beta,
    )


__all__ = [
    "CanonicalEsampVllmDeployment",
    "compile_esamp_vllm_deployment",
]

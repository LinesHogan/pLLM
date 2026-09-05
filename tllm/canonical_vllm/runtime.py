"""Workload-neutral public handle for canonical vLLM MRv2 attachments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Literal

import torch

from tllm.canonical.backend_attachment import (
    compile_parameter_overlay_attachment,
    compile_request_sampler_attachment,
    validate_backend_attachment_physical,
)
from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts
from tllm.canonical.runtime_facts import CanonicalRuntimeFacts
from tllm.canonical_vllm.dense_overlay import (
    VllmDenseAddAdapter,
    inspect_dense_add_targets,
)
from tllm.canonical_vllm.hidden_capture import Vllm027EagerPackedHiddenCapture
from tllm.canonical_vllm.packed_rows import (
    Mrv2PackedStepRows,
    Mrv2RequestIdBatch,
    Mrv2RequestIdBinding,
    Vllm027Mrv2PackedRowCodec,
)
from tllm.canonical_vllm.sampler import (
    CanonicalVllm027PostFilterSampler,
    SignedPostFilterGuidance,
)
from tllm.canonical.workloads.esamp_request import (
    CanonicalRequestStepHandle,
    CanonicalRequestSamplerExecution,
    ESampRequestCompilation,
    ESampRequestWorkload,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions,
    ESampRequestPackedHiddenBindings,
    ESampRequestSpecialization,
    ESampRequestStepInput,
)
from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTCompileOptions,
    InPlaceTTTStepHandle,
    InPlaceTTTOperandPair,
    InPlaceTTTPhase,
    InPlaceTTTSpecialization,
    InPlaceTTTStepBatch,
    InPlaceTTTStepInput,
    InPlaceTTTTargetSpecialization,
)
from tllm.canonical.workloads.inplace_ttt.types import CudaInPlaceTTTStepBatchInput
from tllm.canonical.workloads.inplace_ttt.workload import (
    CanonicalDenseOverlayExecution,
    InPlaceTTTCompilation,
    InPlaceTTTWorkload,
)
from tllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentKind,
    AttachmentOverlayStage,
    AttachmentOverlayStageAnchor,
    AttachmentTaskBinding,
    AttachmentTaskRole,
    AttachmentTopology,
    BackendPlacementProbe,
    SignedBackendAttachment,
)
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.parameter_overlay import (
    ParameterOverlayRowAssignment,
    ParameterOverlayStepPlan,
)
from tllm.contracts.request_identity import RequestIdentity, RequestKeyField
from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    SchedulerTimeline,
    StateResource,
    TimelineAnchor,
)
from tllm.contracts.state_representation import StateDeviceKind
from tllm.contracts.tensor_layout import TensorDType

from vllm import RequestOutput, SamplingParams, __version__ as _VLLM_VERSION
from vllm.config import VllmConfig
from vllm.entrypoints.llm import LLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu_worker import Worker


_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"
_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_WORKER_CLASS = "tllm.canonical_vllm.runtime.CanonicalGPUWorker"
_ADAPTER_ID = "canonical.vllm_mrv2@1"
_SUPPORTED_VLLM_VERSION = "0.27.1"
_BACKEND_ABI_FIELDS = (
    "vllm=0.27.1",
    "runner=vllm.v1.worker.gpu.model_runner.GPUModelRunner",
    "input=vllm.v1.worker.gpu.input_batch.InputBatch",
    "sampler=vllm.v1.worker.gpu.sample.sampler.Sampler",
    "mode=in_process/eager/tp1/pp1/dp1/non_spec",
    "rows=one_logit_per_request",
    "attachment=shared_plan_v2",
    "attachments=request_sampler_training|dense_parameter_overlay",
)
_REQUEST_CAPABILITIES = (
    "backend/vllm_mrv2@1",
    "producer/packed_hidden@1",
    "sampler/post_filter_exact@1",
)
_OVERLAY_CAPABILITIES = (
    "backend/vllm_mrv2@1",
    "operand_capture/mlp_down_projection_pair@1",
    "overlay/dense_add@1",
)


class CanonicalVllmDType(StrEnum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


@dataclass(frozen=True, slots=True)
class CanonicalVllmConfig:
    model: str
    dtype: CanonicalVllmDType
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    seed: int = 0
    trust_remote_code: bool = False
    enable_prefix_caching: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.model) is not str
            or not self.model
            or type(self.dtype) is not CanonicalVllmDType
            or type(self.max_model_len) is not int
            or self.max_model_len < 2
            or type(self.max_num_seqs) is not int
            or self.max_num_seqs < 1
            or type(self.max_num_batched_tokens) is not int
            or self.max_num_batched_tokens < self.max_num_seqs
            or type(self.gpu_memory_utilization) is not float
            or not 0.0 < self.gpu_memory_utilization < 1.0
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.trust_remote_code) is not bool
            or type(self.enable_prefix_caching) is not bool
        ):
            raise ValueError("canonical vLLM configuration is not closed")


def _digest(fields: tuple[str, ...]) -> str:
    return sha256("\x00".join(fields).encode("utf-8")).hexdigest()


def _tensor_dtype(dtype: torch.dtype) -> TensorDType:
    match dtype:
        case torch.float16:
            return TensorDType.FLOAT16
        case torch.bfloat16:
            return TensorDType.BFLOAT16
        case torch.float32:
            return TensorDType.FLOAT32
        case _:
            raise RuntimeError("canonical vLLM runner dtype is unsupported")


def _model_dtype(dtype: CanonicalVllmDType) -> Literal["bfloat16", "float16", "float32"]:
    match dtype:
        case CanonicalVllmDType.BFLOAT16:
            return "bfloat16"
        case CanonicalVllmDType.FLOAT16:
            return "float16"
        case CanonicalVllmDType.FLOAT32:
            return "float32"


@dataclass(frozen=True, slots=True)
class CanonicalVllmDeployment:
    compilation: ESampRequestCompilation | InPlaceTTTCompilation
    attachment: SignedBackendAttachment

    def __post_init__(self) -> None:
        plan = self.compilation.plan_bundle
        execution = self.compilation.execution
        if (
            self.attachment.inputs.plan is not plan
            or plan.schedule_physical is not execution.plan
            or plan.state_physical is not execution.state_plan
            or self.attachment.physical.device_kind is not StateDeviceKind.CUDA
        ):
            raise ValueError("canonical vLLM deployment is not closed")

    @property
    def kind(self) -> AttachmentKind:
        return self.attachment.inputs.request.attachment_kind


@dataclass(frozen=True, slots=True)
class _RequestSamplerVllmDeployment(CanonicalVllmDeployment):
    compilation: ESampRequestCompilation


@dataclass(frozen=True, slots=True)
class _DenseOverlayVllmDeployment(CanonicalVllmDeployment):
    compilation: InPlaceTTTCompilation


_ClosedVllmDeployment = _RequestSamplerVllmDeployment | _DenseOverlayVllmDeployment


@dataclass(frozen=True, slots=True)
class _Idle:
    pass


@dataclass(frozen=True, slots=True)
class _Capturing:
    rows: Mrv2PackedStepRows


@dataclass(frozen=True, slots=True)
class _RequestPredicted:
    handle: CanonicalRequestStepHandle


@dataclass(frozen=True, slots=True)
class _RequestSampled:
    handle: CanonicalRequestStepHandle


@dataclass(frozen=True, slots=True)
class _OverlayPrepared:
    handle: InPlaceTTTStepHandle


@dataclass(frozen=True, slots=True)
class _OverlayTransitioned:
    handle: InPlaceTTTStepHandle


@dataclass(frozen=True, slots=True)
class _OverlaySampled:
    handle: InPlaceTTTStepHandle


_SessionState = (
    _Idle
    | _Capturing
    | _RequestPredicted
    | _RequestSampled
    | _OverlayPrepared
    | _OverlayTransitioned
    | _OverlaySampled
)


class CanonicalVllmSession:
    """One runner-owned closed attachment session."""

    __slots__ = (
        "_capture",
        "_deployment",
        "_dense_adapter",
        "_identities",
        "_next_prompt_index",
        "_next_step_id",
        "_row_codec",
        "_sampler",
        "_state",
    )

    def __init__(
        self,
        deployment: _ClosedVllmDeployment,
        model: Qwen2ForCausalLM,
        upstream_sampler: Sampler,
    ) -> None:
        validate_backend_attachment_physical(
            deployment.attachment.inputs,
            deployment.attachment.decision,
            deployment.attachment.physical,
        )
        self._deployment: _ClosedVllmDeployment = deployment
        self._row_codec = Vllm027Mrv2PackedRowCodec(
            plan_epoch=deployment.attachment.physical.plan_epoch,
        )
        self._sampler: CanonicalVllm027PostFilterSampler | None = None
        self._capture: Vllm027EagerPackedHiddenCapture | None = None
        self._dense_adapter: VllmDenseAddAdapter | None = None
        match deployment:
            case _RequestSamplerVllmDeployment():
                bindings = deployment.compilation.execution.specialization.hidden_bindings
                self._capture = Vllm027EagerPackedHiddenCapture(
                    model=model,
                    source=bindings.source,
                    target=bindings.target,
                    plan_epoch=deployment.attachment.physical.plan_epoch,
                    max_packed_rows=deployment.attachment.inputs.probe.max_packed_rows,
                    hidden_size=deployment.attachment.inputs.probe.hidden_size,
                )
                self._sampler = CanonicalVllm027PostFilterSampler(
                    upstream_sampler,
                    eager=True,
                )
            case _DenseOverlayVllmDeployment():
                self._dense_adapter = VllmDenseAddAdapter(
                    model,
                    deployment.compilation.execution.config.targets,
                )
        self._identities: dict[str, RequestIdentity] = {}
        self._next_prompt_index = 0
        self._next_step_id = 0
        self._state: _SessionState = _Idle()

    @property
    def sampler(self) -> CanonicalVllm027PostFilterSampler:
        if self._sampler is None:
            raise RuntimeError("canonical session has no sampler wrapper")
        return self._sampler

    @property
    def apply_count(self) -> int:
        return 0 if self._dense_adapter is None else self._dense_adapter.apply_count

    @property
    def updated_apply_count(self) -> int:
        if self._dense_adapter is None:
            return 0
        return self._dense_adapter.updated_apply_count

    def before_execute(self, scheduler_output: SchedulerOutput) -> None:
        match self._state:
            case _Idle():
                pass
            case _:
                raise RuntimeError("prior canonical vLLM step is still active")
        deployment = self._deployment
        if isinstance(deployment, _RequestSamplerVllmDeployment):
            deployment.compilation.execution.collect_completed()
        finished = tuple(
            self._identities[request_id]
            for request_id in sorted(scheduler_output.finished_req_ids)
        )
        if finished:
            self._retire(finished)
            for identity in finished:
                self._identities.pop(identity.request_id)
        pending: list[RequestIdentity] = []
        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id in self._identities:
                continue
            sampling_params = request.sampling_params
            if type(request.req_id) is not str or not request.req_id:
                raise RuntimeError("vLLM scheduled an invalid request id")
            if type(sampling_params) is not SamplingParams or sampling_params.n != 1:
                raise RuntimeError("canonical vLLM requires sampling n=1")
            prompt_idx = self._next_prompt_index + len(pending)
            sample_idx = 0
            pending.append(RequestIdentity(request.req_id, prompt_idx, sample_idx))
        if pending:
            device = deployment.compilation.execution.device
            stream = torch.cuda.current_stream(device)
            deployment.compilation.execution.admit(tuple(pending), stream)
            for identity in pending:
                self._identities[identity.request_id] = identity
            self._next_prompt_index += len(pending)

    def prepare_input(self, input_batch: InputBatch) -> None:
        match self._state:
            case _Idle():
                pass
            case _:
                raise RuntimeError("canonical input preparation overlaps a step")
        request_ids = Mrv2RequestIdBatch(
            tuple(
                Mrv2RequestIdBinding(raw_request_id, self._identities[raw_request_id])
                for raw_request_id in input_batch.req_ids
            )
        )
        rows = self._row_codec.encode(
            input_batch,
            request_ids,
            step_id=self._next_step_id,
        )
        self._next_step_id += 1
        match self._deployment:
            case _RequestSamplerVllmDeployment():
                capture = self._require_capture()
                capture.begin(rows.phase_rows)
                self._state = _Capturing(rows)
            case _DenseOverlayVllmDeployment() as deployment:
                adapter = self._require_dense_adapter()
                bindings = deployment.compilation.state_layout.overlay_bindings
                identities = tuple(binding.identity for binding in request_ids.bindings)
                stream = torch.cuda.current_stream(deployment.compilation.execution.device)
                handle = deployment.compilation.execution.begin_batch(
                    identities,
                    stream,
                    bindings,
                    self._stage(AttachmentOverlayStage.PIN).anchor_id,
                )
                ref_by_identity = dict(zip(handle.identities, handle.refs, strict=True))
                assignments: list[ParameterOverlayRowAssignment] = []
                for target in deployment.compilation.execution.config.targets:
                    for identity in handle.identities:
                        selected_rows = tuple(
                            row
                            for row in rows.phase_rows.rows
                            if row.identity == identity
                        )
                        phases = frozenset(row.phase for row in selected_rows)
                        if len(phases) != 1:
                            raise RuntimeError(
                                "one request cannot mix overlay phases in one step"
                            )
                        (phase,) = phases
                        assignments.append(
                            ParameterOverlayRowAssignment(
                                identity,
                                target,
                                ref_by_identity[identity],
                                phase,
                                tuple(row.packed_row_index for row in selected_rows),
                            )
                        )
                canonical_assignments = tuple(
                    sorted(
                        assignments,
                        key=lambda item: (
                            item.target,
                            item.identity.request_id,
                            item.identity.prompt_idx,
                            item.identity.sample_idx,
                            item.phase.value,
                            item.packed_rows,
                        ),
                    )
                )
                try:
                    adapter.prepare(
                        ParameterOverlayStepPlan(
                            deployment.attachment.physical.plan_epoch,
                            self._stage(AttachmentOverlayStage.APPLY).anchor_id,
                            self._stage(AttachmentOverlayStage.CAPTURE).anchor_id,
                            rows.phase_rows.total_rows,
                            canonical_assignments,
                        ),
                        handle.deltas,
                )
                except Exception:
                    deployment.compilation.execution.abort_step(handle)
                    raise
                self._state = _OverlayPrepared(handle)

    def complete_forward(self) -> None:
        match self._deployment.kind:
            case AttachmentKind.REQUEST_SAMPLER_TRAINING:
                self._complete_request_forward()
            case AttachmentKind.DENSE_PARAMETER_OVERLAY:
                self._complete_overlay_forward()

    def begin_sampler(self, model: Qwen2ForCausalLM) -> bool:
        match self._state:
            case _Idle():
                return False
            case _RequestPredicted(handle=handle):
                pass
            case _:
                raise RuntimeError("canonical sampler began in the wrong phase")
        execution = self._request_execution()
        stream = torch.cuda.current_stream(execution.device)
        guidance = execution.sampler_guidance(handle, stream)
        pred_dense_logits = model.compute_logits(
            guidance.pred_hidden.to(dtype=model.lm_head.weight.dtype)
        )
        if type(pred_dense_logits) is not torch.Tensor:
            raise RuntimeError("Qwen2 LM head did not produce dense logits")
        self.sampler.begin_guidance(
            SignedPostFilterGuidance(
                guidance,
                execution.config.intervention_beta,
                pred_dense_logits,
            )
        )
        return True

    def complete_sampler(self, active: bool) -> None:
        match self._deployment.kind:
            case AttachmentKind.REQUEST_SAMPLER_TRAINING:
                self._complete_request_sampler(active)
            case AttachmentKind.DENSE_PARAMETER_OVERLAY:
                self._complete_overlay_sampler()

    def commit_token(self) -> None:
        match self._state:
            case _Idle():
                return
            case _RequestSampled(handle=handle):
                execution = self._request_execution()
                try:
                    execution.execute_anchor(handle, self._task(AttachmentTaskRole.TRANSITION).anchor_id)
                    execution.finish_step(handle)
                    execution.collect_completed()
                except Exception:
                    self._state = _Idle()
                    raise
                self._state = _Idle()
            case _OverlaySampled(handle=handle):
                self._overlay_execution().finish_step(handle)
                self._state = _Idle()
            case _:
                raise RuntimeError("canonical token committed before sampling")

    def abort_active(self) -> None:
        if self._sampler is not None:
            self._sampler.abort_guidance()
        if self._capture is not None:
            self._capture.abort_if_active()
        if self._dense_adapter is not None:
            self._dense_adapter.clear()
        match self._state:
            case _RequestPredicted(handle=handle) | _RequestSampled(handle=handle):
                self._request_execution().cancel_step(handle)
            case (
                _OverlayPrepared(handle=handle)
                | _OverlayTransitioned(handle=handle)
                | _OverlaySampled(handle=handle)
            ):
                self._overlay_execution().abort_step(handle)
            case _Idle() | _Capturing():
                pass
        self._state = _Idle()

    def retire_all(self) -> None:
        match self._state:
            case _Idle():
                pass
            case _:
                raise RuntimeError("canonical requests cannot retire during a step")
        identities = tuple(
            self._identities[request_id] for request_id in sorted(self._identities)
        )
        if identities:
            self._retire(identities)
        self._identities.clear()
        if isinstance(self._deployment, _RequestSamplerVllmDeployment):
            self._deployment.compilation.execution.collect_completed()

    def close(self) -> None:
        self.abort_active()
        self.retire_all()
        if self._capture is not None:
            self._capture.close()
        if self._dense_adapter is not None:
            self._dense_adapter.close()

    def _complete_request_forward(self) -> None:
        match self._state:
            case _Capturing(rows=rows):
                pass
            case _Idle():
                return
            case _:
                raise RuntimeError("canonical request forward is out of order")
        capture = self._require_capture()
        hidden = capture.publish()
        if not rows.phase_rows.decode_rows:
            capture.finish()
            self._state = _Idle()
            return
        execution = self._request_execution()
        stream = torch.cuda.current_stream(execution.device)
        try:
            handle = execution.start_step(
                ESampRequestStepInput(rows.phase_rows, rows.sampler_rows, hidden),
                stream,
            )
            execution.execute_anchor(handle, self._task(AttachmentTaskRole.COMPUTE).anchor_id)
        except Exception:
            capture.finish()
            raise
        capture.finish()
        self._state = _RequestPredicted(handle)

    def _complete_overlay_forward(self) -> None:
        match self._state:
            case _OverlayPrepared(handle=handle):
                pass
            case _Idle():
                return
            case _:
                raise RuntimeError("canonical overlay forward is out of order")
        adapter = self._require_dense_adapter()
        captures = adapter.take_captures()
        actual = frozenset((capture.identity, capture.target) for capture in captures)
        expected = frozenset(
            (identity, target)
            for identity in handle.identities
            for target in self._overlay_execution().config.targets
        )
        if actual != expected:
            raise RuntimeError("vLLM did not capture every signed overlay target")
        captures_by_identity = {
            identity: tuple(
                sorted(
                    (capture for capture in captures if capture.identity == identity),
                    key=lambda capture: capture.target,
                )
            )
            for identity in handle.identities
        }
        batch = InPlaceTTTStepBatch(
            tuple(
                InPlaceTTTStepInput(
                    identity,
                    tuple(
                        InPlaceTTTOperandPair(
                            capture.target,
                            InPlaceTTTPhase.PREFILL
                            if capture.phase is HiddenPhase.PREFILL
                            else InPlaceTTTPhase.DECODE,
                            capture.hidden,
                            capture.desired,
                            capture.base_weight,
                        )
                        for capture in captures_by_identity[identity]
                    ),
                )
                for identity in handle.identities
            )
        )
        stream = torch.cuda.current_stream(self._deployment.compilation.execution.device)
        try:
            self._overlay_execution().submit_captured(
                handle,
                CudaInPlaceTTTStepBatchInput.assert_produced_on(batch, stream),
                self._task(AttachmentTaskRole.TRANSITION),
                self._task(AttachmentTaskRole.PUBLISH),
            )
        except Exception:
            adapter.clear()
            self._state = _Idle()
            raise
        adapter.clear()
        self._state = _OverlayTransitioned(handle)

    def _complete_request_sampler(self, active: bool) -> None:
        if not active:
            match self._state:
                case _Idle():
                    return
                case _:
                    raise RuntimeError("canonical sampler skipped an active step")
        match self._state:
            case _RequestPredicted(handle=handle):
                pass
            case _:
                raise RuntimeError("canonical request sampler is out of order")
        self.sampler.finish_guidance()
        execution = self._request_execution()
        execution.execute_anchor(handle, self._deployment.attachment.inputs.request.sampler_anchor_id)
        stream = torch.cuda.current_stream(execution.device)
        execution.complete_sampler(handle, stream)
        self._state = _RequestSampled(handle)

    def _complete_overlay_sampler(self) -> None:
        match self._state:
            case _OverlayTransitioned(handle=handle):
                pass
            case _Idle():
                return
            case _:
                raise RuntimeError("canonical overlay sampler is out of order")
        execution = self._overlay_execution()
        stream = torch.cuda.current_stream(execution.specialization.device)
        execution.complete_sampler(
            handle,
            stream,
            self._stage(AttachmentOverlayStage.RELEASE).anchor_id,
        )
        self._state = _OverlaySampled(handle)

    def _retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        match self._deployment:
            case _RequestSamplerVllmDeployment() as deployment:
                deployment.compilation.execution.retire(identities)
            case _DenseOverlayVllmDeployment() as deployment:
                deployment.compilation.execution.retire(
                    identities,
                    self._stage(AttachmentOverlayStage.FINISH).anchor_id,
                )

    def _request_execution(self) -> CanonicalRequestSamplerExecution:
        return self._request_deployment().compilation.execution

    def _overlay_execution(self) -> CanonicalDenseOverlayExecution:
        return self._overlay_deployment().compilation.execution

    def _request_deployment(self) -> _RequestSamplerVllmDeployment:
        deployment = self._deployment
        if not isinstance(deployment, _RequestSamplerVllmDeployment):
            raise RuntimeError("canonical session does not own request-sampler deployment")
        return deployment

    def _overlay_deployment(self) -> _DenseOverlayVllmDeployment:
        deployment = self._deployment
        if not isinstance(deployment, _DenseOverlayVllmDeployment):
            raise RuntimeError("canonical session does not own dense overlay deployment")
        return deployment

    def _require_capture(self) -> Vllm027EagerPackedHiddenCapture:
        if self._capture is None:
            raise RuntimeError("canonical session has no hidden capture")
        return self._capture

    def _require_dense_adapter(self) -> VllmDenseAddAdapter:
        if self._dense_adapter is None:
            raise RuntimeError("canonical session has no dense overlay adapter")
        return self._dense_adapter

    def _task(self, role: AttachmentTaskRole) -> AttachmentTaskBinding:
        for binding in self._deployment.attachment.inputs.request.task_bindings:
            if binding.role is role:
                return binding
        raise RuntimeError("canonical attachment task role is missing")

    def _stage(self, stage: AttachmentOverlayStage) -> AttachmentOverlayStageAnchor:
        for anchor in self._deployment.attachment.inputs.request.overlay_stage_anchors:
            if anchor.stage is stage:
                return anchor
        raise RuntimeError("canonical attachment stage is missing")


class CanonicalGPUModelRunner(GPUModelRunner):
    """MRv2 runner subclass with one generic canonical attachment slot."""

    __slots__ = (
        "_canonical_deployment",
        "_canonical_session",
        "_canonical_upstream_sampler",
    )

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self._canonical_deployment: CanonicalVllmDeployment | None = None
        self._canonical_session: CanonicalVllmSession | None = None
        self._canonical_upstream_sampler: Sampler | None = None

    @property
    def canonical_deployment(self) -> CanonicalVllmDeployment | None:
        return self._canonical_deployment

    @property
    def overlay_apply_count(self) -> int:
        session = self._canonical_session
        return 0 if session is None else session.apply_count

    @property
    def overlay_updated_apply_count(self) -> int:
        session = self._canonical_session
        return 0 if session is None else session.updated_apply_count

    def attach_canonical(
        self,
        workload: ESampRequestWorkload | InPlaceTTTWorkload,
    ) -> CanonicalVllmDeployment:
        if self._canonical_session is not None:
            raise RuntimeError("canonical runner already owns a workload session")
        match self.model:
            case Qwen2ForCausalLM() as model:
                pass
            case _:
                raise RuntimeError("canonical vLLM currently requires Qwen2")
        upstream = self.sampler
        if type(upstream) is not Sampler:
            raise RuntimeError("canonical vLLM requires the stock MRv2 sampler")
        deployment = _compile_vllm_deployment(workload, self)
        session = CanonicalVllmSession(deployment, model, upstream)
        self._canonical_upstream_sampler = upstream
        self._canonical_deployment = deployment
        self._canonical_session = session
        if deployment.kind is AttachmentKind.REQUEST_SAMPLER_TRAINING:
            self.sampler = session.sampler
        return deployment

    def detach_canonical(self) -> None:
        session = self._canonical_session
        upstream = self._canonical_upstream_sampler
        if session is None or upstream is None:
            raise RuntimeError("canonical runner has no attached session")
        session.close()
        if self.sampler is not upstream:
            self.sampler = upstream
        self._canonical_session = None
        self._canonical_upstream_sampler = None
        self._canonical_deployment = None

    def retire_completed_canonical_requests(self) -> None:
        session = self._require_session()
        with torch.inference_mode(False):
            session.retire_all()

    def abort_canonical_requests(self) -> None:
        session = self._require_session()
        with torch.inference_mode(False):
            session.abort_active()
            session.retire_all()

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_desc: BatchExecutionDescriptor,
    ) -> InputBatch:
        input_batch = super().prepare_inputs(scheduler_output, batch_desc)
        session = self._canonical_session
        if session is not None:
            session.prepare_input(input_batch)
        return input_batch

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        session = self._canonical_session
        if session is None:
            return super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
        if dummy_run or is_profile:
            raise RuntimeError("attached canonical execution rejects runner warmup")
        with torch.inference_mode(False):
            session.before_execute(scheduler_output)
        try:
            result = super().execute_model(
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
            with torch.inference_mode(False), torch.enable_grad():
                session.complete_forward()
        except Exception:
            session.abort_active()
            raise
        return result

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        session = self._canonical_session
        if session is None:
            return super().sample(hidden_states, input_batch, grammar_output)
        match self.model:
            case Qwen2ForCausalLM() as model:
                pass
            case _:
                raise RuntimeError("attached canonical sampler lost its Qwen2 model")
        try:
            active = False
            if self._canonical_deployment is not None and (
                self._canonical_deployment.kind
                is AttachmentKind.REQUEST_SAMPLER_TRAINING
            ):
                active = session.begin_sampler(model)
            sampled = super().sample(hidden_states, input_batch, grammar_output)
            session.complete_sampler(active)
        except Exception:
            session.abort_active()
            raise
        return sampled

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> AsyncOutput | ModelRunnerOutput | None:
        session = self._canonical_session
        if session is None:
            return super().sample_tokens(grammar_output)
        try:
            output = super().sample_tokens(grammar_output)
            with torch.inference_mode(False), torch.enable_grad():
                session.commit_token()
        except Exception:
            session.abort_active()
            raise
        return output

    def shutdown(self) -> None:
        if self._canonical_session is not None:
            self.detach_canonical()
        super().shutdown()

    def _require_session(self) -> CanonicalVllmSession:
        session = self._canonical_session
        if session is None:
            raise RuntimeError("canonical runner has no attached session")
        return session


class CanonicalGPUWorker(Worker):
    """Single-process GPU worker with a canonical runner before model load."""

    def init_device(self) -> None:
        super().init_device()
        if not self.use_v2_model_runner:
            raise RuntimeError("canonical vLLM requires the MRv2 model runner")
        self.model_runner = CanonicalGPUModelRunner(self.vllm_config, self.device)


@dataclass(slots=True)
class CanonicalVllmLlm:
    """Single public handle for signed canonical vLLM workloads."""

    _llm: LLM
    runner: CanonicalGPUModelRunner
    deployment: CanonicalVllmDeployment
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("canonical vLLM handle is already closed")
        self.runner.detach_canonical()
        self._llm.llm_engine.engine_core.shutdown()
        self._closed = True

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
    ) -> list[RequestOutput]:
        if self._closed:
            raise RuntimeError("canonical vLLM handle is closed")
        if (
            type(prompts) not in (str, list)
            or (type(prompts) is str and not prompts)
            or (
                type(prompts) is list
                and (
                    not prompts
                    or any(type(prompt) is not str or not prompt for prompt in prompts)
                )
            )
            or type(sampling_params) is not SamplingParams
            or sampling_params.n != 1
            or type(use_tqdm) is not bool
        ):
            raise ValueError("canonical vLLM generate requires non-empty text and n=1")
        try:
            outputs = self._llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        except Exception:
            self.runner.abort_canonical_requests()
            raise
        self.runner.retire_completed_canonical_requests()
        return outputs


def create_canonical_vllm_llm(
    workload: object,
    config: CanonicalVllmConfig,
) -> CanonicalVllmLlm:
    if (
        type(config) is not CanonicalVllmConfig
        or not isinstance(workload, (ESampRequestWorkload, InPlaceTTTWorkload))
    ):
        raise TypeError("canonical vLLM construction requires exact inputs")
    if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError("canonical vLLM requires vLLM 0.27.1 exactly")
    mp_env = os.environ.get(_MP_ENV)
    v2_env = os.environ.get(_V2_RUNNER_ENV)
    if mp_env not in (None, "0") or v2_env not in (None, "1"):
        raise RuntimeError(
            "canonical vLLM conflicts with explicit MP or legacy-runner environment"
        )
    os.environ[_MP_ENV] = "0"
    os.environ[_V2_RUNNER_ENV] = "1"
    try:
        llm = LLM(
            model=config.model,
            dtype=_model_dtype(config.dtype),
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
            worker_cls=_WORKER_CLASS,
            enforce_eager=True,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            trust_remote_code=config.trust_remote_code,
            enable_prefix_caching=config.enable_prefix_caching,
            enable_chunked_prefill=False,
        )
    finally:
        if mp_env is None:
            os.environ.pop(_MP_ENV)
        else:
            os.environ[_MP_ENV] = mp_env
        if v2_env is None:
            os.environ.pop(_V2_RUNNER_ENV)
        else:
            os.environ[_V2_RUNNER_ENV] = v2_env

    match llm.llm_engine.engine_core:
        case InprocClient() as client:
            pass
        case _:
            raise RuntimeError("canonical vLLM did not construct an in-process client")
    match client.engine_core.model_executor:
        case UniProcExecutor() as executor:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct a uni-process executor")
    match executor.driver_worker.worker:
        case CanonicalGPUWorker() as worker:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct its worker")
    match worker.model_runner:
        case CanonicalGPUModelRunner() as runner:
            pass
        case _:
            client.shutdown()
            raise RuntimeError("canonical vLLM did not construct its MRv2 runner")
    try:
        deployment = runner.attach_canonical(workload)
    except Exception:
        client.shutdown()
        raise
    return CanonicalVllmLlm(llm, runner, deployment)


def _compile_vllm_deployment(
    workload: ESampRequestWorkload | InPlaceTTTWorkload,
    runner: GPUModelRunner,
) -> _ClosedVllmDeployment:
    _require_runner(runner)
    match runner.model:
        case Qwen2ForCausalLM() as model:
            pass
        case _:
            raise RuntimeError("canonical vLLM currently requires Qwen2")
    if model.quant_config is not None:
        raise RuntimeError("canonical vLLM requires an unquantized model")
    if type(workload) is ESampRequestWorkload:
        return _compile_sampler_training(workload, runner, model)
    if type(workload) is InPlaceTTTWorkload:
        return _compile_dense_overlay_deployment(workload, runner, model)
    raise TypeError("canonical vLLM received an unsupported workload")


def _require_runner(runner: GPUModelRunner) -> None:
    if (
        _VLLM_VERSION != _SUPPORTED_VLLM_VERSION
        or not runner.model_config.enforce_eager
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
        or type(runner.sampler) is not Sampler
    ):
        raise RuntimeError(
            "canonical vLLM requires eager in-process TP1/PP1/DP1 non-spec MRv2"
        )


def _base_probe(
    runner: GPUModelRunner,
    *,
    model_fingerprint: str,
    hidden_size: int,
    hidden_resources: tuple[AttachmentHiddenResource, ...],
    capabilities: tuple[str, ...],
) -> BackendPlacementProbe:
    device_ordinal = runner.device.index
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise RuntimeError("canonical vLLM requires an ordinal CUDA device")
    return BackendPlacementProbe(
        adapter_id=_ADAPTER_ID,
        backend_abi_digest=_digest(_BACKEND_ABI_FIELDS),
        runner_fingerprint=_digest(
            (
                _ADAPTER_ID,
                f"device=cuda:{device_ordinal}",
                f"max_num_tokens={runner.max_num_tokens}",
                f"max_num_reqs={runner.max_num_reqs}",
                f"dtype={runner.dtype}",
                f"seed={runner.model_config.seed}",
            )
        ),
        model_fingerprint=model_fingerprint,
        topology=AttachmentTopology.IN_PROCESS,
        graph_mode=AttachmentGraphMode.EAGER,
        device_kind=StateDeviceKind.CUDA,
        device_ordinal=device_ordinal,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_dtype=_tensor_dtype(runner.dtype),
        hidden_size=hidden_size,
        max_packed_rows=runner.max_num_tokens,
        hidden_resources=hidden_resources,
        capabilities=capabilities,
    )


def _compile_sampler_training(
    workload: ESampRequestWorkload,
    runner: GPUModelRunner,
    model: Qwen2ForCausalLM,
) -> _RequestSamplerVllmDeployment:
    if workload.config.key_fields != (RequestKeyField.REQUEST_ID,):
        raise RuntimeError("canonical request sampler requires request-id state keys")
    layer_count = len(model.model.layers)
    source = _hidden_binding(workload.config.source_layer, layer_count)
    target = _hidden_binding(workload.config.target_layer, layer_count)
    if source.locator_key == target.locator_key:
        raise ValueError("ESamp hidden layers must identify two Qwen2 blocks")
    hidden_size = model.config.hidden_size
    if type(hidden_size) is not int or hidden_size < 1:
        raise RuntimeError("Qwen2 hidden size is not a positive integer")
    device_ordinal = runner.device.index
    free_device_bytes, _total_device_bytes = torch.cuda.mem_get_info(runner.device)
    timeline = SchedulerTimeline(
        anchors=(
            TimelineAnchor(AnchorSymbol("model.execute"), 0),
            TimelineAnchor(AnchorSymbol("sampler.pre_draw"), 1),
            TimelineAnchor(AnchorSymbol("token.commit"), 2),
            TimelineAnchor(AnchorSymbol("next.prepare"), 3),
        ),
        capabilities=("compute/cuda@1", "state/dense_page_cuda@1"),
    )
    specialization = ESampRequestSpecialization(
        hidden_size=hidden_size,
        tensor_dtype=TensorDType.FLOAT32,
        initialization_seed=runner.model_config.seed,
        hidden_bindings=ESampRequestPackedHiddenBindings(source, target),
        device_kind=StateDeviceKind.CUDA,
        device_ordinal=device_ordinal,
    )
    compilation = workload.compile(
        specialization,
        ESampRequestCompileOptions(
            timeline=timeline,
            providers=(
                PageBundleProviderFacts(
                    "canonical.cuda_dense_page@1",
                    max_live_versions_per_lineage=3,
                    device_kind=StateDeviceKind.CUDA,
                    device_ordinal=device_ordinal,
                ),
            ),
            max_admitted_lineages=runner.max_num_reqs,
            state_memory_budget_bytes=free_device_bytes,
        ),
    )
    if compilation.execution.device.type != "cuda":
        raise RuntimeError("vLLM facts did not select CUDA request-sampler execution")
    sorted_hidden_resources = sorted(
        (
            AttachmentHiddenResource(source, HiddenPhase.DECODE),
            AttachmentHiddenResource(target, HiddenPhase.DECODE),
        ),
        key=lambda resource: resource.resource_key,
    )
    hidden_resources = (sorted_hidden_resources[0], sorted_hidden_resources[1])
    probe = _base_probe(
        runner,
        model_fingerprint=_digest(
            (
                f"model={runner.model_config.model}",
                f"revision={runner.model_config.revision or ''}",
                "class=vllm.model_executor.models.qwen2.Qwen2ForCausalLM",
                f"hidden_size={hidden_size}",
                f"layers={layer_count}",
                f"source={source.module_path}",
                f"target={target.module_path}",
            )
        ),
        hidden_size=hidden_size,
        hidden_resources=hidden_resources,
        capabilities=_REQUEST_CAPABILITIES,
    )
    attachment = compile_request_sampler_attachment(
        compilation.plan_bundle,
        probe,
        workload.request_sampler_attachment_spec(),
    )
    return _RequestSamplerVllmDeployment(compilation, attachment)


def _hidden_binding(layer: int, layer_count: int) -> PackedHiddenResourceBinding:
    module_index = layer if layer >= 0 else layer_count + layer
    if module_index < 0 or module_index >= layer_count:
        raise ValueError("hidden layer does not identify a Qwen2 block")
    return PackedHiddenResourceBinding(
        str(layer),
        HiddenLayerKind.INDEX,
        HiddenSite.BLOCK_OUTPUT,
        f"model.layers.{module_index}",
    )


def _compile_dense_overlay_deployment(
    workload: InPlaceTTTWorkload,
    runner: GPUModelRunner,
    model: Qwen2ForCausalLM,
) -> _DenseOverlayVllmDeployment:
    targets = inspect_dense_add_targets(model, workload.config.targets)
    if any(target.dtype is not runner.dtype for target in targets):
        raise RuntimeError("dense overlay target dtype differs from the runner")
    device_ordinal = runner.device.index
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
        _tensor_dtype(runner.dtype),
        runner.device,
    )
    compilation = workload.compile(
        specialization,
        InPlaceTTTCompileOptions(
            CanonicalRuntimeFacts(runner.device).timeline(
                (
                    "state/fenced_reclaim@1",
                    "transition/full_root@1",
                )
            ),
            (
                PageBundleProviderFacts(
                    "canonical.cuda.full_root@1",
                    max_live_versions_per_lineage=3,
                    device_kind=StateDeviceKind.CUDA,
                    device_ordinal=device_ordinal,
                ),
            ),
            max_admitted_lineages=runner.max_num_reqs,
            state_memory_budget_bytes=free_device_bytes,
        ),
    )
    if compilation.execution.device.type != "cuda":
        raise RuntimeError("vLLM facts did not select CUDA dense overlay execution")
    probe = _base_probe(
        runner,
        model_fingerprint=_digest(
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
        ),
        hidden_size=max(target.input_size for target in targets),
        hidden_resources=(),
        capabilities=_OVERLAY_CAPABILITIES,
    )
    attachment = compile_parameter_overlay_attachment(
        compilation.plan_bundle, probe, compilation.state_layout.overlay_bindings,
        StateResource("dense_fast_weights"),
    )
    return _DenseOverlayVllmDeployment(compilation, attachment)


__all__ = [
    "CanonicalGPUModelRunner",
    "CanonicalGPUWorker",
    "CanonicalVllmConfig",
    "CanonicalVllmDType",
    "CanonicalVllmDeployment",
    "CanonicalVllmLlm",
    "CanonicalVllmSession",
    "create_canonical_vllm_llm",
]

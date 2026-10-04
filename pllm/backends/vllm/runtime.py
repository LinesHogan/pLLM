"""Workload-neutral public handle and shared MRv1/MRv2 attachment lifecycle."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from time import perf_counter_ns
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, overload
from types import TracebackType

import torch

from pllm.config import ModelConfig, RunnerKind, VllmDType
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.state_execution import StateDiagnostics, StateTaskContext
from pllm.authoring import BackendCompileFacts, BackendDeployment, BackendRequirements, BackendWorkload, AuxiliaryInput, StateTensorUpdate, map_auxiliary, DataT, ResultT, TensorStepInput
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.host_runtime import CompletionCapture, CompletionInput
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.trace import ExecutionTrace, TraceSelection, TraceSnapshot, TRACE_SCHEMA_VERSION
from pllm.execution.tokenwise_tuning import TokenwiseTuning
from pllm.execution.tokenwise_region import compile_tokenwise_region
from pllm.contracts.region_program import TensorType
from pllm.contracts.region_proposal import REGION_STREAMS_CAPABILITY, RegionScheduleProposal
from pllm.contracts.region_selection import RegionSelection
from pllm.contracts.feedback import BarrierState, Completion, CompletionBatch, CompletionFeedback, FinishReason, RetryAction

from pllm.execution.backend_attachment import attachment_buffer_bytes
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.replay import VllmLayerReplay, VllmTokenwiseReplay
from pllm.backends.vllm.region_kernels import qwen_region_kernels
from pllm.backends.vllm.region_layers import VllmDecoderBodies
from pllm.backends.vllm.parameter_overlay import (
    VllmParameterOverlay,
    inspect_parameter_targets,
)
from pllm.backends.vllm.hidden_capture import Vllm027EagerPackedHiddenCapture
from pllm.backends.vllm.packed_rows import encode_mrv2_rows
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.backends.vllm.sampler import (
    Vllm027Sampler,
    SamplerGuidance,
    SignedSamplerGuidance,
)
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentKind,
    AttachmentTopology,
    BackendPlacementProbe,
    CausalModelDomain,
    DecodeWindow,
    SignedBackendAttachment,
)
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.semantic_graph import (
    AnchorId, ConditionalLayerReplay, SamplerFeedbackInputSource, SchedulerTimeline, StateScope, HiddenInputSource, TokenwiseLayerReplay, TaskId,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec

from vllm import RequestOutput, SamplingParams, __version__ as _VLLM_VERSION
from vllm.config import CacheConfig, VllmConfig
from vllm.v1.sample.sampler import Sampler as Mrv1Sampler
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
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as UpstreamModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu_worker import Worker

if TYPE_CHECKING:
    from pllm.backends.vllm.mrv1 import Mrv1Runner


_MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"
_V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
_WORKER_CLASS = "pllm.worker.GPUWorker"
_SUPPORTED_VLLM_VERSION = "0.27.1"
_BACKEND_ABI_FIELDS = (
    "vllm=0.27.1",
    "mode=in_process|engine_core/eager/tp1/pp1/dp1/non_spec",
    "rows=one_logit_per_request",
    "attachment=shared_plan_v7",
    "attachments=sampler_training|parameter_overlay|activation_patch",
    "activation=phase_windows/exact_step_pins@1",
    "model=causal_indexed_layer@5",
)
_REQUEST_CAPABILITIES = (
    "producer/packed_hidden@1",
    "sampler/post_filter_exact@1",
    "sampler/pre_filter_add@1",
)
_OVERLAY_CAPABILITIES = (
    "operand_capture/mlp_down_projection_pair@1",
    "overlay/dense_add@1",
    "overlay/low_rank_add@1",
)


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
            raise RuntimeError("vLLM workload runner dtype is unsupported")


def _model_dtype(dtype: VllmDType) -> Literal["bfloat16", "float16", "float32"]:
    match dtype:
        case VllmDType.BFLOAT16:
            return "bfloat16"
        case VllmDType.FLOAT16:
            return "float16"
        case VllmDType.FLOAT32:
            return "float32"


class _RunnerSession(Protocol):
    @property
    def trace(self) -> ExecutionTrace: ...
    @property
    def sampler(self) -> SamplerGuidance: ...
    @property
    def next_step_id(self) -> int: ...
    @property
    def apply_count(self) -> int: ...
    @property
    def updated_apply_count(self) -> int: ...
    def stage_input(self, batch: AuxiliaryInput[torch.Tensor] | None, identities: tuple[RequestIdentity, ...]) -> None: ...
    def bind_readonly_input(self, receiver: ReadonlyStepReceiver) -> None: ...
    def identity(self, request_id: str) -> RequestIdentity: ...
    def before_execute(self, scheduler_output: SchedulerOutput) -> None: ...
    def prepare_input(self, rows: PackedSamplerRowBatch) -> None: ...
    def complete_forward(self) -> None: ...
    def begin_sampler(self, model: Qwen2ForCausalLM) -> bool: ...
    def complete_sampler(self, active: bool) -> None: ...
    def commit_token(self) -> None: ...
    def abort_active(self) -> None: ...
    def retire_all(self) -> None: ...
    def close(self) -> None: ...
    def diagnostics(self) -> StateDiagnostics: ...
    def begin_tuning(self, proposals: tuple[RegionScheduleProposal, ...]) -> int: ...
    def tuning_candidate(self, index: int) -> None: ...
    def finish_tuning(self, samples: tuple[tuple[int, ...], ...]) -> RegionSelection: ...
    def abort_tuning(self) -> None: ...
    @property
    def attachment(self) -> SignedBackendAttachment: ...


@dataclass(slots=True)
class _PackedStep(Generic[DataT]):
    rows: PackedSamplerRowBatch
    invocation: TaskGraphInvocation[StateTaskContext[DataT]] | None = None


@dataclass(frozen=True, slots=True)
class ReadonlyStepInput:
    """Internal handoff of complete received grants for one original engine step."""

    plan_epoch: int
    step_id: int
    roots: tuple[BorrowedStateRoot, ...]


ReadonlyStepReceiver = Callable[[PackedSamplerRowBatch, torch.cuda.Stream], ReadonlyStepInput]


class VllmSession(Generic[DataT, ResultT]):
    """Model primitives over one original signed tensor-state invocation."""

    def __init__(self, deployment: BackendDeployment[DataT, ResultT],
                 model: Qwen2ForCausalLM, sampler: SamplerGuidance | None, max_sampler_rows: int) -> None:
        attachment = deployment.attachment
        plan = attachment.inputs.plan
        self._capture_feedback = any(isinstance(port.source, SamplerFeedbackInputSource)
            for task in plan.semantic_graph.tasks for port in task.inputs)
        required = _input_reservations(attachment.inputs.probe, plan.timeline, max_sampler_rows,
            model.lm_head.weight.shape[0], attachment.inputs.request.sampler_stage if sampler is not None else None,
            self._capture_feedback)
        buffers = attachment_buffer_bytes(attachment.inputs.probe, plan.semantic_graph,
            plan.tensor_layout_request, attachment.inputs.request)
        required = tuple(value + buffers for value in required)
        if len(plan.timeline.reserved_bytes) != len(required) or any(
            actual < minimum for actual, minimum in zip(plan.timeline.reserved_bytes, required, strict=True)):
            raise ValueError("signed timeline underfunds the backend input/sampler/effect buffers")
        self._deployment = deployment
        self.trace = deployment.execution.trace
        probe = attachment.inputs.probe
        resources = tuple(resource.binding for resource in probe.hidden_resources)
        self._capture = Vllm027EagerPackedHiddenCapture(model=model, resources=resources,
            plan_epoch=attachment.physical.plan_epoch, max_packed_rows=probe.max_packed_rows,
            hidden_size=probe.hidden_size) if resources else None
        overlay_bindings = attachment.inputs.request.overlay_bindings
        self._overlay_import = bool(overlay_bindings) and (
            plan.semantic_graph.tasks[attachment.inputs.request.task_bindings[0].task_id].timing.after == "request.prepare")
        self._overlay = VllmParameterOverlay(model, overlay_bindings,
            capture=not self._overlay_import) if overlay_bindings else None
        self._activation = VllmActivationPatch(model, attachment.inputs.request.activation_bindings,
            attachment.inputs.request.state_resource) if (
            attachment.inputs.request.activation_bindings) else None
        self._decode_injection = any(binding.decode is DecodeWindow.ALL for binding in attachment.inputs.request.activation_bindings)
        self._token_frontiers: dict[RequestIdentity, int] = {}
        self._readonly_receiver: ReadonlyStepReceiver | None = None
        self._received_readonly: ReadonlyStepInput | None = None
        self._input_failed = False
        tasks = tuple(task for task in attachment.inputs.plan.semantic_graph.tasks if task.replay is not None)
        self._replay: VllmLayerReplay | VllmTokenwiseReplay[DataT, ResultT] | None = None
        if tasks:
            task, = tasks
            if probe.causal_domain is None or probe.causal_domain.layers != len(model.model.layers):
                raise ValueError("backend replay requires its signed causal model domain")
            if isinstance(task.replay, ConditionalLayerReplay) and self._overlay is not None:
                self._replay = VllmLayerReplay(model, task.replay, self._overlay, probe.causal_domain)
            elif isinstance(task.replay, TokenwiseLayerReplay):
                self._replay = VllmTokenwiseReplay(model, task, deployment.execution,
                    attachment.inputs.request.task_bindings[0].anchor_id, source_imports=probe.region_imports,
                    function_bodies=probe.region_bodies)
            else:
                raise ValueError("backend replay does not match its signed primitives")
        self._sampler = sampler
        self._identities: dict[str, RequestIdentity] = {}
        self._pending_identities: dict[str, RequestIdentity] = {}
        self._next_step_id = 0
        self._step: _PackedStep[DataT] | None = None
        self._auxiliary: AuxiliaryInput[torch.Tensor] | None = None
        self._tuning: TokenwiseTuning[TensorStepInput, DataT, ResultT] | None = None
        execution = deployment.execution
        if execution.directory.scope is StateScope.INSTANCE:
            execution.admit((), torch.cuda.current_stream(execution.device))

    def stage_input(self, batch: AuxiliaryInput[torch.Tensor] | None, identities: tuple[RequestIdentity, ...]) -> None:
        self._require_input_live()
        if self._tuning is not None and batch is not None:
            raise ValueError("private native tuning cannot consume user auxiliary input")
        if self._auxiliary is not None or self._step is not None or self._pending_identities or self._identities:
            raise RuntimeError("prior canonical training input is still active")
        validate_auxiliary(batch, self._deployment.attachment.inputs.plan, SharedTensor.describe)
        self._auxiliary = batch
        self._pending_identities = {identity.request_id: identity for identity in identities}

    def bind_readonly_input(self, receiver: ReadonlyStepReceiver) -> None:
        """Install example-owned transport before requests; it supplies no model mathematics."""
        self._require_input_live()
        if (self._readonly_receiver is not None or self._step is not None or self._next_step_id
            or self._identities or self._pending_identities or self._auxiliary is not None or self._tuning is not None
            or self._activation is None or all(state.resource in self._deployment.execution.arena.owned_resources
                for state in self.attachment.inputs.plan.semantic_graph.states)):
            raise RuntimeError("read-only step input requires an idle borrowed activation session")
        self._readonly_receiver = receiver

    def _require_input_live(self) -> None:
        if self._input_failed:
            raise RuntimeError("canonical step input is fail-stopped; received grants remain retained")

    def _receive_readonly(self, rows: PackedSamplerRowBatch, stream: torch.cuda.Stream) -> None:
        if self._readonly_receiver is None:
            return
        try:
            self._received_readonly = self._readonly_receiver(rows, stream)
            received = self._received_readonly
            if (received.plan_epoch, received.step_id) != (rows.plan_epoch, rows.step_id):
                raise ValueError("read-only input differs from the original epoch/step; grants retained")
            if any(root.stream != stream for root in received.roots):
                raise ValueError("read-only input differs from the receiving stream; grants retained")
            self._deployment.execution.stage_readonly(received.roots)
            self._received_readonly = None  # Ownership is now the original execution's pending/active borrow.
        except BaseException:
            self._input_failed = True
            raise

    def identity(self, request_id: str) -> RequestIdentity:
        return self._identities[request_id]

    @property
    def attachment(self) -> SignedBackendAttachment:
        return self._deployment.attachment

    def diagnostics(self) -> StateDiagnostics:
        return self._deployment.execution.diagnostics()

    def _require_tuning_idle(self) -> None:
        if self._step is not None or self._identities or self._pending_identities or self._auxiliary is not None:
            raise RuntimeError("native tuning requires all original requests to finish and retire")

    def begin_tuning(self, proposals: tuple[RegionScheduleProposal, ...]) -> int:
        self._require_tuning_idle()
        if self._tuning is not None or self._next_step_id != 0:
            raise RuntimeError("native tuning is only available before the first request")
        if (not isinstance(self._replay, VllmTokenwiseReplay)
            or self.attachment.inputs.plan.semantic_graph.states
            or self.attachment.inputs.plan.timeline.region_policy.schedule_policy == "serial"):
            return 0
        identity = self.attachment.inputs.request.task_bindings[0].implementation_id
        tuning = TokenwiseTuning(self._deployment.execution, identity, proposals,
            stream_lanes=2 if REGION_STREAMS_CAPABILITY in self.attachment.inputs.probe.capabilities else 1)
        if tuning.count == 1:
            tuning.abort()
            return 0
        self._tuning = tuning
        return tuning.count

    def tuning_candidate(self, index: int) -> None:
        self._require_tuning_idle()
        if self._tuning is None:
            raise RuntimeError("native tuning has not started")
        self._tuning.candidate(index)

    def finish_tuning(self, samples: tuple[tuple[int, ...], ...]) -> RegionSelection:
        self._require_tuning_idle()
        if self._tuning is None:
            raise RuntimeError("native tuning has not started")
        result = self._tuning.finish(samples)
        self._tuning = None
        return result

    def abort_tuning(self) -> None:
        self._require_tuning_idle()
        if self._tuning is not None:
            self._tuning.abort()
            self._tuning = None

    @property
    def sampler(self) -> SamplerGuidance:
        if self._sampler is None:
            raise RuntimeError("canonical session has no sampler intervention")
        return self._sampler

    @property
    def apply_count(self) -> int:
        return 0 if self._overlay is None else self._overlay.apply_count

    @property
    def updated_apply_count(self) -> int:
        return 0 if self._overlay is None else self._overlay.updated_apply_count

    def before_execute(self, scheduler_output: SchedulerOutput) -> None:
        self._require_input_live()
        if self._step is not None:
            raise RuntimeError("prior vLLM workload step is still active")
        if self._decode_injection and (scheduler_output.preempted_req_ids or scheduler_output.scheduled_cached_reqs.resumed_req_ids):
            self._input_failed = True
            raise RuntimeError("per-decode activation injection and vLLM preemption cannot be enabled together")
        execution = self._deployment.execution
        execution.arena.collect_completed()
        finished = tuple(self._identities[request_id] for request_id in sorted(
            self._identities.keys() & scheduler_output.finished_req_ids))
        self._retire(finished)
        pending: list[RequestIdentity] = []
        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id in self._identities:
                continue
            if type(request.req_id) is not str or not request.req_id:
                raise RuntimeError("vLLM scheduled an invalid request id")
            if type(request.sampling_params) is not SamplingParams or request.sampling_params.n != 1:
                raise RuntimeError("vLLM workload requires sampling n=1")
            pending.append(self._pending_identities[request.req_id])
        if pending:
            if execution.directory.scope is StateScope.REQUEST:
                execution.admit(tuple(pending), torch.cuda.current_stream(execution.device))
            for identity in pending:
                self._identities[identity.request_id] = identity
                self._pending_identities.pop(identity.request_id)

    @property
    def next_step_id(self) -> int:
        return self._next_step_id

    def prepare_input(self, rows: PackedSamplerRowBatch) -> None:
        self._require_input_live()
        if self._tuning is not None:
            self._tuning.record_rows(rows.row_batch.rows)
        if self._step is not None:
            raise RuntimeError("canonical input preparation overlaps a step")
        if rows.step_id != self._next_step_id or rows.plan_epoch != self.attachment.physical.plan_epoch:
            raise ValueError("backend rows differ from the original execution step")
        if rows.row_batch.total_rows > self.attachment.inputs.probe.max_packed_rows:
            raise ValueError("packed input exceeds the signed backend row capacity")
        if self._decode_injection:
            if any(row.token_offset < self._token_frontiers.get(row.identity, 0) for row in rows.row_batch.rows):
                self._input_failed = True
                raise RuntimeError("per-decode activation injection and vLLM preemption/history recomputation cannot be enabled together")
            for row in rows.row_batch.rows:
                self._token_frontiers[row.identity] = max(self._token_frontiers.get(row.identity, 0), row.token_offset + 1)
        requests = rows.identities
        self._next_step_id += 1
        step = _PackedStep[DataT](rows)
        self._step = step
        if self._capture is not None:
            self._capture.begin(rows.row_batch)
        if isinstance(self._replay, VllmTokenwiseReplay):
            execution = self._deployment.execution
            invocation = execution.start_step(TensorStepInput(rows.row_batch, rows),
                torch.cuda.current_stream(execution.device))
            step.invocation = invocation
            self._advance(invocation, "request.prepare")
            self._replay.prepare(rows.row_batch, invocation)
            invocation.context.replay = self._replay
            return
        adapter = self._overlay
        if adapter is None and self._activation is None:
            return
        execution = self._deployment.execution
        identities = tuple(sorted(requests,
            key=lambda identity: (identity.request_id, identity.prompt_idx, identity.sample_idx)))
        stream = torch.cuda.current_stream(execution.device)
        if execution.directory.scope is StateScope.INSTANCE or self._activation is not None or self._overlay_import:
            self._receive_readonly(rows, stream)
            value = TensorStepInput(rows.row_batch, rows, auxiliary=self._auxiliary)
            invocation = execution.start_step(value, stream)
            if execution.directory.scope is StateScope.INSTANCE or not self._pending_identities:
                self._auxiliary = None
            step.invocation = invocation
            self._advance(invocation, "request.prepare")
            # Offline publication precedes admission; existing request pins remain unchanged.
            pending = tuple(identity for identity in identities if execution.directory.scope is StateScope.INSTANCE
                and not execution.directory.has_inference_pin(identity))
            if pending:
                execution.admit(pending, stream)
        else:
            invocation = execution.begin(rows.row_batch.request_rows, stream)
            step.invocation = invocation
        if adapter is not None:
            adapter.prepare(rows.row_batch, invocation.context, execution.directory)
        if self._activation is not None:
            self._activation.prepare(rows.row_batch, invocation.context, execution.directory)
        if isinstance(self._replay, VllmLayerReplay):
            self._replay.prepare(rows.row_batch, invocation.context)
            invocation.context.replay = self._replay

    def complete_forward(self) -> None:
        step = self._step
        if step is None:
            return
        execution = self._deployment.execution
        stream = torch.cuda.current_stream(execution.device)
        operands = () if self._overlay is None else self._overlay.take_captures()
        value = TensorStepInput(step.rows.row_batch, step.rows, operands=operands, auxiliary=self._auxiliary)
        if self._capture is not None:
            value = self._capture.publish(value)
        phases = {resource.phase for resource in self._deployment.attachment.inputs.probe.hidden_resources}
        if step.invocation is None and not operands and not any(
            row.phase in phases for row in step.rows.row_batch.rows):
            if self._capture is not None:
                self._capture.finish()
            self._step = None
            return
        if step.invocation is None:
            step.invocation = execution.start_step(value, stream)
            self._auxiliary = None
        elif step.invocation.context.value is None:
            execution.submit_input(step.invocation, value, CudaStreamReadiness(stream))
        self._advance(step.invocation, "sampler.pre_draw")
        self._finish_effects()

    def _finish_effects(self) -> None:
        """End-of-forward lifecycle for every active optional primitive."""
        if self._capture is not None:
            self._capture.finish()
        for effect in (self._overlay, self._activation, self._replay):
            if effect is not None:
                effect.clear()

    def _abort_effects(self) -> None:
        if self._capture is not None:
            self._capture.abort_if_active()
        for effect in (self._overlay, self._activation):
            if effect is not None:
                effect.clear()

    def _close_effects(self) -> None:
        for effect in (self._capture, self._overlay, self._activation, self._replay):
            if effect is not None:
                effect.close()

    def begin_sampler(self, model: Qwen2ForCausalLM) -> bool:
        if self._step is None or self._sampler is None:
            return False
        execution = self._deployment.execution
        guidance = execution.sampler_guidance(self._invocation(), torch.cuda.current_stream(execution.device))
        if guidance.stage is not self._deployment.attachment.inputs.request.sampler_stage:
            raise ValueError("guidance stage differs from its signed attachment")
        logits = (guidance.values if guidance.stage is SamplerStage.PACKED_PRE_FILTER_ADD else
            model.compute_logits(guidance.values.to(dtype=model.lm_head.weight.dtype)))
        if type(logits) is not torch.Tensor:
            raise RuntimeError("Qwen2 LM head did not produce dense logits")
        self._sampler.begin_guidance(SignedSamplerGuidance(guidance, logits, self._capture_feedback))
        return True

    def complete_sampler(self, active: bool) -> None:
        if self._step is None:
            if active:
                raise RuntimeError("sampler completion has no active packed step")
            return
        feedback = None
        if self._sampler is not None:
            if not active:
                raise RuntimeError("canonical sampler skipped an active intervention")
            feedback = self._sampler.finish_guidance()
        execution = self._deployment.execution
        invocation = self._invocation()
        execution.execute_anchor(invocation, self._deployment.attachment.inputs.request.sampler_anchor_id)
        execution.complete_sampler(invocation, torch.cuda.current_stream(execution.device), feedback=feedback)

    def commit_token(self) -> None:
        if self._step is None:
            return
        execution = self._deployment.execution
        invocation = self._invocation()
        self._advance(invocation, "next.prepare")
        execution.finish_step(invocation)
        self._step = None
        execution.arena.collect_completed()

    def abort_active(self) -> None:
        if self._replay is not None:
            self._replay.clear()
        if self._sampler is not None:
            self._sampler.abort_guidance()
        self._abort_effects()
        self._deployment.execution.cancel_active()
        self._auxiliary = None
        self._step = None

    def _retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        execution = self._deployment.execution
        held = (tuple(identity for identity in identities if execution.directory.has_inference_pin(identity))
            if execution.directory.scope is StateScope.INSTANCE else identities)
        if held:
            execution.retire(held)
        for identity in identities:
            self._identities.pop(identity.request_id)
            self._token_frontiers.pop(identity, None)

    def retire_all(self) -> None:
        if self._step is not None:
            raise RuntimeError("canonical requests cannot retire during a step")
        if self._auxiliary is not None:
            raise RuntimeError("generation ended before the signed training input was consumed")
        self._retire(tuple(self._identities[request_id] for request_id in sorted(self._identities)))
        self._pending_identities.clear()
        self._deployment.execution.arena.collect_completed(wait=True)

    def close(self) -> None:
        self.abort_active()
        self.retire_all()
        self.abort_tuning()
        execution = self._deployment.execution
        if execution.directory.scope is StateScope.INSTANCE:
            execution.retire(())
        else:
            execution.retire_dormant()
        self._close_effects()

    def _invocation(self) -> TaskGraphInvocation[StateTaskContext[DataT]]:
        if self._step is None or self._step.invocation is None:
            raise RuntimeError("model inputs have not started the signed invocation")
        return self._step.invocation

    def _advance(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], symbol: str) -> None:
        execution = self._deployment.execution
        for anchor in execution.arena.plan.timeline.anchors:
            if anchor.ordinal > invocation.last_anchor_id:
                execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
            if anchor.symbol == symbol:
                return
        raise RuntimeError("backend event is absent from the original timeline")

SamplerT = TypeVar("SamplerT", Sampler | None, Mrv1Sampler)


class Runner(Generic[SamplerT]):
    """The original attachment slot and cleanup, shared by concrete runner adapters."""

    sampler: SamplerT
    model: torch.nn.Module
    cache_config: CacheConfig
    runner_kind: RunnerKind
    max_num_reqs: int

    def _initialize_workload(self, topology: AttachmentTopology) -> None:
        self.workload_topology = topology
        self._workload_session: _RunnerSession | None = None
        self._workload_upstream_sampler: SamplerT | None = None
        self.completion_capture: CompletionCapture | None = None

    @property
    def overlay_apply_count(self) -> int:
        session = self._workload_session
        return 0 if session is None else session.apply_count

    @property
    def overlay_updated_apply_count(self) -> int:
        session = self._workload_session
        return 0 if session is None else session.updated_apply_count

    def attach_workload(
        self,
        deployment: BackendDeployment[DataT, ResultT],
    ) -> None:
        if self._workload_session is not None:
            raise RuntimeError("canonical runner already owns a workload session")
        if self.cache_config.enable_prefix_caching and deployment.attachment.inputs.request.activation_bindings:
            raise ValueError("signed activation injection does not support shared prefix caching")
        if self.cache_config.enable_prefix_caching and any(task.replay is not None
            for task in deployment.attachment.inputs.plan.semantic_graph.tasks):
            raise ValueError("signed KV replay does not support shared prefix caching")
        match self.model:
            case Qwen2ForCausalLM() as model:
                pass
            case _:
                raise RuntimeError("vLLM workload currently requires Qwen2")
        upstream = self.sampler
        if upstream is None or type(upstream) not in (Sampler, Mrv1Sampler):
            raise RuntimeError("vLLM workload requires the original stock sampler")
        adapter = self._make_sampler(upstream) if (
            deployment.attachment.inputs.request.attachment_kind is AttachmentKind.SAMPLER_TRAINING) else None
        if adapter is not None and not isinstance(adapter, SamplerGuidance):
            raise TypeError("backend sampler must share the original guidance lifecycle")
        session = VllmSession(deployment, model, adapter, self.max_num_reqs)
        self._workload_upstream_sampler = upstream
        self._workload_session = session
        if adapter is not None:
            self.sampler = adapter

    def detach_workload(self) -> None:
        session = self._workload_session
        upstream = self._workload_upstream_sampler
        if session is None or upstream is None:
            raise RuntimeError("canonical runner has no attached session")
        session.close()
        if self.completion_capture is not None:
            self.completion_capture.synchronize()
        if self.sampler is not upstream:
            self.sampler = upstream
        self._workload_session = None
        self._workload_upstream_sampler = None
        self.completion_capture = None

    def stage_workload_input(self, batch: AuxiliaryInput[torch.Tensor] | None, identities: tuple[RequestIdentity, ...],
                              completion: torch.Tensor | None = None) -> None:
        session = self.require_session()
        if self.completion_capture is not None:
            raise RuntimeError("previous completion capture has not finished")
        if completion is not None and completion.shape != (len(identities), session.attachment.inputs.probe.hidden_size):
            raise ValueError("completion destination differs from the original backend probe")
        capture = None if completion is None else CompletionCapture(identities, completion)
        session.stage_input(batch, identities)
        self.completion_capture = capture

    def retire_completed_requests(self) -> None:
        session = self.require_session()
        if self.completion_capture is not None:
            self.completion_capture.require_complete()
            self.completion_capture.synchronize()
        with torch.inference_mode(False):
            session.retire_all()
        self.completion_capture = None

    def worker_state(self) -> BackendWorkerState:
        session = self.require_session()
        return BackendWorkerState(os.getpid(), session.diagnostics())

    @property
    def trace(self) -> ExecutionTrace:
        return self.require_session().trace

    def abort_workload_requests(self) -> None:
        session = self.require_session()
        with torch.inference_mode(False):
            session.abort_active()
            session.retire_all()
        if self.completion_capture is not None:
            self.completion_capture.synchronize()
        self.completion_capture = None

    def require_session(self) -> _RunnerSession:
        session = self._workload_session
        if session is None:
            raise RuntimeError("canonical runner has no attached session")
        return session

    def _make_sampler(self, upstream: SamplerT) -> SamplerT:
        raise NotImplementedError


class WorkloadModelRunner(Runner[Sampler | None], UpstreamModelRunner):
    runner_kind = RunnerKind.MRV2

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 topology: AttachmentTopology = AttachmentTopology.IN_PROCESS) -> None:
        super().__init__(vllm_config, device)
        self._initialize_workload(topology)

    def _make_sampler(self, upstream: Sampler | None) -> Sampler:
        if upstream is None:
            raise RuntimeError("MRv2 sampler has not been initialized")
        return Vllm027Sampler(upstream, eager=True)

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_desc: BatchExecutionDescriptor,
    ) -> InputBatch:
        input_batch = super().prepare_inputs(scheduler_output, batch_desc)
        session = self._workload_session
        if session is not None:
            session.prepare_input(encode_mrv2_rows(input_batch,
                tuple(session.identity(request_id) for request_id in input_batch.req_ids),
                plan_epoch=session.attachment.physical.plan_epoch, step_id=session.next_step_id))
        return input_batch

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        session = self._workload_session
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
        session = self._workload_session
        if session is None:
            return super().sample(hidden_states, input_batch, grammar_output)
        match self.model:
            case Qwen2ForCausalLM() as model:
                pass
            case _:
                raise RuntimeError("attached canonical sampler lost its Qwen2 model")
        try:
            active = session.begin_sampler(model)
            sampled = super().sample(hidden_states, input_batch, grammar_output)
            session.complete_sampler(active)
            if self.completion_capture is not None:
                self.completion_capture.write(tuple(session.identity(request_id) for request_id in input_batch.req_ids),
                    hidden_states[input_batch.logits_indices])
        except Exception:
            session.abort_active()
            raise
        return sampled

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> AsyncOutput | ModelRunnerOutput | None:
        session = self._workload_session
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
        if self._workload_session is not None:
            self.detach_workload()
        super().shutdown()

class BackendCommand(StrEnum):
    PROBE = "probe"
    INSTALL = "install"
    STAGE = "stage"
    FINISH = "finish"
    ABORT = "abort"
    DIAGNOSTICS = "diagnostics"
    TRACE = "trace"
    PROFILE = "profile_tasks"
    OBSERVE = "observe"
    DETACH = "detach"
    TUNE_BEGIN = "tune_begin"
    TUNE_CANDIDATE = "tune_candidate"
    TUNE_FINISH = "tune_finish"
    TUNE_ABORT = "tune_abort"


@dataclass(frozen=True, slots=True)
class BackendInstall:
    attachment: SignedBackendAttachment
    definition: str


@dataclass(frozen=True, slots=True)
class BackendStage:
    identities: tuple[RequestIdentity, ...]
    auxiliary: AuxiliaryInput[SharedTensor] | None
    completion: SharedTensor | None


@dataclass(frozen=True, slots=True)
class BackendWorkerState:
    process_id: int
    diagnostics: StateDiagnostics


FeedbackT = TypeVar("FeedbackT", bound=CompletionFeedback)


@dataclass(slots=True)
class VllmLlm(Generic[DataT, ResultT]):
    """Single public handle for signed vLLM workload workloads."""

    _llm: LLM | None
    _runner: WorkloadModelRunner | Mrv1Runner | None
    _deployment: BackendDeployment[DataT, ResultT] | None
    attachment: SignedBackendAttachment
    _facts: BackendCompileFacts
    _closed: bool = False
    _next_prompt_index: int = 0
    _pending_input: ExitStack | None = None
    region_selection: RegionSelection | None = None

    def tune_native_schedules(self) -> None:
        policy = self.attachment.inputs.plan.timeline.region_policy
        if (policy.schedule_policy == "serial"
            or not any(isinstance(task.replay, TokenwiseLayerReplay)
                       for task in self.attachment.inputs.plan.semantic_graph.tasks)):
            return
        if policy.tuning == "heuristic":
            # Static preparation already installed the funded execution,
            # including stream placement; no private generation is needed.
            return
        task = next(task for task in self.attachment.inputs.plan.semantic_graph.tasks
                    if isinstance(task.replay, TokenwiseLayerReplay))
        probe = self.attachment.inputs.probe
        if not isinstance(task.replay, TokenwiseLayerReplay) or probe.causal_domain is None:
            raise ValueError("native tuning requires its signed causal model domain")
        tensor = TensorType((1, probe.hidden_size), "torch." + probe.tensor_dtype.value, str(self._facts.device))
        budget = next(fact.workspace_bytes for fact in self.attachment.inputs.plan.timeline.task_resources
                      if fact.task == task.symbol)
        proposals = compile_tokenwise_region(tensor, probe.max_packed_rows, probe.causal_domain, task.replay,
            self.attachment.inputs.plan.timeline.region_policy,
            stream_lanes=2 if REGION_STREAMS_CAPABILITY in probe.capabilities else 1,
            source_imports=probe.region_imports, function_bodies=probe.region_bodies).within_payload_budget(budget).candidate_proposals
        count = decode_boundary(self.command(BackendCommand.TUNE_BEGIN, encode_boundary(proposals), timeout=600.0), int)
        if not count:
            return
        prompt_index = self._next_prompt_index
        completed = False
        try:
            probe = self.attachment.inputs.probe
            if probe.causal_domain is None or probe.causal_domain.positions < 2:
                raise ValueError("native prefill tuning requires room for a prompt and one output")
            length = min(8, probe.max_packed_rows, probe.causal_domain.positions - 1)
            prompt = " ".join("a" for _ in range(length))
            sampling = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
            baseline: tuple[tuple[int, ...], ...] | None = None
            samples: list[list[int]] = [[] for _ in range(count)]
            # Warm every candidate before scoring any of them. Rotate the
            # measurement order so the reference is not always timed first.
            # Candidate binding and RPC stay outside completed generate time.
            for trial in range(3):
                for offset in range(count):
                    index = (offset + trial) % count
                    self.command(BackendCommand.TUNE_CANDIDATE, encode_boundary(index), timeout=600.0)
                    started = perf_counter_ns()
                    outputs = self.generate([prompt], sampling)
                    finished = perf_counter_ns() - started
                    tokens = tuple(tuple(output.token_ids) for request in outputs for output in request.outputs)
                    if baseline is None:
                        baseline = tokens
                    elif tokens != baseline:
                        raise RuntimeError("native schedule changed the private sample's greedy output")
                    if trial:
                        samples[index].append(finished)
            self.region_selection = decode_boundary(self.command(BackendCommand.TUNE_FINISH,
                encode_boundary(tuple(tuple(elapsed) for elapsed in samples)), timeout=600.0), RegionSelection)
            completed = True
        finally:
            if not completed and not self._closed:
                self.command(BackendCommand.TUNE_ABORT)
            self._next_prompt_index = prompt_index

    def _require_llm(self) -> LLM:
        if self._closed or self._llm is None:
            raise RuntimeError("model handle is closed")
        return self._llm

    @property
    def hidden_size(self) -> int:
        return self._facts.probe.hidden_size

    @property
    def vocab_size(self) -> int:
        return self._facts.vocab_size

    @property
    def device(self) -> torch.device:
        return self._facts.device

    def __enter__(self) -> VllmLlm[DataT, ResultT]:
        if self._closed:
            raise RuntimeError("model handle is closed")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"model cleanup also failed: {cleanup}")

    def diagnostics(self) -> StateDiagnostics:
        if self._closed:
            raise RuntimeError("model handle is closed")
        return self.worker_state().diagnostics

    @property
    def runner(self) -> WorkloadModelRunner | Mrv1Runner:
        if self._closed:
            raise RuntimeError("model handle is closed")
        if self._runner is None:
            raise RuntimeError("the model runner belongs to the EngineCore child")
        return self._runner

    @property
    def deployment(self) -> BackendDeployment[DataT, ResultT]:
        if self._closed:
            raise RuntimeError("model handle is closed")
        if self._deployment is None:
            raise RuntimeError("the execution and arena belong to the EngineCore child")
        return self._deployment

    def command(self, command: BackendCommand, payload: str = "", *, timeout: float = 30.0) -> str:
        replies = self._require_llm().collective_rpc("workload_command", timeout=timeout, args=(command.value, payload))
        if len(replies) != 1:
            self._shutdown()
            raise RuntimeError("canonical worker quorum differs from the signed single-worker topology")
        return replies[0]

    def worker_state(self) -> BackendWorkerState:
        return decode_boundary(self.command(BackendCommand.DIAGNOSTICS), BackendWorkerState)

    def trace_snapshot(self) -> TraceSnapshot:
        if self._closed:
            raise RuntimeError("vLLM workload handle is closed")
        snapshot = (self._runner.trace.snapshot() if self._runner is not None else
            decode_boundary(self.command(BackendCommand.TRACE), TraceSnapshot))
        if snapshot.schema_version != TRACE_SCHEMA_VERSION:
            raise RuntimeError("worker trace differs from the original schema")
        return snapshot

    def profile_tasks(self, tasks: tuple[TaskId, ...]) -> None:
        if self._closed:
            raise RuntimeError("vLLM workload handle is closed")
        if self._runner is not None:
            self._runner.trace.profile_tasks(tasks)
        else:
            self.command(BackendCommand.PROFILE, encode_boundary(tasks))

    def observe(self, selection: TraceSelection) -> None:
        """Configure bounded explicit tensor/gradient snapshots and opt-in Kineto profiling."""
        if self._closed:
            raise RuntimeError("vLLM workload handle is closed")
        if self._runner is not None:
            self._runner.trace.configure(selection)
        else:
            self.command(BackendCommand.OBSERVE, encode_boundary(selection))

    def close(self) -> None:
        if self._llm is None and self._pending_input is None:
            return
        try:
            if not self._closed:
                self.command(BackendCommand.DETACH)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        # Closed to new work immediately; retain ownership if shutdown needs a retry.
        self._closed = True
        if self._llm is not None:
            self._llm.llm_engine.engine_core.shutdown()
        self._llm = self._runner = self._deployment = None
        if self._pending_input is not None:
            self._pending_input.close()
            self._pending_input = None

    def generate_feedback(self, prompts: list[str], sampling_params: SamplingParams, *,
                          feedback: Callable[[CompletionInput], FeedbackT | None], feedback_group: str,
                          max_rounds: int, prompt_indices: tuple[int, ...] | None = None,
                          use_tqdm: bool = False,
                          completion_device: torch.device | None = None,
                          edit: Callable[[FeedbackT], StateTensorUpdate[torch.Tensor] | None] | None = None,
                          routes: tuple[str, ...] | None = None,
                          ) -> Iterator[tuple[CompletionInput, FeedbackT]]:
        """Generate, settle feedback, then retry eligible original prompts within a finite bound.

        The callable runs locally after generate has released backend requests; it is
        never installed on a worker. Each yield is the original completion and result,
        with no retained history. Multi-request barriers are regenerated as a full
        vote group. Closing the iterator leaves no pending generation or transaction.
        Optional edits reuse StateTensorUpdate and require a signed activation workload;
        the feedback adapter returns original tensors, never a worker callback.
        """
        if type(max_rounds) is not int or max_rounds < 1 or type(prompts) is not list or not prompts:
            raise ValueError("feedback generation requires a nonempty prompt list and positive finite round bound")
        texts: list[str] = list(prompts)
        indices: tuple[int, ...] = (tuple(range(self._next_prompt_index, self._next_prompt_index + len(texts)))
            if prompt_indices is None else prompt_indices)
        route_values: list[str] | None = list(routes) if routes is not None else None
        auxiliary: AuxiliaryInput[torch.Tensor] | None = None
        to_worker: Callable[[torch.Tensor], torch.Tensor] = lambda tensor: tensor.to(self.runner.device)
        for attempt in range(max_rounds):
            batch: CompletionInput = self.generate(texts, sampling_params, use_tqdm=use_tqdm, feedback_group=feedback_group,
                prompt_indices=indices, sample_index=attempt, auxiliary=auxiliary, completion_device=completion_device,
                routes=None if route_values is None else tuple(route_values))
            result = feedback(batch)
            if result is None:
                raise RuntimeError("completion feedback returned no settled control result")
            entries, controls = batch.completions.entries, result.retry_controls
            if len(controls) != len(entries) or any(control.completion is not entry
                for control, entry in zip(controls, entries, strict=True)):
                raise ValueError("feedback controls must reference the original ordered completions")
            by_request = {entry.identity.request_id: entry for entry in entries}
            covered: set[str] = set()
            ready: set[str] = set()
            for barrier in result.update_barriers:
                members = set(barrier.request_ids)
                if (not members or len(members) != len(barrier.request_ids) or covered.intersection(members)
                    or not members.issubset(by_request) or not 0 <= barrier.ready_count <= len(members)
                    or any(by_request[request].group_id != barrier.group_id for request in members)):
                    raise ValueError("feedback barriers must partition original completion identities")
                covered.update(members)
                if barrier.state is BarrierState.READY and barrier.ready_count > 0 and result.update_count > 0:
                    ready.update(members)
            if covered != by_request.keys():
                raise ValueError("feedback barriers omitted a completion")
            retry = {control.completion.identity.request_id for control in controls
                if control.action is RetryAction.RETRY and (not control.retry_after_update
                    or control.completion.identity.request_id in ready)}
            stopped = {control.completion.identity.request_id for control in controls if control.action is RetryAction.STOP}
            for barrier in result.update_barriers:
                members = set(barrier.request_ids)
                if members.intersection(retry):
                    if members.intersection(stopped):
                        retry.difference_update(members)
                    else:
                        retry.update(members)
            selected: tuple[int, ...] = tuple(index for index, entry in enumerate(entries) if entry.identity.request_id in retry)
            auxiliary = edit(result) if edit is not None and selected and attempt + 1 < max_rounds else None
            if auxiliary is not None and self._runner is not None:
                validate_auxiliary(auxiliary, self.attachment.inputs.plan, SharedTensor.describe)
                auxiliary = map_auxiliary(auxiliary, to_worker)
            yield batch, result
            if not selected:
                return
            texts = [texts[index] for index in selected]
            indices = tuple(indices[index] for index in selected)
            if route_values is not None:
                route_values = [route_values[index] for index in selected]

    @overload
    def generate(self, prompts: str | list[str], sampling_params: SamplingParams, *,
                 use_tqdm: bool = False, auxiliary: AuxiliaryInput[torch.Tensor] | None = None,
                 sample_index: int = 0, feedback_group: None = None,
                 prompt_indices: tuple[int, ...] | None = None, completion_device: torch.device | None = None,
                 routes: tuple[str, ...] | None = None) -> list[RequestOutput]: ...

    @overload
    def generate(self, prompts: str | list[str], sampling_params: SamplingParams, *,
                 use_tqdm: bool = False, auxiliary: AuxiliaryInput[torch.Tensor] | None = None,
                 sample_index: int = 0, feedback_group: str,
                 prompt_indices: tuple[int, ...] | None = None, completion_device: torch.device | None = None,
                 routes: tuple[str, ...] | None = None) -> CompletionInput: ...

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams,
        *,
        use_tqdm: bool = False,
        auxiliary: AuxiliaryInput[torch.Tensor] | None = None,
        sample_index: int = 0,
        feedback_group: str | None = None,
        prompt_indices: tuple[int, ...] | None = None,
        completion_device: torch.device | None = None,
        routes: tuple[str, ...] | None = None,
    ) -> list[RequestOutput] | CompletionInput:
        if self._closed:
            raise RuntimeError("vLLM workload handle is closed")
        if completion_device is not None and feedback_group is None:
            raise ValueError("completion device requires completion feedback")
        output_device = completion_device or (torch.device("cpu") if self._runner is None else self.runner.device)
        if output_device.type == "cuda":
            if output_device == torch.device("cuda"):
                output_device = torch.device("cuda", self.attachment.inputs.probe.device_ordinal)
            if output_device.index != self.attachment.inputs.probe.device_ordinal:
                raise ValueError("completion IPC currently requires the model's CUDA device")
        elif output_device.type != "cpu":
            raise ValueError("completion output requires CPU or CUDA storage")
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
            or type(sample_index) is not int or sample_index < 0
            or feedback_group is not None and (type(feedback_group) is not str or not feedback_group)
        ):
            raise ValueError("vLLM workload generate requires non-empty text and n=1")
        texts = [prompts] if isinstance(prompts, str) else prompts
        indices = (tuple(range(self._next_prompt_index, self._next_prompt_index + len(texts)))
            if prompt_indices is None else prompt_indices)
        if len(indices) != len(texts) or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("prompt indices must align with the original input prompts")
        # A route value is user-computed (regex, discriminator, tenant); the signed
        # graph only declares whether any request state keys on it.
        signed_route = any(RequestKeyField.ROUTE in state.key_fields
            for state in self.attachment.inputs.plan.semantic_graph.states)
        if routes is not None and (
            not signed_route or len(routes) != len(texts)
            or any(type(route) is not str or not route for route in routes)):
            raise ValueError("routes require a signed route-keyed state and one value per prompt")
        if signed_route and routes is None:
            raise ValueError("a route-keyed request state requires per-prompt routes")
        groups: dict[int, str] = {}
        for index, prompt in zip(indices, texts, strict=True):
            if groups.setdefault(index, prompt) != prompt:
                raise ValueError("one prompt identity cannot name different input texts")
        validate_auxiliary(auxiliary, self.attachment.inputs.plan, SharedTensor.describe)
        llm = self._require_llm()
        llm.sleep(level=0, mode="keep")
        request_ids: list[str] = []
        lifetime = ExitStack()
        try:
            request_ids = llm.enqueue(prompts, sampling_params, use_tqdm=use_tqdm)
            identities = tuple(RequestIdentity(request_id, index, sample_index, route)
                for request_id, index, route in zip(request_ids, indices,
                    routes if routes is not None else ("",) * len(texts), strict=True))
            # Borrow vLLM's exact frontend mapping before completed entries are removed.
            external = {llm.llm_engine.output_processor.request_states[identity.request_id].external_req_id: identity
                for identity in identities}
            destination = None
            completion_wire = None
            if feedback_group is not None:
                field = TensorFieldSpec(LayoutFieldId(0), (len(identities), self.attachment.inputs.probe.hidden_size),
                    TensorDType.FLOAT32, 4)
                with torch.inference_mode(False):
                    if self._runner is None:
                        destination, completion_wire = SharedTensor.allocate(field, lifetime, output_device)
                    else:
                        destination = torch.empty(field.shape, device=output_device, dtype=torch.float32)
            if self._runner is None:
                auxiliary_wire = map_auxiliary(auxiliary, lambda tensor: SharedTensor.export(tensor, lifetime))
                self.command(BackendCommand.STAGE, encode_boundary(BackendStage(identities, auxiliary_wire, completion_wire)))
            else:
                self.runner.stage_workload_input(auxiliary, identities, destination)
            self._next_prompt_index = max(self._next_prompt_index, max(indices) + 1)
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(RequestOutput, use_tqdm=use_tqdm)
            result: list[RequestOutput] | CompletionInput = outputs
            batch = None
            if feedback_group is not None:
                completions: dict[RequestIdentity, Completion] = {}
                for output in outputs:
                    if not output.finished or len(output.outputs) != 1:
                        raise RuntimeError("completion feedback requires final n=1 outputs")
                    identity = external[output.request_id]
                    completion, = output.outputs
                    if identity in completions:
                        raise RuntimeError("duplicate completion for a submitted request")
                    completions[identity] = Completion(identity, completion.index, feedback_group,
                        tuple(completion.token_ids), completion.text, FinishReason(completion.finish_reason))
                if len(completions) != len(identities):
                    raise RuntimeError("completion feedback did not finish every submitted request")
                batch = CompletionBatch(tuple(completions[identity] for identity in identities))
                if self._runner is not None:
                    capture = self.runner.completion_capture
                    if capture is None:
                        raise RuntimeError("completion capture was not started")
                    result = capture.finish(batch)
            self.command(BackendCommand.FINISH)
            if batch is not None and self._runner is None:
                if destination is None:
                    raise RuntimeError("completion output was not allocated")
                result = CompletionInput(batch, destination, CudaStreamReadiness(torch.cuda.current_stream(destination.device))
                    if destination.device.type == "cuda" else None)
            return result
        except Exception as original:
            try:
                llm.llm_engine.abort_request(request_ids, internal=True)
                self.command(BackendCommand.ABORT)
                llm.wake_up(tags=["scheduling"])
            except Exception as cleanup:
                # Borrowed input storage cannot outlive an unacknowledged worker.
                original.add_note(f"worker abort also failed: {cleanup}")
                try:
                    self._shutdown()
                except Exception as shutdown:
                    original.add_note(f"worker shutdown also failed; close() can retry: {shutdown}")
            raise
        finally:
            if self._closed and self._llm is not None:
                self._pending_input = lifetime
            else:
                lifetime.close()


def create_vllm_llm(
    workload: BackendWorkload[DataT, ResultT],
    config: ModelConfig,
    *, definition: str | None = None,
) -> VllmLlm[DataT, ResultT]:
    if config.enable_prefix_caching and (workload.backend_requirements().parameter_targets
        or workload.backend_requirements().activation_layers):
        raise ValueError("parameter-version overlays do not support shared prefix caching")
    if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError("vLLM workload requires vLLM 0.27.1 exactly")
    environment = {_MP_ENV: "1" if config.multiprocessing else "0",
        _V2_RUNNER_ENV: "1" if config.runner is RunnerKind.MRV2 else "0"}
    if config.multiprocessing:
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    previous_environment = {name: os.environ.get(name) for name in environment}
    if config.multiprocessing and definition is None:
        raise ValueError("EngineCore requires a closed mathematical definition from the public factory")
    if any(previous_environment[name] not in (None, value) for name, value in environment.items()):
        raise RuntimeError("vLLM workload conflicts with explicit MP, spawn or legacy-runner environment")
    os.environ.update(environment)
    try:
        llm = LLM(
            model=config.model,
            revision=config.revision,
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
            async_scheduling=False if config.runner is RunnerKind.MRV1 else None,
        )
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name)
            else:
                os.environ[name] = value

    with ExitStack() as construction:
        construction.callback(llm.llm_engine.engine_core.shutdown)
        if config.multiprocessing:
            replies = llm.collective_rpc("workload_command", timeout=30.0,
                args=(BackendCommand.PROBE.value, encode_boundary(workload.backend_requirements())))
            if len(replies) != 1:
                raise RuntimeError("canonical probe requires the exact single-worker quorum")
            facts = workload.specialize(decode_boundary(replies[0], BackendCompileFacts), config.execution)
            attachment = workload.plan_backend(facts)
            handle = VllmLlm[DataT, ResultT](llm, None, None, attachment, facts)
            installed = decode_boundary(handle.command(BackendCommand.INSTALL,
                encode_boundary(BackendInstall(attachment, definition or ""))), BackendWorkerState)
            if installed.process_id == os.getpid():
                raise RuntimeError("EngineCore did not install the original signed compilation in its child")
        else:
            match llm.llm_engine.engine_core:
                case InprocClient() as client:
                    pass
                case _:
                    raise RuntimeError("vLLM workload did not construct an in-process client")
            match client.engine_core.model_executor:
                case UniProcExecutor() as executor:
                    pass
                case _:
                    raise RuntimeError("vLLM workload did not construct a uni-process executor")
            match executor.driver_worker.worker:
                case Worker() as worker:
                    pass
                case _:
                    raise RuntimeError("vLLM workload did not construct its worker")
            from pllm.backends.vllm.mrv1 import Mrv1Runner
            match worker.model_runner:
                case WorkloadModelRunner() | Mrv1Runner() as runner:
                    pass
                case _:
                    raise RuntimeError("vLLM workload did not construct its selected runner")
            facts = workload.specialize(probe_backend(workload.backend_requirements(), runner), config.execution)
            deployment = workload.compile_backend(facts)
            runner.attach_workload(deployment)
            handle = VllmLlm(llm, runner, deployment, deployment.attachment, facts)
        handle.tune_native_schedules()
        construction.pop_all()
        return handle


def _input_reservations(probe: BackendPlacementProbe, timeline: SchedulerTimeline, rows: int,
                        vocab: int, stage: SamplerStage | None, feedback: bool) -> tuple[int, ...]:
    hidden = len(probe.hidden_resources) * probe.max_packed_rows * probe.hidden_size * 4
    # FP32 envelope for original logits, conversion, projection and rowwise arithmetic.
    # Excludes upstream sampling/GEMM library scratch, model weights and KV storage.
    workspace = (rows * (36 * vocab + 4 * probe.hidden_size + 8) if stage is SamplerStage.PACKED_POST_FILTER_EXACT
        else rows * (12 * vocab + 8) if stage is SamplerStage.PACKED_PRE_FILTER_ADD else 0)
    retained = rows * (4 * vocab + 8) if feedback else 0
    sampler = next(anchor.ordinal for anchor in timeline.anchors if anchor.symbol == "sampler.pre_draw")
    return tuple(hidden + (workspace if anchor.symbol in ("sampler.pre_draw", "sampler.complete") else 0)
        + (retained if anchor.ordinal >= sampler else 0) for anchor in timeline.anchors)


def probe_backend(
    requirements: BackendRequirements,
    runner: WorkloadModelRunner | Mrv1Runner,
    timeline: SchedulerTimeline | None = None,
) -> BackendCompileFacts:
    _require_runner(runner)
    match runner.model:
        case Qwen2ForCausalLM() as model:
            pass
        case _:
            raise RuntimeError("vLLM workload currently requires Qwen2")
    if model.quant_config is not None:
        raise RuntimeError("vLLM workload requires an unquantized model")
    if requirements.model_source is not None and requirements.model_source != (
        runner.model_config.model, runner.model_config.revision or "main"):
        raise ValueError("causal training and inference must use the same model source and revision")
    layer_count = len(model.model.layers)
    if any(layer < 0 or layer >= layer_count for layer in requirements.activation_layers):
        raise ValueError("activation layer is outside the backend model")
    hidden_resources = tuple(sorted((AttachmentHiddenResource(_hidden_binding(source, layer_count), source.phase)
        for source in requirements.hidden_inputs), key=lambda resource: resource.resource_key))
    targets = inspect_parameter_targets(model, requirements.parameter_targets, runner.dtype)
    replay_capabilities = ("model/layer_replay@1", "model/tokenwise_replay@1", REGION_STREAMS_CAPABILITY) if runner.runner_kind is RunnerKind.MRV2 else ()
    probe = _base_probe(runner,
        model_fingerprint=_digest((
            f"model={runner.model_config.model}", f"revision={runner.model_config.revision or ''}",
            "class=vllm.model_executor.models.qwen2.Qwen2ForCausalLM",
            f"hidden_size={model.config.hidden_size}", f"layers={layer_count}",
            f"vocab_size={model.config.vocab_size}",
            *(f"hidden={resource.binding.module_path}" for resource in hidden_resources),
            *(f"activation_layer={layer}" for layer in requirements.activation_layers),
            *(f"target={target.target}:{target.input_size}:{target.output_size}" for target in targets),
        )),
        hidden_size=model.config.hidden_size, layer_count=layer_count, hidden_resources=hidden_resources,
        capabilities=tuple(sorted(set((*_REQUEST_CAPABILITIES, *_OVERLAY_CAPABILITIES, *replay_capabilities,
            f"backend/vllm_{runner.runner_kind.value}@1")))))
    if runner.runner_kind is RunnerKind.MRV2:
        probe = replace(probe, region_imports=qwen_region_kernels(model.model.norm,
            TensorType((1, probe.hidden_size), str(runner.dtype), str(runner.device)), layer_count))
        if (probe.causal_domain is not None and not requirements.hidden_inputs
            and not requirements.parameter_targets and not requirements.activation_layers):
            bodies = VllmDecoderBodies.inspect(tuple(model.model.layers),
                TensorType((1, probe.hidden_size), str(runner.dtype), str(runner.device)), probe.causal_domain, probe.max_packed_rows)
            if bodies is not None:
                probe = replace(probe, region_bodies=bodies.definitions, region_imports=(*probe.region_imports, *bodies.imports))
    observed = RuntimeFacts(runner.device).timeline((
        "state/dense_page_cuda@1", "state/fenced_reclaim@1", "transition/full_root@1", "optimizer/adamw@1",
        *replay_capabilities,
        *(("compute/cuda_graph@1",) if timeline is not None and "compute/cuda_graph@1" in timeline.capabilities else ())))
    required = _input_reservations(probe, observed, runner.max_num_reqs, model.lm_head.weight.shape[0],
        requirements.sampler_stage, requirements.sampler_feedback)
    required = tuple(value + requirements.resident_bytes for value in required)
    observed = replace(observed, reserved_bytes=required)
    if timeline is None:
        timeline = observed
    elif (timeline.anchors != observed.anchors or timeline.capabilities != observed.capabilities
          or timeline.schema_version != observed.schema_version):
        raise ValueError("worker timeline differs from the supported single-device anchors and capabilities")
    elif len(timeline.reserved_bytes) != len(required) or any(
            actual < minimum for actual, minimum in zip(timeline.reserved_bytes, required, strict=True)):
        raise ValueError("worker timeline underfunds the backend input/sampler buffers")
    free_bytes, _total_bytes = torch.cuda.mem_get_info(runner.device)
    # One current and one private root per scheduled request, with publication overlap.
    return BackendCompileFacts(probe, timeline, (PageBundleProviderFacts("canonical.page_bundle_cuda@1", 2 * runner.max_num_reqs + 2,
        probe.device_kind, probe.device_ordinal),), runner.max_num_reqs, free_bytes,
        runner.model_config.seed, targets, model.config.vocab_size, layer_count)


def _require_runner(runner: WorkloadModelRunner | Mrv1Runner) -> None:
    if (
        _VLLM_VERSION != _SUPPORTED_VLLM_VERSION
        or not runner.model_config.enforce_eager
        or runner.parallel_config.tensor_parallel_size != 1
        or runner.parallel_config.pipeline_parallel_size != 1
        or runner.parallel_config.data_parallel_size != 1
        or runner.parallel_config.decode_context_parallel_size != 1
        or runner.speculative_config is not None
        or runner.lora_config is not None
        or runner.device.type != "cuda"
        or runner.max_num_reqs < 1
        or type(runner.sampler) is not (Sampler if runner.runner_kind is RunnerKind.MRV2 else Mrv1Sampler)
    ):
        raise RuntimeError(
            "vLLM workload requires eager TP1/PP1/DP1 non-spec execution"
        )


def _base_probe(
    runner: WorkloadModelRunner | Mrv1Runner,
    *,
    model_fingerprint: str,
    hidden_size: int,
    layer_count: int,
    hidden_resources: tuple[AttachmentHiddenResource, ...],
    capabilities: tuple[str, ...],
) -> BackendPlacementProbe:
    device_ordinal = runner.device.index
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise RuntimeError("vLLM workload requires an ordinal CUDA device")
    return BackendPlacementProbe(
        adapter_id=f"canonical.vllm_{runner.runner_kind.value}@1",
        backend_abi_digest=_digest((*_BACKEND_ABI_FIELDS, f"runner={runner.runner_kind.value}")),
        runner_fingerprint=_digest(
            (
                runner.runner_kind.value,
                f"device=cuda:{device_ordinal}",
                f"max_num_tokens={runner.max_num_tokens}",
                f"max_num_reqs={runner.max_num_reqs}",
                f"dtype={runner.dtype}",
                f"seed={runner.model_config.seed}",
            )
        ),
        model_fingerprint=model_fingerprint,
        topology=runner.workload_topology,
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
        causal_domain=CausalModelDomain(runner.max_num_reqs, layer_count, runner.model_config.max_model_len),
    )




def _hidden_binding(source: HiddenInputSource, layer_count: int) -> PackedHiddenResourceBinding:
    layer = source.layer
    if source.site is HiddenSite.MODEL_OUTPUT and layer == -1:
        return PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX, source.site, "model.norm")
    if source.site is not HiddenSite.BLOCK_OUTPUT:
        raise ValueError("Qwen2 hidden input requires a block or final model output")
    module_index = layer if layer >= 0 else layer_count + layer
    if module_index < 0 or module_index >= layer_count:
        raise ValueError("hidden layer does not identify a Qwen2 block")
    return PackedHiddenResourceBinding(
        str(layer),
        HiddenLayerKind.INDEX,
        HiddenSite.BLOCK_OUTPUT,
        f"model.layers.{module_index}",
    )




__all__ = [
    "WorkloadModelRunner",
    "VllmConfig",
    "VllmDType",
    "VllmLlm",
    "VllmSession",
    "create_vllm_llm",
]

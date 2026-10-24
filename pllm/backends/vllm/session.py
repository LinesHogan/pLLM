"""One vLLM session: a signed tensor-state invocation driven step by step inside the model runner.

VllmSession owns the per-step lifecycle (row preparation, model effects, sampler guidance, completion and
publication) for one attached workload. Used by the runners in runner.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, cast

import torch
from vllm import SamplingParams
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.v1.core.sched.output import SchedulerOutput

from pllm.authoring import (
    AuxiliaryInput,
    BackendDeployment,
    DataT,
    ResultT,
    TensorStepInput,
)
from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.hidden_capture import HiddenTaps
from pllm.backends.vllm.model_scan import VllmModelScan
from pllm.backends.vllm.overlay_graph import (
    HybridParameterOverlay,
    PinnedParameterOverlay,
    parameter_overlay,
)
from pllm.backends.vllm.parameter_overlay import (
    validate_projection_sources,
)
from pllm.backends.vllm.replay import VllmLayerReplay, VllmTokenwiseReplay
from pllm.backends.vllm.sampler import (
    SamplerGuidance,
    SignedSamplerGuidance,
)
from pllm.backends.vllm.window_scheduler import WindowSchedulerOutput, model_scan_admission
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentKind,
    BackendPlacementProbe,
    DecodeWindow,
    SignedBackendAttachment,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.region_proposal import REGION_STREAMS_CAPABILITY, RegionScheduleProposal
from pllm.contracts.region_selection import RegionSelection
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorId,
    ConditionalLayerReplay,
    SamplerFeedbackInputSource,
    SchedulerTimeline,
    StateScope,
    TokenwiseLayerReplay,
)
from pllm.execution.backend_attachment import attachment_buffer_bytes
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.overlay_task import ProjectionTask
from pllm.execution.post_sample import PostSampleOverlap
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.state_execution import StateDiagnostics, StateTaskContext
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tokenwise_tuning import TokenwiseTuning

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class _PackedStep(Generic[DataT]):
    rows: PackedSamplerRowBatch
    invocation: TaskGraphInvocation[StateTaskContext[DataT]] | None = None
    forward_done: bool = False


@dataclass(frozen=True, slots=True)
class ReadonlyStepInput:
    """Internal handoff of complete received grants for one original engine step."""

    plan_epoch: int
    step_id: int
    roots: tuple[BorrowedStateRoot, ...]


@dataclass(frozen=True, slots=True)
class ReadonlyStepAbsent:
    plan_epoch: int
    step_id: int


@dataclass(frozen=True, slots=True)
class ReadonlyStepEnd:
    plan_epoch: int
    step_id: int


ReadonlyStepReceiver = Callable[
    [PackedSamplerRowBatch, torch.cuda.Stream], ReadonlyStepInput | ReadonlyStepAbsent | ReadonlyStepEnd
]


class VllmSession(Generic[DataT, ResultT]):
    """Model primitives over one original signed tensor-state invocation."""

    def __init__(
        self,
        deployment: BackendDeployment[DataT, ResultT],
        model: Qwen2ForCausalLM,
        sampler: SamplerGuidance | None,
        max_sampler_rows: int,
        taps: HiddenTaps | None = None,
    ) -> None:
        attachment = deployment.attachment
        plan = attachment.inputs.plan
        self._capture_feedback = any(
            isinstance(port.source, SamplerFeedbackInputSource)
            for task in plan.semantic_graph.tasks
            for port in task.inputs
        )
        visibility = plan.timeline.visibility
        required = input_reservations(
            attachment.inputs.probe,
            plan.timeline,
            max_sampler_rows,
            model.lm_head.weight.shape[0],
            attachment.inputs.request.sampler_stage if sampler is not None else None,
            self._capture_feedback,
            0 if visibility is None else visibility.lag,
        )
        buffers = attachment_buffer_bytes(
            attachment.inputs.probe, plan.semantic_graph, plan.tensor_layout_request, attachment.inputs.request
        )
        required = tuple(value + buffers for value in required)
        if len(plan.timeline.reserved_bytes) != len(required) or any(
            actual < minimum for actual, minimum in zip(plan.timeline.reserved_bytes, required, strict=True)
        ):
            raise ValueError("signed timeline underfunds the backend input/sampler/effect buffers")
        self._deployment = deployment
        self.trace = deployment.execution.trace
        probe = attachment.inputs.probe
        self._hidden_resources = tuple(resource.binding for resource in probe.hidden_resources)
        if self._hidden_resources and taps is None:
            raise ValueError("hidden inputs require taps installed when the model was loaded")
        self._taps = taps
        overlay_bindings = attachment.inputs.request.overlay_bindings
        model_scan = attachment.inputs.request.attachment_kind is AttachmentKind.MODEL_SCAN
        self._overlay_import = bool(overlay_bindings) and (
            plan.semantic_graph.tasks[attachment.inputs.request.task_bindings[0].task_id].timing.after
            == "request.prepare"
        )
        self._in_forward_overlay = any(
            task.visibility is not None and task.visibility.in_forward for task in plan.semantic_graph.tasks
        )
        if self._in_forward_overlay or probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH:
            validate_projection_sources(model, tuple(binding.target for binding in overlay_bindings))
        self._overlay = (
            parameter_overlay(
                model,
                overlay_bindings,
                deployment.execution.arena,
                probe.max_packed_rows,
                max_sampler_rows,
                buffers,
                plan.timeline.region_policy,
                phases=deployment.execution.math.parameter_phases,
                in_forward=self._in_forward_overlay,
                graphs=probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH,
            )
            if overlay_bindings and not model_scan
            else None
        )
        self._model_scan = (
            VllmModelScan(
                model,
                deployment.execution,
                attachment.inputs.request.task_bindings[0].anchor_id,
                overlay_bindings,
                model_scan_admission(attachment),
            )
            if model_scan
            else None
        )
        tasks = tuple(task for task in attachment.inputs.plan.semantic_graph.tasks if task.replay is not None)
        self._activation = (
            VllmActivationPatch(
                model,
                attachment.inputs.request.activation_bindings,
                attachment.inputs.request.state_resource,
                install=not any(isinstance(task.replay, TokenwiseLayerReplay) for task in tasks),
            )
            if (attachment.inputs.request.activation_bindings)
            else None
        )
        self._decode_injection = any(
            binding.decode is DecodeWindow.ALL for binding in attachment.inputs.request.activation_bindings
        )
        self._token_frontiers: dict[RequestIdentity, int] = {}
        self._readonly_receiver: ReadonlyStepReceiver | None = None
        self._received_readonly: ReadonlyStepInput | ReadonlyStepAbsent | ReadonlyStepEnd | None = None
        self._input_failed = False
        self._replay: VllmLayerReplay | VllmTokenwiseReplay[DataT, ResultT] | None = None
        if tasks:
            (task,) = tasks
            if probe.causal_domain is None or probe.causal_domain.layers != len(model.model.layers):
                raise ValueError("backend replay requires its signed causal model domain")
            if isinstance(task.replay, ConditionalLayerReplay) and self._overlay is not None:
                assert isinstance(self._overlay, PinnedParameterOverlay)
                self._replay = VllmLayerReplay(model, task.replay, self._overlay, probe.causal_domain)
            elif isinstance(task.replay, TokenwiseLayerReplay):
                self._replay = VllmTokenwiseReplay(
                    model,
                    task,
                    deployment.execution,
                    attachment.inputs.request.task_bindings[0].anchor_id,
                    source_imports=probe.region_imports,
                    function_bodies=probe.region_bodies,
                    activation=self._activation,
                )
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
        self._post_sample = None
        if attachment.physical.post_sampler != "main":
            assert visibility is not None
            self._post_sample = PostSampleOverlap(
                execution.arena,
                visibility,
                execution.device,
                readers=attachment.physical.post_sampler == "lagged",
            )
        if execution.directory.scope is StateScope.INSTANCE:
            execution.admit((), torch.cuda.current_stream(execution.device))

    @property
    def parameter_phases(self) -> tuple[HiddenPhase, ...]:
        """Authored update phases used by hybrid model dispatch."""
        return self._deployment.execution.math.parameter_phases

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
        if (
            self._readonly_receiver is not None
            or self._step is not None
            or self._next_step_id
            or self._identities
            or self._pending_identities
            or self._auxiliary is not None
            or self._tuning is not None
            or self._activation is None
            or all(
                state.resource in self._deployment.execution.arena.owned_resources
                for state in self.attachment.inputs.plan.semantic_graph.states
            )
        ):
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
            if isinstance(received, (ReadonlyStepAbsent, ReadonlyStepEnd)):
                self._deployment.execution.stage_hold(
                    tuple(rows.identities),
                    "end" if isinstance(received, ReadonlyStepEnd) else "absent",
                    step_id=rows.step_id,
                )
            else:
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
        if (
            not isinstance(self._replay, VllmTokenwiseReplay)
            or self.attachment.inputs.plan.semantic_graph.states
            or self.attachment.inputs.plan.timeline.region_policy.schedule_policy == "serial"
        ):
            return 0
        identity = self.attachment.inputs.request.task_bindings[0].implementation_id
        tuning = TokenwiseTuning(
            self._deployment.execution,
            identity,
            proposals,
            stream_lanes=2 if REGION_STREAMS_CAPABILITY in self.attachment.inputs.probe.capabilities else 1,
        )
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
        if (self._decode_injection or self._model_scan is not None) and (
            scheduler_output.preempted_req_ids or scheduler_output.scheduled_cached_reqs.resumed_req_ids
        ):
            self._input_failed = True
            raise RuntimeError("activation injection and ModelScan need requests without vLLM preemption")
        if self._model_scan is not None:
            if not isinstance(scheduler_output, WindowSchedulerOutput):
                raise ValueError("ModelScan needs its closed-window scheduler and original allocation grants")
            self._model_scan.accept(scheduler_output.window)
        execution = self._deployment.execution
        execution.arena.collect_completed()
        finished = tuple(
            self._identities[request_id]
            for request_id in sorted(self._identities.keys() & scheduler_output.finished_req_ids)
        )
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
                raise RuntimeError(
                    "per-decode activation injection and vLLM preemption/history recomputation cannot be enabled together"
                )
            for row in rows.row_batch.rows:
                self._token_frontiers[row.identity] = max(
                    self._token_frontiers.get(row.identity, 0), row.token_offset + 1
                )
        requests = rows.identities
        self._next_step_id += 1
        step = _PackedStep[DataT](rows)
        self._step = step
        if self._model_scan is not None:
            execution = self._deployment.execution
            invocation = execution.start_step(
                TensorStepInput(rows.row_batch, rows), torch.cuda.current_stream(execution.device)
            )
            step.invocation = invocation
            self._advance(invocation, "request.prepare")
            self._model_scan.prepare(rows.row_batch, invocation)
            return
        if isinstance(self._replay, VllmTokenwiseReplay) and self._activation is None:
            execution = self._deployment.execution
            invocation = execution.start_step(
                TensorStepInput(rows.row_batch, rows), torch.cuda.current_stream(execution.device)
            )
            step.invocation = invocation
            self._advance(invocation, "request.prepare")
            self._replay.prepare(rows.row_batch, invocation)
            invocation.context.replay = self._replay
            return
        adapter = self._overlay
        if adapter is None and self._activation is None:
            return
        if isinstance(adapter, (PinnedParameterOverlay, HybridParameterOverlay)):
            adapter.reserve()
        execution = self._deployment.execution
        identities = tuple(
            sorted(requests, key=lambda identity: (identity.request_id, identity.prompt_idx, identity.sample_idx))
        )
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
            pending = tuple(
                identity
                for identity in identities
                if execution.directory.scope is StateScope.INSTANCE
                and not execution.directory.has_inference_pin(identity)
            )
            if pending:
                execution.admit(pending, stream)
        else:
            invocation = execution.begin(rows.row_batch.request_rows, stream)
            step.invocation = invocation
        if adapter is not None:
            projection = None
            if self._in_forward_overlay:
                self._advance(invocation, "request.prepare")
                binding = self.attachment.inputs.request.task_bindings[0]
                projection = ProjectionTask(
                    execution,
                    invocation,
                    rows,
                    binding.task_id,
                    tuple(item.target for item in self.attachment.inputs.request.overlay_bindings),
                )
            adapter.prepare(rows.row_batch, invocation.context, execution.directory, projection)
        if self._activation is not None:
            self._activation.prepare(rows.row_batch, invocation.context, execution.directory)
        if isinstance(self._replay, VllmTokenwiseReplay):
            self._replay.prepare(rows.row_batch, invocation)
            invocation.context.replay = self._replay
        elif isinstance(self._replay, VllmLayerReplay):
            self._replay.prepare(rows.row_batch, invocation.context)
            invocation.context.replay = self._replay

    def complete_forward(self) -> None:
        step = self._step
        if step is None or step.forward_done:
            return
        step.forward_done = True
        execution = self._deployment.execution
        stream = torch.cuda.current_stream(execution.device)
        operands = () if self._overlay is None else self._overlay.take_captures()
        value = TensorStepInput(step.rows.row_batch, step.rows, operands=operands, auxiliary=self._auxiliary)
        if self._taps is not None and self._hidden_resources:
            value = self._taps.publish(value, self._hidden_resources)
        phases = {resource.phase for resource in self._deployment.attachment.inputs.probe.hidden_resources}
        if (
            step.invocation is None
            and not operands
            and not any(row.phase in phases for row in step.rows.row_batch.rows)
        ):
            self._step = None
            return
        if step.invocation is None:
            # With no pre-sampler tasks there is no reader to lag. Pin the latest state
            # directly on the writer lane so the main stream never waits for training.
            if self._post_sample is not None and not self._post_sample.readers:
                stream = self._post_sample.stream
            step.invocation = execution.start_step(
                value, stream, history=() if self._post_sample is None else self._post_sample.history
            )
            self._auxiliary = None
        elif step.invocation.context.value is None:
            execution.submit_input(step.invocation, value, CudaStreamReadiness(stream))
        self._advance(step.invocation, "sampler.pre_draw")
        self._finish_effects()

    def _finish_effects(self) -> None:
        """End-of-forward lifecycle for every active optional primitive."""
        for effect in (self._overlay, self._activation, self._replay, self._model_scan):
            if effect is not None:
                effect.clear()

    def _abort_effects(self) -> None:
        for effect in (self._overlay, self._activation, self._model_scan):
            if effect is not None:
                effect.clear()

    def _close_effects(self) -> None:
        for effect in (self._overlay, self._activation, self._replay, self._model_scan):
            if effect is not None:
                effect.close()

    def compute_logits(self, model: Qwen2ForCausalLM, hidden: torch.Tensor) -> torch.Tensor | None:
        """vLLM's LM head. Post-filter guidance projects its predicted hidden rows in the same call.

        Decode-time LM heads are bound by reading the [vocab, hidden] weight, so projecting the
        prediction alongside the model's rows costs about as much as the model's rows alone. A request
        for logits means the transformer forward is over; MRv1 asks before it reports completion.
        """
        with torch.inference_mode(False), torch.enable_grad():
            self.complete_forward()
        lm_head = Qwen2ForCausalLM.compute_logits
        if (
            self._step is None
            or self._sampler is None
            or self._deployment.attachment.inputs.request.sampler_stage is not SamplerStage.PACKED_POST_FILTER_EXACT
        ):
            return lm_head(model, hidden)
        guidance = self._sampler_guidance()
        logits = lm_head(model, torch.cat((hidden, guidance.values.to(hidden.dtype))))
        assert logits is not None
        self._sampler.begin_guidance(SignedSamplerGuidance(guidance, logits[len(hidden) :], self._capture_feedback))
        return logits[: len(hidden)]

    def begin_sampler(self, model: Qwen2ForCausalLM) -> bool:
        if self._step is None or self._sampler is None:
            return False
        # Post-filter guidance began with the LM head; additive logits need no projection.
        if self._deployment.attachment.inputs.request.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD:
            guidance = self._sampler_guidance()
            self._sampler.begin_guidance(SignedSamplerGuidance(guidance, guidance.values, self._capture_feedback))
        return True

    def _sampler_guidance(self) -> PackedSamplerGuidanceBatch:
        execution = self._deployment.execution
        guidance = execution.sampler_guidance(self._invocation(), torch.cuda.current_stream(execution.device))
        if guidance.stage is not self._deployment.attachment.inputs.request.sampler_stage:
            raise ValueError("guidance stage differs from its signed attachment")
        return guidance

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
        overlap = self._post_sample
        if overlap is not None:
            assert isinstance(invocation.context.value, TensorFrame)
            overlap.copy_inputs(cast(StateTaskContext[TensorFrame], invocation.context))
            if overlap.readers:
                with torch.cuda.stream(overlap.stream):
                    overlap.retain(execution.resume_latest(invocation, overlap.stream))
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
        if self._post_sample is not None:
            self._post_sample.clear()
        self._auxiliary = None
        self._step = None

    def _retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        execution = self._deployment.execution
        held = (
            tuple(identity for identity in identities if execution.directory.has_inference_pin(identity))
            if execution.directory.scope is StateScope.INSTANCE
            else identities
        )
        if held:
            # Released pages remain fenced in the arena. Finishing a subset of the
            # requests must not make the host wait for unrelated side-stream work.
            execution.retire(held)
        for identity in identities:
            self._identities.pop(identity.request_id)
            self._token_frontiers.pop(identity, None)

    def retire_all(self, *, wait: bool = False) -> None:
        if self._step is not None:
            raise RuntimeError("canonical requests cannot retire during a step")
        if self._auxiliary is not None:
            raise RuntimeError("generation ended before the signed training input was consumed")
        if self._post_sample is not None:
            self._post_sample.clear()
        self._retire(tuple(self._identities[request_id] for request_id in sorted(self._identities)))
        self._pending_identities.clear()
        self._deployment.execution.arena.collect_completed(wait=wait)

    def close(self) -> None:
        self.abort_active()
        self.retire_all(wait=True)
        self.abort_tuning()
        execution = self._deployment.execution
        if execution.directory.scope is StateScope.INSTANCE:
            execution.retire((), wait=True)
        else:
            execution.retire_dormant(wait=True)
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


def input_reservations(
    probe: BackendPlacementProbe,
    timeline: SchedulerTimeline,
    rows: int,
    vocab: int,
    stage: SamplerStage | None,
    feedback: bool,
    publish_lag: int = 1,
) -> tuple[int, ...]:
    hidden = (1 + bool(publish_lag)) * len(probe.hidden_resources) * probe.max_packed_rows * probe.hidden_size * 4
    # FP32 envelope for original logits, conversion, projection and rowwise arithmetic.
    # Excludes upstream sampling/GEMM library scratch, model weights and KV storage.
    workspace = (
        rows * (36 * vocab + 4 * probe.hidden_size + 8)
        if stage is SamplerStage.PACKED_POST_FILTER_EXACT
        else rows * (12 * vocab + 8)
        if stage is SamplerStage.PACKED_PRE_FILTER_ADD
        else 0
    )
    retained = rows * (4 * vocab + 8) if feedback else 0
    # StepCapture retains its feedback pointers until the next table is prepared.
    # Budget both old and new private copies during handoff, including next forward.
    private_feedback = 2 * retained if publish_lag else 0
    sampler = next(anchor.ordinal for anchor in timeline.anchors if anchor.symbol == "sampler.pre_draw")
    return tuple(
        hidden
        + private_feedback
        + (workspace if anchor.symbol in ("sampler.pre_draw", "sampler.complete") else 0)
        + (retained if anchor.ordinal >= sampler else 0)
        for anchor in timeline.anchors
    )

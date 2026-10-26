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

from pllm.authoring import AuxiliaryInput, BackendDeployment, DataT, ResultT
from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.hidden_capture import HiddenTaps
from pllm.backends.vllm.overlay_graph import (
    HybridParameterOverlay,
    PinnedParameterOverlay,
    parameter_overlay,
)
from pllm.backends.vllm.parameter_overlay import (
    ExactParameterOverlay,
    validate_projection_sources,
)
from pllm.backends.vllm.position_state import PositionCacheLayer
from pllm.backends.vllm.prefix_cache import POLICY_KEY, CachePolicy
from pllm.backends.vllm.replay import VllmTokenwiseReplay
from pllm.backends.vllm.sampler import (
    SamplerGuidance,
    SignedSamplerGuidance,
)
from pllm.contracts.auxiliary import validate_auxiliary
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    SignedBackendAttachment,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorId,
    SamplerFeedbackInputSource,
    StateScope,
    TokenProgramEffect,
)
from pllm.execution.backend_attachment import attachment_buffer_bytes
from pllm.execution.codec import encode_boundary
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.overlay_task import ProjectionTask
from pllm.execution.post_sample import PostSampleOverlap
from pllm.execution.shared_tensor import SharedTensor
from pllm.execution.state_execution import StateDiagnostics, StateTaskContext
from pllm.execution.state_feed import (
    Feed,
    FeedReceiver,
    ReadonlyStepAbsent,
    ReadonlyStepEnd,
    ReadonlyStepInput,
)
from pllm.execution.step_input import TensorStepInput
from pllm.execution.tensor_frame import TensorFrame

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class _PackedStep(Generic[DataT]):
    rows: PackedSamplerRowBatch
    invocation: TaskGraphInvocation[StateTaskContext[DataT]] | None = None
    forward_done: bool = False


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
        position_layers: tuple[PositionCacheLayer, ...] = (),
        allocation_block_size: int = 16,
        *,
        cache_policy: CachePolicy,
    ) -> None:
        self._cache_envelope = encode_boundary(cache_policy)
        attachment = deployment.attachment
        plan = attachment.inputs.plan
        self._capture_feedback = any(
            isinstance(port.source, SamplerFeedbackInputSource)
            for task in plan.semantic_graph.tasks
            for port in task.inputs
        )
        visibility = plan.timeline.visibility
        buffers = attachment_buffer_bytes(
            attachment.inputs.probe, plan.semantic_graph, plan.tensor_layout_request, attachment.inputs.request
        )
        self._deployment = deployment
        self.trace = deployment.execution.trace
        probe = attachment.inputs.probe
        self._hidden_resources = tuple(resource.binding for resource in probe.hidden_resources)
        if self._hidden_resources and taps is None:
            raise ValueError("hidden inputs require taps installed when the model was loaded")
        self._taps = taps
        overlay_bindings = attachment.inputs.request.overlay_bindings
        self._overlay_import = bool(overlay_bindings) and (
            plan.semantic_graph.tasks[attachment.inputs.request.task_bindings[0].task_id].timing.after
            == "request.prepare"
        )
        routed_overlay = bool(overlay_bindings) and any(task.replay is not None for task in plan.semantic_graph.tasks)
        self._in_forward_overlay = plan.semantic_graph.updates_in_forward
        if routed_overlay or self._in_forward_overlay or probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH:
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
                routed=routed_overlay,
            )
            if overlay_bindings
            else None
        )
        tasks = tuple(task for task in attachment.inputs.plan.semantic_graph.tasks if task.replay is not None)
        self._activation = (
            VllmActivationPatch(
                model,
                attachment.inputs.request.activation_bindings,
                attachment.inputs.request.state_resource,
                install=not any(isinstance(task.replay, TokenProgramEffect) for task in tasks),
            )
            if (attachment.inputs.request.activation_bindings)
            else None
        )
        # Recomputing history after preemption would rerun model-side effects on today's state, not the
        # state each position originally saw, so these workloads need every token computed exactly once.
        self._unbroken_history = bool(
            attachment.inputs.request.overlay_bindings or attachment.inputs.request.activation_bindings
        )
        self._token_frontiers: dict[RequestIdentity, int] = {}
        self._feed: FeedReceiver | None = None
        self._readonly_receiver: ReadonlyStepReceiver | None = None
        self._received_readonly: ReadonlyStepInput | ReadonlyStepAbsent | ReadonlyStepEnd | None = None
        self._input_failed = False
        self._replay: VllmTokenwiseReplay[DataT, ResultT] | None = None
        if tasks:
            (task,) = tasks
            if probe.causal_domain is None or probe.causal_domain.layers != len(model.model.layers):
                raise ValueError("backend replay requires its signed causal model domain")
            if isinstance(task.replay, TokenProgramEffect):
                self._replay = VllmTokenwiseReplay(
                    model,
                    task,
                    deployment.execution,
                    attachment.inputs.request.task_bindings[0].anchor_id,
                    source_imports=probe.region_imports,
                    function_bodies=probe.region_bodies,
                    activation=self._activation,
                    position_layers=position_layers,
                    allocation_block_size=allocation_block_size,
                    overlay=self._overlay
                    if isinstance(self._overlay, ExactParameterOverlay) and routed_overlay
                    else None,
                )
            else:
                raise ValueError("backend replay does not match its signed primitives")
        self._sampler = sampler
        self._identities: dict[str, RequestIdentity] = {}
        self._pending_identities: dict[str, RequestIdentity] = {}
        self._next_step_id = 0
        self._step: _PackedStep[DataT] | None = None
        self._auxiliary: AuxiliaryInput[torch.Tensor] | None = None
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
        if execution.feeds:
            (binding,) = execution.feeds
            self._feed = FeedReceiver(execution.arena.plan, binding)
            self.bind_readonly_input(self._feed)

    @property
    def parameter_phases(self) -> tuple[HiddenPhase, ...]:
        """Authored update phases used by hybrid model dispatch."""
        return self._deployment.execution.math.parameter_phases

    def stage_input(self, batch: AuxiliaryInput[torch.Tensor] | None, identities: tuple[RequestIdentity, ...]) -> None:
        self._require_input_live()
        if self._auxiliary is not None or self._step is not None or self._pending_identities or self._identities:
            raise RuntimeError("prior canonical training input is still active")
        validate_auxiliary(batch, self._deployment.attachment.inputs.plan, SharedTensor.describe)
        self._auxiliary = batch
        self._pending_identities = {identity.request_id: identity for identity in identities}

    def feed(self, name: str) -> Feed:
        """Obtain the declared writer while the consumer is idle."""
        if self._feed is None or self._next_step_id or self._step is not None:
            raise RuntimeError("declare a fed state and obtain its writer before generate")
        return self._feed.take(name)

    def bind_readonly_input(self, receiver: ReadonlyStepReceiver) -> None:
        """Install the framework transport before requests; it supplies no model mathematics."""
        self._require_input_live()
        if (
            self._readonly_receiver is not None
            or self._step is not None
            or self._next_step_id
            or self._identities
            or self._pending_identities
            or self._auxiliary is not None
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
            if (request.sampling_params.extra_args or {}).get(POLICY_KEY) != self._cache_envelope:
                raise ValueError("request cache policy differs from installed workload; submit through its handle")
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
        if self._step is not None:
            raise RuntimeError("canonical input preparation overlaps a step")
        if rows.step_id != self._next_step_id or rows.plan_epoch != self.attachment.physical.plan_epoch:
            raise ValueError("backend rows differ from the original execution step")
        if rows.row_batch.total_rows > self.attachment.inputs.probe.max_packed_rows:
            raise ValueError("packed input exceeds the signed backend row capacity")
        if self._unbroken_history:
            if any(row.token_offset < self._token_frontiers.get(row.identity, 0) for row in rows.row_batch.rows):
                self._input_failed = True
                raise RuntimeError(
                    "vLLM preempted a request and would recompute its history, but this workload's parameter "
                    "overlays or activation patches cannot replay past tokens; give vLLM more KV cache "
                    "(gpu_memory_utilization) or fewer concurrent sequences (max_num_seqs)"
                )
            for row in rows.row_batch.rows:
                self._token_frontiers[row.identity] = max(
                    self._token_frontiers.get(row.identity, 0), row.token_offset + 1
                )
        requests = rows.identities
        self._next_step_id += 1
        step = _PackedStep[DataT](rows)
        self._step = step
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
        for effect in (self._overlay, self._activation, self._replay):
            if effect is not None:
                effect.clear()

    def _abort_effects(self) -> None:
        for effect in (self._overlay, self._activation):
            if effect is not None:
                effect.clear()

    def _close_effects(self) -> None:
        for effect in (self._overlay, self._activation, self._replay):
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

    def require_idle(self) -> None:
        """Workload replacement cannot discard admitted or staged requests."""
        if self._step is not None or self._identities or self._pending_identities:
            raise RuntimeError("finish or abort active requests before replacing the workload")

    def close(self) -> None:
        try:
            self.abort_active()
            self.retire_all(wait=True)
            execution = self._deployment.execution
            if execution.directory.scope is StateScope.INSTANCE:
                execution.retire((), wait=True)
            else:
                execution.retire_dormant(wait=True)
            self._close_effects()
        finally:
            if self._feed is not None:
                self._input_failed = True
                self._deployment.execution.discard_borrows()
                self._feed.close()

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

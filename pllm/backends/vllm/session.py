"""One vLLM session: a signed tensor-state invocation driven step by step inside the model runner.

VllmSession owns the per-step lifecycle (row preparation, model effects, sampler guidance, completion and
publication) for one attached workload. Runners drive its hooks; the worker and driver stage and retire requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

import torch
from vllm.v1.core.sched.output import SchedulerOutput

from pllm.authoring import BackendDeployment, DataT, HostBatch, ResultT
from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.architecture import Decoder
from pllm.backends.vllm.custom_forward import VllmCustomForward
from pllm.backends.vllm.hidden_taps import HiddenTaps
from pllm.backends.vllm.layer_logits import LayerLogits
from pllm.backends.vllm.model_sites import ModelSites
from pllm.backends.vllm.parameter_overlay import ExactParameterOverlay, parameter_overlay
from pllm.backends.vllm.position_state import PositionCacheLayer
from pllm.backends.vllm.prefix_cache import POLICY_KEY, CachePolicy
from pllm.backends.vllm.sampler import (
    SamplerGuidance,
    SignedSamplerGuidance,
)
from pllm.contracts.backend_attachment import AttachmentGraphMode, BackendAttachment
from pllm.contracts.codec import encode_boundary
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import (
    SamplerFeedbackInputSource,
)
from pllm.execution.backend_attachment import attachment_buffer_bytes
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.host_runtime import CompletionRows
from pllm.execution.overlay_task import ProjectionTask
from pllm.execution.state_execution import HeldInput, StateDiagnostics, StateTaskContext
from pllm.execution.state_feed import (
    Feed,
    FeedReceiver,
    ReadonlyStepAbsent,
    ReadonlyStepEnd,
    ReadonlyStepInput,
)
from pllm.execution.step_input import StepInput


@dataclass(slots=True)
class _PackedStep(Generic[DataT]):
    rows: PackedSamplerRowBatch
    context: StateTaskContext[DataT] | None = None
    forward_done: bool = False


class VllmSession(Generic[DataT, ResultT]):
    """Model primitives over one original signed tensor-state invocation."""

    def __init__(
        self,
        deployment: BackendDeployment[DataT, ResultT],
        decoder: Decoder,
        sampler: SamplerGuidance | None,
        max_sampler_rows: int,
        taps: HiddenTaps | None = None,
        position_layers: tuple[PositionCacheLayer, ...] = (),
        allocation_block_size: int = 16,
        *,
        cache_policy: CachePolicy,
    ) -> None:
        self._compute_logits = decoder.compute_logits
        self._cache_envelope = encode_boundary(cache_policy)
        attachment = deployment.attachment
        plan = attachment.plan
        self._capture_feedback = any(
            isinstance(port.source, SamplerFeedbackInputSource)
            for task in plan.semantic_graph.tasks
            for port in task.inputs
        )
        buffers = attachment_buffer_bytes(
            attachment.probe, plan.semantic_graph, plan.tensor_layout_request, attachment.spec
        )
        self._deployment = deployment
        self.trace = deployment.execution.trace
        probe = attachment.probe
        self._hidden_resources = tuple(dict.fromkeys(resource.binding for resource in probe.hidden_resources))
        self._hidden_phases = frozenset(resource.phase for resource in probe.hidden_resources)
        self._trains_on_batch = plan.semantic_graph.trains_on_batch
        self._taps = taps
        self._layer_logits = None
        if probe.layer_logits:
            assert taps is not None
            self._layer_logits = LayerLogits(decoder, taps, probe.layer_logits)
        overlay_bindings = attachment.spec.overlay_bindings
        custom_forward_task = plan.semantic_graph.custom_forward_task
        program_overlay = bool(overlay_bindings) and custom_forward_task is not None
        self._overlay = (
            parameter_overlay(
                decoder,
                overlay_bindings,
                deployment.execution.arena,
                probe.max_packed_rows,
                max_sampler_rows,
                buffers,
                phases=deployment.execution.math.parameter_phases,
                in_forward=plan.semantic_graph.updates_in_forward,
                graphs=probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH,
                in_program=program_overlay,
            )
            if overlay_bindings
            else None
        )
        # The overlay prepared for each model forward; a custom forward drives its program overlay itself.
        self._forward_overlay = None if program_overlay else self._overlay
        self._in_forward_overlay = plan.semantic_graph.updates_in_forward and not program_overlay
        in_forward = attachment.spec.in_forward
        self._transition_task = None if in_forward is None else plan.scheduled(in_forward).task_id
        self._activation = (
            VllmActivationPatch(decoder.model, attachment.spec.activation_bindings, install=custom_forward_task is None)
            if attachment.spec.activation_bindings
            else None
        )
        model_sites = deployment.execution.math.model_sites
        self._sites = (
            ModelSites(decoder, model_sites, taps, probe.max_packed_rows, max_sampler_rows) if model_sites else None
        )
        self._feed: FeedReceiver | None = None
        self._received_readonly: ReadonlyStepInput | ReadonlyStepAbsent | ReadonlyStepEnd | None = None
        self._input_failed = False
        self._custom_forward: VllmCustomForward[DataT, ResultT] | None = None
        if custom_forward_task is not None:
            self._custom_forward = VllmCustomForward(
                decoder.kv_writable("a @pllm.custom_forward"),
                custom_forward_task,
                deployment.execution,
                plan.scheduled(custom_forward_task.symbol).anchor_id,
                activation=self._activation,
                sites=self._sites,
                position_layers=position_layers,
                allocation_block_size=allocation_block_size,
                overlay=self._overlay if isinstance(self._overlay, ExactParameterOverlay) and program_overlay else None,
            )
        self._sampler = sampler
        self._identities: dict[str, RequestIdentity] = {}
        self._pending_identities: dict[str, RequestIdentity] = {}
        self._next_step_id = 0
        self._step: _PackedStep[DataT] | None = None
        self._host_batch: HostBatch[torch.Tensor] | None = None
        self.completion_rows: CompletionRows | None = None
        execution = deployment.execution
        execution.admit_instances(torch.cuda.current_stream(execution.device))
        if execution.feeds:
            (binding,) = execution.feeds
            self._feed = FeedReceiver(execution.arena, binding)

    @property
    def parameter_phases(self) -> tuple[HiddenPhase, ...]:
        """Authored update phases used by hybrid model dispatch."""
        return self._deployment.execution.math.parameter_phases

    def stage_input(
        self,
        batch: HostBatch[torch.Tensor] | None,
        identities: tuple[RequestIdentity, ...],
        completion: torch.Tensor | None = None,
    ) -> None:
        """Stage request inputs, already validated by generate, and their optional completion destination."""
        self._require_input_live()
        self.completion_rows = None if completion is None else CompletionRows(identities, completion)
        self._host_batch = batch
        self._pending_identities = {identity.request_id: identity for identity in identities}

    def feed(self, name: str) -> Feed:
        """Obtain the declared writer while the consumer is idle."""
        if self._feed is None or self._next_step_id or self._step is not None:
            raise RuntimeError("declare a fed state and obtain its writer before generate")
        return self._feed.take(name)

    def _require_input_live(self) -> None:
        if self._input_failed:
            raise RuntimeError("canonical step input is fail-stopped; received grants remain retained")

    def _receive_readonly(self, rows: PackedSamplerRowBatch, stream: torch.cuda.Stream) -> HeldInput | None:
        if self._feed is None:
            return None
        try:
            self._received_readonly = self._feed(rows, stream)
            received = self._received_readonly
            if received.step_id != rows.step_id:
                raise ValueError("read-only input differs from the requested step; grants retained")
            held_input = None
            if isinstance(received, (ReadonlyStepAbsent, ReadonlyStepEnd)):
                held_input = self._deployment.execution.stage_hold(
                    tuple(rows.identities),
                    "end" if isinstance(received, ReadonlyStepEnd) else "absent",
                    step_id=rows.step_id,
                )
            else:
                self._deployment.execution.stage_readonly(received.roots)
            self._received_readonly = None  # Ownership is now the original execution's pending/active borrow.
            return held_input
        except BaseException:
            self._input_failed = True
            raise

    def identity(self, request_id: str) -> RequestIdentity:
        return self._identities[request_id]

    @property
    def attachment(self) -> BackendAttachment:
        return self._deployment.attachment

    def diagnostics(self) -> StateDiagnostics:
        return self._deployment.execution.diagnostics()

    @property
    def apply_count(self) -> int:
        return 0 if self._overlay is None else self._overlay.apply_count

    @property
    def updated_apply_count(self) -> int:
        return 0 if self._overlay is None else self._overlay.updated_apply_count

    @torch.inference_mode(False)
    def before_execute(self, scheduler_output: SchedulerOutput) -> None:
        self._require_input_live()
        execution = self._deployment.execution
        execution.arena.collect_completed()
        finished = tuple(
            self._identities[request_id]
            for request_id in sorted(self._identities.keys() & scheduler_output.finished_req_ids)
        )
        self._retire(finished)
        # Sampler/hidden-only workloads retain their current state and preemption behavior.
        if self._overlay is not None or self._activation is not None or self._sites is not None:
            for request_id in scheduler_output.finished_req_ids:
                self._pending_identities.pop(request_id, None)
            # Installed vLLM 0.27.1 emits this set for both runners, despite its MRv2-only field comment.
            self._preempt(scheduler_output.preempted_req_ids or set())
        pending: list[RequestIdentity] = []
        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id in self._identities:
                continue
            params = request.sampling_params
            if params is None or (params.extra_args or {}).get(POLICY_KEY) != self._cache_envelope:
                raise ValueError("request cache policy differs from installed workload; submit through its handle")
            pending.append(self._pending_identities[request.req_id])
        # MRv2 sends resumed requests as new; MRv1 retains their cached request data.
        pending.extend(
            self._pending_identities[request_id]
            for request_id in sorted(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
            if request_id not in self._identities
        )
        if pending:
            if bool(execution.directory.request_states):
                execution.admit(tuple(pending), torch.cuda.current_stream(execution.device))
            for identity in pending:
                self._identities[identity.request_id] = identity
                self._pending_identities.pop(identity.request_id)

    def _preempt(self, request_ids: set[str]) -> None:
        identities = tuple(self._identities[key] for key in sorted(self._identities.keys() & request_ids))
        if not identities:
            return
        rejection = self.attachment.plan.schedule.recompute_rejection
        if rejection is not None:
            self._input_failed = True
            raise RuntimeError(
                "vLLM preempted a request and would recompute its history; "
                f"{rejection}. To avoid preemption, give vLLM more KV cache "
                "(gpu_memory_utilization) or fewer concurrent sequences (max_num_seqs)"
            )
        self._retire(identities)
        self._pending_identities.update((identity.request_id, identity) for identity in identities)

    @property
    def next_step_id(self) -> int:
        return self._next_step_id

    @torch.inference_mode(False)
    @torch.enable_grad()
    def prepare_input(self, rows: PackedSamplerRowBatch) -> None:
        self._require_input_live()
        self._next_step_id += 1
        step = _PackedStep[DataT](rows)
        self._step = step
        if self._taps is not None:
            self._taps.prepare(rows.row_batch)
        adapter = self._forward_overlay
        if (
            adapter is None
            and self._activation is None
            and self._sites is None
            and self._custom_forward is None
            and not self._trains_on_batch
        ):
            return
        if adapter is not None:
            adapter.reserve()
        execution = self._deployment.execution
        identities = tuple(
            sorted(
                rows.identities,
                key=lambda identity: (identity.request_id, identity.prompt_index, identity.sample_index),
            )
        )
        stream = torch.cuda.current_stream(execution.device)
        if not self._in_forward_overlay:
            held_input = self._receive_readonly(rows, stream)
            value = StepInput(rows.row_batch, rows, batch=self._host_batch)
            context = execution.start_step(value, stream, held_input=held_input)
            instance_states = bool(execution.directory.instance_states)
            if instance_states or not self._pending_identities:
                self._host_batch = None
            step.context = context
            execution.advance_to(context, "request.prepare")
            # Offline publication precedes admission; existing request pins remain unchanged.
            pending = tuple(
                identity
                for identity in identities
                if instance_states and not execution.directory.has_inference_pin(identity)
            )
            if pending:
                execution.admit(pending, stream)
        else:
            context = execution.begin(rows.row_batch.request_rows, stream)
            step.context = context
        if adapter is not None:
            projection = None
            if self._in_forward_overlay:
                assert self._transition_task is not None
                execution.advance_to(context, "request.prepare")
                projection = ProjectionTask(
                    execution,
                    context,
                    rows,
                    self._transition_task,
                    tuple(item.target for item in self.attachment.spec.overlay_bindings),
                )
            adapter.prepare(rows.row_batch, context, execution.directory, projection)
        if self._activation is not None:
            self._activation.prepare(rows.row_batch, context, execution.directory)
        if self._sites is not None:
            self._sites.prepare(rows.row_batch, context, execution.directory)
        if self._custom_forward is not None:
            self._custom_forward.prepare(rows.row_batch, context)

    @torch.inference_mode(False)
    @torch.enable_grad()
    def complete_forward(self) -> None:
        step = self._step
        if step is None or step.forward_done:
            return
        step.forward_done = True
        execution = self._deployment.execution
        stream = torch.cuda.current_stream(execution.device)
        operands = () if self._overlay is None else self._overlay.take_operands()
        value = StepInput(step.rows.row_batch, step.rows, operands=operands, batch=self._host_batch)
        if self._taps is not None and self._hidden_resources:
            value = self._taps.publish(value, self._hidden_resources)
        if (
            step.context is None
            and not operands
            and not any(row.phase in self._hidden_phases for row in step.rows.row_batch.rows)
        ):
            self._finish_effects()
            self._step = None
            return
        if step.context is None:
            overlap = execution.post_sample
            if overlap is not None and not overlap.lag:
                stream = overlap.stream
            step.context = execution.start_step(value, stream, history=() if overlap is None else overlap.history)
            self._host_batch = None
        elif step.context.value is None or self._hidden_resources or operands:
            execution.submit_input(step.context, value, CudaStreamReadiness(stream))
        if self._taps is not None and self._hidden_resources:
            self._taps.hold(step.context)
        execution.advance_to(step.context, "sampler.pre_draw")
        self._finish_effects()

    def _finish_effects(self) -> None:
        """Clear every active optional primitive after forward or abort."""
        for effect in (self._taps, self._overlay, self._activation, self._custom_forward):
            if effect is not None:
                effect.clear()

    def _close_effects(self) -> None:
        for effect in (self._overlay, self._activation, self._sites, self._custom_forward):
            if effect is not None:
                effect.close()

    @torch.inference_mode()
    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor | None:
        """vLLM's LM head. Post-filter guidance projects its predicted hidden rows in the same call.

        Decode-time LM heads are bound by reading the [vocab, hidden] weight, so projecting the
        prediction alongside the model's rows costs about as much as the model's rows alone. The extra
        rows change the model rows' logits only as a larger batch would (cuBLAS picks kernels by row
        count); on Qwen2.5-0.5B and a DeepSeek-V2-Lite slice that never exceeded plain vLLM's own
        batch-size sensitivity (0.0625 and 0.5). Steps without guided rows append none, and zero-strength
        guidance declares no sampler. Layer logits need the model's rows before the workload's tasks run,
        so with them the model's rows and the guidance are projected in separate calls. A request for logits
        means the transformer forward is over on both runner ABIs.
        """
        lm_head = self._compute_logits
        step = self._step
        # Prompt logprobs call the LM head again; capture and guidance run only for the sampling rows.
        if step is None or step.forward_done:
            return lm_head(hidden)
        # Forward completion can discard a prefill step that has no workload inputs.
        if self.completion_rows is not None:
            self.completion_rows.write(step.rows.identities, hidden)
        logits = None
        if self._layer_logits is not None:
            logits = lm_head(hidden)
            assert logits is not None
            self._layer_logits.compute(step.rows, logits)
        self.complete_forward()
        if self._step is None or self._sampler is None or self._context().guidance is None:
            return lm_head(hidden) if logits is None else logits
        guidance = self._sampler_guidance()
        if guidance.stage is not SamplerStage.PACKED_POST_FILTER_EXACT:
            predicted = guidance.values
        elif logits is None:
            projected = lm_head(torch.cat((hidden, guidance.values.to(hidden.dtype))))
            assert projected is not None
            logits, predicted = projected[: len(hidden)], projected[len(hidden) :]
        else:
            predicted = lm_head(guidance.values.to(hidden.dtype))
            assert predicted is not None
        logits = lm_head(hidden) if logits is None else logits
        self._sampler.begin_guidance(SignedSamplerGuidance(guidance, predicted, self._capture_feedback))
        return logits

    def _sampler_guidance(self) -> PackedSamplerGuidanceBatch:
        execution = self._deployment.execution
        return execution.sampler_guidance(self._context(), torch.cuda.current_stream(execution.device))

    @torch.inference_mode(False)
    @torch.enable_grad()
    def complete_sampler(self) -> None:
        if self._step is None:
            return
        feedback = None
        if self._sampler is not None:
            feedback = self._sampler.finish_guidance()
        execution = self._deployment.execution
        context = self._context()
        execution.execute_anchor(context, execution.sampler_anchor_id)
        execution.complete_sampler(context, torch.cuda.current_stream(execution.device), feedback=feedback)

    @torch.inference_mode(False)
    @torch.enable_grad()
    def commit_token(self) -> None:
        if self._step is None:
            return
        execution = self._deployment.execution
        context = self._context()
        execution.advance_to(context, "next.prepare")
        execution.finish_step(context)
        self._step = None
        execution.arena.collect_completed()

    @torch.inference_mode(False)
    def abort_active(self) -> None:
        if self._sampler is not None:
            self._sampler.abort_guidance()
        self._finish_effects()
        self._deployment.execution.cancel_active()
        self._host_batch = None
        self._step = None

    def _retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        execution = self._deployment.execution
        held = (
            tuple(identity for identity in identities if execution.directory.has_inference_pin(identity))
            if not execution.directory.request_states
            else identities
        )
        if held:
            # Released pages remain fenced in the arena. Finishing a subset of the
            # requests must not make the host wait for unrelated side-stream work.
            execution.retire(held)
        for identity in identities:
            self._identities.pop(identity.request_id)

    @torch.inference_mode(False)
    def retire_all(self, *, wait: bool = False) -> None:
        if self.completion_rows is not None:
            self.completion_rows.synchronize()
        self._deployment.execution.release_history()
        self._retire(tuple(self._identities[request_id] for request_id in sorted(self._identities)))
        self._pending_identities.clear()
        self._deployment.execution.arena.collect_completed(wait=wait)
        self.completion_rows = None

    def finish_requests(self) -> None:
        """Require every submitted completion row before releasing the generation's requests."""
        if self.completion_rows is not None:
            self.completion_rows.require_complete()
        self.retire_all()

    def require_idle(self) -> None:
        """Workload replacement cannot discard admitted or staged requests."""
        if self._step is not None or self._identities or self._pending_identities:
            raise RuntimeError("finish or abort active requests before replacing the workload")

    @torch.inference_mode(False)
    def close(self) -> None:
        try:
            self.abort_active()
            self.retire_all(wait=True)
            execution = self._deployment.execution
            if bool(execution.directory.instance_states):
                execution.retire((), wait=True)
            if execution.directory.request_states:
                execution.retire_dormant(wait=True)
            self._close_effects()
        finally:
            try:
                self._deployment.execution.close()
            finally:
                if self._feed is not None:
                    self._input_failed = True
                    self._deployment.execution.discard_borrows()
                    self._feed.close()

    def _context(self) -> StateTaskContext[DataT]:
        assert self._step is not None and self._step.context is not None
        return self._step.context

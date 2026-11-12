"""One vLLM session: the model-side effects of an attached workload and the translation of vLLM's runner callbacks
into StepDriver calls.

VllmSession builds the hooks a workload installs on the model (overlays, activation patches, residual sites, custom
attention and forward, hidden taps) and the sampler guidance, and prepares them for each step; the step's order of
state calls is the StepDriver's. Runners call its hooks; the worker and driver stage and retire requests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.v1.core.sched.output import SchedulerOutput

from pllm.backends.vllm.activation import VllmActivationPatch
from pllm.backends.vllm.architecture import Decoder
from pllm.backends.vllm.custom_attention import CustomAttention
from pllm.backends.vllm.custom_forward import VllmCustomForward
from pllm.backends.vllm.hidden_taps import HiddenTaps
from pllm.backends.vllm.layer_logits import LayerLogits
from pllm.backends.vllm.model_sites import ModelSites
from pllm.backends.vllm.parameter_overlay import ParameterOverlay
from pllm.backends.vllm.position_state import PositionCacheLayer
from pllm.backends.vllm.prefix_cache import POLICY_KEY, CachePolicy
from pllm.backends.vllm.sampler import (
    BoundedSamplerGuidance,
    GuidedSampler,
    SignedSamplerGuidance,
    SignedSamplerStages,
)
from pllm.contracts.backend_attachment import AttachmentGraphMode, BackendAttachment
from pllm.contracts.codec import encode_boundary
from pllm.contracts.host_batch import HostBatch
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage
from pllm.contracts.semantic_graph import SamplerFeedbackInputSource, TaskResourceFact, TaskSymbol
from pllm.execution.backend_attachment import attachment_buffer_bytes
from pllm.execution.cuda_stream import PinnedStaging
from pllm.execution.host_runtime import CompletionRows
from pllm.execution.overlay_task import ProjectionTask
from pllm.execution.state_execution import StateDiagnostics
from pllm.execution.step_capture import StepCapture
from pllm.execution.step_driver import StepDriver, StepOpen
from pllm.execution.step_input import StepInput, gather_input_tokens

if TYPE_CHECKING:
    from pllm.workload import BackendDeployment


class VllmSession:
    """Model primitives over one original signed tensor-state invocation."""

    def __init__(
        self,
        deployment: BackendDeployment,
        decoder: Decoder,
        sampler: GuidedSampler | None,
        max_sampler_rows: int,
        taps: HiddenTaps | None = None,
        position_layers: tuple[PositionCacheLayer, ...] = (),
        allocation_block_size: int = 16,
        *,
        cache_policy: CachePolicy,
    ) -> None:
        self._compute_logits = decoder.compute_logits
        capacity = deployment.attachment.spec.guidance_capacity
        self._guidance_weight = None
        if capacity is not None and any(
            stage.stage is SamplerStage.PACKED_POST_FILTER_EXACT for stage in deployment.attachment.spec.sampler
        ):
            assert sampler is not None
            self._guidance_weight = decoder.model.lm_head.weight
        self.cache_envelope = encode_boundary(cache_policy)
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
        self.deployment = deployment
        self.trace = deployment.execution.trace
        probe = attachment.probe
        self._hidden_resources = tuple(dict.fromkeys(resource.binding for resource in probe.hidden_resources))
        self.hidden_taps = taps
        self._layer_logits = None
        if probe.layer_logits:
            assert taps is not None
            self._layer_logits = LayerLogits(decoder, taps, probe.layer_logits)
        overlay_bindings = attachment.spec.overlay_bindings
        custom_forward_task = plan.semantic_graph.custom_forward_task
        program_overlay = attachment.spec.in_forward is not None and custom_forward_task is not None
        self.parameter_overlay = (
            ParameterOverlay(
                decoder,
                overlay_bindings,
                deployment.execution.arena,
                probe.max_packed_rows,
                max_sampler_rows,
                buffers,
                attachment.spec.overlay_phases,
                in_forward=plan.semantic_graph.updates_in_forward,
                graphs=probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH,
                in_program=custom_forward_task is not None,
                math=deployment.execution.math.parameter_overlay,
            )
            if overlay_bindings
            else None
        )
        # The overlay prepared for each model forward; a custom forward drives its program overlay itself.
        self.forward_overlay = None if program_overlay else self.parameter_overlay
        in_forward = attachment.spec.in_forward
        self._transition_task = None if in_forward is None else plan.scheduled(in_forward).task_id
        self.activation_patch = (
            VllmActivationPatch(attachment.spec.activation_bindings, probe.max_packed_rows, deployment.execution.arena)
            if attachment.spec.activation_bindings
            else None
        )
        history = plan.semantic_graph.input_token_history
        self.input_tokens = (
            torch.empty(
                (probe.max_packed_rows, history + 1),
                dtype=torch.long,
                device=decoder.model.lm_head.weight.device,
            )
            if history >= 0
            else None
        )
        self._token_staging = PinnedStaging()
        self._token_table = None if self.input_tokens is None else self.input_tokens.new_empty(max_sampler_rows + 2)
        self.token_capture = (
            None
            if self._token_table is None
            else StepCapture(TaskResourceFact(TaskSymbol("input_tokens"), 0, self._token_table.numel() * 8), 1)
        )
        attention = deployment.execution.math.attention
        model_sites = deployment.execution.math.model_sites
        self.model_sites = (
            ModelSites(
                decoder,
                model_sites,
                taps,
                self.activation_patch,
                probe.max_packed_rows,
                max_sampler_rows,
                arena=deployment.execution.arena,
                graphs=probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH,
                attention_program=attention,
            )
            if model_sites
            or (attention is not None and attention.methods)
            or (custom_forward_task is not None and taps is not None)
            else None
        )
        self._attention = (
            CustomAttention(decoder, attention, self.model_sites)
            if attention is not None and attention.methods and self.model_sites is not None
            else None
        )
        self.driver = StepDriver(deployment.execution, attachment)
        self.custom_forward: VllmCustomForward | None = None
        if custom_forward_task is not None:
            self.custom_forward = VllmCustomForward(
                decoder.kv_writable("a @pllm.custom_forward"),
                custom_forward_task,
                self.driver,
                plan.scheduled(custom_forward_task.symbol).anchor_id,
                activation=self.activation_patch,
                sites=self.model_sites,
                position_layers=position_layers,
                allocation_block_size=allocation_block_size,
                projections=self.parameter_overlay,
            )
        self._sampler = sampler
        self.completion_rows: CompletionRows | None = None
        execution = deployment.execution
        execution.admit_instances(torch.cuda.current_stream(execution.device))

    def stage_input(
        self,
        batch: HostBatch[torch.Tensor] | None,
        identities: tuple[RequestIdentity, ...],
        completion: torch.Tensor | None = None,
        *,
        samples_per_prompt: int = 1,
    ) -> None:
        """Stage request inputs, already validated by generate, and their optional completion destination."""
        self.driver.stage(identities, batch, samples_per_prompt)
        self.completion_rows = None if completion is None else CompletionRows(identities, completion)

    @property
    def attachment(self) -> BackendAttachment:
        return self.deployment.attachment

    def diagnostics(self) -> StateDiagnostics:
        return self.deployment.execution.diagnostics()

    @property
    def apply_count(self) -> int:
        return 0 if self.parameter_overlay is None else self.parameter_overlay.apply_count

    @property
    def updated_apply_count(self) -> int:
        return 0 if self.parameter_overlay is None else self.parameter_overlay.updated_apply_count

    @torch.inference_mode(False)
    def before_execute(self, scheduler_output: SchedulerOutput) -> None:
        new: list[str] = []
        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id in self.driver.admitted:
                continue
            params = request.sampling_params
            if params is None or (params.extra_args or {}).get(POLICY_KEY) != self.cache_envelope:
                raise ValueError("request cache policy differs from installed workload; submit through its handle")
            new.append(request.req_id)
        self.driver.schedule(scheduler_output.finished_req_ids, scheduler_output.preempted_req_ids or set(), new)

    def prepare_tokens(
        self, history: torch.Tensor, indices: tuple[int, ...], starts: torch.Tensor, positions: torch.Tensor, rows: int
    ) -> None:
        """Gather native request histories in one captured kernel using persistent step metadata."""
        table, capture, output = self._token_table, self.token_capture, self.input_tokens
        assert table is not None and capture is not None and output is not None
        capture.order(torch.cuda.current_stream(table.device))
        with self._token_staging.fill(table) as host:
            host[:2] = len(indices), rows
            host[2 : len(indices) + 2] = indices
        capture.run(
            lambda: gather_input_tokens(history, table, starts, positions, output),
            table,
            table[:0],
            (history.data_ptr(), starts.data_ptr(), positions.data_ptr(), output.data_ptr()),
            lambda: self._token_staging.copy_to(table),
        )

    @torch.inference_mode(False)
    @torch.enable_grad()
    def prepare_input(self, rows: PackedSamplerRowBatch) -> None:
        """Prepare the model's hooks for the step's rows, after the driver opened the step's state invocation."""
        if self.hidden_taps is not None:
            self.hidden_taps.prepare(rows.row_batch)
        if self._attention is not None:
            self._attention.prepare(rows.row_batch, rows)
        adapter = self.forward_overlay
        if adapter is not None:
            adapter.reserve()
        execution = self.deployment.execution
        stream = torch.cuda.current_stream(execution.device)
        context = self.driver.begin(StepInput(rows.row_batch, rows, input_tokens=self.input_tokens), stream)
        if context is None:
            return
        if adapter is not None:
            projection = None
            if self.driver.open is StepOpen.IN_FORWARD:
                assert self._transition_task is not None
                targets = tuple(item.target for item in self.attachment.spec.overlay_bindings)
                projection = ProjectionTask(self.driver, context, rows, self._transition_task, targets)
            adapter.prepare(rows.row_batch, context, execution.directory, projection)
        if self.activation_patch is not None:
            self.activation_patch.prepare(rows.row_batch, context, execution.directory)
        if self.model_sites is not None:
            self.model_sites.prepare(rows.row_batch, context, execution.directory)
        if self.custom_forward is not None:
            self.custom_forward.prepare(rows.row_batch, context)

    @torch.inference_mode(False)
    @torch.enable_grad()
    def complete_forward(self) -> None:
        step = self.driver.step
        if step is None or step.forward_done:
            return
        operands = () if self.parameter_overlay is None else self.parameter_overlay.take_operands()
        value = StepInput(step.rows.row_batch, step.rows, operands=operands, input_tokens=self.input_tokens)
        holds = ()
        if self.hidden_taps is not None and self._hidden_resources:
            value = self.hidden_taps.publish(value, self._hidden_resources)
            holds = self.hidden_taps.holds()
        self.driver.after_forward(value, torch.cuda.current_stream(self.deployment.execution.device), holds)
        self._finish_effects()

    def _finish_effects(self) -> None:
        """Clear every active optional primitive after forward or abort."""
        for effect in (self.hidden_taps, self.parameter_overlay, self.activation_patch, self.custom_forward):
            if effect is not None:
                effect.clear()

    def _close_effects(self) -> None:
        for effect in (self._attention, self.parameter_overlay, self.model_sites, self.custom_forward):
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
        means the transformer forward is over.
        """
        lm_head = self._compute_logits
        step = self.driver.step
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
        if self.driver.step is None or self._sampler is None or not self.driver.context().guidance:
            return lm_head(hidden) if logits is None else logits
        stages: list[SignedSamplerGuidance | BoundedSamplerGuidance] = []
        for guidance in self._sampler_guidance():
            if guidance.stage is SamplerStage.PACKED_POST_FILTER_EXACT and self._guidance_weight is not None:
                capacity = self.attachment.spec.guidance_capacity
                assert capacity is not None
                stages.append(BoundedSamplerGuidance(guidance, self._guidance_weight, capacity))
                continue
            if guidance.stage is not SamplerStage.PACKED_POST_FILTER_EXACT:
                predicted = guidance.values
            elif logits is None:
                projected = lm_head(torch.cat((hidden, guidance.values.to(hidden.dtype))))
                assert projected is not None
                logits, predicted = projected[: len(hidden)], projected[len(hidden) :]
            else:
                predicted = lm_head(guidance.values.to(hidden.dtype))
                assert predicted is not None
            stages.append(SignedSamplerGuidance(guidance, predicted))
        logits = lm_head(hidden) if logits is None else logits
        self._sampler.begin_guidance(SignedSamplerStages(tuple(stages), self._capture_feedback))
        return logits

    def _sampler_guidance(self) -> tuple[PackedSamplerGuidanceBatch, ...]:
        execution = self.deployment.execution
        return execution.sampler_guidance(self.driver.context(), torch.cuda.current_stream(execution.device))

    @torch.inference_mode(False)
    @torch.enable_grad()
    def complete_sampler(self) -> None:
        if self.driver.step is None:
            return
        feedback = None if self._sampler is None else self._sampler.finish_guidance()
        self.driver.sample(feedback, torch.cuda.current_stream(self.deployment.execution.device))

    @torch.inference_mode(False)
    @torch.enable_grad()
    def commit_token(self) -> None:
        self.driver.commit()

    @torch.inference_mode(False)
    def abort_active(self) -> None:
        if self._sampler is not None:
            self._sampler.abort_guidance()
        self._finish_effects()
        self.driver.abort()

    @torch.inference_mode(False)
    def retire_all(self, *, wait: bool = False) -> None:
        if self.completion_rows is not None:
            self.completion_rows.synchronize()
        self.driver.finish_call(wait=wait)
        self.completion_rows = None

    def finish_requests(self) -> None:
        """Require every submitted completion row before releasing the generation's requests."""
        if self.completion_rows is not None:
            self.completion_rows.require_complete()
        self.retire_all()

    @torch.inference_mode(False)
    def close(self) -> None:
        try:
            self.abort_active()
            self.retire_all(wait=True)
            self.driver.release()
            self._close_effects()
        finally:
            self.driver.close()

"""Instance-owned MRv2 request/sampler lifecycle for canonical ESamp."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.backend_attachment import (
    validate_backend_attachment_physical,
)
from tllm.canonical.workloads.esamp_request import (
    ESampRequestCudaStepHandle,
)
from tllm.canonical.workloads.esamp_request_types import ESampRequestStepInput
from tllm.canonical_runtime.deployment import CanonicalEsampVllmDeployment
from tllm.canonical_vllm.hidden_capture import (
    Vllm027EagerPackedHiddenCapture,
)
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
from tllm.contracts.backend_attachment import AttachmentCleanupStage
from tllm.contracts.request_identity import RequestIdentity

from vllm import SamplingParams
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.sampler import Sampler


@dataclass(frozen=True, slots=True)
class _Idle:
    pass


@dataclass(frozen=True, slots=True)
class _Capturing:
    rows: Mrv2PackedStepRows


@dataclass(frozen=True, slots=True)
class _Predicted:
    handle: ESampRequestCudaStepHandle


@dataclass(frozen=True, slots=True)
class _Sampled:
    handle: ESampRequestCudaStepHandle


_StepState = _Idle | _Capturing | _Predicted | _Sampled


class CanonicalEsampVllmSession:
    """Own one runner's admitted requests, hooks, sampler, and signed steps."""

    __slots__ = (
        "_capture",
        "_deployment",
        "_identities",
        "_next_prompt_index",
        "_next_step_id",
        "_row_codec",
        "_sampler",
        "_state",
    )

    def __init__(
        self,
        deployment: CanonicalEsampVllmDeployment,
        model: Qwen2ForCausalLM,
        upstream_sampler: Sampler,
    ) -> None:
        if (
            type(deployment) is not CanonicalEsampVllmDeployment
            or type(model) is not Qwen2ForCausalLM
            or type(upstream_sampler) is not Sampler
        ):
            raise TypeError("canonical ESamp session requires exact loaded objects")
        validate_backend_attachment_physical(
            deployment.attachment.inputs,
            deployment.attachment.decision,
            deployment.attachment.physical,
        )
        self._deployment = deployment
        self._row_codec = Vllm027Mrv2PackedRowCodec(
            plan_epoch=deployment.attachment.physical.plan_epoch,
        )
        self._capture = Vllm027EagerPackedHiddenCapture(
            model=model,
            source=deployment.source,
            target=deployment.target,
            plan_epoch=deployment.attachment.physical.plan_epoch,
            max_packed_rows=deployment.max_packed_rows,
            hidden_size=deployment.hidden_size,
        )
        self._sampler = CanonicalVllm027PostFilterSampler(
            upstream_sampler,
            eager=True,
        )
        self._identities: dict[str, RequestIdentity] = {}
        self._next_prompt_index = 0
        self._next_step_id = 0
        self._state: _StepState = _Idle()

    @property
    def sampler(self) -> CanonicalVllm027PostFilterSampler:
        return self._sampler

    def before_execute(self, scheduler_output: SchedulerOutput) -> None:
        """Apply request finish/admission before vLLM mutates its active batch."""

        if type(scheduler_output) is not SchedulerOutput:
            raise TypeError("canonical session requires concrete SchedulerOutput")
        match self._state:
            case _Idle():
                pass
            case _Capturing() | _Predicted() | _Sampled():
                raise RuntimeError("prior canonical runner step is still active")
        execution = self._deployment.execution
        execution.collect_completed()

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
            if type(request.req_id) is not str or not request.req_id:
                raise RuntimeError("vLLM scheduled an invalid request id")
            sampling_params = request.sampling_params
            if (
                type(sampling_params) is not SamplingParams
                or sampling_params.n != 1
            ):
                raise RuntimeError("canonical vLLM requires sampling n=1")
            if request.req_id in self._identities:
                continue
            identity = RequestIdentity(
                request.req_id,
                self._next_prompt_index + len(pending),
                0,
            )
            pending.append(identity)
        if pending:
            stream = torch.cuda.current_stream(self._deployment.execution.device)
            execution.admit(tuple(pending), stream)
            for identity in pending:
                self._identities[identity.request_id] = identity
            self._next_prompt_index += len(pending)

    def prepare_input(self, input_batch: InputBatch) -> None:
        """Open hidden capture from one exact Rust-expanded MRv2 row batch."""

        match self._state:
            case _Idle():
                pass
            case _Capturing() | _Predicted() | _Sampled():
                raise RuntimeError("canonical input preparation overlaps another step")
        request_ids = Mrv2RequestIdBatch(
            tuple(
                Mrv2RequestIdBinding(
                    raw_request_id,
                    self._identities[raw_request_id],
                )
                for raw_request_id in input_batch.req_ids
            )
        )
        rows = self._row_codec.encode(
            input_batch,
            request_ids,
            step_id=self._next_step_id,
        )
        self._capture.begin(rows.phase_rows)
        self._state = _Capturing(rows)
        self._next_step_id += 1

    def complete_forward(self) -> None:
        """Publish captured hidden state and run the signed compute anchor."""

        match self._state:
            case _Idle():
                return
            case _Capturing(rows=rows):
                pass
            case _Predicted() | _Sampled():
                raise RuntimeError("canonical forward was already completed")
        hidden = self._capture.publish()
        if not rows.phase_rows.decode_rows:
            self._capture.finish()
            self._state = _Idle()
            return
        execution = self._deployment.execution
        stream = torch.cuda.current_stream(execution.device)
        try:
            handle = execution.start_step(
                ESampRequestStepInput(
                    rows.phase_rows,
                    rows.sampler_rows,
                    hidden,
                ),
                stream,
            )
            execution.execute_anchor(handle, self._deployment.compute_anchor_id)
        except Exception:
            self._capture.finish()
            raise
        self._capture.finish()
        self._state = _Predicted(handle)

    def begin_sampler(self, model: Qwen2ForCausalLM) -> bool:
        """Order guidance onto the sampler stream and project it through LM head."""

        match self._state:
            case _Idle():
                return False
            case _Predicted(handle=handle):
                pass
            case _Capturing() | _Sampled():
                raise RuntimeError("canonical sampler began in the wrong step phase")
        stream = torch.cuda.current_stream(self._deployment.execution.device)
        guidance = self._deployment.execution.sampler_guidance(handle, stream)
        pred_dense_logits = model.compute_logits(
            guidance.pred_hidden.to(dtype=self._deployment.model_dtype)
        )
        if type(pred_dense_logits) is not torch.Tensor:
            raise RuntimeError("Qwen2 LM head did not produce dense logits")
        self._sampler.begin_guidance(
            SignedPostFilterGuidance(
                guidance,
                self._deployment.sampler_beta,
                pred_dense_logits,
            )
        )
        return True

    def complete_sampler(self, active: bool) -> None:
        """Close the exact draw and publish its sampler completion fence."""

        if type(active) is not bool:
            raise TypeError("canonical sampler activity must be explicit")
        if not active:
            match self._state:
                case _Idle():
                    return
                case _Capturing() | _Predicted() | _Sampled():
                    raise RuntimeError("canonical sampler skipped an active step")
        match self._state:
            case _Predicted(handle=handle):
                pass
            case _Idle() | _Capturing() | _Sampled():
                raise RuntimeError("canonical sampler completed in the wrong phase")
        self._sampler.finish_guidance()
        execution = self._deployment.execution
        execution.execute_anchor(handle, self._deployment.sampler_anchor_id)
        stream = torch.cuda.current_stream(execution.device)
        execution.complete_sampler(handle, stream)
        self._state = _Sampled(handle)

    def commit_token(self) -> None:
        """Run signed loss/transition only after vLLM finishes token processing."""

        match self._state:
            case _Idle():
                return
            case _Sampled(handle=handle):
                pass
            case _Capturing() | _Predicted():
                raise RuntimeError("canonical token commit precedes sampler completion")
        execution = self._deployment.execution
        try:
            execution.execute_anchor(handle, self._deployment.transition_anchor_id)
        except Exception:
            self._state = _Idle()
            raise
        execution.finish_step(handle)
        execution.collect_completed()
        self._state = _Idle()

    def abort_active(self) -> None:
        """Cancel one failed runner step without publishing a state transition."""

        self._sampler.abort_guidance()
        self._capture.abort_if_active()
        match self._state:
            case _Idle() | _Capturing():
                pass
            case _Predicted(handle=handle) | _Sampled(handle=handle):
                self._deployment.execution.cancel_step(handle)
        self._state = _Idle()

    def close(self) -> None:
        """Remove hooks and retire all exact request state owned by this runner."""

        self.abort_active()
        self.retire_all()
        self._capture.close()

    def retire_all(self) -> None:
        """Retire a completed synchronous generation without parsing vLLM ids."""

        match self._state:
            case _Idle():
                pass
            case _Capturing() | _Predicted() | _Sampled():
                raise RuntimeError("canonical requests cannot retire during a step")
        identities = tuple(
            self._identities[request_id]
            for request_id in sorted(self._identities)
        )
        if identities:
            self._retire(identities)
        self._identities.clear()
        self._deployment.execution.collect_completed()

    def _retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        match self._deployment.cleanup_stage:
            case AttachmentCleanupStage.REQUEST_FINISH:
                self._deployment.execution.retire(identities)


__all__ = ["CanonicalEsampVllmSession"]

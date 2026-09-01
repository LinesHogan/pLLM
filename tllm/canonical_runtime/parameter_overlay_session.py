"""Instance-owned MRv2 lifecycle for signed parameter overlays."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.parameter_overlay_attachment import (
    validate_parameter_overlay_attachment_reconstruction,
)
from tllm.canonical.workloads.inplace_ttt import (
    CudaInPlaceTTTStepBatchInput,
    InPlaceTTTCudaStepHandle,
    InPlaceTTTOperandPair,
    InPlaceTTTPhase,
    InPlaceTTTStepInput,
    InPlaceTTTStepBatch,
)
from tllm.canonical_runtime.parameter_overlay_deployment import (
    CanonicalParameterOverlayDeployment,
)
from tllm.canonical.parameter_overlay_plan import (
    ParameterOverlayRuntimePlan,
)
from tllm.canonical_vllm.dense_overlay import VllmDenseAddAdapter
from tllm.canonical_vllm.packed_rows import (
    Mrv2RequestIdBatch,
    Mrv2RequestIdBinding,
    Vllm027Mrv2PackedRowCodec,
)
from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.parameter_overlay import (
    ParameterOverlayRowAssignment,
    ParameterOverlayStepPlan,
)
from tllm.contracts.parameter_overlay_attachment import (
    ParameterOverlayStage,
    ParameterOverlayTaskRole,
)
from tllm.contracts.request_identity import RequestIdentity

from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(frozen=True, slots=True)
class _Idle:
    pass


@dataclass(frozen=True, slots=True)
class _Prepared:
    handle: InPlaceTTTCudaStepHandle


@dataclass(frozen=True, slots=True)
class _Transitioned:
    handle: InPlaceTTTCudaStepHandle


@dataclass(frozen=True, slots=True)
class _Sampled:
    handle: InPlaceTTTCudaStepHandle


_StepState = _Idle | _Prepared | _Transitioned | _Sampled


class CanonicalParameterOverlaySession:
    """Bind exact request roots to one mechanical dense-add adapter."""

    __slots__ = (
        "_adapter",
        "_deployment",
        "_identities",
        "_next_prompt_index",
        "_next_step_id",
        "_row_codec",
        "_runtime_plan",
        "_state",
    )

    def __init__(
        self,
        deployment: CanonicalParameterOverlayDeployment,
        model: Qwen2ForCausalLM,
    ) -> None:
        self._deployment = deployment
        validate_parameter_overlay_attachment_reconstruction(
            deployment.attachment.inputs,
            deployment.attachment.decision,
            deployment.attachment.physical,
            deployment.attachment.inputs,
            deployment.attachment.physical,
        )
        self._runtime_plan = ParameterOverlayRuntimePlan(
            deployment.attachment.physical
        )
        self._row_codec = Vllm027Mrv2PackedRowCodec(
            plan_epoch=deployment.attachment.physical.plan_epoch,
        )
        self._adapter = VllmDenseAddAdapter(
            model,
            self._runtime_plan.targets,
        )
        self._identities: dict[str, RequestIdentity] = {}
        self._next_prompt_index = 0
        self._next_step_id = 0
        self._state: _StepState = _Idle()

    @property
    def apply_count(self) -> int:
        return self._adapter.apply_count

    @property
    def updated_apply_count(self) -> int:
        return self._adapter.updated_apply_count

    def before_execute(self, scheduler_output: SchedulerOutput) -> None:
        match self._state:
            case _Idle():
                pass
            case _Prepared() | _Transitioned() | _Sampled():
                raise RuntimeError("prior parameter-overlay step is active")
        finished = tuple(
            self._identities[request_id]
            for request_id in sorted(scheduler_output.finished_req_ids)
        )
        if finished:
            self._deployment.execution.retire(
                finished,
                self._runtime_plan.stage(
                    ParameterOverlayStage.FINISH
                ).anchor_id,
            )
            for identity in finished:
                self._identities.pop(identity.request_id)
        pending: list[RequestIdentity] = []
        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id in self._identities:
                continue
            sampling_params = request.sampling_params
            if sampling_params is None or sampling_params.n != 1:
                raise RuntimeError("canonical parameter overlay requires sampling n=1")
            pending.append(
                RequestIdentity(
                    request.req_id,
                    self._next_prompt_index + len(pending),
                    0,
                )
            )
        if pending:
            stream = torch.cuda.current_stream(self._deployment.specialization.device)
            self._deployment.execution.admit(tuple(pending), stream)
            for identity in pending:
                self._identities[identity.request_id] = identity
            self._next_prompt_index += len(pending)

    def prepare_input(self, input_batch: InputBatch) -> None:
        match self._state:
            case _Idle():
                pass
            case _Prepared() | _Transitioned() | _Sampled():
                raise RuntimeError("parameter-overlay input overlaps another step")
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
        identities = tuple(binding.identity for binding in request_ids.bindings)
        stream = torch.cuda.current_stream(self._deployment.specialization.device)
        handle = self._deployment.execution.begin_batch(
            identities,
            stream,
            self._runtime_plan.bindings,
            self._runtime_plan.stage(ParameterOverlayStage.PIN).anchor_id,
        )
        self._state = _Prepared(handle)
        ref_by_identity = dict(zip(handle.identities, handle.refs, strict=True))
        assignments: list[ParameterOverlayRowAssignment] = []
        for target in self._runtime_plan.targets:
            for identity in handle.identities:
                selected_rows = tuple(
                    row
                    for row in rows.phase_rows.rows
                    if row.identity == identity
                )
                phases = frozenset(row.phase for row in selected_rows)
                if len(phases) != 1:
                    raise RuntimeError(
                        "one request cannot mix operand phases in one model step"
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
            self._adapter.prepare(
                ParameterOverlayStepPlan(
                    self._deployment.execution.plan.plan_epoch,
                    self._runtime_plan.stage(
                        ParameterOverlayStage.APPLY
                    ).anchor_id,
                    self._runtime_plan.stage(
                        ParameterOverlayStage.CAPTURE
                    ).anchor_id,
                    rows.phase_rows.total_rows,
                    canonical_assignments,
                ),
                handle.deltas,
            )
        except Exception:
            self._deployment.execution.abort_step(handle)
            self._state = _Idle()
            raise
        self._next_step_id += 1

    def complete_forward(self) -> None:
        match self._state:
            case _Prepared(handle=handle):
                pass
            case _Idle():
                return
            case _Transitioned() | _Sampled():
                raise RuntimeError("parameter-overlay forward already completed")
        captures = self._adapter.take_captures()
        actual_capture_keys = frozenset(
            (capture.identity, capture.target) for capture in captures
        )
        expected_capture_keys = frozenset(
            (identity, target)
            for identity in handle.identities
            for target in self._runtime_plan.targets
        )
        if actual_capture_keys != expected_capture_keys:
            raise RuntimeError("vLLM did not capture every signed overlay target")
        captures_by_identity = {
            identity: tuple(
                sorted(
                    (
                        capture
                        for capture in captures
                        if capture.identity == identity
                    ),
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
        stream = torch.cuda.current_stream(self._deployment.specialization.device)
        try:
            self._deployment.execution.submit_captured(
                handle,
                CudaInPlaceTTTStepBatchInput.assert_produced_on(
                    batch,
                    stream,
                ),
                self._runtime_plan.task(
                    ParameterOverlayTaskRole.TRANSITION
                ),
                self._runtime_plan.task(ParameterOverlayTaskRole.PUBLISH),
            )
        except Exception:
            self._adapter.clear()
            self._state = _Idle()
            raise
        self._adapter.clear()
        self._state = _Transitioned(handle)

    def complete_sampler(self) -> None:
        match self._state:
            case _Transitioned(handle=handle):
                pass
            case _Idle() | _Prepared() | _Sampled():
                raise RuntimeError("parameter-overlay sampler completed out of order")
        stream = torch.cuda.current_stream(self._deployment.specialization.device)
        self._deployment.execution.complete_sampler(
            handle,
            stream,
            self._runtime_plan.stage(
                ParameterOverlayStage.RELEASE
            ).anchor_id,
        )
        self._state = _Sampled(handle)

    def commit_token(self) -> None:
        match self._state:
            case _Sampled(handle=handle):
                pass
            case _Idle():
                return
            case _Prepared() | _Transitioned():
                raise RuntimeError("parameter-overlay token committed before sampling")
        self._deployment.execution.finish_step(handle)
        self._state = _Idle()

    def abort_active(self) -> None:
        self._adapter.clear()
        match self._state:
            case _Idle():
                return
            case _Prepared(handle=handle) | _Transitioned(handle=handle) | _Sampled(
                handle=handle
            ):
                self._deployment.execution.abort_step(handle)
        self._state = _Idle()

    def retire_all(self) -> None:
        match self._state:
            case _Idle():
                pass
            case _Prepared() | _Transitioned() | _Sampled():
                raise RuntimeError("cannot retire an active parameter-overlay step")
        identities = tuple(
            self._identities[request_id]
            for request_id in sorted(self._identities)
        )
        if identities:
            self._deployment.execution.retire(
                identities,
                self._runtime_plan.stage(
                    ParameterOverlayStage.FINISH
                ).anchor_id,
            )
        self._identities.clear()

    def close(self) -> None:
        self.abort_active()
        self.retire_all()
        self._adapter.close()


__all__ = ["CanonicalParameterOverlaySession"]

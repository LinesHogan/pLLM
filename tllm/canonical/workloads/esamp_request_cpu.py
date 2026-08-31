"""Staged CPU lifecycle for the clean-room ESamp request graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.execution import TaskGraphExecution, TaskGraphInvocation
from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
    initialize_low_rank_gated,
)
from tllm.canonical.ptts.dense_page import (
    DensePageAdmission,
    DensePageStateProvider,
    DensePageVersionPin,
)
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.canonical.workloads.esamp_request_cpu_tasks import (
    ESampCpuLossTask,
    ESampCpuPredictTask,
    ESampCpuStepContext,
    ESampCpuTrainTask,
    ESampOptimizationPending,
    ESampOptimized,
    ESampPredicted,
    ESampPredictionPending,
    ESampPublicationPending,
    ESampPublished,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestConfig,
    ESampRequestDiagnostics,
    ESampRequestSpecialization,
    ESampRequestStepInput,
    ESampRequestStepResult,
    ESAMP_PARAMETER_FIELD,
)
from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.request_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
    RequestStateKey,
)
from tllm.contracts.sampler import PackedSamplerGuidanceBatch
from tllm.contracts.semantic_graph import AnchorId, PhysicalPlan
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import StatePhysicalPlan


@dataclass(frozen=True, slots=True)
class _SamplerPending:
    pass


@dataclass(frozen=True, slots=True)
class _SamplerCompleted:
    pass


_SamplerState = _SamplerPending | _SamplerCompleted


@dataclass(frozen=True, slots=True)
class _StepActive:
    pass


@dataclass(frozen=True, slots=True)
class _StepFailed:
    pass


@dataclass(frozen=True, slots=True)
class _StepFinished:
    pass


_StepStatus = _StepActive | _StepFailed | _StepFinished


class ESampRequestStepHandle:
    """Opaque staged step advanced only by backend anchor events."""

    __slots__ = (
        "handle_id",
        "owner",
        "invocation",
        "version_pin",
        "sampler",
        "status",
    )

    def __init__(
        self,
        handle_id: int,
        owner: "ESampRequestExecution",
        invocation: TaskGraphInvocation[ESampCpuStepContext],
        version_pin: DensePageVersionPin,
    ) -> None:
        self.handle_id = handle_id
        self.owner = owner
        self.invocation = invocation
        self.version_pin = version_pin
        self.sampler: _SamplerState = _SamplerPending()
        self.status: _StepStatus = _StepActive()


class ESampRequestExecution:
    """CPU request lifecycle around a generic Rust-planned task executor."""

    __slots__ = (
        "_config",
        "_specialization",
        "_plan",
        "_state_plan",
        "_kernel_layout",
        "_provider",
        "_directory",
        "_runner",
        "_sampler_completion_anchor_id",
        "_generator",
        "_loss_count",
        "_update_count",
        "_completed_sampler_pins",
        "_active_handles",
        "_next_handle_id",
    )

    def __init__(
        self,
        *,
        config: ESampRequestConfig,
        specialization: ESampRequestSpecialization,
        plan: PhysicalPlan,
        state_plan: StatePhysicalPlan,
        kernel_layout: LowRankGatedLayout,
        provider: DensePageStateProvider,
        directory: RequestStateDirectory,
        sampler_completion_anchor_id: AnchorId,
    ) -> None:
        if plan.plan_epoch != state_plan.plan_epoch:
            raise ValueError("schedule and state plans belong to different epochs")
        self._config = config
        self._specialization = specialization
        self._plan = plan
        self._state_plan = state_plan
        self._kernel_layout = kernel_layout
        self._provider = provider
        self._directory = directory
        self._sampler_completion_anchor_id = sampler_completion_anchor_id
        self._runner = TaskGraphExecution(
            plan,
            (
                ESampCpuLossTask(kernel_layout),
                ESampCpuPredictTask(config, kernel_layout, provider),
                ESampCpuTrainTask(config, directory, provider),
            ),
        )
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(specialization.initialization_seed)
        self._loss_count = 0
        self._update_count = 0
        self._completed_sampler_pins = 0
        self._active_handles: dict[int, ESampRequestStepHandle] = {}
        self._next_handle_id = 1

    @property
    def plan(self) -> PhysicalPlan:
        return self._plan

    @property
    def state_plan(self) -> StatePhysicalPlan:
        return self._state_plan

    @property
    def free_state_pages(self) -> int:
        return self._provider.free_pages

    def admit(
        self,
        identities: tuple[RequestIdentity, ...],
    ) -> tuple[StateRef, ...]:
        logical = self._directory.prepare_admission(identities)
        try:
            physical_admissions: list[DensePageAdmission] = []
            for admission in logical.admissions:
                initial_tensors: list[torch.Tensor] = list(
                    self._provider.new_zero_root_tensors()
                )
                initial_tensors[int(ESAMP_PARAMETER_FIELD)] = (
                    initialize_low_rank_gated(
                        self._kernel_layout,
                        dtype=initial_tensors[int(ESAMP_PARAMETER_FIELD)].dtype,
                        device=initial_tensors[int(ESAMP_PARAMETER_FIELD)].device,
                        generator=self._generator,
                    )
                )
                physical_admissions.append(
                    DensePageAdmission(admission.ref, tuple(initial_tensors))
                )
            self._provider.admit_batch(tuple(physical_admissions))
        except Exception:
            self._directory.abort_admission(logical)
            raise
        self._directory.commit_admission(logical)
        return tuple(admission.ref for admission in logical.admissions)

    def start_step(self, step: ESampRequestStepInput) -> ESampRequestStepHandle:
        if (
            step.rows.plan_epoch != self._plan.plan_epoch
            or int(step.source.shape[1]) != self._specialization.hidden_size
            or step.source.dtype is not torch.float32
        ):
            raise ValueError("ESamp request step differs from specialization")
        indexed_rows = IndexedRequestIdentityBatch(
            tuple(
                IndexedRequestIdentityRow(
                    row.packed_row_index,
                    row.identity,
                )
                for row in step.rows.rows
                if row.phase is HiddenPhase.DECODE
            )
        )
        if not indexed_rows.rows:
            raise ValueError("ESamp request step requires decode rows")
        selection = self._directory.select(indexed_rows)
        transition_lease = self._directory.acquire_transition_lease(selection)
        try:
            version_pin = self._provider.pin_exact(selection.refs)
        except Exception:
            self._directory.release_transition_lease(transition_lease)
            raise
        context = ESampCpuStepContext(
            step,
            selection,
            transition_lease,
            ESampPredictionPending(),
            ESampOptimizationPending(),
            ESampPublicationPending(),
        )
        handle = ESampRequestStepHandle(
            self._next_handle_id,
            self,
            self._runner.start(context),
            version_pin,
        )
        self._next_handle_id += 1
        self._active_handles[handle.handle_id] = handle
        return handle

    def execute_anchor(
        self,
        handle: ESampRequestStepHandle,
        anchor_id: AnchorId,
    ) -> None:
        self._require_active_handle(handle)
        if int(anchor_id) > int(self._sampler_completion_anchor_id):
            match handle.sampler:
                case _SamplerPending():
                    raise RuntimeError(
                        "ESamp later anchors require sampler completion"
                    )
                case _SamplerCompleted():
                    pass
        try:
            self._runner.execute_anchor(handle.invocation, anchor_id)
        except Exception:
            self._fail_handle(handle)
            raise

    def sampler_guidance(
        self,
        handle: ESampRequestStepHandle,
    ) -> PackedSamplerGuidanceBatch:
        self._require_active_handle(handle)
        match handle.invocation.context.prediction:
            case ESampPredicted(guidance=guidance):
                return guidance
            case _:
                raise RuntimeError("ESamp prediction anchor has not executed")

    def complete_sampler(self, handle: ESampRequestStepHandle) -> None:
        self._require_active_handle(handle)
        if (
            handle.invocation.last_anchor_id
            < int(self._sampler_completion_anchor_id)
        ):
            raise RuntimeError(
                "ESamp sampler cannot complete before its signed anchor"
            )
        match handle.sampler:
            case _SamplerPending():
                handle.sampler = _SamplerCompleted()
                self._completed_sampler_pins += 1
            case _SamplerCompleted():
                raise RuntimeError("ESamp sampler pin is already completed")

    def finish_step(
        self,
        handle: ESampRequestStepHandle,
    ) -> ESampRequestStepResult:
        self._require_active_handle(handle)
        match handle.sampler:
            case _SamplerCompleted():
                pass
            case _SamplerPending():
                raise RuntimeError("ESamp sampler pin must complete before step finish")
        context = self._runner.finish(handle.invocation)
        match context.prediction, context.optimization, context.publication:
            case (
                ESampPredicted() as predicted,
                ESampOptimized() as optimized,
                ESampPublished() as published,
            ):
                pass
            case _:
                raise RuntimeError("ESamp physical plan did not produce all outputs")
        result = ESampRequestStepResult(
            context.step.rows.step_id,
            predicted.guidance,
            self._config.intervention_beta,
            optimized.losses,
            published.before_refs,
            published.after_refs,
        )
        self._loss_count += len(result.losses)
        self._update_count += len(result.after_refs)
        self._provider.release_pin(handle.version_pin)
        self._directory.release_transition_lease(context.transition_lease)
        self._active_handles.pop(handle.handle_id)
        handle.status = _StepFinished()
        return result

    def cancel_step(self, handle: ESampRequestStepHandle) -> None:
        self._require_active_handle(handle)
        self._fail_handle(handle)

    def retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        retirement_keys = {
            RequestStateKey.project(identity, self._config.key_fields)
            for identity in identities
        }
        for handle in tuple(self._active_handles.values()):
            if retirement_keys.intersection(
                handle.invocation.context.selection.keys
            ):
                self.cancel_step(handle)
        logical = self._directory.prepare_retirement(identities)
        try:
            self._provider.retire_batch(logical.refs)
        except Exception:
            self._directory.abort_retirement(logical)
            raise
        self._directory.commit_retirement(logical)

    def snapshot(self, ref: StateRef) -> tuple[torch.Tensor, ...]:
        return self._provider.snapshot(ref)

    def diagnostics(self) -> ESampRequestDiagnostics:
        return ESampRequestDiagnostics(
            self._directory.active_count,
            self._loss_count,
            self._update_count,
            self._completed_sampler_pins,
            self._state_plan,
        )

    def _require_active_handle(self, handle: ESampRequestStepHandle) -> None:
        if handle.owner is not self:
            raise RuntimeError("ESamp step belongs to another execution")
        try:
            issued = self._active_handles[handle.handle_id]
        except KeyError as exc:
            raise RuntimeError("ESamp step is not registered as active") from exc
        if issued is not handle:
            raise RuntimeError("ESamp step differs from the registered handle")
        match handle.status:
            case _StepActive():
                return
            case _StepFailed() | _StepFinished():
                raise RuntimeError("ESamp step is no longer active")

    def _fail_handle(self, handle: ESampRequestStepHandle) -> None:
        self._provider.release_pin(handle.version_pin)
        self._directory.release_transition_lease(
            handle.invocation.context.transition_lease
        )
        self._active_handles.pop(handle.handle_id)
        handle.sampler = _SamplerCompleted()
        handle.status = _StepFailed()


__all__ = ["ESampRequestExecution", "ESampRequestStepHandle"]

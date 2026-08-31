"""Anchor-staged CUDA lifecycle for the clean-room ESamp request graph."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch

from tllm.canonical.cuda_packed_hidden import CudaPackedHiddenSnapshotBatch
from tllm.canonical.execution import TaskGraphExecution, TaskGraphInvocation
from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
    initialize_low_rank_gated,
)
from tllm.canonical.ptts import (
    CudaDensePageAdmission,
    CudaDensePageRelease,
    CudaDensePageStateProvider,
)
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.canonical.workloads.esamp_request_cuda_tasks import (
    ESampCudaLossTask,
    ESampCudaOptimizationPending,
    ESampCudaOptimized,
    ESampCudaPredicted,
    ESampCudaPredictionPending,
    ESampCudaPredictTask,
    ESampCudaPublicationPending,
    ESampCudaPublished,
    ESampCudaStepContext,
    ESampCudaTrainTask,
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
from tllm.contracts.state_representation import StateDeviceKind, StatePhysicalPlan


@dataclass(frozen=True, slots=True)
class _SamplerPending:
    pass


@dataclass(frozen=True, slots=True)
class _SamplerReady:
    stream: torch.cuda.Stream


@dataclass(frozen=True, slots=True)
class _SamplerCompleted:
    stream: torch.cuda.Stream


_SamplerState = _SamplerPending | _SamplerReady | _SamplerCompleted


@dataclass(frozen=True, slots=True)
class ESampRequestCudaStepHandle:
    """Opaque token for one execution-owned CUDA step."""

    handle_id: int

    def __post_init__(self) -> None:
        if type(self.handle_id) is not int or self.handle_id < 1:
            raise ValueError("CUDA ESamp step handle requires an ID")


@dataclass(slots=True)
class _ActiveCudaStep:
    token: ESampRequestCudaStepHandle
    invocation: TaskGraphInvocation[ESampCudaStepContext]
    sampler: _SamplerState


class ESampRequestCudaExecution:
    """CUDA realization; scheduling remains owned by the Rust physical plan."""

    __slots__ = (
        "_config",
        "_specialization",
        "_plan",
        "_state_plan",
        "_kernel_layout",
        "_provider",
        "_directory",
        "_runner",
        "_generator",
        "_sampler_completion_anchor_id",
        "_loss_count",
        "_update_count",
        "_completed_sampler_pins",
        "_active_handles",
        "_next_handle_id",
        "_pending_releases",
        "_lock",
    )

    def __init__(
        self,
        *,
        config: ESampRequestConfig,
        specialization: ESampRequestSpecialization,
        plan: PhysicalPlan,
        state_plan: StatePhysicalPlan,
        kernel_layout: LowRankGatedLayout,
        provider: CudaDensePageStateProvider,
        directory: RequestStateDirectory,
        sampler_completion_anchor_id: AnchorId,
    ) -> None:
        if (
            plan.plan_epoch != state_plan.plan_epoch
            or specialization.device_kind is not StateDeviceKind.CUDA
            or provider.device
            != torch.device("cuda", specialization.device_ordinal)
        ):
            raise ValueError("CUDA ESamp execution plans or placement differ")
        self._config = config
        self._specialization = specialization
        self._plan = plan
        self._state_plan = state_plan
        self._kernel_layout = kernel_layout
        self._provider = provider
        self._directory = directory
        self._runner = TaskGraphExecution(
            plan,
            (
                ESampCudaLossTask(kernel_layout, provider),
                ESampCudaPredictTask(config, kernel_layout, provider),
                ESampCudaTrainTask(config, directory, provider),
            ),
        )
        self._generator = torch.Generator(device=provider.device)
        self._generator.manual_seed(specialization.initialization_seed)
        self._sampler_completion_anchor_id = sampler_completion_anchor_id
        self._loss_count = 0
        self._update_count = 0
        self._completed_sampler_pins = 0
        self._active_handles: dict[int, _ActiveCudaStep] = {}
        self._next_handle_id = 1
        self._pending_releases: list[CudaDensePageRelease] = []
        self._lock = RLock()

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
        stream: torch.cuda.Stream,
    ) -> tuple[StateRef, ...]:
        with self._lock:
            return self._admit(identities, stream)

    def _admit(
        self,
        identities: tuple[RequestIdentity, ...],
        stream: torch.cuda.Stream,
    ) -> tuple[StateRef, ...]:
        self._require_stream(stream)
        logical = self._directory.prepare_admission(identities)
        try:
            admissions: list[CudaDensePageAdmission] = []
            for admission in logical.admissions:
                initial = self._provider.new_zero_root(stream)
                with torch.cuda.stream(stream):
                    parameters = initialize_low_rank_gated(
                        self._kernel_layout,
                        dtype=initial.tensors[int(ESAMP_PARAMETER_FIELD)].dtype,
                        device=self._provider.device,
                        generator=self._generator,
                    )
                    initial.tensors[int(ESAMP_PARAMETER_FIELD)].copy_(parameters)
                admissions.append(CudaDensePageAdmission(admission.ref, initial))
            prepared = self._provider.prepare_admission_batch(tuple(admissions))
            self._provider.submit_admission_batch(prepared, stream)
        except Exception:
            self._directory.abort_admission(logical)
            raise
        self._directory.commit_admission(logical)
        return tuple(admission.ref for admission in logical.admissions)

    def start_step(
        self,
        step: ESampRequestStepInput,
        stream: torch.cuda.Stream,
    ) -> ESampRequestCudaStepHandle:
        with self._lock:
            return self._start_step(step, stream)

    def _start_step(
        self,
        step: ESampRequestStepInput,
        stream: torch.cuda.Stream,
    ) -> ESampRequestCudaStepHandle:
        self._require_stream(stream)
        if type(step.hidden) is not CudaPackedHiddenSnapshotBatch:
            raise TypeError("CUDA ESamp requires stream-owned hidden snapshots")
        step.hidden.order_on(stream)
        hidden = step.hidden.batch
        bindings = self._specialization.hidden_bindings
        actual_resources = tuple(
            snapshot.resource for snapshot in hidden.snapshots
        )
        if actual_resources != bindings.resources:
            raise ValueError(
                "CUDA ESamp snapshots differ from signed resource bindings"
            )
        source = hidden.hidden_for(bindings.source)
        target = hidden.hidden_for(bindings.target)
        if (
            step.rows.plan_epoch != self._plan.plan_epoch
            or int(source.shape[1]) != self._specialization.hidden_size
            or source.dtype is not torch.float32
            or source.device != self._provider.device
            or target.shape != source.shape
        ):
            raise ValueError("CUDA ESamp step differs from specialization")
        indexed_rows = IndexedRequestIdentityBatch(
            tuple(
                IndexedRequestIdentityRow(row.packed_row_index, row.identity)
                for row in step.rows.rows
                if row.phase is HiddenPhase.DECODE
            )
        )
        if not indexed_rows.rows:
            raise ValueError("CUDA ESamp step requires decode rows")
        selection = self._directory.select(indexed_rows)
        transition_lease = self._directory.acquire_transition_lease(selection)
        try:
            state_pin = self._provider.pin_exact(selection.refs, stream)
        except Exception:
            self._directory.release_transition_lease(transition_lease)
            raise
        context = ESampCudaStepContext(
            step,
            source,
            target,
            selection,
            transition_lease,
            state_pin,
            stream,
            ESampCudaPredictionPending(),
            ESampCudaOptimizationPending(),
            ESampCudaPublicationPending(),
        )
        handle = ESampRequestCudaStepHandle(self._next_handle_id)
        active = _ActiveCudaStep(
            handle,
            self._runner.start(context),
            _SamplerPending(),
        )
        self._next_handle_id += 1
        self._active_handles[handle.handle_id] = active
        return handle

    def execute_anchor(
        self,
        handle: ESampRequestCudaStepHandle,
        anchor_id: AnchorId,
    ) -> None:
        with self._lock:
            self._execute_anchor(handle, anchor_id)

    def _execute_anchor(
        self,
        handle: ESampRequestCudaStepHandle,
        anchor_id: AnchorId,
    ) -> None:
        active = self._active_step(handle)
        if int(anchor_id) > int(self._sampler_completion_anchor_id):
            match active.sampler:
                case _SamplerCompleted():
                    pass
                case _SamplerPending() | _SamplerReady():
                    raise RuntimeError(
                        "CUDA ESamp later anchors require sampler completion"
                    )
        try:
            self._runner.execute_anchor(active.invocation, anchor_id)
        except Exception:
            self._fail_step(active)
            raise

    def sampler_guidance(
        self,
        handle: ESampRequestCudaStepHandle,
        stream: torch.cuda.Stream,
    ) -> PackedSamplerGuidanceBatch:
        with self._lock:
            return self._sampler_guidance(handle, stream)

    def _sampler_guidance(
        self,
        handle: ESampRequestCudaStepHandle,
        stream: torch.cuda.Stream,
    ) -> PackedSamplerGuidanceBatch:
        active = self._active_step(handle)
        self._require_stream(stream)
        match active.invocation.context.prediction:
            case ESampCudaPredicted() as predicted:
                pass
            case _:
                raise RuntimeError("CUDA ESamp prediction anchor has not executed")
        match active.sampler:
            case _SamplerPending():
                predicted.readiness.order_on(stream)
                active.sampler = _SamplerReady(stream)
            case _SamplerReady(stream=issued) if (
                issued.cuda_stream == stream.cuda_stream
            ):
                pass
            case _SamplerReady() | _SamplerCompleted():
                raise RuntimeError("CUDA ESamp sampler uses another stream or phase")
        return predicted.guidance

    def complete_sampler(
        self,
        handle: ESampRequestCudaStepHandle,
        stream: torch.cuda.Stream,
    ) -> None:
        with self._lock:
            self._complete_sampler(handle, stream)

    def _complete_sampler(
        self,
        handle: ESampRequestCudaStepHandle,
        stream: torch.cuda.Stream,
    ) -> None:
        active = self._active_step(handle)
        self._require_stream(stream)
        if (
            active.invocation.last_anchor_id
            < int(self._sampler_completion_anchor_id)
        ):
            raise RuntimeError(
                "CUDA ESamp sampler cannot complete before its signed anchor"
            )
        match active.sampler:
            case _SamplerReady(stream=issued) if (
                issued.cuda_stream == stream.cuda_stream
            ):
                completion = torch.cuda.Event(blocking=False)
                completion.record(stream)
                active.invocation.context.stream.wait_event(completion)
                active.sampler = _SamplerCompleted(stream)
                self._completed_sampler_pins += 1
            case _SamplerPending():
                raise RuntimeError("CUDA ESamp guidance was not acquired")
            case _SamplerReady() | _SamplerCompleted():
                raise RuntimeError("CUDA ESamp sampler completion is invalid")

    def finish_step(
        self,
        handle: ESampRequestCudaStepHandle,
    ) -> ESampRequestStepResult:
        with self._lock:
            return self._finish_step(handle)

    def _finish_step(
        self,
        handle: ESampRequestCudaStepHandle,
    ) -> ESampRequestStepResult:
        active = self._active_step(handle)
        match active.sampler:
            case _SamplerCompleted():
                pass
            case _SamplerPending() | _SamplerReady():
                raise RuntimeError("CUDA ESamp sampler must complete before finish")
        context = self._runner.finish(active.invocation)
        match context.prediction, context.optimization, context.publication:
            case (
                ESampCudaPredicted() as predicted,
                ESampCudaOptimized() as optimized,
                ESampCudaPublished() as published,
            ):
                pass
            case _:
                raise RuntimeError("CUDA ESamp plan did not produce all outputs")
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
        self._pending_releases.append(
            self._provider.release_pin(context.state_pin, context.stream)
        )
        self._directory.release_transition_lease(context.transition_lease)
        self._active_handles.pop(handle.handle_id)
        return result

    def collect_completed(self) -> None:
        with self._lock:
            self._collect_completed()

    def _collect_completed(self) -> None:
        pending: list[CudaDensePageRelease] = []
        for release in self._pending_releases:
            if not self._provider.release_complete(release):
                pending.append(release)
        self._pending_releases = pending
        self._provider.collect_completed()

    def cancel_step(self, handle: ESampRequestCudaStepHandle) -> None:
        with self._lock:
            self._fail_step(self._active_step(handle))

    def retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        with self._lock:
            self._retire(identities)

    def _retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        retirement_keys = {
            RequestStateKey.project(identity, self._config.key_fields)
            for identity in identities
        }
        for active in tuple(self._active_handles.values()):
            if retirement_keys.intersection(
                active.invocation.context.selection.keys
            ):
                self._fail_step(active)
        for release in self._pending_releases:
            self._provider.wait_release(release)
        self._pending_releases.clear()
        self._provider.collect_completed()
        logical = self._directory.prepare_retirement(identities)
        try:
            self._provider.retire_batch(logical.refs)
        except Exception:
            self._directory.abort_retirement(logical)
            raise
        self._directory.commit_retirement(logical)

    def diagnostics(self) -> ESampRequestDiagnostics:
        with self._lock:
            return ESampRequestDiagnostics(
                self._directory.active_count,
                self._loss_count,
                self._update_count,
                self._completed_sampler_pins,
                len(self._pending_releases),
                self._state_plan,
            )

    def _require_stream(self, stream: torch.cuda.Stream) -> None:
        if (
            type(stream) is not torch.cuda.Stream
            or stream.device != self._provider.device
        ):
            raise RuntimeError("CUDA ESamp stream uses another device")

    def _active_step(
        self,
        handle: ESampRequestCudaStepHandle,
    ) -> _ActiveCudaStep:
        try:
            active = self._active_handles[handle.handle_id]
        except KeyError as exc:
            raise RuntimeError("CUDA ESamp step is not active") from exc
        if active.token is not handle:
            raise RuntimeError("CUDA ESamp step differs from its issued handle")
        return active

    def _fail_step(self, active: _ActiveCudaStep) -> None:
        context = active.invocation.context
        match context.optimization, context.publication:
            case (
                ESampCudaOptimized(gradients=gradients),
                ESampCudaPublicationPending(),
            ):
                self._provider.discard_gradients(gradients)
            case _:
                pass
        self._directory.release_transition_lease(context.transition_lease)
        self._pending_releases.append(
            self._provider.release_pin(context.state_pin, context.stream)
        )
        self._active_handles.pop(active.token.handle_id)


__all__ = [
    "ESampRequestCudaExecution",
    "ESampRequestCudaStepHandle",
]

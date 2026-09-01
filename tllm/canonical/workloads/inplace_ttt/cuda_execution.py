"""Anchor-staged CUDA lifecycle for canonical dense fast weights."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.execution import TaskGraphExecution
from tllm.canonical.ptts.cuda_publication import (
    CudaRequestStatePublicationCoordinator,
)
from tllm.canonical.ptts.dense_page_cuda import CudaDensePageStateProvider
from tllm.canonical.ptts.dense_page_cuda_types import (
    CudaDensePageAdmission,
    CudaDensePageRelease,
    CudaDensePageVersionPin,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateSelection,
    RequestStateTransitionLease,
)
from tllm.canonical.scheduler import validate_physical_plan, validate_schedule
from tllm.canonical.workloads.inplace_ttt.cuda_tasks import (
    CudaInPlaceTTTCommittedTransition,
    CudaInPlaceTTTPreparedTransition,
    CudaInPlaceTTTPublicationPending,
    CudaInPlaceTTTPublishTask,
    CudaInPlaceTTTPublished,
    CudaInPlaceTTTStepContext,
    CudaInPlaceTTTTransitionPending,
    CudaInPlaceTTTTransitionTask,
    CudaInPlaceTTTValidatedTransition,
    INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION,
    INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION,
)
from tllm.canonical.workloads.inplace_ttt.layout import InPlaceTTTStateLayout
from tllm.canonical.workloads.inplace_ttt.types import (
    CudaInPlaceTTTStepBatchInput,
    InPlaceTTTCleanupReceipt,
    InPlaceTTTConfig,
    InPlaceTTTCudaStepHandle,
    InPlaceTTTDiagnostics,
    InPlaceTTTSpecialization,
    InPlaceTTTStepBatchResult,
    validate_inplace_ttt_step,
)
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.parameter_overlay import ParameterOverlayPinnedTensor
from tllm.contracts.parameter_overlay_attachment import (
    ParameterOverlayAttachmentBinding,
    ParameterOverlayTaskBinding,
    ParameterOverlayTaskRole,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.request_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
)
from tllm.contracts.semantic_graph import AnchorId, ImplementationId, PhysicalPlan
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import StateDeviceKind, StatePhysicalPlan


@dataclass(frozen=True, slots=True)
class _AwaitingCapture:
    pass


@dataclass(frozen=True, slots=True)
class _Executed:
    result: InPlaceTTTStepBatchResult


@dataclass(frozen=True, slots=True)
class _SamplerReleased:
    result: InPlaceTTTStepBatchResult
    release: CudaDensePageRelease


_CudaStepState = _AwaitingCapture | _Executed | _SamplerReleased


@dataclass(frozen=True, slots=True)
class _Live:
    pass


@dataclass(frozen=True, slots=True)
class _Failed:
    pass


_ExecutionState = _Live | _Failed


@dataclass(slots=True)
class _ActiveCudaStep:
    token: InPlaceTTTCudaStepHandle
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    state_pin: CudaDensePageVersionPin
    stream: torch.cuda.Stream
    last_attachment_anchor_id: int
    state: _CudaStepState


class InPlaceTTTCudaExecution:
    """Keep exact StateRef pins alive through model and sampler completion."""

    __slots__ = (
        "_active",
        "_config",
        "_directory",
        "_execution_state",
        "_layout",
        "_lock",
        "_next_handle_id",
        "_pending_releases",
        "_plan_bundle",
        "_provider",
        "_publication",
        "_runner",
        "_specialization",
        "_trained_rows",
        "_update_count",
    )

    def __init__(
        self,
        config: InPlaceTTTConfig,
        specialization: InPlaceTTTSpecialization,
        plan_bundle: CanonicalPlanBundle,
        layout: InPlaceTTTStateLayout,
        provider: CudaDensePageStateProvider,
        directory: RequestStateDirectory,
        publication: CudaRequestStatePublicationCoordinator,
    ) -> None:
        if (
            specialization.device_kind is not StateDeviceKind.CUDA
            or provider.device != specialization.device
            or plan_bundle.semantic_graph.execution_parameters_digest
            != specialization.execution_parameters_digest(config)
            or plan_bundle.schedule_physical.plan_epoch
            != plan_bundle.state_physical.plan_epoch
        ):
            raise ValueError("CUDA In-Place-TTT execution artifacts disagree")
        validate_schedule(
            plan_bundle.semantic_graph,
            plan_bundle.timeline,
            plan_bundle.schedule_decision,
        )
        validate_physical_plan(
            plan_bundle.semantic_graph,
            plan_bundle.timeline,
            plan_bundle.schedule_decision,
            plan_bundle.schedule_physical,
        )
        implementations_by_task = {
            int(task.task_id): task.implementation_id
            for task in plan_bundle.schedule_physical.tasks
        }
        if tuple(sorted(implementations_by_task)) != (0, 1):
            raise ValueError("CUDA In-Place-TTT plan lacks dense task IDs")
        implementation_ids: tuple[ImplementationId, ImplementationId] = (
            implementations_by_task[0],
            implementations_by_task[1],
        )
        if implementation_ids != (
            INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION,
            INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION,
        ):
            raise ValueError("CUDA implementations differ from placement")
        self._config = config
        self._specialization = specialization
        self._plan_bundle = plan_bundle
        self._layout = layout
        self._provider = provider
        self._directory = directory
        self._publication = publication
        self._runner = TaskGraphExecution(
            plan_bundle.schedule_physical,
            (
                CudaInPlaceTTTPublishTask(layout, publication),
                CudaInPlaceTTTTransitionTask(
                    config,
                    layout,
                    directory,
                    provider,
                    publication,
                ),
            ),
        )
        self._active: dict[int, _ActiveCudaStep] = {}
        self._execution_state: _ExecutionState = _Live()
        self._next_handle_id = 1
        self._pending_releases: list[CudaDensePageRelease] = []
        self._update_count = 0
        self._trained_rows = 0
        self._lock = RLock()

    @property
    def plan(self) -> PhysicalPlan:
        return self._plan_bundle.schedule_physical

    @property
    def state_plan(self) -> StatePhysicalPlan:
        return self._plan_bundle.state_physical

    @property
    def config(self) -> InPlaceTTTConfig:
        return self._config

    @property
    def specialization(self) -> InPlaceTTTSpecialization:
        return self._specialization

    @property
    def execution_parameters_digest(self) -> str:
        return self._specialization.execution_parameters_digest(self._config)

    def admit(
        self,
        identities: tuple[RequestIdentity, ...],
        stream: torch.cuda.Stream,
    ) -> tuple[StateRef, ...]:
        with self._lock:
            self._require_live()
            logical = self._directory.prepare_admission(identities)
            try:
                prepared = self._provider.prepare_admission_batch(
                    tuple(
                        CudaDensePageAdmission(
                            admission.ref,
                            self._provider.new_zero_root(stream),
                        )
                        for admission in logical.admissions
                    )
                )
            except Exception:
                self._directory.abort_admission(logical)
                raise
            try:
                self._provider.submit_admission_batch(prepared, stream)
            except Exception:
                self._directory.abort_admission(logical)
                raise
            self._directory.commit_admission(logical)
            return tuple(admission.ref for admission in logical.admissions)

    def begin_batch(
        self,
        identities: tuple[RequestIdentity, ...],
        stream: torch.cuda.Stream,
        bindings: tuple[ParameterOverlayAttachmentBinding, ...],
        pin_anchor_id: AnchorId,
    ) -> InPlaceTTTCudaStepHandle:
        with self._lock:
            self._require_live()
            self._collect_completed()
            if (
                tuple(binding.target for binding in bindings)
                != self._config.targets
                or any(
                    binding.fields
                    != (self._layout.fields_for(binding.target).delta_weight,)
                    for binding in bindings
                )
            ):
                raise ValueError("signed overlay bindings differ from state layout")
            selection = self._directory.select(
                IndexedRequestIdentityBatch(
                    tuple(
                        IndexedRequestIdentityRow(index, identity)
                        for index, identity in enumerate(identities)
                    )
                )
            )
            transition_lease = self._directory.acquire_transition_lease(selection)
            try:
                state_pin = self._provider.pin_exact(selection.refs, stream)
            except Exception:
                self._directory.release_transition_lease(transition_lease)
                raise
            try:
                field_rows = tuple(
                    (
                        binding,
                        self._provider.pinned_field_rows(
                            state_pin,
                            binding.fields[0],
                            stream,
                        ),
                    )
                    for binding in bindings
                )
                deltas = tuple(
                    ParameterOverlayPinnedTensor(binding.target, ref, rows[index])
                    for binding, rows in field_rows
                    for index, ref in enumerate(selection.refs)
                )
            except Exception:
                release = self._provider.release_pin(state_pin, stream)
                self._pending_releases.append(release)
                self._directory.release_transition_lease(transition_lease)
                raise
            token = InPlaceTTTCudaStepHandle(
                self._next_handle_id,
                identities,
                selection.refs,
                deltas,
            )
            self._next_handle_id += 1
            self._active[token.handle_id] = _ActiveCudaStep(
                token,
                selection,
                transition_lease,
                state_pin,
                stream,
                int(pin_anchor_id),
                _AwaitingCapture(),
            )
            return token

    def submit_captured(
        self,
        handle: InPlaceTTTCudaStepHandle,
        captured: CudaInPlaceTTTStepBatchInput,
        transition_task: ParameterOverlayTaskBinding,
        publish_task: ParameterOverlayTaskBinding,
    ) -> None:
        with self._lock:
            self._require_live()
            active = self._active_step(handle)
            match active.state:
                case _AwaitingCapture():
                    pass
                case _Executed() | _SamplerReleased():
                    raise RuntimeError("CUDA operands were already submitted")
            try:
                if (
                    transition_task.role
                    is not ParameterOverlayTaskRole.TRANSITION
                    or publish_task.role is not ParameterOverlayTaskRole.PUBLISH
                    or transition_task.anchor_id > publish_task.anchor_id
                    or int(transition_task.anchor_id)
                    < active.last_attachment_anchor_id
                ):
                    raise ValueError("signed overlay task order is invalid")
                physical_by_id = {
                    task.task_id: task for task in self.plan.tasks
                }
                for binding in (transition_task, publish_task):
                    physical = physical_by_id[binding.task_id]
                    if (
                        physical.implementation_id != binding.implementation_id
                        or physical.anchor_id != binding.anchor_id
                    ):
                        raise ValueError(
                            "signed overlay task differs from physical plan"
                        )
                batch = captured.batch
                if tuple(entry.identity for entry in batch.entries) != handle.identities:
                    raise ValueError(
                        "CUDA captured operands belong to another request"
                    )
                for entry in batch.entries:
                    validate_inplace_ttt_step(entry, self._specialization)
                captured.order_on(active.stream)
            except Exception:
                self._fail_step(active)
                raise
            context = CudaInPlaceTTTStepContext(
                batch,
                active.selection,
                active.transition_lease,
                active.state_pin,
                active.stream,
                CudaInPlaceTTTTransitionPending(),
                CudaInPlaceTTTPublicationPending(),
            )
            invocation = self._runner.start(context)
            try:
                self._runner.execute_anchor(
                    invocation,
                    transition_task.anchor_id,
                )
                self._runner.execute_anchor(
                    invocation,
                    publish_task.anchor_id,
                )
                completed = self._runner.finish(invocation)
            except Exception:
                match context.transition, context.publication:
                    case CudaInPlaceTTTPreparedTransition(
                        logical=logical,
                        physical=physical,
                    ), CudaInPlaceTTTPublicationPending():
                        self._publication.abort_reservation(logical, physical)
                        self._fail_step(active)
                    case (
                        CudaInPlaceTTTValidatedTransition()
                        | CudaInPlaceTTTCommittedTransition(),
                        _,
                    ):
                        self._execution_state = _Failed()
                        self._active.clear()
                    case _:
                        self._fail_step(active)
                raise
            match completed.publication:
                case CudaInPlaceTTTPublished(result=result):
                    active.last_attachment_anchor_id = int(
                        publish_task.anchor_id
                    )
                    active.state = _Executed(result)
                case CudaInPlaceTTTPublicationPending():
                    self._fail_step(active)
                    raise RuntimeError("signed CUDA graph did not publish")

    def complete_sampler(
        self,
        handle: InPlaceTTTCudaStepHandle,
        sampler_stream: torch.cuda.Stream,
        release_anchor_id: AnchorId,
    ) -> None:
        with self._lock:
            self._require_live()
            active = self._active_step(handle)
            match active.state:
                case _Executed(result=result):
                    pass
                case _AwaitingCapture():
                    raise RuntimeError("sampler completed before CUDA transition")
                case _SamplerReleased():
                    raise RuntimeError("sampler completion is already recorded")
            if int(release_anchor_id) < active.last_attachment_anchor_id:
                raise ValueError("signed release anchor precedes publication")
            if sampler_stream.cuda_stream != active.stream.cuda_stream:
                CudaStreamReadiness(sampler_stream).order_on(active.stream)
            release = self._provider.release_pin(active.state_pin, active.stream)
            active.last_attachment_anchor_id = int(release_anchor_id)
            active.state = _SamplerReleased(result, release)

    def finish_step(
        self,
        handle: InPlaceTTTCudaStepHandle,
    ) -> InPlaceTTTStepBatchResult:
        with self._lock:
            self._require_live()
            active = self._active_step(handle)
            match active.state:
                case _SamplerReleased(result=result, release=release):
                    pass
                case _AwaitingCapture() | _Executed():
                    raise RuntimeError("CUDA step requires sampler completion")
            self._directory.release_transition_lease(active.transition_lease)
            self._pending_releases.append(release)
            self._active.pop(handle.handle_id)
            self._update_count += sum(
                target.update_count
                for entry in result.entries
                for target in entry.targets
            )
            self._trained_rows += sum(
                target.trained_rows
                for entry in result.entries
                for target in entry.targets
            )
            return result

    def abort_step(self, handle: InPlaceTTTCudaStepHandle) -> None:
        with self._lock:
            self._require_live()
            self._fail_step(self._active_step(handle))

    def retire(
        self,
        identities: tuple[RequestIdentity, ...],
        finish_anchor_id: AnchorId,
    ) -> InPlaceTTTCleanupReceipt:
        with self._lock:
            self._require_live()
            if int(finish_anchor_id) < 0:
                raise ValueError("signed finish anchor is invalid")
            self._collect_completed()
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
            return InPlaceTTTCleanupReceipt(logical.refs)

    def snapshot(
        self,
        ref: StateRef,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, ...]:
        with self._lock:
            self._require_live()
            pin = self._provider.pin_exact((ref,), stream)
            snapshot = self._provider.snapshot(pin, ref, stream)
            release = self._provider.release_pin(pin, stream)
            self._provider.wait_release(release)
            return snapshot

    def diagnostics(self) -> InPlaceTTTDiagnostics:
        with self._lock:
            self._require_live()
            self._collect_completed()
            return InPlaceTTTDiagnostics(
                self._directory.active_count,
                self._update_count,
                self._trained_rows,
                self.state_plan,
                self.plan.fallback_count,
                self._provider.pending_pin_releases,
            )

    def _active_step(
        self,
        handle: InPlaceTTTCudaStepHandle,
    ) -> _ActiveCudaStep:
        try:
            active = self._active[handle.handle_id]
        except KeyError as exc:
            raise RuntimeError("CUDA step handle is not active") from exc
        if active.token is not handle:
            raise RuntimeError("CUDA step differs from its issued handle")
        return active

    def _fail_step(self, active: _ActiveCudaStep) -> None:
        match active.state:
            case _SamplerReleased(release=release):
                pass
            case _AwaitingCapture() | _Executed():
                release = self._provider.release_pin(
                    active.state_pin,
                    active.stream,
                )
        self._directory.release_transition_lease(active.transition_lease)
        self._pending_releases.append(release)
        self._active.pop(active.token.handle_id)

    def _collect_completed(self) -> None:
        retained: list[CudaDensePageRelease] = []
        for release in self._pending_releases:
            if not self._provider.release_complete(release):
                retained.append(release)
        self._pending_releases = retained
        self._provider.collect_completed()

    def _require_live(self) -> None:
        match self._execution_state:
            case _Live():
                pass
            case _Failed():
                raise RuntimeError("CUDA In-Place-TTT execution is fail-stopped")


__all__ = ["InPlaceTTTCudaExecution"]

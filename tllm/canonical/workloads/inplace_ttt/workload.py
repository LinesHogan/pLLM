"""Clean-room In-Place-TTT declaration and signed CPU specialization."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.execution import TaskGraphExecution
from tllm.canonical.ptts.dense_page import (
    DensePageAdmission,
    DensePageStateProvider,
)
from tllm.canonical.ptts.dense_page_cuda_types import (
    CudaDensePageAdmission,
    CudaDensePageRelease,
    CudaDensePageVersionPin,
)
from tllm.canonical.ptts.dense_page_cuda import CudaDensePageStateProvider
from tllm.canonical.ptts.cuda_publication import (
    CudaRequestStatePublicationCoordinator,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateSelection,
    RequestStateTransitionLease,
)
from tllm.canonical.ptts.publication import RequestStatePublicationCoordinator
from tllm.canonical.scheduler import (
    compile_schedule,
    lower_physical_plan,
    validate_physical_plan,
    validate_schedule,
)
from tllm.canonical.state_representation import (
    compile_state_representation,
    lower_state_physical_plan,
    validate_state_representation,
    validate_state_representation_reconstruction,
)
from tllm.canonical.tensor_layout import (
    intern_tensor_layouts,
    validate_tensor_layouts,
)
from tllm.canonical.workloads.base import CanonicalWorkload
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
from tllm.canonical.workloads.inplace_ttt.layout import (
    InPlaceTTTStateLayout,
    build_inplace_ttt_state_layout,
)
from tllm.canonical.workloads.inplace_ttt.tasks import (
    InPlaceTTTPreparedTransition,
    InPlaceTTTPublicationPending,
    InPlaceTTTPublishTask,
    InPlaceTTTPublished,
    InPlaceTTTStepContext,
    InPlaceTTTTransitionPending,
    InPlaceTTTTransitionTask,
    INPLACE_TTT_PUBLISH_IMPLEMENTATION,
    INPLACE_TTT_TRANSITION_IMPLEMENTATION,
)
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTCompileOptions,
    CudaInPlaceTTTStepBatchInput,
    InPlaceTTTCleanupReceipt,
    InPlaceTTTConfig,
    InPlaceTTTDiagnostics,
    InPlaceTTTCudaStepHandle,
    InPlaceTTTSpecialization,
    InPlaceTTTStepBatchResult,
    InPlaceTTTStepInput,
    InPlaceTTTStepResult,
    validate_inplace_ttt_step,
)
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.backend_attachment import AttachmentTaskBinding, AttachmentTaskRole
from tllm.contracts.parameter_overlay import (
    ParameterOverlayFieldBinding,
    ParameterOverlayPinnedTensor,
)
from tllm.contracts.request_identity import RequestIdentity, RequestKeyField
from tllm.contracts.request_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
)
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    ExecutionKind,
    HostPayloadSchemaId,
    HostServiceId,
    HostServiceInputSource,
    HostServicePort,
    SemanticEdge,
    SemanticEffect,
    ImplementationId,
    PhysicalPlan,
    SemanticGraph,
    SemanticInput,
    SemanticOutput,
    SemanticOutputKind,
    SemanticState,
    SemanticTask,
    StateEffectKind,
    StateResource,
    StateScope,
    TaskImplementationSpec,
    TaskResultInputSource,
    TaskSymbol,
    TimingConstraint,
    WorkloadId,
)
from tllm.contracts.state_representation import (
    StateDeviceKind,
    StatePhysicalPlan,
    StateLayoutRequirement,
    StateRepresentationRequest,
)
from tllm.contracts.state_ref import StateRef
from tllm.contracts.tensor_layout import (
    StateTensorLayout,
    TensorLayoutDecision,
    TensorLayoutRequest,
)

_FAST_WEIGHT_STATE = StateResource("dense_fast_weights")
_PUBLISH_TASK = TaskSymbol("publish")
_TRANSITION_TASK = TaskSymbol("transition")


def _validate_dense_overlay_execution_artifacts(
    *,
    config: InPlaceTTTConfig,
    specialization: InPlaceTTTSpecialization,
    plan_bundle: CanonicalPlanBundle,
    expected_implementations: tuple[ImplementationId, ImplementationId],
    error_prefix: str,
) -> None:
    if (
        plan_bundle.semantic_graph.execution_parameters_digest
        != specialization.execution_parameters_digest(config)
        or plan_bundle.schedule_physical.plan_epoch
        != plan_bundle.state_physical.plan_epoch
    ):
        raise ValueError(f"{error_prefix} execution artifacts disagree")
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
    implementations = {
        int(task.task_id): task.implementation_id
        for task in plan_bundle.schedule_physical.tasks
    }
    if tuple(sorted(implementations)) != (0, 1):
        raise ValueError(f"{error_prefix} plan does not cover dense task IDs")
    implementation_ids: tuple[ImplementationId, ImplementationId] = (
        implementations[0],
        implementations[1],
    )
    if implementation_ids != expected_implementations:
        raise ValueError(f"{error_prefix} implementations differ from placement")


class CanonicalDenseOverlayExecution:
    """In-Place-TTT dense overlay lifecycle over one Rust-planned graph."""

    __slots__ = (
        "_config",
        "_directory",
        "_layout",
        "_lock",
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
        provider: DensePageStateProvider,
        directory: RequestStateDirectory,
        publication: RequestStatePublicationCoordinator,
    ) -> None:
        _validate_dense_overlay_execution_artifacts(
            config=config,
            specialization=specialization,
            plan_bundle=plan_bundle,
            expected_implementations=(
                INPLACE_TTT_PUBLISH_IMPLEMENTATION,
                INPLACE_TTT_TRANSITION_IMPLEMENTATION,
            ),
            error_prefix="dense overlay",
        )
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
                InPlaceTTTPublishTask(layout, publication),
                InPlaceTTTTransitionTask(
                    config,
                    layout,
                    directory,
                    provider,
                    publication,
                ),
            ),
        )
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

    def admit(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            logical = self._directory.prepare_admission(identities)
            try:
                self._provider.admit_batch(
                    tuple(
                        DensePageAdmission(
                            admission.ref,
                            self._provider.new_zero_root_tensors(),
                        )
                        for admission in logical.admissions
                    )
                )
            except Exception:
                self._directory.abort_admission(logical)
                raise
            self._directory.commit_admission(logical)
            return tuple(admission.ref for admission in logical.admissions)

    def step(self, step: InPlaceTTTStepInput) -> InPlaceTTTStepResult:
        with self._lock:
            validate_inplace_ttt_step(step, self._specialization)
            selection = self._directory.select(
                IndexedRequestIdentityBatch(
                    (IndexedRequestIdentityRow(0, step.identity),)
                )
            )
            transition_lease = self._directory.acquire_transition_lease(selection)
            try:
                pin = self._provider.pin_exact(selection.refs)
            except Exception:
                self._directory.release_transition_lease(transition_lease)
                raise
            context = InPlaceTTTStepContext(
                step,
                selection,
                transition_lease,
                InPlaceTTTTransitionPending(),
                InPlaceTTTPublicationPending(),
            )
            try:
                completed = self._runner.run(context)
            except Exception:
                match context.transition, context.publication:
                    case InPlaceTTTPreparedTransition(
                        logical=logical,
                        physical=physical,
                    ), InPlaceTTTPublicationPending():
                        self._publication.abort(logical, physical)
                    case _:
                        pass
                self._provider.release_pin(pin)
                self._directory.release_transition_lease(transition_lease)
                raise
            match completed.publication:
                case InPlaceTTTPublished(result=result):
                    pass
                case InPlaceTTTPublicationPending():
                    self._provider.release_pin(pin)
                    self._directory.release_transition_lease(transition_lease)
                    raise RuntimeError("signed dense overlay graph did not publish")
            self._provider.release_pin(pin)
            self._directory.release_transition_lease(transition_lease)
            self._update_count += sum(target.update_count for target in result.targets)
            self._trained_rows += sum(target.trained_rows for target in result.targets)
            return result

    def retire(
        self,
        identities: tuple[RequestIdentity, ...],
    ) -> InPlaceTTTCleanupReceipt:
        with self._lock:
            logical = self._directory.prepare_retirement(identities)
            try:
                self._provider.retire_batch(logical.refs)
            except Exception:
                self._directory.abort_retirement(logical)
                raise
            self._directory.commit_retirement(logical)
            return InPlaceTTTCleanupReceipt(logical.refs)

    def snapshot(self, ref: StateRef) -> tuple[torch.Tensor, ...]:
        with self._lock:
            return self._provider.snapshot(ref)

    def diagnostics(self) -> InPlaceTTTDiagnostics:
        with self._lock:
            return InPlaceTTTDiagnostics(
                self._directory.active_count,
                self._update_count,
                self._trained_rows,
                self.state_plan,
                self.plan.fallback_count,
                0,
            )


@dataclass(frozen=True, slots=True)
class _DenseCudaAwaitingCapture:
    pass


@dataclass(frozen=True, slots=True)
class _DenseCudaExecuted:
    result: InPlaceTTTStepBatchResult


@dataclass(frozen=True, slots=True)
class _DenseCudaSamplerReleased:
    result: InPlaceTTTStepBatchResult
    release: CudaDensePageRelease


_DenseCudaStepState = (
    _DenseCudaAwaitingCapture | _DenseCudaExecuted | _DenseCudaSamplerReleased
)


@dataclass(frozen=True, slots=True)
class _DenseCudaLive:
    pass


@dataclass(frozen=True, slots=True)
class _DenseCudaFailed:
    pass


_DenseCudaExecutionState = _DenseCudaLive | _DenseCudaFailed


@dataclass(slots=True)
class _ActiveDenseCudaStep:
    token: InPlaceTTTCudaStepHandle
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    state_pin: CudaDensePageVersionPin
    stream: torch.cuda.Stream
    last_attachment_anchor_id: int
    state: _DenseCudaStepState


class CanonicalDenseOverlayCudaExecution:
    """CUDA In-Place-TTT dense-overlay lifecycle."""

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
        if specialization.device_kind is not StateDeviceKind.CUDA or (
            provider.device != specialization.device
        ):
            raise ValueError("CUDA dense overlay execution artifacts disagree")
        _validate_dense_overlay_execution_artifacts(
            config=config,
            specialization=specialization,
            plan_bundle=plan_bundle,
            expected_implementations=(
                INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION,
                INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION,
            ),
            error_prefix="CUDA dense overlay",
        )
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
        self._active: dict[int, _ActiveDenseCudaStep] = {}
        self._execution_state: _DenseCudaExecutionState = _DenseCudaLive()
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

    @property
    def device(self) -> torch.device:
        return self._specialization.device

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
        bindings: tuple[ParameterOverlayFieldBinding, ...],
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
                    ParameterOverlayPinnedTensor(
                        binding.target,
                        ref,
                        rows[index],
                    )
                    for binding, rows in field_rows
                    for index, ref in enumerate(selection.refs)
                )
            except Exception:
                self._pending_releases.append(
                    self._provider.release_pin(state_pin, stream)
                )
                self._directory.release_transition_lease(transition_lease)
                raise
            token = InPlaceTTTCudaStepHandle(
                self._next_handle_id,
                identities,
                selection.refs,
                deltas,
            )
            self._next_handle_id += 1
            self._active[token.handle_id] = _ActiveDenseCudaStep(
                token,
                selection,
                transition_lease,
                state_pin,
                stream,
                int(pin_anchor_id),
                _DenseCudaAwaitingCapture(),
            )
            return token

    def submit_captured(
        self,
        handle: InPlaceTTTCudaStepHandle,
        captured: CudaInPlaceTTTStepBatchInput,
        transition_task: AttachmentTaskBinding,
        publish_task: AttachmentTaskBinding,
    ) -> None:
        with self._lock:
            self._require_live()
            active = self._active_step(handle)
            match active.state:
                case _DenseCudaAwaitingCapture():
                    pass
                case _DenseCudaExecuted() | _DenseCudaSamplerReleased():
                    raise RuntimeError("CUDA dense overlay operands were already submitted")
            try:
                if (
                    transition_task.role is not AttachmentTaskRole.TRANSITION
                    or publish_task.role is not AttachmentTaskRole.PUBLISH
                    or transition_task.anchor_id > publish_task.anchor_id
                    or int(transition_task.anchor_id) < active.last_attachment_anchor_id
                ):
                    raise ValueError("signed overlay task order is invalid")
                physical_by_id = {task.task_id: task for task in self.plan.tasks}
                for binding in (transition_task, publish_task):
                    physical = physical_by_id[binding.task_id]
                    if (
                        physical.implementation_id != binding.implementation_id
                        or physical.anchor_id != binding.anchor_id
                    ):
                        raise ValueError("signed overlay task differs from physical plan")
                batch = captured.batch
                if tuple(entry.identity for entry in batch.entries) != handle.identities:
                    raise ValueError("CUDA dense overlay operands belong elsewhere")
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
                self._runner.execute_anchor(invocation, transition_task.anchor_id)
                self._runner.execute_anchor(invocation, publish_task.anchor_id)
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
                        self._execution_state = _DenseCudaFailed()
                        self._active.clear()
                    case _:
                        self._fail_step(active)
                raise
            match completed.publication:
                case CudaInPlaceTTTPublished(result=result):
                    active.last_attachment_anchor_id = int(publish_task.anchor_id)
                    active.state = _DenseCudaExecuted(result)
                case CudaInPlaceTTTPublicationPending():
                    self._fail_step(active)
                    raise RuntimeError("signed CUDA dense overlay graph did not publish")

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
                case _DenseCudaExecuted(result=result):
                    pass
                case _DenseCudaAwaitingCapture():
                    raise RuntimeError("sampler completed before CUDA dense overlay")
                case _DenseCudaSamplerReleased():
                    raise RuntimeError("sampler completion is already recorded")
            if int(release_anchor_id) < active.last_attachment_anchor_id:
                raise ValueError("signed release anchor precedes publication")
            if sampler_stream.cuda_stream != active.stream.cuda_stream:
                CudaStreamReadiness(sampler_stream).order_on(active.stream)
            release = self._provider.release_pin(active.state_pin, active.stream)
            active.last_attachment_anchor_id = int(release_anchor_id)
            active.state = _DenseCudaSamplerReleased(result, release)

    def finish_step(
        self,
        handle: InPlaceTTTCudaStepHandle,
    ) -> InPlaceTTTStepBatchResult:
        with self._lock:
            self._require_live()
            active = self._active_step(handle)
            match active.state:
                case _DenseCudaSamplerReleased(result=result, release=release):
                    pass
                case _DenseCudaAwaitingCapture() | _DenseCudaExecuted():
                    raise RuntimeError("CUDA dense overlay requires sampler completion")
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
    ) -> _ActiveDenseCudaStep:
        try:
            active = self._active[handle.handle_id]
        except KeyError as exc:
            raise RuntimeError("CUDA dense overlay step handle is not active") from exc
        if active.token is not handle:
            raise RuntimeError("CUDA dense overlay step differs from its handle")
        return active

    def _fail_step(self, active: _ActiveDenseCudaStep) -> None:
        match active.state:
            case _DenseCudaSamplerReleased(release=release):
                pass
            case _DenseCudaAwaitingCapture() | _DenseCudaExecuted():
                release = self._provider.release_pin(active.state_pin, active.stream)
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
            case _DenseCudaLive():
                pass
            case _DenseCudaFailed():
                raise RuntimeError("CUDA dense overlay execution is fail-stopped")

@dataclass(frozen=True, slots=True)
class InPlaceTTTCompilation:
    execution: CanonicalDenseOverlayExecution | CanonicalDenseOverlayCudaExecution
    plan_bundle: CanonicalPlanBundle
    state_layout: InPlaceTTTStateLayout
    tensor_layout_request: TensorLayoutRequest
    tensor_layout_decision: TensorLayoutDecision
    execution_parameters_digest: str

    def __post_init__(self) -> None:
        if (
            self.execution.plan is not self.plan_bundle.schedule_physical
            or self.execution.state_plan is not self.plan_bundle.state_physical
            or self.execution.execution_parameters_digest
            != self.execution_parameters_digest
            or self.plan_bundle.semantic_graph.execution_parameters_digest
            != self.execution_parameters_digest
        ):
            raise ValueError("In-Place-TTT compilation is not execution-aligned")


class InPlaceTTTWorkload(CanonicalWorkload):
    """Declare dense-fast-weight math/effects without physical policy."""

    __slots__ = ("config",)

    def __init__(self, config: InPlaceTTTConfig) -> None:
        self.config = config

    def semantic_graph(self, execution_parameters_digest: str) -> SemanticGraph:
        return SemanticGraph(
            workload_id=WorkloadId("inplace_ttt.dense_fast_weight"),
            tasks=(
                SemanticTask(
                    symbol=_PUBLISH_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("state.publish"),
                        AnchorSymbol("next.prepare"),
                    ),
                    inputs=(
                        SemanticInput(
                            "transition",
                            TaskResultInputSource(_TRANSITION_TASK, "transition"),
                        ),
                    ),
                    outputs=(
                        SemanticOutput(
                            "step_result",
                            SemanticOutputKind.HOST,
                            HostPayloadSchemaId(
                                "canonical.inplace_ttt.step_result@1"
                            ),
                        ),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            (
                                "state/fenced_reclaim@1",
                                "transition/full_root@1",
                            ),
                        ),
                        TaskImplementationSpec(
                            INPLACE_TTT_PUBLISH_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            ("transition/full_root@1",),
                        ),
                    ),
                    effects=(
                        SemanticEffect(_FAST_WEIGHT_STATE, StateEffectKind.PUBLISH),
                    ),
                ),
                SemanticTask(
                    symbol=_TRANSITION_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("model.execute"),
                        AnchorSymbol("state.publish"),
                    ),
                    inputs=(
                        SemanticInput(
                            "operands",
                            HostServiceInputSource(
                                HostServiceId("parameter.operands"),
                                HostServicePort("dense_pairs"),
                                HostPayloadSchemaId(
                                    "canonical.dense_operand_pairs@1"
                                ),
                            ),
                        ),
                    ),
                    outputs=(
                        SemanticOutput(
                            "transition",
                            SemanticOutputKind.TEMPORARY,
                        ),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            INPLACE_TTT_TRANSITION_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            (
                                "compute/cpu@1",
                                "transition/full_root@1",
                            ),
                        ),
                        TaskImplementationSpec(
                            INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            (
                                "compute/cuda@1",
                                "state/fenced_reclaim@1",
                                "transition/full_root@1",
                            ),
                        ),
                    ),
                    effects=(
                        SemanticEffect(
                            _FAST_WEIGHT_STATE,
                            StateEffectKind.TRANSITION_MAYBE,
                        ),
                    ),
                ),
            ),
            execution_parameters_digest=execution_parameters_digest,
            states=(
                SemanticState(
                    _FAST_WEIGHT_STATE,
                    StateScope.REQUEST,
                    (
                        RequestKeyField.REQUEST_ID,
                        RequestKeyField.PROMPT_IDX,
                        RequestKeyField.SAMPLE_IDX,
                    ),
                ),
            ),
            edges=(SemanticEdge(_TRANSITION_TASK, _PUBLISH_TASK),),
        )

    def compile(
        self,
        specialization: InPlaceTTTSpecialization,
        options: InPlaceTTTCompileOptions,
    ) -> InPlaceTTTCompilation:
        execution_parameters_digest = specialization.execution_parameters_digest(
            self.config
        )
        graph = self.semantic_graph(execution_parameters_digest)
        schedule = compile_schedule(graph, options.timeline)
        physical = lower_physical_plan(
            graph,
            options.timeline,
            schedule,
            plan_epoch=options.plan_epoch,
        )
        state_layout = build_inplace_ttt_state_layout(self.config, specialization)
        layout_request = TensorLayoutRequest(
            (
                StateTensorLayout(
                    str(_FAST_WEIGHT_STATE),
                    state_layout.tensor_layout,
                ),
            )
        )
        layout_decision = intern_tensor_layouts(layout_request)
        validate_tensor_layouts(layout_request, layout_decision)
        (layout_binding,) = layout_decision.bindings
        candidates = tuple(
            provider.materialize_plan(
                resource=str(_FAST_WEIGHT_STATE),
                layout=state_layout.tensor_layout,
                binding=layout_binding,
                max_admitted_lineages=options.max_admitted_lineages,
            )
            for provider in options.providers
        )
        match specialization.device_kind:
            case StateDeviceKind.CPU:
                required_capabilities = (
                    "device/cpu@1",
                    "state/exact_versions@1",
                    "transition/full_root@1",
                )
            case StateDeviceKind.CUDA:
                required_capabilities = (
                    "device/cuda@1",
                    "state/exact_versions@1",
                    "state/fenced_reclaim@1",
                    "transition/full_root@1",
                )
        representation_request = StateRepresentationRequest(
            memory_budget_bytes=options.state_memory_budget_bytes,
            states=(
                StateLayoutRequirement(
                    resource=str(_FAST_WEIGHT_STATE),
                    layout_id=layout_binding.layout_id,
                    layout_namespace=layout_binding.layout_namespace,
                    layout_digest=layout_binding.layout_digest,
                    max_admitted_lineages=options.max_admitted_lineages,
                    required_capabilities=required_capabilities,
                ),
            ),
            candidates=candidates,
        )
        representation = compile_state_representation(representation_request)
        validate_state_representation(representation_request, representation)
        validate_state_representation_reconstruction(
            representation_request,
            representation,
            candidates,
        )
        state_physical = lower_state_physical_plan(
            representation_request,
            representation,
            plan_epoch=options.plan_epoch,
        )
        (selected_provider,) = representation.plans
        if (
            selected_provider.device_kind is not specialization.device_kind
            or selected_provider.device_ordinal != specialization.device_ordinal
        ):
            raise RuntimeError("Rust-selected state placement differs from device")
        directory = RequestStateDirectory(
            layout_binding.layout_id,
            (
                RequestKeyField.REQUEST_ID,
                RequestKeyField.PROMPT_IDX,
                RequestKeyField.SAMPLE_IDX,
            ),
        )
        plan_bundle = CanonicalPlanBundle(
            graph,
            options.timeline,
            schedule,
            physical,
            representation_request,
            representation,
            state_physical,
        )
        match selected_provider.device_kind:
            case StateDeviceKind.CPU:
                cpu_provider = DensePageStateProvider(
                    tensor_layout_request=layout_request,
                    tensor_layout_decision=layout_decision,
                    representation_request=representation_request,
                    representation_decision=representation,
                    physical_plan=state_physical,
                    local_candidates=candidates,
                    resource=str(_FAST_WEIGHT_STATE),
                )
                cpu_publication = RequestStatePublicationCoordinator(
                    directory,
                    cpu_provider,
                )
                execution = (
                    CanonicalDenseOverlayExecution(
                        self.config,
                        specialization,
                        plan_bundle,
                        state_layout,
                        cpu_provider,
                        directory,
                        cpu_publication,
                    )
                )
            case StateDeviceKind.CUDA:
                cuda_provider = CudaDensePageStateProvider(
                    tensor_layout_request=layout_request,
                    tensor_layout_decision=layout_decision,
                    representation_request=representation_request,
                    representation_decision=representation,
                    physical_plan=state_physical,
                    local_candidates=candidates,
                    resource=str(_FAST_WEIGHT_STATE),
                    device_ordinal=selected_provider.device_ordinal,
                )
                cuda_publication = CudaRequestStatePublicationCoordinator(
                    directory,
                    cuda_provider,
                )
                execution = CanonicalDenseOverlayCudaExecution(
                    self.config,
                    specialization,
                    plan_bundle,
                    state_layout,
                    cuda_provider,
                    directory,
                    cuda_publication,
                )
        return InPlaceTTTCompilation(
            execution,
            plan_bundle,
            state_layout,
            layout_request,
            layout_decision,
            execution_parameters_digest,
        )


__all__ = [
    "CanonicalDenseOverlayExecution",
    "CanonicalDenseOverlayCudaExecution",
    "InPlaceTTTCompilation",
    "InPlaceTTTWorkload",
]

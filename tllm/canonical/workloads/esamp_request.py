"""Clean-room ESamp request graph and cold specialization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock

import torch

from tllm.canonical.cuda_packed_hidden import CudaPackedHiddenSnapshotBatch
from tllm.canonical.execution import TaskGraphExecution, TaskGraphInvocation
from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
    initialize_low_rank_gated,
)
from tllm.canonical.ptts import (
    CudaDensePageGradientBatch,
    CudaDensePageStateProvider,
)
from tllm.canonical.ptts.dense_page import (
    DensePageAdmission,
    DensePageStateProvider,
    DensePageVersionPin,
)
from tllm.canonical.ptts.dense_page_cuda_types import (
    CudaDensePageAdmission,
    CudaDensePageRelease,
)
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.canonical.request_sampler_attachment import (
    RequestSamplerAttachmentSpec,
)
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
from tllm.canonical.workloads.esamp_request_task_common import (
    ESampCpuLossTask,
    ESampCpuPredictTask,
    ESampCpuStepContext,
    ESampCpuTrainTask,
    ESampCudaLossTask,
    ESampCudaPredictTask,
    ESampCudaStepContext,
    ESampCudaTrainTask,
    ESampOptimizationPending,
    ESampOptimized,
    ESampPredicted,
    ESampPredictionPending,
    ESampPublicationPending,
    ESampPublished,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions,
    ESampRequestConfig,
    ESampRequestDiagnostics,
    ESampRequestSpecialization,
    ESampRequestStepInput,
    ESampRequestStepResult,
    ESAMP_EXP_AVG_FIELD,
    ESAMP_EXP_AVG_SQ_FIELD,
    ESAMP_GRADIENT_FIELD,
    ESAMP_PARAMETER_FIELD,
    ESAMP_REQUEST_CUDA_LOSS_IMPLEMENTATION,
    ESAMP_REQUEST_CUDA_PREDICT_IMPLEMENTATION,
    ESAMP_REQUEST_CUDA_TRAIN_IMPLEMENTATION,
    ESAMP_REQUEST_LOSS_IMPLEMENTATION,
    ESAMP_REQUEST_PREDICT_IMPLEMENTATION,
    ESAMP_REQUEST_TRAIN_IMPLEMENTATION,
    ESAMP_STEP_FIELD,
)
from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.contracts.packed_hidden import PackedHiddenSnapshotBatch
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.request_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
    RequestStateKey,
)
from tllm.contracts.sampler import PackedSamplerGuidanceBatch
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    ExecutionKind,
    HiddenInputSource,
    PackedRowsInputSource,
    PackedSamplerRowsInputSource,
    SemanticEffect,
    SemanticGraph,
    PhysicalPlan,
    SemanticInput,
    SemanticOutput,
    SemanticOutputKind,
    SemanticState,
    SemanticTask,
    StateEffectKind,
    StateResource,
    StateScope,
    StateSelectionInputSource,
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
    MutationGranularity,
    StateTensorLayout,
    TensorDType,
    TensorFieldSpec,
    TensorLayoutRequest,
    TensorLayoutSpec,
)


_STATE_RESOURCE = StateResource("request_distiller")
_LOSS_TASK = TaskSymbol("loss")
_PREDICT_TASK = TaskSymbol("predict")
_TRAIN_TASK = TaskSymbol("train")


def _decode_indexed_rows(
    step: ESampRequestStepInput,
    *,
    empty_message: str,
) -> IndexedRequestIdentityBatch:
    indexed_rows = IndexedRequestIdentityBatch(
        tuple(
            IndexedRequestIdentityRow(row.packed_row_index, row.identity)
            for row in step.rows.rows
            if row.phase is HiddenPhase.DECODE
        )
    )
    if not indexed_rows.rows:
        raise ValueError(empty_message)
    return indexed_rows


def _require_sampler_anchor_reached(
    *,
    last_anchor_id: int,
    sampler_completion_anchor_id: AnchorId,
    message: str,
) -> None:
    if last_anchor_id < int(sampler_completion_anchor_id):
        raise RuntimeError(message)


def _completed_step_result(
    context: ESampCpuStepContext | ESampCudaStepContext,
    *,
    intervention_beta: float,
    error_message: str,
) -> ESampRequestStepResult:
    match context.prediction, context.optimization, context.publication:
        case (
            ESampPredicted() as predicted,
            ESampOptimized() as optimized,
            ESampPublished() as published,
        ):
            return ESampRequestStepResult(
                context.step.rows.step_id,
                predicted.guidance,
                intervention_beta,
                optimized.losses,
                published.before_refs,
                published.after_refs,
            )
        case _:
            raise RuntimeError(error_message)


@dataclass(frozen=True, slots=True)
class _SamplerPending:
    pass


@dataclass(frozen=True, slots=True)
class _SamplerCompleted:
    pass


_SamplerState = _SamplerPending | _SamplerCompleted


@dataclass(frozen=True, slots=True)
class CanonicalRequestStepHandle:
    """Opaque token for one execution-owned CPU step."""

    handle_id: int

    def __post_init__(self) -> None:
        if type(self.handle_id) is not int or self.handle_id < 1:
            raise ValueError("request-sampler step handle requires an ID")


@dataclass(slots=True)
class _ActiveRequestSamplerStep:
    token: CanonicalRequestStepHandle
    invocation: TaskGraphInvocation[ESampCpuStepContext]
    version_pin: DensePageVersionPin
    sampler: _SamplerState


class CanonicalRequestSamplerExecution:
    """CPU ESamp request lifecycle around its Rust-planned task graph."""

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
        provider: DensePageStateProvider,
        directory: RequestStateDirectory,
        sampler_completion_anchor_id: AnchorId,
    ) -> None:
        if (
            plan.plan_epoch != state_plan.plan_epoch
            or specialization.device_kind is not StateDeviceKind.CPU
            or specialization.device_ordinal != 0
        ):
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
        self._active_handles: dict[int, _ActiveRequestSamplerStep] = {}
        self._next_handle_id = 1
        self._lock = RLock()

    @property
    def plan(self) -> PhysicalPlan:
        return self._plan

    @property
    def state_plan(self) -> StatePhysicalPlan:
        return self._state_plan

    @property
    def config(self) -> ESampRequestConfig:
        return self._config

    @property
    def specialization(self) -> ESampRequestSpecialization:
        return self._specialization

    @property
    def execution_parameters_digest(self) -> str:
        return self._specialization.execution_parameters_digest(self._config)

    @property
    def free_state_pages(self) -> int:
        return self._provider.free_pages

    def admit(
        self,
        identities: tuple[RequestIdentity, ...],
    ) -> tuple[StateRef, ...]:
        with self._lock:
            return self._admit(identities)

    def _admit(
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

    def start_step(self, step: ESampRequestStepInput) -> CanonicalRequestStepHandle:
        with self._lock:
            return self._start_step(step)

    def _start_step(
        self,
        step: ESampRequestStepInput,
    ) -> CanonicalRequestStepHandle:
        if type(step.hidden) is not PackedHiddenSnapshotBatch:
            raise TypeError("request-sampler requires ordinary packed hidden snapshots")
        hidden = step.hidden
        bindings = self._specialization.hidden_bindings
        actual_resources = tuple(
            snapshot.resource for snapshot in hidden.snapshots
        )
        if actual_resources != bindings.resources:
            raise ValueError(
                "request-sampler hidden snapshots differ from signed resource bindings"
            )
        source = hidden.hidden_for(bindings.source)
        target = hidden.hidden_for(bindings.target)
        if (
            step.rows.plan_epoch != self._plan.plan_epoch
            or int(source.shape[1]) != self._specialization.hidden_size
            or source.dtype is not torch.float32
            or source.device.type != "cpu"
            or target.shape != source.shape
        ):
            raise ValueError("request-sampler step differs from specialization")
        indexed_rows = _decode_indexed_rows(
            step,
            empty_message="request-sampler step requires decode rows",
        )
        selection = self._directory.select(indexed_rows)
        transition_lease = self._directory.acquire_transition_lease(selection)
        try:
            version_pin = self._provider.pin_exact(selection.refs)
        except Exception:
            self._directory.release_transition_lease(transition_lease)
            raise
        context = ESampCpuStepContext(
            step,
            source,
            target,
            selection,
            transition_lease,
            ESampPredictionPending(),
            ESampOptimizationPending(),
            ESampPublicationPending(),
        )
        handle = CanonicalRequestStepHandle(self._next_handle_id)
        active = _ActiveRequestSamplerStep(
            handle,
            self._runner.start(context),
            version_pin,
            _SamplerPending(),
        )
        self._next_handle_id += 1
        self._active_handles[handle.handle_id] = active
        return handle

    def execute_anchor(
        self,
        handle: CanonicalRequestStepHandle,
        anchor_id: AnchorId,
    ) -> None:
        with self._lock:
            self._execute_anchor(handle, anchor_id)

    def _execute_anchor(
        self,
        handle: CanonicalRequestStepHandle,
        anchor_id: AnchorId,
    ) -> None:
        active = self._active_step(handle)
        if int(anchor_id) > int(self._sampler_completion_anchor_id):
            match active.sampler:
                case _SamplerPending():
                    raise RuntimeError(
                        "generic later anchors require sampler completion"
                    )
                case _SamplerCompleted():
                    pass
        try:
            self._runner.execute_anchor(active.invocation, anchor_id)
        except Exception:
            self._fail_step(active)
            raise

    def sampler_guidance(
        self,
        handle: CanonicalRequestStepHandle,
    ) -> PackedSamplerGuidanceBatch:
        with self._lock:
            return self._sampler_guidance(handle)

    def _sampler_guidance(
        self,
        handle: CanonicalRequestStepHandle,
    ) -> PackedSamplerGuidanceBatch:
        active = self._active_step(handle)
        match active.invocation.context.prediction:
            case ESampPredicted(guidance=guidance):
                return guidance
            case _:
                raise RuntimeError("request-sampler prediction anchor has not executed")

    def complete_sampler(self, handle: CanonicalRequestStepHandle) -> None:
        with self._lock:
            self._complete_sampler(handle)

    def _complete_sampler(self, handle: CanonicalRequestStepHandle) -> None:
        active = self._active_step(handle)
        _require_sampler_anchor_reached(
            last_anchor_id=active.invocation.last_anchor_id,
            sampler_completion_anchor_id=self._sampler_completion_anchor_id,
            message="request-sampler sampler cannot complete before its signed anchor",
        )
        match active.sampler:
            case _SamplerPending():
                active.sampler = _SamplerCompleted()
                self._completed_sampler_pins += 1
            case _SamplerCompleted():
                raise RuntimeError("request-sampler sampler pin is already completed")

    def finish_step(
        self,
        handle: CanonicalRequestStepHandle,
    ) -> ESampRequestStepResult:
        with self._lock:
            return self._finish_step(handle)

    def _finish_step(
        self,
        handle: CanonicalRequestStepHandle,
    ) -> ESampRequestStepResult:
        active = self._active_step(handle)
        match active.sampler:
            case _SamplerCompleted():
                pass
            case _SamplerPending():
                raise RuntimeError("request-sampler sampler pin must complete before step finish")
        context = self._runner.finish(active.invocation)
        result = _completed_step_result(
            context,
            intervention_beta=self._config.intervention_beta,
            error_message="request-sampler physical plan did not produce all outputs",
        )
        self._loss_count += len(result.losses)
        self._update_count += len(result.after_refs)
        self._provider.release_pin(active.version_pin)
        self._directory.release_transition_lease(context.transition_lease)
        self._active_handles.pop(handle.handle_id)
        return result

    def cancel_step(self, handle: CanonicalRequestStepHandle) -> None:
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
        logical = self._directory.prepare_retirement(identities)
        try:
            self._provider.retire_batch(logical.refs)
        except Exception:
            self._directory.abort_retirement(logical)
            raise
        self._directory.commit_retirement(logical)

    def snapshot(self, ref: StateRef) -> tuple[torch.Tensor, ...]:
        with self._lock:
            return self._provider.snapshot(ref)

    def diagnostics(self) -> ESampRequestDiagnostics:
        with self._lock:
            return ESampRequestDiagnostics(
                self._directory.active_count,
                self._loss_count,
                self._update_count,
                self._completed_sampler_pins,
                0,
                self._state_plan,
            )

    def _active_step(
        self,
        handle: CanonicalRequestStepHandle,
    ) -> _ActiveRequestSamplerStep:
        try:
            active = self._active_handles[handle.handle_id]
        except KeyError as exc:
            raise RuntimeError("request-sampler step is not registered as active") from exc
        if active.token is not handle:
            raise RuntimeError("request-sampler step differs from the registered handle")
        return active

    def _fail_step(self, active: _ActiveRequestSamplerStep) -> None:
        self._provider.release_pin(active.version_pin)
        self._directory.release_transition_lease(
            active.invocation.context.transition_lease
        )
        self._active_handles.pop(active.token.handle_id)


@dataclass(frozen=True, slots=True)
class _CudaSamplerReady:
    stream: torch.cuda.Stream


@dataclass(frozen=True, slots=True)
class _CudaSamplerCompleted:
    stream: torch.cuda.Stream


_CudaSamplerState = _SamplerPending | _CudaSamplerReady | _CudaSamplerCompleted


@dataclass(frozen=True, slots=True)
class CanonicalRequestCudaStepHandle:
    """Opaque token for one execution-owned CUDA request-sampler step."""

    handle_id: int

    def __post_init__(self) -> None:
        if type(self.handle_id) is not int or self.handle_id < 1:
            raise ValueError("CUDA request-sampler step handle requires an ID")


@dataclass(slots=True)
class _ActiveCudaRequestSamplerStep:
    token: CanonicalRequestCudaStepHandle
    invocation: TaskGraphInvocation[ESampCudaStepContext]
    sampler: _CudaSamplerState


class CanonicalRequestSamplerCudaExecution:
    """CUDA ESamp request lifecycle around its Rust-planned task graph."""

    __slots__ = (
        "_active_handles",
        "_completed_sampler_pins",
        "_config",
        "_directory",
        "_generator",
        "_kernel_layout",
        "_lock",
        "_loss_count",
        "_next_handle_id",
        "_pending_releases",
        "_plan",
        "_provider",
        "_runner",
        "_sampler_completion_anchor_id",
        "_specialization",
        "_state_plan",
        "_update_count",
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
            raise ValueError("CUDA request-sampler plans or placement differ")
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
        self._active_handles: dict[int, _ActiveCudaRequestSamplerStep] = {}
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
    def config(self) -> ESampRequestConfig:
        return self._config

    @property
    def specialization(self) -> ESampRequestSpecialization:
        return self._specialization

    @property
    def execution_parameters_digest(self) -> str:
        return self._specialization.execution_parameters_digest(self._config)

    @property
    def device(self) -> torch.device:
        return self._provider.device

    @property
    def free_state_pages(self) -> int:
        return self._provider.free_pages

    @property
    def state_pool_addresses(self) -> tuple[int, ...]:
        return self._provider.pool_addresses

    def require_stable_state_pool(self) -> None:
        self._provider.require_stable()

    def admit(
        self,
        identities: tuple[RequestIdentity, ...],
        stream: torch.cuda.Stream,
    ) -> tuple[StateRef, ...]:
        with self._lock:
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
    ) -> CanonicalRequestCudaStepHandle:
        with self._lock:
            self._require_stream(stream)
            if type(step.hidden) is not CudaPackedHiddenSnapshotBatch:
                raise TypeError("CUDA request-sampler requires CUDA hidden snapshots")
            step.hidden.order_on(stream)
            hidden = step.hidden.batch
            bindings = self._specialization.hidden_bindings
            actual_resources = tuple(snapshot.resource for snapshot in hidden.snapshots)
            if actual_resources != bindings.resources:
                raise ValueError(
                    "CUDA request-sampler snapshots differ from signed bindings"
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
                raise ValueError("CUDA request-sampler step differs from specialization")
            indexed_rows = _decode_indexed_rows(
                step,
                empty_message="CUDA request-sampler step requires decode rows",
            )
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
                ESampPredictionPending(),
                ESampOptimizationPending(),
                ESampPublicationPending(),
                state_pin,
                stream,
            )
            handle = CanonicalRequestCudaStepHandle(self._next_handle_id)
            self._next_handle_id += 1
            self._active_handles[handle.handle_id] = _ActiveCudaRequestSamplerStep(
                handle,
                self._runner.start(context),
                _SamplerPending(),
            )
            return handle

    def execute_anchor(
        self,
        handle: CanonicalRequestCudaStepHandle,
        anchor_id: AnchorId,
    ) -> None:
        with self._lock:
            active = self._active_step(handle)
            if int(anchor_id) > int(self._sampler_completion_anchor_id):
                match active.sampler:
                    case _CudaSamplerCompleted():
                        pass
                    case _SamplerPending() | _CudaSamplerReady():
                        raise RuntimeError(
                            "CUDA request-sampler later anchors require sampler completion"
                        )
            try:
                self._runner.execute_anchor(active.invocation, anchor_id)
            except Exception:
                self._fail_step(active)
                raise

    def sampler_guidance(
        self,
        handle: CanonicalRequestCudaStepHandle,
        stream: torch.cuda.Stream,
    ) -> PackedSamplerGuidanceBatch:
        with self._lock:
            self._require_stream(stream)
            active = self._active_step(handle)
            match active.invocation.context.prediction:
                case ESampPredicted() as predicted:
                    pass
                case _:
                    raise RuntimeError("CUDA request-sampler prediction is absent")
            match active.sampler:
                case _SamplerPending():
                    if predicted.readiness is None:
                        raise RuntimeError("CUDA request-sampler readiness is absent")
                    predicted.readiness.order_on(stream)
                    active.sampler = _CudaSamplerReady(stream)
                case _CudaSamplerReady(stream=issued) if (
                    issued.cuda_stream == stream.cuda_stream
                ):
                    pass
                case _CudaSamplerReady() | _CudaSamplerCompleted():
                    raise RuntimeError("CUDA request-sampler sampler phase is invalid")
            return predicted.guidance

    def complete_sampler(
        self,
        handle: CanonicalRequestCudaStepHandle,
        stream: torch.cuda.Stream,
    ) -> None:
        with self._lock:
            self._require_stream(stream)
            active = self._active_step(handle)
            _require_sampler_anchor_reached(
                last_anchor_id=active.invocation.last_anchor_id,
                sampler_completion_anchor_id=self._sampler_completion_anchor_id,
                message="CUDA request-sampler sampler cannot complete before its anchor",
            )
            match active.sampler:
                case _CudaSamplerReady(stream=issued) if (
                    issued.cuda_stream == stream.cuda_stream
                ):
                    completion = torch.cuda.Event(blocking=False)
                    completion.record(stream)
                    active.invocation.context.stream.wait_event(completion)
                    active.sampler = _CudaSamplerCompleted(stream)
                    self._completed_sampler_pins += 1
                case _SamplerPending():
                    raise RuntimeError("CUDA request-sampler guidance was not acquired")
                case _CudaSamplerReady() | _CudaSamplerCompleted():
                    raise RuntimeError("CUDA request-sampler completion is invalid")

    def finish_step(
        self,
        handle: CanonicalRequestCudaStepHandle,
    ) -> ESampRequestStepResult:
        with self._lock:
            active = self._active_step(handle)
            match active.sampler:
                case _CudaSamplerCompleted():
                    pass
                case _SamplerPending() | _CudaSamplerReady():
                    raise RuntimeError("CUDA request-sampler requires sampler completion")
            context = self._runner.finish(active.invocation)
            result = _completed_step_result(
                context,
                intervention_beta=self._config.intervention_beta,
                error_message="CUDA request-sampler plan did not complete",
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

    def cancel_step(self, handle: CanonicalRequestCudaStepHandle) -> None:
        with self._lock:
            self._fail_step(self._active_step(handle))

    def retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        with self._lock:
            retirement_keys = {
                RequestStateKey.project(identity, self._config.key_fields)
                for identity in identities
            }
            for active in tuple(self._active_handles.values()):
                if retirement_keys.intersection(active.invocation.context.selection.keys):
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

    def _collect_completed(self) -> None:
        pending: list[CudaDensePageRelease] = []
        for release in self._pending_releases:
            if not self._provider.release_complete(release):
                pending.append(release)
        self._pending_releases = pending
        self._provider.collect_completed()

    def _require_stream(self, stream: torch.cuda.Stream) -> None:
        if (
            type(stream) is not torch.cuda.Stream
            or stream.device != self._provider.device
        ):
            raise RuntimeError("CUDA request-sampler stream uses another device")

    def _active_step(
        self,
        handle: CanonicalRequestCudaStepHandle,
    ) -> _ActiveCudaRequestSamplerStep:
        try:
            active = self._active_handles[handle.handle_id]
        except KeyError as exc:
            raise RuntimeError("CUDA request-sampler step is not active") from exc
        if active.token is not handle:
            raise RuntimeError("CUDA request-sampler step differs from its handle")
        return active

    def _fail_step(self, active: _ActiveCudaRequestSamplerStep) -> None:
        context = active.invocation.context
        match context.optimization, context.publication:
            case ESampOptimized(gradients=gradients), ESampPublicationPending():
                if type(gradients) is not CudaDensePageGradientBatch:
                    raise RuntimeError("CUDA request-sampler gradient token is absent")
                self._provider.discard_gradients(gradients)
            case _:
                pass
        self._directory.release_transition_lease(context.transition_lease)
        self._pending_releases.append(
            self._provider.release_pin(context.state_pin, context.stream)
        )
        self._active_handles.pop(active.token.handle_id)

@dataclass(frozen=True, slots=True)
class ESampRequestCompilation:
    """One executable ESamp specialization and its complete signed plan bundle."""

    execution: CanonicalRequestSamplerExecution | CanonicalRequestSamplerCudaExecution
    plan_bundle: CanonicalPlanBundle
    execution_parameters_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.execution)
            not in (CanonicalRequestSamplerExecution, CanonicalRequestSamplerCudaExecution)
            or type(self.plan_bundle) is not CanonicalPlanBundle
            or self.execution.plan is not self.plan_bundle.schedule_physical
            or self.execution.state_plan is not self.plan_bundle.state_physical
            or type(self.execution_parameters_digest) is not str
            or len(self.execution_parameters_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.execution_parameters_digest
            )
            or self.execution.execution_parameters_digest
            != self.execution_parameters_digest
            or self.plan_bundle.semantic_graph.execution_parameters_digest
            != self.execution_parameters_digest
        ):
            raise TypeError("ESamp compilation requires exact execution and plan bundle")


class ESampRequestWorkload(CanonicalWorkload):
    """Declare ESamp math/dataflow; Rust owns scheduling and representation."""

    __slots__ = ("config",)

    def __init__(self, config: ESampRequestConfig = ESampRequestConfig()) -> None:
        self.config = config

    def semantic_graph(self) -> SemanticGraph:
        cpu_compute = ("compute/cpu@1",)
        cpu_transition = ("compute/cpu@1", "state/dense_page_cpu@1")
        cuda_compute = ("compute/cuda@1",)
        cuda_transition = (
            "compute/cuda@1",
            "state/dense_page_cuda@1",
        )
        selection = StateSelectionInputSource(
            _STATE_RESOURCE,
            HiddenPhase.DECODE,
        )
        return SemanticGraph(
            workload_id=WorkloadId("esamp.request"),
            tasks=(
                SemanticTask(
                    symbol=_LOSS_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("token.commit"),
                        AnchorSymbol("next.prepare"),
                    ),
                    inputs=(
                        SemanticInput(
                            "row_state_indices",
                            TaskResultInputSource(
                                _PREDICT_TASK,
                                "row_state_indices",
                            ),
                        ),
                        SemanticInput(
                            "source",
                            TaskResultInputSource(
                                _PREDICT_TASK,
                                "training_source",
                            ),
                        ),
                        SemanticInput(
                            "state_vectors",
                            TaskResultInputSource(
                                _PREDICT_TASK,
                                "state_vectors",
                            ),
                        ),
                        SemanticInput(
                            "target",
                            TaskResultInputSource(
                                _PREDICT_TASK,
                                "training_target",
                            ),
                        ),
                    ),
                    outputs=(
                        SemanticOutput(
                            "gradient_batch",
                            SemanticOutputKind.TEMPORARY,
                        ),
                        SemanticOutput(
                            "loss_batch",
                            SemanticOutputKind.TEMPORARY,
                        ),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            ESAMP_REQUEST_LOSS_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu_compute,
                        ),
                        TaskImplementationSpec(
                            ESAMP_REQUEST_CUDA_LOSS_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cuda_compute,
                        ),
                    ),
                ),
                SemanticTask(
                    symbol=_PREDICT_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("model.execute"),
                        AnchorSymbol("sampler.pre_draw"),
                    ),
                    inputs=(
                        SemanticInput("rows", PackedRowsInputSource()),
                        SemanticInput(
                            "sampler_rows",
                            PackedSamplerRowsInputSource(),
                        ),
                        SemanticInput("selection", selection),
                        SemanticInput(
                            "source",
                            HiddenInputSource(
                                self.config.source_layer,
                                HiddenSite.BLOCK_OUTPUT,
                                HiddenPhase.DECODE,
                            ),
                        ),
                        SemanticInput(
                            "target",
                            HiddenInputSource(
                                self.config.target_layer,
                                HiddenSite.BLOCK_OUTPUT,
                                HiddenPhase.DECODE,
                            ),
                        ),
                    ),
                    outputs=(
                        SemanticOutput(
                            "guidance",
                            SemanticOutputKind.SAMPLER,
                        ),
                        SemanticOutput(
                            "row_state_indices",
                            SemanticOutputKind.TEMPORARY,
                        ),
                        SemanticOutput(
                            "state_vectors",
                            SemanticOutputKind.TEMPORARY,
                        ),
                        SemanticOutput(
                            "training_source",
                            SemanticOutputKind.TEMPORARY,
                        ),
                        SemanticOutput(
                            "training_target",
                            SemanticOutputKind.TEMPORARY,
                        ),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            ESAMP_REQUEST_PREDICT_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu_compute,
                        ),
                        TaskImplementationSpec(
                            ESAMP_REQUEST_CUDA_PREDICT_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cuda_compute,
                        ),
                    ),
                    effects=(
                        SemanticEffect(_STATE_RESOURCE, StateEffectKind.READ),
                    ),
                ),
                SemanticTask(
                    symbol=_TRAIN_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("token.commit"),
                        AnchorSymbol("next.prepare"),
                    ),
                    inputs=(
                        SemanticInput(
                            "gradient_batch",
                            TaskResultInputSource(
                                _LOSS_TASK,
                                "gradient_batch",
                            ),
                        ),
                        SemanticInput("selection", selection),
                    ),
                    outputs=(),
                    implementations=(
                        TaskImplementationSpec(
                            ESAMP_REQUEST_TRAIN_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu_transition,
                        ),
                        TaskImplementationSpec(
                            ESAMP_REQUEST_CUDA_TRAIN_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cuda_transition,
                        ),
                    ),
                    effects=(
                        SemanticEffect(
                            _STATE_RESOURCE,
                            StateEffectKind.TRANSITION,
                        ),
                    ),
                ),
            ),
            states=(
                SemanticState(
                    _STATE_RESOURCE,
                    StateScope.REQUEST,
                    self.config.key_fields,
                ),
            ),
            execution_parameters_digest=self.config.execution_parameters_digest(),
        )

    def request_sampler_attachment_spec(
        self,
        execution_parameters_digest: str,
    ) -> RequestSamplerAttachmentSpec:
        """Expose semantic roles; Rust still owns task and provider selection."""

        return RequestSamplerAttachmentSpec(
            compute_task=_PREDICT_TASK,
            transition_task=_TRAIN_TASK,
            sampler_anchor=AnchorSymbol("sampler.pre_draw"),
            state_resource=_STATE_RESOURCE,
            request_key_fields=self.config.key_fields,
            execution_parameters_digest=execution_parameters_digest,
        )

    def compile(
        self,
        specialization: ESampRequestSpecialization,
        options: ESampRequestCompileOptions,
    ) -> ESampRequestCompilation:
        if self.config.rank > specialization.hidden_size:
            raise ValueError("ESamp rank exceeds the specialized hidden size")
        if (
            int(specialization.hidden_bindings.source.layer)
            != self.config.source_layer
            or int(specialization.hidden_bindings.target.layer)
            != self.config.target_layer
        ):
            raise ValueError(
                "ESamp packed hidden bindings differ from semantic locators"
            )
        execution_parameters_digest = specialization.execution_parameters_digest(
            self.config
        )
        graph = replace(
            self.semantic_graph(),
            execution_parameters_digest=execution_parameters_digest,
        )
        schedule = compile_schedule(graph, options.timeline)
        validate_schedule(graph, options.timeline, schedule)
        physical = lower_physical_plan(
            graph,
            options.timeline,
            schedule,
            plan_epoch=options.plan_epoch,
        )
        validate_physical_plan(graph, options.timeline, schedule, physical)

        kernel_layout = LowRankGatedLayout(
            specialization.hidden_size,
            self.config.rank,
            self.config.output_layernorm,
        )
        tensor_layout = TensorLayoutSpec(
            namespace="canonical.low_rank_gated_adamw@1",
            fields=(
                TensorFieldSpec(
                    ESAMP_PARAMETER_FIELD,
                    (kernel_layout.parameter_count,),
                    specialization.tensor_dtype,
                    4,
                    MutationGranularity.PAGE,
                ),
                TensorFieldSpec(
                    ESAMP_GRADIENT_FIELD,
                    (kernel_layout.parameter_count,),
                    specialization.tensor_dtype,
                    4,
                    MutationGranularity.PAGE,
                ),
                TensorFieldSpec(
                    ESAMP_STEP_FIELD,
                    (),
                    TensorDType.FLOAT32,
                    4,
                    MutationGranularity.PAGE,
                ),
                TensorFieldSpec(
                    ESAMP_EXP_AVG_FIELD,
                    (kernel_layout.parameter_count,),
                    specialization.tensor_dtype,
                    4,
                    MutationGranularity.PAGE,
                ),
                TensorFieldSpec(
                    ESAMP_EXP_AVG_SQ_FIELD,
                    (kernel_layout.parameter_count,),
                    specialization.tensor_dtype,
                    4,
                    MutationGranularity.PAGE,
                ),
            ),
        )
        layout_request = TensorLayoutRequest(
            (StateTensorLayout(str(_STATE_RESOURCE), tensor_layout),)
        )
        layout_decision = intern_tensor_layouts(layout_request)
        validate_tensor_layouts(layout_request, layout_decision)
        (binding,) = layout_decision.bindings
        candidates = tuple(
            provider.materialize_plan(
                resource=str(_STATE_RESOURCE),
                layout=tensor_layout,
                binding=binding,
                max_admitted_lineages=options.max_admitted_lineages,
            )
            for provider in options.providers
        )
        match specialization.device_kind:
            case StateDeviceKind.CPU:
                required_state_capabilities = (
                    "device/cpu@1",
                    "optimizer/adamw@1",
                    "state/exact_versions@1",
                )
            case StateDeviceKind.CUDA:
                required_state_capabilities = (
                    "device/cuda@1",
                    "optimizer/adamw@1",
                    "state/exact_versions@1",
                    "state/fenced_reclaim@1",
                )
        representation_request = StateRepresentationRequest(
            memory_budget_bytes=options.state_memory_budget_bytes,
            states=(
                StateLayoutRequirement(
                    resource=str(_STATE_RESOURCE),
                    layout_id=binding.layout_id,
                    layout_namespace=binding.layout_namespace,
                    layout_digest=binding.layout_digest,
                    max_admitted_lineages=options.max_admitted_lineages,
                    required_capabilities=required_state_capabilities,
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
        sampler_anchor_ids = tuple(
            AnchorId(anchor.ordinal)
            for anchor in options.timeline.anchors
            if anchor.symbol == AnchorSymbol("sampler.pre_draw")
        )
        if len(sampler_anchor_ids) != 1:
            raise RuntimeError("ESamp timeline requires one sampler.pre_draw anchor")
        (selected_provider,) = representation.plans
        if (
            selected_provider.device_kind is not specialization.device_kind
            or selected_provider.device_ordinal != specialization.device_ordinal
        ):
            raise RuntimeError(
                "Rust-selected state placement differs from specialization"
            )
        directory = RequestStateDirectory(
            binding.layout_id,
            self.config.key_fields,
        )
        plan_bundle = CanonicalPlanBundle(
            semantic_graph=graph,
            timeline=options.timeline,
            schedule_decision=schedule,
            schedule_physical=physical,
            state_request=representation_request,
            state_decision=representation,
            state_physical=state_physical,
        )
        match selected_provider.device_kind:
            case StateDeviceKind.CPU:
                provider = DensePageStateProvider(
                    tensor_layout_request=layout_request,
                    tensor_layout_decision=layout_decision,
                    representation_request=representation_request,
                    representation_decision=representation,
                    physical_plan=state_physical,
                    local_candidates=candidates,
                    resource=str(_STATE_RESOURCE),
                )
                execution = CanonicalRequestSamplerExecution(
                    config=self.config,
                    specialization=specialization,
                    plan=physical,
                    state_plan=state_physical,
                    kernel_layout=kernel_layout,
                    provider=provider,
                    directory=directory,
                    sampler_completion_anchor_id=sampler_anchor_ids[0],
                )
            case StateDeviceKind.CUDA:
                provider = CudaDensePageStateProvider(
                    tensor_layout_request=layout_request,
                    tensor_layout_decision=layout_decision,
                    representation_request=representation_request,
                    representation_decision=representation,
                    physical_plan=state_physical,
                    local_candidates=candidates,
                    resource=str(_STATE_RESOURCE),
                    device_ordinal=selected_provider.device_ordinal,
                )
                execution = CanonicalRequestSamplerCudaExecution(
                    config=self.config,
                    specialization=specialization,
                    plan=physical,
                    state_plan=state_physical,
                    kernel_layout=kernel_layout,
                    provider=provider,
                    directory=directory,
                    sampler_completion_anchor_id=sampler_anchor_ids[0],
                )
        return ESampRequestCompilation(
            execution,
            plan_bundle,
            execution_parameters_digest,
        )


__all__ = [
    "ESampRequestCompileOptions",
    "ESampRequestCompilation",
    "ESampRequestConfig",
    "ESampRequestSpecialization",
    "CanonicalRequestSamplerExecution",
    "CanonicalRequestSamplerCudaExecution",
    "CanonicalRequestCudaStepHandle",
    "CanonicalRequestStepHandle",
    "ESampRequestWorkload",
]

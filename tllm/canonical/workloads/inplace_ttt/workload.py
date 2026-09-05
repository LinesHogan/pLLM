"""In-Place-TTT declaration and shared signed execution lifecycle."""

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
)
from tllm.canonical.ptts.dense_page_cuda import CudaDensePageStateProvider
from tllm.canonical.ptts.cuda_publication import (
    CudaRequestStatePublicationCoordinator,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
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
from tllm.canonical.workloads.inplace_ttt.layout import (
    InPlaceTTTStateLayout,
    build_inplace_ttt_state_layout,
)
from tllm.canonical.workloads.inplace_ttt.tasks import (
    InPlaceTTTCommittedTransition,
    InPlaceTTTCpuStorage,
    InPlaceTTTCudaStorage,
    InPlaceTTTPublishTask,
    InPlaceTTTStepContext,
    InPlaceTTTSubmitting,
    InPlaceTTTTransitionPending,
    InPlaceTTTTransitionTask,
    INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION,
    INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION,
    INPLACE_TTT_PUBLISH_IMPLEMENTATION,
    INPLACE_TTT_TRANSITION_IMPLEMENTATION,
)
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTCompileOptions,
    CudaInPlaceTTTStepBatchInput,
    InPlaceTTTCleanupReceipt,
    InPlaceTTTConfig,
    InPlaceTTTDiagnostics,
    InPlaceTTTStepHandle,
    InPlaceTTTSpecialization,
    InPlaceTTTStepBatch,
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
    SemanticEffect,
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
    TensorLayoutRequest,
)

_FAST_WEIGHT_STATE = StateResource("dense_fast_weights")
_PUBLISH_TASK = TaskSymbol("publish")
_TRANSITION_TASK = TaskSymbol("transition")


_DenseStorage = (
    tuple[DensePageStateProvider, RequestStatePublicationCoordinator]
    | tuple[CudaDensePageStateProvider, CudaRequestStatePublicationCoordinator]
)


@dataclass(frozen=True, slots=True)
class _DenseSamplerPending:
    pass


@dataclass(frozen=True, slots=True)
class _DenseSamplerReleased:
    release: CudaDensePageRelease | None


@dataclass(frozen=True, slots=True)
class _DenseLive:
    pass


@dataclass(frozen=True, slots=True)
class _DenseFailed:
    pass


@dataclass(slots=True)
class _ActiveDenseStep:
    token: InPlaceTTTStepHandle
    context: InPlaceTTTStepContext
    last_attachment_anchor_id: int
    sampler: _DenseSamplerPending | _DenseSamplerReleased


class CanonicalDenseOverlayExecution:
    """One pinned batch lifecycle, shared by synchronous and captured execution."""

    def __init__(
        self,
        config: InPlaceTTTConfig,
        specialization: InPlaceTTTSpecialization,
        plan_bundle: CanonicalPlanBundle,
        layout: InPlaceTTTStateLayout,
        storage: _DenseStorage,
        directory: RequestStateDirectory,
    ) -> None:
        if (
            plan_bundle.semantic_graph.execution_parameters_digest
            != specialization.execution_parameters_digest(config)
            or plan_bundle.schedule_physical.plan_epoch != plan_bundle.state_physical.plan_epoch
            or storage[0].device != specialization.device
        ):
            raise ValueError("dense overlay execution artifacts disagree")
        validate_schedule(
            plan_bundle.semantic_graph, plan_bundle.timeline, plan_bundle.schedule_decision,
        )
        validate_physical_plan(
            plan_bundle.semantic_graph, plan_bundle.timeline, plan_bundle.schedule_decision,
            plan_bundle.schedule_physical,
        )
        self._config = config
        self._specialization = specialization
        self._plan_bundle = plan_bundle
        self._layout = layout
        self._storage = storage
        self._directory = directory
        publish_id, transition_id = (
            (INPLACE_TTT_PUBLISH_IMPLEMENTATION, INPLACE_TTT_TRANSITION_IMPLEMENTATION)
            if specialization.device_kind is StateDeviceKind.CPU else
            (INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION, INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION)
        )
        self._runner = TaskGraphExecution(
            plan_bundle.schedule_physical,
            (InPlaceTTTPublishTask(publish_id),
             InPlaceTTTTransitionTask(config, layout, directory, transition_id)),
        )
        self._active: dict[int, _ActiveDenseStep] = {}
        self._execution_state: _DenseLive | _DenseFailed = _DenseLive()
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
        self, identities: tuple[RequestIdentity, ...], stream: torch.cuda.Stream | None = None,
    ) -> tuple[StateRef, ...]:
        with self._lock:
            self._require_live()
            self._require_stream(stream)
            logical = self._directory.prepare_admission(identities)
            try:
                match self._storage[0]:
                    case DensePageStateProvider() as provider:
                        provider.admit_batch(tuple(
                            DensePageAdmission(admission.ref, provider.new_zero_root_tensors())
                            for admission in logical.admissions
                        ))
                    case CudaDensePageStateProvider() as provider:
                        if stream is None:
                            raise RuntimeError("CUDA admission requires a stream")
                        prepared = provider.prepare_admission_batch(tuple(
                            CudaDensePageAdmission(admission.ref, provider.new_zero_root(stream))
                            for admission in logical.admissions
                        ))
                        provider.submit_admission_batch(prepared, stream)
            except Exception:
                self._directory.abort_admission(logical)
                raise
            self._directory.commit_admission(logical)
            return tuple(admission.ref for admission in logical.admissions)

    def step(self, step: InPlaceTTTStepInput) -> InPlaceTTTStepResult:
        with self._lock:
            self._require_live()
            self._require_stream(None)
            active = self._begin((step.identity,), None, (), self.plan.tasks[0].anchor_id)
            self._submit(active, InPlaceTTTStepBatch((step,)))
            self.complete_sampler(active.token, None, self.plan.tasks[-1].anchor_id)
            (result,) = self.finish_step(active.token).entries
            return result

    def begin_batch(
        self,
        identities: tuple[RequestIdentity, ...],
        stream: torch.cuda.Stream | None,
        bindings: tuple[ParameterOverlayFieldBinding, ...],
        pin_anchor_id: AnchorId,
    ) -> InPlaceTTTStepHandle:
        with self._lock:
            self._require_live()
            self._require_stream(stream)
            if (
                tuple(binding.target for binding in bindings) != self._config.targets
                or any(binding.fields != (self._layout.fields_for(binding.target).delta_weight,)
                       for binding in bindings)
            ):
                raise ValueError("signed overlay bindings differ from state layout")
            return self._begin(identities, stream, bindings, pin_anchor_id).token

    def _begin(
        self, identities: tuple[RequestIdentity, ...], stream: torch.cuda.Stream | None,
        bindings: tuple[ParameterOverlayFieldBinding, ...], pin_anchor_id: AnchorId,
    ) -> _ActiveDenseStep:
        self._collect_completed()
        selection = self._directory.select(IndexedRequestIdentityBatch(tuple(
            IndexedRequestIdentityRow(index, identity)
            for index, identity in enumerate(identities)
        )))
        lease = self._directory.acquire_transition_lease(selection)
        try:
            storage = self._pin(selection.refs, stream)
        except Exception:
            self._directory.release_transition_lease(lease)
            raise
        context = InPlaceTTTStepContext(
            None, selection, lease, storage, InPlaceTTTTransitionPending(),
        )
        try:
            field_rows = tuple(
                (binding, storage.field_rows(selection.refs, binding.fields[0]))
                for binding in bindings
            )
            deltas = tuple(
                ParameterOverlayPinnedTensor(binding.target, ref, rows[index])
                for binding, rows in field_rows
                for index, ref in enumerate(selection.refs)
            )
        except Exception:
            release = storage.release()
            if release is not None:
                self._pending_releases.append(release)
            self._directory.release_transition_lease(lease)
            raise
        token = InPlaceTTTStepHandle(self._next_handle_id, identities, selection.refs, deltas)
        self._next_handle_id += 1
        active = _ActiveDenseStep(token, context, int(pin_anchor_id), _DenseSamplerPending())
        self._active[token.handle_id] = active
        return active

    def _pin(
        self, refs: tuple[StateRef, ...], stream: torch.cuda.Stream | None,
    ) -> InPlaceTTTCpuStorage | InPlaceTTTCudaStorage:
        match self._storage:
            case DensePageStateProvider() as provider, RequestStatePublicationCoordinator() as publication:
                return InPlaceTTTCpuStorage(provider, publication, provider.pin_exact(refs))
            case CudaDensePageStateProvider() as provider, CudaRequestStatePublicationCoordinator() as publication:
                if stream is None:
                    raise RuntimeError("CUDA pin requires a stream")
                return InPlaceTTTCudaStorage(
                    provider, publication, provider.pin_exact(refs, stream), stream,
                )

    def submit_captured(
        self,
        handle: InPlaceTTTStepHandle,
        captured: InPlaceTTTStepBatch | CudaInPlaceTTTStepBatchInput,
        transition_task: AttachmentTaskBinding,
        publish_task: AttachmentTaskBinding,
    ) -> None:
        with self._lock:
            self._require_live()
            active = self._active_step(handle)
            if active.context.batch is not None:
                raise RuntimeError("dense overlay operands were already submitted")
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
                stream = active.context.storage.stream
                match captured, stream:
                    case InPlaceTTTStepBatch() as batch, None:
                        pass
                    case CudaInPlaceTTTStepBatchInput() as cuda_batch, torch.cuda.Stream():
                        cuda_batch.order_on(stream)
                        batch = cuda_batch.batch
                    case _:
                        raise ValueError("captured operands differ from the execution device")
            except Exception:
                self._release_step(active)
                raise
            self._submit(active, batch)

    def _submit(self, active: _ActiveDenseStep, batch: InPlaceTTTStepBatch) -> None:
        context = active.context
        try:
            if tuple(entry.identity for entry in batch.entries) != active.token.identities:
                raise ValueError("dense overlay operands belong elsewhere")
            for entry in batch.entries:
                validate_inplace_ttt_step(entry, self._specialization)
            context.batch = batch
            self._runner.run(context)
            context.result()
        except Exception:
            match context.transition:
                case InPlaceTTTSubmitting() | InPlaceTTTCommittedTransition():
                    self._execution_state = _DenseFailed()
                case _:
                    context.abort()
                    self._release_step(active)
            raise
        active.last_attachment_anchor_id = int(self.plan.tasks[-1].anchor_id)

    def complete_sampler(
        self, handle: InPlaceTTTStepHandle, sampler_stream: torch.cuda.Stream | None,
        release_anchor_id: AnchorId,
    ) -> None:
        with self._lock:
            self._require_live()
            self._require_stream(sampler_stream)
            active = self._active_step(handle)
            active.context.result()
            match active.sampler:
                case _DenseSamplerPending():
                    pass
                case _DenseSamplerReleased():
                    raise RuntimeError("sampler completion is already recorded")
            if int(release_anchor_id) < active.last_attachment_anchor_id:
                raise ValueError("signed release anchor precedes publication")
            stream = active.context.storage.stream
            if sampler_stream is not None and stream is not None and sampler_stream != stream:
                CudaStreamReadiness(sampler_stream).order_on(stream)
            release = active.context.storage.release()
            active.last_attachment_anchor_id = int(release_anchor_id)
            active.sampler = _DenseSamplerReleased(release)

    def finish_step(self, handle: InPlaceTTTStepHandle) -> InPlaceTTTStepBatchResult:
        with self._lock:
            self._require_live()
            active = self._active_step(handle)
            match active.sampler:
                case _DenseSamplerReleased():
                    pass
                case _DenseSamplerPending():
                    raise RuntimeError("dense overlay requires sampler completion")
            result = active.context.result()
            self._release_step(active)
            self._update_count += sum(
                target.update_count for entry in result.entries for target in entry.targets
            )
            self._trained_rows += sum(
                target.trained_rows for entry in result.entries for target in entry.targets
            )
            return result

    def abort_step(self, handle: InPlaceTTTStepHandle) -> None:
        with self._lock:
            self._require_live()
            self._release_step(self._active_step(handle))

    def _release_step(self, active: _ActiveDenseStep) -> None:
        match active.sampler:
            case _DenseSamplerReleased(release=release):
                pass
            case _DenseSamplerPending():
                release = active.context.storage.release()
        self._directory.release_transition_lease(active.context.transition_lease)
        if release is not None:
            self._pending_releases.append(release)
        self._active.pop(active.token.handle_id)

    def retire(
        self, identities: tuple[RequestIdentity, ...], finish_anchor_id: AnchorId | None = None,
    ) -> InPlaceTTTCleanupReceipt:
        with self._lock:
            self._require_live()
            if finish_anchor_id is not None and int(finish_anchor_id) < 0:
                raise ValueError("signed finish anchor is invalid")
            if self.device.type == "cuda" and finish_anchor_id is None:
                raise ValueError("CUDA retirement requires a signed finish anchor")
            self._collect_completed()
            match self._storage[0]:
                case CudaDensePageStateProvider() as provider:
                    for release in self._pending_releases:
                        provider.wait_release(release)
                    self._pending_releases.clear()
                    provider.collect_completed()
                case DensePageStateProvider():
                    pass
            logical = self._directory.prepare_retirement(identities)
            try:
                self._storage[0].retire_batch(logical.refs)
            except Exception:
                self._directory.abort_retirement(logical)
                raise
            self._directory.commit_retirement(logical)
            return InPlaceTTTCleanupReceipt(logical.refs)

    def snapshot(
        self, ref: StateRef, stream: torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, ...]:
        with self._lock:
            self._require_live()
            self._require_stream(stream)
            match self._storage[0]:
                case DensePageStateProvider() as provider:
                    return provider.snapshot(ref)
                case CudaDensePageStateProvider() as provider:
                    if stream is None:
                        raise RuntimeError("CUDA snapshot requires a stream")
                    pin = provider.pin_exact((ref,), stream)
                    try:
                        return provider.snapshot(pin, ref, stream)
                    finally:
                        release = provider.release_pin(pin, stream)
                        provider.wait_release(release)

    def diagnostics(self) -> InPlaceTTTDiagnostics:
        with self._lock:
            self._require_live()
            self._collect_completed()
            match self._storage[0]:
                case CudaDensePageStateProvider() as provider:
                    pending = provider.pending_pin_releases
                case DensePageStateProvider():
                    pending = 0
            return InPlaceTTTDiagnostics(
                self._directory.active_count, self._update_count, self._trained_rows,
                self.state_plan, self.plan.fallback_count, pending,
            )

    def _collect_completed(self) -> None:
        match self._storage[0]:
            case CudaDensePageStateProvider() as provider:
                self._pending_releases = [
                    release for release in self._pending_releases
                    if not provider.release_complete(release)
                ]
                provider.collect_completed()
            case DensePageStateProvider():
                pass

    def _require_stream(self, stream: torch.cuda.Stream | None) -> None:
        if self.device.type == "cpu":
            if stream is not None:
                raise RuntimeError("CPU dense overlay does not accept a CUDA stream")
        elif stream is None or stream.device != self.device:
            raise RuntimeError("CUDA dense overlay requires a stream on its device")

    def _active_step(self, handle: InPlaceTTTStepHandle) -> _ActiveDenseStep:
        try:
            active = self._active[handle.handle_id]
        except KeyError as exc:
            raise RuntimeError("dense overlay step handle is not active") from exc
        if active.token is not handle:
            raise RuntimeError("dense overlay step differs from its handle")
        return active

    def _require_live(self) -> None:
        match self._execution_state:
            case _DenseLive():
                pass
            case _DenseFailed():
                raise RuntimeError("dense overlay execution is fail-stopped")


@dataclass(frozen=True, slots=True)
class InPlaceTTTCompilation:
    execution: CanonicalDenseOverlayExecution
    plan_bundle: CanonicalPlanBundle
    state_layout: InPlaceTTTStateLayout

    def __post_init__(self) -> None:
        if (
            self.execution.plan is not self.plan_bundle.schedule_physical
            or self.execution.state_plan is not self.plan_bundle.state_physical
            or self.execution.execution_parameters_digest
            != self.plan_bundle.semantic_graph.execution_parameters_digest
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
            layout_request,
            layout_decision,
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
                storage: _DenseStorage = (cpu_provider, cpu_publication)
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
                storage = (cuda_provider, cuda_publication)
        execution = CanonicalDenseOverlayExecution(
            self.config, specialization, plan_bundle, state_layout, storage, directory,
        )
        return InPlaceTTTCompilation(
            execution,
            plan_bundle,
            state_layout,
        )


__all__ = [
    "CanonicalDenseOverlayExecution",
    "InPlaceTTTCompilation",
    "InPlaceTTTWorkload",
]

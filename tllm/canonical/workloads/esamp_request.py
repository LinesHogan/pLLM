"""Clean-room ESamp request graph and cold specialization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import nullcontext
from threading import RLock

import torch

from tllm.canonical.cuda_packed_hidden import CudaPackedHiddenSnapshotBatch
from tllm.canonical.execution import TaskGraphExecution, TaskGraphInvocation
from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
    initialize_low_rank_gated,
)
from tllm.canonical.ptts.page_bundle import PageBundleArena, PagePublication
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.contracts.backend_attachment import (
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
    ESampLossTask,
    ESampPredictTask,
    ESampStepContext,
    ESampTrainTask,
    ESampOptimizationPending,
    ESampOptimized,
    ESampPredicted,
    ESampPredictionPending,
    ESampPublicationPending,
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
    context: ESampStepContext,
    *,
    intervention_beta: float,
    error_message: str,
) -> ESampRequestStepResult:
    match context.prediction, context.optimization, context.publication:
        case (
            ESampPredicted() as predicted,
            ESampOptimized() as optimized,
            PagePublication() as published,
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
class _SamplerReady:
    stream: torch.cuda.Stream


@dataclass(frozen=True, slots=True)
class _SamplerCompleted:
    pass


@dataclass(frozen=True, slots=True)
class CanonicalRequestStepHandle:
    """Opaque token issued by one request-sampler execution."""

    handle_id: int


@dataclass(slots=True)
class _ActiveRequestSamplerStep:
    token: CanonicalRequestStepHandle
    invocation: TaskGraphInvocation[ESampStepContext]
    sampler: _SamplerPending | _SamplerReady | _SamplerCompleted


class CanonicalRequestSamplerExecution:
    """One request lifecycle; device-specific operations bind physical storage."""

    def __init__(
        self,
        *,
        config: ESampRequestConfig,
        specialization: ESampRequestSpecialization,
        plan: PhysicalPlan,
        state_plan: StatePhysicalPlan,
        kernel_layout: LowRankGatedLayout,
        arena: PageBundleArena,
        directory: RequestStateDirectory,
        sampler_completion_anchor_id: AnchorId,
    ) -> None:
        expected_device = (
            torch.device("cpu") if specialization.device_kind is StateDeviceKind.CPU
            else torch.device("cuda", specialization.device_ordinal)
        )
        if plan.plan_epoch != state_plan.plan_epoch or arena.device != expected_device:
            raise ValueError("request-sampler plans or placement differ")
        self._config = config
        self._specialization = specialization
        self._plan = plan
        self._state_plan = state_plan
        self._kernel_layout = kernel_layout
        self._arena = arena
        self._directory = directory
        self._sampler_completion_anchor_id = sampler_completion_anchor_id
        loss_id, predict_id, train_id = (
            (ESAMP_REQUEST_LOSS_IMPLEMENTATION, ESAMP_REQUEST_PREDICT_IMPLEMENTATION,
             ESAMP_REQUEST_TRAIN_IMPLEMENTATION)
            if specialization.device_kind is StateDeviceKind.CPU else
            (ESAMP_REQUEST_CUDA_LOSS_IMPLEMENTATION, ESAMP_REQUEST_CUDA_PREDICT_IMPLEMENTATION,
             ESAMP_REQUEST_CUDA_TRAIN_IMPLEMENTATION)
        )
        self._runner = TaskGraphExecution(
            plan,
            (ESampLossTask(kernel_layout, loss_id),
             ESampPredictTask(config, kernel_layout, predict_id),
             ESampTrainTask(config, directory, train_id)),
        )
        self._generator = torch.Generator(device=arena.device)
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
    def device(self) -> torch.device:
        return self._arena.device

    @property
    def free_state_pages(self) -> int:
        return self._arena.free_pages

    @property
    def state_pool_addresses(self) -> tuple[int, ...]:
        return self._arena.pool_addresses

    def require_stable_state_pool(self) -> None:
        self._arena.require_stable()

    def admit(
        self, identities: tuple[RequestIdentity, ...], stream: torch.cuda.Stream | None = None,
    ) -> tuple[StateRef, ...]:
        with self._lock:
            batch = self._directory.prepare_admission(identities, stream)
            try:
                with nullcontext() if stream is None else torch.cuda.stream(stream):
                    for tensors in self._arena.writable_roots(batch.write):
                        self._initialize(tensors)
            except Exception:
                self._directory.abort_admission(batch)
                raise
            receipt = self._directory.commit_admission(batch)
            self._arena.settle(receipt)
            return receipt.after_refs

    def _initialize(self, tensors: tuple[torch.Tensor, ...]) -> None:
        parameters = tensors[int(ESAMP_PARAMETER_FIELD)]
        parameters.copy_(initialize_low_rank_gated(
            self._kernel_layout, dtype=parameters.dtype, device=parameters.device,
            generator=self._generator,
        ))

    def start_step(
        self, step: ESampRequestStepInput, stream: torch.cuda.Stream | None = None,
    ) -> CanonicalRequestStepHandle:
        with self._lock:
            self._require_stream(stream)
            match step.hidden, stream:
                case PackedHiddenSnapshotBatch() as hidden, None:
                    pass
                case CudaPackedHiddenSnapshotBatch() as captured, torch.cuda.Stream():
                    captured.order_on(stream)
                    hidden = captured.batch
                case _:
                    raise TypeError("request-sampler hidden snapshots differ from device binding")
            bindings = self._specialization.hidden_bindings
            if tuple(snapshot.resource for snapshot in hidden.snapshots) != bindings.resources:
                raise ValueError("request-sampler snapshots differ from signed bindings")
            source = hidden.hidden_for(bindings.source)
            target = hidden.hidden_for(bindings.target)
            if (
                step.rows.plan_epoch != self._plan.plan_epoch
                or int(source.shape[1]) != self._specialization.hidden_size
                or source.dtype is not torch.float32
                or source.device != self.device
                or target.shape != source.shape
            ):
                raise ValueError("request-sampler step differs from specialization")
            selection = self._directory.select(_decode_indexed_rows(
                step, empty_message="request-sampler step requires decode rows",
            ))
            lease = self._directory.acquire_transition_lease(selection)
            try:
                pin = self._arena.pin_exact(selection.refs, stream)
            except Exception:
                self._directory.release_transition_lease(lease)
                raise
            context = ESampStepContext(
                step, source, target, selection, lease, ESampPredictionPending(),
                ESampOptimizationPending(), ESampPublicationPending(), self._arena, pin,
            )
            handle = CanonicalRequestStepHandle(self._next_handle_id)
            self._next_handle_id += 1
            self._active_handles[handle.handle_id] = _ActiveRequestSamplerStep(
                handle, self._runner.start(context), _SamplerPending(),
            )
            return handle

    def execute_anchor(self, handle: CanonicalRequestStepHandle, anchor_id: AnchorId) -> None:
        with self._lock:
            active = self._active_step(handle)
            if int(anchor_id) > int(self._sampler_completion_anchor_id):
                match active.sampler:
                    case _SamplerPending() | _SamplerReady():
                        raise RuntimeError("later anchors require sampler completion")
                    case _SamplerCompleted():
                        pass
            try:
                self._runner.execute_anchor(active.invocation, anchor_id)
            except Exception:
                self._fail_step(active)
                raise

    def sampler_guidance(
        self, handle: CanonicalRequestStepHandle, stream: torch.cuda.Stream | None = None,
    ) -> PackedSamplerGuidanceBatch:
        with self._lock:
            self._require_stream(stream)
            active = self._active_step(handle)
            match active.invocation.context.prediction:
                case ESampPredicted() as predicted:
                    pass
                case ESampPredictionPending():
                    raise RuntimeError("request-sampler prediction anchor has not executed")
            if stream is not None:
                match active.sampler:
                    case _SamplerPending():
                        if predicted.readiness is None:
                            raise RuntimeError("CUDA request-sampler readiness is absent")
                        predicted.readiness.order_on(stream)
                        active.sampler = _SamplerReady(stream)
                    case _SamplerReady(stream=issued) if issued == stream:
                        pass
                    case _SamplerReady() | _SamplerCompleted():
                        raise RuntimeError("request-sampler sampler phase is invalid")
            return predicted.guidance

    def complete_sampler(
        self, handle: CanonicalRequestStepHandle, stream: torch.cuda.Stream | None = None,
    ) -> None:
        with self._lock:
            self._require_stream(stream)
            active = self._active_step(handle)
            _require_sampler_anchor_reached(
                last_anchor_id=active.invocation.last_anchor_id,
                sampler_completion_anchor_id=self._sampler_completion_anchor_id,
                message="request-sampler sampler cannot complete before its signed anchor",
            )
            match active.sampler:
                case _SamplerPending() if stream is None:
                    pass
                case _SamplerReady(stream=issued) if issued == stream:
                    self._order_sampler_completion(active, issued)
                case _SamplerPending():
                    raise RuntimeError("request-sampler guidance was not acquired")
                case _SamplerReady() | _SamplerCompleted():
                    raise RuntimeError("request-sampler completion is invalid or already completed")
            active.sampler = _SamplerCompleted()
            self._completed_sampler_pins += 1

    def _order_sampler_completion(
        self, active: _ActiveRequestSamplerStep, sampler_stream: torch.cuda.Stream,
    ) -> None:
        stream = active.invocation.context.pin.stream
        if stream is not None:
            completion = torch.cuda.Event(blocking=False)
            completion.record(sampler_stream)
            stream.wait_event(completion)

    def finish_step(self, handle: CanonicalRequestStepHandle) -> ESampRequestStepResult:
        with self._lock:
            active = self._active_step(handle)
            match active.sampler:
                case _SamplerCompleted():
                    pass
                case _SamplerPending() | _SamplerReady():
                    raise RuntimeError("request-sampler requires sampler completion")
            context = self._runner.finish(active.invocation)
            result = _completed_step_result(
                context, intervention_beta=self._config.intervention_beta,
                error_message="request-sampler physical plan did not produce all outputs",
            )
            self._loss_count += len(result.losses)
            self._update_count += len(result.after_refs)
            self._release_step(active)
            return result

    def cancel_step(self, handle: CanonicalRequestStepHandle) -> None:
        with self._lock:
            self._fail_step(self._active_step(handle))

    def _release_step(self, active: _ActiveRequestSamplerStep) -> None:
        context = active.invocation.context
        self._arena.release_pin(context.pin)
        if isinstance(context.publication, PagePublication):
            self._arena.settle(context.publication)
        self._directory.release_transition_lease(context.transition_lease)
        self._active_handles.pop(active.token.handle_id)

    def _fail_step(self, active: _ActiveRequestSamplerStep) -> None:
        match active.sampler:
            case _SamplerReady(stream=stream):
                self._order_sampler_completion(active, stream)
            case _SamplerPending() | _SamplerCompleted():
                pass
        self._release_step(active)

    def retire(self, identities: tuple[RequestIdentity, ...]) -> None:
        with self._lock:
            retirement_keys = {
                RequestStateKey.project(identity, self._config.key_fields)
                for identity in identities
            }
            for active in tuple(self._active_handles.values()):
                if retirement_keys.intersection(active.invocation.context.selection.keys):
                    self._fail_step(active)
            self._collect_completed(wait=True)
            self._directory.retire(identities)

    def snapshot(
        self, ref: StateRef, stream: torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, ...]:
        with self._lock:
            pin = self._arena.pin_exact((ref,), stream)
            try:
                return self._arena.snapshot(pin, ref)
            finally:
                self._arena.release_pin(pin)
                self._arena.collect_completed()

    def collect_completed(self) -> None:
        self._arena.collect_completed()

    def _collect_completed(self, *, wait: bool) -> None:
        self._arena.collect_completed(wait=wait)

    def diagnostics(self) -> ESampRequestDiagnostics:
        with self._lock:
            return ESampRequestDiagnostics(
                self._directory.active_count, self._loss_count, self._update_count,
                self._completed_sampler_pins, self._arena.pending_pin_releases, self._state_plan,
            )

    def _require_stream(self, stream: torch.cuda.Stream | None) -> None:
        if self.device.type == "cpu":
            if stream is not None:
                raise RuntimeError("CPU request-sampler does not accept a CUDA stream")
        elif stream is None or stream.device != self.device:
            raise RuntimeError("CUDA request-sampler requires a stream on its device")

    def _active_step(self, handle: CanonicalRequestStepHandle) -> _ActiveRequestSamplerStep:
        try:
            active = self._active_handles[handle.handle_id]
        except KeyError as exc:
            raise RuntimeError("request-sampler step is not registered as active") from exc
        if active.token is not handle:
            raise RuntimeError("request-sampler step differs from the registered handle")
        return active


@dataclass(frozen=True, slots=True)
class ESampRequestCompilation:
    """One executable ESamp specialization and its complete signed plan bundle."""

    execution: CanonicalRequestSamplerExecution
    plan_bundle: CanonicalPlanBundle

    def __post_init__(self) -> None:
        if (
            self.execution.plan is not self.plan_bundle.schedule_physical
            or self.execution.state_plan is not self.plan_bundle.state_physical
            or self.execution.execution_parameters_digest
            != self.plan_bundle.semantic_graph.execution_parameters_digest
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
    ) -> RequestSamplerAttachmentSpec:
        """Expose semantic roles; Rust still owns task and provider selection."""

        return RequestSamplerAttachmentSpec(
            compute_task=_PREDICT_TASK,
            transition_task=_TRAIN_TASK,
            sampler_anchor=AnchorSymbol("sampler.pre_draw"),
            state_resource=_STATE_RESOURCE,
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
        plan_bundle = CanonicalPlanBundle(
            semantic_graph=graph,
            timeline=options.timeline,
            schedule_decision=schedule,
            schedule_physical=physical,
            state_request=representation_request,
            state_decision=representation,
            state_physical=state_physical,
            tensor_layout_request=layout_request,
            tensor_layout_decision=layout_decision,
        )
        arena = PageBundleArena(plan_bundle, candidates)
        directory = RequestStateDirectory(arena, str(_STATE_RESOURCE), self.config.key_fields)
        execution = CanonicalRequestSamplerExecution(
            config=self.config,
            specialization=specialization,
            plan=physical,
            state_plan=state_physical,
            kernel_layout=kernel_layout,
            arena=arena,
            directory=directory,
            sampler_completion_anchor_id=sampler_anchor_ids[0],
        )
        return ESampRequestCompilation(
            execution,
            plan_bundle,
        )


__all__ = [
    "ESampRequestCompileOptions",
    "ESampRequestCompilation",
    "ESampRequestConfig",
    "ESampRequestSpecialization",
    "CanonicalRequestSamplerExecution",
    "CanonicalRequestStepHandle",
    "ESampRequestWorkload",
]

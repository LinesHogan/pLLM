"""Clean-room ESamp request graph and cold specialization."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from tllm.canonical.cuda_packed_hidden import CudaPackedHiddenSnapshotBatch
from tllm.canonical.state_execution import StateExecution, StateTaskContext
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
    ESampData,
    ESampTrainTask,
    ESampOptimizationPending,
    ESampOptimized,
    ESampPredicted,
    ESampPredictionPending,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions,
    ESampRequestConfig,
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
from tllm.contracts.request_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
)
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
    StateLayoutRequirement,
    StateRepresentationRequest,
)
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


@dataclass(slots=True)
class ESampMath:
    config: ESampRequestConfig
    specialization: ESampRequestSpecialization
    layout: LowRankGatedLayout
    generator: torch.Generator

    def initialize(self, roots: tuple[tuple[torch.Tensor, ...], ...]) -> None:
        for root in roots:
            parameters = root[int(ESAMP_PARAMETER_FIELD)]
            parameters.copy_(initialize_low_rank_gated(self.layout, dtype=parameters.dtype,
                device=parameters.device, generator=self.generator))

    def rows(self, value: ESampRequestStepInput) -> IndexedRequestIdentityBatch:
        return _decode_indexed_rows(value, empty_message="ESamp requires decode rows")

    def prepare(self, value: ESampRequestStepInput, context: StateTaskContext[ESampData]) -> ESampData:
        stream = context.pin.stream
        match value.hidden, stream:
            case PackedHiddenSnapshotBatch() as hidden, None:
                pass
            case CudaPackedHiddenSnapshotBatch() as captured, torch.cuda.Stream():
                captured.order_on(stream)
                hidden = captured.batch
            case _:
                raise ValueError("ESamp hidden snapshots differ from the input device")
        bindings = self.specialization.hidden_bindings
        if tuple(snapshot.resource for snapshot in hidden.snapshots) != bindings.resources:
            raise ValueError("ESamp snapshots differ from signed bindings")
        source, target = hidden.hidden_for(bindings.source), hidden.hidden_for(bindings.target)
        if (value.rows.plan_epoch != context.arena.plan.schedule_physical.plan_epoch
            or source.shape[1:] != (self.specialization.hidden_size,) or source.dtype is not torch.float32
            or target.shape != source.shape or source.device != context.arena.device):
            raise ValueError("ESamp input differs from specialization")
        return ESampData(value, source, target, ESampPredictionPending(), ESampOptimizationPending())

    def output(self, context: StateTaskContext[ESampData]) -> ESampRequestStepResult:
        data = context.data()
        match data.prediction, data.optimization, context.publication:
            case ESampPredicted() as predicted, ESampOptimized() as optimized, PagePublication() as published:
                return ESampRequestStepResult(data.step.rows.step_id, predicted.guidance,
                    self.config.intervention_beta, optimized.losses, published.before_refs, published.after_refs)
            case _:
                raise RuntimeError("ESamp signed graph did not produce all outputs")


@dataclass(frozen=True, slots=True)
class ESampRequestCompilation:
    execution: StateExecution[ESampRequestStepInput, ESampData, ESampRequestStepResult]
    plan_bundle: CanonicalPlanBundle
    math: ESampMath


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
        directory = RequestStateDirectory(arena, graph.states[0])
        math = ESampMath(self.config, specialization, kernel_layout,
            torch.Generator(device=arena.device).manual_seed(specialization.initialization_seed))
        loss_id, predict_id, train_id = (
            (ESAMP_REQUEST_LOSS_IMPLEMENTATION, ESAMP_REQUEST_PREDICT_IMPLEMENTATION,
             ESAMP_REQUEST_TRAIN_IMPLEMENTATION) if specialization.device_kind is StateDeviceKind.CPU else
            (ESAMP_REQUEST_CUDA_LOSS_IMPLEMENTATION, ESAMP_REQUEST_CUDA_PREDICT_IMPLEMENTATION,
             ESAMP_REQUEST_CUDA_TRAIN_IMPLEMENTATION)
        )
        execution = StateExecution(arena, directory, math,
            (ESampLossTask(kernel_layout, loss_id), ESampPredictTask(self.config, kernel_layout, predict_id),
             ESampTrainTask(self.config, train_id)), sampler_anchor_ids[0],
        )
        return ESampRequestCompilation(execution, plan_bundle, math)


__all__ = [
    "ESampRequestCompileOptions",
    "ESampRequestCompilation",
    "ESampRequestConfig",
    "ESampRequestSpecialization",
    "ESampRequestWorkload",
]

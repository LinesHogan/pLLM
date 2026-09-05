"""In-Place-TTT declaration and shared signed execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.state_execution import StateExecution, StateTaskContext
from tllm.canonical.ptts.page_bundle import PageBundleArena
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.canonical.scheduler import (
    compile_schedule,
    lower_physical_plan,
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
    InPlaceTTTPublishTask,
    InPlaceTTTData,
    InPlaceTTTTransitionTask,
    INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION,
    INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION,
    INPLACE_TTT_PUBLISH_IMPLEMENTATION,
    INPLACE_TTT_TRANSITION_IMPLEMENTATION,
)
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTSpecialization,
    InPlaceTTTStepBatchResult,
    validate_inplace_ttt_step,
)
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.parameter_overlay import CapturedParameterOperand
from tllm.contracts.request_identity import RequestKeyField
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
    StateLayoutRequirement,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import (
    StateTensorLayout,
    TensorLayoutRequest,
)

_FAST_WEIGHT_STATE = StateResource("dense_fast_weights")
_PUBLISH_TASK = TaskSymbol("publish")
_TRANSITION_TASK = TaskSymbol("transition")


@dataclass(frozen=True, slots=True)
class InPlaceTTTMath:
    config: InPlaceTTTConfig
    specialization: InPlaceTTTSpecialization

    def initialize(self, roots: tuple[tuple[torch.Tensor, ...], ...]) -> None:
        # The arena's zero initialization is the mathematical initial state.
        pass

    def rows(self, value: tuple[CapturedParameterOperand, ...]) -> IndexedRequestIdentityBatch:
        identities = sorted(set(operand.identity for operand in value),
            key=lambda identity: (identity.request_id, identity.prompt_idx, identity.sample_idx))
        return IndexedRequestIdentityBatch(tuple(IndexedRequestIdentityRow(index, identity)
            for index, identity in enumerate(identities)))

    def prepare(self, value: tuple[CapturedParameterOperand, ...],
                context: StateTaskContext[InPlaceTTTData]) -> InPlaceTTTData:
        actual = tuple((operand.identity, operand.target) for operand in value)
        expected = {(row.identity, target) for row in context.selection.rows for target in self.config.targets}
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("captured operands must cover each selected request and target exactly once")
        validate_inplace_ttt_step(value, self.specialization)
        return InPlaceTTTData(value)

    def output(self, context: StateTaskContext[InPlaceTTTData]) -> InPlaceTTTStepBatchResult:
        result = context.data().result
        if result is None or context.write is not None:
            raise RuntimeError("signed dense overlay graph did not publish")
        return result


@dataclass(frozen=True, slots=True)
class InPlaceTTTCompilation:
    execution: StateExecution[tuple[CapturedParameterOperand, ...], InPlaceTTTData, InPlaceTTTStepBatchResult]
    plan_bundle: CanonicalPlanBundle
    state_layout: InPlaceTTTStateLayout
    math: InPlaceTTTMath


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
        arena = PageBundleArena(plan_bundle, candidates)
        directory = RequestStateDirectory(arena, graph.states[0])
        math = InPlaceTTTMath(self.config, specialization)
        publish_id, transition_id = (
            (INPLACE_TTT_PUBLISH_IMPLEMENTATION, INPLACE_TTT_TRANSITION_IMPLEMENTATION)
            if specialization.device_kind is StateDeviceKind.CPU else
            (INPLACE_TTT_CUDA_PUBLISH_IMPLEMENTATION, INPLACE_TTT_CUDA_TRANSITION_IMPLEMENTATION)
        )
        sampler_anchor, = (AnchorId(anchor.ordinal) for anchor in options.timeline.anchors
            if anchor.symbol == AnchorSymbol("sampler.complete"))
        execution = StateExecution(arena, directory, math,
            (InPlaceTTTPublishTask(publish_id), InPlaceTTTTransitionTask(self.config, state_layout, transition_id)),
            sampler_anchor,
        )
        return InPlaceTTTCompilation(execution, plan_bundle, state_layout, math)


__all__ = [
    "InPlaceTTTCompilation",
    "InPlaceTTTWorkload",
]

"""In-Place-TTT declaration and shared signed execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.state_execution import PublishStateTask, StateExecution, StateTaskContext
from tllm.canonical.backend_program import BackendCompileFacts, BackendDeployment, BackendRequirements, BackendWorkload, TensorStepInput
from tllm.canonical.backend_attachment import bind_backend_attachment
from tllm.canonical.ptts.page_bundle import PageBundleArena
from tllm.canonical.ptts.page_bundle_planning import compile_page_bundle_plan
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.canonical.workloads.inplace_ttt.layout import (
    InPlaceTTTStateLayout,
    build_inplace_ttt_state_layout,
)
from tllm.canonical.workloads.inplace_ttt.tasks import (
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
from tllm.contracts.backend_attachment import BackendAttachmentSpec, AttachmentKind, AttachmentTaskRole
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

    def rows(self, value: TensorStepInput) -> IndexedRequestIdentityBatch:
        identities = sorted(set(row.identity for row in value.rows.rows),
            key=lambda identity: (identity.request_id, identity.prompt_idx, identity.sample_idx))
        return IndexedRequestIdentityBatch(tuple(IndexedRequestIdentityRow(index, identity)
            for index, identity in enumerate(identities)))

    def prepare(self, value: TensorStepInput,
                context: StateTaskContext[InPlaceTTTData]) -> InPlaceTTTData:
        actual = tuple((operand.identity, operand.target) for operand in value.operands)
        expected = {(row.identity, target) for row in context.selection.rows for target in self.config.targets}
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("captured operands must cover each selected request and target exactly once")
        validate_inplace_ttt_step(value.operands, self.specialization)
        return InPlaceTTTData(value.operands)

    def output(self, context: StateTaskContext[InPlaceTTTData]) -> InPlaceTTTStepBatchResult:
        result = context.data().result
        if result is None or context.write is not None:
            raise RuntimeError("signed dense overlay graph did not publish")
        return result


@dataclass(frozen=True, slots=True)
class InPlaceTTTCompilation:
    execution: StateExecution[TensorStepInput, InPlaceTTTData, InPlaceTTTStepBatchResult]
    plan_bundle: CanonicalPlanBundle
    state_layout: InPlaceTTTStateLayout
    math: InPlaceTTTMath


class InPlaceTTTWorkload(BackendWorkload[InPlaceTTTData, InPlaceTTTStepBatchResult]):
    """Declare dense-fast-weight math/effects without physical policy."""

    __slots__ = ("config",)

    def __init__(self, config: InPlaceTTTConfig) -> None:
        self.config = config

    def backend_requirements(self) -> BackendRequirements:
        return BackendRequirements(parameter_targets=self.config.targets)

    def compile_backend(self, facts: BackendCompileFacts) -> BackendDeployment[InPlaceTTTData, InPlaceTTTStepBatchResult]:
        device = torch.device("cpu") if facts.probe.device_kind is StateDeviceKind.CPU else torch.device("cuda", facts.probe.device_ordinal)
        compiled = self.compile(
            InPlaceTTTSpecialization(facts.targets, facts.probe.tensor_dtype, device),
            InPlaceTTTCompileOptions(facts.timeline, facts.providers,
                facts.max_admitted_lineages, facts.state_memory_budget_bytes),
        )
        bindings = compiled.state_layout.overlay_bindings
        attachment = bind_backend_attachment(compiled.plan_bundle, facts.probe,
            BackendAttachmentSpec(AttachmentKind.PARAMETER_OVERLAY,
                ((AttachmentTaskRole.TRANSITION, _TRANSITION_TASK), (AttachmentTaskRole.PUBLISH, _PUBLISH_TASK)),
                AnchorSymbol("sampler.complete"), _FAST_WEIGHT_STATE, bindings))
        return BackendDeployment(compiled.execution, attachment)

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
        state_layout = build_inplace_ttt_state_layout(self.config, specialization)
        plan_bundle = compile_page_bundle_plan(graph, options.timeline,
            TensorLayoutRequest((StateTensorLayout(str(_FAST_WEIGHT_STATE), state_layout.tensor_layout),)),
            options.providers, options.max_admitted_lineages, options.state_memory_budget_bytes,
            specialization.device, plan_epoch=options.plan_epoch)
        arena = PageBundleArena(plan_bundle)
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
            (PublishStateTask[InPlaceTTTData](publish_id), InPlaceTTTTransitionTask(self.config, state_layout, transition_id)),
            sampler_anchor,
        )
        return InPlaceTTTCompilation(execution, plan_bundle, state_layout, math)


__all__ = [
    "InPlaceTTTCompilation",
    "InPlaceTTTWorkload",
]

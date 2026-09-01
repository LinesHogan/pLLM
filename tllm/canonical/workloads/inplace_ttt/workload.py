"""Clean-room In-Place-TTT declaration and signed CPU specialization."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.canonical.ptts.dense_page import DensePageStateProvider
from tllm.canonical.ptts.request_state import RequestStateDirectory
from tllm.canonical.ptts.publication import RequestStatePublicationCoordinator
from tllm.canonical.scheduler import compile_schedule, lower_physical_plan
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
from tllm.canonical.workloads.inplace_ttt.execution import InPlaceTTTExecution
from tllm.canonical.workloads.inplace_ttt.layout import (
    InPlaceTTTStateLayout,
    build_inplace_ttt_state_layout,
)
from tllm.canonical.workloads.inplace_ttt.tasks import (
    INPLACE_TTT_PUBLISH_IMPLEMENTATION,
    INPLACE_TTT_TRANSITION_IMPLEMENTATION,
)
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTSpecialization,
)
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.request_identity import RequestKeyField
from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    ExecutionKind,
    HostPayloadSchemaId,
    HostServiceId,
    HostServiceInputSource,
    HostServicePort,
    SemanticEdge,
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
from tllm.contracts.tensor_layout import StateTensorLayout, TensorLayoutRequest


_FAST_WEIGHT_STATE = StateResource("dense_fast_weights")
_PUBLISH_TASK = TaskSymbol("publish")
_TRANSITION_TASK = TaskSymbol("transition")


@dataclass(frozen=True, slots=True)
class InPlaceTTTCompilation:
    execution: InPlaceTTTExecution
    plan_bundle: CanonicalPlanBundle
    state_layout: InPlaceTTTStateLayout
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
                        AnchorSymbol("host.output.publish"),
                        AnchorSymbol("host.next"),
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
                        AnchorSymbol("host.input.ready"),
                        AnchorSymbol("host.output.publish"),
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
        representation_request = StateRepresentationRequest(
            memory_budget_bytes=options.state_memory_budget_bytes,
            states=(
                StateLayoutRequirement(
                    resource=str(_FAST_WEIGHT_STATE),
                    layout_id=layout_binding.layout_id,
                    layout_namespace=layout_binding.layout_namespace,
                    layout_digest=layout_binding.layout_digest,
                    max_admitted_lineages=options.max_admitted_lineages,
                    required_capabilities=(
                        "device/cpu@1",
                        "state/exact_versions@1",
                        "transition/full_root@1",
                    ),
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
        if selected_provider.device_kind is not StateDeviceKind.CPU:
            raise RuntimeError("In-Place-TTT CPU execution received another provider")
        provider = DensePageStateProvider(
            tensor_layout_request=layout_request,
            tensor_layout_decision=layout_decision,
            representation_request=representation_request,
            representation_decision=representation,
            physical_plan=state_physical,
            local_candidates=candidates,
            resource=str(_FAST_WEIGHT_STATE),
        )
        directory = RequestStateDirectory(
            layout_binding.layout_id,
            (
                RequestKeyField.REQUEST_ID,
                RequestKeyField.PROMPT_IDX,
                RequestKeyField.SAMPLE_IDX,
            ),
        )
        publication = RequestStatePublicationCoordinator(directory, provider)
        plan_bundle = CanonicalPlanBundle(
            graph,
            options.timeline,
            schedule,
            physical,
            representation_request,
            representation,
            state_physical,
        )
        execution = InPlaceTTTExecution(
            self.config,
            specialization,
            plan_bundle,
            state_layout,
            provider,
            directory,
            publication,
        )
        return InPlaceTTTCompilation(
            execution,
            plan_bundle,
            state_layout,
            execution_parameters_digest,
        )


__all__ = ["InPlaceTTTCompilation", "InPlaceTTTWorkload"]

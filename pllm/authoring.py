"""Typed authoring and invocation boundary shared with model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import torch

from pllm.config import ExecutionOptions
from pllm.contracts.auxiliary import (
    AuxiliaryInput as AuxiliaryInput,
    CausalTrainBatch as CausalTrainBatch,
    StateTensorUpdate as StateTensorUpdate,
    WeightedCEBatch as WeightedCEBatch,
    map_auxiliary as map_auxiliary,
)
from pllm.contracts.backend_attachment import BackendAttachment, BackendAttachmentSpec
from pllm.contracts.compilation import PlanBundle as PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase as HiddenPhase, HiddenSite as HiddenSite
from pllm.contracts.parameter_overlay import ParameterTarget
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage as SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    MathContract as MathContract,
    SchedulerTimeline,
    SemanticGraph,
    StateScope,
)
from pllm.contracts.tensor_layout import TensorLayoutRequest
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.backend_attachment import (
    attachment_buffer_bytes,
    bind_backend_attachment,
    derive_attachment_resources,
)
from pllm.execution.causal_batch import CausalBinder
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.declaration import (
    Builder as Builder,
    Call as Call,
    Dimensions as Dimensions,
    Field as Field,
    State as State,
    When as When,
    adam_fields as adam_fields,
    native as native,
)
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import compile_page_bundle_plan
from pllm.execution.state_execution import (
    MathBinding as MathBinding,
    StateExecution,
    StateTaskContext as StateTaskContext,
)
from pllm.execution.step_input import TensorStepInput
from pllm.execution.tensor_binding import TensorResult as TensorResult
from pllm.execution.tensor_program import (
    FunctionMath as FunctionMath,
    InvocationOutput as InvocationOutput,
    Observation as Observation,
    Roots as Roots,
    TaskOutput as TaskOutput,
    TensorBatch as TensorBatch,
    TensorOperation as TensorOperation,
)
from pllm.execution.trace import MAX_SNAPSHOT_BYTES as MAX_SNAPSHOT_BYTES
from pllm.kernels.adamw import AdamWConfig

DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
AdamW = AdamWConfig
Guidance = PackedSamplerGuidanceBatch
__all__ = [
    "AdamW",
    "Guidance",
    "Builder",
    "When",
    "Field",
    "State",
    "Call",
    "Dimensions",
    "MathContract",
    "FunctionMath",
    "TensorBatch",
    "TaskOutput",
    "InvocationOutput",
    "Observation",
    "MathBinding",
    "PlanBundle",
    "Roots",
    "HiddenPhase",
    "HiddenSite",
    "SamplerStage",
    "CausalTrainBatch",
    "StateTensorUpdate",
    "WeightedCEBatch",
    "native",
    "adam_fields",
    "TensorResult",
    "MAX_SNAPSHOT_BYTES",
]


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    hidden_inputs: tuple[HiddenInputSource, ...] = ()
    parameter_targets: tuple[ParameterTarget, ...] = ()
    activation_layers: tuple[int, ...] = ()
    sampler_stage: SamplerStage | None = None
    sampler_feedback: bool = False
    position_states: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class BackendDeployment(Generic[DataT, ResultT]):
    execution: StateExecution[TensorStepInput, DataT, ResultT]
    attachment: BackendAttachment

    def __post_init__(self) -> None:
        assert self.attachment.plan is self.execution.arena.plan


class BackendAuthoring(ABC):
    @abstractmethod
    def backend_requirements(self) -> BackendRequirements: ...

    def resource_facts(self, facts: BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline:
        if options.explicit_math:
            raise ValueError("this workload has no explicit tensor/capture implementation")
        return facts.timeline

    def specialize(self, facts: BackendCompileFacts, options: ExecutionOptions) -> BackendCompileFacts:
        from pllm.contracts.tensor_policy import TensorPolicy

        facts = replace(
            facts,
            timeline=replace(
                facts.timeline,
                capabilities=tuple(
                    sorted(
                        set(
                            (
                                *facts.timeline.capabilities,
                                *(("compute/cuda_graph@1",) if "compute/cuda@1" in facts.timeline.capabilities else ()),
                            )
                        )
                    )
                ),
                memory_budget_bytes=min(
                    (n for n in (facts.timeline.memory_budget_bytes, options.memory_budget_bytes) if n is not None),
                    default=None,
                ),
                tensor_policy=TensorPolicy(options.allow_numerical_transforms, options.schedule_policy, options.tuning),
            ),
        )
        if options.state_memory_bytes:
            if options.state_memory_bytes > facts.state_memory_budget_bytes:
                raise ValueError("requested state capacity exceeds available backend memory")
            facts = replace(facts, state_memory_budget_bytes=options.state_memory_bytes)
        timeline = self.resource_facts(facts, options)
        budgets = tuple(
            value
            for value in (facts.timeline.memory_budget_bytes, timeline.memory_budget_bytes, options.memory_budget_bytes)
            if value is not None
        )
        capabilities = tuple(
            sorted(
                set(
                    (
                        *timeline.capabilities,
                        *(("compute/explicit_math@1",) if options.explicit_math else ()),
                    )
                )
            )
        )
        timeline = replace(
            timeline,
            capabilities=capabilities,
            memory_budget_bytes=min(budgets, default=None),
            tensor_policy=facts.timeline.tensor_policy,
            visibility=VisibilitySource.post_sampler(options.publish_lag),
        )
        graph, layouts, attachment = self.declare(replace(facts, timeline=timeline))
        buffers = attachment_buffer_bytes(facts.probe, graph, layouts, attachment)
        reserved = (
            tuple(value + buffers for value in (timeline.reserved_bytes or (0,) * len(timeline.anchors)))
            if buffers
            else timeline.reserved_bytes
        )
        timeline = derive_attachment_resources(
            facts.probe,
            graph,
            replace(timeline, reserved_bytes=reserved),
            capture_memory_bytes=options.capture_memory_bytes,
        )
        return replace(facts, timeline=timeline)

    @abstractmethod
    def declare(self, facts: BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        """Return original graph, logical layouts and effects, never a physical plan."""
        ...

    def plan_backend(self, facts: BackendCompileFacts) -> BackendAttachment:
        graph, layouts, attachment = self.declare(facts)
        timeline = derive_attachment_resources(facts.probe, graph, facts.timeline)
        plan = compile_page_bundle_plan(
            (graph, layouts),
            timeline,
            facts.providers,
            facts.max_admitted_lineages if any(state.scope is StateScope.REQUEST for state in graph.states) else 1,
            facts.state_memory_budget_bytes,
            facts.device,
        )
        return bind_backend_attachment(plan, facts.probe, attachment)


class BackendWorkload(BackendAuthoring, Generic[DataT, ResultT]):
    @abstractmethod
    def bind_math(
        self, facts: BackendCompileFacts, plan: PlanBundle, *, causal: CausalBinder | None = None
    ) -> MathBinding[TensorStepInput, DataT, ResultT]:
        """Bind mathematics to selected implementations; no storage or execution ownership."""
        ...

    def bind_backend(
        self, facts: BackendCompileFacts, attachment: BackendAttachment, *, causal: CausalBinder | None = None
    ) -> BackendDeployment[DataT, ResultT]:
        plan = attachment.plan
        math = self.bind_math(facts, plan, causal=causal)
        device = (
            torch.device("cpu")
            if facts.probe.device_kind.value == "cpu"
            else torch.device("cuda", facts.probe.device_ordinal)
        )
        arena = PageBundleArena(plan, device=device, _owned_resources=math.local_resources(plan))
        return BackendDeployment(math.bind(arena, attachment.sampler_anchor_id), attachment)

    def compile_backend(
        self, facts: BackendCompileFacts, *, causal: CausalBinder | None = None
    ) -> BackendDeployment[DataT, ResultT]:
        return self.bind_backend(facts, self.plan_backend(facts), causal=causal)

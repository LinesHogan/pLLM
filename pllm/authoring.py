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
    WeightedRowBatch as WeightedRowBatch,
    map_auxiliary as map_auxiliary,
)
from pllm.contracts.backend_attachment import BackendAttachment
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenSite as HiddenSite
from pllm.contracts.parameter_overlay import ParameterTarget
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import HiddenInputSource, SchedulerTimeline, StateScope
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.backend_attachment import (
    attachment_buffer_bytes,
    bind_backend_attachment,
    derive_attachment_resources,
)
from pllm.execution.causal_batch import CausalBinder
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.declaration import Dimensions as Dimensions
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import compile_page_bundle_plan
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.step_input import TensorStepInput as TensorStepInput
from pllm.execution.tensor_binding import TensorBinding, TensorResult as TensorResult
from pllm.execution.tensor_program import FunctionMath as FunctionMath
from pllm.kernels.adamw import AdamWConfig

DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
AdamW = AdamWConfig
__all__ = [
    "AdamW",
    "AuxiliaryInput",
    "BackendDeployment",
    "BackendRequirements",
    "BackendWorkload",
    "CausalTrainBatch",
    "Dimensions",
    "FunctionMath",
    "HiddenSite",
    "StateTensorUpdate",
    "TensorResult",
    "TensorStepInput",
    "WeightedCEBatch",
    "WeightedRowBatch",
    "map_auxiliary",
]


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    hidden_inputs: tuple[HiddenInputSource, ...] = ()
    parameter_targets: tuple[ParameterTarget, ...] = ()
    activation_layers: tuple[int, ...] = ()
    sampler_stage: SamplerStage | None = None
    sampler_feedback: bool = False
    position_states: tuple[tuple[str, float], ...] = ()
    # The smallest pllm.samples(limit=); requests must use a lower sample index.
    sample_limit: int | None = None


@dataclass(frozen=True, slots=True)
class BackendDeployment(Generic[DataT, ResultT]):
    execution: StateExecution[TensorStepInput, DataT, ResultT]
    attachment: BackendAttachment

    def __post_init__(self) -> None:
        assert self.attachment.plan is self.execution.arena.plan


class BackendWorkload(ABC, Generic[DataT, ResultT]):
    """A workload compiled against a model backend: declare, plan, bind math, then deploy."""

    @abstractmethod
    def backend_requirements(self) -> BackendRequirements: ...

    @abstractmethod
    def resource_facts(self, facts: BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline: ...

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
        timeline = replace(
            timeline,
            memory_budget_bytes=min(budgets, default=None),
            tensor_policy=facts.timeline.tensor_policy,
            visibility=VisibilitySource.post_sampler(options.publish_lag),
        )
        graph, layouts, attachment = self.declaration(replace(facts, timeline=timeline)).finish()
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
    def declaration(self, facts: BackendCompileFacts, *, causal: CausalBinder | None = None) -> TensorBinding:
        """Declare the graph, logical layouts, effects and tensor tasks; never a physical plan."""
        ...

    def plan_backend(
        self, facts: BackendCompileFacts, *, causal: CausalBinder | None = None
    ) -> tuple[BackendAttachment, TensorBinding]:
        """Plan one declaration and return it with the attachment, so binding reuses what was planned."""
        declaration = self.declaration(facts, causal=causal)
        graph, layouts, attachment = declaration.finish()
        timeline = derive_attachment_resources(facts.probe, graph, facts.timeline)
        plan = compile_page_bundle_plan(
            (graph, layouts),
            timeline,
            facts.providers,
            facts.max_admitted_lineages if any(state.scope is StateScope.REQUEST for state in graph.states) else 1,
            facts.state_memory_budget_bytes,
            facts.device,
        )
        return bind_backend_attachment(plan, facts.probe, attachment), declaration

    @abstractmethod
    def bind_math(
        self, facts: BackendCompileFacts, plan: PlanBundle, declaration: TensorBinding
    ) -> MathBinding[TensorStepInput, DataT, ResultT]:
        """Bind the planned declaration to selected implementations; no storage or execution ownership."""
        ...

    def bind_backend(
        self, facts: BackendCompileFacts, attachment: BackendAttachment, declaration: TensorBinding
    ) -> BackendDeployment[DataT, ResultT]:
        plan = attachment.plan
        math = self.bind_math(facts, plan, declaration)
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
        return self.bind_backend(facts, *self.plan_backend(facts, causal=causal))

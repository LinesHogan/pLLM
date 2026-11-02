"""Typed authoring and invocation boundary shared with model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import BackendAttachment, ModelProbe
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.host_batch import (
    CausalTrainBatch as CausalTrainBatch,
    HostBatch as HostBatch,
    StateImport as StateImport,
    WeightedCEBatch as WeightedCEBatch,
    WeightedRowBatch as WeightedRowBatch,
    map_host_batch as map_host_batch,
)
from pllm.contracts.parameter_overlay import Projection
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
from pllm.execution.step_input import StepInput as StepInput
from pllm.execution.tensor_binding import StepResult as StepResult, TensorBinding
from pllm.execution.tensor_program import FunctionMath as FunctionMath
from pllm.kernels.adamw import AdamWConfig

DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
AdamW = AdamWConfig
__all__ = [
    "AdamW",
    "HostBatch",
    "BackendDeployment",
    "BackendRequirements",
    "BackendWorkload",
    "CausalTrainBatch",
    "Dimensions",
    "FunctionMath",
    "StateImport",
    "StepResult",
    "StepInput",
    "WeightedCEBatch",
    "WeightedRowBatch",
    "map_host_batch",
]


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    checkpoint_names: tuple[str, ...] = field(default=(), kw_only=True)
    exact_decode: bool = field(default=False, kw_only=True)
    hidden_inputs: tuple[HiddenInputSource, ...] = ()
    parameter_targets: tuple[Projection, ...] = ()
    activation_layers: tuple[int, ...] = ()
    sampler_stage: SamplerStage | None = None
    sampler_feedback: bool = False
    position_states: tuple[tuple[str, float], ...] = ()
    # Whether a @pllm.custom_forward reruns model layers, which vLLM's CUDA graphs cannot capture.
    custom_forward: bool = False
    # Whether a pllm.model.Adapter reads an InstanceState; CUDA-graph overlays bind request state only.
    instance_adapter: bool = False
    causal_training: bool = False
    feeds: bool = False
    # The smallest pllm.sample_index(limit=); requests must use a lower sample index.
    sample_limit: int | None = None


@dataclass(frozen=True, slots=True)
class BackendDeployment(Generic[DataT, ResultT]):
    execution: StateExecution[StepInput, DataT, ResultT]
    attachment: BackendAttachment

    def __post_init__(self) -> None:
        assert self.attachment.plan is self.execution.arena.plan


class BackendWorkload(ABC, Generic[DataT, ResultT]):
    """A workload compiled against a model backend: declare, plan, bind math, then deploy."""

    @abstractmethod
    def backend_requirements(self) -> BackendRequirements: ...

    @abstractmethod
    def resource_facts(
        self, declaration: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions
    ) -> SchedulerTimeline:
        """Measure the declaration's task resources and reserve its retained state templates."""
        ...

    def specialize(
        self, facts: BackendCompileFacts, options: ExecutionOptions, *, causal: CausalBinder | None = None
    ) -> tuple[BackendCompileFacts, TensorBinding]:
        """Measure and fund the declaration on this backend, then declare against the funded facts as workers do;
        planning and binding reuse both."""
        facts = _apply_options(facts, options)
        measured = self.build(facts, causal=causal)
        timeline = _fund_attachment(measured, facts.probe, self.resource_facts(measured, facts, options), options)
        facts = replace(facts, timeline=timeline)
        return facts, self.funded_declaration(facts, causal=causal)

    def funded_declaration(self, facts: BackendCompileFacts, *, causal: CausalBinder | None = None) -> TensorBinding:
        """The declaration on funded facts: tensor tasks funded resident memory run as captured CUDA graphs. The
        driver and every worker process declare here, so their graphs agree."""
        declaration = self.build(facts, causal=causal)
        declaration.select_capture(facts.timeline.task_resources)
        return declaration

    @abstractmethod
    def build(self, facts: BackendCompileFacts, *, causal: CausalBinder | None = None) -> TensorBinding:
        """Declare the graph, logical layouts, effects and tensor tasks; never a physical plan."""
        ...

    def plan_backend(self, facts: BackendCompileFacts, declaration: TensorBinding) -> BackendAttachment:
        """Plan the specialized declaration; binding reuses the same declaration."""
        graph, layouts, attachment = declaration.finish()
        plan = compile_page_bundle_plan(
            (graph, layouts),
            facts.timeline,
            facts.providers,
            facts.max_admitted_lineages if any(state.scope is StateScope.REQUEST for state in graph.states) else 1,
            facts.state_memory_budget_bytes,
            facts.device,
        )
        return bind_backend_attachment(plan, facts.probe, attachment)

    @abstractmethod
    def bind_math(
        self, facts: BackendCompileFacts, plan: PlanBundle, declaration: TensorBinding
    ) -> MathBinding[StepInput, DataT, ResultT]:
        """Bind the planned declaration to selected implementations; no storage or execution ownership."""
        ...

    def bind_backend(
        self,
        facts: BackendCompileFacts,
        attachment: BackendAttachment,
        declaration: TensorBinding,
        *,
        is_output_rank: bool = True,
    ) -> BackendDeployment[DataT, ResultT]:
        plan = attachment.plan
        math = self.bind_math(facts, plan, declaration)
        if not is_output_rank:
            math = replace(math, on_export=None)
        arena = PageBundleArena(plan, device=facts.device, _owned_resources=math.local_resources(plan))
        return BackendDeployment(math.bind(arena, attachment.sampler_anchor_id), attachment)

    def compile_backend(
        self,
        facts: BackendCompileFacts,
        options: ExecutionOptions = ExecutionOptions(),
        *,
        causal: CausalBinder | None = None,
    ) -> BackendDeployment[DataT, ResultT]:
        """Specialize, plan and bind: the whole compile on probed backend facts."""
        facts, declaration = self.specialize(facts, options, causal=causal)
        return self.bind_backend(facts, self.plan_backend(facts, declaration), declaration)


def _apply_options(facts: BackendCompileFacts, options: ExecutionOptions) -> BackendCompileFacts:
    """The execution options the declaration and plan depend on: capabilities, budgets, tensor policy, visibility."""
    from pllm.contracts.tensor_policy import TensorPolicy

    if options.state_memory_bytes > facts.state_memory_budget_bytes:
        raise ValueError(
            f"ExecutionOptions.state_memory_bytes={options.state_memory_bytes} exceeds the "
            f"{facts.state_memory_budget_bytes} bytes free on the device; lower it, or lower gpu_memory_utilization "
            "so vLLM leaves more memory free"
        )
    capabilities = set(facts.timeline.capabilities)
    if "compute/cuda@1" in capabilities:
        capabilities.add("compute/cuda_graph@1")
    timeline = replace(
        facts.timeline,
        capabilities=tuple(sorted(capabilities)),
        memory_budget_bytes=min(
            (n for n in (facts.timeline.memory_budget_bytes, options.memory_budget_bytes) if n is not None),
            default=None,
        ),
        tensor_policy=TensorPolicy(options.allow_numerical_transforms, options.tuning),
        visibility=VisibilitySource.post_sampler(options.publish_lag),
    )
    return replace(
        facts,
        timeline=timeline,
        state_memory_budget_bytes=options.state_memory_bytes or facts.state_memory_budget_bytes,
    )


def _fund_attachment(
    declaration: TensorBinding, probe: ModelProbe, timeline: SchedulerTimeline, options: ExecutionOptions
) -> SchedulerTimeline:
    """Reserve the attachment's effect buffers and fund the custom-forward task."""
    graph, layouts, attachment = declaration.finish()
    buffers = attachment_buffer_bytes(probe, graph, layouts, attachment)
    if buffers:
        timeline = replace(
            timeline,
            reserved_bytes=tuple(
                value + buffers for value in (timeline.reserved_bytes or (0,) * len(timeline.anchors))
            ),
        )
    return derive_attachment_resources(probe, graph, timeline, capture_memory_bytes=options.model_capture_bytes)

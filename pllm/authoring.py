"""Typed authoring and invocation boundary shared with model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeVar

import torch
from pllm.execution.declaration import (
    Builder as Builder, Call as Call, Dimensions as Dimensions, Field as Field, State as State, When as When,
    adam_fields as adam_fields, native as native,
)
from pllm.execution.tensor_program import (
    FunctionMath as FunctionMath, InvocationOutput as InvocationOutput, Roots as Roots, Observation as Observation,
    TaskOutput as TaskOutput, TensorBatch as TensorBatch, TensorOperation as TensorOperation,
)
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.backend_attachment import attachment_buffer_bytes, bind_backend_attachment, derive_attachment_resources, validate_backend_attachment_physical
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.state_execution import MathBinding as MathBinding, StateExecution, StateTaskContext as StateTaskContext
from pllm.contracts.backend_attachment import BackendAttachmentSpec, BackendPlacementProbe, SignedBackendAttachment
from pllm.contracts.compilation import PlanBundle as PlanBundle
from pllm.kernels.adamw import AdamWConfig
from pllm.contracts.hidden_locator import HiddenPhase as HiddenPhase, HiddenSite as HiddenSite
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerStage as SamplerStage
from pllm.contracts.tensor_layout import TensorDType, TensorLayoutRequest
from pllm.contracts.packed_hidden import PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.parameter_overlay import ParameterTarget, CapturedParameterOperand
from pllm.contracts.semantic_graph import HiddenInputSource, SchedulerTimeline, SemanticGraph, StateScope, TaskResourceFact, ValueResourceFact
from pllm.contracts.auxiliary import (AuxiliaryInput as AuxiliaryInput, CausalTrainBatch as CausalTrainBatch,
    StateTensorUpdate as StateTensorUpdate, WeightedCEBatch as WeightedCEBatch, map_auxiliary as map_auxiliary)
from pllm.config import ExecutionOptions
from pllm.contracts.semantic_graph import MathContract as MathContract
from pllm.execution.tensor_binding import TensorResult as TensorResult
from pllm.execution.trace import MAX_SNAPSHOT_BYTES as MAX_SNAPSHOT_BYTES


DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
AdamW = AdamWConfig
Guidance = PackedSamplerGuidanceBatch
__all__ = ["AdamW", "Guidance", "Builder", "When", "Field", "State", "Call",
    "Dimensions", "MathContract", "FunctionMath", "TensorBatch", "TaskOutput", "InvocationOutput", "Observation",
    "BackendCompileFacts", "TensorStepInput", "MathBinding", "PlanBundle", "Roots", "HiddenPhase", "HiddenSite",
    "SamplerStage", "CausalTrainBatch", "StateTensorUpdate", "WeightedCEBatch", "native", "adam_fields",
    "TensorResult",
    "MAX_SNAPSHOT_BYTES"]


@dataclass(frozen=True, slots=True)
class TensorStepInput:
    rows: PackedPhaseRowBatch
    sampler_rows: PackedSamplerRowBatch
    hidden: PackedHiddenSnapshotBatch | None = None
    operands: tuple[CapturedParameterOperand, ...] = ()
    auxiliary: AuxiliaryInput[torch.Tensor] | None = None
    hidden_readiness: CudaStreamReadiness | None = None

    def __post_init__(self) -> None:
        if self.sampler_rows.row_batch is not self.rows:
            raise ValueError("tensor input requires the original packed row provenance")
        if self.hidden is None:
            if self.hidden_readiness is not None:
                raise ValueError("hidden readiness requires its original snapshots")
        else:
            if self.hidden.row_batch is not self.rows:
                raise ValueError("hidden input belongs to another packed step")
            device = self.hidden.snapshots[0].hidden.device
            if ((device.type == "cuda") != (self.hidden_readiness is not None)
                or self.hidden_readiness is not None and self.hidden_readiness.device != device):
                raise ValueError("hidden snapshots require readiness from their original device")

    def hidden_batch(self, stream: torch.cuda.Stream | None) -> PackedHiddenSnapshotBatch:
        if self.hidden is None or (self.hidden_readiness is None) != (stream is None):
            raise ValueError("hidden snapshots differ from the input device")
        if self.hidden_readiness is not None and stream is not None:
            self.hidden_readiness.order_on(stream)
            for snapshot in self.hidden.snapshots:
                snapshot.hidden.record_stream(stream)
        return self.hidden


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    hidden_inputs: tuple[HiddenInputSource, ...] = ()
    parameter_targets: tuple[ParameterTarget, ...] = ()
    model_source: tuple[str, str] | None = None
    activation_layers: tuple[int, ...] = ()
    sampler_stage: SamplerStage | None = None
    sampler_feedback: bool = False
    resident_bytes: int = 0

    def __post_init__(self) -> None:
        if self.resident_bytes < 0:
            raise ValueError("frozen dependency residency must be nonnegative")


@dataclass(frozen=True, slots=True)
class ParameterTargetShape:
    target: ParameterTarget
    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        if self.input_size < 1 or self.output_size < 1:
            raise ValueError("parameter target dimensions must be positive")


class CausalModel(Protocol):
    """Frozen base mathematics and factor templates; live factors belong to PTTS."""

    @property
    def targets(self) -> tuple[ParameterTargetShape, ...]: ...
    @property
    def initial_factors(self) -> tuple[torch.Tensor, ...]: ...
    @property
    def tensor_dtype(self) -> TensorDType: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def fingerprint(self) -> str: ...
    @property
    def scale(self) -> float: ...
    @property
    def model_source(self) -> tuple[str, str]: ...
    @property
    def resident_bytes(self) -> int: ...
    def logits(self, batch: CausalTrainBatch[torch.Tensor], factors: tuple[torch.Tensor, ...] | None) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class BackendCompileFacts:
    probe: BackendPlacementProbe
    timeline: SchedulerTimeline
    providers: tuple[PageBundleProviderFacts, ...]
    max_admitted_lineages: int
    state_memory_budget_bytes: int
    initialization_seed: int
    targets: tuple[ParameterTargetShape, ...] = ()
    vocab_size: int = 0
    layer_count: int = 0

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.probe.device_ordinal) if self.probe.device_kind.value == "cuda" else torch.device("cpu")


@dataclass(frozen=True, slots=True)
class BackendDeployment(Generic[DataT, ResultT]):
    execution: StateExecution[TensorStepInput, DataT, ResultT]
    attachment: SignedBackendAttachment

    def __post_init__(self) -> None:
        if self.attachment.inputs.plan is not self.execution.arena.plan:
            raise ValueError("backend deployment requires the original execution plan")


_ZERO_WORKSPACE_IMPLEMENTATIONS = frozenset({
    # Framework-owned canonical import/publish tasks: copies land directly in allocated
    # private pages and publication moves metadata, so their derived workspace is zero.
    "canonical.overlay:import@1", "canonical.activation:import@1",
    "canonical.state:publish@1", "canonical.state:publish_parameter_overlay@1",
    "canonical.state:cuda_publish_parameter_overlay@1",
})


def derive_framework_resources(graph: SemanticGraph, timeline: SchedulerTimeline) -> SchedulerTimeline:
    """Fill resource facts for framework-owned canonical tasks the timeline omits.

    Memory-constrained AER requires a fact per task and per output; users cost their own
    mathematics, never the framework's import/publish plumbing (whose copies land directly
    in allocated private pages and whose receipts are host metadata).
    """
    canonical = tuple(task for task in graph.tasks
        if task.implementations and all(
            implementation.implementation_id in _ZERO_WORKSPACE_IMPLEMENTATIONS
            for implementation in task.implementations))
    if not canonical:
        return timeline
    known_tasks = {fact.task for fact in timeline.task_resources}
    known_values = {(fact.task, fact.output) for fact in timeline.value_resources}
    tasks = tuple(TaskResourceFact(task.symbol, 0) for task in canonical if task.symbol not in known_tasks)
    values = tuple(ValueResourceFact(task.symbol, output.name, 0) for task in canonical
        for output in task.outputs if (task.symbol, output.name) not in known_values)
    if not tasks and not values:
        return timeline
    return replace(timeline,
        task_resources=tuple(sorted((*timeline.task_resources, *tasks), key=lambda fact: fact.task)),
        value_resources=tuple(sorted((*timeline.value_resources, *values), key=lambda fact: (fact.task, fact.output))))


def strip_framework_resources(graph: SemanticGraph, timeline: SchedulerTimeline) -> SchedulerTimeline:
    """Remove exactly the facts derive_framework_resources would add.

    Worker bindings recompute minimum costs from the signed plan timeline, which already
    carries the derived zero-cost canonical facts; the driver decided from the raw probe
    timeline, so the recompute must see the same view or cost guards diverge across the
    process boundary.
    """
    canonical = {task.symbol for task in graph.tasks
        if task.implementations and all(
            implementation.implementation_id in _ZERO_WORKSPACE_IMPLEMENTATIONS
            for implementation in task.implementations)}
    if not canonical:
        return timeline
    return replace(timeline,
        task_resources=tuple(fact for fact in timeline.task_resources
            if fact.task not in canonical or fact != TaskResourceFact(fact.task, 0)),
        value_resources=tuple(fact for fact in timeline.value_resources
            if fact.task not in canonical or fact.bytes != 0))


class BackendAuthoring(ABC):
    @abstractmethod
    def backend_requirements(self) -> BackendRequirements: ...

    def resource_facts(self, facts: BackendCompileFacts, options: ExecutionOptions) -> SchedulerTimeline:
        if options.capture or options.explicit_math:
            raise ValueError("this workload has no explicit tensor/capture implementation")
        return facts.timeline

    def specialize(self, facts: BackendCompileFacts, options: ExecutionOptions) -> BackendCompileFacts:
        from pllm.contracts.region_selection import RegionSelectionPolicy
        facts = replace(facts, timeline=replace(facts.timeline,
            region_policy=RegionSelectionPolicy(options.allow_numerical_transforms, options.schedule_policy)))
        if options.state_memory_bytes:
            if options.state_memory_bytes > facts.state_memory_budget_bytes:
                raise ValueError("requested state capacity exceeds available backend memory")
            facts = replace(facts, state_memory_budget_bytes=options.state_memory_bytes)
        timeline = self.resource_facts(facts, options)
        budgets = tuple(value for value in (facts.timeline.memory_budget_bytes, timeline.memory_budget_bytes,
            options.memory_budget_bytes) if value is not None)
        if options.capture and facts.device.type != "cuda":
            raise ValueError("capture requires CUDA")
        capabilities = tuple(sorted(set((*timeline.capabilities,
            *(("compute/cuda_graph@1",) if options.capture else ()),
            *(("compute/explicit_math@1",) if options.explicit_math else ())))))
        timeline = replace(timeline, capabilities=capabilities, memory_budget_bytes=min(budgets, default=None),
                           region_policy=facts.timeline.region_policy)
        graph, layouts, attachment = self.declare(replace(facts, timeline=timeline))
        timeline = derive_framework_resources(graph, timeline)
        buffers = attachment_buffer_bytes(facts.probe, graph, layouts, attachment)
        reserved = (tuple(value + buffers for value in (timeline.reserved_bytes or (0,) * len(timeline.anchors)))
            if buffers else timeline.reserved_bytes)
        timeline = derive_attachment_resources(facts.probe, graph, replace(timeline, reserved_bytes=reserved))
        return replace(facts, timeline=timeline)

    @abstractmethod
    def declare(self, facts: BackendCompileFacts) -> tuple[SemanticGraph, TensorLayoutRequest, BackendAttachmentSpec]:
        """Return original graph, logical layouts and effects, never a physical plan."""
        ...

    def plan_backend(self, facts: BackendCompileFacts) -> SignedBackendAttachment:
        graph, layouts, attachment = self.declare(facts)
        timeline = derive_attachment_resources(facts.probe, graph, derive_framework_resources(graph, facts.timeline))
        plan = compile_page_bundle_plan((graph, layouts), timeline, facts.providers,
            facts.max_admitted_lineages if any(state.scope is StateScope.REQUEST for state in graph.states) else 1,
            facts.state_memory_budget_bytes, facts.device)
        return bind_backend_attachment(plan, facts.probe, attachment)


class BackendWorkload(BackendAuthoring, Generic[DataT, ResultT]):
    @abstractmethod
    def bind_math(self, facts: BackendCompileFacts, plan: PlanBundle) -> MathBinding[TensorStepInput, DataT, ResultT]:
        """Bind mathematics to selected implementations; no storage or execution ownership."""
        ...

    def minimum_resources(self, facts: BackendCompileFacts, plan: PlanBundle) -> SchedulerTimeline:
        """Recompute the driver's minimum-cost view against a signed plan.

        The signed timeline carries framework-derived zero-cost canonical facts the driver
        never saw when it declared costs; strip exactly those so cost guards inside
        resource_facts decide identically on driver and worker.
        """
        return self.resource_facts(replace(facts,
            timeline=strip_framework_resources(plan.semantic_graph, facts.timeline)), ExecutionOptions())

    @staticmethod
    def require_resources(plan: PlanBundle, minimum: SchedulerTimeline) -> None:
        """Check implementation costs against the original signed facts, without replanning."""
        tasks = {item.task: item for item in plan.timeline.task_resources}
        for required in minimum.task_resources:
            actual = tasks.get(required.task)
            if (actual is None or actual.input_shapes != required.input_shapes
                or actual.workspace_bytes < required.workspace_bytes or actual.resident_bytes < required.resident_bytes):
                raise ValueError(f"{required.task} workspace/residency differs from its signed envelope")
        values = {(item.task, item.output): item.bytes for item in plan.timeline.value_resources}
        for required in minimum.value_resources:
            if values.get((required.task, required.output), -1) < required.bytes:
                raise ValueError(f"{required.task}.{required.output} output storage is smaller than its signed envelope")

    def bind_backend(self, facts: BackendCompileFacts, attachment: SignedBackendAttachment) -> BackendDeployment[DataT, ResultT]:
        plan = attachment.inputs.plan
        if (attachment.inputs.probe != facts.probe
            or plan.timeline != derive_attachment_resources(facts.probe, plan.semantic_graph,
                derive_framework_resources(plan.semantic_graph, facts.timeline))):
            raise ValueError("worker backend facts differ from the original signed attachment")
        if (plan.state_decision.reserved_bytes > facts.state_memory_budget_bytes
            or any(item.max_admitted_lineages > facts.max_admitted_lineages for item in plan.state_request.states)):
            raise ValueError("worker capacity is smaller than the signed entitlement")
        candidates = tuple(provider.materialize_plan(resource=entry.resource, layout=entry.layout, binding=binding,
            max_admitted_lineages=requirement.max_admitted_lineages)
            for entry, binding, requirement in zip(plan.tensor_layout_request.layouts,
                plan.tensor_layout_decision.bindings, plan.state_request.states, strict=True)
            for provider in facts.providers)
        validate_backend_attachment_physical(attachment.inputs, attachment.decision, attachment.physical)
        math = self.bind_math(facts, plan)
        device = torch.device("cpu") if facts.probe.device_kind.value == "cpu" else torch.device("cuda", facts.probe.device_ordinal)
        arena = PageBundleArena(plan, candidates, device=device)
        return BackendDeployment(math.bind(arena, attachment.inputs.request.sampler_anchor_id), attachment)

    def compile_backend(self, facts: BackendCompileFacts) -> BackendDeployment[DataT, ResultT]:
        return self.bind_backend(facts, self.plan_backend(facts))

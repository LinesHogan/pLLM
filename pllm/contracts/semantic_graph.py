"""Shared immutable graph/plan data; Rust checks the declaration facts a user can get wrong at compilation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, NewType

from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.tensor_policy import TensorPolicy
from pllm.contracts.visibility import VisibilitySource

WorkloadId = NewType("WorkloadId", str)
TaskSymbol = NewType("TaskSymbol", str)
ImplementationId = NewType("ImplementationId", str)
AnchorSymbol = NewType("AnchorSymbol", str)
TaskId = NewType("TaskId", int)
AnchorId = NewType("AnchorId", int)
StateResource = NewType("StateResource", str)
HostServiceId = NewType("HostServiceId", str)
HostServicePort = NewType("HostServicePort", str)
HostPayloadSchemaId = NewType("HostPayloadSchemaId", str)


class StateScope(StrEnum):
    REQUEST = "request"
    INSTANCE = "instance"


class SemanticOutputKind(StrEnum):
    HOST = "host"
    TEMPORARY = "temporary"
    SAMPLER = "sampler"


class StateEffectKind(StrEnum):
    READ = "read"
    TRANSITION = "transition"
    TRANSITION_MAYBE = "transition_maybe"
    UPDATE = "update"
    UPDATE_MAYBE = "update_maybe"
    PUBLISH = "publish"


@dataclass(frozen=True, slots=True)
class HiddenInputSource:
    layer: int
    site: HiddenSite
    phase: HiddenPhase


@dataclass(frozen=True, slots=True)
class InputTokensInputSource:
    """Current and preceding request ids, available before the model forward."""

    history: int


@dataclass(frozen=True, slots=True)
class StepIdInputSource:
    pass


@dataclass(frozen=True, slots=True)
class PackedRowsInputSource:
    pass


@dataclass(frozen=True, slots=True)
class PackedSamplerRowsInputSource:
    pass


@dataclass(frozen=True, slots=True)
class SamplerFeedbackInputSource:
    pass


@dataclass(frozen=True, slots=True)
class HostServiceInputSource:
    service: HostServiceId
    port: HostServicePort
    schema: HostPayloadSchemaId


@dataclass(frozen=True, slots=True)
class StateSelectionInputSource:
    resource: StateResource
    phase: HiddenPhase
    metadata: bool = False


@dataclass(frozen=True, slots=True)
class TaskResultInputSource:
    task: TaskSymbol
    output: str


SemanticInputSource = (
    HiddenInputSource
    | InputTokensInputSource
    | StepIdInputSource
    | PackedRowsInputSource
    | PackedSamplerRowsInputSource
    | SamplerFeedbackInputSource
    | HostServiceInputSource
    | StateSelectionInputSource
    | TaskResultInputSource
)


@dataclass(frozen=True, slots=True)
class SemanticInput:
    name: str
    source: SemanticInputSource


@dataclass(frozen=True, slots=True)
class SemanticOutput:
    name: str
    kind: SemanticOutputKind = SemanticOutputKind.HOST
    schema: HostPayloadSchemaId = HostPayloadSchemaId("")


@dataclass(frozen=True, slots=True)
class SemanticState:
    resource: StateResource
    scope: StateScope
    key_fields: tuple[RequestKeyField, ...]
    immutable: bool = False


@dataclass(frozen=True, slots=True)
class SemanticEffect:
    resource: StateResource
    kind: StateEffectKind
    # None names the invocation's input version, not an implicit mutable latest.
    version: TaskSymbol | None = None


@dataclass(frozen=True, slots=True)
class TimingConstraint:
    after: AnchorSymbol
    before: AnchorSymbol


@dataclass(frozen=True, slots=True)
class TaskImplementationSpec:
    implementation_id: ImplementationId
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenProgramEffect:
    """A token program's model domain and selected rows; mathematics lives in its traced source."""

    layer_count: int
    rows: Literal["prefill", "decode", "all"] = "all"
    position_states: tuple[tuple[str, float], ...] = ()
    digest: str = ""
    role_sensitive: bool = False
    apply_updates: bool = False
    span: bool = False
    adaptive: bool = False


# How a request-training loss averages its rows: the author's @pllm.train choice ("mean", "state_mean"), or "none".
# Batch training still sets "custom" (its method reduces the loss); nothing reads it.
LossReduction = Literal["none", "mean", "state_mean", "custom"]
# What a workload's @pllm.train method trains on, read from its inputs: the rows of running requests, a host batch
# with one AdamW step, or a WeightedCEBatch with SGD steps per state.
TrainingMode = Literal["request", "batch", "weighted_batch"]


@dataclass(frozen=True, slots=True)
class MathContract:
    """Explicit differentiation and row domains of a native mathematical operation."""

    input_gradients: tuple[str, ...] = ()
    parameter_gradients: tuple[StateResource, ...] = ()
    detached_inputs: tuple[str, ...] = ()
    forward_rows: str = ""
    input_gradient_rows: str = ""
    parameter_gradient_rows: str = ""
    update_rows: str = ""
    reduction: LossReduction = "none"


@dataclass(frozen=True, slots=True)
class SemanticTask:
    symbol: TaskSymbol
    timing: TimingConstraint
    inputs: tuple[SemanticInput, ...]
    outputs: tuple[SemanticOutput, ...]
    implementations: tuple[TaskImplementationSpec, ...]
    effects: tuple[SemanticEffect, ...] = ()
    custom_forward: TokenProgramEffect | None = None
    math: MathContract | None = None
    visibility: VisibilitySource | None = None


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    source: TaskSymbol
    target: TaskSymbol
    step_distance: int = 0
    separation: int = 0


@dataclass(frozen=True, slots=True)
class ConditionalInputGroup:
    """Atomic producer closure; absent input reads the invocation's pinned version.

    Existing edges/effects describe the complete-input branch and conservatively
    schedule both branches. Only the named readers may omit publication inputs.
    """

    resource: StateResource
    borrowed_resources: tuple[StateResource, ...]
    tasks: tuple[TaskSymbol, ...]
    publication: TaskSymbol
    readers: tuple[TaskSymbol, ...]
    allow_absent: bool
    held_version: Literal["input"] = "input"


@dataclass(frozen=True, slots=True)
class SemanticGraph:
    workload_id: WorkloadId
    tasks: tuple[SemanticTask, ...]
    execution_parameters_digest: str
    states: tuple[SemanticState, ...] = ()
    edges: tuple[SemanticEdge, ...] = ()
    input_groups: tuple[ConditionalInputGroup, ...] = ()
    model_reads: tuple[StateResource, ...] = ()
    training_mode: TrainingMode | None = None

    @property
    def input_token_history(self) -> int:
        """Largest token lookback (-1 when the graph does not read token ids)."""
        return max(
            (
                port.source.history
                for task in self.tasks
                for port in task.inputs
                if isinstance(port.source, InputTokensInputSource)
            ),
            default=-1,
        )

    def host_state(self, schema: str) -> SemanticState | None:
        """The state replaced by a host input schema, or None when no such input is declared."""
        resources = {
            effect.resource
            for task in self.tasks
            if any(
                isinstance(port.source, HostServiceInputSource) and port.source.schema == schema for port in task.inputs
            )
            for effect in task.effects
            if effect.kind is StateEffectKind.TRANSITION_MAYBE
        }
        return next((state for state in self.states if state.resource in resources), None)

    @property
    def trains_on_batch(self) -> bool:
        """Whether the workload trains on a host CausalTrainBatch or WeightedRowBatch (declare_batch_train)."""
        return self.training_mode == "batch"

    @property
    def custom_forward_task(self) -> SemanticTask | None:
        """The custom-forward task; a graph declares at most one."""
        return next((task for task in self.tasks if task.custom_forward is not None), None)

    @property
    def updates_in_forward(self) -> bool:
        """Some task applies state updates inside the model forward (an Exact overlay)."""
        return any(task.visibility is not None and task.visibility.in_forward for task in self.tasks)


@dataclass(frozen=True, slots=True)
class TimelineAnchor:
    symbol: AnchorSymbol
    ordinal: int


@dataclass(frozen=True, slots=True)
class ValueResourceFact:
    task: TaskSymbol
    output: str
    bytes: int


@dataclass(frozen=True, slots=True)
class TaskResourceFact:
    task: TaskSymbol
    workspace_bytes: int
    resident_bytes: int = 0
    # Fixed cold buffers; native bindings define which input axes may be padded.
    input_shapes: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class SchedulerTimeline:
    anchors: tuple[TimelineAnchor, ...]
    capabilities: tuple[str, ...] = ()
    value_resources: tuple[ValueResourceFact, ...] = ()
    task_resources: tuple[TaskResourceFact, ...] = ()
    memory_budget_bytes: int | None = None
    reserved_bytes: tuple[int, ...] = ()
    tensor_policy: TensorPolicy = TensorPolicy()
    visibility: VisibilitySource | None = None


@dataclass(frozen=True, slots=True)
class PhysicalTask:
    task_id: TaskId
    implementation_id: ImplementationId
    anchor_id: AnchorId
    release_values: tuple[int, ...] = ()
    lane: Literal["main", "side"] = "main"
    read_origin: Literal["entry", "boundary", "latest"] = "entry"
    read_distance: int = 0


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    """Rust-selected task placements and their peak memory usage."""

    tasks: tuple[PhysicalTask, ...]
    peak_memory_bytes: int
    recompute_rejection: str | None


__all__ = [
    "TaskResourceFact",
    "ValueResourceFact",
    "AnchorId",
    "AnchorSymbol",
    "HiddenInputSource",
    "InputTokensInputSource",
    "HostServiceId",
    "HostServiceInputSource",
    "HostServicePort",
    "HostPayloadSchemaId",
    "ImplementationId",
    "PhysicalTask",
    "PackedRowsInputSource",
    "PackedSamplerRowsInputSource",
    "SamplerFeedbackInputSource",
    "ScheduleDecision",
    "SchedulerTimeline",
    "SemanticEdge",
    "SemanticEffect",
    "SemanticGraph",
    "ConditionalInputGroup",
    "SemanticInput",
    "SemanticOutput",
    "SemanticOutputKind",
    "SemanticState",
    "SemanticTask",
    "StepIdInputSource",
    "StateEffectKind",
    "StateResource",
    "StateScope",
    "StateSelectionInputSource",
    "TaskId",
    "TaskImplementationSpec",
    "TaskSymbol",
    "TaskResultInputSource",
    "TimelineAnchor",
    "TimingConstraint",
    "WorkloadId",
]

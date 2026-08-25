"""Type stub for the Rust extension module ``tllm._core`` (pyo3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict

class PlanPortSpecDescriptor(Protocol):
    @property
    def port(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def writable(self) -> bool: ...
    @property
    def write_modes(self) -> Sequence[str]: ...
    @property
    def phases(self) -> Sequence[str]: ...

class PlanProducerDescriptor(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def priority(self) -> int: ...
    @property
    def outputs(self) -> Sequence[PlanPortSpecDescriptor]: ...
    @property
    def requires(self) -> Sequence[str]: ...
    @property
    def hook_points(self) -> Sequence[str]: ...
    @property
    def graph_safety(self) -> str: ...

class PlanConsumerReadDescriptor(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def role(self) -> str: ...
    @property
    def phase(self) -> str: ...

class PlanConsumerWriteDescriptor(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def mode(self) -> str: ...
    @property
    def phase(self) -> str: ...

class PlanConsumerFlowDescriptor(Protocol):
    @property
    def reads(self) -> Sequence[PlanConsumerReadDescriptor]: ...
    @property
    def writes(self) -> Sequence[PlanConsumerWriteDescriptor]: ...
    @property
    def bundle_key(self) -> Sequence[str]: ...

class PlanConsumerDescriptor(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def flows(self) -> Sequence[PlanConsumerFlowDescriptor]: ...

class LocatorFieldDescriptor(Protocol):
    @property
    def path(self) -> Sequence[str]: ...
    @property
    def value_type(self) -> str: ...
    @property
    def value(self) -> str: ...

class PortAddressDescriptor(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def locator_type(self) -> str: ...
    @property
    def locator_fields(self) -> Sequence[LocatorFieldDescriptor]: ...

class PlacementWriteDescriptor(Protocol):
    @property
    def port(self) -> PortAddressDescriptor: ...
    @property
    def order(self) -> int: ...
    @property
    def mode(self) -> str: ...
    @property
    def fusion_candidate(self) -> bool: ...

class VersionedStateResourceDescriptor(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def namespace(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> int: ...

class PlacementResourceWriteDescriptor(Protocol):
    @property
    def resource(self) -> VersionedStateResourceDescriptor: ...

class PlacementFlowDescriptor(Protocol):
    @property
    def flow_id(self) -> str: ...
    @property
    def consumer_id(self) -> str: ...
    @property
    def declared_hook(self) -> str: ...
    @property
    def mobility(self) -> str | None: ...
    @property
    def deadline(self) -> str | None: ...
    @property
    def max_defer_steps(self) -> int | None: ...
    @property
    def observes_version(self) -> str | None: ...
    @property
    def reads(self) -> Sequence[PortAddressDescriptor]: ...
    @property
    def writes(self) -> Sequence[PlacementWriteDescriptor]: ...
    @property
    def delivery(self) -> str: ...
    @property
    def ownership(self) -> str: ...
    @property
    def lease_lifetime(self) -> str: ...
    @property
    def max_bundle_rows(self) -> int: ...
    @property
    def effect_hook(self) -> str | None: ...
    @property
    def earliest_hook(self) -> str | None: ...
    @property
    def allowed_hooks(self) -> Sequence[str]: ...
    @property
    def fusion_kind(self) -> str: ...
    @property
    def resource_reads(self) -> Sequence[VersionedStateResourceDescriptor]: ...
    @property
    def resource_writes(self) -> Sequence[PlacementResourceWriteDescriptor]: ...

class PlacementHookCostDescriptor(Protocol):
    @property
    def hook(self) -> str: ...
    @property
    def bubble_score(self) -> int: ...

class DecodeLocalization:
    positions: list[int]
    row_idx: list[int]
    contiguous: bool

class DecodePositions:
    positions: list[int]
    contiguous: bool

class RowPartition:
    affected_greedy: list[int]
    affected_random: list[int]
    unaffected_greedy: list[int]
    unaffected_random: list[int]

class SamplerPlan:
    path: str
    use_candidate_kernel: bool
    dense_inplace_applied: bool
    base: str

class CompiledPlan:
    ordered_producer_ids: list[str]
    port_sources: list[tuple[str, str]]
    capture_resident_producer_ids: list[str]

class PlacementSchedule:
    placements: list[tuple[str, str]]
    execution_order: list[str]
    fusion_groups: list[list[str]]
    fusion_rejections: list[tuple[str, str]]

class CompiledRecurrenceSchedule:
    groups: list[list[str]]
    execution_order: list[str]
    kind: str
    fusion_rejections: list[tuple[str, str, str]]

def compile_recurrence_schedule(
    ops: Sequence[object],
    edges: Sequence[object],
    internalized_edges: Sequence[str] | None = None,
    max_rows: int = 2,
) -> CompiledRecurrenceSchedule: ...

class OperandKind:
    Seed: OperandKind
    RegionValue: OperandKind
    BackendResource: OperandKind

class TypedOperand:
    kind: OperandKind
    binding_id: str | None
    value_id: str | None
    resource_id: str | None

class EffectKind:
    StateRead: EffectKind
    StateWrite: EffectKind
    StateTransition: EffectKind
    VersionPublish: EffectKind
    ExternIo: EffectKind

class TypedEffect:
    kind: EffectKind
    resource_id: str

class TypedOperation:
    operation_id: str
    primitive_key: str
    implementation_id: str
    operands: list[TypedOperand]
    results: list[str]
    effects: list[TypedEffect]

class TypedExecutionGroup:
    group_id: int
    primitive_key: str
    implementation_id: str
    operations: list[TypedOperation]
    release_after: list[str]

class ExecutionGroupingKind:
    Serial: ExecutionGroupingKind
    Wavefront: ExecutionGroupingKind

class TypedRecurrencePlan:
    schema_version: int
    groups: list[TypedExecutionGroup]
    grouping_kind: ExecutionGroupingKind
    retained_values: list[str]
    fusion_rejections: list[tuple[str, str, str]]

def compile_typed_recurrence_plan(spec_json: str) -> TypedRecurrencePlan: ...
def compile_typed_recurrence_plan_json(spec_json: str) -> str: ...
def validate_typed_recurrence_plan(plan_json: str) -> None: ...

class TypedStageBinding:
    binding_id: str
    kind: str
    coordinates: list[int]

class TypedStageOutputBinding:
    output_id: str
    value_id: str
    kind: str
    coordinates: list[int]

class TypedStageResourceBinding:
    resource_id: str
    kind: str
    coordinates: list[int]

class TypedStageRequestSpan(TypedDict):
    """Closed evidence that one request owns a contiguous packed row span."""
    request_ordinal: int
    packed_row_start: int
    packed_row_end: int
    token_offset_start: int
    token_offset_end: int
    phase: str

class TypedStageFacts(TypedDict):
    schema_version: int
    shape_specialization: list[tuple[str, str]]
    row_count: int
    request_count: int
    request_spans: list[TypedStageRequestSpan]

class TypedRecurrenceStageArtifact:
    schema_version: int
    signature: str
    shape_specialization: list[tuple[str, str]]
    row_count: int
    request_count: int
    plan: TypedRecurrencePlan
    seed_bindings: list[TypedStageBinding]
    resource_bindings: list[TypedStageResourceBinding]
    output_bindings: list[TypedStageOutputBinding]

class TypedBoundRecurrenceStage:
    signature: str
    plan: TypedRecurrencePlan
    seed_bindings: list[TypedStageBinding]
    resource_bindings: list[TypedStageResourceBinding]
    output_bindings: list[TypedStageOutputBinding]

class TypedObservedStageSpan:
    request_ordinal: int
    packed_row_start: int
    packed_row_end: int
    token_offset_start: int
    token_offset_end: int
    phase: SchedulerBatchPhase

class TypedPackedRecurrenceStageBinding:
    stage: TypedBoundRecurrenceStage
    observed_span: TypedObservedStageSpan

class TypedRecurrenceSpecializationRequired(ValueError): ...

class TypedStagePlanAdmission:
    Execute: TypedStagePlanAdmission
    Noop: TypedStagePlanAdmission

class TypedRecurrenceStageArtifactBundleEntry:
    row_count: int
    artifact: TypedRecurrenceStageArtifact

class TypedRecurrenceStageArtifactBundle:
    schema_version: int
    signature: str
    artifacts: list[TypedRecurrenceStageArtifactBundleEntry]

def compile_typed_recurrence_stage_artifact(spec_json: str) -> TypedRecurrenceStageArtifact: ...
def compile_typed_recurrence_stage_artifact_json(spec_json: str) -> str: ...
def validate_typed_recurrence_stage_artifact(artifact_json: str) -> None: ...
def load_typed_recurrence_stage_artifact(artifact_json: str) -> TypedRecurrenceStageArtifact: ...
def bind_typed_recurrence_stage(artifact: TypedRecurrenceStageArtifact, facts_json: str) -> TypedBoundRecurrenceStage: ...
def compile_typed_recurrence_stage_artifact_bundle_json(spec_json: str) -> str: ...
def validate_typed_recurrence_stage_artifact_bundle(bundle_json: str) -> None: ...
def load_typed_recurrence_stage_artifact_bundle(bundle_json: str) -> TypedRecurrenceStageArtifactBundle: ...
def typed_recurrence_stage_artifact_for_rows(
    bundle: TypedRecurrenceStageArtifactBundle, row_count: int
) -> TypedRecurrenceStageArtifact: ...

class PrefillRowSpan:
    request_ordinal: int
    packed_row_start: int
    packed_row_end: int
    token_offset_start: int
    token_offset_end: int
    prefill_row_count: int
    decode_row_count: int

class PackedPrefillRowPlan:
    request_ordinals: list[int]
    packed_row_indices: list[int]
    token_offsets: list[int]
    phases: list[str]
    request_spans: list[PrefillRowSpan]
    total_rows: int

class SchedulerBatchPhase:
    Empty: SchedulerBatchPhase
    Prefill: SchedulerBatchPhase
    Decode: SchedulerBatchPhase
    Mixed: SchedulerBatchPhase

def classify_scheduler_batch_phase(
    scheduled_tokens: list[int],
    prompt_tokens: list[int],
    computed_tokens: list[int],
) -> SchedulerBatchPhase: ...

def plan_packed_prefill_rows(
    scheduled_tokens: list[int],
    prompt_tokens: list[int],
    computed_tokens: list[int],
    expected_total_rows: int | None = None,
) -> PackedPrefillRowPlan: ...

def bind_typed_recurrence_stage_bundle_for_packed_span(
    bundle: TypedRecurrenceStageArtifactBundle,
    row_plan: PackedPrefillRowPlan,
    request_ordinal: int,
) -> TypedPackedRecurrenceStageBinding: ...

def admit_typed_recurrence_stage_bundle_for_packed_plan(
    bundle: TypedRecurrenceStageArtifactBundle,
    row_plan: PackedPrefillRowPlan,
) -> TypedStagePlanAdmission: ...

def bind_typed_recurrence_stage_bundle_for_request_rows(
    bundle: TypedRecurrenceStageArtifactBundle,
    row_count: int,
) -> TypedPackedRecurrenceStageBinding: ...

def plan_prefill_rows(
    scheduled_tokens: list[int],
    prompt_tokens: list[int],
    computed_tokens: list[int],
    expected_total_rows: int | None = None,
) -> PackedPrefillRowPlan: ...

# Legacy-only compatibility while historical vLLM replay is migrated.
class RecirculationInputKind:
    Seed: RecirculationInputKind
    Ordinal: RecirculationInputKind

class RecirculationPrefillGroupKind:
    ModelLayer: RecirculationPrefillGroupKind
    Mix: RecirculationPrefillGroupKind

class RecirculationPrefillOpKind:
    Normal: RecirculationPrefillOpKind
    Mix: RecirculationPrefillOpKind
    Recurrent: RecirculationPrefillOpKind

class RecirculationPrefillOp:
    ordinal: int
    kind: RecirculationPrefillOpKind
    token: int
    layer: int
    data_predecessor_ordinals: list[int]
    model_input_kind: RecirculationInputKind | None
    model_input_ordinal: int | None
    mix_source_ordinal: int | None
    mix_destination_ordinal: int | None
    mix_final_normal_ordinal: int | None

class RecirculationPrefillGroup:
    ordinal: int
    kind: RecirculationPrefillGroupKind
    model_layer: int | None
    operations: list[RecirculationPrefillOp]
    release_after: list[int]

class RecirculationPrefillSchedule:
    token_count: int
    layer_count: int
    source_layer: int
    destination_layer: int
    kind: str
    groups: list[RecirculationPrefillGroup]
    final_normal_ordinals: list[int]
    retained_ordinals: list[int]
    fusion_rejections: list[tuple[int, int, str]]

def compile_recirculation_prefill_schedule(
    token_count: int,
    layer_count: int,
    source_layer: int,
    destination_layer: int,
    fusion_policy: str,
) -> RecirculationPrefillSchedule: ...

class RecirculationPrefillRequest:
    request_id: str
    request_ordinal: int
    scheduled_tokens: int
    prompt_tokens: int
    computed_tokens: int
    packed_row_start: int
    packed_row_end: int
    packed_row_indices: list[int]
    token_position_start: int
    token_position_end: int
    token_positions: list[int]

class RecirculationPrefillPlan:
    schema_version: int
    phase: str
    layer_count: int
    total_rows: int
    requests: list[RecirculationPrefillRequest]

def normalize_recirculation_prefill_plan(plan_json: str) -> str: ...
def parse_recirculation_prefill_plan(plan_json: str) -> RecirculationPrefillPlan: ...
def build_recirculation_prefill_plan(
    request_ids: list[str],
    scheduled_tokens: list[int],
    prompt_tokens: list[int],
    computed_tokens: list[int],
    layer_count: int,
) -> RecirculationPrefillPlan: ...

class WorkGraphNodeKeyDescriptor(Protocol):
    namespace: str
    owner: str
    ordinal: int

class WorkGraphPortIdDescriptor(Protocol):
    namespace: str
    name: str
    version: int

class WorkGraphLocatorFieldDescriptor(Protocol):
    path: Sequence[str]
    value_type: str
    value: str

class WorkGraphPortAddressDescriptor(Protocol):
    kind: str
    locator_type: str
    locator_fields: Sequence[WorkGraphLocatorFieldDescriptor]

WorkGraphPortRefDescriptor = WorkGraphPortIdDescriptor | WorkGraphPortAddressDescriptor

class WorkGraphValueDescriptor(Protocol):
    value_id: str
    version: int
    port: WorkGraphPortRefDescriptor | None
    shape: Sequence[str]
    dtype: str
    layout: str
    alias_set: str

class WorkGraphAccessDescriptor(Protocol):
    node_key: WorkGraphNodeKeyDescriptor
    value: WorkGraphValueDescriptor
    mode: str
    port: WorkGraphPortRefDescriptor | None
    subset: Sequence[str]
    consistency: str

class WorkGraphTimingDescriptor(Protocol):
    window: str
    mobility: str
    deadline: str
    max_defer_steps: int
    observes_version: str

class WorkGraphExecutorDescriptor(Protocol):
    namespace: str
    name: str
    version: str

class WorkGraphAnchorDescriptor(Protocol):
    backend: str
    runner: str
    graph_mode: str
    hook: str
    externally_owned: bool
    mobility: str
    service_class: str

class WorkGraphBackendDescriptor(Protocol):
    backend_id: str
    runner_family: str
    graph_mode: str
    backend_version: str
    capabilities: Sequence[str]
    device: str
    rank: int
    tp_size: int
    pp_size: int
    dp_size: int
    topology: Sequence[str]

class WorkGraphSemanticNodeDescriptor(Protocol):
    node_key: WorkGraphNodeKeyDescriptor
    kind: str
    executor: WorkGraphExecutorDescriptor | None
    reads: Sequence[WorkGraphAccessDescriptor]
    writes: Sequence[WorkGraphAccessDescriptor]
    effects: Sequence[str]
    timing: WorkGraphTimingDescriptor
    anchor: WorkGraphAnchorDescriptor | None
    graph_safety: str
    source: str
    legacy_boundary: str | None
    shape_guard: Sequence[str]

class WorkGraphSemanticDescriptor(Protocol):
    schema_version: int
    nodes: Sequence[WorkGraphSemanticNodeDescriptor]
    edges: Sequence[tuple[WorkGraphNodeKeyDescriptor, WorkGraphNodeKeyDescriptor, str]]
    values: Sequence[WorkGraphValueDescriptor]
    backend: WorkGraphBackendDescriptor | None
    graph_mode: str

class WorkGraphExecutionNodeDescriptor(Protocol):
    node_key: WorkGraphNodeKeyDescriptor
    semantic_key: WorkGraphNodeKeyDescriptor | None
    kind: str
    executor: WorkGraphExecutorDescriptor | None
    accesses: Sequence[WorkGraphAccessDescriptor]
    residence_keys: Sequence[str]
    move_keys: Sequence[str]
    lease_keys: Sequence[str]
    device: str
    rank: int
    anchor: WorkGraphAnchorDescriptor | None
    graph_safety: str
    fusion_group: str

class WorkGraphResidenceDescriptor(Protocol):
    value: WorkGraphValueDescriptor
    device: str
    rank: int
    memory_kind: str
    valid_version: int

class WorkGraphMoveDescriptor(Protocol):
    value: WorkGraphValueDescriptor
    source: WorkGraphResidenceDescriptor
    destination: WorkGraphResidenceDescriptor
    subset: Sequence[str]
    layout: str
    engine: str
    asynchronous: bool

class WorkGraphLeaseDescriptor(Protocol):
    value: WorkGraphValueDescriptor
    storage: str
    live_start: str
    live_end: str
    ready_fence: str
    retire_fence: str
    reader_count: int
    stable_address: bool
    materialized: bool

class WorkGraphDataDescriptor(Protocol):
    schema_version: int
    nodes: Sequence[WorkGraphExecutionNodeDescriptor]
    values: Sequence[WorkGraphValueDescriptor]
    accesses: Sequence[WorkGraphAccessDescriptor]
    residences: Sequence[WorkGraphResidenceDescriptor]
    moves: Sequence[WorkGraphMoveDescriptor]
    leases: Sequence[WorkGraphLeaseDescriptor]
    edges: Sequence[tuple[WorkGraphNodeKeyDescriptor, WorkGraphNodeKeyDescriptor, str]]
    backend: WorkGraphBackendDescriptor | None
    graph_mode: str

class WorkGraphValidation:
    topological_order: list[str]

def compile_placement_schedule(
    flows: Sequence[PlacementFlowDescriptor],
    graph_mode: str,
    hook_costs: Sequence[PlacementHookCostDescriptor] | None = None,
) -> PlacementSchedule: ...

def localize_decode_rows(
    is_decode: list[bool],
    logits_indices: list[int],
    req_count: int,
    num_actual_tokens: int,
    max_decode_rows: int,
) -> DecodeLocalization: ...

def localize_decode_positions(
    is_decode: list[bool],
    n_logits: int,
    req_count: int,
    max_decode_rows: int,
) -> DecodePositions: ...

def plan_sampler_step(
    backend: str,
    temperature_mode: str,
    n_rows: int,
    n_affected: int,
    n_greedy: int,
    greedy_sampled_available: bool,
    candidate_eligible: bool,
) -> SamplerPlan: ...

def partition_rows(
    n_rows: int,
    affected_rows: list[int],
    greedy_rows: list[int],
    greedy_sampled_available: bool,
) -> RowPartition: ...

def compile_execution_plan(
    producers: Sequence[PlanProducerDescriptor],
    consumers: Sequence[PlanConsumerDescriptor],
    enforced_kinds: list[str] | None = ...,
    supported_hooks: list[str] | None = ...,
    backend: tuple[str, str] | None = ...,
    backend_capabilities: list[str] | None = ...,
) -> CompiledPlan: ...

def compile_schedule(graph: str, timeline: str, options: str) -> str: ...
def validate_schedule(graph: str, timeline: str, options: str, decision: str) -> None: ...
def validate_workload_compilation_wire(
    cache_key: str,
    graph_json: str,
    timeline_json: str,
    options_json: str,
    decision_json: str,
    plan_epoch: int,
) -> None: ...
def validate_worker_reconstruction_specs(
    cache_key: str,
    graph_json: str,
    timeline_json: str,
    options_json: str,
    decision_json: str,
    plan_epoch: int,
    reconstruction_specs_json: str,
) -> None: ...
def validate_worker_reconstruction_wire(
    cache_key: str,
    graph_json: str,
    timeline_json: str,
    options_json: str,
    decision_json: str,
    plan_epoch: int,
    reconstructed_graph_json: str,
    runtime_specialization_facts_json: str,
    reconstructed_residual_resource_manifest_json: str,
    reconstructed_static_binding_manifest_json: str,
) -> None: ...
def validate_local_execution_binding_wire(
    cache_key: str,
    graph_json: str,
    timeline_json: str,
    options_json: str,
    decision_json: str,
    plan_epoch: int,
    reconstructed_graph_json: str,
    reconstructed_residual_resource_manifest_json: str,
    reconstructed_static_binding_manifest_json: str,
) -> None: ...
def normalize_runtime_specialization_options(
    options_json: str,
    facts_json: str,
) -> str: ...
def apply_region_shape_specializations(
    cache_key: str,
    graph_json: str,
    timeline_json: str,
    options_json: str,
    decision_json: str,
    plan_epoch: int,
    requests_json: str,
) -> str: ...
def normalize_region_shape_specialization_request(payload: str) -> str: ...
def validate_region_shape_specialization_request(payload: str) -> None: ...

def validate_workgraph(
    semantic: WorkGraphSemanticDescriptor,
    data: WorkGraphDataDescriptor,
) -> WorkGraphValidation: ...

def validate_workgraph_ir(
    semantic: WorkGraphSemanticDescriptor,
    data: WorkGraphDataDescriptor,
) -> WorkGraphValidation: ...

def validate_capture_hierarchy(
    hierarchy: object,
    semantic: WorkGraphSemanticDescriptor,
) -> WorkGraphValidation: ...

def validate_physical_flow_plan(plan: object) -> WorkGraphValidation: ...

def validate_placement_plan(plan: object) -> WorkGraphValidation: ...

def validate_distributed_flow_plan(plan: object) -> WorkGraphValidation: ...

def validate_adaptive_flow_plan(plan: object) -> WorkGraphValidation: ...

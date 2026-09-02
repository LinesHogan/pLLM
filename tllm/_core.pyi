"""Type stub for the Rust extension module ``tllm._core`` (pyo3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

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

class OperandKind:
    Seed: OperandKind
    RegionValue: OperandKind
    BackendResource: OperandKind

class TypedOperand:
    kind: OperandKind
    binding_id: str | None
    value_id: str | None
    resource_id: str | None
    @staticmethod
    def seed(binding_id: str) -> TypedOperand: ...
    @staticmethod
    def region_value(value_id: str) -> TypedOperand: ...
    @staticmethod
    def backend_resource(resource_id: str) -> TypedOperand: ...

class EffectKind:
    StateRead: EffectKind
    StateWrite: EffectKind
    StateTransition: EffectKind
    VersionPublish: EffectKind
    ExternIo: EffectKind

class TypedEffect:
    kind: EffectKind
    resource_id: str
    def __init__(self, kind: EffectKind, resource_id: str) -> None: ...

class TypedOperation:
    operation_id: str
    primitive_key: str
    implementation_id: str
    operands: list[TypedOperand]
    results: list[str]
    effects: list[TypedEffect]
    def __init__(
        self,
        operation_id: str,
        primitive_key: str,
        implementation_id: str,
        operands: Sequence[TypedOperand],
        results: Sequence[str],
        effects: Sequence[TypedEffect] = (),
    ) -> None: ...

class TypedDependency:
    source: str
    target: str
    kind: str
    def __init__(self, source: str, target: str, kind: str) -> None: ...

class TypedStageAdmission:
    @staticmethod
    def prefill_reject() -> TypedStageAdmission: ...
    @staticmethod
    def prefill_select() -> TypedStageAdmission: ...
    @staticmethod
    def decode_reject() -> TypedStageAdmission: ...

class TypedStageFusionPolicy:
    @staticmethod
    def automatic() -> TypedStageFusionPolicy: ...
    @staticmethod
    def serial() -> TypedStageFusionPolicy: ...
    @staticmethod
    def ordered(
        max_rows: int, internalized_edges: Sequence[str]
    ) -> TypedStageFusionPolicy: ...

class TypedRecurrence:
    def __init__(
        self,
        region_id: str,
        implementation_id: str,
        admission: TypedStageAdmission,
        fusion_policy: TypedStageFusionPolicy,
        operations: Sequence[TypedOperation],
        dependencies: Sequence[TypedDependency],
        retained_values: Sequence[str],
    ) -> None: ...

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
    def __init__(
        self, binding_id: str, kind: str, coordinates: Sequence[int]
    ) -> None: ...

class TypedStageOutputBinding:
    output_id: str
    value_id: str
    kind: str
    coordinates: list[int]
    def __init__(
        self,
        output_id: str,
        value_id: str,
        kind: str,
        coordinates: Sequence[int],
    ) -> None: ...

class TypedStageResourceBinding:
    resource_id: str
    kind: str
    coordinates: list[int]
    def __init__(
        self, resource_id: str, kind: str, coordinates: Sequence[int]
    ) -> None: ...

class TypedRecurrenceStageDefinition:
    def __init__(
        self,
        row_count: int,
        recurrence: TypedRecurrence,
        seed_bindings: Sequence[TypedStageBinding],
        resource_bindings: Sequence[TypedStageResourceBinding],
        output_bindings: Sequence[TypedStageOutputBinding],
    ) -> None: ...

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

class TypedPackedRecurrenceStageBatch:
    admission: TypedStagePlanAdmission
    total_rows: int
    requests: list[TypedPackedRecurrenceStageBinding]

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
    max_row_count: int
    @staticmethod
    def compile(
        definitions: Sequence[TypedRecurrenceStageDefinition],
    ) -> TypedRecurrenceStageArtifactBundle: ...
    @staticmethod
    def from_wire(bundle_json: str) -> TypedRecurrenceStageArtifactBundle: ...
    def to_wire(self) -> str: ...
    def bind_packed_plan(
        self, row_plan: PackedPrefillRowPlan
    ) -> TypedPackedRecurrenceStageBatch: ...

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

def compile_schedule(graph: str, timeline: str, options: str) -> str: ...
def validate_schedule(graph: str, timeline: str, options: str, decision: str) -> None: ...
def compile_state_representation_plan(request_json: str) -> str: ...
def validate_state_representation_plan(
    request_json: str,
    decision_json: str,
) -> None: ...
def validate_state_representation_reconstruction(
    request_json: str,
    decision_json: str,
    local_candidates_json: str,
) -> None: ...
def lower_canonical_state_physical_plan(
    request_json: str,
    decision_json: str,
    plan_epoch: int,
) -> str: ...
def validate_canonical_state_physical_plan(
    request_json: str,
    decision_json: str,
    physical_json: str,
) -> None: ...
def intern_canonical_tensor_layouts(request_json: str) -> str: ...
def validate_canonical_tensor_layouts(
    request_json: str,
    decision_json: str,
) -> None: ...
def compile_canonical_schedule(graph_json: str, timeline_json: str) -> str: ...
def validate_canonical_schedule(
    graph_json: str,
    timeline_json: str,
    decision_json: str,
) -> None: ...
def lower_canonical_physical_plan(
    graph_json: str,
    timeline_json: str,
    decision_json: str,
    plan_epoch: int,
) -> str: ...
def validate_canonical_physical_plan(
    graph_json: str,
    timeline_json: str,
    decision_json: str,
    physical_json: str,
) -> None: ...

def compile_backend_attachment(
    inputs_json: str,
) -> str: ...

def validate_backend_attachment(
    inputs_json: str,
    decision_json: str,
) -> None: ...

def lower_backend_attachment(
    inputs_json: str,
    decision_json: str,
) -> str: ...

def validate_backend_attachment_physical(
    inputs_json: str,
    decision_json: str,
    physical_json: str,
) -> None: ...

def validate_backend_attachment_reconstruction(
    inputs_json: str,
    decision_json: str,
    physical_json: str,
    local_inputs_json: str,
    local_physical_json: str,
) -> None: ...

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
    reconstructed_packed_hidden_resource_manifest_json: str,
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
    reconstructed_packed_hidden_resource_manifest_json: str,
    reconstructed_static_binding_manifest_json: str,
) -> None: ...
def normalize_runtime_specialization_options(
    options_json: str,
    facts_json: str,
) -> str: ...
def intern_state_layout_ids(graph_json: str) -> str: ...
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

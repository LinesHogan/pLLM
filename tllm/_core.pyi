"""Type stub for the Rust extension module ``tllm._core`` (pyo3)."""

from __future__ import annotations

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

def normalize_region_shape_specialization_request(payload: str) -> str: ...
def validate_region_shape_specialization_request(payload: str) -> None: ...

"""Type stub for the Rust extension module ``pllm._core`` (pyo3)."""

from __future__ import annotations

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
    prompt_lengths: list[int]
    phases: list[str]
    request_spans: list[PrefillRowSpan]
    total_rows: int

def plan_packed_prefill_rows(
    scheduled_tokens: list[int],
    prompt_tokens: list[int],
    computed_tokens: list[int],
    expected_total_rows: int | None = None,
) -> PackedPrefillRowPlan: ...
def intern_tensor_layouts(request_json: str) -> str: ...
def compile_schedule(graph_json: str, timeline_json: str, state_json: str | None = None) -> str: ...
def compile_state_schedule(
    graph_json: str,
    timeline_json: str,
    request_json: str,
) -> tuple[str, str]: ...

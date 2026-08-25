"""Pure Python oracle for packed prefill row planning.

The oracle intentionally mirrors only integer planning semantics; it does not
know request ids or inspect tensors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OraclePrefillRowSpan:
    request_ordinal: int
    packed_row_start: int
    packed_row_end: int
    token_offset_start: int
    token_offset_end: int
    prefill_row_count: int
    decode_row_count: int


@dataclass(frozen=True)
class OraclePackedPrefillRowPlan:
    request_ordinals: tuple[int, ...]
    packed_row_indices: tuple[int, ...]
    token_offsets: tuple[int, ...]
    phases: tuple[str, ...]
    request_spans: tuple[OraclePrefillRowSpan, ...]
    total_rows: int


def oracle_plan_packed_prefill_rows(
    scheduled_tokens: list[int] | tuple[int, ...],
    prompt_tokens: list[int] | tuple[int, ...],
    computed_tokens: list[int] | tuple[int, ...],
    expected_total_rows: int | None = None,
) -> OraclePackedPrefillRowPlan:
    if not (len(scheduled_tokens) == len(prompt_tokens) == len(computed_tokens)):
        raise ValueError("scheduled_tokens, prompt_tokens, and computed_tokens must have equal lengths")
    for name, values in (
        ("scheduled_tokens", scheduled_tokens),
        ("prompt_tokens", prompt_tokens),
        ("computed_tokens", computed_tokens),
    ):
        if any(value < 0 for value in values):
            raise ValueError(f"{name} must contain only non-negative values")
    if expected_total_rows is not None and expected_total_rows < 0:
        raise ValueError("expected_total_rows must be non-negative")
    total_rows = sum(scheduled_tokens)
    if expected_total_rows is not None and expected_total_rows != total_rows:
        raise ValueError("packed row count mismatch")

    request_ordinals: list[int] = []
    packed_row_indices: list[int] = []
    token_offsets: list[int] = []
    phases: list[str] = []
    spans: list[OraclePrefillRowSpan] = []
    row_base = 0
    for ordinal, (scheduled, prompt, computed) in enumerate(
        zip(scheduled_tokens, prompt_tokens, computed_tokens)
    ):
        row_end = row_base + scheduled
        token_end = computed + scheduled
        prefill_rows = min(scheduled, max(0, prompt - computed))
        for local_row in range(scheduled):
            request_ordinals.append(ordinal)
            packed_row_indices.append(row_base + local_row)
            token_offset = computed + local_row
            token_offsets.append(token_offset)
            phases.append("prefill" if token_offset < prompt else "decode")
        spans.append(
            OraclePrefillRowSpan(
                request_ordinal=ordinal,
                packed_row_start=row_base,
                packed_row_end=row_end,
                token_offset_start=computed,
                token_offset_end=token_end,
                prefill_row_count=prefill_rows,
                decode_row_count=scheduled - prefill_rows,
            )
        )
        row_base = row_end
    return OraclePackedPrefillRowPlan(
        request_ordinals=tuple(request_ordinals),
        packed_row_indices=tuple(packed_row_indices),
        token_offsets=tuple(token_offsets),
        phases=tuple(phases),
        request_spans=tuple(spans),
        total_rows=total_rows,
    )

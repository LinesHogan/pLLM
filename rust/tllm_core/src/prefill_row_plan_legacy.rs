//! Packed prefill/decode row planning.
//!
//! This module only deals with host-side integer metadata. Request ids and
//! tensors stay in Python; the returned request ordinals are indexes into the
//! caller's request-order arrays.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Write;

/// Closed classification for all scheduled rows in one backend step.
///
/// The value is intentionally derived from the same packed-row arithmetic as
/// the prefill plan, rather than from a runner-specific boolean.  Python may
/// map this enum onto a hook filter, but cannot decide the phase itself.
#[pyclass(frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SchedulerBatchPhase {
    Empty,
    Prefill,
    Decode,
    Mixed,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrefillRowSpan {
    /// Request ordinal in the input arrays.
    #[pyo3(get)]
    pub(crate) request_ordinal: i64,
    /// Half-open range in the packed hidden tensor for this request.
    #[pyo3(get)]
    pub(crate) packed_row_start: i64,
    #[pyo3(get)]
    pub(crate) packed_row_end: i64,
    /// Half-open range of token positions represented by this step.
    #[pyo3(get)]
    pub(crate) token_offset_start: i64,
    #[pyo3(get)]
    pub(crate) token_offset_end: i64,
    /// Number of rows in each phase within the request span.
    #[pyo3(get)]
    pub(crate) prefill_row_count: i64,
    #[pyo3(get)]
    pub(crate) decode_row_count: i64,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackedPrefillRowPlan {
    /// One entry per scheduled packed row, in packed-row order.
    #[pyo3(get)]
    pub(crate) request_ordinals: Vec<i64>,
    #[pyo3(get)]
    pub(crate) packed_row_indices: Vec<i64>,
    #[pyo3(get)]
    pub(crate) token_offsets: Vec<i64>,
    /// Values are exactly ``"prefill"`` or ``"decode"``.
    #[pyo3(get)]
    pub(crate) phases: Vec<String>,
    /// One span per input request, including requests with zero scheduled rows.
    #[pyo3(get)]
    pub(crate) request_spans: Vec<PrefillRowSpan>,
    /// Total number of packed rows represented by this plan.
    #[pyo3(get)]
    pub(crate) total_rows: i64,
}

fn validation_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn decode<'a, T: Deserialize<'a>>(payload: &'a str, label: &str) -> PyResult<T> {
    serde_json::from_str(payload)
        .map_err(|error| validation_error(format!("invalid {label}: {error}")))
}

fn canonical_ascii_json<T: Serialize>(value: &T) -> PyResult<String> {
    let value = serde_json::to_value(value).map_err(|error| {
        validation_error(format!(
            "failed to encode canonical Recirculation prefill plan: {error}"
        ))
    })?;
    let encoded = serde_json::to_string(&value).map_err(|error| {
        validation_error(format!(
            "failed to serialize canonical Recirculation prefill plan: {error}"
        ))
    })?;
    let mut canonical = String::with_capacity(encoded.len());
    for character in encoded.chars() {
        if character.is_ascii() {
            canonical.push(character);
            continue;
        }
        let scalar = character as u32;
        if scalar <= 0xffff {
            write!(&mut canonical, "\\u{scalar:04x}").expect("writing to String cannot fail");
            continue;
        }
        let surrogate = scalar - 0x1_0000;
        let high = 0xd800 + (surrogate >> 10);
        let low = 0xdc00 + (surrogate & 0x3ff);
        write!(&mut canonical, "\\u{high:04x}\\u{low:04x}").expect("writing to String cannot fail");
    }
    Ok(canonical)
}

/// Closed host-side plan for binding one pure-prefill Recirculation batch.
///
/// The request descriptors retain vLLM's scheduler facts alongside the exact
/// packed row and token-position expansion. Rust re-derives that expansion
/// before the backend can bind a recurrence executor.
#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecirculationPrefillPlan {
    #[pyo3(get)]
    schema_version: i64,
    #[pyo3(get)]
    phase: String,
    #[pyo3(get)]
    layer_count: i64,
    #[pyo3(get)]
    total_rows: i64,
    #[pyo3(get)]
    requests: Vec<RecirculationPrefillRequest>,
}

#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecirculationPrefillRequest {
    #[pyo3(get)]
    request_id: String,
    #[pyo3(get)]
    request_ordinal: i64,
    #[pyo3(get)]
    scheduled_tokens: i64,
    #[pyo3(get)]
    prompt_tokens: i64,
    #[pyo3(get)]
    computed_tokens: i64,
    #[pyo3(get)]
    packed_row_start: i64,
    #[pyo3(get)]
    packed_row_end: i64,
    #[pyo3(get)]
    packed_row_indices: Vec<i64>,
    #[pyo3(get)]
    token_position_start: i64,
    #[pyo3(get)]
    token_position_end: i64,
    #[pyo3(get)]
    token_positions: Vec<i64>,
}

fn validate_recirculation_prefill_plan(plan: &RecirculationPrefillPlan) -> PyResult<()> {
    if plan.schema_version != 1 {
        return Err(validation_error(
            "unsupported RecirculationPrefillPlan schema_version",
        ));
    }
    if plan.phase != "prefill" {
        return Err(validation_error(
            "RecirculationPrefillPlan phase must be prefill",
        ));
    }
    if plan.layer_count < 1 {
        return Err(validation_error(
            "RecirculationPrefillPlan layer_count must be positive",
        ));
    }
    if plan.total_rows < 0 {
        return Err(validation_error(
            "RecirculationPrefillPlan total_rows must be non-negative",
        ));
    }
    if plan.requests.is_empty() && plan.total_rows != 0 {
        return Err(validation_error(
            "an empty RecirculationPrefillPlan requires total_rows=0",
        ));
    }

    let mut request_ids = BTreeSet::new();
    let mut scheduled = Vec::with_capacity(plan.requests.len());
    let mut prompt = Vec::with_capacity(plan.requests.len());
    let mut computed = Vec::with_capacity(plan.requests.len());
    for (index, request) in plan.requests.iter().enumerate() {
        if request.request_id.is_empty() || !request_ids.insert(request.request_id.as_str()) {
            return Err(validation_error(
                "RecirculationPrefillPlan request_id values must be non-empty and unique",
            ));
        }
        if request.request_ordinal != index as i64 {
            return Err(validation_error(
                "RecirculationPrefillPlan request ordinals must be contiguous canonical order",
            ));
        }
        if request.scheduled_tokens < 1 {
            return Err(validation_error(
                "RecirculationPrefillPlan scheduled_tokens must be positive",
            ));
        }
        scheduled.push(request.scheduled_tokens);
        prompt.push(request.prompt_tokens);
        computed.push(request.computed_tokens);
    }

    let declared_scheduled_total = scheduled.iter().try_fold(0_i64, |total, &count| {
        total.checked_add(count).ok_or_else(|| {
            validation_error(
                "RecirculationPrefillPlan scheduled token total exceeds the supported row index range",
            )
        })
    })?;
    if declared_scheduled_total != plan.total_rows {
        return Err(validation_error(
            "RecirculationPrefillPlan total_rows does not match scheduled token total",
        ));
    }

    let packed = plan_packed_prefill_rows(scheduled, prompt, computed, Some(plan.total_rows))?;
    for (request, span) in plan.requests.iter().zip(&packed.request_spans) {
        if span.prefill_row_count != request.scheduled_tokens || span.decode_row_count != 0 {
            return Err(validation_error(
                "RecirculationPrefillPlan cannot mix prefill and decode rows",
            ));
        }
        let expected_rows: Vec<i64> = (span.packed_row_start..span.packed_row_end).collect();
        let expected_positions = packed.token_offsets
            [span.packed_row_start as usize..span.packed_row_end as usize]
            .to_vec();
        if request.packed_row_start != span.packed_row_start
            || request.packed_row_end != span.packed_row_end
            || request.packed_row_indices != expected_rows
        {
            return Err(validation_error(
                "RecirculationPrefillPlan packed row spans or indices disagree with packed-row planning",
            ));
        }
        if request.token_position_start != span.token_offset_start
            || request.token_position_end != span.token_offset_end
            || request.token_positions != expected_positions
        {
            return Err(validation_error(
                "RecirculationPrefillPlan token positions disagree with packed-row planning",
            ));
        }
    }
    Ok(())
}

/// Validate and canonically encode one vLLM prepare-inputs Recirculation
/// batch. Python transports this immutable JSON only; it cannot repair row
/// spans, token positions, phase, or request ordering.
#[pyfunction]
pub(crate) fn normalize_recirculation_prefill_plan(plan_json: &str) -> PyResult<String> {
    let plan: RecirculationPrefillPlan = decode(plan_json, "RecirculationPrefillPlan")?;
    validate_recirculation_prefill_plan(&plan)?;
    canonical_ascii_json(&plan)
}

/// Decode one closed Recirculation prefill plan into its validated typed view.
///
/// This shares the normalizer's closed-wire and row-planning validation. The
/// returned PyO3 objects intentionally expose getters only, so Python cannot
/// alter the verified plan in place.
#[pyfunction]
pub(crate) fn parse_recirculation_prefill_plan(
    plan_json: &str,
) -> PyResult<RecirculationPrefillPlan> {
    let plan: RecirculationPrefillPlan = decode(plan_json, "RecirculationPrefillPlan")?;
    validate_recirculation_prefill_plan(&plan)?;
    Ok(plan)
}

/// Build a closed Recirculation plan directly from vLLM scheduler columns.
///
/// Python supplies only scheduler facts. Rust derives every packed-row span,
/// token position, and phase, then validates the resulting typed plan through
/// the same closed-plan validator used for the JSON transport boundary.
/// A completely decode-only step is represented as the validated zero-row
/// no-op. A batch containing both prefill and decode work is not bindable.
#[pyfunction]
pub(crate) fn build_recirculation_prefill_plan(
    request_ids: Vec<String>,
    scheduled_tokens: Vec<i64>,
    prompt_tokens: Vec<i64>,
    computed_tokens: Vec<i64>,
    layer_count: i64,
) -> PyResult<RecirculationPrefillPlan> {
    let request_count = request_ids.len();
    if scheduled_tokens.len() != request_count
        || prompt_tokens.len() != request_count
        || computed_tokens.len() != request_count
    {
        return Err(validation_error(format!(
            "request_ids, scheduled_tokens, prompt_tokens, and computed_tokens must have equal lengths; got {}, {}, {}, {}",
            request_count,
            scheduled_tokens.len(),
            prompt_tokens.len(),
            computed_tokens.len()
        )));
    }
    if layer_count < 1 {
        return Err(validation_error(
            "RecirculationPrefillPlan layer_count must be positive",
        ));
    }

    let mut unique_request_ids = BTreeSet::new();
    for request_id in &request_ids {
        if request_id.is_empty() || !unique_request_ids.insert(request_id.as_str()) {
            return Err(validation_error(
                "RecirculationPrefillPlan request_id values must be non-empty and unique",
            ));
        }
    }

    let packed = plan_packed_prefill_rows(
        scheduled_tokens.clone(),
        prompt_tokens.clone(),
        computed_tokens.clone(),
        None,
    )?;
    let has_prefill = packed.phases.iter().any(|phase| phase == "prefill");
    let has_decode = packed.phases.iter().any(|phase| phase == "decode");
    if has_prefill && has_decode {
        return Err(validation_error(
            "RecirculationPrefillPlan cannot mix prefill and decode rows",
        ));
    }

    if !has_prefill {
        let plan = RecirculationPrefillPlan {
            schema_version: 1,
            phase: "prefill".to_string(),
            layer_count,
            total_rows: 0,
            requests: Vec::new(),
        };
        validate_recirculation_prefill_plan(&plan)?;
        return Ok(plan);
    }

    let mut requests = Vec::with_capacity(request_count);
    for (request_ordinal, span) in packed.request_spans.iter().enumerate() {
        if span.prefill_row_count != scheduled_tokens[request_ordinal] || span.decode_row_count != 0
        {
            return Err(validation_error(
                "RecirculationPrefillPlan cannot mix prefill and decode rows",
            ));
        }
        if scheduled_tokens[request_ordinal] < 1 {
            return Err(validation_error(
                "RecirculationPrefillPlan scheduled_tokens must be positive for prefill requests",
            ));
        }
        let packed_row_indices = (span.packed_row_start..span.packed_row_end).collect();
        let token_positions = packed.token_offsets
            [span.packed_row_start as usize..span.packed_row_end as usize]
            .to_vec();
        requests.push(RecirculationPrefillRequest {
            request_id: request_ids[request_ordinal].clone(),
            request_ordinal: request_ordinal as i64,
            scheduled_tokens: scheduled_tokens[request_ordinal],
            prompt_tokens: prompt_tokens[request_ordinal],
            computed_tokens: computed_tokens[request_ordinal],
            packed_row_start: span.packed_row_start,
            packed_row_end: span.packed_row_end,
            packed_row_indices,
            token_position_start: span.token_offset_start,
            token_position_end: span.token_offset_end,
            token_positions,
        });
    }
    let plan = RecirculationPrefillPlan {
        schema_version: 1,
        phase: "prefill".to_string(),
        layer_count,
        total_rows: packed.total_rows,
        requests,
    };
    validate_recirculation_prefill_plan(&plan)?;
    Ok(plan)
}

fn validate_non_negative(values: &[i64], name: &str) -> PyResult<()> {
    if let Some((index, value)) = values.iter().enumerate().find(|(_, value)| **value < 0) {
        return Err(validation_error(format!(
            "{name}[{index}] must be non-negative, got {value}"
        )));
    }
    Ok(())
}

/// Plan every row in a vLLM packed step.
///
/// For request ``r`` with ``scheduled[r]`` rows, the packed rows are
/// contiguous. The token position of row ``i`` is ``computed[r] + i``;
/// positions below ``prompt[r]`` are prefill rows and the remainder are decode
/// rows. ``expected_total_rows`` is an optional boundary check against a
/// tensor's first dimension.
#[pyfunction]
#[pyo3(signature = (scheduled_tokens, prompt_tokens, computed_tokens, expected_total_rows=None))]
pub(crate) fn plan_packed_prefill_rows(
    scheduled_tokens: Vec<i64>,
    prompt_tokens: Vec<i64>,
    computed_tokens: Vec<i64>,
    expected_total_rows: Option<i64>,
) -> PyResult<PackedPrefillRowPlan> {
    let request_count = scheduled_tokens.len();
    if prompt_tokens.len() != request_count || computed_tokens.len() != request_count {
        return Err(validation_error(format!(
            "scheduled_tokens, prompt_tokens, and computed_tokens must have equal lengths; got {}, {}, {}",
            scheduled_tokens.len(),
            prompt_tokens.len(),
            computed_tokens.len()
        )));
    }
    validate_non_negative(&scheduled_tokens, "scheduled_tokens")?;
    validate_non_negative(&prompt_tokens, "prompt_tokens")?;
    validate_non_negative(&computed_tokens, "computed_tokens")?;
    if let Some(total_rows) = expected_total_rows {
        if total_rows < 0 {
            return Err(validation_error(format!(
                "expected_total_rows must be non-negative, got {total_rows}"
            )));
        }
    }

    let total_rows = scheduled_tokens
        .iter()
        .try_fold(0_i64, |total, &scheduled| {
            total.checked_add(scheduled).ok_or_else(|| {
                validation_error("scheduled token total exceeds the supported row index range")
            })
        })?;
    if let Some(expected) = expected_total_rows {
        if expected != total_rows {
            return Err(validation_error(format!(
                "packed row count mismatch: scheduled token total is {total_rows}, expected {expected}"
            )));
        }
    }

    let mut request_ordinals = Vec::with_capacity(total_rows as usize);
    let mut packed_row_indices = Vec::with_capacity(total_rows as usize);
    let mut token_offsets = Vec::with_capacity(total_rows as usize);
    let mut phases = Vec::with_capacity(total_rows as usize);
    let mut request_spans = Vec::with_capacity(request_count);
    let mut row_base = 0_i64;

    for (request_ordinal, (&scheduled, (&prompt, &computed))) in scheduled_tokens
        .iter()
        .zip(prompt_tokens.iter().zip(computed_tokens.iter()))
        .enumerate()
    {
        let row_end = row_base.checked_add(scheduled).ok_or_else(|| {
            validation_error("packed row index exceeds the supported row index range")
        })?;
        let token_end = computed.checked_add(scheduled).ok_or_else(|| {
            validation_error(format!(
                "computed_tokens[{request_ordinal}] plus scheduled_tokens[{request_ordinal}] exceeds the supported token offset range"
            ))
        })?;
        let prefill_rows = (prompt - computed).max(0).min(scheduled);
        let decode_rows = scheduled - prefill_rows;

        for local_row in 0..scheduled {
            let packed_row = row_base + local_row;
            let token_offset = computed + local_row;
            request_ordinals.push(request_ordinal as i64);
            packed_row_indices.push(packed_row);
            token_offsets.push(token_offset);
            phases.push(if token_offset < prompt {
                "prefill".to_string()
            } else {
                "decode".to_string()
            });
        }
        request_spans.push(PrefillRowSpan {
            request_ordinal: request_ordinal as i64,
            packed_row_start: row_base,
            packed_row_end: row_end,
            token_offset_start: computed,
            token_offset_end: token_end,
            prefill_row_count: prefill_rows,
            decode_row_count: decode_rows,
        });
        row_base = row_end;
    }

    debug_assert_eq!(row_base, total_rows);
    Ok(PackedPrefillRowPlan {
        request_ordinals,
        packed_row_indices,
        token_offsets,
        phases,
        request_spans,
        total_rows,
    })
}

/// Short alias retained for callers that refer to the operation as row
/// planning rather than packed-prefill planning.
#[pyfunction]
#[pyo3(signature = (scheduled_tokens, prompt_tokens, computed_tokens, expected_total_rows=None))]
pub(crate) fn plan_prefill_rows(
    scheduled_tokens: Vec<i64>,
    prompt_tokens: Vec<i64>,
    computed_tokens: Vec<i64>,
    expected_total_rows: Option<i64>,
) -> PyResult<PackedPrefillRowPlan> {
    plan_packed_prefill_rows(
        scheduled_tokens,
        prompt_tokens,
        computed_tokens,
        expected_total_rows,
    )
}

/// Classify the complete scheduler batch from its request-ordered columns.
///
/// ``Mixed`` is distinct from both pure phases. Consumers that only expose a
/// single phase filter must reject it rather than silently treating it as a
/// decode batch.
#[pyfunction]
pub(crate) fn classify_scheduler_batch_phase(
    scheduled_tokens: Vec<i64>,
    prompt_tokens: Vec<i64>,
    computed_tokens: Vec<i64>,
) -> PyResult<SchedulerBatchPhase> {
    let packed = plan_packed_prefill_rows(scheduled_tokens, prompt_tokens, computed_tokens, None)?;
    let has_prefill = packed.phases.iter().any(|phase| phase == "prefill");
    let has_decode = packed.phases.iter().any(|phase| phase == "decode");
    Ok(match (has_prefill, has_decode) {
        (false, false) => SchedulerBatchPhase::Empty,
        (true, false) => SchedulerBatchPhase::Prefill,
        (false, true) => SchedulerBatchPhase::Decode,
        (true, true) => SchedulerBatchPhase::Mixed,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_recirculation_prefill_plan, m)?)?;
    m.add_function(wrap_pyfunction!(parse_recirculation_prefill_plan, m)?)?;
    m.add_function(wrap_pyfunction!(build_recirculation_prefill_plan, m)?)?;
    m.add_class::<RecirculationPrefillPlan>()?;
    m.add_class::<RecirculationPrefillRequest>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_recirculation_prefill_plan, classify_scheduler_batch_phase, plan_packed_prefill_rows,
        validate_recirculation_prefill_plan, RecirculationPrefillPlan, SchedulerBatchPhase,
    };

    fn valid_recirculation_plan() -> RecirculationPrefillPlan {
        serde_json::from_str(
            r#"{
                "schema_version": 1,
                "phase": "prefill",
                "layer_count": 3,
                "total_rows": 3,
                "requests": [
                    {
                        "request_id": "req-a",
                        "request_ordinal": 0,
                        "scheduled_tokens": 2,
                        "prompt_tokens": 5,
                        "computed_tokens": 1,
                        "packed_row_start": 0,
                        "packed_row_end": 2,
                        "packed_row_indices": [0, 1],
                        "token_position_start": 1,
                        "token_position_end": 3,
                        "token_positions": [1, 2]
                    },
                    {
                        "request_id": "req-b",
                        "request_ordinal": 1,
                        "scheduled_tokens": 1,
                        "prompt_tokens": 2,
                        "computed_tokens": 1,
                        "packed_row_start": 2,
                        "packed_row_end": 3,
                        "packed_row_indices": [2],
                        "token_position_start": 1,
                        "token_position_end": 2,
                        "token_positions": [1]
                    }
                ]
            }"#,
        )
        .expect("test fixture is a valid closed plan")
    }

    #[test]
    fn mixed_request_rows_have_stable_packed_offsets_and_spans() {
        let plan = plan_packed_prefill_rows(vec![3, 2, 0], vec![5, 2, 4], vec![1, 2, 4], None)
            .expect("valid plan");
        assert_eq!(plan.request_ordinals, vec![0, 0, 0, 1, 1]);
        assert_eq!(plan.packed_row_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(plan.token_offsets, vec![1, 2, 3, 2, 3]);
        assert_eq!(
            plan.phases,
            vec!["prefill", "prefill", "prefill", "decode", "decode"]
        );
        assert_eq!(plan.request_spans[0].packed_row_start, 0);
        assert_eq!(plan.request_spans[0].packed_row_end, 3);
        assert_eq!(plan.request_spans[1].packed_row_start, 3);
        assert_eq!(plan.request_spans[1].packed_row_end, 5);
        assert_eq!(plan.request_spans[2].packed_row_start, 5);
        assert_eq!(plan.request_spans[2].packed_row_end, 5);
    }

    #[test]
    fn invalid_counts_are_rejected() {
        assert!(plan_packed_prefill_rows(vec![1], vec![1, 1], vec![0], None).is_err());
        assert!(plan_packed_prefill_rows(vec![-1], vec![1], vec![0], None).is_err());
        assert!(plan_packed_prefill_rows(vec![1], vec![1], vec![0], Some(2)).is_err());
    }

    #[test]
    fn computed_tokens_may_advance_past_prompt_during_decode() {
        let plan = plan_packed_prefill_rows(vec![2], vec![3], vec![5], None)
            .expect("decode rows after the prompt are valid");
        assert_eq!(plan.token_offsets, vec![5, 6]);
        assert_eq!(plan.phases, vec!["decode", "decode"]);
    }

    #[test]
    fn recirculation_prefill_plan_accepts_exact_rust_derived_rows() {
        assert!(validate_recirculation_prefill_plan(&valid_recirculation_plan()).is_ok());
    }

    #[test]
    fn recirculation_prefill_plan_rejects_decode_rows_and_packed_drift() {
        let mut decode = valid_recirculation_plan();
        decode.requests[0].prompt_tokens = 2;
        assert!(validate_recirculation_prefill_plan(&decode).is_err());

        let mut drifted_rows = valid_recirculation_plan();
        drifted_rows.requests[1].packed_row_start = 1;
        drifted_rows.requests[1].packed_row_end = 2;
        drifted_rows.requests[1].packed_row_indices = vec![1];
        assert!(validate_recirculation_prefill_plan(&drifted_rows).is_err());
    }

    #[test]
    fn recirculation_prefill_plan_rejects_closed_wire_drift() {
        let malformed = r#"{
            "schema_version": 1,
            "phase": "prefill",
            "layer_count": 1,
            "total_rows": 1,
            "requests": [],
            "unexpected": true
        }"#;
        assert!(serde_json::from_str::<RecirculationPrefillPlan>(malformed).is_err());
    }

    #[test]
    fn builder_derives_complete_prefill_requests_and_revalidates_them() {
        let plan = build_recirculation_prefill_plan(
            vec!["req-a".to_string(), "req-b".to_string()],
            vec![2, 1],
            vec![5, 2],
            vec![1, 1],
            3,
        )
        .expect("pure prefill batch should build");
        assert_eq!(plan.total_rows, 3);
        assert_eq!(plan.requests.len(), 2);
        assert_eq!(plan.requests[0].packed_row_indices, vec![0, 1]);
        assert_eq!(plan.requests[0].token_positions, vec![1, 2]);
        assert_eq!(plan.requests[1].packed_row_indices, vec![2]);
        assert_eq!(plan.requests[1].token_positions, vec![1]);
        assert!(validate_recirculation_prefill_plan(&plan).is_ok());
    }

    #[test]
    fn builder_returns_zero_row_noop_for_decode_only_batch() {
        let plan = build_recirculation_prefill_plan(
            vec!["req-a".to_string()],
            vec![2],
            vec![3],
            vec![5],
            3,
        )
        .expect("decode-only batch should be a no-op");
        assert_eq!(plan.total_rows, 0);
        assert!(plan.requests.is_empty());
        assert!(validate_recirculation_prefill_plan(&plan).is_ok());
    }

    #[test]
    fn builder_rejects_mixed_and_invalid_input_columns() {
        assert!(build_recirculation_prefill_plan(
            vec!["req-a".to_string()],
            vec![2],
            vec![2],
            vec![1],
            3,
        )
        .is_err());
        assert!(build_recirculation_prefill_plan(
            vec!["req-a".to_string(), "req-b".to_string()],
            vec![1, 1],
            vec![2, 2],
            vec![1, 2],
            3,
        )
        .is_err());
        assert!(build_recirculation_prefill_plan(
            vec!["req-a".to_string(), "req-a".to_string()],
            vec![1, 1],
            vec![2, 2],
            vec![1, 1],
            3,
        )
        .is_err());
        assert!(build_recirculation_prefill_plan(
            vec!["req-a".to_string()],
            vec![1],
            vec![2],
            vec![1],
            0,
        )
        .is_err());
    }

    #[test]
    fn classifies_full_scheduler_batch_without_collapsing_mixed_or_empty() {
        assert_eq!(
            classify_scheduler_batch_phase(vec![], vec![], vec![]).unwrap(),
            SchedulerBatchPhase::Empty
        );
        assert_eq!(
            classify_scheduler_batch_phase(vec![2], vec![4], vec![0]).unwrap(),
            SchedulerBatchPhase::Prefill
        );
        assert_eq!(
            classify_scheduler_batch_phase(vec![2], vec![4], vec![4]).unwrap(),
            SchedulerBatchPhase::Decode
        );
        assert_eq!(
            classify_scheduler_batch_phase(vec![2], vec![4], vec![3]).unwrap(),
            SchedulerBatchPhase::Mixed
        );
        assert_eq!(
            classify_scheduler_batch_phase(vec![1, 1], vec![2, 3], vec![0, 3]).unwrap(),
            SchedulerBatchPhase::Mixed
        );
    }
}

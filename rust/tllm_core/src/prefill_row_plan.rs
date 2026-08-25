//! Packed scheduler-row planning.
//!
//! This module derives generic host-side row metadata from scheduler columns.
//! Workload policy is deliberately absent: a `FiniteRecurrenceSpec` decides
//! which phases a region admits.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    #[pyo3(get)]
    pub(crate) request_ordinal: i64,
    #[pyo3(get)]
    pub(crate) packed_row_start: i64,
    #[pyo3(get)]
    pub(crate) packed_row_end: i64,
    #[pyo3(get)]
    pub(crate) token_offset_start: i64,
    #[pyo3(get)]
    pub(crate) token_offset_end: i64,
    #[pyo3(get)]
    pub(crate) prefill_row_count: i64,
    #[pyo3(get)]
    pub(crate) decode_row_count: i64,
}

/// Exact packed-row expansion. Request IDs and tensors remain owned by the
/// caller; ordinals index the request-order scheduler columns.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackedPrefillRowPlan {
    #[pyo3(get)]
    pub(crate) request_ordinals: Vec<i64>,
    #[pyo3(get)]
    pub(crate) packed_row_indices: Vec<i64>,
    #[pyo3(get)]
    pub(crate) token_offsets: Vec<i64>,
    #[pyo3(get)]
    pub(crate) phases: Vec<String>,
    #[pyo3(get)]
    pub(crate) request_spans: Vec<PrefillRowSpan>,
    #[pyo3(get)]
    pub(crate) total_rows: i64,
}

fn validation_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_non_negative(values: &[i64], name: &str) -> PyResult<()> {
    if let Some((index, value)) = values.iter().enumerate().find(|(_, value)| **value < 0) {
        return Err(validation_error(format!(
            "{name}[{index}] must be non-negative, got {value}"
        )));
    }
    Ok(())
}

/// Plan every row in a scheduler-packed step.
///
/// For request `r` with `scheduled[r]` rows, token position `i` is
/// `computed[r] + i`; positions below `prompt[r]` are prefill rows. The
/// optional expected total checks the first tensor dimension at the boundary.
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
            scheduled_tokens.len(), prompt_tokens.len(), computed_tokens.len()
        )));
    }
    validate_non_negative(&scheduled_tokens, "scheduled_tokens")?;
    validate_non_negative(&prompt_tokens, "prompt_tokens")?;
    validate_non_negative(&computed_tokens, "computed_tokens")?;
    if expected_total_rows.is_some_and(|value| value < 0) {
        return Err(validation_error("expected_total_rows must be non-negative"));
    }
    let total_rows = scheduled_tokens
        .iter()
        .try_fold(0_i64, |total, &scheduled| {
            total.checked_add(scheduled).ok_or_else(|| {
                validation_error("scheduled token total exceeds supported row index range")
            })
        })?;
    if expected_total_rows.is_some_and(|expected| expected != total_rows) {
        return Err(validation_error(format!(
            "packed row count mismatch: scheduled token total is {total_rows}"
        )));
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
        let row_end = row_base
            .checked_add(scheduled)
            .ok_or_else(|| validation_error("packed row index exceeds supported range"))?;
        let token_end = computed
            .checked_add(scheduled)
            .ok_or_else(|| validation_error("token offset exceeds supported range"))?;
        let prefill_rows = (prompt - computed).max(0).min(scheduled);
        for local_row in 0..scheduled {
            let token_offset = computed + local_row;
            request_ordinals.push(request_ordinal as i64);
            packed_row_indices.push(row_base + local_row);
            token_offsets.push(token_offset);
            phases.push(if token_offset < prompt {
                "prefill".into()
            } else {
                "decode".into()
            });
        }
        request_spans.push(PrefillRowSpan {
            request_ordinal: request_ordinal as i64,
            packed_row_start: row_base,
            packed_row_end: row_end,
            token_offset_start: computed,
            token_offset_end: token_end,
            prefill_row_count: prefill_rows,
            decode_row_count: scheduled - prefill_rows,
        });
        row_base = row_end;
    }
    Ok(PackedPrefillRowPlan {
        request_ordinals,
        packed_row_indices,
        token_offsets,
        phases,
        request_spans,
        total_rows,
    })
}

/// Compatibility alias for callers that use the older row-planning name.
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
    m.add_function(wrap_pyfunction!(plan_packed_prefill_rows, m)?)?;
    m.add_function(wrap_pyfunction!(plan_prefill_rows, m)?)?;
    m.add_function(wrap_pyfunction!(classify_scheduler_batch_phase, m)?)?;
    m.add_class::<PackedPrefillRowPlan>()?;
    m.add_class::<PrefillRowSpan>()?;
    m.add_class::<SchedulerBatchPhase>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{classify_scheduler_batch_phase, plan_packed_prefill_rows, SchedulerBatchPhase};

    #[test]
    fn derives_mixed_rows_without_workload_policy() {
        let plan = plan_packed_prefill_rows(vec![3, 2], vec![5, 2], vec![1, 2], None).unwrap();
        assert_eq!(plan.request_ordinals, vec![0, 0, 0, 1, 1]);
        assert_eq!(
            plan.phases,
            vec!["prefill", "prefill", "prefill", "decode", "decode"]
        );
        assert_eq!(plan.request_spans[1].packed_row_start, 3);
        assert_eq!(
            classify_scheduler_batch_phase(vec![2], vec![4], vec![3]).unwrap(),
            SchedulerBatchPhase::Mixed
        );
    }
}

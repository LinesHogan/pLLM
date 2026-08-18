//! Decode-row integer bookkeeping with no tensor/GPU values crossing FFI.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
struct DecodeLocalization {
    #[pyo3(get)]
    positions: Vec<i64>,
    #[pyo3(get)]
    row_idx: Vec<i64>,
    #[pyo3(get)]
    contiguous: bool,
}

/// Integer math of decode-row localization.
///
/// `is_decode[i]` marks request i in decode phase; `logits_indices[i]` is the
/// packed hidden row used for request i's sampling; `req_count` is the number
/// of known request ids (request-id resolution stays in Python).
#[pyfunction]
fn localize_decode_rows(
    is_decode: Vec<bool>,
    logits_indices: Vec<i64>,
    req_count: usize,
    num_actual_tokens: i64,
    max_decode_rows: i64,
) -> PyResult<DecodeLocalization> {
    if num_actual_tokens <= 0 || logits_indices.is_empty() {
        return Ok(DecodeLocalization {
            positions: vec![],
            row_idx: vec![],
            contiguous: true,
        });
    }
    let n_logits = logits_indices.len();
    let mut positions: Vec<i64> = Vec::new();
    for (i, &d) in is_decode.iter().enumerate() {
        if d && i < n_logits && i < req_count {
            positions.push(i as i64);
        }
    }
    if max_decode_rows > 0 {
        positions.truncate(max_decode_rows as usize);
    }
    if positions.is_empty() {
        return Ok(DecodeLocalization {
            positions,
            row_idx: vec![],
            contiguous: true,
        });
    }
    let first = positions[0];
    let contiguous = positions
        .iter()
        .enumerate()
        .all(|(k, &p)| p == first + k as i64);
    let row_idx: Vec<i64> = positions
        .iter()
        .map(|&p| logits_indices[p as usize])
        .collect();
    let min_idx = *row_idx.iter().min().unwrap();
    let max_idx = *row_idx.iter().max().unwrap();
    if min_idx < 0 || max_idx >= num_actual_tokens {
        return Err(PyRuntimeError::new_err(format!(
            "logits_indices out of range in prepare_inputs: min={} max={} num_actual_tokens={}",
            min_idx, max_idx, num_actual_tokens
        )));
    }
    Ok(DecodeLocalization {
        positions,
        row_idx,
        contiguous,
    })
}

#[pyclass]
#[derive(Clone)]
struct DecodePositions {
    #[pyo3(get)]
    positions: Vec<i64>,
    #[pyo3(get)]
    contiguous: bool,
}

/// Position selection half of decode localization: always host-known.
#[pyfunction]
fn localize_decode_positions(
    is_decode: Vec<bool>,
    n_logits: i64,
    req_count: i64,
    max_decode_rows: i64,
) -> DecodePositions {
    let mut positions: Vec<i64> = Vec::new();
    for (i, &d) in is_decode.iter().enumerate() {
        if d && (i as i64) < n_logits && (i as i64) < req_count {
            positions.push(i as i64);
        }
    }
    if max_decode_rows > 0 {
        positions.truncate(max_decode_rows as usize);
    }
    let contiguous = !positions.is_empty()
        && positions
            .iter()
            .enumerate()
            .all(|(k, &p)| p == positions[0] + k as i64);
    DecodePositions {
        positions,
        contiguous,
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(localize_decode_rows, m)?)?;
    m.add_function(wrap_pyfunction!(localize_decode_positions, m)?)?;
    m.add_class::<DecodeLocalization>()?;
    m.add_class::<DecodePositions>()?;
    Ok(())
}

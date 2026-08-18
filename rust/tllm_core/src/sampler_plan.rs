//! CPU-knowable sampler path selection and row-set partitioning.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashSet;

#[pyclass]
#[derive(Clone, Default)]
struct RowPartition {
    #[pyo3(get)]
    affected_greedy: Vec<i64>,
    #[pyo3(get)]
    affected_random: Vec<i64>,
    #[pyo3(get)]
    unaffected_greedy: Vec<i64>,
    #[pyo3(get)]
    unaffected_random: Vec<i64>,
}

#[pyclass]
#[derive(Clone)]
struct SamplerPlan {
    #[pyo3(get)]
    path: String,
    #[pyo3(get)]
    use_candidate_kernel: bool,
    #[pyo3(get)]
    dense_inplace_applied: bool,
    #[pyo3(get)]
    base: String,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn plan_sampler_step(
    backend: &str,
    temperature_mode: &str,
    n_rows: i64,
    n_affected: i64,
    n_greedy: i64,
    greedy_sampled_available: bool,
    candidate_eligible: bool,
) -> PyResult<SamplerPlan> {
    let is_dense = backend == "pre_filter_dense";
    let is_exact = backend == "post_filter_exact" || backend == "post_filter_dense_cache";

    if is_dense && n_affected == n_rows {
        if temperature_mode == "all_greedy" || n_greedy == n_rows {
            return Ok(plan_full("full_dense_greedy", false));
        }
        if temperature_mode == "all_random" || n_greedy == 0 {
            return Ok(plan_full("full_dense_random", false));
        }
        return Ok(plan_partitioned_meta(true));
    }
    if is_exact && n_affected == n_rows {
        if temperature_mode == "all_greedy" || n_greedy == n_rows {
            return Ok(plan_full("full_exact_greedy", false));
        }
        if temperature_mode == "all_random" || n_greedy == 0 {
            return Ok(plan_full("full_exact_random", candidate_eligible));
        }
        return Ok(plan_partitioned_meta(false));
    }
    match temperature_mode {
        "all_greedy" => {
            let base = if greedy_sampled_available {
                "greedy_sampled"
            } else {
                "argmax"
            };
            Ok(SamplerPlan {
                path: "partitioned".to_string(),
                use_candidate_kernel: false,
                dense_inplace_applied: false,
                base: base.to_string(),
            })
        }
        "all_random" | "mixed" => Ok(plan_partitioned_meta(false)),
        other => Err(PyRuntimeError::new_err(format!(
            "unsupported temperature mode `{other}`"
        ))),
    }
}

fn plan_full(path: &str, use_candidate_kernel: bool) -> SamplerPlan {
    SamplerPlan {
        path: path.to_string(),
        use_candidate_kernel,
        dense_inplace_applied: false,
        base: String::new(),
    }
}

fn sorted(mut values: Vec<i64>) -> Vec<i64> {
    values.sort_unstable();
    values
}

fn complement(n_rows: i64, rows: &[i64]) -> Vec<i64> {
    let set: HashSet<i64> = rows.iter().copied().collect();
    (0..n_rows).filter(|row| !set.contains(row)).collect()
}

fn intersect_sorted(left: &[i64], right: &[i64]) -> Vec<i64> {
    let set: HashSet<i64> = right.iter().copied().collect();
    let mut out: Vec<i64> = left
        .iter()
        .copied()
        .filter(|row| set.contains(row))
        .collect();
    out.sort_unstable();
    out
}

fn plan_partitioned_meta(dense_inplace_applied: bool) -> SamplerPlan {
    SamplerPlan {
        path: "partitioned".to_string(),
        use_candidate_kernel: false,
        dense_inplace_applied,
        base: "empty".to_string(),
    }
}

#[pyfunction]
fn partition_rows(
    n_rows: i64,
    affected_rows: Vec<i64>,
    greedy_rows: Vec<i64>,
    greedy_sampled_available: bool,
) -> RowPartition {
    if !greedy_sampled_available {
        return RowPartition {
            affected_random: sorted(affected_rows.clone()),
            unaffected_random: complement(n_rows, &affected_rows),
            ..Default::default()
        };
    }
    let affected = sorted(affected_rows);
    let greedy = sorted(greedy_rows);
    let affected_set: HashSet<i64> = affected.iter().copied().collect();
    let greedy_set: HashSet<i64> = greedy.iter().copied().collect();
    RowPartition {
        affected_greedy: intersect_sorted(&affected, &greedy),
        affected_random: affected
            .iter()
            .copied()
            .filter(|row| !greedy_set.contains(row))
            .collect(),
        unaffected_greedy: greedy
            .iter()
            .copied()
            .filter(|row| !affected_set.contains(row))
            .collect(),
        unaffected_random: (0..n_rows)
            .filter(|row| !affected_set.contains(row) && !greedy_set.contains(row))
            .collect(),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plan_sampler_step, m)?)?;
    m.add_function(wrap_pyfunction!(partition_rows, m)?)?;
    m.add_class::<RowPartition>()?;
    m.add_class::<SamplerPlan>()?;
    Ok(())
}

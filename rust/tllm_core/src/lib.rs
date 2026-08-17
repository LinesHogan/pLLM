//! tllm._core — small, cohesive Rust core for tLLM.
//!
//! Three coarse-grained entry points, one call per concern per step:
//! - `plan_sampler_step`: sampler intervention path + row partition decision
//! - `compile_execution_plan`: producer DAG compile/validation at startup
//! - `localize_decode_rows`: decode-row integer bookkeeping
//!
//! No torch/GPU types cross the FFI; tensors stay in Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// localize_decode_rows
// ---------------------------------------------------------------------------

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
    let row_idx: Vec<i64> = positions.iter().map(|&p| logits_indices[p as usize]).collect();
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

/// Position selection half of decode localization — the part that is always
/// host-known (no logits values, so no GPU sync). The device-side gather and
/// bounds diagnostics stay in Python.
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

// ---------------------------------------------------------------------------
// plan_sampler_step
// ---------------------------------------------------------------------------

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
    /// One of: full_dense_greedy, full_dense_random, full_exact_greedy,
    /// full_exact_random, partitioned.
    #[pyo3(get)]
    path: String,
    /// Full-row exact random path may use the fused candidate kernel.
    #[pyo3(get)]
    use_candidate_kernel: bool,
    /// Legacy quirk: mixed mode after an in-place full-dense modify.
    #[pyo3(get)]
    dense_inplace_applied: bool,
    /// Partitioned base rows: "greedy_sampled" | "argmax" | "empty" | "".
    #[pyo3(get)]
    base: String,
}

/// Decide the sampler intervention path from CPU-knowable metadata only
/// (counts, never row values — no GPU sync crosses this FFI).
///
/// `temperature_mode` is one of all_greedy/all_random/mixed; `n_affected` the
/// intervention row count; `n_greedy` the greedy-row count under mixed mode
/// (0 when no greedy_sampled is available); `candidate_eligible` marks the
/// fused min-p candidate kernel preconditions.
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
        // Mixed with partial greedy rows: the legacy path already modified
        // logits in place, then falls through to the partitioned section.
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
        "all_random" => Ok(plan_partitioned_meta(false)),
        "mixed" => Ok(plan_partitioned_meta(false)),
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

fn sorted(mut v: Vec<i64>) -> Vec<i64> {
    v.sort_unstable();
    v
}

fn complement(n_rows: i64, rows: &[i64]) -> Vec<i64> {
    let set: HashSet<i64> = rows.iter().copied().collect();
    (0..n_rows).filter(|r| !set.contains(r)).collect()
}

fn intersect_sorted(a: &[i64], b: &[i64]) -> Vec<i64> {
    let set: HashSet<i64> = b.iter().copied().collect();
    let mut out: Vec<i64> = a.iter().copied().filter(|r| set.contains(r)).collect();
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

/// Pure row-partition set math, for callers whose row sets are host-known
/// (tests, eager debug paths). The hot bridge path partitions on-device and
/// never calls this — values would cost a GPU sync.
#[pyfunction]
fn partition_rows(
    n_rows: i64,
    affected_rows: Vec<i64>,
    greedy_rows: Vec<i64>,
    greedy_sampled_available: bool,
) -> RowPartition {
    if !greedy_sampled_available {
        // No greedy mask: every non-affected row is vanilla, every affected
        // row is random.
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
            .filter(|r| !greedy_set.contains(r))
            .collect(),
        unaffected_greedy: greedy
            .iter()
            .copied()
            .filter(|r| !affected_set.contains(r))
            .collect(),
        unaffected_random: (0..n_rows)
            .filter(|r| !affected_set.contains(r) && !greedy_set.contains(r))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// compile_execution_plan
// ---------------------------------------------------------------------------

#[derive(Clone, FromPyObject)]
#[pyo3(from_item_all)]
struct PortSpecDesc {
    port: String,
    kind: String,
    writable: Option<bool>,
    write_modes: Option<Vec<String>>,
}

impl PortSpecDesc {
    fn writable(&self) -> bool {
        self.writable.unwrap_or(false)
    }
    fn write_modes(&self) -> Vec<String> {
        self.write_modes.clone().unwrap_or_default()
    }
}

#[derive(Clone, FromPyObject)]
#[pyo3(from_item_all)]
struct ProducerDesc {
    id: String,
    priority: Option<i64>,
    outputs: Vec<PortSpecDesc>,
    requires: Option<Vec<String>>,
    hook_points: Option<Vec<String>>,
    graph_safety: Option<String>,
}

impl ProducerDesc {
    fn priority(&self) -> i64 {
        self.priority.unwrap_or(0)
    }
    fn requires(&self) -> &[String] {
        self.requires.as_deref().unwrap_or(&[])
    }
    fn hook_points(&self) -> &[String] {
        self.hook_points.as_deref().unwrap_or(&[])
    }
    fn graph_safety(&self) -> &str {
        self.graph_safety.as_deref().unwrap_or("CAPTURE_SAFE")
    }
}

#[derive(Clone, FromPyObject)]
#[pyo3(from_item_all)]
struct ConsumerWriteDesc {
    kind: String,
    mode: Option<String>,
}

impl ConsumerWriteDesc {
    fn mode(&self) -> &str {
        self.mode.as_deref().unwrap_or("")
    }
}

#[derive(Clone, FromPyObject)]
#[pyo3(from_item_all)]
struct ConsumerDesc {
    id: String,
    reads: Option<Vec<String>>,
    writes: Option<Vec<ConsumerWriteDesc>>,
}

impl ConsumerDesc {
    fn reads(&self) -> &[String] {
        self.reads.as_deref().unwrap_or(&[])
    }
    fn writes(&self) -> &[ConsumerWriteDesc] {
        self.writes.as_deref().unwrap_or(&[])
    }
}

#[pyclass]
#[derive(Clone)]
struct CompiledPlan {
    #[pyo3(get)]
    ordered_producer_ids: Vec<String>,
    #[pyo3(get)]
    port_sources: Vec<(String, String)>,
}


/// Python repr() of a string (single-quoted; our vocabulary never contains
/// quotes or backslashes, so the simple form is exact).
fn qr(s: &str) -> String {
    format!("'{}'", s)
}

/// Python repr() of a list of strings.
fn qr_list(v: &[String]) -> String {
    let inner: Vec<String> = v.iter().map(|s| qr(s)).collect();
    format!("[{}]", inner.join(", "))
}

fn qr_list_str(v: &[&str]) -> String {
    let inner: Vec<String> = v.iter().map(|s| qr(s)).collect();
    format!("[{}]", inner.join(", "))
}

fn plan_err(msg: String) -> PyErr {
    PyRuntimeError::new_err(msg)
}

/// Compile the producer DAG. Error messages match the legacy Python
/// implementation verbatim (the oracle matrix pins them).
#[pyfunction]
#[pyo3(signature = (producers, consumers, enforced_kinds=None, supported_hooks=None, backend=None))]
fn compile_execution_plan(
    producers: Vec<ProducerDesc>,
    consumers: Vec<ConsumerDesc>,
    enforced_kinds: Option<Vec<String>>,
    supported_hooks: Option<Vec<String>>,
    backend: Option<(String, String)>,
) -> PyResult<CompiledPlan> {
    let enforced: Option<HashSet<String>> = enforced_kinds.map(|v| v.into_iter().collect());

    // 0. Consumer write validation (mirrors _validate_consumer_writes).
    for consumer in &consumers {
        for write in consumer.writes() {
            if let Some(enf) = &enforced {
                if !enf.contains(&write.kind) {
                    continue;
                }
            }
            let specs: Vec<&PortSpecDesc> = producers
                .iter()
                .flat_map(|p| p.outputs.iter())
                .filter(|s| s.kind == write.kind)
                .collect();
            if specs.is_empty() {
                return Err(plan_err(format!(
                    "consumer {} writes port kind {} but no registered producer offers it",
                    qr(&consumer.id), qr(&write.kind)
                )));
            }
            let writable: Vec<&&PortSpecDesc> = specs.iter().filter(|s| s.writable()).collect();
            if writable.is_empty() {
                return Err(plan_err(format!(
                    "consumer {} writes port kind {} but no producer for that kind declares it writable",
                    qr(&consumer.id), qr(&write.kind)
                )));
            }
            if !write.mode().is_empty() {
                let mut supported_modes: Vec<String> = writable
                    .iter()
                    .flat_map(|s| s.write_modes())
                    .collect();
                supported_modes.sort();
                supported_modes.dedup();
                if !supported_modes.iter().any(|m| m == write.mode()) {
                    return Err(plan_err(format!(
                        "consumer {} writes port kind {} with mode {} but producers only support: {}",
                        qr(&consumer.id), qr(&write.kind), qr(write.mode()), qr_list(&supported_modes)
                    )));
                }
            }
        }
    }

    // 1. Consumer-read resolution with priority-based per-port winners.
    let mut selected: HashMap<String, usize> = HashMap::new(); // producer_id -> index
    let mut selected_order: Vec<usize> = Vec::new();
    let mut port_winners: HashMap<String, usize> = HashMap::new(); // port -> producer index
    let mut port_winner_order: Vec<String> = Vec::new();
    let mut kinds_done: HashSet<String> = HashSet::new();

    for consumer in &consumers {
        for kind in consumer.reads() {
            if let Some(enf) = &enforced {
                if !enf.contains(kind) {
                    continue;
                }
            }
            let candidates: Vec<usize> = producers
                .iter()
                .enumerate()
                .filter(|(_, p)| p.outputs.iter().any(|s| &s.kind == kind))
                .map(|(i, _)| i)
                .collect();
            if candidates.is_empty() {
                return Err(plan_err(format!(
                    "consumer {} reads port kind {} but no registered producer offers it",
                    qr(&consumer.id), qr(kind)
                )));
            }
            if kinds_done.contains(kind) {
                continue;
            }
            kinds_done.insert(kind.clone());
            let mut by_port: HashMap<String, Vec<usize>> = HashMap::new();
            let mut port_order: Vec<String> = Vec::new();
            for &i in &candidates {
                for spec in &producers[i].outputs {
                    if &spec.kind == kind {
                        if !by_port.contains_key(&spec.port) {
                            port_order.push(spec.port.clone());
                        }
                        by_port.entry(spec.port.clone()).or_default().push(i);
                    }
                }
            }
            for port in &port_order {
                if port_winners.contains_key(port) {
                    continue;
                }
                let port_entries = &by_port[port];
                let best_priority = port_entries
                    .iter()
                    .map(|&i| producers[i].priority())
                    .max()
                    .unwrap();
                let top: Vec<usize> = port_entries
                    .iter()
                    .copied()
                    .filter(|&i| producers[i].priority() == best_priority)
                    .collect();
                if top.len() > 1 {
                    let mut ids: Vec<&str> =
                        top.iter().map(|&i| producers[i].id.as_str()).collect();
                    ids.sort();
                    return Err(plan_err(format!(
                        "port {} required by consumer {} has {} producers at priority {}: {}; disambiguate via priority",
                        port,
                        qr(&consumer.id),
                        top.len(),
                        best_priority,
                        qr_list_str(&ids)
                    )));
                }
                port_winners.insert(port.clone(), top[0]);
                port_winner_order.push(port.clone());
            }
            for port in &port_winner_order {
                let winner = port_winners[port];
                if !selected.contains_key(&producers[winner].id) {
                    selected.insert(producers[winner].id.clone(), winner);
                    selected_order.push(winner);
                }
            }
        }
    }

    // Transitively include producers required by selected producers.
    let by_id: HashMap<&str, usize> = producers
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id.as_str(), i))
        .collect();
    let mut offered: HashMap<String, usize> = HashMap::new();
    for (i, p) in producers.iter().enumerate() {
        for spec in &p.outputs {
            offered.insert(spec.port.clone(), i);
        }
    }
    for (port, &winner) in &port_winners {
        offered.insert(port.clone(), winner);
    }
    let mut included: HashMap<String, usize> = HashMap::new();
    let mut included_order: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = selected_order.clone();
    while let Some(i) = stack.pop() {
        let pid = producers[i].id.clone();
        if included.contains_key(&pid) {
            continue;
        }
        included.insert(pid, i);
        included_order.push(i);
        for req in producers[i].requires() {
            match offered.get(req) {
                None => {
                    return Err(plan_err(format!(
                        "producer {} requires port {} but no registered producer offers it",
                        qr(&producers[i].id), req
                    )));
                }
                Some(&src) => {
                    if !included.contains_key(&producers[src].id) {
                        stack.push(src);
                    }
                }
            }
        }
    }

    // 2. Topological order (Kahn).
    let mut indegree: HashMap<&str, i64> = included.keys().map(|k| (k.as_str(), 0)).collect();
    let mut edges: HashMap<&str, Vec<&str>> = included.keys().map(|k| (k.as_str(), vec![])).collect();
    let included_snapshot: Vec<(String, usize)> =
        included.iter().map(|(k, &v)| (k.clone(), v)).collect();
    for (pid, i) in &included_snapshot {
        for req in producers[*i].requires() {
            let src = offered[req];
            let src_id = producers[src].id.as_str();
            if included.contains_key(src_id) {
                edges.get_mut(src_id).unwrap().push(pid.as_str());
                *indegree.get_mut(pid.as_str()).unwrap() += 1;
            }
        }
    }
    let mut queue: Vec<&str> = indegree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(k, _)| *k)
        .collect();
    queue.sort();
    let mut ordered: Vec<String> = Vec::new();
    while !queue.is_empty() {
        let pid = queue.remove(0);
        ordered.push(pid.to_string());
        let mut nexts = edges[pid].clone();
        nexts.sort();
        for nxt in nexts {
            let d = indegree.get_mut(nxt).unwrap();
            *d -= 1;
            if *d == 0 {
                queue.push(nxt);
            }
        }
    }
    if ordered.len() != included.len() {
        let mut remaining: Vec<&str> = indegree
            .iter()
            .filter(|(_, &d)| d > 0)
            .map(|(k, _)| *k)
            .collect();
        remaining.sort();
        return Err(plan_err(format!(
            "producer dependency cycle involving: {}",
            qr_list_str(&remaining)
        )));
    }

    // 3. Backend feasibility: hook points and graph safety.
    let all_hooks: HashSet<&str> = [
        "load_model.pre",
        "load_model.post",
        "prepare_inputs.pre",
        "prepare_inputs.post",
        "layer.pre",
        "layer.post",
        "block.end",
        "stack.end",
        "execute_model.pre",
        "execute_model.post",
        "sample_tokens.pre",
        "sample_tokens.post",
    ]
    .into_iter()
    .collect();
    let supported: HashSet<String> = supported_hooks
        .map(|v| v.into_iter().collect())
        .unwrap_or_else(|| all_hooks.iter().map(|s| s.to_string()).collect());
    for pid in &ordered {
        let i = by_id[pid.as_str()];
        let mut unsupported: Vec<&str> = producers[i]
            .hook_points()
            .iter()
            .filter(|h| !supported.contains(h.as_str()))
            .map(|h| h.as_str())
            .collect();
        if !unsupported.is_empty() {
            unsupported.sort();
            return Err(plan_err(format!(
                "producer {} needs hook points the backend does not support: {}",
                qr(pid), qr_list_str(&unsupported)
            )));
        }
        if let Some((backend_id, backend_safety)) = &backend {
            if backend_safety == "EAGER_ONLY" && producers[i].graph_safety() != "EAGER_ONLY" {
                return Err(plan_err(format!(
                    "producer {} declares graph safety {} but backend {} is EAGER_ONLY",
                    qr(pid), producers[i].graph_safety(), qr(backend_id)
                )));
            }
        }
    }

    let mut port_sources: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    // included iteration order in Python is insertion order; mirror it and
    // let later duplicates overwrite, as dict comprehension does.
    let mut sources_map: HashMap<String, String> = HashMap::new();
    for &i in &included_order {
        for spec in &producers[i].outputs {
            sources_map.insert(spec.port.clone(), producers[i].id.clone());
        }
    }
    for &i in &included_order {
        for spec in &producers[i].outputs {
            if seen.insert(spec.port.clone()) {
                port_sources.push((spec.port.clone(), sources_map[&spec.port].clone()));
            }
        }
    }
    Ok(CompiledPlan {
        ordered_producer_ids: ordered,
        port_sources,
    })
}

// ---------------------------------------------------------------------------

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(localize_decode_rows, m)?)?;
    m.add_function(wrap_pyfunction!(localize_decode_positions, m)?)?;
    m.add_function(wrap_pyfunction!(plan_sampler_step, m)?)?;
    m.add_function(wrap_pyfunction!(partition_rows, m)?)?;
    m.add_function(wrap_pyfunction!(compile_execution_plan, m)?)?;
    m.add_class::<DecodeLocalization>()?;
    m.add_class::<DecodePositions>()?;
    m.add_class::<RowPartition>()?;
    m.add_class::<SamplerPlan>()?;
    m.add_class::<CompiledPlan>()?;
    Ok(())
}

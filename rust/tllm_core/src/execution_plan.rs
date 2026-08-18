//! Startup-time producer selection, schema validation, and DAG compilation.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, FromPyObject)]
struct PortSpecDesc {
    #[pyo3(attribute)]
    port: String,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    writable: bool,
    #[pyo3(attribute)]
    write_modes: Vec<String>,
    #[pyo3(attribute)]
    phases: Vec<String>,
}

#[derive(Clone, FromPyObject)]
struct ProducerDesc {
    #[pyo3(attribute)]
    id: String,
    #[pyo3(attribute)]
    priority: i64,
    #[pyo3(attribute)]
    outputs: Vec<PortSpecDesc>,
    #[pyo3(attribute)]
    requires: Vec<String>,
    #[pyo3(attribute)]
    hook_points: Vec<String>,
    #[pyo3(attribute)]
    graph_safety: String,
}

#[derive(Clone, FromPyObject)]
struct ConsumerWriteDesc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    mode: String,
    #[pyo3(attribute)]
    phase: String,
}

#[derive(Clone, FromPyObject)]
struct ConsumerReadDesc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    role: String,
    #[pyo3(attribute)]
    phase: String,
}

#[derive(Clone, FromPyObject)]
struct ConsumerFlowDesc {
    #[pyo3(attribute)]
    reads: Vec<ConsumerReadDesc>,
    #[pyo3(attribute)]
    writes: Vec<ConsumerWriteDesc>,
    #[pyo3(attribute)]
    bundle_key: Vec<String>,
}

#[derive(Clone, FromPyObject)]
struct ConsumerDesc {
    #[pyo3(attribute)]
    id: String,
    #[pyo3(attribute)]
    flows: Vec<ConsumerFlowDesc>,
}

#[pyclass]
#[derive(Clone)]
struct CompiledPlan {
    #[pyo3(get)]
    ordered_producer_ids: Vec<String>,
    #[pyo3(get)]
    port_sources: Vec<(String, String)>,
    #[pyo3(get)]
    capture_resident_producer_ids: Vec<String>,
}

fn spec_matches(spec: &PortSpecDesc, kind: &str, phase: &str) -> bool {
    spec.kind == kind
        && (phase.is_empty()
            || spec.phases.is_empty()
            || spec.phases.iter().any(|candidate| candidate == phase))
}

fn qr(value: &str) -> String {
    format!("'{}'", value)
}

fn qr_list(values: &[String]) -> String {
    let inner: Vec<String> = values.iter().map(|value| qr(value)).collect();
    format!("[{}]", inner.join(", "))
}

fn qr_list_str(values: &[&str]) -> String {
    let inner: Vec<String> = values.iter().map(|value| qr(value)).collect();
    format!("[{}]", inner.join(", "))
}

fn plan_err(message: String) -> PyErr {
    PyRuntimeError::new_err(message)
}

#[pyfunction]
#[pyo3(signature = (producers, consumers, enforced_kinds=None, supported_hooks=None, backend=None))]
fn compile_execution_plan(
    producers: Vec<ProducerDesc>,
    consumers: Vec<ConsumerDesc>,
    enforced_kinds: Option<Vec<String>>,
    supported_hooks: Option<Vec<String>>,
    backend: Option<(String, String)>,
) -> PyResult<CompiledPlan> {
    let enforced: Option<HashSet<String>> =
        enforced_kinds.map(|values| values.into_iter().collect());

    validate_consumers(&producers, &consumers, enforced.as_ref())?;

    let mut selected: HashMap<String, usize> = HashMap::new();
    let mut selected_order: Vec<usize> = Vec::new();
    let mut port_winners: HashMap<String, usize> = HashMap::new();
    let mut port_winner_order: Vec<String> = Vec::new();
    let mut selections_done: HashSet<(String, String)> = HashSet::new();

    for consumer in &consumers {
        for read in consumer.flows.iter().flat_map(|flow| &flow.reads) {
            let kind = &read.kind;
            if enforced.as_ref().is_some_and(|kinds| !kinds.contains(kind)) {
                continue;
            }
            let candidates: Vec<usize> = producers
                .iter()
                .enumerate()
                .filter(|(_, producer)| {
                    producer
                        .outputs
                        .iter()
                        .any(|spec| spec_matches(spec, kind, &read.phase))
                })
                .map(|(index, _)| index)
                .collect();
            if candidates.is_empty() {
                return Err(plan_err(format!(
                    "consumer {} reads port kind {} but no registered producer offers it",
                    qr(&consumer.id),
                    qr(kind)
                )));
            }
            if !selections_done.insert((kind.clone(), read.phase.clone())) {
                continue;
            }
            let mut by_port: HashMap<String, Vec<usize>> = HashMap::new();
            let mut port_order: Vec<String> = Vec::new();
            for &index in &candidates {
                for spec in &producers[index].outputs {
                    if spec_matches(spec, kind, &read.phase) {
                        if !by_port.contains_key(&spec.port) {
                            port_order.push(spec.port.clone());
                        }
                        by_port.entry(spec.port.clone()).or_default().push(index);
                    }
                }
            }
            for port in &port_order {
                if port_winners.contains_key(port) {
                    continue;
                }
                let entries = &by_port[port];
                let best_priority = entries
                    .iter()
                    .map(|&index| producers[index].priority)
                    .max()
                    .unwrap();
                let top: Vec<usize> = entries
                    .iter()
                    .copied()
                    .filter(|&index| producers[index].priority == best_priority)
                    .collect();
                if top.len() > 1 {
                    let mut ids: Vec<&str> = top
                        .iter()
                        .map(|&index| producers[index].id.as_str())
                        .collect();
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

    let by_id: HashMap<&str, usize> = producers
        .iter()
        .enumerate()
        .map(|(index, producer)| (producer.id.as_str(), index))
        .collect();
    let mut providers_by_port: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, producer) in producers.iter().enumerate() {
        for spec in &producer.outputs {
            providers_by_port
                .entry(spec.port.clone())
                .or_default()
                .push(index);
        }
    }
    let mut offered = port_winners;

    let mut included: HashMap<String, usize> = HashMap::new();
    let mut included_order: Vec<usize> = Vec::new();
    let mut stack = selected_order;
    while let Some(index) = stack.pop() {
        let producer_id = producers[index].id.clone();
        if included.contains_key(&producer_id) {
            continue;
        }
        included.insert(producer_id, index);
        included_order.push(index);
        for required in &producers[index].requires {
            let source = if let Some(&source) = offered.get(required) {
                source
            } else {
                let Some(candidates) = providers_by_port.get(required) else {
                    return Err(plan_err(format!(
                        "producer {} requires port {} but no registered producer offers it",
                        qr(&producers[index].id),
                        required
                    )));
                };
                let best_priority = candidates
                    .iter()
                    .map(|&candidate| producers[candidate].priority)
                    .max()
                    .expect("provider candidate list is non-empty");
                let top: Vec<usize> = candidates
                    .iter()
                    .copied()
                    .filter(|&candidate| producers[candidate].priority == best_priority)
                    .collect();
                if top.len() > 1 {
                    let mut ids: Vec<&str> = top
                        .iter()
                        .map(|&candidate| producers[candidate].id.as_str())
                        .collect();
                    ids.sort();
                    return Err(plan_err(format!(
                        "port {} required by producer {} has {} producers at priority {}: {}; disambiguate via priority",
                        required,
                        qr(&producers[index].id),
                        top.len(),
                        best_priority,
                        qr_list_str(&ids)
                    )));
                }
                let source = top[0];
                offered.insert(required.clone(), source);
                source
            };
            if !included.contains_key(&producers[source].id) {
                stack.push(source);
            }
        }
    }

    let ordered = topological_order(&producers, &included, &offered)?;
    validate_backend(
        &producers,
        &by_id,
        &ordered,
        supported_hooks,
        backend.as_ref(),
    )?;
    let capture_resident_producer_ids: Vec<String> = ordered
        .iter()
        .filter(|producer_id| {
            let producer = &producers[by_id[producer_id.as_str()]];
            producer.graph_safety != "EAGER_ONLY"
                && producer.hook_points.iter().any(|hook| hook == "layer.post")
        })
        .cloned()
        .collect();

    let mut port_sources: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut sources_map: HashMap<String, String> = HashMap::new();
    for &index in &included_order {
        for spec in &producers[index].outputs {
            sources_map.insert(spec.port.clone(), producers[index].id.clone());
        }
    }
    for &index in &included_order {
        for spec in &producers[index].outputs {
            if seen.insert(spec.port.clone()) {
                port_sources.push((spec.port.clone(), sources_map[&spec.port].clone()));
            }
        }
    }
    Ok(CompiledPlan {
        ordered_producer_ids: ordered,
        port_sources,
        capture_resident_producer_ids,
    })
}

fn validate_consumers(
    producers: &[ProducerDesc],
    consumers: &[ConsumerDesc],
    enforced: Option<&HashSet<String>>,
) -> PyResult<()> {
    let valid_bundle_keys: HashSet<&str> = [
        "engine_step_id",
        "generation_group_id",
        "phase",
        "request_id",
        "sample_idx",
    ]
    .into_iter()
    .collect();
    for consumer in consumers {
        for flow in &consumer.flows {
            let mut fields: HashSet<String> = HashSet::new();
            for read in &flow.reads {
                let role = read.role.trim();
                let field = if role.is_empty() { &read.kind } else { role };
                if !fields.insert(field.to_string()) {
                    return Err(plan_err(format!(
                        "consumer {} declares duplicate bundle field {}",
                        qr(&consumer.id),
                        qr(field)
                    )));
                }
            }
            let mut bundle_keys: HashSet<&str> = HashSet::new();
            for field in &flow.bundle_key {
                if !valid_bundle_keys.contains(field.as_str()) {
                    return Err(plan_err(format!(
                        "consumer {} declares unknown bundle_key field {}",
                        qr(&consumer.id),
                        qr(field)
                    )));
                }
                if !bundle_keys.insert(field) {
                    return Err(plan_err(format!(
                        "consumer {} declares duplicate bundle_key field {}",
                        qr(&consumer.id),
                        qr(field)
                    )));
                }
            }
        }
        for write in consumer.flows.iter().flat_map(|flow| &flow.writes) {
            if enforced.is_some_and(|kinds| !kinds.contains(&write.kind)) {
                continue;
            }
            let specs: Vec<&PortSpecDesc> = producers
                .iter()
                .flat_map(|producer| producer.outputs.iter())
                .filter(|spec| spec_matches(spec, &write.kind, &write.phase))
                .collect();
            if specs.is_empty() {
                return Err(plan_err(format!(
                    "consumer {} writes port kind {} but no registered producer offers it",
                    qr(&consumer.id),
                    qr(&write.kind)
                )));
            }
            let writable: Vec<&&PortSpecDesc> = specs.iter().filter(|spec| spec.writable).collect();
            if writable.is_empty() {
                return Err(plan_err(format!(
                    "consumer {} writes port kind {} but no producer for that kind declares it writable",
                    qr(&consumer.id),
                    qr(&write.kind)
                )));
            }
            if !write.mode.is_empty() {
                let mut supported_modes: Vec<String> = writable
                    .iter()
                    .flat_map(|spec| spec.write_modes.clone())
                    .collect();
                supported_modes.sort();
                supported_modes.dedup();
                if !supported_modes.iter().any(|mode| mode == &write.mode) {
                    return Err(plan_err(format!(
                        "consumer {} writes port kind {} with mode {} but producers only support: {}",
                        qr(&consumer.id),
                        qr(&write.kind),
                        qr(&write.mode),
                        qr_list(&supported_modes)
                    )));
                }
            }
        }
    }
    Ok(())
}

fn topological_order(
    producers: &[ProducerDesc],
    included: &HashMap<String, usize>,
    offered: &HashMap<String, usize>,
) -> PyResult<Vec<String>> {
    let mut indegree: HashMap<&str, i64> = included.keys().map(|key| (key.as_str(), 0)).collect();
    let mut edges: HashMap<&str, Vec<&str>> = included
        .keys()
        .map(|key| (key.as_str(), Vec::new()))
        .collect();
    let snapshot: Vec<(String, usize)> = included
        .iter()
        .map(|(key, &index)| (key.clone(), index))
        .collect();
    for (producer_id, index) in &snapshot {
        for required in &producers[*index].requires {
            let source = offered[required];
            let source_id = producers[source].id.as_str();
            if included.contains_key(source_id) {
                edges.get_mut(source_id).unwrap().push(producer_id.as_str());
                *indegree.get_mut(producer_id.as_str()).unwrap() += 1;
            }
        }
    }
    let mut queue: Vec<&str> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(key, _)| *key)
        .collect();
    queue.sort();
    let mut ordered: Vec<String> = Vec::new();
    while !queue.is_empty() {
        let producer_id = queue.remove(0);
        ordered.push(producer_id.to_string());
        let mut successors = edges[producer_id].clone();
        successors.sort();
        for successor in successors {
            let degree = indegree.get_mut(successor).unwrap();
            *degree -= 1;
            if *degree == 0 {
                queue.push(successor);
            }
        }
    }
    if ordered.len() != included.len() {
        let mut remaining: Vec<&str> = indegree
            .iter()
            .filter(|(_, degree)| **degree > 0)
            .map(|(key, _)| *key)
            .collect();
        remaining.sort();
        return Err(plan_err(format!(
            "producer dependency cycle involving: {}",
            qr_list_str(&remaining)
        )));
    }
    Ok(ordered)
}

fn validate_backend(
    producers: &[ProducerDesc],
    by_id: &HashMap<&str, usize>,
    ordered: &[String],
    supported_hooks: Option<Vec<String>>,
    backend: Option<&(String, String)>,
) -> PyResult<()> {
    let all_hooks = [
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
        "sampler.pre_draw",
        "sample_tokens.post",
    ];
    let supported: HashSet<String> = supported_hooks
        .map(|values| values.into_iter().collect())
        .unwrap_or_else(|| all_hooks.iter().map(|hook| hook.to_string()).collect());
    for producer_id in ordered {
        let index = by_id[producer_id.as_str()];
        let mut unsupported: Vec<&str> = producers[index]
            .hook_points
            .iter()
            .filter(|hook| !supported.contains(hook.as_str()))
            .map(String::as_str)
            .collect();
        if !unsupported.is_empty() {
            unsupported.sort();
            return Err(plan_err(format!(
                "producer {} needs hook points the backend does not support: {}",
                qr(producer_id),
                qr_list_str(&unsupported)
            )));
        }
        if let Some((backend_id, backend_safety)) = backend {
            if backend_safety == "EAGER_ONLY" && producers[index].graph_safety != "EAGER_ONLY" {
                return Err(plan_err(format!(
                    "producer {} declares graph safety {} but backend {} is EAGER_ONLY",
                    qr(producer_id),
                    producers[index].graph_safety,
                    qr(backend_id)
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_execution_plan, m)?)?;
    m.add_class::<CompiledPlan>()?;
    Ok(())
}

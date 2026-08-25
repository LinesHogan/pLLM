//! Generic finite recurrence scheduling.
//!
//! The compiler sees operation instances, dependence edges, and dependence
//! kinds a backend can satisfy inside an ordered-row kernel.  It never sees a
//! consumer or algorithm name.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, FromPyObject)]
struct IterationPointDesc {
    #[pyo3(attribute)]
    token: i64,
    #[pyo3(attribute)]
    layer: i64,
    #[pyo3(attribute)]
    pass_name: String,
    #[pyo3(attribute)]
    round: i64,
}

#[derive(Clone, FromPyObject)]
struct RecurrenceOpDesc {
    #[pyo3(attribute)]
    op_id: String,
    #[pyo3(attribute)]
    executor: String,
    #[pyo3(attribute)]
    point: IterationPointDesc,
}

#[derive(Clone, FromPyObject)]
struct RecurrenceEdgeDesc {
    #[pyo3(attribute)]
    source: String,
    #[pyo3(attribute)]
    target: String,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    distance: Vec<i64>,
    #[pyo3(attribute)]
    visibility: String,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompiledRecurrenceSchedule {
    #[pyo3(get)]
    groups: Vec<Vec<String>>,
    #[pyo3(get)]
    execution_order: Vec<String>,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    fusion_rejections: Vec<(String, String, String)>,
}

/// Closed execution tags for the exact Recirculation backend adapter.
///
/// This is deliberately an enum rather than an executor string. Python may
/// dispatch a vLLM layer or the mix callback from this tag, but it does not
/// parse an algorithm-specific executor naming convention.
#[pyclass(frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RecirculationPrefillOpKind {
    Normal,
    Mix,
    Recurrent,
}

/// Closed source of a model-layer input.  The only non-value input is the
/// original model row captured by the backend; all other inputs are stable
/// operation ordinals selected by Rust.
#[pyclass(frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RecirculationInputKind {
    Seed,
    Ordinal,
}

/// One backend primitive group.  Python dispatches this enum mechanically;
/// it must not infer the primitive or layer from the first operation.
#[pyclass(frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RecirculationPrefillGroupKind {
    ModelLayer,
    Mix,
}

/// One immutable operation in a Rust-derived exact Recirculation schedule.
///
/// ``data_predecessor_ordinals`` is the complete ordered data-input list.
/// Mix operations additionally name their three semantic normal inputs so a
/// tensor binder never has to reconstruct an incoming-edge map.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RecirculationPrefillOp {
    op_id: String,
    #[pyo3(get)]
    ordinal: i64,
    #[pyo3(get)]
    kind: RecirculationPrefillOpKind,
    #[pyo3(get)]
    token: i64,
    #[pyo3(get)]
    layer: i64,
    #[pyo3(get)]
    data_predecessor_ordinals: Vec<i64>,
    #[pyo3(get)]
    model_input_kind: Option<RecirculationInputKind>,
    #[pyo3(get)]
    model_input_ordinal: Option<i64>,
    #[pyo3(get)]
    mix_source_ordinal: Option<i64>,
    #[pyo3(get)]
    mix_destination_ordinal: Option<i64>,
    #[pyo3(get)]
    mix_final_normal_ordinal: Option<i64>,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RecirculationPrefillGroup {
    #[pyo3(get)]
    ordinal: i64,
    #[pyo3(get)]
    kind: RecirculationPrefillGroupKind,
    #[pyo3(get)]
    model_layer: Option<i64>,
    #[pyo3(get)]
    operations: Vec<RecirculationPrefillOp>,
    /// Region-local values whose final use occurs in this group.
    #[pyo3(get)]
    release_after: Vec<i64>,
}

/// Closed, backend-bindable finite recurrence schedule for exact prefill.
///
/// The groups are already in legal execution order. Every member is a typed
/// operation with stable ordinal inputs; no graph construction, effect
/// declaration, executor-string parsing, or anchor choice is left to Python.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RecirculationPrefillSchedule {
    #[pyo3(get)]
    token_count: i64,
    #[pyo3(get)]
    layer_count: i64,
    #[pyo3(get)]
    source_layer: i64,
    #[pyo3(get)]
    destination_layer: i64,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    groups: Vec<RecirculationPrefillGroup>,
    #[pyo3(get)]
    final_normal_ordinals: Vec<i64>,
    /// Values intentionally retained past the final group for observable
    /// outputs. All other operation values are released by a group action.
    #[pyo3(get)]
    retained_ordinals: Vec<i64>,
    #[pyo3(get)]
    fusion_rejections: Vec<(i64, i64, String)>,
}

fn validation_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

struct OrderedRowFusionPolicy {
    internalized_edges: Vec<String>,
    max_rows: usize,
}

fn parse_ordered_row_fusion_policy(policy: &str) -> PyResult<Option<OrderedRowFusionPolicy>> {
    if policy == "none" {
        return Ok(None);
    }
    let (max_rows_field, edges_field) = policy
        .split_once(';')
        .ok_or_else(|| validation_error("ordered-row fusion policy must be canonical"))?;
    let max_rows_text = max_rows_field
        .strip_prefix("max_rows=")
        .ok_or_else(|| validation_error("ordered-row fusion policy must start with max_rows="))?;
    let edge_text = edges_field
        .strip_prefix("internalized_edges=")
        .ok_or_else(|| {
            validation_error("ordered-row fusion policy must contain internalized_edges=")
        })?;
    let max_rows = max_rows_text.parse::<usize>().map_err(|_| {
        validation_error("ordered-row fusion policy max_rows must be a positive integer")
    })?;
    if max_rows < 2 || max_rows.to_string() != max_rows_text {
        return Err(validation_error(
            "ordered-row fusion policy max_rows must be canonical and >= 2",
        ));
    }
    let internalized_edges: Vec<String> = edge_text.split(',').map(str::to_owned).collect();
    if internalized_edges.is_empty()
        || internalized_edges.iter().any(|edge| edge.is_empty())
        || internalized_edges
            .windows(2)
            .any(|pair| pair[0].as_str() >= pair[1].as_str())
        || internalized_edges.iter().any(|edge| {
            edge.chars()
                .any(|character| matches!(character, ',' | ';' | '='))
        })
    {
        return Err(validation_error(
            "ordered-row fusion policy internalized_edges must be non-empty, unique, and sorted",
        ));
    }
    let canonical = format!(
        "max_rows={max_rows};internalized_edges={}",
        internalized_edges.join(",")
    );
    if policy != canonical {
        return Err(validation_error(
            "ordered-row fusion policy must be canonical",
        ));
    }
    Ok(Some(OrderedRowFusionPolicy {
        internalized_edges,
        max_rows,
    }))
}

fn validate_graph(ops: &[RecurrenceOpDesc], edges: &[RecurrenceEdgeDesc]) -> PyResult<()> {
    let mut known = HashSet::new();
    for op in ops {
        if op.op_id.is_empty() || op.executor.is_empty() || op.point.pass_name.is_empty() {
            return Err(validation_error("recurrence op strings must be non-empty"));
        }
        if op.point.token < 0 || op.point.round < 0 {
            return Err(validation_error(
                "recurrence token and round must be non-negative",
            ));
        }
        if !known.insert(op.op_id.clone()) {
            return Err(validation_error(format!(
                "recurrence graph contains duplicate op id {:?}",
                op.op_id
            )));
        }
    }
    for edge in edges {
        if !known.contains(&edge.source) || !known.contains(&edge.target) {
            return Err(validation_error(format!(
                "recurrence edge references unknown op: {}->{}",
                edge.source, edge.target
            )));
        }
        if edge.source == edge.target {
            return Err(validation_error("recurrence graph contains a self edge"));
        }
        if edge.kind == "carried"
            && (edge.distance.is_empty()
                || edge.distance.iter().any(|value| *value < 0)
                || !edge.distance.iter().any(|value| *value > 0))
        {
            return Err(validation_error(
                "carried dependence requires a non-negative, positive distance",
            ));
        }
        if edge.visibility != "current" && edge.visibility != "previous_round" {
            return Err(validation_error(format!(
                "unsupported recurrence visibility {:?}",
                edge.visibility
            )));
        }
    }
    Ok(())
}

fn topological_groups(
    ops: &[RecurrenceOpDesc],
    edges: &[RecurrenceEdgeDesc],
    pairs: &[(String, String)],
) -> PyResult<Vec<Vec<String>>> {
    let ordinal: HashMap<String, usize> = ops
        .iter()
        .enumerate()
        .map(|(index, op)| (op.op_id.clone(), index))
        .collect();
    let paired: HashSet<&str> = pairs
        .iter()
        .flat_map(|(source, target)| [source.as_str(), target.as_str()])
        .collect();
    let mut groups: Vec<Vec<String>> = pairs
        .iter()
        .map(|(source, target)| vec![source.clone(), target.clone()])
        .collect();
    for op in ops {
        if !paired.contains(op.op_id.as_str()) {
            groups.push(vec![op.op_id.clone()]);
        }
    }
    let mut group_of = HashMap::new();
    for (index, group) in groups.iter().enumerate() {
        for op_id in group {
            group_of.insert(op_id.clone(), index);
        }
    }
    let mut indegree = vec![0_usize; groups.len()];
    let mut successors: Vec<HashSet<usize>> = vec![HashSet::new(); groups.len()];
    for edge in edges {
        let source = group_of[&edge.source];
        let target = group_of[&edge.target];
        if source != target && successors[source].insert(target) {
            indegree[target] += 1;
        }
    }
    let group_ordinal: Vec<usize> = groups
        .iter()
        .map(|group| group.iter().map(|op_id| ordinal[op_id]).min().unwrap_or(0))
        .collect();
    let mut ready: Vec<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    ready.sort_by_key(|index| group_ordinal[*index]);
    let mut result = Vec::with_capacity(groups.len());
    while !ready.is_empty() {
        let index = ready.remove(0);
        result.push(groups[index].clone());
        let mut next: Vec<usize> = successors[index].iter().copied().collect();
        next.sort_by_key(|target| group_ordinal[*target]);
        for target in next {
            indegree[target] -= 1;
            if indegree[target] == 0 {
                ready.push(target);
            }
        }
        ready.sort_by_key(|candidate| group_ordinal[*candidate]);
    }
    if result.len() != groups.len() {
        return Err(validation_error(
            "recurrence instance graph or contracted fusion graph contains a cycle",
        ));
    }
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (ops, edges, internalized_edges=None, max_rows=2))]
fn compile_recurrence_schedule(
    ops: Vec<RecurrenceOpDesc>,
    edges: Vec<RecurrenceEdgeDesc>,
    internalized_edges: Option<Vec<String>>,
    max_rows: usize,
) -> PyResult<CompiledRecurrenceSchedule> {
    validate_graph(&ops, &edges)?;
    if max_rows < 1 {
        return Err(validation_error("recurrence max_rows must be positive"));
    }
    let internalized: HashSet<String> =
        internalized_edges.unwrap_or_default().into_iter().collect();
    let by_id: HashMap<&str, &RecurrenceOpDesc> =
        ops.iter().map(|op| (op.op_id.as_str(), op)).collect();
    let mut used = HashSet::new();
    let mut pairs = Vec::new();
    let mut rejections = Vec::new();
    if max_rows >= 2 {
        for edge in &edges {
            if !internalized.contains(&edge.kind) {
                continue;
            }
            let source = by_id[edge.source.as_str()];
            let target = by_id[edge.target.as_str()];
            let reason = if source.executor != target.executor {
                Some("executor_mismatch")
            } else if used.contains(&edge.source) || used.contains(&edge.target) {
                Some("row_already_fused")
            } else if source.point.token >= target.point.token {
                Some("noncausal_row_order")
            } else if source.point.layer != target.point.layer {
                Some("layer_mismatch")
            } else {
                None
            };
            if let Some(reason) = reason {
                rejections.push((edge.source.clone(), edge.target.clone(), reason.to_string()));
                continue;
            }
            used.insert(edge.source.clone());
            used.insert(edge.target.clone());
            pairs.push((edge.source.clone(), edge.target.clone()));
        }
    }
    let groups = topological_groups(&ops, &edges, &pairs)?;
    let execution_order = groups.iter().flatten().cloned().collect();
    Ok(CompiledRecurrenceSchedule {
        groups,
        execution_order,
        kind: if pairs.is_empty() {
            "serial".to_string()
        } else {
            "wavefront".to_string()
        },
        fusion_rejections: rejections,
    })
}

fn resolve_recirculation_layer(raw: i64, layer_count: i64, name: &str) -> PyResult<i64> {
    let resolved = if raw < 0 {
        layer_count.checked_add(raw).ok_or_else(|| {
            validation_error(format!(
                "Recirculation {name} is outside the supported layer index range"
            ))
        })?
    } else {
        raw
    };
    if resolved < 0 || resolved >= layer_count {
        return Err(validation_error(format!(
            "Recirculation {name} must resolve inside [0, layer_count)"
        )));
    }
    Ok(resolved)
}

fn append_recirculation_op(
    ops: &mut Vec<RecirculationPrefillOp>,
    kind: RecirculationPrefillOpKind,
    token: i64,
    layer: i64,
) -> String {
    let ordinal = ops.len() as i64;
    let op_id = format!("recirculation/{ordinal}");
    ops.push(RecirculationPrefillOp {
        op_id: op_id.clone(),
        ordinal,
        kind,
        token,
        layer,
        data_predecessor_ordinals: Vec::new(),
        model_input_kind: None,
        model_input_ordinal: None,
        mix_source_ordinal: None,
        mix_destination_ordinal: None,
        mix_final_normal_ordinal: None,
    });
    op_id
}

fn checked_recirculation_operation_count(
    token_count: i64,
    layer_count: i64,
    destination_layer: i64,
) -> PyResult<i64> {
    let per_token = layer_count
        .checked_mul(2)
        .and_then(|value| value.checked_sub(destination_layer))
        .ok_or_else(|| {
            validation_error(
                "Recirculation token/layer domain exceeds the supported operation range",
            )
        })?;
    token_count.checked_mul(per_token).ok_or_else(|| {
        validation_error("Recirculation token/layer domain exceeds the supported operation range")
    })
}

fn checked_usize(value: i64, name: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| validation_error(format!("Recirculation {name} exceeds the host index range")))
}

fn reserve_recirculation<T>(values: &mut Vec<T>, additional: usize) -> PyResult<()> {
    values.try_reserve_exact(additional).map_err(|_| {
        validation_error("Recirculation schedule allocation exceeds available host memory")
    })
}

fn add_edge(
    edges: &mut Vec<RecurrenceEdgeDesc>,
    source: &str,
    target: &str,
    kind: &str,
    distance: Vec<i64>,
) {
    edges.push(RecurrenceEdgeDesc {
        source: source.to_string(),
        target: target.to_string(),
        kind: kind.to_string(),
        distance,
        visibility: "current".to_string(),
    });
}

fn derive_recirculation_group_releases(
    groups: &mut [RecirculationPrefillGroup],
    retained_ordinals: &[i64],
) -> PyResult<()> {
    let retained: HashSet<i64> = retained_ordinals.iter().copied().collect();
    let mut producer_group = HashMap::new();
    for (group_index, group) in groups.iter().enumerate() {
        for operation in &group.operations {
            if producer_group
                .insert(operation.ordinal, group_index)
                .is_some()
            {
                return Err(validation_error(
                    "Recirculation group operations must have unique ordinals",
                ));
            }
        }
    }
    let mut last_use = producer_group.clone();
    for (group_index, group) in groups.iter().enumerate() {
        for operation in &group.operations {
            for source in &operation.data_predecessor_ordinals {
                let Some(entry) = last_use.get_mut(source) else {
                    return Err(validation_error(
                        "Recirculation operation data predecessor must identify a group operation",
                    ));
                };
                *entry = (*entry).max(group_index);
            }
        }
    }
    for group in groups.iter_mut() {
        group.release_after.clear();
    }
    for (ordinal, group_index) in last_use {
        if !retained.contains(&ordinal) {
            groups[group_index].release_after.push(ordinal);
        }
    }
    for group in groups.iter_mut() {
        group.release_after.sort_unstable();
    }
    Ok(())
}

fn validate_recirculation_prefill_schedule(
    schedule: &RecirculationPrefillSchedule,
    all_ops: &[RecirculationPrefillOp],
    edges: &[RecurrenceEdgeDesc],
) -> PyResult<()> {
    if schedule.token_count < 1 || schedule.layer_count < 1 {
        return Err(validation_error(
            "Recirculation schedule token_count and layer_count must be positive",
        ));
    }
    if schedule.destination_layer < 0
        || schedule.destination_layer >= schedule.source_layer
        || schedule.source_layer >= schedule.layer_count
    {
        return Err(validation_error(
            "Recirculation schedule layers must satisfy 0 <= destination < source < layer_count",
        ));
    }

    let expected_count = checked_recirculation_operation_count(
        schedule.token_count,
        schedule.layer_count,
        schedule.destination_layer,
    )?;
    let mut flattened = Vec::new();
    for (group_index, group) in schedule.groups.iter().enumerate() {
        if group.ordinal != group_index as i64 {
            return Err(validation_error(
                "Recirculation schedule group ordinals must be contiguous",
            ));
        }
        if group.operations.is_empty() || group.operations.len() > 2 {
            return Err(validation_error(
                "Recirculation schedule groups must contain one or two operations",
            ));
        }
        match group.kind {
            RecirculationPrefillGroupKind::ModelLayer => {
                let Some(layer) = group.model_layer else {
                    return Err(validation_error(
                        "Recirculation model-layer group must name its layer",
                    ));
                };
                if layer < 0 || layer >= schedule.layer_count {
                    return Err(validation_error(
                        "Recirculation model-layer group layer is outside its domain",
                    ));
                }
                if group.operations.iter().any(|op| {
                    !matches!(
                        op.kind,
                        RecirculationPrefillOpKind::Normal | RecirculationPrefillOpKind::Recurrent
                    ) || op.layer != layer
                }) {
                    return Err(validation_error(
                        "Recirculation model-layer group must contain one-layer normal or recurrent operations",
                    ));
                }
            }
            RecirculationPrefillGroupKind::Mix => {
                if group.model_layer.is_some()
                    || group.operations.len() != 1
                    || group.operations[0].kind != RecirculationPrefillOpKind::Mix
                {
                    return Err(validation_error(
                        "Recirculation mix group must contain exactly one mix operation",
                    ));
                }
            }
        }
        flattened.extend(group.operations.iter());
    }
    if flattened.len() as i64 != expected_count || all_ops.len() as i64 != expected_count {
        return Err(validation_error(
            "Recirculation schedule operation count does not cover its token/layer domain",
        ));
    }

    let mut by_ordinal = HashMap::new();
    let mut execution_position = HashMap::new();
    for (group_index, group) in schedule.groups.iter().enumerate() {
        for (within_group, op) in group.operations.iter().enumerate() {
            if op.ordinal < 0 || !by_ordinal.insert(op.ordinal, op).is_none() {
                return Err(validation_error(
                    "Recirculation schedule operation ordinals must be unique and non-negative",
                ));
            }
            execution_position.insert(op.ordinal, (group_index, within_group));
            if op.token < 0
                || op.token >= schedule.token_count
                || op.layer < 0
                || op.layer >= schedule.layer_count
            {
                return Err(validation_error(
                    "Recirculation schedule operation token/layer is outside its domain",
                ));
            }
            match op.kind {
                RecirculationPrefillOpKind::Normal => {
                    if op.model_input_kind.is_none() {
                        return Err(validation_error(
                            "Recirculation normal operation must name a typed model input",
                        ));
                    }
                    if op.mix_source_ordinal.is_some()
                        || op.mix_destination_ordinal.is_some()
                        || op.mix_final_normal_ordinal.is_some()
                    {
                        return Err(validation_error(
                            "only Recirculation mix operations may carry mix input ordinals",
                        ));
                    }
                }
                RecirculationPrefillOpKind::Mix => {
                    if op.model_input_kind.is_some() || op.model_input_ordinal.is_some() {
                        return Err(validation_error(
                            "Recirculation mix operation cannot carry a model input",
                        ));
                    }
                    if op.layer != schedule.destination_layer {
                        return Err(validation_error(
                            "Recirculation mix operation must execute at destination_layer",
                        ));
                    }
                    let (Some(source), Some(destination), Some(final_normal)) = (
                        op.mix_source_ordinal,
                        op.mix_destination_ordinal,
                        op.mix_final_normal_ordinal,
                    ) else {
                        return Err(validation_error(
                            "Recirculation mix operation must name source, destination, and final normal inputs",
                        ));
                    };
                    for (ordinal, expected_layer) in [
                        (source, schedule.source_layer),
                        (destination, schedule.destination_layer),
                        (final_normal, schedule.layer_count - 1),
                    ] {
                        let Some(input) = by_ordinal.get(&ordinal) else {
                            return Err(validation_error(
                                "Recirculation mix input ordinal must identify a schedule operation",
                            ));
                        };
                        if input.kind != RecirculationPrefillOpKind::Normal
                            || input.token != op.token
                            || input.layer != expected_layer
                            || !op.data_predecessor_ordinals.contains(&ordinal)
                        {
                            return Err(validation_error(
                                "Recirculation mix inputs must be current-token normal source, destination, and final outputs",
                            ));
                        }
                    }
                }
                RecirculationPrefillOpKind::Recurrent => {
                    if op.layer <= schedule.destination_layer {
                        return Err(validation_error(
                            "Recirculation recurrent operations must be above destination_layer",
                        ));
                    }
                    if op.mix_source_ordinal.is_some()
                        || op.mix_destination_ordinal.is_some()
                        || op.mix_final_normal_ordinal.is_some()
                    {
                        return Err(validation_error(
                            "only Recirculation mix operations may carry mix input ordinals",
                        ));
                    }
                    if op.model_input_kind != Some(RecirculationInputKind::Ordinal) {
                        return Err(validation_error(
                            "Recirculation recurrent operation must consume an ordinal model input",
                        ));
                    }
                }
            }
        }
    }
    for ordinal in 0..expected_count {
        if !by_ordinal.contains_key(&ordinal) {
            return Err(validation_error(
                "Recirculation schedule operation ordinals must be contiguous",
            ));
        }
    }

    let id_to_ordinal: HashMap<&str, i64> = all_ops
        .iter()
        .map(|op| (op.op_id.as_str(), op.ordinal))
        .collect();
    let mut expected_data: HashMap<i64, Vec<i64>> = HashMap::new();
    for edge in edges {
        let source = id_to_ordinal[edge.source.as_str()];
        let target = id_to_ordinal[edge.target.as_str()];
        let (source_group, source_index) = execution_position[&source];
        let (target_group, target_index) = execution_position[&target];
        if source_group > target_group
            || (source_group == target_group && source_index >= target_index)
        {
            return Err(validation_error(
                "Recirculation schedule does not place every dependency before its consumer",
            ));
        }
        if edge.kind == "data" {
            let data = expected_data.entry(target).or_default();
            if !data.contains(&source) {
                data.push(source);
            }
        }
    }
    for op in &flattened {
        if expected_data.get(&op.ordinal).cloned().unwrap_or_default()
            != op.data_predecessor_ordinals
        {
            return Err(validation_error(
                "Recirculation operation data predecessor ordinals disagree with its dependence graph",
            ));
        }
        match op.kind {
            RecirculationPrefillOpKind::Mix => {}
            RecirculationPrefillOpKind::Normal if op.layer == 0 => {
                if op.model_input_kind != Some(RecirculationInputKind::Seed)
                    || op.model_input_ordinal.is_some()
                    || !op.data_predecessor_ordinals.is_empty()
                {
                    return Err(validation_error(
                        "Recirculation layer-zero normal operation must consume only the seed input",
                    ));
                }
            }
            RecirculationPrefillOpKind::Normal | RecirculationPrefillOpKind::Recurrent => {
                if op.model_input_kind != Some(RecirculationInputKind::Ordinal)
                    || op.data_predecessor_ordinals.len() != 1
                    || op.model_input_ordinal != Some(op.data_predecessor_ordinals[0])
                {
                    return Err(validation_error(
                        "Recirculation model operation must consume its sole typed data predecessor",
                    ));
                }
            }
        }
    }

    if schedule.final_normal_ordinals.len() != checked_usize(schedule.token_count, "token_count")? {
        return Err(validation_error(
            "Recirculation schedule must expose one final normal output per token",
        ));
    }
    for (token, ordinal) in schedule.final_normal_ordinals.iter().enumerate() {
        let op = by_ordinal.get(ordinal).ok_or_else(|| {
            validation_error(
                "Recirculation final normal ordinal must identify a schedule operation",
            )
        })?;
        if op.kind != RecirculationPrefillOpKind::Normal
            || op.token != token as i64
            || op.layer != schedule.layer_count - 1
        {
            return Err(validation_error(
                "Recirculation final normal outputs must cover every token at the final layer",
            ));
        }
    }

    let retained: HashSet<i64> = schedule.retained_ordinals.iter().copied().collect();
    if retained.len() != schedule.retained_ordinals.len()
        || retained.len() != schedule.final_normal_ordinals.len()
        || !schedule
            .final_normal_ordinals
            .iter()
            .all(|ordinal| retained.contains(ordinal))
    {
        return Err(validation_error(
            "Recirculation retained ordinals must exactly name final normal outputs",
        ));
    }
    let mut last_use = HashMap::new();
    for (ordinal, (group_index, _)) in &execution_position {
        last_use.insert(*ordinal, *group_index);
    }
    for op in &flattened {
        let (consumer_group, _) = execution_position[&op.ordinal];
        for source in &op.data_predecessor_ordinals {
            let entry = last_use
                .get_mut(source)
                .expect("validated data predecessor must identify an operation");
            *entry = (*entry).max(consumer_group);
        }
    }
    let mut expected_releases: Vec<Vec<i64>> = vec![Vec::new(); schedule.groups.len()];
    for (ordinal, group_index) in last_use {
        if !retained.contains(&ordinal) {
            expected_releases[group_index].push(ordinal);
        }
    }
    for releases in &mut expected_releases {
        releases.sort_unstable();
    }
    for (group, expected) in schedule.groups.iter().zip(expected_releases) {
        let mut actual = group.release_after.clone();
        actual.sort_unstable();
        if actual != expected {
            return Err(validation_error(
                "Recirculation group release_after does not match Rust-derived value lifetime",
            ));
        }
    }
    Ok(())
}

/// Compile the exact Recirculation prefill recurrence from typed shape and
/// binding facts. This is the native alternative to Python graph building and
/// effect declaration; ordered-row fusion is selected only from the signed,
/// canonical backend policy supplied by the caller.
#[pyfunction]
fn compile_recirculation_prefill_schedule(
    token_count: i64,
    layer_count: i64,
    source_layer: i64,
    destination_layer: i64,
    ordered_row_fusion_policy: String,
) -> PyResult<RecirculationPrefillSchedule> {
    if token_count < 1 || layer_count < 1 {
        return Err(validation_error(
            "Recirculation token_count and layer_count must be positive",
        ));
    }
    let source_layer = resolve_recirculation_layer(source_layer, layer_count, "source_layer")?;
    let destination_layer =
        resolve_recirculation_layer(destination_layer, layer_count, "destination_layer")?;
    if destination_layer >= source_layer {
        return Err(validation_error(
            "Recirculation layers must satisfy destination_layer < source_layer",
        ));
    }
    let ordered_row_fusion = parse_ordered_row_fusion_policy(&ordered_row_fusion_policy)?;

    let operation_count =
        checked_recirculation_operation_count(token_count, layer_count, destination_layer)?;
    let token_len = checked_usize(token_count, "token_count")?;
    let layer_len = checked_usize(layer_count, "layer_count")?;
    let operation_len = checked_usize(operation_count, "operation count")?;

    let mut typed_ops = Vec::new();
    let mut ops = Vec::new();
    let mut edges = Vec::new();
    reserve_recirculation(&mut typed_ops, operation_len)?;
    reserve_recirculation(&mut ops, operation_len)?;
    reserve_recirculation(
        &mut edges,
        operation_len.checked_mul(3).ok_or_else(|| {
            validation_error("Recirculation dependence count exceeds the supported host range")
        })?,
    )?;
    let mut normal_ids = Vec::new();
    let mut recurrent_ids = Vec::new();
    reserve_recirculation(&mut normal_ids, token_len)?;
    reserve_recirculation(&mut recurrent_ids, token_len)?;
    for _ in 0..token_len {
        let mut normal_layers = Vec::new();
        let mut recurrent_layers = Vec::new();
        reserve_recirculation(&mut normal_layers, layer_len)?;
        reserve_recirculation(&mut recurrent_layers, layer_len)?;
        normal_layers.resize(layer_len, String::new());
        recurrent_layers.resize(layer_len, None);
        normal_ids.push(normal_layers);
        recurrent_ids.push(recurrent_layers);
    }
    let mut mix_ids = Vec::new();
    let mut final_normal_ordinals = Vec::new();
    reserve_recirculation(&mut mix_ids, token_len)?;
    reserve_recirculation(&mut final_normal_ordinals, token_len)?;

    for token in 0..token_count {
        for layer in 0..layer_count {
            let id = append_recirculation_op(
                &mut typed_ops,
                RecirculationPrefillOpKind::Normal,
                token,
                layer,
            );
            normal_ids[token as usize][layer as usize] = id.clone();
            ops.push(RecurrenceOpDesc {
                op_id: id.clone(),
                executor: format!("model_layer:{layer}"),
                point: IterationPointDesc {
                    token,
                    layer,
                    pass_name: "normal".to_string(),
                    round: 0,
                },
            });
            if layer > 0 {
                add_edge(
                    &mut edges,
                    &normal_ids[token as usize][(layer - 1) as usize],
                    &id,
                    "data",
                    Vec::new(),
                );
            }
            if token > 0 {
                let prior = if layer <= destination_layer {
                    &normal_ids[(token - 1) as usize][layer as usize]
                } else {
                    recurrent_ids[(token - 1) as usize][layer as usize]
                        .as_ref()
                        .expect("prior token recurrent operation must exist")
                };
                add_edge(
                    &mut edges,
                    prior,
                    &id,
                    if layer <= destination_layer {
                        "kv_write_read"
                    } else {
                        "kv_overwrite_read"
                    },
                    vec![1],
                );
            }
        }

        let mix_id = append_recirculation_op(
            &mut typed_ops,
            RecirculationPrefillOpKind::Mix,
            token,
            destination_layer,
        );
        let source_ordinal = typed_ops
            .iter()
            .find(|op| op.op_id == normal_ids[token as usize][source_layer as usize])
            .expect("source normal operation must exist")
            .ordinal;
        let destination_ordinal = typed_ops
            .iter()
            .find(|op| op.op_id == normal_ids[token as usize][destination_layer as usize])
            .expect("destination normal operation must exist")
            .ordinal;
        let final_normal_ordinal = typed_ops
            .iter()
            .find(|op| op.op_id == normal_ids[token as usize][(layer_count - 1) as usize])
            .expect("final normal operation must exist")
            .ordinal;
        let mix = typed_ops
            .last_mut()
            .expect("mix operation was just appended");
        mix.mix_source_ordinal = Some(source_ordinal);
        mix.mix_destination_ordinal = Some(destination_ordinal);
        mix.mix_final_normal_ordinal = Some(final_normal_ordinal);
        final_normal_ordinals.push(final_normal_ordinal);
        for dependency in [
            &normal_ids[token as usize][destination_layer as usize],
            &normal_ids[token as usize][source_layer as usize],
            &normal_ids[token as usize][(layer_count - 1) as usize],
        ] {
            add_edge(&mut edges, dependency, &mix_id, "data", Vec::new());
        }
        ops.push(RecurrenceOpDesc {
            op_id: mix_id.clone(),
            executor: "recirculation_mix".to_string(),
            point: IterationPointDesc {
                token,
                layer: destination_layer,
                pass_name: "mix".to_string(),
                round: 0,
            },
        });
        mix_ids.push(mix_id.clone());

        for layer in (destination_layer + 1)..layer_count {
            let id = append_recirculation_op(
                &mut typed_ops,
                RecirculationPrefillOpKind::Recurrent,
                token,
                layer,
            );
            recurrent_ids[token as usize][layer as usize] = Some(id.clone());
            let previous = if layer == destination_layer + 1 {
                &mix_ids[token as usize]
            } else {
                recurrent_ids[token as usize][(layer - 1) as usize]
                    .as_ref()
                    .expect("prior recurrent operation must exist")
            };
            add_edge(&mut edges, previous, &id, "data", Vec::new());
            add_edge(
                &mut edges,
                &normal_ids[token as usize][layer as usize],
                &id,
                "control",
                Vec::new(),
            );
            ops.push(RecurrenceOpDesc {
                op_id: id,
                executor: format!("model_layer:{layer}"),
                point: IterationPointDesc {
                    token,
                    layer,
                    pass_name: "recurrent".to_string(),
                    round: 0,
                },
            });
        }
    }
    let by_id: HashMap<String, usize> = typed_ops
        .iter()
        .enumerate()
        .map(|(index, op)| (op.op_id.clone(), index))
        .collect();
    for edge in &edges {
        if edge.kind != "data" {
            continue;
        }
        let source = typed_ops[by_id[&edge.source]].ordinal;
        let target = &mut typed_ops[by_id[&edge.target]];
        if !target.data_predecessor_ordinals.contains(&source) {
            target.data_predecessor_ordinals.push(source);
        }
    }
    for operation in &mut typed_ops {
        match operation.kind {
            RecirculationPrefillOpKind::Mix => {}
            RecirculationPrefillOpKind::Normal if operation.layer == 0 => {
                operation.model_input_kind = Some(RecirculationInputKind::Seed);
            }
            RecirculationPrefillOpKind::Normal | RecirculationPrefillOpKind::Recurrent => {
                let input = *operation.data_predecessor_ordinals.first().ok_or_else(|| {
                    validation_error("Recirculation model operation is missing a data input")
                })?;
                if operation.data_predecessor_ordinals.len() != 1 {
                    return Err(validation_error(
                        "Recirculation model operation must have exactly one data input",
                    ));
                }
                operation.model_input_kind = Some(RecirculationInputKind::Ordinal);
                operation.model_input_ordinal = Some(input);
            }
        }
    }

    let compiled = compile_recurrence_schedule(
        ops,
        edges.clone(),
        ordered_row_fusion
            .as_ref()
            .map(|policy| policy.internalized_edges.clone()),
        ordered_row_fusion
            .as_ref()
            .map_or(1, |policy| policy.max_rows),
    )?;
    let mut groups: Vec<RecirculationPrefillGroup> = compiled
        .groups
        .iter()
        .enumerate()
        .map(|(ordinal, group)| {
            let operations: Vec<RecirculationPrefillOp> = group
                .iter()
                .map(|id| typed_ops[by_id[id]].clone())
                .collect();
            let (kind, model_layer) = match operations[0].kind {
                RecirculationPrefillOpKind::Mix => (RecirculationPrefillGroupKind::Mix, None),
                RecirculationPrefillOpKind::Normal | RecirculationPrefillOpKind::Recurrent => (
                    RecirculationPrefillGroupKind::ModelLayer,
                    Some(operations[0].layer),
                ),
            };
            RecirculationPrefillGroup {
                ordinal: ordinal as i64,
                kind,
                model_layer,
                operations,
                release_after: Vec::new(),
            }
        })
        .collect();
    derive_recirculation_group_releases(&mut groups, &final_normal_ordinals)?;
    let id_to_ordinal: HashMap<&str, i64> = typed_ops
        .iter()
        .map(|op| (op.op_id.as_str(), op.ordinal))
        .collect();
    let schedule = RecirculationPrefillSchedule {
        token_count,
        layer_count,
        source_layer,
        destination_layer,
        kind: compiled.kind,
        groups,
        retained_ordinals: final_normal_ordinals.clone(),
        final_normal_ordinals,
        fusion_rejections: compiled
            .fusion_rejections
            .into_iter()
            .map(|(source, target, reason)| {
                (
                    id_to_ordinal[source.as_str()],
                    id_to_ordinal[target.as_str()],
                    reason,
                )
            })
            .collect(),
    };
    validate_recirculation_prefill_schedule(&schedule, &typed_ops, &edges)?;
    Ok(schedule)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_recirculation_prefill_schedule, m)?)?;
    m.add_class::<RecirculationInputKind>()?;
    m.add_class::<RecirculationPrefillGroupKind>()?;
    m.add_class::<RecirculationPrefillOpKind>()?;
    m.add_class::<RecirculationPrefillOp>()?;
    m.add_class::<RecirculationPrefillGroup>()?;
    m.add_class::<RecirculationPrefillSchedule>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        compile_recirculation_prefill_schedule, compile_recurrence_schedule, IterationPointDesc,
        RecirculationPrefillOpKind, RecurrenceEdgeDesc, RecurrenceOpDesc,
    };

    #[test]
    fn fuses_generic_ordered_rows_without_algorithm_identity() {
        let ops = vec![
            RecurrenceOpDesc {
                op_id: "a".into(),
                executor: "layer:1".into(),
                point: IterationPointDesc {
                    token: 0,
                    layer: 1,
                    pass_name: "x".into(),
                    round: 0,
                },
            },
            RecurrenceOpDesc {
                op_id: "b".into(),
                executor: "layer:1".into(),
                point: IterationPointDesc {
                    token: 1,
                    layer: 1,
                    pass_name: "y".into(),
                    round: 0,
                },
            },
        ];
        let edges = vec![RecurrenceEdgeDesc {
            source: "a".into(),
            target: "b".into(),
            kind: "kv_overwrite_read".into(),
            distance: vec![1],
            visibility: "current".into(),
        }];
        let plan =
            compile_recurrence_schedule(ops, edges, Some(vec!["kv_overwrite_read".into()]), 2)
                .expect("valid plan");
        assert_eq!(plan.kind, "wavefront");
        assert_eq!(plan.groups, vec![vec!["a", "b"]]);
    }

    #[test]
    fn exact_recirculation_descriptor_closes_graph_effects_and_data_inputs() {
        let schedule = compile_recirculation_prefill_schedule(
            2,
            4,
            2,
            0,
            "max_rows=2;internalized_edges=kv_overwrite_read".to_string(),
        )
        .expect("valid exact Recirculation schedule");
        assert_eq!(schedule.kind, "wavefront");
        assert_eq!(schedule.source_layer, 2);
        assert_eq!(schedule.destination_layer, 0);
        let ops: Vec<_> = schedule
            .groups
            .iter()
            .flat_map(|group| group.operations.iter())
            .collect();
        assert_eq!(ops.len(), 16);
        let mut ordinals: Vec<_> = ops.iter().map(|op| op.ordinal).collect();
        ordinals.sort_unstable();
        assert_eq!(ordinals, (0..16).collect::<Vec<_>>());
        let mixes: Vec<_> = ops
            .iter()
            .filter(|op| op.kind == RecirculationPrefillOpKind::Mix)
            .collect();
        assert_eq!(mixes.len(), 2);
        for mix in mixes {
            assert!(mix
                .data_predecessor_ordinals
                .contains(&mix.mix_source_ordinal.expect("source ordinal")));
            assert!(mix
                .data_predecessor_ordinals
                .contains(&mix.mix_destination_ordinal.expect("destination ordinal")));
            assert!(mix
                .data_predecessor_ordinals
                .contains(&mix.mix_final_normal_ordinal.expect("final ordinal")));
        }
        assert_eq!(schedule.final_normal_ordinals.len(), 2);
        let by_ordinal: HashMap<_, _> = ops.iter().map(|op| (op.ordinal, *op)).collect();
        for (token, ordinal) in schedule.final_normal_ordinals.iter().enumerate() {
            let op = by_ordinal[ordinal];
            assert_eq!(op.kind, RecirculationPrefillOpKind::Normal);
            assert_eq!(op.token, token as i64);
            assert_eq!(op.layer, 3);
        }
    }

    #[test]
    fn exact_recirculation_descriptor_rejects_invalid_static_contracts() {
        assert!(compile_recirculation_prefill_schedule(0, 4, 2, 0, "none".to_string()).is_err());
        assert!(compile_recirculation_prefill_schedule(1, 0, 0, 0, "none".to_string()).is_err());
        assert!(compile_recirculation_prefill_schedule(1, 4, 0, 0, "none".to_string()).is_err());
        assert!(compile_recirculation_prefill_schedule(1, 4, -5, 0, "none".to_string()).is_err());
    }

    #[test]
    fn recirculation_policy_is_parsed_in_rust() {
        let wavefront = compile_recirculation_prefill_schedule(
            2,
            4,
            2,
            0,
            "max_rows=2;internalized_edges=kv_overwrite_read".to_string(),
        )
        .expect("canonical policy must compile");
        assert_eq!(wavefront.kind, "wavefront");
        let serial = compile_recirculation_prefill_schedule(2, 4, 2, 0, "none".to_string())
            .expect("none policy must compile");
        assert_eq!(serial.kind, "serial");
        assert!(compile_recirculation_prefill_schedule(
            2,
            4,
            2,
            0,
            "max_rows=2;internalized_edges=kv_overwrite_read,control".to_string(),
        )
        .is_err());
    }
}

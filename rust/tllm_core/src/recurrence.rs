//! Generic finite recurrence planning.
//!
//! The JSON contract in this module is intentionally backend and workload
//! neutral. Rust owns graph validation, deterministic grouping, and value
//! liveness; Python binds the resulting typed operands to runtime resources.

use crate::prefill_row_plan::{PackedPrefillRowPlan, PrefillRowSpan, SchedulerBatchPhase};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};

pyo3::create_exception!(
    _core,
    TypedRecurrenceSpecializationRequired,
    pyo3::exceptions::PyValueError
);

fn validation_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

#[pyclass(frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OperandKind {
    Seed,
    RegionValue,
    BackendResource,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedOperand {
    #[pyo3(get)]
    kind: OperandKind,
    #[pyo3(get)]
    #[serde(skip_serializing_if = "Option::is_none")]
    binding_id: Option<String>,
    #[pyo3(get)]
    #[serde(skip_serializing_if = "Option::is_none")]
    value_id: Option<String>,
    #[pyo3(get)]
    #[serde(skip_serializing_if = "Option::is_none")]
    resource_id: Option<String>,
}

#[pymethods]
impl TypedOperand {
    #[staticmethod]
    fn seed(binding_id: String) -> Self {
        Self {
            kind: OperandKind::Seed,
            binding_id: Some(binding_id),
            value_id: None,
            resource_id: None,
        }
    }

    #[staticmethod]
    fn region_value(value_id: String) -> Self {
        Self {
            kind: OperandKind::RegionValue,
            binding_id: None,
            value_id: Some(value_id),
            resource_id: None,
        }
    }

    #[staticmethod]
    fn backend_resource(resource_id: String) -> Self {
        Self {
            kind: OperandKind::BackendResource,
            binding_id: None,
            value_id: None,
            resource_id: Some(resource_id),
        }
    }
}

#[pyclass(frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EffectKind {
    StateRead,
    StateWrite,
    StateTransition,
    VersionPublish,
    ExternIo,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedEffect {
    #[pyo3(get)]
    kind: EffectKind,
    #[pyo3(get)]
    resource_id: String,
}

#[pymethods]
impl TypedEffect {
    #[new]
    fn new(kind: EffectKind, resource_id: String) -> Self {
        Self { kind, resource_id }
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedOperation {
    #[pyo3(get)]
    operation_id: String,
    #[pyo3(get)]
    primitive_key: String,
    #[pyo3(get)]
    implementation_id: String,
    #[pyo3(get)]
    operands: Vec<TypedOperand>,
    #[pyo3(get)]
    results: Vec<String>,
    #[pyo3(get)]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    effects: Vec<TypedEffect>,
}

#[pymethods]
impl TypedOperation {
    #[new]
    #[pyo3(signature = (operation_id, primitive_key, implementation_id, operands, results, effects=Vec::new()))]
    fn new(
        py: Python<'_>,
        operation_id: String,
        primitive_key: String,
        implementation_id: String,
        operands: Vec<Py<TypedOperand>>,
        results: Vec<String>,
        effects: Vec<Py<TypedEffect>>,
    ) -> Self {
        Self {
            operation_id,
            primitive_key,
            implementation_id,
            operands: operands
                .into_iter()
                .map(|operand| operand.borrow(py).clone())
                .collect(),
            results,
            effects: effects
                .into_iter()
                .map(|effect| effect.borrow(py).clone())
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[pyclass]
struct TypedDependency {
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    kind: String,
}

#[pymethods]
impl TypedDependency {
    #[new]
    fn new(source: String, target: String, kind: String) -> Self {
        Self {
            source,
            target,
            kind,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MixedBatchPolicy {
    Reject,
    Noop,
    SelectAllowed,
}

impl<'de> Deserialize<'de> for MixedBatchPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer)?.as_str() {
            "reject" => Ok(Self::Reject),
            "noop" => Ok(Self::Noop),
            "select_allowed" => Ok(Self::SelectAllowed),
            _ => Err(serde::de::Error::custom(
                "mixed_batch must be reject, noop, or select_allowed",
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RowAdmission {
    allowed_phases: Vec<String>,
    mixed_batch: MixedBatchPolicy,
}

#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedStageAdmission {
    value: RowAdmission,
}

#[pymethods]
impl TypedStageAdmission {
    #[staticmethod]
    fn prefill_reject() -> Self {
        Self {
            value: RowAdmission {
                allowed_phases: vec!["prefill".into()],
                mixed_batch: MixedBatchPolicy::Reject,
            },
        }
    }

    #[staticmethod]
    fn prefill_select() -> Self {
        Self {
            value: RowAdmission {
                allowed_phases: vec!["prefill".into()],
                mixed_batch: MixedBatchPolicy::SelectAllowed,
            },
        }
    }

    #[staticmethod]
    fn decode_reject() -> Self {
        Self {
            value: RowAdmission {
                allowed_phases: vec!["decode".into()],
                mixed_batch: MixedBatchPolicy::Reject,
            },
        }
    }
}

#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedStageFusionPolicy {
    value: String,
}

#[pymethods]
impl TypedStageFusionPolicy {
    #[staticmethod]
    fn automatic() -> Self {
        Self {
            value: "auto".into(),
        }
    }

    #[staticmethod]
    fn serial() -> Self {
        Self {
            value: "none".into(),
        }
    }

    #[staticmethod]
    fn ordered(max_rows: usize, mut internalized_edges: Vec<String>) -> PyResult<Self> {
        internalized_edges.sort();
        internalized_edges.dedup();
        let value = format!(
            "max_rows={max_rows};internalized_edges={}",
            internalized_edges.join(",")
        );
        parse_fusion_policy(&value)?;
        Ok(Self { value })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[pyclass(name = "TypedRecurrence")]
struct FiniteRecurrenceSpec {
    schema_version: u64,
    region_id: String,
    implementation_id: String,
    row_admission: RowAdmission,
    fusion_policy: String,
    operations: Vec<TypedOperation>,
    dependencies: Vec<TypedDependency>,
    retained_values: Vec<String>,
}

#[pymethods]
impl FiniteRecurrenceSpec {
    #[new]
    #[pyo3(signature = (region_id, implementation_id, admission, fusion_policy, operations, dependencies, retained_values))]
    fn new(
        py: Python<'_>,
        region_id: String,
        implementation_id: String,
        admission: Py<TypedStageAdmission>,
        fusion_policy: Py<TypedStageFusionPolicy>,
        operations: Vec<Py<TypedOperation>>,
        dependencies: Vec<Py<TypedDependency>>,
        retained_values: Vec<String>,
    ) -> Self {
        Self {
            schema_version: 1,
            region_id,
            implementation_id,
            row_admission: admission.borrow(py).value.clone(),
            fusion_policy: fusion_policy.borrow(py).value.clone(),
            operations: operations
                .into_iter()
                .map(|operation| operation.borrow(py).clone())
                .collect(),
            dependencies: dependencies
                .into_iter()
                .map(|dependency| dependency.borrow(py).clone())
                .collect(),
            retained_values,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedExecutionGroup {
    #[pyo3(get)]
    group_id: u64,
    #[pyo3(get)]
    primitive_key: String,
    #[pyo3(get)]
    implementation_id: String,
    #[pyo3(get)]
    operations: Vec<TypedOperation>,
    #[pyo3(get)]
    release_after: Vec<String>,
}

/// Closed execution grouping derived from the accepted typed plan.
///
/// This is generic planner evidence: consumers cannot relabel singleton
/// groups as serial or infer a schedule from Python-side operation details.
#[pyclass(frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionGroupingKind {
    Serial,
    Wavefront,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedRecurrencePlan {
    #[pyo3(get)]
    schema_version: u64,
    spec: FiniteRecurrenceSpec,
    #[pyo3(get)]
    groups: Vec<TypedExecutionGroup>,
    #[pyo3(get)]
    grouping_kind: ExecutionGroupingKind,
    #[pyo3(get)]
    retained_values: Vec<String>,
    #[pyo3(get)]
    fusion_rejections: Vec<(String, String, String)>,
}

struct FusionPolicy {
    max_rows: usize,
    internalized_edges: BTreeSet<String>,
    automatic: bool,
}

fn is_nonempty(value: &str) -> bool {
    !value.is_empty()
}

fn parse_fusion_policy(value: &str) -> PyResult<FusionPolicy> {
    if value == "none" {
        return Ok(FusionPolicy {
            max_rows: 1,
            internalized_edges: BTreeSet::new(),
            automatic: false,
        });
    }
    if value == "auto" {
        return Ok(FusionPolicy {
            max_rows: 2,
            internalized_edges: BTreeSet::new(),
            automatic: true,
        });
    }
    let (max_rows_field, edge_field) = value
        .split_once(';')
        .ok_or_else(|| validation_error("fusion_policy must be canonical"))?;
    let max_rows_text = max_rows_field
        .strip_prefix("max_rows=")
        .ok_or_else(|| validation_error("fusion_policy must start with max_rows="))?;
    let edge_text = edge_field
        .strip_prefix("internalized_edges=")
        .ok_or_else(|| validation_error("fusion_policy must contain internalized_edges="))?;
    let max_rows = max_rows_text
        .parse::<usize>()
        .map_err(|_| validation_error("fusion_policy max_rows must be a positive integer"))?;
    if max_rows < 2 || max_rows.to_string() != max_rows_text {
        return Err(validation_error(
            "fusion_policy max_rows must be canonical and >= 2",
        ));
    }
    let edge_values: Vec<String> = edge_text.split(',').map(str::to_owned).collect();
    let internalized_edges: BTreeSet<String> = edge_values.iter().cloned().collect();
    if edge_values.is_empty()
        || edge_values.iter().any(|edge| !is_nonempty(edge))
        || internalized_edges.len() != edge_values.len()
        || edge_values.windows(2).any(|pair| pair[0] >= pair[1])
        || edge_values
            .iter()
            .any(|edge| edge.contains([',', ';', '=']))
    {
        return Err(validation_error(
            "fusion_policy internalized_edges must be non-empty, unique, and sorted",
        ));
    }
    if value
        != format!(
            "max_rows={max_rows};internalized_edges={}",
            edge_values.join(",")
        )
    {
        return Err(validation_error("fusion_policy must be canonical"));
    }
    Ok(FusionPolicy {
        max_rows,
        internalized_edges,
        automatic: false,
    })
}

fn validate_operand(operand: &TypedOperand) -> PyResult<()> {
    let valid = match operand.kind {
        OperandKind::Seed => {
            operand.binding_id.as_deref().is_some_and(is_nonempty)
                && operand.value_id.is_none()
                && operand.resource_id.is_none()
        }
        OperandKind::RegionValue => {
            operand.value_id.as_deref().is_some_and(is_nonempty)
                && operand.binding_id.is_none()
                && operand.resource_id.is_none()
        }
        OperandKind::BackendResource => {
            operand.resource_id.as_deref().is_some_and(is_nonempty)
                && operand.binding_id.is_none()
                && operand.value_id.is_none()
        }
    };
    if valid {
        Ok(())
    } else {
        Err(validation_error(
            "typed operand fields do not match its kind",
        ))
    }
}

fn validate_spec(
    spec: &FiniteRecurrenceSpec,
) -> PyResult<(
    FusionPolicy,
    HashMap<String, usize>,
    HashMap<String, String>,
)> {
    if spec.schema_version != 1 {
        return Err(validation_error(
            "unsupported finite recurrence schema_version",
        ));
    }
    if !is_nonempty(&spec.region_id) || !is_nonempty(&spec.implementation_id) {
        return Err(validation_error(
            "region_id and implementation_id must be non-empty",
        ));
    }
    if spec.row_admission.allowed_phases.is_empty()
        || spec
            .row_admission
            .allowed_phases
            .iter()
            .any(|phase| !matches!(phase.as_str(), "prefill" | "decode"))
        || spec
            .row_admission
            .allowed_phases
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(validation_error(
            "allowed_phases must be non-empty, unique, and sorted supported phases",
        ));
    }
    let policy = parse_fusion_policy(&spec.fusion_policy)?;
    if spec.operations.is_empty() {
        return Err(validation_error("operations must be non-empty"));
    }
    let mut operation_index = HashMap::new();
    let mut producer = HashMap::new();
    for (index, operation) in spec.operations.iter().enumerate() {
        if !is_nonempty(&operation.operation_id)
            || !is_nonempty(&operation.primitive_key)
            || !is_nonempty(&operation.implementation_id)
            || operation.implementation_id != spec.implementation_id
            || !operation_index
                .insert(operation.operation_id.clone(), index)
                .is_none()
        {
            return Err(validation_error(
                "operation identifiers must be unique and match the plan implementation",
            ));
        }
        if operation.results.is_empty() {
            return Err(validation_error(
                "each operation must produce at least one result",
            ));
        }
        for operand in &operation.operands {
            validate_operand(operand)?;
        }
        for effect in &operation.effects {
            if !is_nonempty(&effect.resource_id) {
                return Err(validation_error("effect fields must be non-empty"));
            }
        }
        for result in &operation.results {
            if !is_nonempty(result)
                || producer
                    .insert(result.clone(), operation.operation_id.clone())
                    .is_some()
            {
                return Err(validation_error(
                    "result identifiers must be non-empty and globally unique",
                ));
            }
        }
    }
    for operation in &spec.operations {
        for operand in &operation.operands {
            if let Some(value_id) = &operand.value_id {
                if !producer.contains_key(value_id) {
                    return Err(validation_error(
                        "region_value operand references an unknown result",
                    ));
                }
            }
        }
    }
    let mut seen_dependencies = HashSet::new();
    for dependency in &spec.dependencies {
        if !operation_index.contains_key(&dependency.source)
            || !operation_index.contains_key(&dependency.target)
            || dependency.source == dependency.target
            || !is_nonempty(&dependency.kind)
            || !seen_dependencies.insert((
                dependency.source.as_str(),
                dependency.target.as_str(),
                dependency.kind.as_str(),
            ))
        {
            return Err(validation_error(
                "dependencies must be unique and reference distinct known operations",
            ));
        }
    }
    let mut retained = HashSet::new();
    for value in &spec.retained_values {
        if !producer.contains_key(value) || !retained.insert(value) {
            return Err(validation_error(
                "retained_values must be unique produced results",
            ));
        }
    }
    Ok((policy, operation_index, producer))
}

fn derived_dependencies(
    spec: &FiniteRecurrenceSpec,
    producer: &HashMap<String, String>,
) -> Vec<TypedDependency> {
    let mut all = spec.dependencies.clone();
    let mut seen: HashSet<(String, String, String)> = all
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone(), edge.kind.clone()))
        .collect();
    for operation in &spec.operations {
        for operand in &operation.operands {
            if let Some(value_id) = &operand.value_id {
                let source = producer[value_id].clone();
                let edge = (
                    source,
                    operation.operation_id.clone(),
                    "region_value".to_string(),
                );
                if seen.insert(edge.clone()) {
                    all.push(TypedDependency {
                        source: edge.0,
                        target: edge.1,
                        kind: edge.2,
                    });
                }
            }
        }
    }
    all
}

fn stable_topological_order(
    operations: &[TypedOperation],
    operation_index: &HashMap<String, usize>,
    dependencies: &[TypedDependency],
) -> PyResult<Vec<String>> {
    let mut indegree = vec![0_usize; operations.len()];
    let mut successors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); operations.len()];
    for dependency in dependencies {
        let source = operation_index[&dependency.source];
        let target = operation_index[&dependency.target];
        if successors[source].insert(target) {
            indegree[target] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    let mut order = Vec::with_capacity(operations.len());
    while let Some(index) = ready.pop_first() {
        order.push(operations[index].operation_id.clone());
        for target in &successors[index] {
            indegree[*target] -= 1;
            if indegree[*target] == 0 {
                ready.insert(*target);
            }
        }
    }
    if order.len() != operations.len() {
        Err(validation_error("recurrence dependencies contain a cycle"))
    } else {
        Ok(order)
    }
}

/// Order contracted execution groups after fusion. A valid operation DAG can
/// become cyclic when a later operation is appended to an earlier group, so
/// this is a separate legality check from operation-level topological order.
fn stable_group_order(
    groups: &[TypedExecutionGroup],
    group_of: &HashMap<String, usize>,
    dependencies: &[TypedDependency],
) -> Option<Vec<usize>> {
    let mut indegree = vec![0_usize; groups.len()];
    let mut successors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); groups.len()];
    for dependency in dependencies {
        let (Some(&source), Some(&target)) = (
            group_of.get(&dependency.source),
            group_of.get(&dependency.target),
        ) else {
            // During incremental contraction, future operations do not have
            // a group yet. The final call below sees the complete map.
            continue;
        };
        if source != target && successors[source].insert(target) {
            indegree[target] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    let mut order = Vec::with_capacity(groups.len());
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for target in &successors[index] {
            indegree[*target] -= 1;
            if indegree[*target] == 0 {
                ready.insert(*target);
            }
        }
    }
    (order.len() == groups.len()).then_some(order)
}

fn compile_typed(spec: FiniteRecurrenceSpec) -> PyResult<TypedRecurrencePlan> {
    let (mut policy, operation_index, producer) = validate_spec(&spec)?;
    let dependencies = derived_dependencies(&spec, &producer);
    if policy.automatic {
        policy.internalized_edges = dependencies
            .iter()
            .map(|dependency| dependency.kind.clone())
            .collect();
    }
    let order = stable_topological_order(&spec.operations, &operation_index, &dependencies)?;
    let mut inbound: HashMap<&str, Vec<&TypedDependency>> = HashMap::new();
    for dependency in &dependencies {
        inbound
            .entry(&dependency.target)
            .or_default()
            .push(dependency);
    }
    let by_id: HashMap<&str, &TypedOperation> = spec
        .operations
        .iter()
        .map(|op| (op.operation_id.as_str(), op))
        .collect();
    let mut groups: Vec<TypedExecutionGroup> = Vec::new();
    let mut group_of = HashMap::new();
    let mut fusion_rejections = Vec::new();
    for operation_id in &order {
        let operation = by_id[operation_id.as_str()];
        let candidates = inbound
            .get(operation_id.as_str())
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let predecessor_groups: BTreeSet<usize> = candidates
            .iter()
            .filter(|edge| policy.internalized_edges.contains(&edge.kind))
            .filter_map(|edge| group_of.get(&edge.source).copied())
            .collect();
        let mut append_to = None;
        if predecessor_groups.len() == 1 {
            let group_index = *predecessor_groups.iter().next().unwrap();
            let group = &groups[group_index];
            if group.operations.len() < policy.max_rows
                && group.primitive_key == operation.primitive_key
                && group.implementation_id == operation.implementation_id
                && group.operations.last().is_some_and(|previous| {
                    candidates.iter().any(|edge| {
                        edge.source == previous.operation_id
                            && policy.internalized_edges.contains(&edge.kind)
                    })
                })
            {
                append_to = Some(group_index);
            }
        }
        let group_index = match append_to {
            Some(index) => {
                groups[index].operations.push(operation.clone());
                index
            }
            None => {
                let index = groups.len();
                groups.push(TypedExecutionGroup {
                    group_id: index as u64,
                    primitive_key: operation.primitive_key.clone(),
                    implementation_id: operation.implementation_id.clone(),
                    operations: vec![operation.clone()],
                    release_after: Vec::new(),
                });
                index
            }
        };
        group_of.insert(operation.operation_id.clone(), group_index);
        if append_to.is_some() && stable_group_order(&groups, &group_of, &dependencies).is_none() {
            let fused_source = groups[group_index]
                .operations
                .get(groups[group_index].operations.len() - 2)
                .expect("newly fused group has a predecessor")
                .operation_id
                .clone();
            groups[group_index].operations.pop();
            let index = groups.len();
            groups.push(TypedExecutionGroup {
                group_id: index as u64,
                primitive_key: operation.primitive_key.clone(),
                implementation_id: operation.implementation_id.clone(),
                operations: vec![operation.clone()],
                release_after: Vec::new(),
            });
            group_of.insert(operation.operation_id.clone(), index);
            fusion_rejections.push((
                fused_source,
                operation.operation_id.clone(),
                "contracted_group_cycle".into(),
            ));
        }
    }
    let group_order = stable_group_order(&groups, &group_of, &dependencies)
        .ok_or_else(|| validation_error("accepted fusion produced a cyclic group graph"))?;
    let mut old_to_new = vec![0_usize; groups.len()];
    let mut reordered = Vec::with_capacity(groups.len());
    for (new_index, old_index) in group_order.into_iter().enumerate() {
        old_to_new[old_index] = new_index;
        let mut group = groups[old_index].clone();
        group.group_id = new_index as u64;
        reordered.push(group);
    }
    for group_index in group_of.values_mut() {
        *group_index = old_to_new[*group_index];
    }
    groups = reordered;
    let retained: HashSet<&str> = spec.retained_values.iter().map(String::as_str).collect();
    let mut last_use: HashMap<&str, usize> = HashMap::new();
    for operation in &spec.operations {
        let producer_group = group_of[&operation.operation_id];
        for result in &operation.results {
            last_use.insert(result, producer_group);
        }
    }
    for operation in &spec.operations {
        let consumer_group = group_of[&operation.operation_id];
        for operand in &operation.operands {
            if let Some(value_id) = &operand.value_id {
                last_use
                    .entry(value_id)
                    .and_modify(|last| *last = (*last).max(consumer_group));
            }
        }
    }
    for (value, group_index) in last_use {
        if !retained.contains(value) {
            groups[group_index].release_after.push(value.to_string());
        }
    }
    for group in &mut groups {
        group.release_after.sort();
    }
    let grouping_kind = if groups.iter().any(|group| group.operations.len() > 1) {
        ExecutionGroupingKind::Wavefront
    } else {
        ExecutionGroupingKind::Serial
    };
    Ok(TypedRecurrencePlan {
        schema_version: 1,
        grouping_kind,
        retained_values: spec.retained_values.clone(),
        spec,
        groups,
        fusion_rejections,
    })
}

/// One opaque resource coordinate selected by an artifact-owning workload.
/// Rust validates identity/coverage; the workload alone interprets the kind.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedStageBinding {
    #[pyo3(get)]
    binding_id: String,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    coordinates: Vec<i64>,
}

#[pymethods]
impl TypedStageBinding {
    #[new]
    fn new(binding_id: String, kind: String, coordinates: Vec<i64>) -> Self {
        Self {
            binding_id,
            kind,
            coordinates,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedStageResourceBinding {
    #[pyo3(get)]
    resource_id: String,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    coordinates: Vec<i64>,
}

#[pymethods]
impl TypedStageResourceBinding {
    #[new]
    fn new(resource_id: String, kind: String, coordinates: Vec<i64>) -> Self {
        Self {
            resource_id,
            kind,
            coordinates,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedStageOutputBinding {
    #[pyo3(get)]
    output_id: String,
    #[pyo3(get)]
    value_id: String,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    coordinates: Vec<i64>,
}

#[pymethods]
impl TypedStageOutputBinding {
    #[new]
    fn new(output_id: String, value_id: String, kind: String, coordinates: Vec<i64>) -> Self {
        Self {
            output_id,
            value_id,
            kind,
            coordinates,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[pyclass(name = "TypedRecurrenceStageDefinition")]
struct TypedRecurrenceStageSpec {
    schema_version: u64,
    shape_specialization: Vec<(String, String)>,
    row_count: u64,
    request_count: u64,
    recurrence: FiniteRecurrenceSpec,
    seed_bindings: Vec<TypedStageBinding>,
    resource_bindings: Vec<TypedStageResourceBinding>,
    output_bindings: Vec<TypedStageOutputBinding>,
}

#[pymethods]
impl TypedRecurrenceStageSpec {
    #[new]
    fn new(
        py: Python<'_>,
        row_count: u64,
        recurrence: Py<FiniteRecurrenceSpec>,
        seed_bindings: Vec<Py<TypedStageBinding>>,
        resource_bindings: Vec<Py<TypedStageResourceBinding>>,
        output_bindings: Vec<Py<TypedStageOutputBinding>>,
    ) -> Self {
        let mut seed_bindings: Vec<_> = seed_bindings
            .into_iter()
            .map(|binding| binding.borrow(py).clone())
            .collect();
        seed_bindings.sort_by(|left, right| left.binding_id.cmp(&right.binding_id));
        let mut resource_bindings: Vec<_> = resource_bindings
            .into_iter()
            .map(|binding| binding.borrow(py).clone())
            .collect();
        resource_bindings.sort_by(|left, right| left.resource_id.cmp(&right.resource_id));
        let mut output_bindings: Vec<_> = output_bindings
            .into_iter()
            .map(|binding| binding.borrow(py).clone())
            .collect();
        output_bindings.sort_by(|left, right| left.output_id.cmp(&right.output_id));
        Self {
            schema_version: 1,
            shape_specialization: vec![
                ("stage/request_count".into(), "1".into()),
                ("stage/row_count".into(), row_count.to_string()),
            ],
            row_count,
            request_count: 1,
            recurrence: recurrence.borrow(py).clone(),
            seed_bindings,
            resource_bindings,
            output_bindings,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedRecurrenceStageArtifact {
    #[pyo3(get)]
    schema_version: u64,
    #[pyo3(get)]
    signature: String,
    #[pyo3(get)]
    shape_specialization: Vec<(String, String)>,
    #[pyo3(get)]
    row_count: u64,
    #[pyo3(get)]
    request_count: u64,
    #[pyo3(get)]
    plan: TypedRecurrencePlan,
    #[pyo3(get)]
    seed_bindings: Vec<TypedStageBinding>,
    #[pyo3(get)]
    resource_bindings: Vec<TypedStageResourceBinding>,
    #[pyo3(get)]
    output_bindings: Vec<TypedStageOutputBinding>,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedBoundRecurrenceStage {
    #[pyo3(get)]
    signature: String,
    #[pyo3(get)]
    plan: TypedRecurrencePlan,
    #[pyo3(get)]
    seed_bindings: Vec<TypedStageBinding>,
    #[pyo3(get)]
    resource_bindings: Vec<TypedStageResourceBinding>,
    #[pyo3(get)]
    output_bindings: Vec<TypedStageOutputBinding>,
}

/// Backend-observed row ownership after Rust has validated stage admission.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedObservedStageSpan {
    #[pyo3(get)]
    request_ordinal: i64,
    #[pyo3(get)]
    packed_row_start: i64,
    #[pyo3(get)]
    packed_row_end: i64,
    #[pyo3(get)]
    token_offset_start: i64,
    #[pyo3(get)]
    token_offset_end: i64,
    #[pyo3(get)]
    phase: SchedulerBatchPhase,
}

/// A signed stage plus the exact packed-row span admitted by Rust.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedPackedRecurrenceStageBinding {
    #[pyo3(get)]
    stage: TypedBoundRecurrenceStage,
    #[pyo3(get)]
    observed_span: TypedObservedStageSpan,
}

/// Whole packed-plan admission derived from signed row policy and Rust facts.
#[pyclass(frozen, eq, eq_int, hash)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TypedStagePlanAdmission {
    Execute,
    Noop,
}

/// One closed packed-plan result. `Noop` always carries an empty request list;
/// `Execute` carries every request-local signed stage selected by Rust.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedPackedRecurrenceStageBatch {
    #[pyo3(get)]
    admission: TypedStagePlanAdmission,
    #[pyo3(get)]
    total_rows: i64,
    #[pyo3(get)]
    requests: Vec<TypedPackedRecurrenceStageBinding>,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedRecurrenceStageArtifactBundleEntry {
    #[pyo3(get)]
    row_count: u64,
    #[pyo3(get)]
    artifact: TypedRecurrenceStageArtifact,
}

/// A signed, cold-specialized family of one stage artifact per row count.
/// Python may select an exact member, but cannot reinterpret or reassemble
/// the signed wire representation.
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TypedRecurrenceStageArtifactBundle {
    #[pyo3(get)]
    schema_version: u64,
    #[pyo3(get)]
    signature: String,
    #[pyo3(get)]
    artifacts: Vec<TypedRecurrenceStageArtifactBundleEntry>,
}

#[pymethods]
impl TypedRecurrenceStageArtifactBundle {
    #[staticmethod]
    fn compile(py: Python<'_>, definitions: Vec<Py<TypedRecurrenceStageSpec>>) -> PyResult<Self> {
        let artifacts = definitions
            .into_iter()
            .map(|definition| {
                let definition = definition.borrow(py).clone();
                let row_count = definition.row_count;
                stage_artifact_from_spec(definition).and_then(|artifact| {
                    let artifact_json = serde_json::to_string(&artifact).map_err(|error| {
                        validation_error(format!(
                            "failed to encode typed recurrence stage artifact: {error}"
                        ))
                    })?;
                    Ok(TypedRecurrenceStageArtifactBundleWireEntry {
                        row_count,
                        artifact_json,
                    })
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        bundle_from_wire(1, artifacts)
    }

    #[staticmethod]
    fn from_wire(bundle_json: &str) -> PyResult<Self> {
        let wire: TypedRecurrenceStageArtifactBundleWire =
            decode_json(bundle_json, "typed recurrence stage artifact bundle")?;
        validate_stage_artifact_bundle_wire(wire)
    }

    fn to_wire(&self) -> PyResult<String> {
        encode_stage_artifact_bundle(self)
    }

    #[getter]
    fn max_row_count(&self) -> PyResult<u64> {
        self.artifacts
            .last()
            .map(|entry| entry.row_count)
            .ok_or_else(|| validation_error("typed recurrence stage artifact bundle is empty"))
    }

    fn bind_packed_plan(
        &self,
        row_plan: PyRef<'_, PackedPrefillRowPlan>,
    ) -> PyResult<TypedPackedRecurrenceStageBatch> {
        bind_stage_bundle_for_packed_plan(self, &row_plan)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedRecurrenceStageArtifactBundleSpec {
    schema_version: u64,
    artifacts: Vec<TypedRecurrenceStageArtifactBundleWireEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedRecurrenceStageArtifactBundleWireEntry {
    row_count: u64,
    artifact_json: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedRecurrenceStageArtifactBundleWire {
    schema_version: u64,
    signature: String,
    artifacts: Vec<TypedRecurrenceStageArtifactBundleWireEntry>,
}

#[derive(Serialize)]
struct TypedRecurrenceStageArtifactBundleUnsigned<'a> {
    schema_version: u64,
    artifacts: &'a [TypedRecurrenceStageArtifactBundleWireEntry],
}

/// Closed runtime evidence for one request's contiguous packed stage rows.
///
/// The row plan owns these coordinates. A stage artifact never infers a
/// request association from Python objects or a backend-specific batch type.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedStageRequestSpan {
    request_ordinal: u64,
    packed_row_start: u64,
    packed_row_end: u64,
    token_offset_start: u64,
    token_offset_end: u64,
    phase: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedStageFacts {
    schema_version: u64,
    shape_specialization: Vec<(String, String)>,
    row_count: u64,
    request_count: u64,
    request_spans: Vec<TypedStageRequestSpan>,
}

fn stage_digest<T: Serialize>(value: &T) -> PyResult<String> {
    let wire = serde_json::to_string(value).map_err(|error| {
        validation_error(format!("failed to encode typed recurrence stage: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(wire.as_bytes())))
}

fn validate_shape_specialization(entries: &[(String, String)]) -> PyResult<()> {
    if entries.is_empty()
        || entries
            .iter()
            .any(|(key, value)| !is_nonempty(key) || !is_nonempty(value))
        || entries.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err(validation_error(
            "typed recurrence stage shape_specialization must be non-empty and sorted unique",
        ));
    }
    Ok(())
}

fn validate_stage_shape_specialization(
    entries: &[(String, String)],
    row_count: u64,
    request_count: u64,
) -> PyResult<()> {
    validate_shape_specialization(entries)?;
    let required = [
        ("stage/request_count", request_count.to_string()),
        ("stage/row_count", row_count.to_string()),
    ];
    for (key, expected) in required {
        let Some((_, actual)) = entries.iter().find(|(candidate, _)| candidate == key) else {
            return Err(validation_error(format!(
                "typed recurrence stage shape_specialization requires {key}"
            )));
        };
        if actual != &expected {
            return Err(validation_error(format!(
                "typed recurrence stage shape_specialization {key} must equal {expected}"
            )));
        }
    }
    Ok(())
}

fn coordinates_are_nonnegative(coordinates: &[i64]) -> bool {
    coordinates.iter().all(|coordinate| *coordinate >= 0)
}

fn validate_stage_binding_list(
    bindings: &[TypedStageBinding],
    expected: &BTreeSet<String>,
    label: &str,
) -> PyResult<()> {
    let actual: BTreeSet<String> = bindings
        .iter()
        .map(|binding| binding.binding_id.clone())
        .collect();
    if bindings.iter().any(|binding| {
        !is_nonempty(&binding.binding_id)
            || !is_nonempty(&binding.kind)
            || !coordinates_are_nonnegative(&binding.coordinates)
    }) || actual.len() != bindings.len()
        || actual != *expected
        || bindings
            .windows(2)
            .any(|pair| pair[0].binding_id >= pair[1].binding_id)
    {
        return Err(validation_error(format!(
            "typed recurrence stage {label} bindings must exactly and canonically cover signed operands"
        )));
    }
    Ok(())
}

fn validate_stage_resource_binding_list(
    bindings: &[TypedStageResourceBinding],
    expected: &BTreeSet<String>,
) -> PyResult<()> {
    let actual: BTreeSet<String> = bindings
        .iter()
        .map(|binding| binding.resource_id.clone())
        .collect();
    if bindings.iter().any(|binding| {
        !is_nonempty(&binding.resource_id)
            || !is_nonempty(&binding.kind)
            || !coordinates_are_nonnegative(&binding.coordinates)
    }) || actual.len() != bindings.len()
        || actual != *expected
        || bindings
            .windows(2)
            .any(|pair| pair[0].resource_id >= pair[1].resource_id)
    {
        return Err(validation_error(
            "typed recurrence stage resource bindings must exactly and canonically cover signed operands",
        ));
    }
    Ok(())
}

fn validate_stage_outputs(
    bindings: &[TypedStageOutputBinding],
    retained_values: &[String],
) -> PyResult<()> {
    let retained: BTreeSet<&str> = retained_values.iter().map(String::as_str).collect();
    let output_ids: BTreeSet<String> = bindings
        .iter()
        .map(|binding| binding.output_id.clone())
        .collect();
    let value_ids: BTreeSet<String> = bindings
        .iter()
        .map(|binding| binding.value_id.clone())
        .collect();
    if bindings.iter().any(|binding| {
        !is_nonempty(&binding.output_id)
            || !is_nonempty(&binding.value_id)
            || !is_nonempty(&binding.kind)
            || !coordinates_are_nonnegative(&binding.coordinates)
            || !retained.contains(binding.value_id.as_str())
    }) || output_ids.len() != bindings.len()
        || value_ids.len() != bindings.len()
        || value_ids != retained.into_iter().map(str::to_string).collect()
        || bindings
            .windows(2)
            .any(|pair| pair[0].output_id >= pair[1].output_id)
    {
        return Err(validation_error(
            "typed recurrence stage output bindings must be canonical retained values",
        ));
    }
    Ok(())
}

fn stage_artifact_from_spec(
    spec: TypedRecurrenceStageSpec,
) -> PyResult<TypedRecurrenceStageArtifact> {
    if spec.schema_version != 1 || spec.row_count == 0 || spec.request_count == 0 {
        return Err(validation_error(
            "typed recurrence stage schema_version, row_count, and request_count are invalid",
        ));
    }
    validate_stage_shape_specialization(
        &spec.shape_specialization,
        spec.row_count,
        spec.request_count,
    )?;
    if spec.recurrence.row_admission.mixed_batch == MixedBatchPolicy::Noop {
        return Err(validation_error(
            "typed recurrence stage artifacts do not support mixed_batch=noop",
        ));
    }
    let plan = compile_typed(spec.recurrence)?;
    let mut expected_seeds = BTreeSet::new();
    let mut expected_resources = BTreeSet::new();
    for operation in &plan.spec.operations {
        for operand in &operation.operands {
            if let Some(binding_id) = &operand.binding_id {
                expected_seeds.insert(binding_id.clone());
            }
            if let Some(resource_id) = &operand.resource_id {
                expected_resources.insert(resource_id.clone());
            }
        }
    }
    validate_stage_binding_list(&spec.seed_bindings, &expected_seeds, "seed")?;
    validate_stage_resource_binding_list(&spec.resource_bindings, &expected_resources)?;
    validate_stage_outputs(&spec.output_bindings, &plan.retained_values)?;
    let unsigned = TypedRecurrenceStageArtifact {
        schema_version: 1,
        signature: String::new(),
        shape_specialization: spec.shape_specialization,
        row_count: spec.row_count,
        request_count: spec.request_count,
        plan,
        seed_bindings: spec.seed_bindings,
        resource_bindings: spec.resource_bindings,
        output_bindings: spec.output_bindings,
    };
    let mut artifact = unsigned;
    artifact.signature = stage_digest(&artifact_without_signature(&artifact))?;
    Ok(artifact)
}

#[derive(Serialize)]
struct TypedRecurrenceStageUnsigned<'a> {
    schema_version: u64,
    shape_specialization: &'a [(String, String)],
    row_count: u64,
    request_count: u64,
    plan: &'a TypedRecurrencePlan,
    seed_bindings: &'a [TypedStageBinding],
    resource_bindings: &'a [TypedStageResourceBinding],
    output_bindings: &'a [TypedStageOutputBinding],
}

fn artifact_without_signature(
    artifact: &TypedRecurrenceStageArtifact,
) -> TypedRecurrenceStageUnsigned<'_> {
    TypedRecurrenceStageUnsigned {
        schema_version: artifact.schema_version,
        shape_specialization: &artifact.shape_specialization,
        row_count: artifact.row_count,
        request_count: artifact.request_count,
        plan: &artifact.plan,
        seed_bindings: &artifact.seed_bindings,
        resource_bindings: &artifact.resource_bindings,
        output_bindings: &artifact.output_bindings,
    }
}

fn validate_stage_artifact(artifact: &TypedRecurrenceStageArtifact) -> PyResult<()> {
    let expected = stage_artifact_from_spec(TypedRecurrenceStageSpec {
        schema_version: artifact.schema_version,
        shape_specialization: artifact.shape_specialization.clone(),
        row_count: artifact.row_count,
        request_count: artifact.request_count,
        recurrence: artifact.plan.spec.clone(),
        seed_bindings: artifact.seed_bindings.clone(),
        resource_bindings: artifact.resource_bindings.clone(),
        output_bindings: artifact.output_bindings.clone(),
    })?;
    if artifact != &expected {
        return Err(validation_error(
            "typed recurrence stage artifact plan, bindings, or signature differ from Rust-derived values",
        ));
    }
    Ok(())
}

fn validate_stage_facts(
    artifact: &TypedRecurrenceStageArtifact,
    facts: &TypedStageFacts,
) -> PyResult<()> {
    if facts.schema_version != 1
        || facts.shape_specialization != artifact.shape_specialization
        || facts.row_count != artifact.row_count
        || facts.request_count != artifact.request_count
    {
        return Err(validation_error(
            "typed recurrence stage facts shape_specialization does not match the signed artifact",
        ));
    }
    validate_stage_shape_specialization(
        &facts.shape_specialization,
        facts.row_count,
        facts.request_count,
    )?;
    if facts.request_spans.len() != artifact.request_count as usize {
        return Err(validation_error(
            "typed recurrence stage facts must contain one canonical request span per request",
        ));
    }
    let mut next_packed_row = 0_u64;
    for (expected_ordinal, span) in facts.request_spans.iter().enumerate() {
        if span.request_ordinal != expected_ordinal as u64
            || span.packed_row_start != next_packed_row
            || span.packed_row_end <= span.packed_row_start
            || span.token_offset_end < span.token_offset_start
            || span.token_offset_end - span.token_offset_start
                != span.packed_row_end - span.packed_row_start
            || !matches!(span.phase.as_str(), "prefill" | "decode")
        {
            return Err(validation_error(
                "typed recurrence stage facts request_spans must canonically partition packed rows with coherent phases",
            ));
        }
        next_packed_row = span.packed_row_end;
    }
    if next_packed_row != artifact.row_count {
        return Err(validation_error(
            "typed recurrence stage facts request_spans do not cover the signed row_count",
        ));
    }
    let allowed: BTreeSet<&str> = artifact
        .plan
        .spec
        .row_admission
        .allowed_phases
        .iter()
        .map(String::as_str)
        .collect();
    let all_allowed = facts
        .request_spans
        .iter()
        .all(|span| allowed.contains(span.phase.as_str()));
    let mixed = facts
        .request_spans
        .windows(2)
        .any(|pair| pair[0].phase != pair[1].phase);
    if !all_allowed || mixed {
        return Err(validation_error(
            "typed recurrence stage row admission rejects backend facts",
        ));
    }
    Ok(())
}

fn bundle_digest(
    schema_version: u64,
    artifacts: &[TypedRecurrenceStageArtifactBundleWireEntry],
) -> PyResult<String> {
    stage_digest(&TypedRecurrenceStageArtifactBundleUnsigned {
        schema_version,
        artifacts,
    })
}

fn bundle_from_wire(
    schema_version: u64,
    artifacts: Vec<TypedRecurrenceStageArtifactBundleWireEntry>,
) -> PyResult<TypedRecurrenceStageArtifactBundle> {
    if schema_version != 1 || artifacts.is_empty() {
        return Err(validation_error(
            "typed recurrence stage artifact bundle schema_version and artifacts are invalid",
        ));
    }
    let mut entries = Vec::with_capacity(artifacts.len());
    let mut row_admission: Option<RowAdmission> = None;
    for (index, wire_entry) in artifacts.iter().enumerate() {
        let expected_row_count = index as u64 + 1;
        if wire_entry.row_count != expected_row_count || wire_entry.artifact_json.is_empty() {
            return Err(validation_error(
                "typed recurrence stage artifact bundle rows must canonically cover 1 through N",
            ));
        }
        let artifact: TypedRecurrenceStageArtifact = decode_json(
            &wire_entry.artifact_json,
            "typed recurrence stage artifact bundle member",
        )?;
        validate_stage_artifact(&artifact)?;
        if row_admission
            .as_ref()
            .is_some_and(|expected| expected != &artifact.plan.spec.row_admission)
        {
            return Err(validation_error(
                "typed recurrence stage artifact bundle members must share one row admission policy",
            ));
        }
        row_admission.get_or_insert_with(|| artifact.plan.spec.row_admission.clone());
        let canonical_artifact_json = serde_json::to_string(&artifact).map_err(|error| {
            validation_error(format!(
                "failed to encode typed recurrence stage artifact bundle member: {error}"
            ))
        })?;
        if wire_entry.artifact_json != canonical_artifact_json
            || artifact.row_count != wire_entry.row_count
        {
            return Err(validation_error(
                "typed recurrence stage artifact bundle member does not match its canonical row specialization",
            ));
        }
        entries.push(TypedRecurrenceStageArtifactBundleEntry {
            row_count: wire_entry.row_count,
            artifact,
        });
    }
    let signature = bundle_digest(schema_version, &artifacts)?;
    Ok(TypedRecurrenceStageArtifactBundle {
        schema_version,
        signature,
        artifacts: entries,
    })
}

fn validate_stage_artifact_bundle_wire(
    wire: TypedRecurrenceStageArtifactBundleWire,
) -> PyResult<TypedRecurrenceStageArtifactBundle> {
    let bundle = bundle_from_wire(wire.schema_version, wire.artifacts)?;
    if bundle.signature != wire.signature {
        return Err(validation_error(
            "typed recurrence stage artifact bundle signature differs from Rust-derived values",
        ));
    }
    Ok(bundle)
}

#[pyfunction]
fn compile_typed_recurrence_stage_artifact(
    spec_json: &str,
) -> PyResult<TypedRecurrenceStageArtifact> {
    let spec: TypedRecurrenceStageSpec = decode_json(spec_json, "typed recurrence stage spec")?;
    stage_artifact_from_spec(spec)
}

#[pyfunction]
fn compile_typed_recurrence_stage_artifact_json(spec_json: &str) -> PyResult<String> {
    let artifact = compile_typed_recurrence_stage_artifact(spec_json)?;
    serde_json::to_string(&artifact).map_err(|error| {
        validation_error(format!(
            "failed to encode typed recurrence stage artifact: {error}"
        ))
    })
}

#[pyfunction]
fn validate_typed_recurrence_stage_artifact(artifact_json: &str) -> PyResult<()> {
    let artifact: TypedRecurrenceStageArtifact =
        decode_json(artifact_json, "typed recurrence stage artifact")?;
    validate_stage_artifact(&artifact)
}

#[pyfunction]
fn load_typed_recurrence_stage_artifact(
    artifact_json: &str,
) -> PyResult<TypedRecurrenceStageArtifact> {
    let artifact: TypedRecurrenceStageArtifact =
        decode_json(artifact_json, "typed recurrence stage artifact")?;
    validate_stage_artifact(&artifact)?;
    Ok(artifact)
}

#[pyfunction]
fn bind_typed_recurrence_stage(
    artifact: PyRef<'_, TypedRecurrenceStageArtifact>,
    facts_json: &str,
) -> PyResult<TypedBoundRecurrenceStage> {
    let facts: TypedStageFacts = decode_json(facts_json, "typed recurrence stage facts")?;
    validate_stage_facts(&artifact, &facts)?;
    Ok(TypedBoundRecurrenceStage {
        signature: artifact.signature.clone(),
        plan: artifact.plan.clone(),
        seed_bindings: artifact.seed_bindings.clone(),
        resource_bindings: artifact.resource_bindings.clone(),
        output_bindings: artifact.output_bindings.clone(),
    })
}

fn bind_single_request_observed_stage(
    artifact: &TypedRecurrenceStageArtifact,
    observed_span: TypedObservedStageSpan,
    phase_name: &str,
) -> PyResult<TypedPackedRecurrenceStageBinding> {
    if artifact.request_count != 1 {
        return Err(validation_error(
            "observed stage binding requires a one-request signed artifact",
        ));
    }
    let packed_row_count = observed_span
        .packed_row_end
        .checked_sub(observed_span.packed_row_start)
        .ok_or_else(|| validation_error("typed recurrence observed span has invalid row bounds"))?;
    let row_count = u64::try_from(packed_row_count)
        .map_err(|_| validation_error("typed recurrence observed span row count is unsupported"))?;
    let token_offset_start = u64::try_from(observed_span.token_offset_start)
        .map_err(|_| validation_error("typed recurrence observed token offset is unsupported"))?;
    let token_offset_end = u64::try_from(observed_span.token_offset_end)
        .map_err(|_| validation_error("typed recurrence observed token offset is unsupported"))?;
    let facts = TypedStageFacts {
        schema_version: 1,
        shape_specialization: artifact.shape_specialization.clone(),
        row_count,
        request_count: 1,
        request_spans: vec![TypedStageRequestSpan {
            request_ordinal: 0,
            packed_row_start: 0,
            packed_row_end: row_count,
            token_offset_start,
            token_offset_end,
            phase: phase_name.into(),
        }],
    };
    validate_stage_facts(artifact, &facts)?;
    Ok(TypedPackedRecurrenceStageBinding {
        stage: TypedBoundRecurrenceStage {
            signature: artifact.signature.clone(),
            plan: artifact.plan.clone(),
            seed_bindings: artifact.seed_bindings.clone(),
            resource_bindings: artifact.resource_bindings.clone(),
            output_bindings: artifact.output_bindings.clone(),
        },
        observed_span,
    })
}

#[pyfunction]
fn bind_typed_recurrence_stage_bundle_for_request_rows(
    bundle: PyRef<'_, TypedRecurrenceStageArtifactBundle>,
    row_count: u64,
) -> PyResult<TypedPackedRecurrenceStageBinding> {
    let artifact = bundle
        .artifacts
        .iter()
        .find(|entry| entry.row_count == row_count)
        .map(|entry| &entry.artifact)
        .ok_or_else(|| {
            TypedRecurrenceSpecializationRequired::new_err(format!(
                "typed recurrence stage artifact bundle has no exact row specialization for {row_count}"
            ))
        })?;
    let (phase, phase_name) = match artifact.plan.spec.row_admission.allowed_phases.as_slice() {
        [phase] if phase == "prefill" => (SchedulerBatchPhase::Prefill, "prefill"),
        [phase] if phase == "decode" => (SchedulerBatchPhase::Decode, "decode"),
        _ => {
            return Err(validation_error(
                "request-row stage binding requires one signed prefill or decode admission phase",
            ));
        }
    };
    let row_end = i64::try_from(row_count)
        .map_err(|_| validation_error("typed recurrence request row count is unsupported"))?;
    bind_single_request_observed_stage(
        artifact,
        TypedObservedStageSpan {
            request_ordinal: 0,
            packed_row_start: 0,
            packed_row_end: row_end,
            token_offset_start: 0,
            token_offset_end: row_end,
            phase,
        },
        phase_name,
    )
}

#[pyfunction]
fn admit_typed_recurrence_stage_bundle_for_packed_plan(
    bundle: PyRef<'_, TypedRecurrenceStageArtifactBundle>,
    row_plan: PyRef<'_, PackedPrefillRowPlan>,
) -> PyResult<TypedStagePlanAdmission> {
    admit_stage_bundle_for_packed_plan(&bundle, &row_plan)
}

fn packed_span_phase(span: &PrefillRowSpan) -> PyResult<(SchedulerBatchPhase, &'static str)> {
    let packed_row_count = span
        .packed_row_end
        .checked_sub(span.packed_row_start)
        .ok_or_else(|| validation_error("typed recurrence packed span has invalid row bounds"))?;
    if packed_row_count <= 0
        || span.prefill_row_count < 0
        || span.decode_row_count < 0
        || span.prefill_row_count + span.decode_row_count != packed_row_count
        || span.token_offset_end - span.token_offset_start != packed_row_count
    {
        return Err(validation_error(
            "typed recurrence packed span has inconsistent observed row facts",
        ));
    }
    match (span.prefill_row_count, span.decode_row_count) {
        (prefill, 0) if prefill > 0 => Ok((SchedulerBatchPhase::Prefill, "prefill")),
        (0, decode) if decode > 0 => Ok((SchedulerBatchPhase::Decode, "decode")),
        _ => Err(validation_error(
            "typed recurrence packed span contains mixed prefill/decode rows",
        )),
    }
}

fn admit_stage_bundle_for_packed_plan(
    bundle: &TypedRecurrenceStageArtifactBundle,
    row_plan: &PackedPrefillRowPlan,
) -> PyResult<TypedStagePlanAdmission> {
    let artifact = &bundle
        .artifacts
        .first()
        .ok_or_else(|| validation_error("typed recurrence stage artifact bundle is empty"))?
        .artifact;
    let observed: BTreeSet<&str> = row_plan.phases.iter().map(String::as_str).collect();
    if observed.is_empty() {
        return Ok(TypedStagePlanAdmission::Noop);
    }
    if observed.len() > 1 {
        return match artifact.plan.spec.row_admission.mixed_batch {
            MixedBatchPolicy::Reject => Err(validation_error(
                "typed recurrence stage row admission rejects mixed backend phases",
            )),
            MixedBatchPolicy::Noop => Ok(TypedStagePlanAdmission::Noop),
            MixedBatchPolicy::SelectAllowed => {
                for span in &row_plan.request_spans {
                    let (_, phase_name) = packed_span_phase(span)?;
                    if artifact
                        .plan
                        .spec
                        .row_admission
                        .allowed_phases
                        .iter()
                        .any(|allowed| allowed == phase_name)
                    {
                        return Ok(TypedStagePlanAdmission::Execute);
                    }
                }
                Ok(TypedStagePlanAdmission::Noop)
            }
        };
    }
    let phase = observed
        .first()
        .ok_or_else(|| validation_error("typed recurrence packed plan has no observed phase"))?;
    if artifact
        .plan
        .spec
        .row_admission
        .allowed_phases
        .iter()
        .any(|allowed| allowed == phase)
    {
        Ok(TypedStagePlanAdmission::Execute)
    } else {
        Ok(TypedStagePlanAdmission::Noop)
    }
}

#[pyfunction]
fn bind_typed_recurrence_stage_bundle_for_packed_span(
    bundle: PyRef<'_, TypedRecurrenceStageArtifactBundle>,
    row_plan: PyRef<'_, PackedPrefillRowPlan>,
    request_ordinal: i64,
) -> PyResult<TypedPackedRecurrenceStageBinding> {
    bind_stage_bundle_for_packed_span(&bundle, &row_plan, request_ordinal)
}

fn bind_stage_bundle_for_packed_span(
    bundle: &TypedRecurrenceStageArtifactBundle,
    row_plan: &PackedPrefillRowPlan,
    request_ordinal: i64,
) -> PyResult<TypedPackedRecurrenceStageBinding> {
    let span = row_plan
        .request_spans
        .iter()
        .find(|span| span.request_ordinal == request_ordinal)
        .ok_or_else(|| {
            validation_error(format!(
                "typed recurrence packed row plan has no request ordinal {request_ordinal}"
            ))
        })?;
    let (phase, phase_name) = packed_span_phase(span)?;
    let packed_row_count = span
        .packed_row_end
        .checked_sub(span.packed_row_start)
        .ok_or_else(|| validation_error("typed recurrence packed span has invalid row bounds"))?;
    let row_count = u64::try_from(packed_row_count)
        .map_err(|_| validation_error("typed recurrence packed span row count is unsupported"))?;
    let artifact = bundle
        .artifacts
        .iter()
        .find(|entry| entry.row_count == row_count)
        .map(|entry| &entry.artifact)
        .ok_or_else(|| {
            TypedRecurrenceSpecializationRequired::new_err(format!(
                "typed recurrence stage artifact bundle has no exact row specialization for {row_count}"
            ))
        })?;
    bind_single_request_observed_stage(
        artifact,
        TypedObservedStageSpan {
            request_ordinal: span.request_ordinal,
            packed_row_start: span.packed_row_start,
            packed_row_end: span.packed_row_end,
            token_offset_start: span.token_offset_start,
            token_offset_end: span.token_offset_end,
            phase,
        },
        phase_name,
    )
}

fn bind_stage_bundle_for_packed_plan(
    bundle: &TypedRecurrenceStageArtifactBundle,
    row_plan: &PackedPrefillRowPlan,
) -> PyResult<TypedPackedRecurrenceStageBatch> {
    let admission = admit_stage_bundle_for_packed_plan(bundle, row_plan)?;
    if admission == TypedStagePlanAdmission::Noop {
        return Ok(TypedPackedRecurrenceStageBatch {
            admission,
            total_rows: row_plan.total_rows,
            requests: Vec::new(),
        });
    }
    let allowed_phases = &bundle
        .artifacts
        .first()
        .ok_or_else(|| validation_error("typed recurrence stage artifact bundle is empty"))?
        .artifact
        .plan
        .spec
        .row_admission
        .allowed_phases;
    let mut requests = Vec::new();
    for span in &row_plan.request_spans {
        let (_, phase_name) = packed_span_phase(span)?;
        if allowed_phases.iter().any(|allowed| allowed == phase_name) {
            requests.push(bind_stage_bundle_for_packed_span(
                bundle,
                row_plan,
                span.request_ordinal,
            )?);
        }
    }
    let admission = if requests.is_empty() {
        TypedStagePlanAdmission::Noop
    } else {
        admission
    };
    Ok(TypedPackedRecurrenceStageBatch {
        admission,
        total_rows: row_plan.total_rows,
        requests,
    })
}

#[pyfunction]
fn compile_typed_recurrence_stage_artifact_bundle_json(spec_json: &str) -> PyResult<String> {
    let spec: TypedRecurrenceStageArtifactBundleSpec =
        decode_json(spec_json, "typed recurrence stage artifact bundle spec")?;
    let bundle = bundle_from_wire(spec.schema_version, spec.artifacts)?;
    encode_stage_artifact_bundle(&bundle)
}

fn encode_stage_artifact_bundle(bundle: &TypedRecurrenceStageArtifactBundle) -> PyResult<String> {
    let wire = TypedRecurrenceStageArtifactBundleWire {
        schema_version: bundle.schema_version,
        signature: bundle.signature.clone(),
        artifacts: bundle
            .artifacts
            .iter()
            .map(|entry| {
                serde_json::to_string(&entry.artifact)
                    .map(|artifact_json| TypedRecurrenceStageArtifactBundleWireEntry {
                        row_count: entry.row_count,
                        artifact_json,
                    })
                    .map_err(|error| {
                        validation_error(format!(
                            "failed to encode typed recurrence stage artifact bundle member: {error}"
                        ))
                    })
            })
            .collect::<PyResult<Vec<_>>>()?,
    };
    serde_json::to_string(&wire).map_err(|error| {
        validation_error(format!(
            "failed to encode typed recurrence stage artifact bundle: {error}"
        ))
    })
}

#[pyfunction]
fn validate_typed_recurrence_stage_artifact_bundle(bundle_json: &str) -> PyResult<()> {
    let wire: TypedRecurrenceStageArtifactBundleWire =
        decode_json(bundle_json, "typed recurrence stage artifact bundle")?;
    validate_stage_artifact_bundle_wire(wire).map(|_| ())
}

#[pyfunction]
fn load_typed_recurrence_stage_artifact_bundle(
    bundle_json: &str,
) -> PyResult<TypedRecurrenceStageArtifactBundle> {
    let wire: TypedRecurrenceStageArtifactBundleWire =
        decode_json(bundle_json, "typed recurrence stage artifact bundle")?;
    validate_stage_artifact_bundle_wire(wire)
}

#[pyfunction]
fn typed_recurrence_stage_artifact_for_rows(
    bundle: PyRef<'_, TypedRecurrenceStageArtifactBundle>,
    row_count: u64,
) -> PyResult<TypedRecurrenceStageArtifact> {
    bundle
        .artifacts
        .iter()
        .find(|entry| entry.row_count == row_count)
        .map(|entry| entry.artifact.clone())
        .ok_or_else(|| {
            validation_error(format!(
                "typed recurrence stage artifact bundle has no exact row specialization for {row_count}"
            ))
        })
}

fn decode_json<T: for<'de> Deserialize<'de>>(json: &str, name: &str) -> PyResult<T> {
    serde_json::from_str(json)
        .map_err(|error| validation_error(format!("invalid {name} JSON: {error}")))
}

#[pyfunction]
fn compile_typed_recurrence_plan(spec_json: &str) -> PyResult<TypedRecurrencePlan> {
    let spec: FiniteRecurrenceSpec = decode_json(spec_json, "finite recurrence spec")?;
    compile_typed(spec)
}

#[pyfunction]
fn compile_typed_recurrence_plan_json(spec_json: &str) -> PyResult<String> {
    let plan = compile_typed_recurrence_plan(spec_json)?;
    serde_json::to_string(&plan).map_err(|error| {
        validation_error(format!("failed to encode typed recurrence plan: {error}"))
    })
}

#[pyfunction]
fn validate_typed_recurrence_plan(plan_json: &str) -> PyResult<()> {
    let submitted: TypedRecurrencePlan = decode_json(plan_json, "typed recurrence plan")?;
    let expected = compile_typed(submitted.spec.clone())?;
    if submitted != expected {
        return Err(validation_error(
            "typed recurrence plan groups or release_after do not match Rust-derived plan",
        ));
    }
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "TypedRecurrenceSpecializationRequired",
        m.py().get_type::<TypedRecurrenceSpecializationRequired>(),
    )?;
    m.add_function(wrap_pyfunction!(compile_typed_recurrence_plan, m)?)?;
    m.add_function(wrap_pyfunction!(compile_typed_recurrence_plan_json, m)?)?;
    m.add_function(wrap_pyfunction!(validate_typed_recurrence_plan, m)?)?;
    m.add_function(wrap_pyfunction!(
        compile_typed_recurrence_stage_artifact,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compile_typed_recurrence_stage_artifact_json,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        validate_typed_recurrence_stage_artifact,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(load_typed_recurrence_stage_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(bind_typed_recurrence_stage, m)?)?;
    m.add_function(wrap_pyfunction!(
        admit_typed_recurrence_stage_bundle_for_packed_plan,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bind_typed_recurrence_stage_bundle_for_request_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bind_typed_recurrence_stage_bundle_for_packed_span,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compile_typed_recurrence_stage_artifact_bundle_json,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        validate_typed_recurrence_stage_artifact_bundle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        load_typed_recurrence_stage_artifact_bundle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        typed_recurrence_stage_artifact_for_rows,
        m
    )?)?;
    m.add_class::<OperandKind>()?;
    m.add_class::<EffectKind>()?;
    m.add_class::<TypedOperand>()?;
    m.add_class::<TypedEffect>()?;
    m.add_class::<TypedOperation>()?;
    m.add_class::<TypedDependency>()?;
    m.add_class::<TypedStageAdmission>()?;
    m.add_class::<TypedStageFusionPolicy>()?;
    m.add_class::<FiniteRecurrenceSpec>()?;
    m.add_class::<TypedExecutionGroup>()?;
    m.add_class::<ExecutionGroupingKind>()?;
    m.add_class::<TypedRecurrencePlan>()?;
    m.add_class::<TypedStageBinding>()?;
    m.add_class::<TypedStageResourceBinding>()?;
    m.add_class::<TypedStageOutputBinding>()?;
    m.add_class::<TypedRecurrenceStageSpec>()?;
    m.add_class::<TypedRecurrenceStageArtifact>()?;
    m.add_class::<TypedBoundRecurrenceStage>()?;
    m.add_class::<TypedObservedStageSpan>()?;
    m.add_class::<TypedPackedRecurrenceStageBinding>()?;
    m.add_class::<TypedPackedRecurrenceStageBatch>()?;
    m.add_class::<TypedStagePlanAdmission>()?;
    m.add_class::<TypedRecurrenceStageArtifactBundleEntry>()?;
    m.add_class::<TypedRecurrenceStageArtifactBundle>()?;
    Ok(())
}

//! Deterministic clean-room scheduler and mechanical physical projection.

use super::semantic_graph::{
    canonical_ascii_json, decode_graph, decode_timeline, digest, validate_graph, validate_timeline,
    ImplementationCandidate, SemanticGraph, Timeline, SCHEMA_VERSION,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ScheduledTask {
    task_id: u32,
    task_symbol: String,
    implementation_id: String,
    anchor_id: u32,
    anchor_symbol: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ScheduleDecision {
    graph_digest: String,
    timeline_digest: String,
    tasks: Vec<ScheduledTask>,
    fallback_count: u64,
    signature: String,
    schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PhysicalTask {
    task_id: u32,
    implementation_id: String,
    anchor_id: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PhysicalPlan {
    plan_epoch: u64,
    schedule_signature: String,
    tasks: Vec<PhysicalTask>,
    fallback_count: u64,
    schema_version: u64,
}

fn error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn decode_decision(payload: &str) -> Result<ScheduleDecision, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid ScheduleDecision: {source}"))
}

fn decode_physical(payload: &str) -> Result<PhysicalPlan, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid PhysicalPlan: {source}"))
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn decision_unsigned(decision: &ScheduleDecision) -> ScheduleDecision {
    let mut unsigned = decision.clone();
    unsigned.signature.clear();
    unsigned
}

fn decision_signature(decision: &ScheduleDecision) -> Result<String, String> {
    Ok(digest(&canonical_ascii_json(&decision_unsigned(decision))?))
}

fn task_index(graph: &SemanticGraph, symbol: &str) -> Result<usize, String> {
    graph
        .tasks
        .binary_search_by(|task| task.symbol.as_str().cmp(symbol))
        .map_err(|_| "semantic graph edge references an unknown task".to_owned())
}

fn topological_order(graph: &SemanticGraph) -> Result<(Vec<usize>, Vec<Vec<usize>>), String> {
    let mut successors = vec![Vec::new(); graph.tasks.len()];
    let mut predecessors = vec![Vec::new(); graph.tasks.len()];
    let mut indegree = vec![0_usize; graph.tasks.len()];
    for edge in &graph.edges {
        let source = task_index(graph, &edge.source)?;
        let target = task_index(graph, &edge.target)?;
        successors[source].push(target);
        predecessors[target].push(source);
        indegree[target] += 1;
    }
    let mut ready = BTreeSet::new();
    for (index, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            ready.insert(index);
        }
    }
    let mut order = Vec::with_capacity(graph.tasks.len());
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for successor in &successors[index] {
            indegree[*successor] -= 1;
            if indegree[*successor] == 0 {
                ready.insert(*successor);
            }
        }
    }
    if order.len() != graph.tasks.len() {
        return Err("semantic graph contains a dependency cycle".to_owned());
    }
    Ok((order, predecessors))
}

fn selected_implementation<'a>(
    candidates: &'a [ImplementationCandidate],
    timeline: &Timeline,
) -> Result<&'a ImplementationCandidate, String> {
    candidates
        .iter()
        .filter(|candidate| {
            candidate
                .required_capabilities
                .iter()
                .all(|capability| timeline.capabilities.binary_search(capability).is_ok())
        })
        .min_by(|left, right| left.implementation_id.cmp(&right.implementation_id))
        .ok_or_else(|| "no implementation candidate is covered by timeline capabilities".to_owned())
}

fn anchor_ids(timeline: &Timeline) -> Result<HashMap<&str, u32>, String> {
    timeline
        .anchors
        .iter()
        .enumerate()
        .map(|(index, anchor)| {
            let dense = u32::try_from(index)
                .map_err(|_| "canonical schedule anchor IDs exhausted u32".to_owned())?;
            Ok((anchor.symbol.as_str(), dense))
        })
        .collect()
}

fn compile(graph: &SemanticGraph, timeline: &Timeline) -> Result<ScheduleDecision, String> {
    validate_graph(graph)?;
    validate_timeline(timeline)?;
    let anchors = anchor_ids(timeline)?;
    let (order, predecessors) = topological_order(graph)?;
    let mut selected_anchors = vec![0_u32; graph.tasks.len()];
    let mut plans = Vec::with_capacity(graph.tasks.len());
    for task_index in order {
        let task = &graph.tasks[task_index];
        let after = *anchors
            .get(task.timing.after.as_str())
            .ok_or_else(|| "task timing.after references an unknown anchor".to_owned())?;
        let before = *anchors
            .get(task.timing.before.as_str())
            .ok_or_else(|| "task timing.before references an unknown anchor".to_owned())?;
        if after >= before {
            return Err("task timing window is empty".to_owned());
        }
        let dependency_anchor = predecessors[task_index]
            .iter()
            .map(|predecessor| selected_anchors[*predecessor])
            .max()
            .unwrap_or(0);
        let anchor_id = after.max(dependency_anchor);
        if anchor_id >= before {
            return Err("task dependencies do not fit its timing window".to_owned());
        }
        let implementation = selected_implementation(&task.implementations, timeline)?;
        let task_id = u32::try_from(task_index)
            .map_err(|_| "canonical schedule task IDs exhausted u32".to_owned())?;
        selected_anchors[task_index] = anchor_id;
        plans.push(ScheduledTask {
            task_id,
            task_symbol: task.symbol.clone(),
            implementation_id: implementation.implementation_id.clone(),
            anchor_id,
            anchor_symbol: timeline.anchors[anchor_id as usize].symbol.clone(),
        });
    }
    plans.sort_by_key(|plan| plan.anchor_id);
    let mut decision = ScheduleDecision {
        graph_digest: digest(&canonical_ascii_json(graph)?),
        timeline_digest: digest(&canonical_ascii_json(timeline)?),
        tasks: plans,
        fallback_count: 0,
        signature: String::new(),
        schema_version: SCHEMA_VERSION,
    };
    decision.signature = decision_signature(&decision)?;
    Ok(decision)
}

fn validate_decision_shape(decision: &ScheduleDecision) -> Result<(), String> {
    if decision.schema_version != SCHEMA_VERSION || decision.fallback_count != 0 {
        return Err("invalid canonical schedule decision envelope".to_owned());
    }
    require_digest(&decision.graph_digest, "decision.graph_digest")?;
    require_digest(&decision.timeline_digest, "decision.timeline_digest")?;
    require_digest(&decision.signature, "decision.signature")?;
    if decision.tasks.is_empty() {
        return Err("canonical schedule decision requires task plans".to_owned());
    }
    Ok(())
}

fn validate_decision(
    graph: &SemanticGraph,
    timeline: &Timeline,
    decision: &ScheduleDecision,
) -> Result<(), String> {
    validate_decision_shape(decision)?;
    let expected = compile(graph, timeline)?;
    if decision != &expected {
        return Err("canonical schedule decision differs from Rust recomputation".to_owned());
    }
    Ok(())
}

fn lower_physical(
    graph: &SemanticGraph,
    timeline: &Timeline,
    decision: &ScheduleDecision,
    plan_epoch: u64,
) -> Result<PhysicalPlan, String> {
    if plan_epoch == 0 {
        return Err("canonical physical plan epoch must be positive".to_owned());
    }
    validate_decision(graph, timeline, decision)?;
    Ok(PhysicalPlan {
        plan_epoch,
        schedule_signature: decision.signature.clone(),
        tasks: decision
            .tasks
            .iter()
            .map(|plan| PhysicalTask {
                task_id: plan.task_id,
                implementation_id: plan.implementation_id.clone(),
                anchor_id: plan.anchor_id,
            })
            .collect(),
        fallback_count: decision.fallback_count,
        schema_version: SCHEMA_VERSION,
    })
}

fn validate_physical_shape(physical: &PhysicalPlan) -> Result<(), String> {
    if physical.plan_epoch == 0 || physical.schema_version != SCHEMA_VERSION {
        return Err("invalid canonical physical plan envelope".to_owned());
    }
    require_digest(&physical.schedule_signature, "physical.schedule_signature")?;
    if physical.tasks.is_empty() || physical.fallback_count != 0 {
        return Err("invalid canonical physical plan tasks or fallback count".to_owned());
    }
    Ok(())
}

fn validate_physical(
    graph: &SemanticGraph,
    timeline: &Timeline,
    decision: &ScheduleDecision,
    physical: &PhysicalPlan,
) -> Result<(), String> {
    validate_physical_shape(physical)?;
    let expected = lower_physical(graph, timeline, decision, physical.plan_epoch)?;
    if physical != &expected {
        return Err("canonical physical plan differs from Rust projection".to_owned());
    }
    Ok(())
}

#[pyfunction]
fn compile_canonical_schedule(graph_json: &str, timeline_json: &str) -> PyResult<String> {
    let graph = decode_graph(graph_json).map_err(error)?;
    let timeline = decode_timeline(timeline_json).map_err(error)?;
    let decision = compile(&graph, &timeline).map_err(error)?;
    canonical_ascii_json(&decision).map_err(error)
}

#[pyfunction]
fn validate_canonical_schedule(
    graph_json: &str,
    timeline_json: &str,
    decision_json: &str,
) -> PyResult<()> {
    let graph = decode_graph(graph_json).map_err(error)?;
    let timeline = decode_timeline(timeline_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    validate_decision(&graph, &timeline, &decision).map_err(error)
}

#[pyfunction]
fn lower_canonical_physical_plan(
    graph_json: &str,
    timeline_json: &str,
    decision_json: &str,
    plan_epoch: u64,
) -> PyResult<String> {
    let graph = decode_graph(graph_json).map_err(error)?;
    let timeline = decode_timeline(timeline_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    let physical = lower_physical(&graph, &timeline, &decision, plan_epoch).map_err(error)?;
    canonical_ascii_json(&physical).map_err(error)
}

#[pyfunction]
fn validate_canonical_physical_plan(
    graph_json: &str,
    timeline_json: &str,
    decision_json: &str,
    physical_json: &str,
) -> PyResult<()> {
    let graph = decode_graph(graph_json).map_err(error)?;
    let timeline = decode_timeline(timeline_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    let physical = decode_physical(physical_json).map_err(error)?;
    validate_physical(&graph, &timeline, &decision, &physical).map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_canonical_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_canonical_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(lower_canonical_physical_plan, module)?)?;
    module.add_function(wrap_pyfunction!(validate_canonical_physical_plan, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compile, lower_physical, validate_decision, validate_physical, ScheduleDecision};
    use crate::canonical::semantic_graph::{decode_graph, decode_timeline};

    #[test]
    fn scheduler_selects_capabilities_dense_ids_timing_dependencies_and_rejects_tampering() {
        let graph = decode_graph(
            r#"{"workload_id":"esamp.request","tasks":[{"symbol":"predict","timing":{"after":"begin","before":"decode"},"inputs":[{"name":"hidden","source":{"kind":"hidden","layer":0,"site":"block_output","phase":"decode"}},{"name":"rows","source":{"kind":"packed_rows"}},{"name":"sampler_rows","source":{"kind":"packed_sampler_rows"}},{"name":"selection","source":{"kind":"state_selection","resource":"request_distiller","phase":"decode"}}],"outputs":[{"name":"guidance","kind":"sampler"},{"name":"loss_batch","kind":"temporary"}],"effects":[{"resource":"request_distiller","kind":"read"}],"implementations":[{"implementation_id":"esamp:cpu@1","execution_kind":"native_task","required_capabilities":["cpu"]},{"implementation_id":"esamp:gpu@1","execution_kind":"native_task","required_capabilities":["cuda"]}]},{"symbol":"train","timing":{"after":"decode","before":"end"},"inputs":[{"name":"loss_batch","source":{"kind":"result","task":"predict","output":"loss_batch"}},{"name":"selection","source":{"kind":"state_selection","resource":"request_distiller","phase":"decode"}}],"outputs":[],"effects":[{"resource":"request_distiller","kind":"transition"}],"implementations":[{"implementation_id":"esamp:train@1","execution_kind":"native_task","required_capabilities":[]}]}],"edges":[{"source":"predict","target":"train"}],"states":[{"resource":"request_distiller","scope":"request","key_fields":["request_id"]}],"schema_version":1}"#,
        )
        .unwrap();
        let timeline = decode_timeline(
            r#"{"anchors":[{"symbol":"begin","ordinal":0},{"symbol":"decode","ordinal":1},{"symbol":"end","ordinal":2}],"capabilities":["cpu"],"schema_version":1}"#,
        )
        .unwrap();
        let decision = compile(&graph, &timeline).unwrap();
        assert_eq!(decision.tasks[0].task_symbol, "predict");
        assert_eq!(decision.tasks[0].task_id, 0);
        assert_eq!(decision.tasks[0].anchor_id, 0);
        assert_eq!(decision.tasks[1].task_symbol, "train");
        assert_eq!(decision.tasks[1].task_id, 1);
        assert_eq!(decision.tasks[0].implementation_id, "esamp:cpu@1");
        assert_eq!(decision.tasks[1].anchor_id, 1);
        validate_decision(&graph, &timeline, &decision).unwrap();
        let physical = lower_physical(&graph, &timeline, &decision, 4).unwrap();
        validate_physical(&graph, &timeline, &decision, &physical).unwrap();
        let mut physical_tampered = physical.clone();
        physical_tampered.tasks[1].anchor_id = 0;
        assert!(validate_physical(&graph, &timeline, &decision, &physical_tampered).is_err());
        let mut tampered: ScheduleDecision = decision.clone();
        tampered.tasks[1].anchor_id = 0;
        assert!(validate_decision(&graph, &timeline, &tampered).is_err());
    }
}

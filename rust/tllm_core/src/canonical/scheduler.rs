//! Deterministic clean-room scheduler and mechanical physical projection.

use super::semantic_graph::{
    canonical_ascii_json, decode_graph, decode_timeline, digest, validate_graph, validate_timeline,
    ImplementationCandidate, InputSource, SemanticGraph, StateEffectKind, Timeline, SCHEMA_VERSION,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScheduledTask {
    task_id: u32,
    task_symbol: String,
    implementation_id: String,
    anchor_id: u32,
    anchor_symbol: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScheduleDecision {
    graph_digest: String,
    timeline_digest: String,
    tasks: Vec<ScheduledTask>,
    fallback_count: u64,
    signature: String,
    schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PhysicalTask {
    pub(crate) task_id: u32,
    pub(crate) implementation_id: String,
    pub(crate) anchor_id: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PhysicalPlan {
    pub(crate) plan_epoch: u64,
    pub(crate) schedule_signature: String,
    pub(crate) tasks: Vec<PhysicalTask>,
    pub(crate) fallback_count: u64,
    pub(crate) schema_version: u64,
}

fn error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

pub(crate) fn decode_decision(payload: &str) -> Result<ScheduleDecision, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid ScheduleDecision: {source}"))
}

pub(crate) fn decode_physical(payload: &str) -> Result<PhysicalPlan, String> {
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

struct GraphIndex {
    order: Vec<usize>,
    predecessors: Vec<BTreeSet<usize>>,
    windows: Vec<(u32, u32)>,
}

fn graph_index(graph: &SemanticGraph, timeline: &Timeline) -> Result<GraphIndex, String> {
    let anchors = anchor_ids(timeline)?;
    let windows: Vec<_> = graph
        .tasks
        .iter()
        .map(|task| {
            let after = *anchors
                .get(task.timing.after.as_str())
                .ok_or_else(|| "task timing.after references an unknown anchor".to_owned())?;
            let before = *anchors
                .get(task.timing.before.as_str())
                .ok_or_else(|| "task timing.before references an unknown anchor".to_owned())?;
            if after >= before {
                return Err("task timing window is empty".to_owned());
            }
            Ok((after, before))
        })
        .collect::<Result<_, String>>()?;
    let mut predecessors = vec![BTreeSet::new(); graph.tasks.len()];
    for edge in &graph.edges {
        predecessors[task_index(graph, &edge.target)?].insert(task_index(graph, &edge.source)?);
    }
    for (consumer, task) in graph.tasks.iter().enumerate() {
        for input in &task.inputs {
            if let InputSource::Result { task: producer, .. } = &input.source {
                predecessors[consumer].insert(task_index(graph, producer)?);
            }
        }
    }
    let mut conflicts = Vec::new();
    for (left, task) in graph.tasks.iter().enumerate() {
        for (right, other) in graph.tasks.iter().enumerate().skip(left + 1) {
            if task.effects.iter().any(|a| {
                other.effects.iter().any(|b| {
                    a.resource == b.resource
                        && (a.kind != StateEffectKind::Read || b.kind != StateEffectKind::Read)
                })
            }) {
                conflicts.push((left, right));
                // Disjoint semantic windows prove an order; overlapping writes need causality.
                if windows[left].1 <= windows[right].0 {
                    predecessors[right].insert(left);
                } else if windows[right].1 <= windows[left].0 {
                    predecessors[left].insert(right);
                }
            }
        }
    }
    let mut successors = vec![Vec::new(); graph.tasks.len()];
    for (target, sources) in predecessors.iter().enumerate() {
        for &source in sources {
            successors[source].push(target);
        }
    }
    let mut indegree: Vec<_> = predecessors.iter().map(BTreeSet::len).collect();
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
    let mut ancestors = vec![vec![false; graph.tasks.len()]; graph.tasks.len()];
    for &task in &order {
        for &parent in &predecessors[task] {
            ancestors[task][parent] = true;
            for ancestor in 0..graph.tasks.len() {
                ancestors[task][ancestor] |= ancestors[parent][ancestor];
            }
        }
    }
    for (left, right) in conflicts {
        if !ancestors[left][right] && !ancestors[right][left] {
            return Err(format!(
                "unordered state effects between {} and {}",
                graph.tasks[left].symbol, graph.tasks[right].symbol
            ));
        }
    }
    Ok(GraphIndex {
        order,
        predecessors,
        windows,
    })
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
    let index = graph_index(graph, timeline)?;
    let mut selected_anchors = vec![0_u32; graph.tasks.len()];
    let mut plans = Vec::with_capacity(graph.tasks.len());
    for task_index in index.order {
        let task = &graph.tasks[task_index];
        let (after, before) = index.windows[task_index];
        let dependency_anchor = index.predecessors[task_index]
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

pub(crate) fn validate_physical(
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
    use serde_json::json;

    fn graph_fixture() -> crate::canonical::semantic_graph::SemanticGraph {
        decode_graph(&json!({
            "workload_id": "oracle", "execution_parameters_digest": "a".repeat(64),
            "schema_version": 1, "edges": [], "states": [],
            "tasks": (0..3).map(|i| json!({
                "symbol": format!("t{i}"), "timing": {"after": "a0", "before": "a3"},
                "inputs": [], "outputs": [{"name": "value", "kind": "temporary", "schema": ""}],
                "effects": [], "implementations": [{"implementation_id": "cpu", "execution_kind": "native_task", "required_capabilities": []}]
            })).collect::<Vec<_>>()
        }).to_string()).unwrap()
    }

    fn check_exhaustive_graphs() {
        use crate::canonical::semantic_graph::{InputSource, SemanticEdge, TypedInput};
        let timeline = decode_timeline(&json!({
            "schema_version": 1, "capabilities": [],
            "anchors": (0..4).map(|i| json!({"symbol": format!("a{i}"), "ordinal": i})).collect::<Vec<_>>()
        }).to_string()).unwrap();
        let windows = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let arcs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for mask in 0..64 {
            let edges: Vec<_> = arcs
                .iter()
                .enumerate()
                .filter_map(|(bit, &arc)| (mask & (1 << bit) != 0).then_some(arc))
                .collect();
            let acyclic = permutations.iter().any(|order| {
                edges.iter().all(|(a, b)| {
                    order.iter().position(|i| i == a) < order.iter().position(|i| i == b)
                })
            });
            for window_code in 0..216 {
                let bounds = [
                    windows[window_code % 6],
                    windows[window_code / 6 % 6],
                    windows[window_code / 36],
                ];
                let mut graph = graph_fixture();
                for (task, (after, before)) in graph.tasks.iter_mut().zip(bounds) {
                    task.timing.after = format!("a{after}");
                    task.timing.before = format!("a{before}");
                }
                for &(a, b) in &edges {
                    if a < b {
                        graph.tasks[b].inputs.push(TypedInput {
                            name: format!("from{a}"),
                            source: InputSource::Result {
                                task: format!("t{a}"),
                                output: "value".into(),
                            },
                        });
                    } else {
                        graph.edges.push(SemanticEdge {
                            source: format!("t{a}"),
                            target: format!("t{b}"),
                        });
                    }
                }
                // Enumerate assignments independently; no planner index or placement helper.
                let expected = (0..27)
                    .map(|code| [code % 3, code / 3 % 3, code / 9])
                    .filter(|slots| {
                        acyclic
                            && bounds
                                .iter()
                                .enumerate()
                                .all(|(i, &(lo, hi))| lo <= slots[i] && slots[i] < hi)
                            && edges.iter().all(|&(a, b)| slots[a] <= slots[b])
                    })
                    .min();
                let actual = compile(&graph, &timeline);
                assert_eq!(
                    actual.is_ok(),
                    expected.is_some(),
                    "mask={mask}, windows={bounds:?}: {actual:?}"
                );
                if let Some(expected) = expected {
                    let decision = actual.unwrap();
                    let mut slots = [0; 3];
                    for task in &decision.tasks {
                        slots[task.task_id as usize] = task.anchor_id;
                    }
                    assert_eq!(slots, expected);
                    for &(a, b) in &edges {
                        assert!(
                            decision.tasks.iter().position(|t| t.task_id == a as u32)
                                < decision.tasks.iter().position(|t| t.task_id == b as u32)
                        );
                    }
                }
            }
        }
        let mut graph = graph_fixture();
        graph.tasks[0].inputs.push(TypedInput {
            name: "self".into(),
            source: InputSource::Result {
                task: "t0".into(),
                output: "value".into(),
            },
        });
        assert!(compile(&graph, &timeline).is_err());
        graph = graph_fixture();
        graph.states = serde_json::from_value(
            json!([{"resource": "state", "scope": "instance", "key_fields": []}]),
        )
        .unwrap();
        for left in ["read", "transition", "transition_maybe", "publish"] {
            for right in ["read", "transition", "transition_maybe", "publish"] {
                for (i, kind) in [left, right].iter().enumerate() {
                    graph.tasks[i].effects =
                        serde_json::from_value(json!([{"resource": "state", "kind": kind}]))
                            .unwrap();
                }
                assert_eq!(
                    compile(&graph, &timeline).is_ok(),
                    left == "read" && right == "read"
                );
            }
        }
        graph.tasks[0].timing.before = "a1".into();
        graph.tasks[1].timing.after = "a1".into();
        assert!(compile(&graph, &timeline).is_ok());
        graph.tasks[0].timing.before = "a3".into();
        graph.edges.push(SemanticEdge {
            source: "t0".into(),
            target: "t1".into(),
        });
        assert!(compile(&graph, &timeline).is_ok());
        graph.edges = vec![
            SemanticEdge {
                source: "t0".into(),
                target: "t2".into(),
            },
            SemanticEdge {
                source: "t2".into(),
                target: "t1".into(),
            },
        ];
        assert!(compile(&graph, &timeline).is_ok());
        graph.tasks[0].timing.after = "a2".into();
        graph.tasks[1].timing.after = "a0".into();
        graph.tasks[1].timing.before = "a1".into();
        assert!(compile(&graph, &timeline).is_err());
    }

    #[test]
    fn scheduler_matches_exhaustive_small_dag_oracle() {
        check_exhaustive_graphs();
        let graph = decode_graph(
            r#"{"workload_id":"esamp.request","execution_parameters_digest":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","tasks":[{"symbol":"predict","timing":{"after":"begin","before":"decode"},"inputs":[{"name":"hidden","source":{"kind":"hidden","layer":0,"site":"block_output","phase":"decode"}},{"name":"rows","source":{"kind":"packed_rows"}},{"name":"sampler_rows","source":{"kind":"packed_sampler_rows"}},{"name":"selection","source":{"kind":"state_selection","resource":"request_distiller","phase":"decode"}}],"outputs":[{"name":"guidance","kind":"sampler","schema":""},{"name":"loss_batch","kind":"temporary","schema":""}],"effects":[{"resource":"request_distiller","kind":"read"}],"implementations":[{"implementation_id":"esamp:cpu@1","execution_kind":"native_task","required_capabilities":["cpu"]},{"implementation_id":"esamp:gpu@1","execution_kind":"native_task","required_capabilities":["cuda"]}]},{"symbol":"train","timing":{"after":"decode","before":"end"},"inputs":[{"name":"loss_batch","source":{"kind":"result","task":"predict","output":"loss_batch"}},{"name":"selection","source":{"kind":"state_selection","resource":"request_distiller","phase":"decode"}}],"outputs":[],"effects":[{"resource":"request_distiller","kind":"transition"}],"implementations":[{"implementation_id":"esamp:train@1","execution_kind":"native_task","required_capabilities":[]}]}],"edges":[{"source":"predict","target":"train"}],"states":[{"resource":"request_distiller","scope":"request","key_fields":["request_id"]}],"schema_version":1}"#,
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

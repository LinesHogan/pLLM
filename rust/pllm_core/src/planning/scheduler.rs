//! Deterministic clean-room scheduler and mechanical physical projection.

use super::aer;
use super::semantic_graph::{
    canonical_ascii_json, decode_graph, decode_timeline, digest, validate_graph, validate_timeline,
    ImplementationCandidate, InputSource, SemanticGraph, Timeline, SCHEMA_VERSION,
};
use super::state_representation::{decode_physical_plan as decode_state, StatePhysicalPlan};
use super::version_hazards;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScheduleDecision {
    graph_digest: String,
    timeline_digest: String,
    tasks: Vec<PhysicalTask>,
    fallback_count: u64,
    signature: String,
    schema_version: u64,
    region_ends: Vec<u32>,
    materialized_values: Vec<u32>,
    materialization_bytes: u64,
    peak_memory_bytes: u64,
    fence_count: u64,
    state_plan_digest: Option<String>,
}

impl ScheduleDecision {
    pub(super) fn score(&self) -> [u64; 4] {
        [
            self.materialization_bytes,
            self.region_ends.len() as u64,
            self.peak_memory_bytes,
            self.fence_count,
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PhysicalTask {
    pub(crate) task_id: u32,
    pub(crate) implementation_id: String,
    pub(crate) anchor_id: u32,
    #[serde(default)]
    pub(crate) release_values: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PhysicalPlan {
    pub(crate) plan_epoch: u64,
    pub(crate) schedule_signature: String,
    pub(crate) tasks: Vec<PhysicalTask>,
    pub(crate) fallback_count: u64,
    pub(crate) schema_version: u64,
    pub(crate) region_ends: Vec<u32>,
    pub(crate) materialized_values: Vec<u32>,
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

pub(super) struct GraphIndex {
    pub(super) order: Vec<usize>,
    pub(super) predecessors: Vec<BTreeSet<usize>>,
    pub(super) windows: Vec<(u32, u32)>,
}

fn graph_index(
    graph: &SemanticGraph,
    timeline: &Timeline,
    state: Option<&StatePhysicalPlan>,
) -> Result<GraphIndex, String> {
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
    version_hazards::dependencies(graph, state, &mut predecessors)?;
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
    Ok(GraphIndex {
        order,
        predecessors,
        windows,
    })
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

pub(super) fn compile(
    graph: &SemanticGraph,
    timeline: &Timeline,
    state: Option<&StatePhysicalPlan>,
) -> Result<ScheduleDecision, String> {
    validate_graph(graph)?;
    validate_timeline(timeline)?;
    let index = graph_index(graph, timeline, state)?;
    fn choose<'a>(graph: &'a SemanticGraph, timeline: &Timeline, index: &GraphIndex,
                  state: Option<&StatePhysicalPlan>, selected: &mut Vec<&'a ImplementationCandidate>,
                  best: &mut Option<ScheduleDecision>) -> Result<(), String> {
        if selected.len() == graph.tasks.len() {
            if let Ok(solution) = aer::solve(graph, timeline, index, selected, state) {
                let decision = sign_solution(graph, timeline, state, selected, solution)?;
                if best.as_ref().is_none_or(|best| decision.score() < best.score()) {
                    *best = Some(decision);
                }
            }
        } else {
            for candidate in &graph.tasks[selected.len()].implementations {
                if candidate.required_capabilities.iter().all(|cap| timeline.capabilities.binary_search(cap).is_ok()) {
                    selected.push(candidate);
                    choose(graph, timeline, index, state, selected, best)?;
                    selected.pop();
                }
            }
        }
        Ok(())
    }
    let mut best = None;
    choose(graph, timeline, &index, state, &mut Vec::new(), &mut best)?;
    best.ok_or_else(|| "no implementation, AER partition and placement fits the signed resource facts".into())
}

fn sign_solution(
    graph: &SemanticGraph,
    timeline: &Timeline,
    state: Option<&StatePhysicalPlan>,
    implementations: &[&ImplementationCandidate],
    solution: aer::Solution,
) -> Result<ScheduleDecision, String> {
    let mut plans = Vec::with_capacity(graph.tasks.len());
    let mut region_ends = Vec::new();
    for (members, anchor_id) in &solution.regions {
        for &task_index in members {
            let implementation = implementations[task_index];
            let task_id = u32::try_from(task_index)
                .map_err(|_| "canonical schedule task IDs exhausted u32".to_owned())?;
            plans.push(PhysicalTask {
                task_id,
                implementation_id: implementation.implementation_id.clone(),
                anchor_id: *anchor_id,
                release_values: solution.release_values[task_index].clone(),
            });
        }
        region_ends.push(plans.len() as u32);
    }
    let mut decision = ScheduleDecision {
        graph_digest: digest(&canonical_ascii_json(graph)?),
        timeline_digest: digest(&canonical_ascii_json(timeline)?),
        tasks: plans,
        fallback_count: 0,
        signature: String::new(),
        schema_version: SCHEMA_VERSION,
        region_ends,
        materialized_values: solution.materialized_values,
        materialization_bytes: solution.score[0],
        peak_memory_bytes: solution.score[2],
        fence_count: solution.score[3],
        state_plan_digest: state
            .map(canonical_ascii_json)
            .transpose()?
            .map(|wire| digest(&wire)),
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
    state: Option<&StatePhysicalPlan>,
) -> Result<(), String> {
    validate_decision_shape(decision)?;
    validate_graph(graph)?;
    validate_timeline(timeline)?;
    if decision.signature != decision_signature(decision)? {
        return Err("canonical schedule signature differs from its contents".into());
    }
    let index = graph_index(graph, timeline, state)?;
    let implementations = graph.tasks.iter().enumerate().map(|(id, task)| {
        let selected = decision.tasks.iter().find(|selected| selected.task_id as usize == id)
            .ok_or("canonical schedule is missing a task")?;
        task.implementations.iter().find(|candidate| candidate.implementation_id == selected.implementation_id
            && candidate.required_capabilities.iter().all(|capability| timeline.capabilities.binary_search(capability).is_ok()))
            .ok_or_else(|| "selected implementation is not available in the original timeline".to_owned())
    }).collect::<Result<Vec<_>, String>>()?;
    let solution = aer::check_placement(graph, timeline, &index, &implementations, state,
        &decision.tasks, &decision.region_ends)?;
    let expected = sign_solution(graph, timeline, state, &implementations, solution)?;
    if decision != &expected {
        return Err("canonical schedule decision differs from its selected placement witness".to_owned());
    }
    Ok(())
}

fn lower_physical(
    graph: &SemanticGraph,
    timeline: &Timeline,
    decision: &ScheduleDecision,
    plan_epoch: u64,
    state: Option<&StatePhysicalPlan>,
) -> Result<PhysicalPlan, String> {
    validate_decision(graph, timeline, decision, state)?;
    project_physical(decision, plan_epoch)
}

pub(super) fn project_physical(
    decision: &ScheduleDecision,
    plan_epoch: u64,
) -> Result<PhysicalPlan, String> {
    if plan_epoch == 0 {
        return Err("canonical physical plan epoch must be positive".to_owned());
    }
    Ok(PhysicalPlan {
        plan_epoch,
        schedule_signature: decision.signature.clone(),
        tasks: decision.tasks.clone(),
        fallback_count: decision.fallback_count,
        schema_version: SCHEMA_VERSION,
        region_ends: decision.region_ends.clone(),
        materialized_values: decision.materialized_values.clone(),
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
    state: Option<&StatePhysicalPlan>,
) -> Result<(), String> {
    validate_physical_shape(physical)?;
    let expected = lower_physical(graph, timeline, decision, physical.plan_epoch, state)?;
    if physical != &expected {
        return Err("canonical physical plan differs from Rust projection".to_owned());
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (graph_json, timeline_json, state_json=None))]
fn compile_schedule(
    graph_json: &str,
    timeline_json: &str,
    state_json: Option<&str>,
) -> PyResult<String> {
    let graph = decode_graph(graph_json).map_err(error)?;
    let timeline = decode_timeline(timeline_json).map_err(error)?;
    let state = state_json.map(decode_state).transpose().map_err(error)?;
    let decision = compile(&graph, &timeline, state.as_ref()).map_err(error)?;
    canonical_ascii_json(&decision).map_err(error)
}

#[pyfunction]
#[pyo3(signature = (graph_json, timeline_json, decision_json, state_json=None))]
fn validate_schedule(
    graph_json: &str,
    timeline_json: &str,
    decision_json: &str,
    state_json: Option<&str>,
) -> PyResult<()> {
    let graph = decode_graph(graph_json).map_err(error)?;
    let timeline = decode_timeline(timeline_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    let state = state_json.map(decode_state).transpose().map_err(error)?;
    validate_decision(&graph, &timeline, &decision, state.as_ref()).map_err(error)
}

#[pyfunction]
#[pyo3(signature = (graph_json, timeline_json, decision_json, plan_epoch, state_json=None))]
fn lower_physical_plan(
    graph_json: &str,
    timeline_json: &str,
    decision_json: &str,
    plan_epoch: u64,
    state_json: Option<&str>,
) -> PyResult<String> {
    let graph = decode_graph(graph_json).map_err(error)?;
    let timeline = decode_timeline(timeline_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    let state = state_json.map(decode_state).transpose().map_err(error)?;
    let physical =
        lower_physical(&graph, &timeline, &decision, plan_epoch, state.as_ref()).map_err(error)?;
    canonical_ascii_json(&physical).map_err(error)
}

#[pyfunction]
#[pyo3(signature = (graph_json, timeline_json, decision_json, physical_json, state_json=None))]
fn validate_physical_plan(
    graph_json: &str,
    timeline_json: &str,
    decision_json: &str,
    physical_json: &str,
    state_json: Option<&str>,
) -> PyResult<()> {
    let graph = decode_graph(graph_json).map_err(error)?;
    let timeline = decode_timeline(timeline_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    let physical = decode_physical(physical_json).map_err(error)?;
    let state = state_json.map(decode_state).transpose().map_err(error)?;
    validate_physical(&graph, &timeline, &decision, &physical, state.as_ref()).map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(lower_physical_plan, module)?)?;
    module.add_function(wrap_pyfunction!(validate_physical_plan, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{aer, decision_signature, graph_index, sign_solution};
    use super::{compile, lower_physical, validate_decision, validate_physical, ScheduleDecision};
    use crate::planning::semantic_graph::{decode_graph, decode_timeline};
    use serde_json::json;

    #[test]
    fn versioned_state_hazards_and_timing_match_oracle() {
        let timeline = decode_timeline(&json!({
            "schema_version": 1, "capabilities": [],
            "anchors": (0..4).map(|i| json!({"symbol": format!("a{i}"), "ordinal": i})).collect::<Vec<_>>()
        }).to_string()).unwrap();
        let state = super::decode_state(&json!({
            "plan_epoch": 1, "schema_version": 1, "request_digest": "a".repeat(64),
            "representation_signature": "b".repeat(64), "reserved_bytes": 32,
            "representations": [{"resource": "state", "provider_id": "pages",
                "device_kind": "cpu", "device_ordinal": 0, "layout_id": 0,
                "layout_digest": "c".repeat(64), "layout_namespace": "d".repeat(64),
                "max_admitted_lineages": 1, "page_capacity": 3, "root_page_count": 1,
                "reserved_bytes": 32, "capture_stable": false, "capabilities": ["state/exact_versions@1"]}]
        }).to_string()).unwrap();
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut valid = graph_fixture();
        valid.states = serde_json::from_value(json!([
            {"resource": "state", "scope": "instance", "key_fields": []}
        ]))
        .unwrap();
        // Enumerate full executions, maintaining values/current state independently
        // of the planner's dependency construction and topological search.
        for split in [false, true] {
            for chain in [false, true] {
                for retained in [false, true] {
                    for read_version in [None, Some(0), Some(1)] {
                        for order in permutations {
                            let mut graph = valid.clone();
                            let sources = [None, chain.then_some(0), read_version];
                            for task in 0..3 {
                                let rank = order.iter().position(|&i| i == task).unwrap();
                                graph.tasks[task].timing.after = format!("a{rank}");
                                graph.tasks[task].timing.before = format!("a{}", rank + 1);
                                graph.tasks[task].effects = serde_json::from_value(json!([{
                                    "resource": "state", "kind": match task {
                                        0 if split => "transition_maybe", 0 => "update",
                                        1 if split => "publish", 1 => "update_maybe", _ => "read"
                                    },
                                    "version": sources[task].map(|i| format!("t{i}"))
                                }]))
                                .unwrap();
                            }
                            let mut current = None;
                            let mut produced = [false; 3];
                            let mut expected = !split || chain;
                            for task in order {
                                if sources[task].is_some_and(|i| !produced[i]) {
                                    expected = false;
                                }
                                if task == 2 {
                                    // The private value is available before its publication;
                                    // the publication symbol names that same value afterward.
                                    let value = sources[task].map(|i| if split { 0 } else { i });
                                    if !retained && !(split && value == Some(0)) && value != current
                                    {
                                        expected = false;
                                    }
                                } else if split && task == 0 {
                                    produced[task] = true;
                                } else {
                                    let input = if split { None } else { sources[task] };
                                    if current != input {
                                        expected = false;
                                    }
                                    current = Some(if split { 0 } else { task });
                                    produced[task] = true;
                                }
                            }
                            let result = compile(&graph, &timeline, retained.then_some(&state));
                            assert_eq!(result.is_ok(), expected,
                                "split={split} chain={chain} retained={retained} read={read_version:?} order={order:?}: {result:?}");
                            if expected && split && retained {
                                valid = graph;
                            }
                        }
                    }
                }
            }
        }
        let decision = compile(&valid, &timeline, Some(&state)).unwrap();
        let physical = lower_physical(&valid, &timeline, &decision, 1, Some(&state)).unwrap();
        let mut drift = state.clone();
        drift.representations[0].capabilities.clear();
        assert!(validate_physical(&valid, &timeline, &decision, &physical, Some(&drift)).is_err());
        assert!(validate_decision(&valid, &timeline, &decision, None).is_err());
        for mutation in 0..7 {
            let mut graph = valid.clone();
            match mutation {
                0 => graph.tasks[2].effects[0].version = Some("missing".into()),
                1 => graph.tasks[2].effects[0].version = Some("t2".into()),
                2 => graph.tasks[0].effects[0].version = Some("t2".into()),
                3 => graph.tasks[1].effects.clear(),
                4 => graph.tasks[2].effects = graph.tasks[1].effects.clone(),
                5 => {
                    graph.states.push(
                        serde_json::from_value(json!({
                            "resource": "unrelated", "scope": "instance", "key_fields": []
                        }))
                        .unwrap(),
                    );
                    graph.tasks[2].effects[0].resource = "unrelated".into();
                }
                _ => {
                    graph.edges = serde_json::from_value(json!([
                        {"source": "t1", "target": "t0"}
                    ]))
                    .unwrap()
                }
            }
            assert!(
                compile(&graph, &timeline, Some(&state)).is_err(),
                "mutation={mutation}"
            );
        }
    }

    #[test]
    fn physical_plan_is_mechanical_and_reconstruction_is_exact() {
        let mut graph = graph_fixture();
        graph.tasks[0].timing.before = "a1".into();
        for task in 1..3 {
            graph.tasks[task].timing.after = "a1".into();
            graph.tasks[task].inputs = serde_json::from_value(json!([{
                "name": "input", "source": {"kind": "result", "task": format!("t{}", task - 1), "output": "value"}
            }])).unwrap();
        }
        let timeline = decode_timeline(&json!({
            "schema_version": 1, "capabilities": [], "memory_budget_bytes": 100,
            "reserved_bytes": [0, 0, 24, 0],
            "anchors": (0..4).map(|anchor| json!({"symbol": format!("a{anchor}"), "ordinal": anchor})).collect::<Vec<_>>(),
            "value_resources": (0..3).map(|task| json!({"task": format!("t{task}"), "output": "value", "bytes": if task == 0 {8} else {32}})).collect::<Vec<_>>(),
            "task_resources": (0..3).map(|task| json!({"task": format!("t{task}"), "workspace_bytes": 16})).collect::<Vec<_>>()
        }).to_string()).unwrap();
        let decision = compile(&graph, &timeline, None).unwrap();
        let physical = lower_physical(&graph, &timeline, &decision, 7, None).unwrap();
        assert_eq!(physical.tasks, decision.tasks);
        assert_eq!(physical.region_ends, vec![1, 3]);
        assert_eq!(physical.materialized_values, vec![0]);
        assert_eq!(physical.tasks.iter().map(|task| task.release_values.clone()).collect::<Vec<_>>(),
            vec![vec![], vec![0], vec![1, 2]]);
        let reconstructed =
            super::decode_physical(&serde_json::to_string(&physical).unwrap()).unwrap();
        validate_physical(&graph, &timeline, &decision, &reconstructed, None).unwrap();
        // A worker checks this exact legal witness, not the optimizer's preferred
        // alternative. Compilation remains the only placement-selection path.
        let mut roomy = timeline.clone();
        roomy.memory_budget_bytes = Some(200);
        let mut later = decision.tasks.clone();
        later[1].anchor_id = 2;
        later[2].anchor_id = 2;
        let index = graph_index(&graph, &roomy, None).unwrap();
        let implementations: Vec<_> = graph.tasks.iter().map(|task| &task.implementations[0]).collect();
        let witness = aer::check_placement(&graph, &roomy, &index, &implementations, None,
            &later, &decision.region_ends).unwrap();
        let signed = sign_solution(&graph, &roomy, None, &implementations, witness).unwrap();
        assert_ne!(signed, compile(&graph, &roomy, None).unwrap());
        validate_decision(&graph, &roomy, &signed, None).unwrap();
        for mutation in 0..5 {
            let mut changed = signed.clone();
            match mutation {
                0 => changed.tasks[0].anchor_id = 3,
                1 => changed.tasks[1].task_id = changed.tasks[0].task_id,
                2 => changed.tasks.swap(1, 2),
                3 => changed.peak_memory_bytes = 1,
                _ => changed.tasks[1].release_values.clear(),
            }
            changed.signature = decision_signature(&changed).unwrap();
            assert!(validate_decision(&graph, &roomy, &changed, None).is_err());
        }
        for mutation in 0..7 {
            let mut changed = reconstructed.clone();
            match mutation {
                0 => changed.region_ends[0] = 2,
                1 => changed.materialized_values.clear(),
                2 => changed.tasks.swap(0, 1),
                3 => changed.tasks[1].anchor_id = 2,
                4 => changed.tasks[1].implementation_id = "local-choice".into(),
                5 => changed.schedule_signature = "0".repeat(64),
                _ => changed.plan_epoch = 0,
            }
            assert!(validate_physical(&graph, &timeline, &decision, &changed, None).is_err());
        }
        let mut drift = timeline.clone();
        drift.reserved_bytes[1] = 24;
        assert!(validate_physical(&graph, &drift, &decision, &physical, None).is_err());
        let mut incomplete = serde_json::to_value(&physical).unwrap();
        incomplete.as_object_mut().unwrap().remove("region_ends");
        assert!(super::decode_physical(&incomplete.to_string()).is_err());

        graph.tasks[1].timing.before = "a2".into();
        graph.tasks[2].timing.after = "a2".into();
        graph.tasks[2].outputs[0].kind = super::super::semantic_graph::OutputKind::Host;
        graph.tasks[2].outputs[0].schema = "test.result@1".into();
        let ordinary = compile(&graph, &roomy, None).unwrap();
        assert_eq!(ordinary.peak_memory_bytes, 104);
        for task in &mut graph.tasks[1..] {
            task.math = Some(serde_json::from_value(json!({"input_gradients": ["input"],
                "parameter_gradients": [], "detached_inputs": [], "forward_rows": "", "input_gradient_rows": "",
                "parameter_gradient_rows": "", "update_states": "", "reduction": "none"})).unwrap());
        }
        let backward = compile(&graph, &roomy, None).unwrap();
        // The first value is saved by t1 for the gradient consumed at t2.
        // 8 + 32 + 32 output bytes, 16 workspace, and 24 backend reservation.
        assert_eq!(backward.peak_memory_bytes, 112);
        assert_eq!(backward.tasks.iter().map(|task| task.release_values.clone()).collect::<Vec<_>>(),
            vec![vec![], vec![], vec![0, 1]]);
        roomy.memory_budget_bytes = Some(111);
        assert!(compile(&graph, &roomy, None).is_err());
    }

    fn graph_fixture() -> crate::planning::semantic_graph::SemanticGraph {
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
        use crate::planning::semantic_graph::{InputSource, SemanticEdge, TypedInput};
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
                let actual = compile(&graph, &timeline, None);
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
                    for task in 0..3 {
                        assert!(slots[task] >= expected[task]);
                        assert!(bounds[task].0 <= slots[task] && slots[task] < bounds[task].1);
                    }
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
        assert!(compile(&graph, &timeline, None).is_err());
    }

    #[test]
    fn scheduler_matches_exhaustive_small_dag_oracle() {
        check_exhaustive_graphs();
        let graph = decode_graph(
            r#"{"workload_id":"esamp.request","execution_parameters_digest":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","tasks":[{"symbol":"predict","timing":{"after":"begin","before":"decode"},"inputs":[{"name":"hidden","source":{"kind":"hidden","layer":0,"site":"block_output","phase":"decode"}},{"name":"rows","source":{"kind":"packed_rows"}},{"name":"sampler_rows","source":{"kind":"packed_sampler_rows"}},{"name":"selection","source":{"kind":"state_selection","resource":"request_distiller","phase":"decode"}}],"outputs":[{"name":"guidance","kind":"sampler","schema":""},{"name":"loss_batch","kind":"temporary","schema":""}],"effects":[{"resource":"request_distiller","kind":"read"}],"implementations":[{"implementation_id":"esamp:cpu@1","execution_kind":"native_task","required_capabilities":["cpu"]},{"implementation_id":"esamp:gpu@1","execution_kind":"native_task","required_capabilities":["cuda"]}]},{"symbol":"train","timing":{"after":"decode","before":"end"},"inputs":[{"name":"loss_batch","source":{"kind":"result","task":"predict","output":"loss_batch"}},{"name":"selection","source":{"kind":"state_selection","resource":"request_distiller","phase":"decode"}}],"outputs":[],"effects":[{"resource":"request_distiller","kind":"update"}],"implementations":[{"implementation_id":"esamp:train@1","execution_kind":"native_task","required_capabilities":[]}]}],"edges":[{"source":"predict","target":"train"}],"states":[{"resource":"request_distiller","scope":"request","key_fields":["request_id"]}],"schema_version":1}"#,
        )
        .unwrap();
        let timeline = decode_timeline(
            r#"{"anchors":[{"symbol":"begin","ordinal":0},{"symbol":"decode","ordinal":1},{"symbol":"end","ordinal":2}],"capabilities":["cpu"],"schema_version":1}"#,
        )
        .unwrap();
        let decision = compile(&graph, &timeline, None).unwrap();
        assert_eq!(
            graph.tasks[decision.tasks[0].task_id as usize].symbol,
            "predict"
        );
        assert_eq!(decision.tasks[0].task_id, 0);
        assert_eq!(decision.tasks[0].anchor_id, 0);
        assert_eq!(
            graph.tasks[decision.tasks[1].task_id as usize].symbol,
            "train"
        );
        assert_eq!(decision.tasks[1].task_id, 1);
        assert_eq!(decision.tasks[0].implementation_id, "esamp:cpu@1");
        assert_eq!(decision.tasks[1].anchor_id, 1);
        validate_decision(&graph, &timeline, &decision, None).unwrap();
        let physical = lower_physical(&graph, &timeline, &decision, 4, None).unwrap();
        validate_physical(&graph, &timeline, &decision, &physical, None).unwrap();
        let mut physical_tampered = physical.clone();
        physical_tampered.tasks[1].anchor_id = 0;
        assert!(validate_physical(&graph, &timeline, &decision, &physical_tampered, None).is_err());
        let mut tampered: ScheduleDecision = decision.clone();
        tampered.tasks[1].anchor_id = 0;
        assert!(validate_decision(&graph, &timeline, &tampered, None).is_err());
    }
}

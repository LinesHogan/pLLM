//! Select task points and lanes, and account the memory of the placed tasks.

use super::semantic_graph::{
    canonical_ascii_json, decode_graph, decode_timeline, validate_graph, ImplementationCandidate,
    InputSource, OutputKind, RequestKeyField, SemanticGraph, StateEffectKind, StateScope, Timeline,
    VisibilityKind,
};
use super::state_representation::{decode_decision as decode_state, StateRepresentationDecision};
use super::version_hazards;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScheduleDecision {
    pub(crate) tasks: Vec<PhysicalTask>,
    peak_memory_bytes: u64,
    prefix_replay_rejection: Option<String>,
}

impl ScheduleDecision {
    pub(super) fn score(&self) -> [u64; 2] {
        [self.peak_memory_bytes, self.tasks.len() as u64]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Lane {
    Main,
    Side,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReadOrigin {
    Entry,
    Boundary,
    Latest,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PhysicalTask {
    pub(crate) task_id: u32,
    pub(crate) implementation_id: String,
    pub(crate) anchor_id: u32,
    #[serde(default)]
    pub(crate) release_values: Vec<u32>,
    pub(crate) lane: Lane,
    pub(crate) read_origin: ReadOrigin,
    pub(crate) read_distance: u64,
}

fn error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn task_index(graph: &SemanticGraph, symbol: &str) -> Result<usize, String> {
    graph
        .tasks
        .binary_search_by(|task| task.symbol.as_str().cmp(symbol))
        .map_err(|_| "semantic graph edge references an unknown task".to_owned())
}

pub(super) struct GraphIndex {
    pub(super) order: Vec<usize>,
    pub(super) windows: Vec<(u32, u32)>,
}

fn graph_index(
    graph: &SemanticGraph,
    timeline: &Timeline,
    state: Option<&StateRepresentationDecision>,
) -> Result<GraphIndex, String> {
    let anchors: HashMap<_, _> = (timeline.anchors.iter())
        .enumerate()
        .map(|(index, anchor)| (anchor.symbol.as_str(), index as u32))
        .collect();
    // Declared phases name backend anchors in order, so every window is non-empty.
    let mut windows: Vec<_> = graph
        .tasks
        .iter()
        .map(|task| {
            (
                anchors[task.timing.after.as_str()],
                anchors[task.timing.before.as_str()],
            )
        })
        .collect();
    let mut predecessors = vec![BTreeSet::new(); graph.tasks.len()];
    for edge in graph.edges.iter().filter(|e| e.step_distance == 0) {
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
    let period =
        i64::try_from(timeline.anchors.len()).map_err(|_| "timeline rank overflows i64")?;
    let mut dependencies: Vec<_> = predecessors
        .iter()
        .enumerate()
        .flat_map(|(v, parents)| parents.iter().map(move |&u| (u, v, 0_i64)))
        .collect();
    for edge in &graph.edges {
        let shift = edge
            .step_distance
            .checked_mul(period)
            .and_then(|distance| i64::from(edge.separation).checked_sub(distance))
            .ok_or("step distance times timeline period overflows i64")?;
        dependencies.push((
            task_index(graph, &edge.source)?,
            task_index(graph, &edge.target)?,
            shift,
        ));
    }
    for (id, task) in graph.tasks.iter().enumerate() {
        for input in &task.inputs {
            let point = match &input.source {
                // A hidden input is available once the completed forward is published.
                InputSource::Hidden { .. } => anchors
                    .get("state.publish")
                    .or_else(|| anchors.get("model.execute")),
                InputSource::SamplerFeedback => anchors.get("sampler.complete"),
                _ => None,
            };
            if let Some(&point) = point {
                windows[id].0 = windows[id].0.max(point);
            }
        }
    }
    windows = propagate(&windows, &dependencies)?;
    Ok(GraphIndex { order, windows })
}

/// Reduced uniform dependence graph: u(t) -> v(t+k) has weight d-k*M.
fn propagate(
    bounds: &[(u32, u32)],
    edges: &[(usize, usize, i64)],
) -> Result<Vec<(u32, u32)>, String> {
    let mut lo: Vec<_> = bounds.iter().map(|b| i64::from(b.0)).collect();
    let mut hi: Vec<_> = bounds.iter().map(|b| i64::from(b.1) - 1).collect();
    for pass in 0..=bounds.len() {
        let mut changed = false;
        for &(u, v, weight) in edges {
            let lower = lo[u]
                .checked_add(weight)
                .ok_or("placement lower bound overflows i64")?;
            let upper = hi[v]
                .checked_sub(weight)
                .ok_or("placement upper bound overflows i64")?;
            if lower > lo[v] {
                lo[v] = lower;
                changed = true;
            }
            if upper < hi[u] {
                hi[u] = upper;
                changed = true;
            }
        }
        if lo.iter().zip(&hi).any(|(a, b)| a > b) {
            return Err(
                "a task reads a value (sampled tokens, hidden states or another task's \
                 result) that exists only after its phase ends; declare it in a later phase"
                    .into(),
            );
        }
        if !changed {
            return Ok(lo
                .iter()
                .zip(&hi)
                .map(|(&a, &b)| (a as u32, b as u32 + 1))
                .collect());
        }
        if pass == bounds.len() {
            return Err("placement constraints contain a contradictory cycle".into());
        }
    }
    unreachable!()
}

struct ValueUse {
    producer: usize,
    users: BTreeSet<usize>,
    bytes: u64,
    exported: bool,
}

struct Solution {
    tasks: Vec<PhysicalTask>,
    peak: u64,
}

fn place(
    graph: &SemanticGraph,
    timeline: &Timeline,
    index: &GraphIndex,
    state: Option<&StateRepresentationDecision>,
    implementations: &[&ImplementationCandidate],
) -> Result<Solution, String> {
    let mut order = index.order.clone();
    order.sort_by_key(|&id| index.windows[id].0);
    let mut tasks: Vec<_> = order
        .into_iter()
        .map(|id| PhysicalTask {
            task_id: id as u32,
            implementation_id: implementations[id].implementation_id.clone(),
            anchor_id: index.windows[id].0,
            release_values: Vec::new(),
            lane: Lane::Main,
            read_origin: ReadOrigin::Entry,
            read_distance: 0,
        })
        .collect();
    select_lanes(graph, timeline, &index.windows, &mut tasks);
    plan_memory(graph, timeline, index, state, &tasks).or_else(|_| {
        // Memory fallback preserves the selected logical read boundaries.
        for task in &mut tasks {
            task.lane = Lane::Main;
        }
        plan_memory(graph, timeline, index, state, &tasks)
    })
}

fn select_lanes(
    graph: &SemanticGraph,
    timeline: &Timeline,
    windows: &[(u32, u32)],
    tasks: &mut [PhysicalTask],
) {
    use super::semantic_graph::StateEffectKind;
    let Some(source) = &timeline.visibility else {
        return;
    };
    let Some(model) = timeline
        .anchors
        .iter()
        .position(|a| a.symbol == "model.execute")
    else {
        return;
    };
    if !timeline.capabilities.iter().any(|c| c == "compute/cuda@1")
        || source.lag == 0 && graph.model_reads.is_empty()
        || graph
            .tasks
            .iter()
            .any(|t| t.replay.is_some() || t.visibility.is_some())
    {
        return;
    }
    let Some(mut first) = tasks.iter().position(|t| {
        graph.tasks[t.task_id as usize]
            .effects
            .iter()
            .any(|e| e.kind != StateEffectKind::Read)
    }) else {
        return;
    };
    // A pure producer's temporary results have no consumer on the model/sampler lane.
    if graph.tasks.iter().all(|t| {
        t.outputs.iter().all(|o| o.kind != OutputKind::Sampler)
            && t.inputs
                .iter()
                .all(|i| !matches!(i.source, InputSource::HostService { .. }))
    }) {
        first = 0;
    }
    if tasks[first].anchor_id as usize <= model
        || tasks[first..].iter().any(|t| {
            let semantic = &graph.tasks[t.task_id as usize];
            semantic
                .inputs
                .iter()
                .any(|i| matches!(i.source, InputSource::HostService { .. }))
                || semantic
                    .outputs
                    .iter()
                    .any(|o| o.kind == OutputKind::Sampler)
                || t.anchor_id + 1 >= windows[t.task_id as usize].1
        })
        || tasks[..first].iter().any(|t| {
            graph.tasks[t.task_id as usize]
                .math
                .as_ref()
                .is_some_and(|m| !m.input_gradients.is_empty() || !m.parameter_gradients.is_empty())
        })
    {
        return;
    }
    // Explicit cross-step task events keep every task on the main lane on this backend.
    let cross_step = graph.edges.iter().any(|edge| edge.step_distance != 0);
    let captured = !cross_step
        && tasks[first..].iter().all(|task| {
            let semantic = &graph.tasks[task.task_id as usize];
            semantic.math.is_none()
                || semantic.implementations.iter().any(|candidate| {
                    candidate.implementation_id == task.implementation_id
                        && candidate
                            .required_capabilities
                            .iter()
                            .any(|cap| cap == "compute/cuda_graph@1")
                })
        });
    for (position, task) in tasks.iter_mut().enumerate() {
        if position >= first {
            task.lane = if captured { Lane::Side } else { Lane::Main };
            task.read_origin = ReadOrigin::Latest;
        } else if graph.model_reads.is_empty() {
            task.read_origin = ReadOrigin::Boundary;
            task.read_distance = source.lag;
        }
    }
}

/// Size every value, workspace and reservation of the placed tasks, release temporaries after their last
/// use, and check the peak against the memory budget.
fn plan_memory(
    graph: &SemanticGraph,
    timeline: &Timeline,
    index: &GraphIndex,
    state: Option<&StateRepresentationDecision>,
    tasks: &[PhysicalTask],
) -> Result<Solution, String> {
    let mut positions = vec![0; tasks.len()];
    for (position, task) in tasks.iter().enumerate() {
        positions[task.task_id as usize] = position;
    }
    let mut workspace = vec![0; graph.tasks.len()];
    for fact in &timeline.task_resources {
        let task = graph
            .tasks
            .binary_search_by(|task| task.symbol.cmp(&fact.task))
            .expect("workspace facts name declared tasks");
        workspace[task] = fact.workspace_bytes;
    }
    let mut values = Vec::new();
    for (producer, task) in graph.tasks.iter().enumerate() {
        for output in &task.outputs {
            let fact = timeline
                .value_resources
                .iter()
                .find(|fact| fact.task == task.symbol && fact.output == output.name);
            let mut users: BTreeSet<_> = graph.tasks.iter().enumerate().filter_map(|(user, task)|
                task.inputs.iter().any(|input| matches!(&input.source,
                    InputSource::Result { task, output: name } if task == &graph.tasks[producer].symbol && name == &output.name))
                    .then_some(user)).collect();
            // Autograd may save a consumer's inputs until its downstream gradient sink.
            for &user in &index.order {
                if graph.tasks[user].math.as_ref().is_some_and(|math| {
                    graph.tasks[user].inputs.iter().any(|input| {
                        math.input_gradients.contains(&input.name)
                            && matches!(&input.source, InputSource::Result { task, .. }
                            if users.iter().any(|&parent| &graph.tasks[parent].symbol == task))
                    })
                }) {
                    users.insert(user);
                }
            }
            values.push(ValueUse {
                producer,
                users,
                bytes: fact.map_or(0, |fact| fact.bytes),
                exported: output.kind != OutputKind::Temporary,
            });
        }
    }
    let mut planned = tasks.to_vec();
    for task in &mut planned {
        task.release_values.clear();
    }
    let mut live = vec![0_u128; tasks.len()];
    let mut exported = 0_u128;
    for (id, value) in values.iter().enumerate() {
        let first = positions[value.producer];
        let last = if value.exported {
            tasks.len() - 1
        } else {
            value
                .users
                .iter()
                .map(|&user| positions[user])
                .chain([first])
                .max()
                .unwrap()
        };
        for bytes in &mut live[first..=last] {
            *bytes += value.bytes as u128;
        }
        if value.exported {
            exported += value.bytes as u128;
        } else {
            planned[last].release_values.push(id as u32);
        }
    }
    let resident = timeline
        .task_resources
        .iter()
        .map(|f| f.resident_bytes as u128)
        .sum::<u128>()
        + state.map_or(0, |s| s.reserved_bytes as u128);
    let copies = if tasks.iter().any(|t| t.read_origin == ReadOrigin::Latest) {
        values
            .iter()
            .filter(|v| {
                tasks[positions[v.producer]].read_origin != ReadOrigin::Latest
                    && (v.exported
                        || v.users
                            .iter()
                            .any(|&u| tasks[positions[u]].read_origin == ReadOrigin::Latest))
            })
            .map(|v| v.bytes as u128)
            .sum::<u128>()
    } else {
        0
    };
    let side_workspace = copies
        + tasks
            .iter()
            .filter(|t| t.lane == Lane::Side)
            .map(|t| workspace[t.task_id as usize] as u128 + live[positions[t.task_id as usize]])
            .max()
            .unwrap_or(0);
    let mut peak = 0;
    for (anchor, _) in timeline.anchors.iter().enumerate() {
        let reserved = timeline.reserved_bytes.get(anchor).copied().unwrap_or(0) as u128 + resident;
        let at: Vec<_> = tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.anchor_id as usize == anchor)
            .collect();
        let bytes = if at.is_empty() {
            // Values spanning an anchor still occupy storage when no task runs there.
            values
                .iter()
                .filter(|v| {
                    tasks[positions[v.producer]].anchor_id as usize <= anchor
                        && (v.exported
                            || v.users
                                .iter()
                                .any(|&u| tasks[positions[u]].anchor_id as usize >= anchor))
                })
                .map(|v| v.bytes as u128)
                .sum()
        } else {
            at.iter()
                .map(|(pos, t)| live[*pos] + workspace[t.task_id as usize] as u128)
                .max()
                .unwrap()
        };
        peak = peak.max(reserved + bytes + side_workspace);
    }
    peak = peak.max(resident + exported);
    let peak = u64::try_from(peak).map_err(|_| "placement memory envelope overflows u64")?;
    if let Some(budget) = timeline.memory_budget_bytes.filter(|&budget| peak > budget) {
        return Err(format!(
            "workload tasks need {peak} bytes at their peak but only {budget} fit; lower \
             max_num_batched_tokens or raise ExecutionOptions.memory_budget_bytes"
        ));
    }
    Ok(Solution {
        tasks: planned,
        peak,
    })
}

pub(super) fn compile(
    graph: &SemanticGraph,
    timeline: &Timeline,
    state: Option<&StateRepresentationDecision>,
) -> Result<ScheduleDecision, String> {
    validate_graph(graph)?;
    debug_assert!(
        timeline
            .capabilities
            .windows(2)
            .all(|pair| pair[0] < pair[1]),
        "capabilities are sorted for binary search"
    );
    let index = graph_index(graph, timeline, state)?;
    fn choose<'a>(
        graph: &'a SemanticGraph,
        timeline: &Timeline,
        index: &GraphIndex,
        state: Option<&StateRepresentationDecision>,
        selected: &mut Vec<&'a ImplementationCandidate>,
        best: &mut Option<ScheduleDecision>,
        failure: &mut Option<String>,
    ) -> Result<(), String> {
        if selected.len() == graph.tasks.len() {
            match place(graph, timeline, index, state, selected) {
                Ok(solution) => {
                    let decision = build_decision(graph, selected, solution);
                    if best
                        .as_ref()
                        .is_none_or(|best| decision.score() < best.score())
                    {
                        *best = Some(decision);
                    }
                }
                Err(error) => *failure = Some(error),
            }
        } else {
            for candidate in &graph.tasks[selected.len()].implementations {
                if candidate
                    .required_capabilities
                    .iter()
                    .all(|cap| timeline.capabilities.binary_search(cap).is_ok())
                {
                    selected.push(candidate);
                    choose(graph, timeline, index, state, selected, best, failure)?;
                    selected.pop();
                }
            }
        }
        Ok(())
    }
    let mut best = None;
    let mut failure = None;
    choose(
        graph,
        timeline,
        &index,
        state,
        &mut Vec::new(),
        &mut best,
        &mut failure,
    )?;
    best.ok_or_else(|| {
        failure.unwrap_or_else(|| "no task implementation is supported by this backend".into())
    })
}

// Torch transitions are assumed deterministic. Initialization and execution support are separate
// from this structural eligibility fact; it does not enable preemption or prefix-cache hits.
fn prefix_replay_rejection(
    graph: &SemanticGraph,
    planned: &[PhysicalTask],
) -> Option<&'static str> {
    if graph.states.iter().any(|state| match state.scope {
        StateScope::Request => !state.key_fields.contains(&RequestKeyField::RequestId),
        StateScope::Instance => !state.immutable,
    }) {
        return Some(
            "use request-owned keys containing request_id and remove mutable instance state",
        );
    }
    if !graph.input_groups.is_empty()
        || graph
            .tasks
            .iter()
            .flat_map(|task| &task.inputs)
            .any(|input| matches!(&input.source, InputSource::HostService { service, .. } if service != "parameter.operands"))
    {
        return Some(
            "remove external feeds, imports and host inputs from the request state computation",
        );
    }
    for task in &graph.tasks {
        for input in &task.inputs {
            match input.source {
                InputSource::SamplerFeedback => {
                    return Some("remove sampler feedback dependencies")
                }
                InputSource::StepId | InputSource::StateSelection { metadata: true, .. } => {
                    return Some("remove reads of step ids and PTTS versions or slot indices");
                }
                _ => {}
            }
        }
        if task.replay.as_ref().is_some_and(|replay| replay.span) {
            return Some("use tokenwise layer routes instead of span sweeps");
        }
    }
    if graph
        .tasks
        .iter()
        .filter_map(|task| task.visibility.as_ref())
        .any(|visibility| {
            visibility.lag != 0 || visibility.policy.kind == VisibilityKind::ForwardPinned
        })
        || planned.iter().any(|task| task.read_distance != 0)
        || graph.edges.iter().any(|edge| edge.step_distance != 0)
    {
        return Some("use Exact visibility with zero publish lag and no cross-step dependencies");
    }
    if graph.tasks.iter().any(|task| {
        task.visibility.is_none()
            && task.effects.iter().any(|effect| {
                effect.kind != StateEffectKind::Publish
                    && graph
                        .states
                        .iter()
                        .any(|state| state.resource == effect.resource && !state.immutable)
            })
    }) {
        return Some("use Exact visibility for request state reads and transitions instead of forward_pinned execution");
    }
    None
}

fn build_decision(
    graph: &SemanticGraph,
    implementations: &[&ImplementationCandidate],
    solution: Solution,
) -> ScheduleDecision {
    let plans: Vec<_> = solution
        .tasks
        .into_iter()
        .map(|mut task| {
            task.implementation_id = implementations[task.task_id as usize]
                .implementation_id
                .clone();
            task
        })
        .collect();
    let rejection = prefix_replay_rejection(graph, &plans);
    ScheduleDecision {
        tasks: plans,
        peak_memory_bytes: solution.peak,
        prefix_replay_rejection: rejection.map(str::to_owned),
    }
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_schedule, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::compile;
    use crate::planning::semantic_graph::{decode_graph, decode_timeline};
    use serde_json::json;

    #[test]
    fn inline_memory_fallback_preserves_boundary_reads() {
        let mut graph = graph_fixture();
        graph.states = serde_json::from_value(
            json!([{"resource":"s", "scope":"request", "key_fields":["request_id"]}]),
        )
        .unwrap();
        graph.tasks[0].outputs[0].kind = super::OutputKind::Sampler;
        for (i, task) in graph.tasks.iter_mut().enumerate() {
            task.timing.after = if i == 0 {
                "model.execute"
            } else {
                "token.commit"
            }
            .into();
            task.timing.before = if i == 0 {
                "sampler.pre_draw"
            } else {
                "next.prepare"
            }
            .into();
            if i < 2 {
                task.effects = serde_json::from_value(json!([{
                    "resource":"s", "kind":if i==0 {"read"} else {"update"}
                }]))
                .unwrap();
            }
        }
        let mut timeline = decode_timeline(&json!({
            "capabilities":["compute/cuda@1"],
            "anchors":vec!["request.prepare","model.execute","state.publish","sampler.pre_draw","sampler.complete","token.commit","request.finish","next.prepare"]
                .into_iter().enumerate().map(|(i,s)| json!({"symbol":s,"ordinal":i})).collect::<Vec<_>>(),
            "visibility":{"policy":{"kind":"step_lag"},"scope":"post_sampler","equations_digest":"retained-step-inputs@1","resources":[],"chunk_size":0,"tail":"transition","lag":2},
            "task_resources":(0..3).map(|i|json!({"task":format!("t{i}"),"workspace_bytes":16})).collect::<Vec<_>>(),
            "value_resources":(0..3).map(|i|json!({"task":format!("t{i}"),"output":"value","bytes":8})).collect::<Vec<_>>()
        }).to_string()).unwrap();
        let side = compile(&graph, &timeline, None).unwrap();
        assert_eq!(side.tasks[1].lane, super::Lane::Side);
        timeline.memory_budget_bytes = Some(side.peak_memory_bytes - 1);
        let inline = compile(&graph, &timeline, None).unwrap();
        assert!(inline.tasks.iter().all(|t| t.lane == super::Lane::Main));
        assert_eq!(inline.tasks[0].read_distance, 2);
        assert_eq!(inline.tasks[0].read_origin, super::ReadOrigin::Boundary);
        assert_eq!(inline.tasks[1].read_origin, super::ReadOrigin::Latest);
        timeline.memory_budget_bytes = Some(inline.peak_memory_bytes - 1);
        assert!(compile(&graph, &timeline, None).is_err());
        timeline.memory_budget_bytes = None;
        graph.tasks[1].math = Some(serde_json::from_value(json!({"input_gradients":[],"parameter_gradients":[],"detached_inputs":[],"forward_rows":"","input_gradient_rows":"","parameter_gradient_rows":"","update_states":"","reduction":"none"})).unwrap());
        let main = compile(&graph, &timeline, None).unwrap();
        assert!(main.tasks.iter().all(|t| t.lane == super::Lane::Main));
        assert_eq!(main.tasks[0].read_distance, 2);
        graph.tasks[1].implementations[0]
            .required_capabilities
            .push("compute/cuda_graph@1".into());
        timeline.capabilities.push("compute/cuda_graph@1".into());
        assert_eq!(
            compile(&graph, &timeline, None).unwrap().tasks[1].lane,
            super::Lane::Side
        );
    }

    #[test]
    fn uniform_windows_match_independent_enumeration() {
        let windows: Vec<_> = (0..4)
            .flat_map(|a| (a + 1..=4).map(move |b| (a, b)))
            .collect();
        for &a in &windows {
            for &b in &windows {
                for forward in [None, Some(-1), Some(0), Some(1), Some(2)] {
                    for backward in [None, Some(-1), Some(0), Some(1), Some(2)] {
                        let edges: Vec<_> = [(0, 1, forward), (1, 0, backward)]
                            .into_iter()
                            .filter_map(|(u, v, k)| k.map(|k| (u, v, -4 * k)))
                            .collect();
                        let legal: Vec<_> = (0..4)
                            .flat_map(|x| (0..4).map(move |y| [x, y]))
                            .filter(|x| {
                                a.0 <= x[0]
                                    && x[0] < a.1
                                    && b.0 <= x[1]
                                    && x[1] < b.1
                                    && edges
                                        .iter()
                                        .all(|&(u, v, w)| i64::from(x[v]) >= i64::from(x[u]) + w)
                            })
                            .collect();
                        let result = super::propagate(&[a, b], &edges);
                        assert_eq!(result.is_ok(), !legal.is_empty());
                        if !legal.is_empty() {
                            assert_eq!(
                                result.unwrap(),
                                (0..2)
                                    .map(|i| (
                                        legal.iter().map(|x| x[i]).min().unwrap(),
                                        legal.iter().map(|x| x[i]).max().unwrap() + 1
                                    ))
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
        assert!(super::propagate(&[(0, 4), (0, 4)], &[(0, 1, i64::MAX), (1, 0, 1)]).is_err());
    }

    #[test]
    fn serial_memory_matches_allocation_intervals() {
        for mask in 0..8 {
            let mut graph = graph_fixture();
            let sizes = [3_u64, 5, 7];
            let scratch = [2_u64, 4, 1];
            let arcs: Vec<_> = [(0, 1), (0, 2), (1, 2)]
                .into_iter()
                .enumerate()
                .filter_map(|(i, e)| (mask & (1 << i) != 0).then_some(e))
                .collect();
            for &(u, v) in &arcs {
                graph.tasks[v].inputs.push(serde_json::from_value(json!({
                    "name": format!("from{u}"), "source": {"kind":"result", "task":format!("t{u}"), "output":"value"}
                })).unwrap());
            }
            let mut timeline = decode_timeline(&json!({
                "capabilities":[], "reserved_bytes":[9,0,0,0],
                "anchors":(0..4).map(|i| json!({"symbol":format!("a{i}"),"ordinal":i})).collect::<Vec<_>>(),
                "value_resources":(0..3).map(|i| json!({"task":format!("t{i}"),"output":"value","bytes":sizes[i]})).collect::<Vec<_>>(),
                "task_resources":(0..3).map(|i| json!({"task":format!("t{i}"),"workspace_bytes":scratch[i],"resident_bytes":1})).collect::<Vec<_>>()
            }).to_string()).unwrap();
            let expected = (0..3)
                .map(|at| {
                    12 + scratch[at]
                        + (0..=at)
                            .filter(|&u| {
                                arcs.iter()
                                    .filter(|&&(from, _)| from == u)
                                    .map(|&(_, to)| to)
                                    .chain([u])
                                    .max()
                                    .unwrap()
                                    >= at
                            })
                            .map(|u| sizes[u])
                            .sum::<u64>()
                })
                .max()
                .unwrap();
            timeline.memory_budget_bytes = Some(expected);
            assert_eq!(
                compile(&graph, &timeline, None).unwrap().peak_memory_bytes,
                expected
            );
            timeline.memory_budget_bytes = Some(expected - 1);
            assert!(compile(&graph, &timeline, None).is_err());
        }
    }

    #[test]
    fn versioned_state_hazards_and_timing_match_oracle() {
        let timeline = decode_timeline(&json!({
            "capabilities": [],
            "anchors": (0..4).map(|i| json!({"symbol": format!("a{i}"), "ordinal": i})).collect::<Vec<_>>()
        }).to_string()).unwrap();
        let state = super::decode_state(&json!({
            "reserved_bytes": 32,
            "plans": [{"resource": "state", "provider_id": "pages",
                "device_kind": "cpu", "device_ordinal": 0, "layout_id": 0,
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
    fn placement_witness_and_memory_accounting_are_exact() {
        let mut graph = graph_fixture();
        graph.tasks[0].timing.before = "a1".into();
        for task in 1..3 {
            graph.tasks[task].timing.after = "a1".into();
            graph.tasks[task].inputs = serde_json::from_value(json!([{
                "name": "input", "source": {"kind": "result", "task": format!("t{}", task - 1), "output": "value"}
            }])).unwrap();
        }
        let timeline = decode_timeline(&json!({
            "capabilities": [], "memory_budget_bytes": 100,
            "reserved_bytes": [0, 0, 24, 0],
            "anchors": (0..4).map(|anchor| json!({"symbol": format!("a{anchor}"), "ordinal": anchor})).collect::<Vec<_>>(),
            "value_resources": (0..3).map(|task| json!({"task": format!("t{task}"), "output": "value", "bytes": if task == 0 {8} else {32}})).collect::<Vec<_>>(),
            "task_resources": (0..3).map(|task| json!({"task": format!("t{task}"), "workspace_bytes": 16})).collect::<Vec<_>>()
        }).to_string()).unwrap();
        let decision = compile(&graph, &timeline, None).unwrap();
        assert_eq!(
            decision
                .tasks
                .iter()
                .map(|task| task.release_values.clone())
                .collect::<Vec<_>>(),
            vec![vec![], vec![0], vec![1, 2]]
        );
        let mut roomy = timeline.clone();
        roomy.memory_budget_bytes = Some(200);
        graph.tasks[1].timing.before = "a2".into();
        graph.tasks[2].timing.after = "a2".into();
        graph.tasks[2].outputs[0].kind = super::super::semantic_graph::OutputKind::Host;
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
        assert_eq!(
            backward
                .tasks
                .iter()
                .map(|task| task.release_values.clone())
                .collect::<Vec<_>>(),
            vec![vec![], vec![], vec![0, 1]]
        );
        roomy.memory_budget_bytes = Some(111);
        assert!(compile(&graph, &roomy, None).is_err());
    }

    fn graph_fixture() -> crate::planning::semantic_graph::SemanticGraph {
        decode_graph(&json!({
            "workload_id": "oracle", "execution_parameters_digest": "a".repeat(64),
            "input_groups": [], "edges": [], "states": [],
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
            "capabilities": [],
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
                            step_distance: 0,
                            separation: 0,
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
            r#"{"workload_id":"esamp.request","execution_parameters_digest":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","tasks":[{"symbol":"predict","timing":{"after":"begin","before":"decode"},"inputs":[{"name":"hidden","source":{"kind":"hidden","layer":0,"site":"block_output","phase":"decode"}},{"name":"rows","source":{"kind":"packed_rows"}},{"name":"sampler_rows","source":{"kind":"packed_sampler_rows"}},{"name":"selection","source":{"kind":"state_selection","resource":"request_distiller","phase":"decode"}}],"outputs":[{"name":"guidance","kind":"sampler","schema":""},{"name":"loss_batch","kind":"temporary","schema":""}],"effects":[{"resource":"request_distiller","kind":"read"}],"implementations":[{"implementation_id":"esamp:cpu@1","execution_kind":"native_task","required_capabilities":["cpu"]},{"implementation_id":"esamp:gpu@1","execution_kind":"native_task","required_capabilities":["cuda"]}]},{"symbol":"train","timing":{"after":"decode","before":"end"},"inputs":[{"name":"loss_batch","source":{"kind":"result","task":"predict","output":"loss_batch"}},{"name":"selection","source":{"kind":"state_selection","resource":"request_distiller","phase":"decode"}}],"outputs":[],"effects":[{"resource":"request_distiller","kind":"update"}],"implementations":[{"implementation_id":"esamp:train@1","execution_kind":"native_task","required_capabilities":[]}]}],"input_groups": [], "edges":[{"source":"predict","target":"train"}],"states":[{"resource":"request_distiller","scope":"request","key_fields":["request_id"]}]}"#,
        )
        .unwrap();
        let timeline = decode_timeline(
            r#"{"anchors":[{"symbol":"begin","ordinal":0},{"symbol":"decode","ordinal":1},{"symbol":"end","ordinal":2}],"capabilities":["cpu"]}"#,
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
    }
}

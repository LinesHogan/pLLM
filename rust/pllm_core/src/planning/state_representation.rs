//! Clean-room selection and validation for request-state representations.
//!
//! This module owns a closed, pure-data FFI contract.  It deliberately knows
//! no workload names, Tensor objects, Python callbacks, or runtime handles.

use super::scheduler::{self, ScheduleDecision};
use super::semantic_graph::{validate_graph, SemanticGraph, Timeline};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateDeviceKind {
    Cpu,
    Cuda,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateProviderPlan {
    pub(crate) resource: String,
    provider_id: String,
    pub(crate) device_kind: StateDeviceKind,
    pub(crate) layout_id: u64,
    pub(crate) max_admitted_lineages: u64,
    pub(crate) page_capacity: u64,
    root_page_count: u64,
    pub(crate) reserved_bytes: u64,
    capture_stable: bool,
    pub(crate) capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StateLayoutRequirement {
    resource: String,
    layout_id: u64,
    max_admitted_lineages: u64,
    required_capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateRepresentationRequest {
    memory_budget_bytes: u64,
    states: Vec<StateLayoutRequirement>,
    pub(crate) candidates: Vec<StateProviderPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateRepresentationDecision {
    pub(crate) plans: Vec<StateProviderPlan>,
    pub(crate) reserved_bytes: u64,
}

pub(crate) fn decode_request(payload: &str) -> Result<StateRepresentationRequest, String> {
    serde_json::from_str(payload)
        .map_err(|source| format!("invalid StateRepresentationRequest: {source}"))
}

pub(crate) fn decode_decision(payload: &str) -> Result<StateRepresentationDecision, String> {
    serde_json::from_str(payload)
        .map_err(|source| format!("invalid StateRepresentationDecision: {source}"))
}

fn validate_request(request: &StateRepresentationRequest) -> Result<(), String> {
    for candidate in &request.candidates {
        let requirement = request
            .states
            .iter()
            .find(|state| state.resource == candidate.resource)
            .ok_or_else(|| {
                format!(
                    "candidate resource {:?} has no state requirement",
                    candidate.resource
                )
            })?;
        validate_candidate_matches(requirement, candidate)?;
    }
    for requirement in &request.states {
        if !request
            .candidates
            .iter()
            .any(|candidate| candidate.resource == requirement.resource)
        {
            return Err(format!(
                "state resource {:?} has no provider candidates",
                requirement.resource
            ));
        }
    }
    Ok(())
}

fn validate_candidate_matches(
    requirement: &StateLayoutRequirement,
    candidate: &StateProviderPlan,
) -> Result<(), String> {
    if candidate.layout_id != requirement.layout_id
        || candidate.max_admitted_lineages != requirement.max_admitted_lineages
    {
        return Err(format!(
            "candidate {:?}/{:?} does not match its state layout requirement",
            candidate.resource, candidate.provider_id
        ));
    }
    if requirement
        .required_capabilities
        .iter()
        .any(|capability| candidate.capabilities.binary_search(capability).is_err())
    {
        return Err(format!(
            "candidate {:?}/{:?} lacks a required capability",
            candidate.resource, candidate.provider_id
        ));
    }
    Ok(())
}

pub(super) fn compile(
    request: &StateRepresentationRequest,
    graph: &SemanticGraph,
    timeline: &Timeline,
) -> Result<(StateRepresentationDecision, ScheduleDecision), String> {
    validate_request(request)?;
    validate_graph(graph)?;
    // Candidate catalogs are sorted. Strict score improvement therefore also
    // gives a stable provider-key tie-break without another decision ledger.
    fn choose(
        request: &StateRepresentationRequest,
        graph: &SemanticGraph,
        timeline: &Timeline,
        plans: &mut Vec<StateProviderPlan>,
        reserved_bytes: u64,
        best: &mut Option<(StateRepresentationDecision, ScheduleDecision)>,
        failure: &mut Option<String>,
    ) -> Result<(), String> {
        if request.memory_budget_bytes != 0 && reserved_bytes > request.memory_budget_bytes {
            return Ok(());
        }
        if plans.len() < request.states.len() {
            let resource = &request.states[plans.len()].resource;
            for candidate in request
                .candidates
                .iter()
                .filter(|candidate| &candidate.resource == resource)
            {
                let Some(bytes) = reserved_bytes.checked_add(candidate.reserved_bytes) else {
                    continue;
                };
                plans.push(candidate.clone());
                choose(request, graph, timeline, plans, bytes, best, failure)?;
                plans.pop();
            }
            return Ok(());
        }
        if plans.first().is_some_and(|first| {
            plans
                .iter()
                .any(|plan| plan.device_kind != first.device_kind)
        }) {
            return Ok(());
        }
        let decision = StateRepresentationDecision {
            plans: plans.clone(),
            reserved_bytes,
        };
        // task borrows this exact candidate, including retention and reserved
        // capacity. An infeasible cheap provider must not hide a legal COW plan.
        match scheduler::compile(graph, timeline, Some(&decision)) {
            Ok(schedule) => {
                if best
                    .as_ref()
                    .is_none_or(|(_, previous)| schedule.score() < previous.score())
                {
                    *best = Some((decision, schedule));
                }
            }
            Err(error) => *failure = Some(error),
        }
        Ok(())
    }
    let mut best = None;
    let mut failure = None;
    choose(
        request,
        graph,
        timeline,
        &mut Vec::new(),
        0,
        &mut best,
        &mut failure,
    )?;
    best.ok_or_else(|| {
        failure.unwrap_or_else(|| {
            "state representations exceed ExecutionOptions.memory_budget_bytes".into()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::super::semantic_graph::{decode_graph, decode_timeline};
    use super::*;
    use serde_json::json;

    #[test]
    fn representation_selection_and_capacity_match_oracle() {
        let requirement = |resource| {
            json!({"resource": resource, "layout_id": 0,
            "max_admitted_lineages": 1, "required_capabilities": []})
        };
        let candidate = |resource, provider, bytes, retained| {
            json!({
                "resource": resource, "provider_id": provider,
                "device_kind": "cpu", "layout_id": 0,
                "max_admitted_lineages": 1, "page_capacity": 3,
                "root_page_count": 1, "reserved_bytes": bytes, "capture_stable": false,
                "capabilities": if retained {vec!["state/exact_versions@1"]} else {vec![]}
            })
        };
        let mut undersized = candidate("state", "a_too_small", 4, true);
        undersized["page_capacity"] = json!(1);
        for late_read in [false, true] {
            let graph = decode_graph(&json!({
                "input_groups": [], "edges": [], "states": (["state", "unused"].map(|resource| json!({
                    "resource": resource, "scope": "instance", "key_fields": []}))),
                "tasks": (["read", "update"].iter().enumerate().map(|(task, kind)| {
                    let at = usize::from((task == 0) == late_read);
                    json!({"symbol": kind, "timing": {"after": format!("a{at}"), "before": format!("a{}", at + 1)},
                        "inputs": [], "outputs": [], "effects": [{"resource": "state", "kind": kind}],
                        "implementations": [{"implementation_id": "cpu", "required_capabilities": []}]})
                }).collect::<Vec<_>>())
            }).to_string()).unwrap();
            for arena_budget in [0, 8, 16, 24, 32] {
                let request = decode_request(&json!({
                    "memory_budget_bytes": arena_budget,
                    "states": [requirement("state"), requirement("unused")],
                    "candidates": [undersized, candidate("state", "cow", 16, true), candidate("state", "mutable", 8, false),
                        candidate("unused", "large", 5, false), candidate("unused", "small", 3, false)]
                }).to_string()).unwrap();
                for memory_budget in [8, 16, 20, 24, 28, 32, 40] {
                    let timeline = decode_timeline(&json!({
                        "capabilities": [], "memory_budget_bytes": memory_budget,
                        "anchors": (0..3).map(|i| json!({"symbol": format!("a{i}"), "ordinal": i})).collect::<Vec<_>>(),
                        "task_resources": (["read", "update"].map(|task| json!({"task": task, "workspace_bytes": 4}))),
                        "reserved_bytes": [5, 0, 0]
                    }).to_string()).unwrap();
                    // Independent product of providers, with direct old-value and
                    // resident+workspace+backend arithmetic, no planner helpers.
                    let expected = [("cow", 16, true), ("mutable", 8, false)]
                        .into_iter()
                        .flat_map(|(provider, bytes, retained)| {
                            [3, 5].map(|extra| (provider, bytes + extra, retained))
                        })
                        .filter(|&(_, bytes, retained)| {
                            (!late_read || retained)
                                && (arena_budget == 0 || bytes <= arena_budget)
                                && bytes + 9 <= memory_budget
                        })
                        .min_by_key(|&(provider, bytes, _)| (bytes + 9, provider));
                    let actual = compile(&request, &graph, &timeline);
                    assert_eq!(actual.is_ok(), expected.is_some());
                    if let Some((provider, bytes, _)) = expected {
                        let (decision, schedule) = actual.unwrap();
                        assert_eq!(decision.plans[0].provider_id, provider);
                        assert_eq!(decision.reserved_bytes, bytes);
                        assert_eq!(schedule.score()[0], bytes + 9);
                    }
                }
            }
        }
    }
}

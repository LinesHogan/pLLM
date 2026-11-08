//! Selection of one provider plan per state resource within the state-page budget.
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
    layout_id: u64,
    max_admitted_lineages: u64,
    page_capacity: u64,
    root_page_count: u64,
    pub(crate) reserved_bytes: u64,
    #[serde(default)]
    host_bytes: u64,
    capture_stable: bool,
    pub(crate) capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateRepresentationRequest {
    state_page_budget_bytes: u64,
    resources: Vec<String>,
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

pub(super) fn compile(
    request: &StateRepresentationRequest,
    graph: &SemanticGraph,
    timeline: &Timeline,
) -> Result<(StateRepresentationDecision, ScheduleDecision), String> {
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
        if request.state_page_budget_bytes != 0 && reserved_bytes > request.state_page_budget_bytes
        {
            return Ok(());
        }
        if plans.len() < request.resources.len() {
            let resource = &request.resources[plans.len()];
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
            let required: u64 = request
                .resources
                .iter()
                .map(|resource| {
                    request
                        .candidates
                        .iter()
                        .filter(|candidate| &candidate.resource == resource)
                        .map(|candidate| candidate.reserved_bytes)
                        .min()
                        .unwrap()
                })
                .sum();
            format!(
                "state pages need at least {required} bytes, but the state-page budget is {} bytes; \
                 lower gpu_memory_utilization to leave at least {} more bytes free, \
                 or lower max_num_seqs to reduce request-state storage. \
                 If you set ExecutionOptions.state_page_budget_bytes, raise it to at least {required}; \
                 for a read-only pretrained lookup table, reduce it or explicitly set placement=\"host\"",
                request.state_page_budget_bytes,
                required.saturating_sub(request.state_page_budget_bytes),
            )
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
        let candidate = |resource, provider, bytes, retained| {
            json!({
                "resource": resource, "provider_id": provider,
                "device_kind": "cpu", "layout_id": 0,
                "max_admitted_lineages": 1, "page_capacity": 3,
                "root_page_count": 1, "reserved_bytes": bytes, "capture_stable": false,
                "capabilities": if retained {vec!["state/exact_versions@1"]} else {vec![]}
            })
        };
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
                    "state_page_budget_bytes": arena_budget,
                    "resources": ["state", "unused"],
                    "candidates": [candidate("state", "cow", 16, true), candidate("state", "mutable", 8, false),
                        candidate("unused", "large", 5, false), candidate("unused", "small", 3, false)]
                }).to_string()).unwrap();
                for memory_budget in [8, 16, 20, 24, 28, 32, 40] {
                    let timeline = decode_timeline(&json!({
                        "capabilities": [], "execution_peak_budget_bytes": memory_budget,
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

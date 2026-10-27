//! Clean-room selection and validation for request-state representations.
//!
//! This module owns a closed, pure-data FFI contract.  It deliberately knows
//! no workload names, Tensor objects, Python callbacks, or runtime handles.

use super::scheduler::{self, ScheduleDecision};
use super::semantic_graph::{
    canonical_ascii_json, digest, validate_graph, validate_timeline, SemanticGraph, Timeline,
};
use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: u64 = 1;

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
    pub(crate) device_ordinal: u64,
    pub(crate) layout_id: u64,
    pub(crate) layout_digest: String,
    pub(crate) layout_namespace: String,
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
    layout_digest: String,
    layout_namespace: String,
    max_admitted_lineages: u64,
    required_capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateRepresentationRequest {
    memory_budget_bytes: u64,
    states: Vec<StateLayoutRequirement>,
    pub(crate) candidates: Vec<StateProviderPlan>,
    schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateRepresentationDecision {
    request_digest: String,
    plans: Vec<StateProviderPlan>,
    reserved_bytes: u64,
    signature: String,
    schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StatePhysicalPlan {
    pub(crate) plan_epoch: u64,
    pub(crate) request_digest: String,
    pub(crate) representation_signature: String,
    pub(crate) representations: Vec<StateProviderPlan>,
    pub(crate) reserved_bytes: u64,
    pub(crate) schema_version: u64,
}

pub(crate) fn decode_request(payload: &str) -> Result<StateRepresentationRequest, String> {
    serde_json::from_str(payload)
        .map_err(|source| format!("invalid StateRepresentationRequest: {source}"))
}

pub(crate) fn decode_physical_plan(payload: &str) -> Result<StatePhysicalPlan, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid StatePhysicalPlan: {source}"))
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{label} must be a non-empty string"));
    }
    Ok(())
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

fn require_sorted_unique_text(values: &[String], label: &str) -> Result<(), String> {
    for value in values {
        require_text(value, label)?;
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(format!("{label} must be sorted unique names"));
    }
    Ok(())
}

fn validate_requirement(requirement: &StateLayoutRequirement) -> Result<(), String> {
    require_text(&requirement.resource, "states.resource")?;
    require_digest(&requirement.layout_digest, "states.layout_digest")?;
    require_digest(&requirement.layout_namespace, "states.layout_namespace")?;
    if requirement.max_admitted_lineages == 0 {
        return Err("states.max_admitted_lineages must be positive".to_owned());
    }
    require_sorted_unique_text(
        &requirement.required_capabilities,
        "states.required_capabilities",
    )
}

fn validate_candidate(candidate: &StateProviderPlan) -> Result<(), String> {
    require_text(&candidate.resource, "candidates.resource")?;
    require_text(&candidate.provider_id, "candidates.provider_id")?;
    require_digest(&candidate.layout_digest, "candidates.layout_digest")?;
    require_digest(&candidate.layout_namespace, "candidates.layout_namespace")?;
    if candidate.device_kind == StateDeviceKind::Cpu && candidate.device_ordinal != 0 {
        return Err("CPU state provider device ordinal must be zero".to_owned());
    }
    if candidate.max_admitted_lineages == 0
        || candidate.page_capacity < candidate.max_admitted_lineages
        || candidate.root_page_count == 0
    {
        return Err("state provider capacity facts must be positive".to_owned());
    }
    require_sorted_unique_text(&candidate.capabilities, "candidates.capabilities")
}

fn validate_request(request: &StateRepresentationRequest) -> Result<(), String> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported StateRepresentationRequest schema_version: {}",
            request.schema_version
        ));
    }
    for requirement in &request.states {
        validate_requirement(requirement)?;
    }
    if request
        .states
        .windows(2)
        .any(|pair| pair[0].resource >= pair[1].resource)
    {
        return Err("request states must be sorted by unique resource".to_owned());
    }

    for candidate in &request.candidates {
        validate_candidate(candidate)?;
    }
    if request.candidates.windows(2).any(|pair| {
        (pair[0].resource.as_str(), pair[0].provider_id.as_str())
            >= (pair[1].resource.as_str(), pair[1].provider_id.as_str())
    }) {
        return Err(
            "state provider candidates must be sorted by unique resource/provider_id".to_owned(),
        );
    }

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
        || candidate.layout_digest != requirement.layout_digest
        || candidate.layout_namespace != requirement.layout_namespace
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
    plan_epoch: u64,
) -> Result<
    (
        StateRepresentationDecision,
        StatePhysicalPlan,
        ScheduleDecision,
    ),
    String,
> {
    validate_request(request)?;
    validate_graph(graph)?;
    validate_timeline(timeline)?;
    if plan_epoch == 0 {
        return Err("state physical plan epoch must be positive".into());
    }
    // Candidate catalogs are sorted. Strict score improvement therefore also
    // gives a stable provider-key tie-break without another decision ledger.
    fn choose(
        request: &StateRepresentationRequest,
        graph: &SemanticGraph,
        timeline: &Timeline,
        plan_epoch: u64,
        plans: &mut Vec<StateProviderPlan>,
        reserved_bytes: u64,
        best: &mut Option<(
            StateRepresentationDecision,
            StatePhysicalPlan,
            ScheduleDecision,
        )>,
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
                choose(
                    request, graph, timeline, plan_epoch, plans, bytes, best, failure,
                )?;
                plans.pop();
            }
            return Ok(());
        }
        if plans.first().is_some_and(|first| {
            plans.iter().any(|plan| {
                plan.device_kind != first.device_kind || plan.device_ordinal != first.device_ordinal
            })
        }) {
            return Ok(());
        }
        let mut decision = StateRepresentationDecision {
            request_digest: digest(&canonical_ascii_json(request)?),
            plans: plans.clone(),
            reserved_bytes,
            signature: String::new(),
            schema_version: SCHEMA_VERSION,
        };
        decision.signature = digest(&canonical_ascii_json(&decision)?);
        let physical = project(&decision, plan_epoch);
        // task borrows this exact candidate, including retention and reserved
        // capacity. An infeasible cheap provider must not hide a legal COW plan.
        match scheduler::compile(graph, timeline, Some(&physical)) {
            Ok(schedule) => {
                if best
                    .as_ref()
                    .is_none_or(|(_, _, previous)| schedule.score() < previous.score())
                {
                    *best = Some((decision, physical, schedule));
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
        plan_epoch,
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

fn project(decision: &StateRepresentationDecision, plan_epoch: u64) -> StatePhysicalPlan {
    StatePhysicalPlan {
        plan_epoch,
        request_digest: decision.request_digest.clone(),
        representation_signature: decision.signature.clone(),
        representations: decision.plans.clone(),
        reserved_bytes: decision.reserved_bytes,
        schema_version: SCHEMA_VERSION,
    }
}

pub(super) fn validate(
    request: &StateRepresentationRequest,
    decision: &StateRepresentationDecision,
    physical: &StatePhysicalPlan,
) -> Result<(), String> {
    validate_request(request)?;
    if decision.schema_version != SCHEMA_VERSION
        || physical.plan_epoch == 0
        || decision.request_digest != digest(&canonical_ascii_json(request)?)
        || decision.plans.len() != request.states.len()
    {
        return Err("selected state plan has an invalid request identity".into());
    }
    for (plan, requirement) in decision.plans.iter().zip(&request.states) {
        if plan.resource != requirement.resource || !request.candidates.contains(plan) {
            return Err("selected state plan differs from its original provider catalog".into());
        }
    }
    let bytes = decision
        .plans
        .iter()
        .try_fold(0_u64, |sum, plan| sum.checked_add(plan.reserved_bytes))
        .ok_or("selected state capacity overflows u64")?;
    if bytes != decision.reserved_bytes
        || (request.memory_budget_bytes != 0 && bytes > request.memory_budget_bytes)
        || decision.plans.first().is_some_and(|first| {
            decision.plans.iter().any(|plan| {
                plan.device_kind != first.device_kind || plan.device_ordinal != first.device_ordinal
            })
        })
    {
        return Err("selected state capacity or device differs from its request".into());
    }
    let mut unsigned = decision.clone();
    unsigned.signature.clear();
    if decision.signature != digest(&canonical_ascii_json(&unsigned)?)
        || physical != &project(decision, physical.plan_epoch)
    {
        return Err("state physical plan differs from its signed decision".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::semantic_graph::{decode_graph, decode_timeline};
    use super::*;
    use serde_json::json;

    #[test]
    fn representation_selection_and_capacity_are_signed_once() {
        let requirement = |resource| {
            json!({"resource": resource, "layout_id": 0,
            "layout_digest": "a".repeat(64), "layout_namespace": "b".repeat(64),
            "max_admitted_lineages": 1, "required_capabilities": []})
        };
        let candidate = |resource, provider, bytes, retained| {
            json!({
                "resource": resource, "provider_id": provider,
                "device_kind": "cpu", "device_ordinal": 0, "layout_id": 0,
                "layout_digest": "a".repeat(64), "layout_namespace": "b".repeat(64),
                "max_admitted_lineages": 1, "page_capacity": 3,
                "root_page_count": 1, "reserved_bytes": bytes, "capture_stable": false,
                "capabilities": if retained {vec!["state/exact_versions@1"]} else {vec![]}
            })
        };
        let mut undersized = candidate("state", "a_too_small", 4, true);
        undersized["page_capacity"] = json!(1);
        for late_read in [false, true] {
            let graph = decode_graph(&json!({
                "schema_version": 1, "workload_id": "joint", "execution_parameters_digest": "c".repeat(64),
                "input_groups": [], "edges": [], "states": (["state", "unused"].map(|resource| json!({
                    "resource": resource, "scope": "instance", "key_fields": []}))),
                "tasks": (["read", "update"].iter().enumerate().map(|(task, kind)| {
                    let at = usize::from((task == 0) == late_read);
                    json!({"symbol": kind, "timing": {"after": format!("a{at}"), "before": format!("a{}", at + 1)},
                        "inputs": [], "outputs": [], "effects": [{"resource": "state", "kind": kind}],
                        "implementations": [{"implementation_id": "cpu", "execution_kind": "native_task", "required_capabilities": []}]})
                }).collect::<Vec<_>>())
            }).to_string()).unwrap();
            for arena_budget in [0, 8, 16, 24, 32] {
                let request = decode_request(&json!({
                    "schema_version": 1, "memory_budget_bytes": arena_budget,
                    "states": [requirement("state"), requirement("unused")],
                    "candidates": [undersized, candidate("state", "cow", 16, true), candidate("state", "mutable", 8, false),
                        candidate("unused", "large", 5, false), candidate("unused", "small", 3, false)]
                }).to_string()).unwrap();
                for memory_budget in [8, 16, 20, 24, 28, 32, 40] {
                    let timeline = decode_timeline(&json!({
                        "schema_version": 1, "capabilities": [], "memory_budget_bytes": memory_budget,
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
                    let actual = compile(&request, &graph, &timeline, 7);
                    assert_eq!(actual.is_ok(), expected.is_some());
                    if let Some((provider, bytes, _)) = expected {
                        let (decision, physical, schedule) = actual.unwrap();
                        assert_eq!(decision.plans[0].provider_id, provider);
                        assert_eq!(decision.reserved_bytes, bytes);
                        assert_eq!(schedule.score()[0], bytes + 9);
                        assert_eq!(physical.representations, decision.plans);
                        assert_eq!(physical.representation_signature, decision.signature);
                        assert_eq!(physical.plan_epoch, 7);
                        let mut unsigned = decision.clone();
                        unsigned.signature.clear();
                        assert_eq!(
                            digest(&canonical_ascii_json(&unsigned).unwrap()),
                            decision.signature
                        );
                    }
                }
            }
        }
    }
}

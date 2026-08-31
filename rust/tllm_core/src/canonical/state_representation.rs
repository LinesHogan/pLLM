//! Clean-room selection and validation for request-state representations.
//!
//! This module owns a closed, pure-data FFI contract.  It deliberately knows
//! no workload names, Tensor objects, Python callbacks, or runtime handles.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write;

const SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StateRepresentation {
    ResidentModule,
    DensePageArena,
    SparsePageArena,
    VersionedOverlay,
    ShardedState,
    HostService,
    ExternalHandle,
}

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
    representation: StateRepresentation,
    pub(crate) device_kind: StateDeviceKind,
    pub(crate) device_ordinal: u64,
    layout_id: u64,
    layout_digest: String,
    layout_namespace: String,
    max_admitted_lineages: u64,
    max_live_versions_per_lineage: u64,
    root_page_count: u64,
    reserved_bytes: u64,
    capture_stable: bool,
    capabilities: Vec<String>,
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
    candidates: Vec<StateProviderPlan>,
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

fn error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

pub(crate) fn decode_request(payload: &str) -> Result<StateRepresentationRequest, String> {
    serde_json::from_str(payload)
        .map_err(|source| format!("invalid StateRepresentationRequest: {source}"))
}

pub(crate) fn decode_decision(payload: &str) -> Result<StateRepresentationDecision, String> {
    serde_json::from_str(payload)
        .map_err(|source| format!("invalid StateRepresentationDecision: {source}"))
}

fn decode_candidates(payload: &str) -> Result<Vec<StateProviderPlan>, String> {
    serde_json::from_str(payload)
        .map_err(|source| format!("invalid local state provider candidates: {source}"))
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
        || candidate.max_live_versions_per_lineage == 0
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

// serde_json maps preserve lexicographic object-key ordering by default.  The
// Python contract uses ASCII JSON, so normalize non-ASCII scalars explicitly.
fn canonical_ascii_json<T: Serialize>(value: &T) -> Result<String, String> {
    let value = serde_json::to_value(value)
        .map_err(|source| format!("failed to encode canonical JSON: {source}"))?;
    let encoded = serde_json::to_string(&value)
        .map_err(|source| format!("failed to serialize canonical JSON: {source}"))?;
    let mut canonical = String::with_capacity(encoded.len());
    for character in encoded.chars() {
        if character.is_ascii() {
            canonical.push(character);
            continue;
        }
        let scalar = character as u32;
        if scalar <= 0xffff {
            write!(&mut canonical, "\\u{scalar:04x}").expect("writing to String cannot fail");
            continue;
        }
        let surrogate = scalar - 0x1_0000;
        let high = 0xd800 + (surrogate >> 10);
        let low = 0xdc00 + (surrogate & 0x3ff);
        write!(&mut canonical, "\\u{high:04x}\\u{low:04x}").expect("writing to String cannot fail");
    }
    Ok(canonical)
}

fn digest(payload: &str) -> String {
    format!("{:x}", Sha256::digest(payload.as_bytes()))
}

fn decision_unsigned(decision: &StateRepresentationDecision) -> StateRepresentationDecision {
    let mut unsigned = decision.clone();
    unsigned.signature.clear();
    unsigned
}

fn decision_signature(decision: &StateRepresentationDecision) -> Result<String, String> {
    Ok(digest(&canonical_ascii_json(&decision_unsigned(decision))?))
}

fn compile(request: &StateRepresentationRequest) -> Result<StateRepresentationDecision, String> {
    validate_request(request)?;
    let mut plans = Vec::with_capacity(request.states.len());
    let mut reserved_bytes = 0_u64;
    for requirement in &request.states {
        let selected = request
            .candidates
            .iter()
            .filter(|candidate| candidate.resource == requirement.resource)
            .min_by(|left, right| {
                left.reserved_bytes
                    .cmp(&right.reserved_bytes)
                    .then_with(|| left.provider_id.cmp(&right.provider_id))
            })
            .ok_or_else(|| {
                format!(
                    "state resource {:?} has no compatible provider candidates",
                    requirement.resource
                )
            })?
            .clone();
        reserved_bytes = reserved_bytes
            .checked_add(selected.reserved_bytes)
            .ok_or_else(|| "state representation reserved bytes overflow".to_owned())?;
        plans.push(selected);
    }
    if request.memory_budget_bytes != 0 && reserved_bytes > request.memory_budget_bytes {
        return Err("state representation reserved bytes exceed memory budget".to_owned());
    }
    let mut decision = StateRepresentationDecision {
        request_digest: digest(&canonical_ascii_json(request)?),
        plans,
        reserved_bytes,
        signature: String::new(),
        schema_version: SCHEMA_VERSION,
    };
    decision.signature = decision_signature(&decision)?;
    Ok(decision)
}

fn validate_decision_shape(decision: &StateRepresentationDecision) -> Result<(), String> {
    if decision.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported StateRepresentationDecision schema_version: {}",
            decision.schema_version
        ));
    }
    require_digest(&decision.request_digest, "decision.request_digest")?;
    require_digest(&decision.signature, "decision.signature")?;
    for plan in &decision.plans {
        validate_candidate(plan)?;
    }
    if decision
        .plans
        .windows(2)
        .any(|pair| pair[0].resource >= pair[1].resource)
    {
        return Err("decision plans must be sorted by unique resource".to_owned());
    }
    Ok(())
}

fn validate_plan(
    request: &StateRepresentationRequest,
    decision: &StateRepresentationDecision,
) -> Result<(), String> {
    validate_decision_shape(decision)?;
    let expected = compile(request)?;
    if decision != &expected {
        return Err("state representation decision differs from Rust recomputation".to_owned());
    }
    Ok(())
}

fn lower_physical_plan(
    request: &StateRepresentationRequest,
    decision: &StateRepresentationDecision,
    plan_epoch: u64,
) -> Result<StatePhysicalPlan, String> {
    if plan_epoch == 0 {
        return Err("state physical plan epoch must be positive".to_owned());
    }
    validate_plan(request, decision)?;
    Ok(StatePhysicalPlan {
        plan_epoch,
        request_digest: decision.request_digest.clone(),
        representation_signature: decision.signature.clone(),
        representations: decision.plans.clone(),
        reserved_bytes: decision.reserved_bytes,
        schema_version: SCHEMA_VERSION,
    })
}

fn validate_physical_plan_shape(plan: &StatePhysicalPlan) -> Result<(), String> {
    if plan.plan_epoch == 0 || plan.schema_version != SCHEMA_VERSION {
        return Err("invalid state physical plan envelope".to_owned());
    }
    require_digest(&plan.request_digest, "physical.request_digest")?;
    require_digest(
        &plan.representation_signature,
        "physical.representation_signature",
    )?;
    for representation in &plan.representations {
        validate_candidate(representation)?;
    }
    if plan
        .representations
        .windows(2)
        .any(|pair| pair[0].resource >= pair[1].resource)
    {
        return Err("physical representations must be sorted by unique resource".to_owned());
    }
    Ok(())
}

pub(crate) fn validate_physical_plan(
    request: &StateRepresentationRequest,
    decision: &StateRepresentationDecision,
    physical: &StatePhysicalPlan,
) -> Result<(), String> {
    validate_physical_plan_shape(physical)?;
    let expected = lower_physical_plan(request, decision, physical.plan_epoch)?;
    if physical != &expected {
        return Err("state physical plan differs from Rust projection".to_owned());
    }
    Ok(())
}

fn validate_reconstruction(
    request: &StateRepresentationRequest,
    decision: &StateRepresentationDecision,
    local_candidates: &[StateProviderPlan],
) -> Result<(), String> {
    validate_plan(request, decision)?;
    if local_candidates != request.candidates {
        return Err(
            "local state provider candidates differ from the parent-signed request catalog"
                .to_owned(),
        );
    }
    for selected in &decision.plans {
        if !local_candidates
            .iter()
            .any(|candidate| candidate == selected)
        {
            return Err(
                "selected state provider is absent from local candidate catalog".to_owned(),
            );
        }
    }
    Ok(())
}

#[pyfunction]
fn compile_state_representation_plan(request_json: &str) -> PyResult<String> {
    let request = decode_request(request_json).map_err(error)?;
    let decision = compile(&request).map_err(error)?;
    canonical_ascii_json(&decision).map_err(error)
}

#[pyfunction]
fn validate_state_representation_plan(request_json: &str, decision_json: &str) -> PyResult<()> {
    let request = decode_request(request_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    validate_plan(&request, &decision).map_err(error)
}

#[pyfunction]
fn lower_canonical_state_physical_plan(
    request_json: &str,
    decision_json: &str,
    plan_epoch: u64,
) -> PyResult<String> {
    let request = decode_request(request_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    let physical = lower_physical_plan(&request, &decision, plan_epoch).map_err(error)?;
    canonical_ascii_json(&physical).map_err(error)
}

#[pyfunction]
fn validate_canonical_state_physical_plan(
    request_json: &str,
    decision_json: &str,
    physical_json: &str,
) -> PyResult<()> {
    let request = decode_request(request_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    let physical = decode_physical_plan(physical_json).map_err(error)?;
    validate_physical_plan(&request, &decision, &physical).map_err(error)
}

#[pyfunction]
fn validate_state_representation_reconstruction(
    request_json: &str,
    decision_json: &str,
    local_candidates_json: &str,
) -> PyResult<()> {
    let request = decode_request(request_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    let local_candidates = decode_candidates(local_candidates_json).map_err(error)?;
    validate_reconstruction(&request, &decision, &local_candidates).map_err(PyRuntimeError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_state_representation_plan, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_state_representation_plan,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        lower_canonical_state_physical_plan,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_canonical_state_physical_plan,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_state_representation_reconstruction,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        compile, lower_physical_plan, validate_physical_plan, validate_plan,
        validate_reconstruction, StateRepresentationDecision, StateRepresentationRequest,
    };

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn request_json() -> String {
        format!(
            r#"{{"memory_budget_bytes":24,"states":[{{"resource":"request_state","layout_id":7,"layout_digest":"{A}","layout_namespace":"{C}","max_admitted_lineages":2,"required_capabilities":["optimizer"]}}],"candidates":[{{"resource":"request_state","provider_id":"dense","representation":"dense_page_arena","device_kind":"cuda","device_ordinal":0,"layout_id":7,"layout_digest":"{A}","layout_namespace":"{C}","max_admitted_lineages":2,"max_live_versions_per_lineage":3,"root_page_count":1,"reserved_bytes":16,"capture_stable":true,"capabilities":["optimizer"]}},{{"resource":"request_state","provider_id":"resident","representation":"resident_module","device_kind":"cpu","device_ordinal":0,"layout_id":7,"layout_digest":"{A}","layout_namespace":"{C}","max_admitted_lineages":2,"max_live_versions_per_lineage":1,"root_page_count":1,"reserved_bytes":8,"capture_stable":false,"capabilities":["optimizer"]}}],"schema_version":1}}"#
        )
    }

    #[test]
    fn planner_selects_signs_rejects_tampering_and_requires_exact_reconstruction() {
        let request: StateRepresentationRequest = serde_json::from_str(&request_json()).unwrap();
        let decision = compile(&request).unwrap();
        assert_eq!(decision.plans[0].provider_id, "resident");
        assert_eq!(decision.reserved_bytes, 8);
        validate_plan(&request, &decision).unwrap();
        let physical = lower_physical_plan(&request, &decision, 3).unwrap();
        validate_physical_plan(&request, &decision, &physical).unwrap();
        let mut physical_tampered = physical.clone();
        physical_tampered.reserved_bytes = 9;
        assert!(validate_physical_plan(&request, &decision, &physical_tampered).is_err());

        let mut tampered: StateRepresentationDecision = decision.clone();
        tampered.signature = B.to_owned();
        assert!(validate_plan(&request, &tampered).is_err());

        let local = request.candidates.clone();
        validate_reconstruction(&request, &decision, &local).unwrap();
        let missing_selected = vec![local[0].clone()];
        assert!(validate_reconstruction(&request, &decision, &missing_selected).is_err());
    }
}

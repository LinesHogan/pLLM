use super::super::semantic_graph::{canonical_ascii_json, digest};
use super::types::*;
use super::validation::{digest as valid_digest, selected};

pub(crate) fn compile(inputs: &Inputs) -> Result<Decision, String> {
    let selected = selected(inputs)?;
    let mut value = Decision {
        request_digest: digest(&canonical_ascii_json(inputs)?),
        selected_candidate_id: selected.candidate_id.clone(),
        signature: String::new(),
        schema_version: SCHEMA_VERSION,
    };
    value.signature = digest(&canonical_ascii_json(&Decision {
        signature: String::new(),
        ..value.clone()
    })?);
    Ok(value)
}
pub(crate) fn validate_decision(inputs: &Inputs, decision: &Decision) -> Result<(), String> {
    if decision.schema_version != SCHEMA_VERSION
        || !valid_digest(&decision.request_digest)
        || !valid_digest(&decision.signature)
        || decision != &compile(inputs)?
    {
        return Err("parameter-overlay decision differs from Rust recomputation".to_owned());
    }
    Ok(())
}
pub(crate) fn lower(inputs: &Inputs, decision: &Decision) -> Result<Physical, String> {
    validate_decision(inputs, decision)?;
    let request = &inputs.request;
    let mut value = Physical {
        plan_epoch: request.plan_epoch,
        selected_candidate_id: decision.selected_candidate_id.clone(),
        attachment_signature: decision.signature.clone(),
        schedule_signature: request.schedule_signature.clone(),
        state_representation_signature: request.state_representation_signature.clone(),
        bindings: request.bindings.clone(),
        stage_anchors: request.stage_anchors.clone(),
        task_bindings: request.task_bindings.clone(),
        bundle_signature: String::new(),
        schema_version: SCHEMA_VERSION,
    };
    value.bundle_signature = digest(&canonical_ascii_json(&Physical {
        bundle_signature: String::new(),
        ..value.clone()
    })?);
    Ok(value)
}
pub(crate) fn validate_physical(
    inputs: &Inputs,
    decision: &Decision,
    physical: &Physical,
) -> Result<(), String> {
    if !valid_digest(&physical.bundle_signature) || physical != &lower(inputs, decision)? {
        return Err("parameter-overlay physical plan differs from Rust projection".to_owned());
    }
    Ok(())
}

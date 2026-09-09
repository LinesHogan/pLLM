//! Deterministic signing and mechanical physical projection for attachments.

use super::super::semantic_graph::{canonical_ascii_json, digest};
use super::types::{
    AttachmentCompilerInputs, BackendAttachmentDecision, BackendAttachmentPhysicalPlan,
    SCHEMA_VERSION,
};
use super::validation::{require_digest, validate_inputs};

fn decision_signature(decision: &BackendAttachmentDecision) -> Result<String, String> {
    let mut unsigned = decision.clone();
    unsigned.signature.clear();
    Ok(digest(&canonical_ascii_json(&unsigned)?))
}

pub(crate) fn compile(
    inputs: &AttachmentCompilerInputs,
) -> Result<BackendAttachmentDecision, String> {
    validate_inputs(inputs)?;
    let mut decision = BackendAttachmentDecision {
        request_digest: digest(&canonical_ascii_json(inputs)?),
        signature: String::new(),
        schema_version: SCHEMA_VERSION,
    };
    decision.signature = decision_signature(&decision)?;
    Ok(decision)
}

pub(crate) fn validate_decision(
    inputs: &AttachmentCompilerInputs,
    decision: &BackendAttachmentDecision,
) -> Result<(), String> {
    if decision.schema_version != SCHEMA_VERSION {
        return Err("unsupported attachment decision schema".to_owned());
    }
    require_digest(&decision.request_digest, "decision.request_digest")?;
    require_digest(&decision.signature, "decision.signature")?;
    if decision != &compile(inputs)? {
        return Err("attachment decision differs from Rust recomputation".to_owned());
    }
    Ok(())
}

pub(crate) fn lower(
    inputs: &AttachmentCompilerInputs,
    decision: &BackendAttachmentDecision,
) -> Result<BackendAttachmentPhysicalPlan, String> {
    validate_decision(inputs, decision)?;
    let probe = &inputs.probe;
    let plan = &inputs.plan;
    let mut physical = BackendAttachmentPhysicalPlan {
        plan_epoch: plan.schedule_physical.plan_epoch,
        adapter_id: probe.adapter_id.clone(),
        attachment_signature: decision.signature.clone(),
        schedule_signature: plan.schedule_physical.schedule_signature.clone(),
        state_representation_signature: plan.state_physical.representation_signature.clone(),
        device_kind: probe.device_kind.clone(),
        device_ordinal: probe.device_ordinal,
        topology: probe.topology.clone(),
        graph_mode: probe.graph_mode.clone(),
        bundle_signature: String::new(),
        schema_version: SCHEMA_VERSION,
    };
    physical.bundle_signature = physical_signature(&physical)?;
    Ok(physical)
}

fn physical_signature(physical: &BackendAttachmentPhysicalPlan) -> Result<String, String> {
    let mut unsigned = physical.clone();
    unsigned.bundle_signature.clear();
    Ok(digest(&canonical_ascii_json(&unsigned)?))
}

pub(crate) fn validate_physical(
    inputs: &AttachmentCompilerInputs,
    decision: &BackendAttachmentDecision,
    physical: &BackendAttachmentPhysicalPlan,
) -> Result<(), String> {
    require_digest(&physical.bundle_signature, "physical.bundle_signature")?;
    if physical != &lower(inputs, decision)? {
        return Err("physical attachment differs from Rust projection".to_owned());
    }
    Ok(())
}

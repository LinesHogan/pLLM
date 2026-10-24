//! Deterministic signing and mechanical physical projection for attachments.

use super::super::semantic_graph::{canonical_ascii_json, digest};
use super::types::{
    AttachmentCompilerInputs, BackendAttachmentDecision, BackendAttachmentPhysicalPlan,
    PostSamplerPlacement, SCHEMA_VERSION,
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
        post_sampler: post_sampler(inputs)?,
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
        post_sampler: decision.post_sampler.clone(),
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

fn post_sampler(inputs: &AttachmentCompilerInputs) -> Result<PostSamplerPlacement, String> {
    use super::super::semantic_graph::{InputSource, StateEffectKind};
    let graph = &inputs.plan.semantic_graph;
    let timeline = &inputs.plan.timeline;
    let tasks = &inputs.plan.schedule_physical.tasks;
    let Some(source) = &timeline.visibility else { return Ok(PostSamplerPlacement::Main); };
    let Some(sampler) = timeline.anchors.iter().find(|a| a.symbol == "sampler.pre_draw") else { return Ok(PostSamplerPlacement::Main); };
    let prefix: Vec<_> = tasks.iter().filter(|t| u64::from(t.anchor_id) <= sampler.ordinal).map(|t| &graph.tasks[t.task_id as usize]).collect();
    if !timeline.capabilities.iter().any(|c| c == "compute/cuda@1")
        || !tasks.iter().any(|t| u64::from(t.anchor_id) > sampler.ordinal)
        || !inputs.request.overlay_bindings.is_empty() || !inputs.request.activation_bindings.is_empty()
        || graph.tasks.iter().any(|t| t.replay.is_some() || t.visibility.is_some())
        || prefix.iter().any(|t| t.effects.iter().any(|e| e.kind != StateEffectKind::Read)
            || t.math.as_ref().is_some_and(|m| !m.input_gradients.is_empty() || !m.parameter_gradients.is_empty()))
        || !super::super::visibility::permits_overlap(source)?
    { return Ok(PostSamplerPlacement::Main); }
    Ok(if !prefix.is_empty() || graph.tasks.iter().any(|t| t.inputs.iter().any(|p| matches!(p.source, InputSource::SamplerFeedback))) {
        PostSamplerPlacement::Lagged
    } else { PostSamplerPlacement::Writer })
}

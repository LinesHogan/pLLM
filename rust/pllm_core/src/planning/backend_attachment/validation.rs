//! Reject attachments a workload author can declare but the backend cannot run.
//!
//! Python planning builds every input from one declaration and one backend probe, so only
//! mistakes reachable from that declaration are checked here: export and training inputs,
//! the trained state's scope, sampled-token readers and CUDA-graph overlay state. Buffer
//! shapes and dtypes are checked earlier, where Python resolves them against the model.

use super::super::semantic_graph::{InputSource, OutputKind, StateEffectKind, StateScope};
use super::types::{AttachmentGraphMode, AttachmentKind, BackendAttachment, SamplerStage};

pub(crate) fn validate_attachment(attachment: &BackendAttachment) -> Result<(), String> {
    match attachment.spec.kind {
        AttachmentKind::Observation => validate_observation(attachment)?,
        AttachmentKind::RequestTraining => validate_request_training(attachment)?,
        _ => {}
    }
    let graph = &attachment.plan.semantic_graph;
    if attachment.probe.graph_mode == AttachmentGraphMode::CudaGraph
        && graph.states.iter().any(|state| {
            graph.model_reads.contains(&state.resource) && state.scope != StateScope::Request
        })
    {
        return Err(
            "CUDA-graph overlays read request state; declare the overlay state as \
                    pllm.RequestState or set cuda_graphs=False"
                .into(),
        );
    }
    Ok(())
}

fn validate_observation(attachment: &BackendAttachment) -> Result<(), String> {
    if attachment.probe.hidden_resources.is_empty() {
        return Err("exports need a hidden input, e.g. `h: torch.Tensor = pllm.hidden(0)`".into());
    }
    let graph = &attachment.plan.semantic_graph;
    // The observation path runs only bound export tasks between publication and sampling.
    if attachment.task_bindings.len() != graph.tasks.len()
        || graph.tasks.iter().any(|task| {
            task.effects
                .iter()
                .any(|effect| effect.kind != StateEffectKind::Read)
                || task
                    .outputs
                    .iter()
                    .any(|output| output.kind != OutputKind::Host)
                || task.timing.after != "state.publish"
                || task.timing.before != "sampler.pre_draw"
                || task.inputs.iter().any(|input| {
                    !matches!(
                        input.source,
                        InputSource::Hidden { .. }
                            | InputSource::StepId
                            | InputSource::StateSelection { .. }
                    )
                })
        })
    {
        return Err(
            "a workload with exports runs only exports in Phase.OBSERVE that read hidden \
                    rows and state; remove its other operations, state writes and phase overrides"
                .into(),
        );
    }
    Ok(())
}

fn validate_request_training(attachment: &BackendAttachment) -> Result<(), String> {
    if attachment.probe.hidden_resources.is_empty() {
        return Err("train and sampler operations need a hidden input, \
                    e.g. `h: torch.Tensor = pllm.hidden(0)`"
            .into());
    }
    let graph = &attachment.plan.semantic_graph;
    let state = graph
        .states
        .iter()
        .find(|state| state.resource == attachment.spec.state_resource)
        .expect("Python planning declares the trained state");
    if state.scope != StateScope::Request
        && attachment.spec.sampler_stage == SamplerStage::PackedPostFilterExact
    {
        return Err(
            "declare the trained state as pllm.RequestState; only workloads with \
                    @pllm.add_logits train an InstanceState"
                .into(),
        );
    }
    let transition = attachment.task_bindings.last().unwrap().task_id;
    let mut past_update = false;
    for scheduled in &attachment.plan.schedule.tasks {
        let task = &graph.tasks[scheduled.task_id as usize];
        let reads_feedback = task
            .inputs
            .iter()
            .any(|input| matches!(input.source, InputSource::SamplerFeedback));
        let read_only_helper = task
            .effects
            .iter()
            .all(|effect| effect.kind == StateEffectKind::Read)
            && task
                .outputs
                .iter()
                .all(|output| output.kind == OutputKind::Temporary);
        if reads_feedback
            && (attachment.spec.sampler_stage != SamplerStage::PackedPreFilterAdd
                || scheduled.anchor_id <= attachment.sampler_anchor_id
                || past_update
                || (scheduled.task_id != transition && !read_only_helper))
        {
            return Err(
                "sampled tokens are readable only beside @pllm.add_logits, by the train \
                        operation or by read-only operations that feed it"
                    .into(),
            );
        }
        past_update |= scheduled.task_id == transition;
    }
    Ok(())
}

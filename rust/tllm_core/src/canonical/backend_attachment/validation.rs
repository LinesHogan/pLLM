//! Semantic, plan, and backend-fact validation for attachment signing.

use super::super::scheduler::{validate_physical as validate_schedule_physical, PhysicalPlan};
use super::super::semantic_graph::{
    HiddenPhase as SemanticHiddenPhase, HiddenSite as SemanticHiddenSite, InputSource, OutputKind,
    SemanticGraph, SemanticTask, StateEffectKind, StateScope, Timeline,
};
use super::super::state_representation::{
    validate_physical_plan as validate_state_physical, StateDeviceKind, StatePhysicalPlan,
};
use super::types::{
    AttachmentCompilerInputs, AttachmentHiddenResource, AttachmentKind, AttachmentOverlayStage,
    AttachmentTaskBinding, AttachmentTaskRole, BackendAttachmentRequest, BackendPlacementProbe,
    HiddenLayerKind, HiddenPhase, HiddenSite, StreamDependency, SCHEMA_VERSION,
};

pub(crate) fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{label} must be a non-empty string"));
    }
    Ok(())
}

pub(crate) fn require_digest(value: &str, label: &str) -> Result<(), String> {
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
        return Err(format!("{label} must be sorted unique strings"));
    }
    Ok(())
}

fn validate_request(
    request: &BackendAttachmentRequest,
    probe: &BackendPlacementProbe,
) -> Result<(), String> {
    if request.schema_version != SCHEMA_VERSION || request.packed_row_schema_version != 1 {
        return Err("unsupported backend attachment schema".to_owned());
    }
    require_text(&request.state_resource, "state_resource")?;
    if request.attachment_kind == AttachmentKind::RequestSamplerTraining
        && probe.hidden_resources.is_empty()
    {
        return Err("attachment requires hidden resources".to_owned());
    }
    require_text(&request.sampler_anchor_symbol, "sampler_anchor_symbol")?;
    match request.attachment_kind {
        AttachmentKind::RequestSamplerTraining => {
            if request.task_bindings.len() != 2
                || request.task_bindings[0].role != AttachmentTaskRole::Compute
                || request.task_bindings[1].role != AttachmentTaskRole::Transition
                || request.task_bindings[0].task_id == request.task_bindings[1].task_id
            {
                return Err("sampler attachment tasks must be compute then transition".to_owned());
            }
        }
        AttachmentKind::DenseParameterOverlay => {
            if request.task_bindings.len() != 2
                || request.task_bindings[0].role != AttachmentTaskRole::Transition
                || request.task_bindings[1].role != AttachmentTaskRole::Publish
                || request.task_bindings[0].task_id == request.task_bindings[1].task_id
            {
                return Err(
                    "dense overlay attachment tasks must be transition then publish".to_owned(),
                );
            }
        }
    }
    for binding in &request.task_bindings {
        require_text(
            &binding.implementation_id,
            "task_bindings.implementation_id",
        )?;
    }
    if request.stream_dependencies.is_empty()
        || request
            .stream_dependencies
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err("stream dependencies must be sorted unique".to_owned());
    }
    match request.attachment_kind {
        AttachmentKind::RequestSamplerTraining => {
            if !request.overlay_field_ids.is_empty() || !request.overlay_stage_anchors.is_empty() {
                return Err("request sampler attachment cannot bind overlay fields".to_owned());
            }
            if request.stream_dependencies
                != vec![
                    StreamDependency::HiddenProducerToCompute,
                    StreamDependency::ComputeToSampler,
                    StreamDependency::SamplerToTransition,
                    StreamDependency::TransitionToRelease,
                ]
            {
                return Err(
                    "request sampler training attachment requires all stream dependencies"
                        .to_owned(),
                );
            }
        }
        AttachmentKind::DenseParameterOverlay => {
            if request.overlay_field_ids.is_empty()
                || request
                    .overlay_field_ids
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
            {
                return Err("dense overlay attachment requires sorted unique fields".to_owned());
            }
            let stages = request
                .overlay_stage_anchors
                .iter()
                .map(|stage| &stage.stage)
                .collect::<Vec<_>>();
            if stages
                != vec![
                    &AttachmentOverlayStage::Pin,
                    &AttachmentOverlayStage::Apply,
                    &AttachmentOverlayStage::Capture,
                    &AttachmentOverlayStage::Transition,
                    &AttachmentOverlayStage::Publish,
                    &AttachmentOverlayStage::Release,
                    &AttachmentOverlayStage::Finish,
                ]
            {
                return Err("dense overlay attachment stages are not closed".to_owned());
            }
            for stage in &request.overlay_stage_anchors {
                require_text(&stage.anchor_symbol, "overlay_stage_anchors.anchor_symbol")?;
            }
        }
    }
    Ok(())
}

fn validate_probe(probe: &BackendPlacementProbe) -> Result<(), String> {
    if probe.schema_version != 1 {
        return Err("unsupported backend placement probe schema".to_owned());
    }
    require_text(&probe.adapter_id, "probe.adapter_id")?;
    require_digest(&probe.backend_abi_digest, "probe.backend_abi_digest")?;
    require_digest(&probe.runner_fingerprint, "probe.runner_fingerprint")?;
    require_digest(&probe.model_fingerprint, "probe.model_fingerprint")?;
    if probe.device_kind == StateDeviceKind::Cpu && probe.device_ordinal != 0 {
        return Err("CPU backend probe device ordinal must be zero".to_owned());
    }
    if probe.tensor_parallel_size != 1
        || probe.pipeline_parallel_size != 1
        || probe.hidden_size == 0
        || probe.max_packed_rows == 0
    {
        return Err("backend placement probe topology is unsupported".to_owned());
    }
    for resource in &probe.hidden_resources {
        require_text(&resource.binding.layer, "probe.hidden_resources.layer")?;
        require_text(
            &resource.binding.module_path,
            "probe.hidden_resources.module_path",
        )?;
        if resource.binding.layer_kind == HiddenLayerKind::Index {
            let index = resource
                .binding
                .layer
                .parse::<i64>()
                .map_err(|_| "probe indexed hidden layer is invalid".to_owned())?;
            if index.to_string() != resource.binding.layer {
                return Err("probe indexed hidden layer is not canonical".to_owned());
            }
        }
    }
    if probe.hidden_resources.windows(2).any(|pair| {
        (
            pair[0].binding.layer.as_str(),
            &pair[0].binding.layer_kind,
            &pair[0].binding.site,
        ) >= (
            pair[1].binding.layer.as_str(),
            &pair[1].binding.layer_kind,
            &pair[1].binding.site,
        )
    }) {
        return Err("probe hidden resources must be sorted unique".to_owned());
    }
    require_sorted_unique_text(&probe.capabilities, "probe.capabilities")
}

fn semantic_hidden_key(input: &InputSource) -> Option<(i64, u8, u8)> {
    let InputSource::Hidden { layer, site, phase } = input else {
        return None;
    };
    let site = match site {
        SemanticHiddenSite::Post => 0,
        SemanticHiddenSite::BlockOutput => 1,
    };
    let phase = match phase {
        SemanticHiddenPhase::Prefill => 0,
        SemanticHiddenPhase::Decode => 1,
    };
    Some((*layer, site, phase))
}

fn attachment_hidden_key(resource: &AttachmentHiddenResource) -> Result<(i64, u8, u8), String> {
    if resource.binding.layer_kind != HiddenLayerKind::Index {
        return Err("attachment hidden resources must use indexed graph locators".to_owned());
    }
    let layer = resource
        .binding
        .layer
        .parse::<i64>()
        .map_err(|_| "attachment hidden resource layer is invalid".to_owned())?;
    let site = match resource.binding.site {
        HiddenSite::BlockOutput => 1,
    };
    let phase = match resource.phase {
        HiddenPhase::Prefill => 0,
        HiddenPhase::Decode => 1,
    };
    Ok((layer, site, phase))
}

fn task_for_dense_id<'a>(
    graph: &'a SemanticGraph,
    binding: &AttachmentTaskBinding,
) -> Result<&'a SemanticTask, String> {
    let task_id = usize::try_from(binding.task_id)
        .map_err(|_| "attachment task ID cannot address the semantic graph".to_owned())?;
    graph
        .tasks
        .get(task_id)
        .ok_or_else(|| "attachment task ID is not a dense semantic graph ID".to_owned())
}

fn has_decode_selection(task: &SemanticTask, state_resource: &str) -> bool {
    task.inputs.iter().any(|input| {
        matches!(
            &input.source,
            InputSource::StateSelection {
                resource,
                phase: SemanticHiddenPhase::Decode,
            } if resource == state_resource
        )
    })
}

fn validate_request_sampler_training_graph(
    request: &BackendAttachmentRequest,
    probe: &BackendPlacementProbe,
    graph: &SemanticGraph,
) -> Result<(), String> {
    let state = graph
        .states
        .binary_search_by(|state| state.resource.cmp(&request.state_resource))
        .map(|index| &graph.states[index])
        .map_err(|_| "attachment state resource is absent from the semantic graph".to_owned())?;
    if state.scope != StateScope::Request {
        return Err("attachment request state differs from semantic graph state".to_owned());
    }

    let compute = task_for_dense_id(graph, &request.task_bindings[0])?;
    if !compute.effects.iter().any(|effect| {
        effect.resource == request.state_resource && effect.kind == StateEffectKind::Read
    }) || !has_decode_selection(compute, &request.state_resource)
    {
        return Err("attachment compute task is missing request-state read semantics".to_owned());
    }
    if !compute
        .inputs
        .iter()
        .any(|input| matches!(&input.source, InputSource::PackedRows))
        || !compute
            .inputs
            .iter()
            .any(|input| matches!(&input.source, InputSource::PackedSamplerRows))
        || !compute
            .outputs
            .iter()
            .any(|output| output.kind == OutputKind::Sampler)
    {
        return Err("attachment compute task is missing packed sampler semantics".to_owned());
    }
    let mut task_hidden = compute
        .inputs
        .iter()
        .filter_map(|input| semantic_hidden_key(&input.source))
        .collect::<Vec<_>>();
    let mut signed_hidden = probe
        .hidden_resources
        .iter()
        .map(attachment_hidden_key)
        .collect::<Result<Vec<_>, _>>()?;
    task_hidden.sort_unstable();
    signed_hidden.sort_unstable();
    if task_hidden != signed_hidden {
        return Err(
            "attachment compute hidden inputs differ from signed hidden resources".to_owned(),
        );
    }

    let transition = task_for_dense_id(graph, &request.task_bindings[1])?;
    if !transition.effects.iter().any(|effect| {
        effect.resource == request.state_resource && effect.kind == StateEffectKind::Transition
    }) || !has_decode_selection(transition, &request.state_resource)
    {
        return Err(
            "attachment transition task is missing request-state transition semantics".to_owned(),
        );
    }
    Ok(())
}

fn validate_dense_parameter_overlay_graph(
    request: &BackendAttachmentRequest,
    graph: &SemanticGraph,
) -> Result<(), String> {
    let state = graph
        .states
        .binary_search_by(|state| state.resource.cmp(&request.state_resource))
        .map(|index| &graph.states[index])
        .map_err(|_| "dense overlay state resource is absent from the semantic graph".to_owned())?;
    if state.scope != StateScope::Request {
        return Err("dense overlay state differs from semantic graph state".to_owned());
    }
    let transition = task_for_dense_id(graph, &request.task_bindings[0])?;
    if !transition.effects.iter().any(|effect| {
        effect.resource == request.state_resource
            && matches!(
                effect.kind,
                StateEffectKind::Transition | StateEffectKind::TransitionMaybe
            )
    }) {
        return Err("dense overlay transition task is missing transition semantics".to_owned());
    }
    let publish = task_for_dense_id(graph, &request.task_bindings[1])?;
    if !publish.effects.iter().any(|effect| {
        effect.resource == request.state_resource && effect.kind == StateEffectKind::Publish
    }) {
        return Err("dense overlay publish task is missing publish semantics".to_owned());
    }
    if request.task_bindings[0].anchor_id > request.task_bindings[1].anchor_id {
        return Err("dense overlay task anchors are out of order".to_owned());
    }
    Ok(())
}

fn validate_cross_binding(
    request: &BackendAttachmentRequest,
    probe: &BackendPlacementProbe,
    graph: &SemanticGraph,
    timeline: &Timeline,
    schedule: &PhysicalPlan,
    state: &StatePhysicalPlan,
) -> Result<(), String> {
    validate_request(request, probe)?;
    validate_probe(probe)?;
    if schedule.plan_epoch != state.plan_epoch {
        return Err("schedule and state epochs differ".to_owned());
    }
    match request.attachment_kind {
        AttachmentKind::RequestSamplerTraining => {
            validate_request_sampler_training_graph(request, probe, graph)?;
        }
        AttachmentKind::DenseParameterOverlay => {
            validate_dense_parameter_overlay_graph(request, graph)?;
        }
    }
    let provider = state
        .representations
        .binary_search_by(|provider| provider.resource.cmp(&request.state_resource))
        .map(|index| &state.representations[index])
        .map_err(|_| "attachment state resource is not physically selected".to_owned())?;
    if provider.device_kind != probe.device_kind || provider.device_ordinal != probe.device_ordinal
    {
        return Err("attachment placement differs from selected state provider".to_owned());
    }
    let sampler_anchor = timeline
        .anchors
        .get(request.sampler_anchor_id as usize)
        .ok_or_else(|| "attachment sampler anchor is outside the timeline".to_owned())?;
    if sampler_anchor.ordinal != request.sampler_anchor_id as u64
        || sampler_anchor.symbol != request.sampler_anchor_symbol
    {
        return Err("attachment sampler anchor differs from timeline".to_owned());
    }
    for stage in &request.overlay_stage_anchors {
        let anchor = timeline
            .anchors
            .get(stage.anchor_id as usize)
            .ok_or_else(|| "dense overlay stage anchor is outside the timeline".to_owned())?;
        if anchor.ordinal != stage.anchor_id as u64 || anchor.symbol != stage.anchor_symbol {
            return Err("dense overlay stage anchor differs from timeline".to_owned());
        }
    }
    for binding in &request.task_bindings {
        let task = schedule
            .tasks
            .iter()
            .find(|task| task.task_id == binding.task_id)
            .ok_or_else(|| "attachment task binding is not in physical plan".to_owned())?;
        if task.implementation_id != binding.implementation_id
            || task.anchor_id != binding.anchor_id
        {
            return Err("attachment task binding differs from physical plan".to_owned());
        }
    }
    match request.attachment_kind {
        AttachmentKind::RequestSamplerTraining => {
            let compute = &request.task_bindings[0];
            let transition = &request.task_bindings[1];
            if compute.anchor_id >= request.sampler_anchor_id
                || transition.anchor_id <= request.sampler_anchor_id
            {
                return Err("attachment task anchors violate sampler timing".to_owned());
            }
        }
        AttachmentKind::DenseParameterOverlay => {
            let transition = &request.task_bindings[0];
            let publish = &request.task_bindings[1];
            if request
                .overlay_stage_anchors
                .windows(2)
                .any(|pair| pair[0].anchor_id > pair[1].anchor_id)
                || request.overlay_stage_anchors[3].anchor_id != transition.anchor_id
                || request.overlay_stage_anchors[4].anchor_id != publish.anchor_id
                || request.overlay_stage_anchors[5].anchor_id != request.sampler_anchor_id
                || transition.anchor_id > publish.anchor_id
            {
                return Err("dense overlay task anchors violate publication timing".to_owned());
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_inputs(inputs: &AttachmentCompilerInputs) -> Result<(), String> {
    let plan = &inputs.plan;
    super::super::tensor_layout::validate_decision(
        &plan.tensor_layout_request,
        &plan.tensor_layout_decision,
    )?;
    validate_schedule_physical(
        &plan.semantic_graph,
        &plan.timeline,
        &plan.schedule_decision,
        &plan.schedule_physical,
    )?;
    validate_state_physical(
        &plan.state_request,
        &plan.state_decision,
        &plan.state_physical,
    )?;
    for state in &plan.state_physical.representations {
        let binding = plan
            .tensor_layout_decision
            .bindings
            .iter()
            .find(|binding| binding.resource == state.resource)
            .ok_or_else(|| "state representation lacks a signed tensor layout".to_owned())?;
        if state.layout_id != u64::from(binding.layout_id)
            || state.layout_namespace != binding.layout_namespace
            || state.layout_digest != binding.layout_digest
        {
            return Err("state representation differs from signed tensor layout".to_owned());
        }
    }
    let layout = plan
        .tensor_layout_request
        .layouts
        .iter()
        .find(|layout| layout.resource == inputs.request.state_resource)
        .ok_or_else(|| "attachment state has no tensor layout".to_owned())?;
    if inputs.request.overlay_field_ids.iter().any(|field| {
        !layout
            .layout
            .fields
            .iter()
            .any(|known| known.field_id == u64::from(*field))
    }) {
        return Err("overlay field is absent from the signed tensor layout".to_owned());
    }
    validate_cross_binding(
        &inputs.request,
        &inputs.probe,
        &plan.semantic_graph,
        &plan.timeline,
        &plan.schedule_physical,
        &plan.state_physical,
    )
}

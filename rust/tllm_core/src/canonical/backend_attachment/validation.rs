//! Semantic, plan, and backend-fact validation for attachment signing.

use super::super::scheduler::{validate_physical as validate_schedule_physical, PhysicalPlan};
use super::super::semantic_graph::{
    HiddenPhase as SemanticHiddenPhase, HiddenSite as SemanticHiddenSite, InputSource, OutputKind,
    RequestKeyField as SemanticRequestKeyField, SemanticGraph, SemanticTask, StateEffectKind,
    StateScope, Timeline,
};
use super::super::state_representation::{
    validate_physical_plan as validate_state_physical, StateDeviceKind, StatePhysicalPlan,
};
use super::types::{
    AttachmentCompilerInputs, AttachmentKind, AttachmentTaskBinding, AttachmentTaskRole,
    BackendAttachmentRequest, BackendPlacementProbe, HiddenBinding, HiddenLayerKind, HiddenPhase,
    HiddenSite, RequestKeyField, StreamDependency, SCHEMA_VERSION,
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

fn validate_request(request: &BackendAttachmentRequest) -> Result<(), String> {
    if request.schema_version != SCHEMA_VERSION {
        return Err("unsupported backend attachment schema".to_owned());
    }
    require_text(&request.adapter_id, "adapter_id")?;
    require_text(&request.state_resource, "state_resource")?;
    for (value, label) in [
        (&request.backend_abi_digest, "backend_abi_digest"),
        (&request.runner_fingerprint, "runner_fingerprint"),
        (&request.model_fingerprint, "model_fingerprint"),
        (&request.schedule_signature, "schedule_signature"),
        (&request.state_request_digest, "state_request_digest"),
        (
            &request.state_representation_signature,
            "state_representation_signature",
        ),
        (
            &request.execution_parameters_digest,
            "execution_parameters_digest",
        ),
    ] {
        require_digest(value, label)?;
    }
    if request.device_kind == StateDeviceKind::Cpu && request.device_ordinal != 0 {
        return Err("CPU attachment device ordinal must be zero".to_owned());
    }
    if request.tensor_parallel_size != 1
        || request.pipeline_parallel_size != 1
        || request.hidden_size == 0
        || request.max_packed_rows == 0
        || request.packed_row_schema_version != 1
        || request.plan_epoch == 0
    {
        return Err("attachment topology or capacity facts are unsupported".to_owned());
    }
    if request.hidden_resources.is_empty() {
        return Err("attachment requires hidden resources".to_owned());
    }
    for resource in &request.hidden_resources {
        require_text(&resource.layer, "hidden_resources.layer")?;
        require_text(&resource.module_path, "hidden_resources.module_path")?;
        if resource.layer_kind == HiddenLayerKind::Index {
            let index = resource
                .layer
                .parse::<i64>()
                .map_err(|_| "indexed hidden layer must be a signed integer".to_owned())?;
            if index.to_string() != resource.layer {
                return Err("indexed hidden layer must be canonical".to_owned());
            }
        }
    }
    if request.hidden_resources.windows(2).any(|pair| {
        (pair[0].layer.as_str(), &pair[0].layer_kind, &pair[0].site)
            >= (pair[1].layer.as_str(), &pair[1].layer_kind, &pair[1].site)
    }) {
        return Err("hidden resources must be sorted unique".to_owned());
    }
    require_text(&request.sampler_anchor_symbol, "sampler_anchor_symbol")?;
    if request.task_bindings.len() != 2
        || request.task_bindings[0].role != AttachmentTaskRole::Compute
        || request.task_bindings[1].role != AttachmentTaskRole::Transition
        || request.task_bindings[0].task_id == request.task_bindings[1].task_id
    {
        return Err("attachment task bindings must be compute then transition".to_owned());
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
    }
    if request.request_key_fields.is_empty()
        || request
            .request_key_fields
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err("request key fields must be sorted unique".to_owned());
    }
    require_sorted_unique_text(&request.capabilities, "capabilities")
}

fn validate_probe(probe: &BackendPlacementProbe) -> Result<(), String> {
    if probe.schema_version != SCHEMA_VERSION {
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
        || probe.hidden_resources.is_empty()
    {
        return Err("backend placement probe topology is unsupported".to_owned());
    }
    for resource in &probe.hidden_resources {
        require_text(&resource.layer, "probe.hidden_resources.layer")?;
        require_text(&resource.module_path, "probe.hidden_resources.module_path")?;
        if resource.layer_kind == HiddenLayerKind::Index {
            let index = resource
                .layer
                .parse::<i64>()
                .map_err(|_| "probe indexed hidden layer is invalid".to_owned())?;
            if index.to_string() != resource.layer {
                return Err("probe indexed hidden layer is not canonical".to_owned());
            }
        }
    }
    if probe.hidden_resources.windows(2).any(|pair| {
        (pair[0].layer.as_str(), &pair[0].layer_kind, &pair[0].site)
            >= (pair[1].layer.as_str(), &pair[1].layer_kind, &pair[1].site)
    }) {
        return Err("probe hidden resources must be sorted unique".to_owned());
    }
    require_sorted_unique_text(&probe.capabilities, "probe.capabilities")
}

fn request_key_fields_match(
    semantic_fields: &[SemanticRequestKeyField],
    attachment_fields: &[RequestKeyField],
) -> bool {
    semantic_fields.len() == attachment_fields.len()
        && semantic_fields
            .iter()
            .zip(attachment_fields)
            .all(|(semantic, attachment)| {
                matches!(
                    (semantic, attachment),
                    (
                        SemanticRequestKeyField::RequestId,
                        RequestKeyField::RequestId
                    ) | (
                        SemanticRequestKeyField::PromptIdx,
                        RequestKeyField::PromptIdx
                    ) | (
                        SemanticRequestKeyField::SampleIdx,
                        RequestKeyField::SampleIdx
                    )
                )
            })
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

fn attachment_hidden_key(resource: &HiddenBinding) -> Result<(i64, u8, u8), String> {
    if resource.layer_kind != HiddenLayerKind::Index {
        return Err("attachment hidden resources must use indexed graph locators".to_owned());
    }
    let layer = resource
        .layer
        .parse::<i64>()
        .map_err(|_| "attachment hidden resource layer is invalid".to_owned())?;
    let site = match resource.site {
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
    graph: &SemanticGraph,
) -> Result<(), String> {
    let state = graph
        .states
        .binary_search_by(|state| state.resource.cmp(&request.state_resource))
        .map(|index| &graph.states[index])
        .map_err(|_| "attachment state resource is absent from the semantic graph".to_owned())?;
    if state.scope != StateScope::Request
        || !request_key_fields_match(&state.key_fields, &request.request_key_fields)
    {
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
    let mut signed_hidden = request
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

fn validate_cross_binding(
    request: &BackendAttachmentRequest,
    probe: &BackendPlacementProbe,
    graph: &SemanticGraph,
    timeline: &Timeline,
    schedule: &PhysicalPlan,
    state: &StatePhysicalPlan,
) -> Result<(), String> {
    validate_request(request)?;
    validate_probe(probe)?;
    require_digest(&schedule.schedule_signature, "schedule.schedule_signature")?;
    require_digest(&state.request_digest, "state.request_digest")?;
    require_digest(
        &state.representation_signature,
        "state.representation_signature",
    )?;
    if state.representations.is_empty()
        || state
            .representations
            .windows(2)
            .any(|pair| pair[0].resource >= pair[1].resource)
    {
        return Err("state providers must be sorted unique resources".to_owned());
    }
    for provider in &state.representations {
        require_text(&provider.resource, "state.providers.resource")?;
        if provider.device_kind == StateDeviceKind::Cpu && provider.device_ordinal != 0 {
            return Err("CPU state provider device ordinal must be zero".to_owned());
        }
    }
    if request.plan_epoch != schedule.plan_epoch
        || request.plan_epoch != state.plan_epoch
        || request.schedule_signature != schedule.schedule_signature
        || request.state_request_digest != state.request_digest
        || request.state_representation_signature != state.representation_signature
    {
        return Err("attachment does not match canonical schedule/state plans".to_owned());
    }
    if request.adapter_id != probe.adapter_id
        || request.backend_abi_digest != probe.backend_abi_digest
        || request.runner_fingerprint != probe.runner_fingerprint
        || request.model_fingerprint != probe.model_fingerprint
        || request.topology != probe.topology
        || request.graph_mode != probe.graph_mode
        || request.device_kind != probe.device_kind
        || request.device_ordinal != probe.device_ordinal
        || request.tensor_parallel_size != probe.tensor_parallel_size
        || request.pipeline_parallel_size != probe.pipeline_parallel_size
        || request.tensor_dtype != probe.tensor_dtype
        || request.hidden_size != probe.hidden_size
        || request.max_packed_rows != probe.max_packed_rows
        || request.hidden_resources != probe.hidden_resources
        || request.capabilities != probe.capabilities
    {
        return Err("attachment differs from backend placement probe".to_owned());
    }
    match request.attachment_kind {
        AttachmentKind::RequestSamplerTraining => {
            validate_request_sampler_training_graph(request, graph)?;
        }
    }
    let provider = state
        .representations
        .binary_search_by(|provider| provider.resource.cmp(&request.state_resource))
        .map(|index| &state.representations[index])
        .map_err(|_| "attachment state resource is not physically selected".to_owned())?;
    if provider.device_kind != request.device_kind
        || provider.device_ordinal != request.device_ordinal
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
    let compute = &request.task_bindings[0];
    let transition = &request.task_bindings[1];
    if compute.anchor_id >= request.sampler_anchor_id
        || transition.anchor_id <= request.sampler_anchor_id
    {
        return Err("attachment task anchors violate sampler timing".to_owned());
    }
    Ok(())
}

pub(crate) fn validate_inputs(inputs: &AttachmentCompilerInputs) -> Result<(), String> {
    validate_schedule_physical(
        &inputs.graph,
        &inputs.timeline,
        &inputs.schedule_decision,
        &inputs.schedule_physical,
    )?;
    validate_state_physical(
        &inputs.state_request,
        &inputs.state_decision,
        &inputs.state_physical,
    )?;
    validate_cross_binding(
        &inputs.request,
        &inputs.probe,
        &inputs.graph,
        &inputs.timeline,
        &inputs.schedule_physical,
        &inputs.state_physical,
    )
}

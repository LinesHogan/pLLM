//! Semantic, plan, and backend-fact validation for attachment signing.

use super::super::scheduler::PhysicalPlan;
use super::super::semantic_graph::{
    HiddenPhase as SemanticHiddenPhase, HiddenSite as SemanticHiddenSite, InputSource, OutputKind,
    LayerReplay, SemanticGraph, SemanticTask, StateEffectKind, StateScope, Timeline,
};
use super::super::state_representation::{StateDeviceKind, StatePhysicalPlan};
use super::types::{
    ActivationPatchOperator, AttachmentCompilerInputs, AttachmentHiddenResource, AttachmentKind, AttachmentTaskBinding,
    AttachmentTaskRole, BackendAttachmentRequest, BackendPlacementProbe, HiddenLayerKind,
    HiddenPhase, HiddenSite, OverlayOperator, SamplerStage,
    StreamDependency, SCHEMA_VERSION,
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
    if !matches!(request.attachment_kind, AttachmentKind::ModelReplay | AttachmentKind::Observation) {
        require_text(&request.state_resource, "state_resource")?;
    }
    if request.attachment_kind != AttachmentKind::SamplerTraining
        && request.sampler_stage != SamplerStage::PackedPostFilterExact
    {
        return Err("sampler intervention requires a sampler attachment".into());
    }
    if request.attachment_kind == AttachmentKind::SamplerTraining
        && probe.hidden_resources.is_empty()
    {
        return Err("attachment requires hidden resources".to_owned());
    }
    require_text(&request.sampler_anchor_symbol, "sampler_anchor_symbol")?;
    match request.attachment_kind {
        AttachmentKind::ModelReplay | AttachmentKind::Observation => {
            if !request.state_resource.is_empty() || request.task_bindings.len() != 1
                || request.task_bindings[0].role != AttachmentTaskRole::Compute
            {
                return Err("stateless attachment requires one compute task".into());
            }
        }
        AttachmentKind::SamplerTraining => {
            if request.task_bindings.len() != 2
                || request.task_bindings[0].role != AttachmentTaskRole::Compute
                || request.task_bindings[1].role != AttachmentTaskRole::Transition
                || request.task_bindings[0].task_id == request.task_bindings[1].task_id
            {
                return Err("sampler attachment tasks must be compute then transition".to_owned());
            }
        }
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            if request.task_bindings.len() != 2
                || request.task_bindings[0].role != AttachmentTaskRole::Transition
                || request.task_bindings[1].role != AttachmentTaskRole::Publish
                || request.task_bindings[0].task_id == request.task_bindings[1].task_id
            {
                return Err(
                    "parameter overlay attachment tasks must be transition then publish".to_owned(),
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
        AttachmentKind::Observation => {
            if probe.hidden_resources.is_empty() || !request.overlay_bindings.is_empty()
                || !request.activation_bindings.is_empty()
                || request.stream_dependencies != vec![StreamDependency::HiddenProducerToCompute, StreamDependency::TransitionToRelease]
            {
                return Err("observation requires only signed hidden inputs and release ordering".into());
            }
        }
        AttachmentKind::ModelReplay => {
            if !probe.hidden_resources.is_empty() || !request.overlay_bindings.is_empty()
                || !request.activation_bindings.is_empty()
                || request.stream_dependencies != vec![StreamDependency::TransitionToRelease]
                || !probe.capabilities.iter().any(|value| value == "model/tokenwise_replay@1")
            {
                return Err("model replay requires the signed tokenwise primitive only".into());
            }
        }
        AttachmentKind::SamplerTraining => {
            if !request.overlay_bindings.is_empty() || !request.activation_bindings.is_empty() {
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
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            if !probe.hidden_resources.is_empty() {
                return Err(
                    "hidden/overlay composition is not supported by this attachment".to_owned(),
                );
            }
            if request.stream_dependencies != vec![StreamDependency::TransitionToRelease] {
                return Err("parameter overlay stream dependencies differ from execution".to_owned());
            }
            if request.attachment_kind == AttachmentKind::ActivationPatch {
                let bindings = &request.activation_bindings;
                if !request.overlay_bindings.is_empty()
                    || bindings.is_empty()
                    || bindings
                        .windows(2)
                        .any(|pair| pair[0].layer >= pair[1].layer)
                    || bindings.iter().any(|binding| {
                        match (&binding.operator, binding.fields.as_slice()) {
                            (ActivationPatchOperator::LowRankAffine, [a, b, c]) => a == b || a == c || b == c,
                            (ActivationPatchOperator::Add, [_]) => false,
                            _ => true,
                        }
                    })
                {
                    return Err(
                        "activation patch requires unique layers and exact operator fields"
                            .to_owned(),
                    );
                }
                return Ok(());
            }
            if !request.activation_bindings.is_empty() {
                return Err("parameter/activation composition is not supported".to_owned());
            }
            let bindings = &request.overlay_bindings;
            if bindings.is_empty()
                || bindings
                    .windows(2)
                    .any(|pair| pair[0].target >= pair[1].target)
                || bindings.iter().any(|binding| {
                    !binding.scale.is_finite()
                        || match binding.operator {
                            OverlayOperator::DenseAdd => binding.fields.len() != 1,
                            OverlayOperator::LowRankAdd => {
                                binding.fields.len() != 2 || binding.fields[0] == binding.fields[1]
                            }
                        }
                })
            {
                return Err(
                    "parameter overlay requires unique targets and exact operator arity".to_owned(),
                );
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
        SemanticHiddenSite::ModelOutput => 2,
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
        HiddenSite::ModelOutput => 2,
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
    if state.scope != StateScope::Request
        && request.sampler_stage == SamplerStage::PackedPostFilterExact
    {
        return Err("attachment request state differs from semantic graph state".to_owned());
    }

    let compute = task_for_dense_id(graph, &request.task_bindings[0])?;
    if !compute.effects.iter().any(|effect| {
        effect.resource == request.state_resource
            && effect.kind == StateEffectKind::Read
            && effect.version.is_none()
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
    let mut task_hidden = graph.tasks.iter()
        .flat_map(|task| task.inputs.iter())
        .filter_map(|input| semantic_hidden_key(&input.source))
        .collect::<Vec<_>>();
    let mut signed_hidden = probe
        .hidden_resources
        .iter()
        .map(attachment_hidden_key)
        .collect::<Result<Vec<_>, _>>()?;
    task_hidden.sort_unstable();
    task_hidden.dedup();
    signed_hidden.sort_unstable();
    if task_hidden != signed_hidden {
        return Err(
            "attachment compute hidden inputs differ from signed hidden resources".to_owned(),
        );
    }

    let transition = task_for_dense_id(graph, &request.task_bindings[1])?;
    if !transition.effects.iter().any(|effect| {
        effect.resource == request.state_resource
            && (effect.kind == StateEffectKind::Update
                || (effect.kind == StateEffectKind::UpdateMaybe
                    && request.sampler_stage == SamplerStage::PackedPreFilterAdd))
            && effect.version.is_none()
    }) || !has_decode_selection(transition, &request.state_resource)
    {
        return Err(
            "attachment transition task is missing request-state transition semantics".to_owned(),
        );
    }
    Ok(())
}

fn validate_parameter_overlay_graph(
    request: &BackendAttachmentRequest,
    graph: &SemanticGraph,
) -> Result<(), String> {
    if graph.tasks.iter().any(|task| {
        task.outputs
            .iter()
            .any(|output| output.kind == OutputKind::Sampler)
            || task
                .inputs
                .iter()
                .any(|input| matches!(input.source, InputSource::Hidden { .. }))
    }) {
        return Err(
            "hidden/sampler/overlay composition is not supported by this attachment".to_owned(),
        );
    }
    graph
        .states
        .binary_search_by(|state| state.resource.cmp(&request.state_resource))
        .map_err(|_| "parameter overlay state resource is absent from the semantic graph".to_owned())?;
    let transition = task_for_dense_id(graph, &request.task_bindings[0])?;
    if !transition.effects.iter().any(|effect| {
        effect.resource == request.state_resource
            && effect.version.is_none()
            && matches!(
                effect.kind,
                StateEffectKind::Transition | StateEffectKind::TransitionMaybe
            )
    }) {
        return Err("parameter overlay transition task is missing transition semantics".to_owned());
    }
    let publish = task_for_dense_id(graph, &request.task_bindings[1])?;
    if !publish.effects.iter().any(|effect| {
        effect.resource == request.state_resource
            && effect.kind == StateEffectKind::Publish
            && effect.version.as_ref() == Some(&transition.symbol)
    }) {
        return Err("parameter overlay publish task is missing publish semantics".to_owned());
    }
    if request.task_bindings[0].anchor_id > request.task_bindings[1].anchor_id {
        return Err("parameter overlay task anchors are out of order".to_owned());
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
    let replay_tasks = graph
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, task)| task.replay.is_some())
        .collect::<Vec<_>>();
    if replay_tasks.len() > 1 {
        return Err("backend supports one conditional replay task".to_owned());
    }
    for (task_id, task) in replay_tasks {
        let Some(LayerReplay::Conditional(replay)) = task.replay.as_ref() else {
            if request.attachment_kind != AttachmentKind::ModelReplay {
                return Err("tokenwise replay requires a model replay attachment".into());
            }
            continue;
        };
        let scheduled = schedule
            .tasks
            .iter()
            .find(|item| item.task_id as usize == task_id)
            .ok_or_else(|| "replay task is absent from the physical plan".to_owned())?;
        if request.attachment_kind != AttachmentKind::ParameterOverlay
            || !task
                .effects
                .iter()
                .any(|effect| effect.resource == request.state_resource)
            || !request.overlay_bindings.iter().any(|binding| {
                binding.target.layer == replay.layer
            })
            || task.timing.after != "state.publish"
            || task.timing.before != "sampler.pre_draw"
            || scheduled.anchor_id < request.task_bindings[1].anchor_id
            || scheduled.anchor_id >= request.sampler_anchor_id
        {
            return Err("replay requires a published parameter overlay before sampling".to_owned());
        }
    }
    match request.attachment_kind {
        AttachmentKind::Observation => {
            let task = task_for_dense_id(graph, &request.task_bindings[0])?;
            let mut hidden = task.inputs.iter().filter_map(|input| semantic_hidden_key(&input.source)).collect::<Vec<_>>();
            let mut signed = probe.hidden_resources.iter().map(attachment_hidden_key).collect::<Result<Vec<_>, _>>()?;
            hidden.sort_unstable();
            signed.sort_unstable();
            if graph.tasks.len() != 1 || !graph.states.is_empty() || !state.representations.is_empty()
                || task.replay.is_some() || !task.effects.is_empty() || hidden != signed
                || task.outputs.iter().any(|output| output.kind != OutputKind::Host)
                || task.timing.after != "state.publish" || task.timing.before != "sampler.pre_draw"
                || task.inputs.iter().any(|input| !matches!(input.source, InputSource::Hidden { .. } | InputSource::StepId))
            {
                return Err("observation must read its original hidden inputs without changing model state".into());
            }
        }
        AttachmentKind::ModelReplay => {
            let task = task_for_dense_id(graph, &request.task_bindings[0])?;
            if graph.tasks.len() != 1 || !graph.states.is_empty() || !state.representations.is_empty()
                || !matches!(task.replay, Some(LayerReplay::Tokenwise(_)))
                || task.timing.after != "model.execute" || task.timing.before != "state.publish"
            {
                return Err("model replay must execute its original stateless effect during forward".into());
            }
        }
        AttachmentKind::SamplerTraining => {
            validate_request_sampler_training_graph(request, probe, graph)?;
        }
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            validate_parameter_overlay_graph(request, graph)?;
        }
    }
    if !matches!(request.attachment_kind, AttachmentKind::ModelReplay | AttachmentKind::Observation) {
    let provider = state
        .representations
        .binary_search_by(|provider| provider.resource.cmp(&request.state_resource))
        .map(|index| &state.representations[index])
        .map_err(|_| "attachment state resource is not physically selected".to_owned())?;
    if provider.device_kind != probe.device_kind || provider.device_ordinal != probe.device_ordinal
    {
        return Err("attachment placement differs from selected state provider".to_owned());
    }
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
    match request.attachment_kind {
        AttachmentKind::Observation => {
            if !timeline.anchors.iter().any(|anchor| anchor.symbol == "state.publish"
                && anchor.ordinal == request.task_bindings[0].anchor_id as u64)
                || request.task_bindings[0].anchor_id >= request.sampler_anchor_id
            {
                return Err("observation must follow model execution and precede sampling".into());
            }
        }
        AttachmentKind::ModelReplay => {
            if !timeline.anchors.iter().any(|anchor| anchor.symbol == "model.execute"
                && anchor.ordinal == request.task_bindings[0].anchor_id as u64)
                || request.task_bindings[0].anchor_id >= request.sampler_anchor_id
            {
                return Err("model replay task must bind the model forward anchor".into());
            }
        }
        AttachmentKind::SamplerTraining => {
            let compute = &request.task_bindings[0];
            let transition = &request.task_bindings[1];
            if compute.anchor_id >= request.sampler_anchor_id
                || transition.anchor_id <= request.sampler_anchor_id
            {
                return Err("attachment task anchors violate sampler timing".to_owned());
            }
            let mut past_update = false;
            for scheduled in &schedule.tasks {
                let task = &graph.tasks[scheduled.task_id as usize];
                if task.inputs.iter().any(|input| matches!(input.source, InputSource::SamplerFeedback))
                    && (request.sampler_stage != SamplerStage::PackedPreFilterAdd
                        || scheduled.anchor_id <= request.sampler_anchor_id
                        || past_update
                        || (scheduled.task_id != transition.task_id
                            && (task.effects.iter().any(|effect| effect.kind != StateEffectKind::Read)
                                || task.outputs.iter().any(|output| output.kind != OutputKind::Temporary))))
                {
                    return Err("sampler feedback requires a post-sample read or its signed transition".into());
                }
                past_update |= scheduled.task_id == transition.task_id;
            }
        }
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            let transition = &request.task_bindings[0];
            let publish = &request.task_bindings[1];
            let anchor_id = |symbol: &str| {
                timeline
                    .anchors
                    .iter()
                    .find(|anchor| anchor.symbol == symbol)
                    .map(|anchor| anchor.ordinal as u32)
            };
            let request_state = graph
                .states
                .iter()
                .find(|state| state.resource == request.state_resource)
                .is_some_and(|state| state.scope == StateScope::Request);
            let before_model = Some(transition.anchor_id) == anchor_id("request.prepare")
                && Some(publish.anchor_id) == anchor_id("request.prepare");
            let supported = if request.attachment_kind == AttachmentKind::ActivationPatch {
                before_model
            } else if request_state {
                Some(transition.anchor_id) == anchor_id("model.execute")
                    && Some(publish.anchor_id) == anchor_id("state.publish")
            } else {
                before_model || (Some(transition.anchor_id) == anchor_id("token.commit")
                        && Some(publish.anchor_id) == anchor_id("token.commit"))
            };
            if !supported {
                return Err(
                    "parameter overlay tasks use unsupported backend event timing".to_owned(),
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_inputs(inputs: &AttachmentCompilerInputs) -> Result<(), String> {
    let plan = &inputs.plan;
    plan.validate()?;
    if !matches!(inputs.request.attachment_kind, AttachmentKind::ModelReplay | AttachmentKind::Observation) {
    let layout = plan
        .tensor_layout_request
        .layouts
        .iter()
        .find(|layout| layout.resource == inputs.request.state_resource)
        .ok_or_else(|| "attachment state has no tensor layout".to_owned())?;
    if inputs
        .request
        .overlay_bindings
        .iter()
        .flat_map(|binding| &binding.fields)
        .chain(
            inputs
                .request
                .activation_bindings
                .iter()
                .flat_map(|binding| &binding.fields),
        )
        .any(|field| {
            !layout
                .layout
                .fields
                .iter()
                .any(|known| known.field_id == u64::from(*field))
        })
    {
        return Err("overlay field is absent from the signed tensor layout".to_owned());
    }
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

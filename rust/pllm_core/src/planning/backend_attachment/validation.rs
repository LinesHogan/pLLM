//! Semantic, plan, and backend-fact validation for attachment signing.

use super::super::scheduler::PhysicalPlan;
use super::super::semantic_graph::{
    HiddenPhase as SemanticHiddenPhase, HiddenSite as SemanticHiddenSite, InputSource, OutputKind,
    SemanticGraph, SemanticTask, StateEffectKind, StateScope, Timeline,
};
use super::super::state_representation::{StateDeviceKind, StatePhysicalPlan};
use super::types::{
    ActivationPatchOperator, AttachmentCompilerInputs, AttachmentGraphMode, AttachmentHiddenResource, AttachmentKind, AttachmentTaskBinding,
    AttachmentTaskRole, BackendAttachmentRequest, BackendPlacementProbe, HiddenLayerKind,
    DecodeRefresh, DecodeWindow, HiddenPhase, HiddenSite, MissingPolicy, OverlayOperator, SamplerStage,
    SCHEMA_VERSION,
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
    if request.schema_version != SCHEMA_VERSION || request.packed_row_schema_version != 2 {
        return Err("unsupported backend attachment schema".to_owned());
    }
    let composed_replay = request.attachment_kind == AttachmentKind::ModelReplay
        && !request.activation_bindings.is_empty();
    if composed_replay || !matches!(request.attachment_kind, AttachmentKind::ModelReplay | AttachmentKind::Observation) {
        require_text(&request.state_resource, "state_resource")?;
    }
    let guides_sampler = request.attachment_kind == AttachmentKind::RequestTraining
        && request.task_bindings.first().is_some_and(|binding| binding.role == AttachmentTaskRole::Compute);
    if !guides_sampler && request.sampler_stage != SamplerStage::PackedPostFilterExact {
        return Err("sampler_stage needs a guide_sampling or add_logits operation".into());
    }
    if request.attachment_kind == AttachmentKind::RequestTraining
        && probe.hidden_resources.is_empty()
    {
        return Err("attachment requires hidden resources".to_owned());
    }
    require_text(&request.sampler_anchor_symbol, "sampler_anchor_symbol")?;
    match request.attachment_kind {
        AttachmentKind::ModelReplay if composed_replay => {
            if request.task_bindings.iter().map(|binding| &binding.role).collect::<Vec<_>>()
                != vec![&AttachmentTaskRole::Compute, &AttachmentTaskRole::Transition, &AttachmentTaskRole::Publish]
                && request.task_bindings.iter().map(|binding| &binding.role).collect::<Vec<_>>() != vec![&AttachmentTaskRole::Compute]
                || request.task_bindings.iter().map(|binding| binding.task_id)
                    .collect::<std::collections::BTreeSet<_>>().len() != request.task_bindings.len()
            {
                return Err("activation replay requires distinct compute, transition and publish roles".into());
            }
        }
        AttachmentKind::ModelReplay | AttachmentKind::Observation => {
            if !request.state_resource.is_empty() || request.task_bindings.len() != 1
                || request.task_bindings[0].role != AttachmentTaskRole::Compute
            {
                return Err("stateless attachment requires one compute task".into());
            }
        }
        AttachmentKind::RequestTraining => {
            // A compute task that guides the sampler is optional; training alone binds only the transition.
            let roles = request.task_bindings.iter().map(|binding| &binding.role).collect::<Vec<_>>();
            if !matches!(
                roles.as_slice(),
                [AttachmentTaskRole::Transition] | [AttachmentTaskRole::Compute, AttachmentTaskRole::Transition]
            ) || (roles.len() == 2 && request.task_bindings[0].task_id == request.task_bindings[1].task_id)
            {
                return Err("request training binds its transition task, after an optional sampler compute task".to_owned());
            }
        }
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            if !request.task_bindings.is_empty() && (request.task_bindings.len() != 2
                || request.task_bindings[0].role != AttachmentTaskRole::Transition
                || request.task_bindings[1].role != AttachmentTaskRole::Publish
                || request.task_bindings[0].task_id == request.task_bindings[1].task_id) {
                return Err("inline model effects require their transition and publication".into());
            }
        }
    }
    for binding in &request.task_bindings {
        require_text(
            &binding.implementation_id,
            "task_bindings.implementation_id",
        )?;
    }
    for binding in &request.activation_bindings {
        if binding.decode == DecodeWindow::None
            && (binding.decode_refresh != DecodeRefresh::Inherit
                || binding.on_missing != MissingPolicy::Stop)
        {
            return Err("decode window is off; refresh and missing policies require decode rows".into());
        }
    }
    if !request.activation_bindings.is_empty() {
        if request.activation_bindings.windows(2).any(|pair| pair[0].layer >= pair[1].layer)
            || request.activation_bindings.iter().any(|binding| match (&binding.operator, binding.fields.as_slice()) {
                (ActivationPatchOperator::LowRankAffine, [a, b, c]) => a == b || a == c || b == c,
                (ActivationPatchOperator::Add, [_]) => false,
                _ => true,
            })
        {
            return Err("activation patch requires unique layers and exact operator fields".into());
        }
    }
    match request.attachment_kind {
        AttachmentKind::Observation => {
            if probe.hidden_resources.is_empty() || !request.overlay_bindings.is_empty()
                || !request.activation_bindings.is_empty()
            {
                return Err("observation requires only signed hidden inputs".into());
            }
        }
        AttachmentKind::ModelReplay => {
            if !probe.hidden_resources.is_empty() || !request.overlay_bindings.is_empty()
                || !probe.capabilities.iter().any(|value| value == "model/tokenwise_replay@1")
            {
                return Err("model replay requires the signed tokenwise primitive only".into());
            }
        }
        AttachmentKind::RequestTraining => {
            if !request.overlay_bindings.is_empty() || !request.activation_bindings.is_empty() {
                return Err("request training cannot bind overlay or activation fields".to_owned());
            }
        }
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            if request.attachment_kind == AttachmentKind::ActivationPatch {
                if !request.overlay_bindings.is_empty()
                    || request.activation_bindings.is_empty()
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
                            OverlayOperator::DenseAdd | OverlayOperator::HeadAdd => binding.fields.len() != 1,
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
    if probe.schema_version != 3 {
        return Err("unsupported backend placement probe schema".to_owned());
    }
    require_text(&probe.adapter_id, "probe.adapter_id")?;
    require_digest(&probe.backend_abi_digest, "probe.backend_abi_digest")?;
    require_digest(&probe.runner_fingerprint, "probe.runner_fingerprint")?;
    require_digest(&probe.model_fingerprint, "probe.model_fingerprint")?;
    if probe.device_kind == StateDeviceKind::Cpu && probe.device_ordinal != 0 {
        return Err("CPU backend probe device ordinal must be zero".to_owned());
    }
    if probe.capabilities.iter().any(|capability| capability == "execution/region_streams@1")
        && (probe.device_kind != StateDeviceKind::Cuda || probe.graph_mode != AttachmentGraphMode::Eager)
    {
        return Err("region stream capability requires the original eager CUDA backend".into());
    }
    if probe.tensor_parallel_size != 1
        || probe.pipeline_parallel_size != 1
        || probe.hidden_size == 0
        || probe.max_packed_rows == 0
        || probe.causal_domain.as_ref().is_some_and(|domain|
            [domain.requests, domain.layers, domain.positions].iter().any(|&size| size == 0 || size > i64::MAX as u64))
    {
        return Err("backend placement probe topology is unsupported".to_owned());
    }
    if probe.targets.windows(2).any(|pair| pair[0].target >= pair[1].target) || probe.targets.iter().any(|target| target.input_size == 0 || target.output_size == 0) { return Err("target dimensions must be positive and ordered uniquely".into()); }
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
            matches!(pair[0].phase, HiddenPhase::Decode),
        ) >= (
            pair[1].binding.layer.as_str(),
            &pair[1].binding.layer_kind,
            &pair[1].binding.site,
            matches!(pair[1].phase, HiddenPhase::Decode),
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

fn validate_request_training_graph(
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

    if let [compute, _] = request.task_bindings.as_slice() {
        validate_sampler_compute(request, task_for_dense_id(graph, compute)?)?;
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
            "attachment hidden inputs differ from the backend's hidden resources".to_owned(),
        );
    }

    let transition = task_for_dense_id(graph, request.task_bindings.last().unwrap())?;
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

/// The prediction task that feeds the sampler reads request state over packed sampler rows.
fn validate_sampler_compute(request: &BackendAttachmentRequest, compute: &SemanticTask) -> Result<(), String> {
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
    Ok(())
}

fn validate_parameter_overlay_graph(
    request: &BackendAttachmentRequest,
    graph: &SemanticGraph,
    bindings: &[AttachmentTaskBinding],
) -> Result<(), String> {
    if graph.tasks.iter().any(|task| task.outputs.iter().any(|output| output.kind == OutputKind::Sampler)) {
        return Err("model state consumers cannot also guide the sampler".into());
    }
    for state in &graph.states {
        if graph.tasks.iter().flat_map(|task| &task.effects).filter(|effect| effect.resource == state.resource && matches!(effect.kind, StateEffectKind::Update | StateEffectKind::UpdateMaybe | StateEffectKind::Transition | StateEffectKind::TransitionMaybe)).count() > 1 {
            return Err("declare one mutation producer per state resource".into());
        }
    }
    graph
        .states
        .binary_search_by(|state| state.resource.cmp(&request.state_resource))
        .map_err(|_| "parameter overlay state resource is absent from the semantic graph".to_owned())?;
    if bindings.is_empty() { return Ok(()); }
    let transition = task_for_dense_id(graph, &bindings[0])?;
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
    let publish = task_for_dense_id(graph, &bindings[1])?;
    if !publish.effects.iter().any(|effect| {
        effect.resource == request.state_resource
            && effect.kind == StateEffectKind::Publish
            && effect.version.as_ref() == Some(&transition.symbol)
    }) {
        return Err("parameter overlay publish task is missing publish semantics".to_owned());
    }
    if bindings[0].anchor_id > bindings[1].anchor_id {
        return Err("parameter overlay task anchors are out of order".to_owned());
    }
    Ok(())
}

fn validate_activation_replay(
    request: &BackendAttachmentRequest,
    probe: &BackendPlacementProbe,
    graph: &SemanticGraph,
) -> Result<(), String> {
    validate_parameter_overlay_graph(request, graph, &request.task_bindings[1..])?;
    let forward = task_for_dense_id(graph, &request.task_bindings[0])?;
    if request.task_bindings.len() == 1 {
        if graph.tasks.len() != 1 || forward.effects.len() != 1 || forward.effects[0].kind != StateEffectKind::Read
            || forward.effects[0].version.is_some() || forward.effects[0].resource != request.state_resource
            || !graph.input_groups.is_empty() || request.activation_bindings.iter().any(|binding| probe.causal_domain.as_ref().is_none_or(|domain| u64::from(binding.layer) >= domain.layers)) {
            return Err("initialized activation replay needs its entry READ and signed layers".into());
        }
        return Ok(());
    }
    let transition = task_for_dense_id(graph, &request.task_bindings[1])?;
    let publication = task_for_dense_id(graph, &request.task_bindings[2])?;
    let depends_on = |consumer: &SemanticTask, producer: &SemanticTask| {
        consumer.inputs.iter().any(|input| matches!(&input.source,
            InputSource::Result { task, output } if task == &producer.symbol
                && producer.outputs.iter().any(|value| &value.name == output)))
    };
    if forward.effects.len() != 1 || forward.effects.iter().any(|effect|
        effect.kind != StateEffectKind::Read || effect.resource != request.state_resource
            || effect.version.as_ref() != Some(&publication.symbol))
        || !depends_on(forward, publication) || !depends_on(publication, transition)
        || transition.effects.len() != 1 || publication.effects.len() != 1
        || !forward.inputs.iter().any(|input| matches!(input.source, InputSource::PackedRows))
        || forward.inputs.iter().any(|input| !matches!(&input.source,
            InputSource::PackedRows | InputSource::Result { .. }))
        || forward.inputs.iter().any(|input| matches!(&input.source,
            InputSource::Result { task, .. } if task != &publication.symbol))
    {
        return Err("activation replay requires an exact READ of its signed inject publication".into());
    }
    if graph.tasks.iter().filter(|task| task.symbol != transition.symbol && task.symbol != publication.symbol)
        .any(|task| task.effects.iter().any(|effect| effect.kind != StateEffectKind::Read))
        || graph.states.iter().any(|state| !graph.tasks.iter().any(|task|
            task.effects.iter().any(|effect| effect.resource == state.resource)))
    {
        return Err("activation replay cannot hide unrelated state or transitions".into());
    }
    let domain = probe.causal_domain.as_ref().ok_or("activation replay requires its signed model domain")?;
    if !matches!(&forward.replay, Some(effect) if u64::from(effect.layer_count) == domain.layers)
        || request.activation_bindings.iter().any(|binding| u64::from(binding.layer) >= domain.layers)
    {
        return Err("activation replay layers differ from the signed model domain".into());
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
    for group in &graph.input_groups {
        let readers: Vec<_> = if request.attachment_kind == AttachmentKind::ModelReplay {
            vec![task_for_dense_id(graph, &request.task_bindings[0])?.symbol.clone()]
        } else { vec![] };
        if request.activation_bindings.is_empty() || group.resource != request.state_resource
            || group.allow_absent != request.activation_bindings.iter().all(|binding| binding.on_missing == MissingPolicy::Hold)
            || group.readers != readers
            || !matches!(request.attachment_kind, AttachmentKind::ActivationPatch | AttachmentKind::ModelReplay)
            || request.task_bindings.iter().filter(|binding| binding.role != AttachmentTaskRole::Compute)
                .any(|binding| task_for_dense_id(graph, binding).map_or(true, |task| !group.tasks.contains(&task.symbol)))
            || task_for_dense_id(graph, request.task_bindings.last().unwrap())?.symbol != group.publication
        {
            return Err("conditional activation requires its exact inject producer group and pinned READ".into());
        }
    }
    let replay_tasks = graph
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, task)| task.replay.is_some())
        .collect::<Vec<_>>();
    if replay_tasks.len() > 1 {
        return Err("backend supports one layer route".to_owned());
    }
    for (_, task) in replay_tasks {
        validate_position_source(task, graph, probe)?;
        let domain = probe.causal_domain.as_ref().ok_or("layer routes need a signed model domain")?;
        if u64::from(task.replay.as_ref().unwrap().layer_count) != domain.layers
            || task.timing.after != "model.execute" || task.timing.before != "state.publish"
            || task.replay.as_ref().unwrap().digest.len() != 64
            || !task.replay.as_ref().unwrap().digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || (task.replay.as_ref().unwrap().apply_updates && request.attachment_kind != AttachmentKind::ParameterOverlay)
        {
            return Err("layer routes need their model domain, source and forward timing".into());
        }
        if request.attachment_kind == AttachmentKind::ParameterOverlay {
            let publish = task_for_dense_id(graph, &request.task_bindings[1])?;
            if graph.tasks.len() != 2 || graph.states.len() != 1
                || graph.states[0].scope != StateScope::Request
                || publish.effects.len() != 1
                || !publish.inputs.iter().any(|input| matches!(&input.source,
                    InputSource::Result { task: source, .. } if source == &task.symbol))
                || task_for_dense_id(graph, &request.task_bindings[0])?.symbol != task.symbol
                || task.effects.len() != 1 || task.effects[0].kind != StateEffectKind::TransitionMaybe
                || task.effects[0].resource != request.state_resource
                || task.inputs.iter().any(|input| match &input.source {
                    InputSource::PackedRows => false,
                    InputSource::StateSelection { resource, .. } => resource != &request.state_resource,
                    InputSource::HostService { service, port, schema } => service != "parameter.operands"
                        || port != "dense_pairs" || schema != "canonical.dense_operand_pairs@1",
                    _ => true,
                })
                || !task.inputs.iter().any(|input| matches!(input.source, InputSource::PackedRows))
                || !request.activation_bindings.is_empty()
            {
                return Err("routed overlays need one request transition and its publication".into());
            }
        } else if request.attachment_kind != AttachmentKind::ModelReplay {
            return Err("layer routes need a model replay or parameter overlay attachment".into());
        }
    }
    match request.attachment_kind {
        AttachmentKind::Observation => {
            let task = task_for_dense_id(graph, &request.task_bindings[0])?;
            let mut hidden = task.inputs.iter().filter_map(|input| semantic_hidden_key(&input.source)).collect::<Vec<_>>();
            let mut signed = probe.hidden_resources.iter().map(attachment_hidden_key).collect::<Result<Vec<_>, _>>()?;
            hidden.sort_unstable();
            signed.sort_unstable();
            if graph.tasks.len() != 1
                || task.effects.iter().any(|effect| {
                    effect.kind != StateEffectKind::Read || effect.version.is_some()
                })
                || task.replay.is_some() || hidden != signed
                || task.outputs.iter().any(|output| output.kind != OutputKind::Host)
                || task.timing.after != "state.publish" || task.timing.before != "sampler.pre_draw"
                || task.inputs.iter().any(|input| match &input.source {
                    InputSource::Hidden { .. } | InputSource::StepId => false,
                    InputSource::StateSelection { resource, .. } => !task.effects.iter().any(|effect| {
                        effect.kind == StateEffectKind::Read
                            && effect.version.is_none()
                            && &effect.resource == resource
                    }),
                    _ => true,
                })
            {
                return Err("observation must read its original hidden inputs without changing model state".into());
            }
        }
        AttachmentKind::ModelReplay => {
            let task = task_for_dense_id(graph, &request.task_bindings[0])?;
            if task.replay.is_none()
                || task.timing.after != "model.execute" || task.timing.before != "state.publish"
            {
                return Err("model replay must execute its original tokenwise effect during forward".into());
            }
            if request.activation_bindings.is_empty() {
                if graph.tasks.len() != 1 || !graph.states.is_empty() || !state.representations.is_empty()
                    || !task.effects.is_empty()
                {
                    return Err("model replay must execute its original stateless effect during forward".into());
                }
            } else {
                validate_activation_replay(request, probe, graph)?;
            }
        }
        AttachmentKind::RequestTraining => {
            validate_request_training_graph(request, probe, graph)?;
        }
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            validate_parameter_overlay_graph(request, graph, &request.task_bindings)?;
            let mut hidden: Vec<_> = graph.tasks.iter().flat_map(|task| &task.inputs).filter_map(|input| semantic_hidden_key(&input.source)).collect();
            let mut signed = probe.hidden_resources.iter().map(attachment_hidden_key).collect::<Result<Vec<_>, _>>()?;
            hidden.sort_unstable(); hidden.dedup(); signed.sort_unstable();
            if hidden != signed {
                return Err("model producers must use the probed hidden inputs".into());
            }
        }
    }
    if request.attachment_kind != AttachmentKind::ModelReplay || !request.activation_bindings.is_empty()
    {
        if state.representations.iter().any(|provider| {
            provider.device_kind != probe.device_kind || provider.device_ordinal != probe.device_ordinal
        }) {
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
            if !request.activation_bindings.is_empty()
                && (request.sampler_anchor_symbol != "sampler.pre_draw"
                    || request.task_bindings[1..].iter().any(|binding|
                        !timeline.anchors.iter().any(|anchor| anchor.symbol == "request.prepare"
                            && anchor.ordinal == u64::from(binding.anchor_id))))
            {
                return Err("activation replay must publish at request.prepare before model forward".into());
            }
        }
        AttachmentKind::RequestTraining => {
            let transition = request.task_bindings.last().unwrap();
            if request.task_bindings.len() == 2 && request.task_bindings[0].anchor_id >= request.sampler_anchor_id
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
            if request.task_bindings.is_empty() { return Ok(()); }
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
                before_model || (Some(transition.anchor_id) == anchor_id("state.publish") && Some(publish.anchor_id) == anchor_id("state.publish"))
            } else if request_state {
                // Publication before the request's first forward is KV-safe; mid-request
                // updates continue through the model-execute publication path.
                before_model || (Some(transition.anchor_id) == anchor_id("model.execute")
                    && Some(publish.anchor_id) == anchor_id("state.publish"))
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

fn validate_position_source(task: &SemanticTask, graph: &SemanticGraph, probe: &BackendPlacementProbe) -> Result<(), String> {
    let effect = task.replay.as_ref().unwrap();
    if effect.position_states != probe.position_states
        || (!effect.position_states.is_empty() && (!effect.role_sensitive || !task.effects.is_empty() || !graph.states.is_empty())) {
        return Err("PositionState needs a role-sensitive pure route and its pre-allocated schema".into());
    }

    Ok(())
}

pub(crate) fn validate_inputs(inputs: &AttachmentCompilerInputs) -> Result<(), String> {
    let plan = &inputs.plan;
    plan.validate()?;
    let mut reads: std::collections::BTreeSet<_> = inputs.request.overlay_bindings.iter().map(|b| b.resource.clone()).collect();
    if !inputs.request.activation_bindings.is_empty() { reads.insert(inputs.request.state_resource.clone()); }
    if reads.into_iter().collect::<Vec<_>>() != plan.semantic_graph.model_reads {
        return Err("model consumer resources differ from the semantic graph".into());
    }
    if inputs.probe.graph_mode == AttachmentGraphMode::CudaGraph {
        let request = &inputs.request;
        if inputs.probe.device_kind != StateDeviceKind::Cuda
            || !inputs.probe.capabilities.iter().any(|value| value == "backend/vllm_mrv2@1")
            || request.attachment_kind != AttachmentKind::ParameterOverlay
            || plan.semantic_graph.states.iter().any(|state| state.scope != StateScope::Request)
            || plan.semantic_graph.tasks.iter().any(|task| task.replay.is_some() || task.inputs.iter().any(|input| matches!(input.source, InputSource::Hidden { .. })))
        {
            return Err("Graph overlays need request-owned parameter reads without replay".into());
        }
    }
    if !inputs.request.activation_bindings.is_empty()
        || !matches!(inputs.request.attachment_kind, AttachmentKind::ModelReplay | AttachmentKind::Observation)
    {
    let layout = plan
        .tensor_layout_request
        .layouts
        .iter()
        .find(|layout| layout.resource == inputs.request.state_resource)
        .ok_or_else(|| "attachment state has no tensor layout".to_owned())?;
    for binding in &inputs.request.overlay_bindings {
        let target_layout = plan.tensor_layout_request.layouts.iter().find(|layout| layout.resource == binding.resource)
            .ok_or("projection resource has no signed tensor layout")?;
        if binding.resource != inputs.request.state_resource {
            return Err("projection resource differs from the attachment state".into());
        }
        validate_views(&binding.fields, &target_layout.layout.fields)?;
        let dtype = if binding.operator == OverlayOperator::HeadAdd { super::super::tensor_layout::TensorDType::Float32 } else { inputs.probe.tensor_dtype.clone() };
        if binding.fields.iter().any(|view| target_layout.layout.fields[view.field as usize].dtype != dtype) { return Err("projection dtype differs from signed model".into()); }
        let target = inputs.probe.targets.iter().find(|target| target.target == binding.target)
            .ok_or("projection target has no signed dimensions")?;
        let shapes: Vec<_> = binding.fields.iter().map(|view| view.shape.as_slice()).collect();
        let valid = match (&binding.operator, shapes.as_slice()) {
            (OverlayOperator::DenseAdd, [shape]) => *shape == [target.output_size, target.input_size],
            (OverlayOperator::LowRankAdd, [a, b]) => a.len() == 2 && b.len() == 2 && a[0] > 0 && a[0] == b[1] && a[1] == target.input_size && b[0] == target.output_size,
            (OverlayOperator::HeadAdd, [shape]) => shape.len() == 3 && shape[0] > 0 && shape[1] == shape[2] && shape[0] * shape[1] == target.input_size && target.input_size == target.output_size,
            _ => false,
        };
        if !valid { return Err("projection views differ from signed target dimensions".into()); }
    }
    for binding in &inputs.request.activation_bindings {
        validate_views(&binding.fields, &layout.layout.fields)?;
        let shapes: Vec<_> = binding.fields.iter().map(|view| view.shape.as_slice()).collect();
        let hidden = inputs.probe.hidden_size;
        let valid = match (&binding.operator, shapes.as_slice()) {
            (ActivationPatchOperator::Add, [shape]) => *shape == [hidden],
            (ActivationPatchOperator::LowRankAffine, [a, b, c]) => a.len() == 2 && b.len() == 1 && c.len() == 2 && a[0] > 0 && a[0] <= hidden && a[0] == b[0] && a[0] == c[1] && a[1] == hidden && c[0] == hidden,
            _ => false,
        };
        let dtype = if binding.operator == ActivationPatchOperator::Add { super::super::tensor_layout::TensorDType::Float32 } else { inputs.probe.tensor_dtype.clone() };
        if !valid || binding.fields.iter().any(|view| layout.layout.fields[view.field as usize].dtype != dtype) { return Err("activation views differ from signed operator shape or dtype".into()); }
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

fn validate_views(views: &[super::types::FieldView], fields: &[super::super::tensor_layout::TensorFieldSpec]) -> Result<(), String> {
    for view in views {
        let field = fields.iter().find(|field| field.field_id == u64::from(view.field)).ok_or("view field is absent from layout")?;
        let size = view.shape.iter().try_fold(1_u64, |a, b| a.checked_mul(*b)).ok_or("view size overflow")?;
        if view.offset.checked_add(size).is_none_or(|end| end > field.shape.iter().product()) || size == 0 {
            return Err("view is outside its physical field".into());
        }
    }
    Ok(())
}

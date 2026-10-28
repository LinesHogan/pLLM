//! Check workload inputs against backend facts and attachment execution limits.

use super::super::scheduler::ScheduleDecision;
use super::super::semantic_graph::{
    HiddenPhase as SemanticHiddenPhase, HiddenSite as SemanticHiddenSite, InputSource, OutputKind,
    SemanticGraph, SemanticTask, StateEffectKind, StateScope, Timeline,
};
use super::super::state_representation::{StateDeviceKind, StateRepresentationDecision};
use super::types::{
    ActivationPatchOperator, AttachmentGraphMode, AttachmentKind, AttachmentTaskBinding,
    AttachmentTaskRole, BackendAttachment, BackendPlacementProbe, HiddenLayerKind, HiddenPhase,
    HiddenSite, MissingPolicy, OverlayOperator, SamplerStage,
};

fn validate_declaration(
    attachment: &BackendAttachment,
    probe: &BackendPlacementProbe,
) -> Result<(), String> {
    match attachment.spec.kind {
        AttachmentKind::Observation => {
            if !attachment.spec.state_resource.is_empty()
                || attachment.task_bindings.is_empty()
                || probe.hidden_resources.is_empty()
                || !attachment.spec.overlay_bindings.is_empty()
                || !attachment.spec.activation_bindings.is_empty()
            {
                return Err(
                    "observation requires hidden inputs and no model state bindings".into(),
                );
            }
        }
        AttachmentKind::ModelReplay => {
            if !probe.hidden_resources.is_empty()
                || !attachment.spec.overlay_bindings.is_empty()
                || !probe
                    .capabilities
                    .iter()
                    .any(|value| value == "model/tokenwise_replay@1")
            {
                return Err("model replay requires tokenwise replay support without hidden or overlay bindings".into());
            }
            // Stateless replay execution supports one forward task.
            if attachment.spec.activation_bindings.is_empty()
                && (!attachment.spec.state_resource.is_empty()
                    || attachment.task_bindings.len() != 1)
            {
                return Err("stateless model replay requires one compute task".into());
            }
        }
        AttachmentKind::RequestTraining => {
            if probe.hidden_resources.is_empty() {
                return Err("request training requires hidden resources".into());
            }
            if !attachment.spec.overlay_bindings.is_empty()
                || !attachment.spec.activation_bindings.is_empty()
            {
                return Err("request training cannot bind overlay or activation fields".to_owned());
            }
        }
        AttachmentKind::ActivationPatch => {
            if !attachment.spec.overlay_bindings.is_empty()
                || attachment.spec.activation_bindings.is_empty()
            {
                return Err(
                    "activation patch requires activation fields without parameter overlays".into(),
                );
            }
        }
        AttachmentKind::ParameterOverlay => {
            if !attachment.spec.activation_bindings.is_empty() {
                return Err("parameter/activation composition is not supported".to_owned());
            }
            if attachment.spec.overlay_bindings.is_empty() {
                return Err("parameter overlay requires target bindings".into());
            }
            if attachment
                .spec
                .overlay_bindings
                .iter()
                .any(|binding| !binding.scale.is_finite())
            {
                return Err("parameter overlay scales must be finite".into());
            }
        }
    }
    Ok(())
}

fn hidden_inputs_match(graph: &SemanticGraph, probe: &BackendPlacementProbe) -> Result<(), String> {
    let mut hidden = std::collections::BTreeSet::new();
    for input in graph.tasks.iter().flat_map(|task| &task.inputs) {
        if let InputSource::Hidden { layer, site, phase } = &input.source {
            let site = match site {
                SemanticHiddenSite::Post => 0,
                SemanticHiddenSite::BlockOutput => 1,
                SemanticHiddenSite::ModelOutput => 2,
            };
            hidden.insert((*layer, site, matches!(phase, SemanticHiddenPhase::Decode)));
        }
    }
    let mut probed = std::collections::BTreeSet::new();
    for resource in &probe.hidden_resources {
        if resource.binding.layer_kind != HiddenLayerKind::Index {
            return Err("attachment hidden resources must use indexed graph locators".into());
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
        probed.insert((layer, site, matches!(resource.phase, HiddenPhase::Decode)));
    }
    if hidden != probed {
        return Err("attachment hidden inputs differ from the backend's hidden resources".into());
    }
    Ok(())
}

fn validate_parameter_overlay_graph(
    graph: &SemanticGraph,
    bindings: &[AttachmentTaskBinding],
) -> Result<(), String> {
    if graph.tasks.iter().any(|task| {
        task.outputs
            .iter()
            .any(|output| output.kind == OutputKind::Sampler)
    }) {
        return Err("model state consumers cannot also guide the sampler".into());
    }
    for state in &graph.states {
        if graph
            .tasks
            .iter()
            .flat_map(|task| &task.effects)
            .filter(|effect| {
                effect.resource == state.resource
                    && matches!(
                        effect.kind,
                        StateEffectKind::Update
                            | StateEffectKind::UpdateMaybe
                            | StateEffectKind::Transition
                            | StateEffectKind::TransitionMaybe
                    )
            })
            .count()
            > 1
        {
            return Err("declare one mutation producer per state resource".into());
        }
    }
    if bindings.is_empty() {
        return Ok(());
    }
    if bindings[0].anchor_id > bindings[1].anchor_id {
        return Err("parameter overlay task anchors are out of order".to_owned());
    }
    Ok(())
}

fn validate_activation_replay(
    attachment: &BackendAttachment,
    graph: &SemanticGraph,
) -> Result<(), String> {
    validate_parameter_overlay_graph(graph, &attachment.task_bindings[1..])?;
    let forward = &graph.tasks[attachment.task_bindings[0].task_id as usize];
    if attachment.task_bindings.len() == 1 {
        if graph.tasks.len() != 1
            || forward.effects.len() != 1
            || forward.effects[0].kind != StateEffectKind::Read
            || forward.effects[0].version.is_some()
            || forward.effects[0].resource != attachment.spec.state_resource
            || !graph.input_groups.is_empty()
        {
            return Err(
                "initialized activation replay needs one forward task reading entry state".into(),
            );
        }
        return Ok(());
    }
    let transition = &graph.tasks[attachment.task_bindings[1].task_id as usize];
    let publication = &graph.tasks[attachment.task_bindings[2].task_id as usize];
    let depends_on = |consumer: &SemanticTask, producer: &SemanticTask| {
        consumer.inputs.iter().any(|input| {
            matches!(&input.source,
            InputSource::Result { task, output } if task == &producer.symbol
                && producer.outputs.iter().any(|value| &value.name == output))
        })
    };
    if forward.effects.len() != 1
        || forward.effects.iter().any(|effect| {
            effect.kind != StateEffectKind::Read
                || effect.resource != attachment.spec.state_resource
                || effect.version.as_ref() != Some(&publication.symbol)
        })
        || !depends_on(forward, publication)
        || !depends_on(publication, transition)
        || transition.effects.len() != 1
        || publication.effects.len() != 1
        || !forward
            .inputs
            .iter()
            .any(|input| matches!(input.source, InputSource::PackedRows))
        || forward.inputs.iter().any(|input| {
            !matches!(
                &input.source,
                InputSource::PackedRows | InputSource::Result { .. }
            )
        })
        || forward.inputs.iter().any(|input| {
            matches!(&input.source,
            InputSource::Result { task, .. } if task != &publication.symbol)
        })
    {
        return Err(
            "activation replay must read the state version published by its inject task".into(),
        );
    }
    if graph
        .tasks
        .iter()
        .filter(|task| task.symbol != transition.symbol && task.symbol != publication.symbol)
        .any(|task| {
            task.effects
                .iter()
                .any(|effect| effect.kind != StateEffectKind::Read)
        })
        || graph.states.iter().any(|state| {
            !graph.tasks.iter().any(|task| {
                task.effects
                    .iter()
                    .any(|effect| effect.resource == state.resource)
            })
        })
    {
        return Err("activation replay cannot hide unrelated state or transitions".into());
    }
    Ok(())
}

fn validate_cross_binding(
    attachment: &BackendAttachment,
    probe: &BackendPlacementProbe,
    graph: &SemanticGraph,
    timeline: &Timeline,
    schedule: &ScheduleDecision,
    state: &StateRepresentationDecision,
) -> Result<(), String> {
    validate_declaration(attachment, probe)?;
    hidden_inputs_match(graph, probe)?;
    let attachment_state = graph
        .states
        .iter()
        .find(|state| state.resource == attachment.spec.state_resource);
    if attachment_state.is_none()
        && (!attachment.spec.activation_bindings.is_empty()
            || !matches!(
                attachment.spec.kind,
                AttachmentKind::ModelReplay | AttachmentKind::Observation
            ))
    {
        return Err("attachment state resource is absent from the semantic graph".into());
    }
    for group in &graph.input_groups {
        let readers: Vec<_> = if attachment.spec.kind == AttachmentKind::ModelReplay {
            vec![graph.tasks[attachment.task_bindings[0].task_id as usize]
                .symbol
                .clone()]
        } else {
            vec![]
        };
        if attachment.spec.activation_bindings.is_empty()
            || group.resource != attachment.spec.state_resource
            || group.allow_absent
                != attachment
                    .spec
                    .activation_bindings
                    .iter()
                    .all(|binding| binding.on_missing == MissingPolicy::Hold)
            || group.readers != readers
            || !matches!(
                attachment.spec.kind,
                AttachmentKind::ActivationPatch | AttachmentKind::ModelReplay
            )
            || attachment
                .task_bindings
                .iter()
                .filter(|binding| binding.role != AttachmentTaskRole::Compute)
                .any(|binding| {
                    !group
                        .tasks
                        .contains(&graph.tasks[binding.task_id as usize].symbol)
                })
            || graph.tasks[attachment.task_bindings.last().unwrap().task_id as usize].symbol
                != group.publication
        {
            return Err(
                "conditional activation requires its exact inject producer group and pinned READ"
                    .into(),
            );
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
        let domain = probe
            .causal_domain
            .as_ref()
            .ok_or("layer routes require backend model dimensions")?;
        if u64::from(task.replay.as_ref().unwrap().layer_count) != domain.layers
            || attachment
                .spec
                .activation_bindings
                .iter()
                .any(|binding| u64::from(binding.layer) >= domain.layers)
            || task.timing.after != "model.execute"
            || task.timing.before != "state.publish"
            || task.replay.as_ref().unwrap().digest.len() != 64
            || !task
                .replay
                .as_ref()
                .unwrap()
                .digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || (task.replay.as_ref().unwrap().apply_updates
                && attachment.spec.kind != AttachmentKind::ParameterOverlay)
        {
            return Err("layer routes need their model domain, source and forward timing".into());
        }
        if attachment.spec.kind == AttachmentKind::ParameterOverlay {
            let publish = &graph.tasks[attachment.task_bindings[1].task_id as usize];
            if graph.tasks.len() != 2
                || graph.states.len() != 1
                || graph.states[0].scope != StateScope::Request
                || publish.effects.len() != 1
                || !publish.inputs.iter().any(|input| {
                    matches!(&input.source,
                    InputSource::Result { task: source, .. } if source == &task.symbol)
                })
                || graph.tasks[attachment.task_bindings[0].task_id as usize].symbol != task.symbol
                || task.effects.len() != 1
                || task.effects[0].kind != StateEffectKind::TransitionMaybe
                || task.effects[0].resource != attachment.spec.state_resource
                || task.inputs.iter().any(|input| match &input.source {
                    InputSource::PackedRows => false,
                    InputSource::StateSelection { resource, .. } => {
                        resource != &attachment.spec.state_resource
                    }
                    InputSource::HostService {
                        service,
                        port,
                        schema,
                    } => {
                        service != "parameter.operands"
                            || port != "dense_pairs"
                            || schema != "canonical.dense_operand_pairs@1"
                    }
                    _ => true,
                })
                || !task
                    .inputs
                    .iter()
                    .any(|input| matches!(input.source, InputSource::PackedRows))
                || !attachment.spec.activation_bindings.is_empty()
            {
                return Err(
                    "routed overlays need one request transition and its publication".into(),
                );
            }
        } else if attachment.spec.kind != AttachmentKind::ModelReplay {
            return Err("layer routes need a model replay or parameter overlay attachment".into());
        }
    }
    match attachment.spec.kind {
        AttachmentKind::Observation => {
            if attachment.task_bindings.len() != graph.tasks.len()
                || graph.tasks.iter().any(|task| {
                    task.effects.iter().any(|effect| {
                        effect.kind != StateEffectKind::Read || effect.version.is_some()
                    }) || task.replay.is_some()
                        || task
                            .outputs
                            .iter()
                            .any(|output| output.kind != OutputKind::Host)
                        || task.timing.after != "state.publish"
                        || task.timing.before != "sampler.pre_draw"
                        || task.inputs.iter().any(|input| match &input.source {
                            InputSource::Hidden { .. } | InputSource::StepId => false,
                            InputSource::StateSelection { resource, .. } => {
                                !task.effects.iter().any(|effect| {
                                    effect.kind == StateEffectKind::Read
                                        && effect.version.is_none()
                                        && &effect.resource == resource
                                })
                            }
                            _ => true,
                        })
                })
            {
                return Err(
                    "observation must read hidden inputs without changing model state".into(),
                );
            }
        }
        AttachmentKind::ModelReplay => {
            let task = &graph.tasks[attachment.task_bindings[0].task_id as usize];
            if task.replay.is_none() {
                return Err("model replay requires a tokenwise effect during forward".into());
            }
            if attachment.spec.activation_bindings.is_empty() {
                if graph.tasks.len() != 1
                    || !graph.states.is_empty()
                    || !state.plans.is_empty()
                    || !task.effects.is_empty()
                {
                    return Err(
                        "stateless model replay requires one forward task without state effects"
                            .into(),
                    );
                }
            } else {
                validate_activation_replay(attachment, graph)?;
            }
        }
        AttachmentKind::RequestTraining => {
            if attachment_state.unwrap().scope != StateScope::Request
                && attachment.spec.sampler_stage == SamplerStage::PackedPostFilterExact
            {
                return Err("attachment request state differs from semantic graph state".into());
            }
        }
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            validate_parameter_overlay_graph(graph, &attachment.task_bindings)?;
        }
    }
    match attachment.spec.kind {
        AttachmentKind::Observation => {
            if attachment.task_bindings.iter().any(|binding| {
                !timeline.anchors.iter().any(|anchor| {
                    anchor.symbol == "state.publish" && anchor.ordinal == binding.anchor_id as u64
                }) || binding.anchor_id >= attachment.sampler_anchor_id
            }) {
                return Err("observation must follow model execution and precede sampling".into());
            }
        }
        AttachmentKind::ModelReplay => {
            if !timeline.anchors.iter().any(|anchor| {
                anchor.symbol == "model.execute"
                    && anchor.ordinal == attachment.task_bindings[0].anchor_id as u64
            }) || attachment.task_bindings[0].anchor_id >= attachment.sampler_anchor_id
            {
                return Err("model replay task must bind the model forward anchor".into());
            }
            if !attachment.spec.activation_bindings.is_empty()
                && (attachment.spec.sampler_anchor != "sampler.pre_draw"
                    || attachment.task_bindings[1..].iter().any(|binding| {
                        !timeline.anchors.iter().any(|anchor| {
                            anchor.symbol == "request.prepare"
                                && anchor.ordinal == u64::from(binding.anchor_id)
                        })
                    }))
            {
                return Err(
                    "activation replay must publish at request.prepare before model forward".into(),
                );
            }
        }
        AttachmentKind::RequestTraining => {
            let transition = attachment.task_bindings.last().unwrap();
            if attachment.task_bindings.len() == 2
                && attachment.task_bindings[0].anchor_id >= attachment.sampler_anchor_id
            {
                return Err("attachment task anchors violate sampler timing".to_owned());
            }
            let mut past_update = false;
            for scheduled in &schedule.tasks {
                let task = &graph.tasks[scheduled.task_id as usize];
                if task
                    .inputs
                    .iter()
                    .any(|input| matches!(input.source, InputSource::SamplerFeedback))
                    && (attachment.spec.sampler_stage != SamplerStage::PackedPreFilterAdd
                        || scheduled.anchor_id <= attachment.sampler_anchor_id
                        || past_update
                        || (scheduled.task_id != transition.task_id
                            && (task
                                .effects
                                .iter()
                                .any(|effect| effect.kind != StateEffectKind::Read)
                                || task
                                    .outputs
                                    .iter()
                                    .any(|output| output.kind != OutputKind::Temporary))))
                {
                    return Err(
                        "sampler feedback requires a post-sample read or the request transition"
                            .into(),
                    );
                }
                past_update |= scheduled.task_id == transition.task_id;
            }
        }
        AttachmentKind::ParameterOverlay | AttachmentKind::ActivationPatch => {
            if attachment.task_bindings.is_empty() {
                return Ok(());
            }
            let transition = &attachment.task_bindings[0];
            let publish = &attachment.task_bindings[1];
            let anchor_id = |symbol: &str| {
                timeline
                    .anchors
                    .iter()
                    .find(|anchor| anchor.symbol == symbol)
                    .map(|anchor| anchor.ordinal as u32)
            };
            let request_state =
                attachment_state.is_some_and(|state| state.scope == StateScope::Request);
            let before_model = Some(transition.anchor_id) == anchor_id("request.prepare")
                && Some(publish.anchor_id) == anchor_id("request.prepare");
            let supported = if attachment.spec.kind == AttachmentKind::ActivationPatch {
                before_model
                    || (Some(transition.anchor_id) == anchor_id("state.publish")
                        && Some(publish.anchor_id) == anchor_id("state.publish"))
            } else if request_state {
                // Publication before the request's first forward is KV-safe; mid-request
                // updates continue through the model-execute publication path.
                before_model
                    || (Some(transition.anchor_id) == anchor_id("model.execute")
                        && Some(publish.anchor_id) == anchor_id("state.publish"))
            } else {
                before_model
                    || (Some(transition.anchor_id) == anchor_id("token.commit")
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

fn validate_position_source(
    task: &SemanticTask,
    graph: &SemanticGraph,
    probe: &BackendPlacementProbe,
) -> Result<(), String> {
    let effect = task.replay.as_ref().unwrap();
    if effect.position_states != probe.position_states
        || (!effect.position_states.is_empty()
            && (!effect.role_sensitive || !task.effects.is_empty() || !graph.states.is_empty()))
    {
        return Err(
            "PositionState needs a role-sensitive pure route and its pre-allocated schema".into(),
        );
    }

    Ok(())
}

pub(crate) fn validate_attachment(attachment: &BackendAttachment) -> Result<(), String> {
    let plan = &attachment.plan;
    if attachment.probe.graph_mode == AttachmentGraphMode::CudaGraph {
        if attachment.probe.device_kind != StateDeviceKind::Cuda
            || !attachment
                .probe
                .capabilities
                .iter()
                .any(|value| value == "backend/vllm_mrv2@1")
            || attachment.spec.kind != AttachmentKind::ParameterOverlay
            || plan.semantic_graph.states.iter().any(|state| {
                plan.semantic_graph.model_reads.contains(&state.resource)
                    && state.scope != StateScope::Request
            })
            || plan
                .semantic_graph
                .tasks
                .iter()
                .any(|task| task.replay.is_some())
        {
            return Err("Graph overlays need request-owned parameter reads without replay".into());
        }
    }
    if !attachment.spec.activation_bindings.is_empty()
        || !matches!(
            attachment.spec.kind,
            AttachmentKind::ModelReplay | AttachmentKind::Observation
        )
    {
        let layout = plan
            .tensor_layout_request
            .layouts
            .iter()
            .find(|layout| layout.resource == attachment.spec.state_resource)
            .ok_or_else(|| "attachment state has no tensor layout".to_owned())?;
        for binding in &attachment.spec.overlay_bindings {
            let target_layout = plan
                .tensor_layout_request
                .layouts
                .iter()
                .find(|layout| layout.resource == binding.resource)
                .ok_or("projection resource has no tensor layout")?;
            if binding.resource != attachment.spec.state_resource {
                return Err("projection resource differs from the attachment state".into());
            }
            validate_views(&binding.fields, &target_layout.layout.fields)?;
            let dtype = if binding.operator == OverlayOperator::HeadAdd {
                super::super::tensor_layout::TensorDType::Float32
            } else {
                attachment.probe.tensor_dtype.clone()
            };
            if binding
                .fields
                .iter()
                .any(|view| target_layout.layout.fields[view.field as usize].dtype != dtype)
            {
                return Err("projection dtype must match the model dtype".into());
            }
            let target = attachment
                .probe
                .targets
                .iter()
                .find(|target| target.target == binding.target)
                .ok_or("projection target has no backend dimensions")?;
            let shapes: Vec<_> = binding
                .fields
                .iter()
                .map(|view| view.shape.as_slice())
                .collect();
            let valid = match (&binding.operator, shapes.as_slice()) {
                (OverlayOperator::DenseAdd, [shape]) => {
                    *shape == [target.output_size, target.input_size]
                }
                (OverlayOperator::LowRankAdd, [a, b]) => {
                    a.len() == 2
                        && b.len() == 2
                        && a[0] > 0
                        && a[0] == b[1]
                        && a[1] == target.input_size
                        && b[0] == target.output_size
                }
                (OverlayOperator::HeadAdd, [shape]) => {
                    shape.len() == 3
                        && shape[0] > 0
                        && shape[1] == shape[2]
                        && shape[0] * shape[1] == target.input_size
                        && target.input_size == target.output_size
                }
                _ => false,
            };
            if !valid {
                return Err("projection views must match the target dimensions".into());
            }
        }
        for binding in &attachment.spec.activation_bindings {
            validate_views(&binding.fields, &layout.layout.fields)?;
            let shapes: Vec<_> = binding
                .fields
                .iter()
                .map(|view| view.shape.as_slice())
                .collect();
            let hidden = attachment.probe.hidden_size;
            let valid = match (&binding.operator, shapes.as_slice()) {
                (ActivationPatchOperator::Add, [shape]) => *shape == [hidden],
                (ActivationPatchOperator::LowRankAffine, [a, b, c]) => {
                    a.len() == 2
                        && b.len() == 1
                        && c.len() == 2
                        && a[0] > 0
                        && a[0] <= hidden
                        && a[0] == b[0]
                        && a[0] == c[1]
                        && a[1] == hidden
                        && c[0] == hidden
                }
                _ => false,
            };
            let dtype = if binding.operator == ActivationPatchOperator::Add {
                super::super::tensor_layout::TensorDType::Float32
            } else {
                attachment.probe.tensor_dtype.clone()
            };
            if !valid
                || binding
                    .fields
                    .iter()
                    .any(|view| layout.layout.fields[view.field as usize].dtype != dtype)
            {
                return Err("activation views must match the operator shape and dtype".into());
            }
        }
    }
    validate_cross_binding(
        attachment,
        &attachment.probe,
        &plan.semantic_graph,
        &plan.timeline,
        &plan.schedule,
        &plan.states,
    )
}

fn validate_views(
    views: &[super::types::FieldView],
    fields: &[super::super::tensor_layout::TensorFieldSpec],
) -> Result<(), String> {
    for view in views {
        let field = fields
            .iter()
            .find(|field| field.field_id == u64::from(view.field))
            .ok_or("view field is absent from layout")?;
        let size = view
            .shape
            .iter()
            .try_fold(1_u64, |a, b| a.checked_mul(*b))
            .ok_or("view size overflow")?;
        if view
            .offset
            .checked_add(size)
            .is_none_or(|end| end > field.shape.iter().product())
            || size == 0
        {
            return Err("view is outside its physical field".into());
        }
    }
    Ok(())
}

use super::super::scheduler::validate_physical as validate_schedule;
use super::super::semantic_graph::{StateEffectKind, StateScope};
use super::super::state_representation::validate_physical_plan as validate_state;
use super::super::tensor_layout::validate_decision as validate_layouts;
use super::types::*;

pub(crate) fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn ordered<T: Ord>(values: &[T]) -> bool {
    !values.is_empty() && values.windows(2).all(|pair| pair[0] < pair[1])
}
fn stages(values: &[StageAnchor]) -> bool {
    values.iter().map(|value| &value.stage).eq([
        &ParameterOverlayStage::Pin,
        &ParameterOverlayStage::Apply,
        &ParameterOverlayStage::Capture,
        &ParameterOverlayStage::Transition,
        &ParameterOverlayStage::Publish,
        &ParameterOverlayStage::Release,
        &ParameterOverlayStage::Finish,
    ]) && values.iter().all(|value| !value.anchor_symbol.is_empty())
}
fn tasks(values: &[TaskBinding]) -> bool {
    values.iter().map(|value| &value.role).eq([
        &ParameterOverlayTaskRole::Transition,
        &ParameterOverlayTaskRole::Publish,
    ]) && values
        .iter()
        .all(|value| !value.implementation_id.is_empty())
}
fn timeline_stages(values: &[StageAnchor], inputs: &Inputs) -> bool {
    stages(values)
        && values
            .windows(2)
            .all(|pair| pair[0].anchor_id <= pair[1].anchor_id)
        && values.iter().all(|value| {
            inputs
                .timeline
                .anchors
                .get(value.anchor_id as usize)
                .is_some_and(|anchor| {
                    anchor.ordinal == value.anchor_id as u64 && anchor.symbol == value.anchor_symbol
                })
        })
}
fn candidate(value: &Candidate) -> bool {
    !value.candidate_id.is_empty()
        && !value.adapter_id.is_empty()
        && digest(&value.backend_abi_digest)
        && digest(&value.runner_fingerprint)
        && digest(&value.model_fingerprint)
        && ordered(&value.bindings)
        && stages(&value.stage_anchors)
        && tasks(&value.task_bindings)
        && ordered(&value.capabilities)
}
pub(crate) fn validate(inputs: &Inputs) -> Result<(), String> {
    validate_schedule(
        &inputs.graph,
        &inputs.timeline,
        &inputs.schedule_decision,
        &inputs.schedule_physical,
    )?;
    validate_state(
        &inputs.state_request,
        &inputs.state_decision,
        &inputs.state_physical,
    )?;
    validate_layouts(
        &inputs.tensor_layout_request,
        &inputs.tensor_layout_decision,
    )?;
    let request = &inputs.request;
    if request.schema_version != SCHEMA_VERSION
        || request.plan_epoch == 0
        || request.plan_epoch != inputs.schedule_physical.plan_epoch
        || request.plan_epoch != inputs.state_physical.plan_epoch
        || request.state_resource.is_empty()
        || request.request_key_fields.is_empty()
        || !ordered(&request.bindings)
        || !timeline_stages(&request.stage_anchors, inputs)
        || !tasks(&request.task_bindings)
        || !ordered(&request.capabilities)
        || ![
            &request.backend_abi_digest,
            &request.runner_fingerprint,
            &request.model_fingerprint,
            &request.graph_signature,
            &request.schedule_signature,
            &request.state_representation_signature,
        ]
        .into_iter()
        .all(|value| digest(value))
    {
        return Err("parameter-overlay attachment request is invalid".to_owned());
    }
    if request.graph_signature != inputs.graph.execution_parameters_digest
        || request.schedule_signature != inputs.schedule_physical.schedule_signature
        || request.state_representation_signature != inputs.state_physical.representation_signature
    {
        return Err("parameter-overlay attachment differs from signed plans".to_owned());
    }
    let state = inputs
        .graph
        .states
        .binary_search_by(|state| state.resource.cmp(&request.state_resource))
        .map(|index| &inputs.graph.states[index])
        .map_err(|_| "parameter-overlay state resource is absent".to_owned())?;
    if state.scope != StateScope::Request || state.key_fields != request.request_key_fields {
        return Err("parameter-overlay request state differs from graph".to_owned());
    }
    let provider = inputs
        .state_physical
        .representations
        .binary_search_by(|provider| provider.resource.cmp(&request.state_resource))
        .map(|index| &inputs.state_physical.representations[index])
        .map_err(|_| "parameter-overlay state resource is not selected".to_owned())?;
    let layout = inputs
        .tensor_layout_request
        .layouts
        .binary_search_by(|layout| layout.resource.cmp(&request.state_resource))
        .map(|index| &inputs.tensor_layout_request.layouts[index])
        .map_err(|_| "parameter-overlay tensor layout resource is absent".to_owned())?;
    let binding = inputs
        .tensor_layout_decision
        .bindings
        .binary_search_by(|binding| binding.resource.cmp(&request.state_resource))
        .map(|index| &inputs.tensor_layout_decision.bindings[index])
        .map_err(|_| "parameter-overlay tensor layout binding is absent".to_owned())?;
    if provider.layout_id != u64::from(binding.layout_id)
        || provider.layout_namespace != binding.layout_namespace
        || provider.layout_digest != binding.layout_digest
        || request
            .bindings
            .iter()
            .any(|value| value.field_id as usize >= layout.layout.fields.len())
    {
        return Err("parameter-overlay bindings differ from frozen tensor layout".to_owned());
    }
    let transition_binding = &request.task_bindings[0];
    let publish_binding = &request.task_bindings[1];
    for task_binding in &request.task_bindings {
        let semantic = inputs
            .graph
            .tasks
            .get(task_binding.task_id as usize)
            .ok_or_else(|| "parameter-overlay task ID is outside the graph".to_owned())?;
        let physical = inputs
            .schedule_physical
            .tasks
            .iter()
            .find(|task| task.task_id == task_binding.task_id)
            .ok_or_else(|| "parameter-overlay task is absent from physical plan".to_owned())?;
        if physical.implementation_id != task_binding.implementation_id
            || physical.anchor_id != task_binding.anchor_id
            || !semantic.implementations.iter().any(|implementation| {
                implementation.implementation_id == task_binding.implementation_id
            })
        {
            return Err("parameter-overlay task binding differs from physical plan".to_owned());
        }
        let expected_effect = match task_binding.role {
            ParameterOverlayTaskRole::Transition => StateEffectKind::TransitionMaybe,
            ParameterOverlayTaskRole::Publish => StateEffectKind::Publish,
        };
        if !semantic.effects.iter().any(|effect| {
            effect.resource == request.state_resource && effect.kind == expected_effect
        }) {
            return Err("parameter-overlay task effect differs from semantic graph".to_owned());
        }
    }
    let transition_task = &inputs.graph.tasks[transition_binding.task_id as usize];
    let publish_task = &inputs.graph.tasks[publish_binding.task_id as usize];
    if !inputs
        .graph
        .edges
        .iter()
        .any(|edge| edge.source == transition_task.symbol && edge.target == publish_task.symbol)
    {
        return Err("parameter-overlay transition lacks a publish edge".to_owned());
    }
    let transition_stage = &request.stage_anchors[3];
    let publish_stage = &request.stage_anchors[4];
    if transition_stage.anchor_id != transition_binding.anchor_id
        || publish_stage.anchor_id != publish_binding.anchor_id
    {
        return Err("parameter-overlay stages differ from task anchors".to_owned());
    }
    if !inputs
        .candidates
        .windows(2)
        .all(|pair| pair[0].candidate_id < pair[1].candidate_id)
        || inputs.candidates.is_empty()
        || inputs.candidates.iter().any(|value| !candidate(value))
    {
        return Err("parameter-overlay candidates are invalid".to_owned());
    }
    let selected = inputs
        .candidates
        .iter()
        .find(|value| {
            value.adapter_id == request.adapter_id
                && value.backend_abi_digest == request.backend_abi_digest
                && value.runner_fingerprint == request.runner_fingerprint
                && value.model_fingerprint == request.model_fingerprint
                && value.bindings == request.bindings
                && value.operand_capture == request.operand_capture
                && value.stage_anchors == request.stage_anchors
                && value.task_bindings == request.task_bindings
                && value.capabilities == request.capabilities
        })
        .ok_or_else(|| "no exact parameter-overlay candidate".to_owned())?;
    if selected.candidate_id.is_empty() {
        return Err("parameter-overlay candidate ID is invalid".to_owned());
    }
    Ok(())
}
pub(crate) fn selected(inputs: &Inputs) -> Result<&Candidate, String> {
    validate(inputs)?;
    let request = &inputs.request;
    inputs
        .candidates
        .iter()
        .find(|value| {
            value.adapter_id == request.adapter_id
                && value.backend_abi_digest == request.backend_abi_digest
                && value.runner_fingerprint == request.runner_fingerprint
                && value.model_fingerprint == request.model_fingerprint
                && value.bindings == request.bindings
                && value.operand_capture == request.operand_capture
                && value.stage_anchors == request.stage_anchors
                && value.task_bindings == request.task_bindings
                && value.capabilities == request.capabilities
        })
        .ok_or_else(|| "no exact parameter-overlay candidate".to_owned())
}

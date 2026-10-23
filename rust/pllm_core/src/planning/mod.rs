//! Canonical closed planning contracts shared across Python and Rust.

use pyo3::prelude::*;

mod aer;
mod region_program;
mod region_cache;
mod region_dependence;
mod region_ast;
mod region_control;
mod region_bindings;
mod region_streams;
mod tensor_access;
mod region_storage;
mod region_scratch;
mod region_payload;
mod region_retention;
mod region_imports;
mod region_capacity;
mod region_selection;
mod region_tiling;
mod region_window;
mod region_outlining;
mod affine_scan;
pub(crate) mod backend_attachment;
pub(crate) mod scheduler;
pub(crate) mod semantic_graph;
pub(crate) mod state_representation;
pub(crate) mod tensor_layout;
mod version_hazards;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlanBundle {
    pub(crate) semantic_graph: semantic_graph::SemanticGraph,
    pub(crate) timeline: semantic_graph::Timeline,
    pub(crate) schedule_decision: scheduler::ScheduleDecision,
    pub(crate) schedule_physical: scheduler::PhysicalPlan,
    pub(crate) state_request: state_representation::StateRepresentationRequest,
    pub(crate) state_decision: state_representation::StateRepresentationDecision,
    pub(crate) state_physical: state_representation::StatePhysicalPlan,
    pub(crate) tensor_layout_request: tensor_layout::TensorLayoutRequest,
    pub(crate) tensor_layout_decision: tensor_layout::TensorLayoutDecision,
}

impl PlanBundle {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.schedule_physical.plan_epoch != self.state_physical.plan_epoch {
            return Err("schedule and state epochs differ".to_owned());
        }
        tensor_layout::validate_decision(
            &self.tensor_layout_request,
            &self.tensor_layout_decision,
        )?;
        state_representation::validate(&self.state_request, &self.state_decision, &self.state_physical)?;
        scheduler::validate_physical(&self.semantic_graph, &self.timeline, &self.schedule_decision,
            &self.schedule_physical, Some(&self.state_physical))?;
        for state in &self.state_physical.representations {
            if !self
                .semantic_graph
                .states
                .iter()
                .any(|known| known.resource == state.resource)
            {
                return Err("state representation is absent from the semantic graph".to_owned());
            }
            let binding = self
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
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature = (plan_json, local_candidates_json=None))]
fn validate_plan_bundle(
    plan_json: &str,
    local_candidates_json: Option<&str>,
) -> PyResult<()> {
    let plan: PlanBundle = serde_json::from_str(plan_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    if let Some(payload) = local_candidates_json {
        let local: Vec<state_representation::StateProviderPlan> = serde_json::from_str(payload)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        if local != plan.state_request.candidates {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "local state provider candidates differ from the signed catalog",
            ));
        }
    }
    plan.validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn compile_state_schedule(
    graph_json: &str,
    timeline_json: &str,
    request_json: &str,
    plan_epoch: u64,
) -> PyResult<(String, String, String, String)> {
    let compile = || -> Result<_, String> {
        let graph = semantic_graph::decode_graph(graph_json)?;
        let timeline = semantic_graph::decode_timeline(timeline_json)?;
        let request = state_representation::decode_request(request_json)?;
        let (decision, state, schedule) =
            state_representation::compile(&request, &graph, &timeline, plan_epoch)?;
        let physical = scheduler::project_physical(&schedule, plan_epoch)?;
        Ok((
            semantic_graph::canonical_ascii_json(&decision)?,
            semantic_graph::canonical_ascii_json(&state)?,
            semantic_graph::canonical_ascii_json(&schedule)?,
            semantic_graph::canonical_ascii_json(&physical)?,
        ))
    };
    compile().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    region_program::register(module)?;
    region_dependence::register(module)?;
    region_ast::register(module)?;
    region_bindings::register(module)?;
    region_streams::register(module)?;
    tensor_access::register(module)?;
    region_storage::register(module)?;
    region_retention::register(module)?;
    region_payload::register(module)?;
    region_capacity::register(module)?;
    region_selection::register(module)?;
    region_tiling::register(module)?;
    region_window::register(module)?;
    region_outlining::register(module)?;
    affine_scan::register(module)?;
    module.add_function(wrap_pyfunction!(compile_state_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_plan_bundle, module)?)?;
    Ok(())
}

//! Canonical closed planning contracts shared across Python and Rust.

use pyo3::prelude::*;

pub(crate) mod backend_attachment;
pub(crate) mod scheduler;
pub(crate) mod semantic_graph;
pub(crate) mod state_representation;
pub(crate) mod tensor_layout;
mod tensor_policy;
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
        tensor_layout::validate_decision(
            &self.tensor_layout_request,
            &self.tensor_layout_decision,
        )?;
        state_representation::validate(
            &self.state_request,
            &self.state_decision,
            &self.state_physical,
        )?;
        scheduler::validate_physical(
            &self.semantic_graph,
            &self.timeline,
            &self.schedule_decision,
            &self.schedule_physical,
            Some(&self.state_physical),
        )?;
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
fn validate_plan_bundle(plan_json: &str, local_candidates_json: Option<&str>) -> PyResult<()> {
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
) -> PyResult<(String, String, String, String)> {
    let compile = || -> Result<_, String> {
        let graph = semantic_graph::decode_graph(graph_json)?;
        let timeline = semantic_graph::decode_timeline(timeline_json)?;
        let request = state_representation::decode_request(request_json)?;
        let (decision, state, schedule) =
            state_representation::compile(&request, &graph, &timeline)?;
        let physical = scheduler::project_physical(&schedule);
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
    module.add_function(wrap_pyfunction!(
        semantic_graph::explain_state_visibility,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(compile_state_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_plan_bundle, module)?)?;
    Ok(())
}

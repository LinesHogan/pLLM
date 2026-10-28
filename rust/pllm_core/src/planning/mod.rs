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
    pub(crate) schedule: scheduler::ScheduleDecision,
    pub(crate) states: state_representation::StateRepresentationDecision,
    pub(crate) tensor_layout_request: tensor_layout::TensorLayoutRequest,
    pub(crate) tensor_layout_decision: tensor_layout::TensorLayoutDecision,
}

impl PlanBundle {
    pub(crate) fn validate(&self) -> Result<(), String> {
        tensor_layout::validate_decision(
            &self.tensor_layout_request,
            &self.tensor_layout_decision,
        )?;
        scheduler::validate_decision(
            &self.semantic_graph,
            &self.timeline,
            &self.schedule,
            Some(&self.states),
        )?;
        for state in &self.states.plans {
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
fn compile_state_schedule(
    graph_json: &str,
    timeline_json: &str,
    request_json: &str,
) -> PyResult<(String, String)> {
    let compile = || -> Result<_, String> {
        let graph = semantic_graph::decode_graph(graph_json)?;
        let timeline = semantic_graph::decode_timeline(timeline_json)?;
        let request = state_representation::decode_request(request_json)?;
        let (decision, schedule) = state_representation::compile(&request, &graph, &timeline)?;
        Ok((
            semantic_graph::canonical_ascii_json(&decision)?,
            semantic_graph::canonical_ascii_json(&schedule)?,
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
    Ok(())
}

//! Canonical closed planning contracts shared across Python and Rust.

use pyo3::prelude::*;

pub(crate) mod backend_attachment;
pub(crate) mod scheduler;
pub(crate) mod semantic_graph;
pub(crate) mod state_representation;
pub(crate) mod tensor_layout;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CanonicalPlanBundle {
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

impl CanonicalPlanBundle {
    pub(crate) fn validate(&self) -> Result<(), String> {
        tensor_layout::validate_decision(
            &self.tensor_layout_request,
            &self.tensor_layout_decision,
        )?;
        scheduler::validate_physical(
            &self.semantic_graph,
            &self.timeline,
            &self.schedule_decision,
            &self.schedule_physical,
        )?;
        state_representation::validate_physical_plan(
            &self.state_request,
            &self.state_decision,
            &self.state_physical,
        )?;
        if self.schedule_physical.plan_epoch != self.state_physical.plan_epoch {
            return Err("schedule and state epochs differ".to_owned());
        }
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
fn validate_canonical_plan_bundle(plan_json: &str) -> PyResult<()> {
    let plan: CanonicalPlanBundle = serde_json::from_str(plan_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    plan.validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(validate_canonical_plan_bundle, module)?)?;
    Ok(())
}

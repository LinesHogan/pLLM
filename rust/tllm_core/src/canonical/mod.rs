//! Canonical closed planning contracts shared across Python and Rust.

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

use serde::{Deserialize, Serialize};

use super::super::scheduler::{PhysicalPlan, ScheduleDecision};
use super::super::semantic_graph::{RequestKeyField, SemanticGraph, Timeline};
use super::super::state_representation::{
    StatePhysicalPlan, StateRepresentationDecision, StateRepresentationRequest,
};
use super::super::tensor_layout::{TensorLayoutDecision, TensorLayoutRequest};

pub(crate) const SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ParameterModuleKind {
    Attention,
    Mlp,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ParameterRole {
    DownProjectionWeight,
    QueryProjectionWeight,
    ValueProjectionWeight,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ParameterOverlayOperator {
    DenseAdd,
    LowRankAdd,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ParameterOverlayScope {
    Request,
    Global,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OperandCaptureKind {
    MlpDownProjectionPair,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ParameterOverlayStage {
    Pin,
    Apply,
    Capture,
    Transition,
    Publish,
    Release,
    Finish,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ParameterOverlayTaskRole {
    Transition,
    Publish,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub(crate) struct Binding {
    pub(crate) layer: i64,
    pub(crate) module: ParameterModuleKind,
    pub(crate) role: ParameterRole,
    pub(crate) operator: ParameterOverlayOperator,
    pub(crate) field_ids: Vec<u32>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub(crate) struct StageAnchor {
    pub(crate) stage: ParameterOverlayStage,
    pub(crate) anchor_id: u32,
    pub(crate) anchor_symbol: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskBinding {
    pub(crate) role: ParameterOverlayTaskRole,
    pub(crate) task_id: u32,
    pub(crate) implementation_id: String,
    pub(crate) anchor_id: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Candidate {
    pub(crate) candidate_id: String,
    pub(crate) adapter_id: String,
    pub(crate) backend_abi_digest: String,
    pub(crate) runner_fingerprint: String,
    pub(crate) model_fingerprint: String,
    pub(crate) scope: ParameterOverlayScope,
    pub(crate) bindings: Vec<Binding>,
    pub(crate) operand_capture: OperandCaptureKind,
    pub(crate) stage_anchors: Vec<StageAnchor>,
    pub(crate) task_bindings: Vec<TaskBinding>,
    pub(crate) capabilities: Vec<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) adapter_id: String,
    pub(crate) backend_abi_digest: String,
    pub(crate) runner_fingerprint: String,
    pub(crate) model_fingerprint: String,
    pub(crate) state_resource: String,
    pub(crate) scope: ParameterOverlayScope,
    pub(crate) request_key_fields: Vec<RequestKeyField>,
    pub(crate) bindings: Vec<Binding>,
    pub(crate) operand_capture: OperandCaptureKind,
    pub(crate) stage_anchors: Vec<StageAnchor>,
    pub(crate) task_bindings: Vec<TaskBinding>,
    pub(crate) plan_epoch: u64,
    pub(crate) graph_signature: String,
    pub(crate) schedule_signature: String,
    pub(crate) state_representation_signature: String,
    pub(crate) capabilities: Vec<String>,
    pub(crate) schema_version: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Inputs {
    pub(crate) request: Request,
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) graph: SemanticGraph,
    pub(crate) timeline: Timeline,
    pub(crate) schedule_decision: ScheduleDecision,
    pub(crate) schedule_physical: PhysicalPlan,
    pub(crate) state_request: StateRepresentationRequest,
    pub(crate) state_decision: StateRepresentationDecision,
    pub(crate) state_physical: StatePhysicalPlan,
    pub(crate) tensor_layout_request: TensorLayoutRequest,
    pub(crate) tensor_layout_decision: TensorLayoutDecision,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Decision {
    pub(crate) request_digest: String,
    pub(crate) selected_candidate_id: String,
    pub(crate) signature: String,
    pub(crate) schema_version: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Physical {
    pub(crate) plan_epoch: u64,
    pub(crate) selected_candidate_id: String,
    pub(crate) attachment_signature: String,
    pub(crate) schedule_signature: String,
    pub(crate) state_representation_signature: String,
    pub(crate) scope: ParameterOverlayScope,
    pub(crate) bindings: Vec<Binding>,
    pub(crate) stage_anchors: Vec<StageAnchor>,
    pub(crate) task_bindings: Vec<TaskBinding>,
    pub(crate) bundle_signature: String,
    pub(crate) schema_version: u64,
}

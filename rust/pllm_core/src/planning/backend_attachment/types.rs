//! Closed wire contracts signed by the backend attachment compiler.

use serde::{Deserialize, Serialize};

use super::super::state_representation::StateDeviceKind;
use super::super::tensor_layout::TensorDType;
use super::super::PlanBundle;

pub(crate) const SCHEMA_VERSION: u64 = 7;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttachmentTopology {
    InProcess,
    EngineCore,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttachmentKind {
    Observation,
    ModelReplay,
    RequestTraining,
    ParameterOverlay,
    ActivationPatch,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttachmentGraphMode {
    Eager,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SamplerStage {
    PackedPostFilterExact,
    PackedPreFilterAdd,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttachmentCleanupStage {
    RequestFinish,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StreamDependency {
    HiddenProducerToCompute,
    ComputeToSampler,
    SamplerToTransition,
    TransitionToRelease,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttachmentTaskRole {
    Compute,
    Transition,
    Publish,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HiddenLayerKind {
    Index,
    Path,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HiddenSite {
    BlockOutput,
    ModelOutput,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HiddenPhase {
    Prefill,
    Decode,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct HiddenBinding {
    pub(crate) layer: String,
    pub(crate) layer_kind: HiddenLayerKind,
    pub(crate) site: HiddenSite,
    pub(crate) module_path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AttachmentHiddenResource {
    pub(crate) binding: HiddenBinding,
    pub(crate) phase: HiddenPhase,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AttachmentTaskBinding {
    pub(crate) role: AttachmentTaskRole,
    pub(crate) task_id: u32,
    pub(crate) implementation_id: String,
    pub(crate) anchor_id: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CausalModelDomain {
    pub(crate) requests: u64,
    pub(crate) layers: u64,
    pub(crate) positions: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct RegionFunctionBody {
    pub declaration: super::super::region_program::Function,
    pub program: super::super::region_program::Program,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackendPlacementProbe {
    pub(crate) adapter_id: String,
    pub(crate) backend_abi_digest: String,
    pub(crate) runner_fingerprint: String,
    pub(crate) model_fingerprint: String,
    pub(crate) topology: AttachmentTopology,
    pub(crate) graph_mode: AttachmentGraphMode,
    pub(crate) device_kind: StateDeviceKind,
    pub(crate) device_ordinal: u64,
    pub(crate) tensor_parallel_size: u64,
    pub(crate) pipeline_parallel_size: u64,
    pub(crate) tensor_dtype: TensorDType,
    pub(crate) hidden_size: u64,
    pub(crate) max_packed_rows: u64,
    pub(crate) hidden_resources: Vec<AttachmentHiddenResource>,
    pub(crate) capabilities: Vec<String>,
    pub(crate) schema_version: u64,
    #[serde(default)]
    pub(crate) causal_domain: Option<CausalModelDomain>,
    #[serde(default)]
    pub(super) region_imports: Vec<super::super::region_imports::TensorImport>,
    #[serde(default)]
    pub(super) region_bodies: Vec<RegionFunctionBody>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AttachmentCompilerInputs {
    pub(crate) request: BackendAttachmentRequest,
    pub(crate) probe: BackendPlacementProbe,
    pub(crate) plan: PlanBundle,
}

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
#[serde(deny_unknown_fields)]
pub(crate) struct ParameterTarget {
    pub(crate) layer: u32,
    pub(crate) module: ParameterModuleKind,
    pub(crate) role: ParameterRole,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OverlayOperator {
    DenseAdd,
    LowRankAdd,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct OverlayBinding {
    pub(crate) target: ParameterTarget,
    pub(crate) operator: OverlayOperator,
    pub(crate) fields: Vec<u32>,
    pub(crate) scale: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActivationPatchOperator {
    LowRankAffine,
    Add,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecodeWindow {
    None,
    All,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecodeRefresh {
    Inherit,
    Step,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MissingPolicy {
    Stop,
    Hold,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActivationPatchBinding {
    pub(crate) layer: u32,
    pub(crate) fields: Vec<u32>,
    pub(crate) first_tokens: u32,
    pub(crate) last_tokens: u32,
    pub(crate) operator: ActivationPatchOperator,
    pub(crate) decode: DecodeWindow,
    pub(crate) decode_refresh: DecodeRefresh,
    pub(crate) on_missing: MissingPolicy,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackendAttachmentRequest {
    pub(crate) attachment_kind: AttachmentKind,
    pub(crate) packed_row_schema_version: u64,
    pub(crate) task_bindings: Vec<AttachmentTaskBinding>,
    pub(crate) sampler_anchor_id: u32,
    pub(crate) sampler_anchor_symbol: String,
    pub(crate) stream_dependencies: Vec<StreamDependency>,
    pub(crate) sampler_stage: SamplerStage,
    pub(crate) cleanup_stage: AttachmentCleanupStage,
    pub(crate) state_resource: String,
    pub(crate) overlay_bindings: Vec<OverlayBinding>,
    pub(crate) activation_bindings: Vec<ActivationPatchBinding>,
    pub(crate) schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackendAttachmentDecision {
    pub(crate) request_digest: String,
    pub(crate) signature: String,
    pub(crate) schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackendAttachmentPhysicalPlan {
    pub(crate) plan_epoch: u64,
    pub(crate) adapter_id: String,
    pub(crate) attachment_signature: String,
    pub(crate) schedule_signature: String,
    pub(crate) state_representation_signature: String,
    pub(crate) device_kind: StateDeviceKind,
    pub(crate) device_ordinal: u64,
    pub(crate) topology: AttachmentTopology,
    pub(crate) graph_mode: AttachmentGraphMode,
    pub(crate) bundle_signature: String,
    pub(crate) schema_version: u64,
}

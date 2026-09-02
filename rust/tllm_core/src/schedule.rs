//! Pure-JSON schedule compiler boundary.
//!
//! Only primitive, versioned metadata crosses this module. Python callables,
//! tensors, CUDA handles, and backend objects cannot be extracted here.

use isl_rs::{Context, Schedule, ScheduleConstraints, UnionAccessInfo, UnionMap, UnionSet};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyInt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::Write;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TaskImplementationKind {
    Captured,
    Extern,
    OptimizerTransition,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ScheduleInputKind {
    Hidden,
    PackedRequestRows,
    PackedSamplerRows,
    PackedHidden,
    Result,
    StateSelection,
    Provided,
    StepId,
    Staged,
    SamplerRequestIds,
    SamplerSampleIdxs,
    SamplerSampleIdxTensor,
    SamplerLogits,
    SampledTokens,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum HiddenLayerKind {
    #[default]
    None,
    Index,
    Path,
    Wildcard,
}

impl HiddenLayerKind {
    fn as_wire(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Index => "index",
            Self::Path => "path",
            Self::Wildcard => "wildcard",
        }
    }
}

/// Closed hidden-activation tap sites. `None` retains the legacy empty wire
/// value for non-hidden inputs without reintroducing an open string field.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum HiddenSite {
    #[default]
    #[serde(rename = "")]
    None,
    Post,
    BlockOutput,
}

impl HiddenSite {
    fn as_wire(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Post => "post",
            Self::BlockOutput => "block_output",
        }
    }
}

/// Closed hidden-activation phases. `None` retains the legacy empty wire
/// value for non-hidden inputs.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum HiddenPhase {
    #[default]
    #[serde(rename = "")]
    None,
    Prefill,
    Decode,
}

impl HiddenPhase {
    fn as_wire(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

fn hidden_locator_capability(site: HiddenSite, phase: HiddenPhase) -> Option<&'static str> {
    match (site, phase) {
        (HiddenSite::Post, HiddenPhase::Prefill) => Some("hidden/post/prefill"),
        (HiddenSite::Post, HiddenPhase::Decode) => Some("hidden/post/decode"),
        (HiddenSite::BlockOutput, HiddenPhase::Prefill) => Some("hidden/block_output/prefill"),
        (HiddenSite::BlockOutput, HiddenPhase::Decode) => Some("hidden/block_output/decode"),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ScheduleOutputKind {
    Sink,
    Temporary,
    Host,
    Sampler,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ScheduleEffectKind {
    StateRead,
    StateWrite,
    StateTransition,
    StateTransitionMaybe,
    VersionPublish,
    ExternIo,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ScheduleEdgeKind {
    Data,
    Effect,
    Control,
    Anchor,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum StateScope {
    Request,
    Replica,
    Shard,
    Global,
}

/// Closed request-identity projection for request-scoped native state.
/// Python authoring exports the matching ``RequestKeyField`` enum; only its
/// string wire representation crosses this JSON boundary.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum RequestKeyField {
    RequestId,
    PromptIdx,
    SampleIdx,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ScheduleStateStorage {
    Dynamic,
    OptimizerStable,
    FixedBatched,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum RegionInputKind {
    Hidden,
    PackedRequestRows,
    PackedSamplerRows,
    PackedHidden,
    Provided,
    Materialized,
    StateSelection,
    StepId,
    Staged,
    SamplerRequestIds,
    SamplerSampleIdxs,
    SamplerSampleIdxTensor,
    SamplerLogits,
    SampledTokens,
}

impl RegionInputKind {
    fn as_wire(self) -> &'static str {
        match self {
            Self::Hidden => "hidden",
            Self::PackedRequestRows => "packed_request_rows",
            Self::PackedSamplerRows => "packed_sampler_rows",
            Self::PackedHidden => "packed_hidden",
            Self::Provided => "provided",
            Self::Materialized => "materialized",
            Self::StateSelection => "state_selection",
            Self::StepId => "step_id",
            Self::Staged => "staged",
            Self::SamplerRequestIds => "sampler_request_ids",
            Self::SamplerSampleIdxs => "sampler_sample_idxs",
            Self::SamplerSampleIdxTensor => "sampler_sample_idx_tensor",
            Self::SamplerLogits => "sampler_logits",
            Self::SampledTokens => "sampled_tokens",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum RegionOutputKind {
    Sink,
    Materialized,
    Host,
    Sampler,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegionImplementationKind {
    Compiled,
    Extern,
}

/// Closed compiler adapters.  The planner, rather than Python callable
/// inspection, selects one for every ordinary compiled Region.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegionCompilerAdapter {
    TorchCompile,
    Eager,
}

/// Auxiliary Regions currently own their gradient context.  Keep this as a
/// closed command because inheriting the ambient vLLM inference context would
/// be an execution-policy choice made outside the signed decision.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegionAutogradPolicy {
    EnableGrad,
}

/// The complete compiler command for a normal Python/Torch ExecutionRegion.
/// It is part of both the schedule signature and the artifact key.  Python
/// may only dispatch this record to a registered adapter.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RegionCompileCommand {
    adapter: RegionCompilerAdapter,
    fullgraph: bool,
    dynamic: bool,
    shape_specialization: Vec<(String, String)>,
    autograd_policy: RegionAutogradPolicy,
    fallback_policy: RegionFallbackPolicy,
}

fn deserialize_required_compile_command<'de, D>(
    deserializer: D,
) -> Result<Option<RegionCompileCommand>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<RegionCompileCommand>::deserialize(deserializer)
}

/// The closed capture command for a fully-bound ExecutionRegion.
///
/// This is intentionally distinct from `capture_safe`: the latter records
/// capability evidence, while this enum is the signed physical decision.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegionCaptureMode {
    Disabled,
    Require,
}

/// The closed stream command for a fully-bound ExecutionRegion.
///
/// The planner owns this value. Python lowering must only bind it to the
/// runtime descriptor and cannot select a task-local stream.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegionStreamClass {
    Default,
    SideForward,
}

/// The closed host-side submission command for a fully-bound Region.
///
/// This is distinct from the GPU stream class. `AsyncSerial` permits the
/// runtime to submit an explicitly safe extern Region to the one Rust-owned
/// serial host lane; it does not let Python or a workload create a lane or
/// choose its capacity.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegionHostSubmission {
    Inline,
    AsyncSerial,
}

/// A closed runtime command over the fixed-capacity async host lane.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum HostCommandKind {
    WaitAsync,
    ExecuteInline,
    SubmitAsync,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AsyncBatchCaptureMode {
    Disabled,
    OptimizerTransition,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RuntimeHostCommand {
    kind: HostCommandKind,
    submission_id: String,
    region_ids: Vec<String>,
    async_capture_mode: AsyncBatchCaptureMode,
    async_artifact_key: String,
}

/// The closed fallback command for a fully-bound ExecutionRegion.
///
/// An unsupported Region must reject before execution instead of silently
/// rebuilding a task-level eager plan.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegionFallbackPolicy {
    Reject,
}

/// A backend hook boundary that the runtime can observe without inferring it
/// from an anchor name.  The wire spelling deliberately matches the backend
/// event protocol.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
enum BackendObservableEvent {
    #[serde(rename = "prepare_inputs.pre")]
    PrepareInputsPre,
    #[serde(rename = "prepare_inputs.post")]
    PrepareInputsPost,
    #[serde(rename = "execute_model.pre")]
    ExecuteModelPre,
    #[serde(rename = "execute_model.post")]
    ExecuteModelPost,
    #[serde(rename = "sample_tokens.pre")]
    SampleTokensPre,
    #[serde(rename = "sampler.pre_draw")]
    SamplerPreDraw,
    #[serde(rename = "sample_tokens.post")]
    SampleTokensPost,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskTiming {
    after: String,
    before: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct AffineDomain {
    name: String,
    indices: Vec<String>,
    bounds: Vec<(String, String, String)>,
    parameters: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AffineExpr {
    coefficients: Vec<(String, i64)>,
    constant: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AffineAccess {
    value: String,
    indices: Vec<AffineExpr>,
    mode: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleTask {
    task_id: String,
    timing: TaskTiming,
    temporary_bytes: u64,
    implementation_ids: Vec<String>,
    #[serde(default = "default_implementation_kinds")]
    implementation_kinds: Vec<TaskImplementationKind>,
    /// Raw per-candidate capability evidence.  Schema v2 requires this so
    /// Rust can derive the backend-facing capability set without trusting
    /// Python lowering's filtered view.
    #[serde(default)]
    implementation_capabilities: Vec<Vec<String>>,
    required_capabilities: Vec<String>,
    capture_safe: bool,
    #[serde(default)]
    effects: Vec<ScheduleEffect>,
    #[serde(default)]
    domain: AffineDomain,
    #[serde(default)]
    accesses: Vec<AffineAccess>,
    #[serde(default)]
    inputs: Vec<ScheduleInput>,
    #[serde(default)]
    outputs: Vec<ScheduleOutput>,
}

/// The one implementation selected by the Rust planner for a task.  The
/// ScheduleGraph retains every declaration so the planner can make this
/// choice against the coordinator's timeline capabilities; nothing in Python
/// is allowed to pick a different member of that declaration later.
#[derive(Clone, Debug)]
struct SelectedTaskImplementation {
    kind: TaskImplementationKind,
    implementation_id: String,
    required_capabilities: Vec<String>,
    compile_safe: bool,
    capture_safe: bool,
    side_stream_safe: bool,
    async_host_safe: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleInput {
    name: String,
    kind: ScheduleInputKind,
    source: String,
    #[serde(default)]
    output: String,
    #[serde(default)]
    layer: String,
    #[serde(default = "default_layer_kind")]
    layer_kind: HiddenLayerKind,
    #[serde(default)]
    site: HiddenSite,
    #[serde(default)]
    phase: HiddenPhase,
    #[serde(default)]
    packed_rows_port: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleOutput {
    name: String,
    kind: ScheduleOutputKind,
    target: String,
}

fn default_implementation_kinds() -> Vec<TaskImplementationKind> {
    vec![TaskImplementationKind::Captured]
}

fn default_layer_kind() -> HiddenLayerKind {
    HiddenLayerKind::None
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ScheduleEffect {
    kind: ScheduleEffectKind,
    resource: String,
    /// Schema v4 binds request-scoped effects to a closed StateSelection
    /// input of the owning task.  Earlier schema versions did not carry a
    /// selection and therefore retain the empty wire value.
    selection_port: Option<String>,
}

/// An effect emitted on a signed ExecutionRegion.  Request-scoped effects
/// retain the canonical Region input binding selected by their task port;
/// this prevents a worker from substituting an unrelated request slot after
/// Rust has signed the region.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RegionEffect {
    kind: ScheduleEffectKind,
    resource: String,
    selection_binding_id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleEdge {
    source: String,
    target: String,
    kind: ScheduleEdgeKind,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleValue {
    value_id: String,
    version: i64,
    producer: String,
    #[serde(default)]
    producer_output: String,
    consumers: Vec<String>,
    size_bytes: u64,
    alias_set: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleState {
    resource: String,
    scope: StateScope,
    #[serde(default)]
    request_key: Vec<RequestKeyField>,
    #[serde(default)]
    storage: Option<ScheduleStateStorage>,
    transition: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleGraph {
    workload_key: String,
    tasks: Vec<ScheduleTask>,
    edges: Vec<ScheduleEdge>,
    values: Vec<ScheduleValue>,
    #[serde(default)]
    states: Vec<ScheduleState>,
    persistent_state_bytes: u64,
    schema_version: i64,
    /// (workload_id, approximation policy) for non-exact workloads.  Absent
    /// (not empty) on the wire for exact-only graphs.
    #[serde(default)]
    workload_approximations: Vec<(String, String)>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TimelineAnchor {
    anchor_id: String,
    ordinal: usize,
    observable: bool,
    graph_boundary: bool,
    visible_values: Vec<String>,
    /// `None` is only accepted for a legacy timeline schema.  Schema v2
    /// requires the backend to make every event-to-anchor relation explicit.
    #[serde(default)]
    backend_events: Option<Vec<BackendObservableEvent>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reservation {
    memory_bytes: u64,
    compute_units: u64,
    bandwidth_units: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MainRegion {
    region_id: String,
    start: String,
    end: String,
    pinned: bool,
    reservation: Reservation,
    capture_region: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Timeline {
    backend_id: String,
    backend_version: String,
    runner_family: String,
    graph_mode: String,
    anchors: Vec<TimelineAnchor>,
    regions: Vec<MainRegion>,
    static_headroom_bytes: u64,
    capabilities: Vec<String>,
    schema_version: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ResidualResourceBinding {
    layer_index: i64,
    site: HiddenSite,
    phase: HiddenPhase,
    module_path: String,
}

impl ResidualResourceBinding {
    fn locator_key(&self) -> (i64, &'static str, &'static str) {
        (self.layer_index, self.site.as_wire(), self.phase.as_wire())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PackedHiddenResourceBinding {
    layer: String,
    layer_kind: HiddenLayerKind,
    site: HiddenSite,
    module_path: String,
}

fn packed_hidden_locator_key(
    layer: &str,
    layer_kind: HiddenLayerKind,
    site: HiddenSite,
) -> PyResult<(&str, &'static str, &'static str)> {
    if layer.is_empty()
        || !matches!(layer_kind, HiddenLayerKind::Index | HiddenLayerKind::Path)
        || site != HiddenSite::BlockOutput
    {
        return Err(value_error(
            "packed hidden resource locator requires a non-empty INDEX or PATH layer and block_output site",
        ));
    }
    if layer_kind == HiddenLayerKind::Index {
        let index = layer.parse::<i64>().map_err(|_| {
            value_error(
                "packed hidden resource INDEX layer must be a canonical signed 64-bit integer",
            )
        })?;
        if index.to_string() != layer {
            return Err(value_error(
                "packed hidden resource INDEX layer must be a canonical signed 64-bit integer",
            ));
        }
    }
    Ok((layer, layer_kind.as_wire(), site.as_wire()))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompileOptions {
    memory_budget_bytes: u64,
    allow_serial_fallback: bool,
    #[serde(default)]
    shape_specialization: Vec<(String, String)>,
    #[serde(default)]
    residual_resource_manifest: Vec<ResidualResourceBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    packed_hidden_resource_manifest: Option<Vec<PackedHiddenResourceBinding>>,
    #[serde(default = "default_static_binding_manifest")]
    static_binding_manifest: String,
    #[serde(default)]
    compiler_fingerprint: String,
    #[serde(default = "default_compile_policy")]
    compile_policy: String,
    #[serde(default = "default_capture_policy")]
    capture_policy: String,
    /// Internal search-space gate, enabled by default. The planner alone
    /// selects stream-safe pure or ordered-transition Regions and signs
    /// every wait; workload declarations never choose a lane or placement.
    #[serde(default = "default_allow_side_stream")]
    allow_side_stream: bool,
    schema_version: i64,
}

fn default_static_binding_manifest() -> String {
    "[]".to_owned()
}

/// Closed, coordinator-signed static semantics for a backend-owned Region.
///
/// Python transports this record verbatim from parent and worker factories.
/// Rust parses, validates, and compares it before callbacks, tensors, or
/// model objects can be bound in a worker.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct StaticBackendBinding {
    workload_id: String,
    factory: String,
    config: serde_json::Value,
    specialization_key: Vec<(String, String)>,
    backend_region_implementation_ids: Vec<String>,
}

/// Import recipes are executable authority, so their pure-data identity must
/// be checked against the coordinator-signed static manifest before Python
/// imports a worker factory.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerReconstructionSpec {
    factory: String,
    config: serde_json::Value,
}

fn default_compile_policy() -> String {
    "torch_compile".to_owned()
}

fn default_capture_policy() -> String {
    "auto".to_owned()
}

fn default_allow_side_stream() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Placement {
    task_id: String,
    after_anchor: String,
    implementation_id: String,
    stream: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ScheduleTree {
    kind: String,
    tasks: Vec<String>,
    #[serde(default)]
    affine_regions: Vec<AffineRegionSchedule>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AffineRegionSchedule {
    tasks: Vec<String>,
    schedule: String,
    dependences: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Allocation {
    task_id: String,
    size_bytes: u64,
    live_start: String,
    live_end: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct MemoryPlan {
    budget_bytes: u64,
    persistent_bytes: u64,
    peak_bytes: u64,
    allocations: Vec<Allocation>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Fence {
    fence_id: String,
    source: String,
    target: String,
    kind: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Cost {
    peak_memory_bytes: u64,
    transfer_bytes: u64,
    main_path_extension_ns: u64,
    compute_cost: u64,
    #[serde(default)]
    materialization_bytes: u64,
    #[serde(default)]
    capture_boundary_count: u64,
    #[serde(default)]
    launch_group_count: u64,
    #[serde(default)]
    cross_region_fence_count: u64,
    #[serde(default)]
    critical_path_extension: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Materialization {
    materialization_id: String,
    value_id: String,
    version: i64,
    producer_region: String,
    consumer_region: String,
    size_bytes: u64,
    live_start: String,
    live_end: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct RegionPort {
    task_id: String,
    port_name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RegionInputBinding {
    binding_id: String,
    kind: RegionInputKind,
    source: String,
    consumers: Vec<RegionPort>,
    layer: String,
    layer_index: Option<i64>,
    layer_kind: HiddenLayerKind,
    #[serde(default)]
    site: HiddenSite,
    #[serde(default)]
    phase: HiddenPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    packed_rows_binding_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RegionOutputBinding {
    binding_id: String,
    kind: RegionOutputKind,
    target: String,
    producer: RegionPort,
    value_binding_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RegionTaskPortBinding {
    port_name: String,
    value_binding_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RegionTaskInvocation {
    task_id: String,
    inputs: Vec<RegionTaskPortBinding>,
    outputs: Vec<RegionTaskPortBinding>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ExecutionRegion {
    region_id: String,
    tasks: Vec<String>,
    after_anchor: String,
    before_anchor: String,
    placement_anchor: String,
    implementation_kind: RegionImplementationKind,
    implementation_ids: Vec<String>,
    #[serde(deserialize_with = "deserialize_required_compile_command")]
    compile_command: Option<RegionCompileCommand>,
    required_capabilities: Vec<String>,
    dependencies: Vec<String>,
    materialized_inputs: Vec<String>,
    materialized_outputs: Vec<String>,
    input_bindings: Vec<RegionInputBinding>,
    output_bindings: Vec<RegionOutputBinding>,
    task_invocations: Vec<RegionTaskInvocation>,
    effects: Vec<RegionEffect>,
    compile_safe: bool,
    capture_safe: bool,
    side_stream_safe: bool,
    capture_mode: RegionCaptureMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stream: Option<RegionStreamClass>,
    host_submission: RegionHostSubmission,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback_policy: Option<RegionFallbackPolicy>,
    artifact_key: String,
    serial_fallback: bool,
    affine_schedule: String,
}

/// One signed unit of runtime advancement.  Multiple entries may share a
/// backend event; their order is the canonical timeline-anchor order.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RuntimeAdvancePlan {
    backend_event: BackendObservableEvent,
    anchor_id: String,
    region_ids: Vec<String>,
    host_commands: Vec<RuntimeHostCommand>,
    backend_region_binder_stages: Vec<BackendRegionBinderStage>,
    wait_for_prior_side_work: bool,
    wait_for_side_outputs: bool,
}

/// A backend-owned extern binding selected by the signed physical plan.
///
/// Python must consume this record mechanically.  It must not infer binder
/// eligibility again from Region kind or capability strings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BackendRegionBinderStage {
    backend_event: BackendObservableEvent,
    region_id: String,
    implementation_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Decision {
    graph_key: String,
    timeline_key: String,
    execution_order: Vec<String>,
    placements: Vec<Placement>,
    schedule_tree: ScheduleTree,
    memory_plan: MemoryPlan,
    #[serde(default)]
    execution_regions: Vec<ExecutionRegion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_advance_plan: Option<Vec<RuntimeAdvancePlan>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completion_anchor: Option<String>,
    #[serde(default)]
    materializations: Vec<Materialization>,
    moves: Vec<serde_json::Value>,
    fences: Vec<Fence>,
    capture_partitions: Vec<Vec<String>>,
    fallback: String,
    cost: Cost,
    explanations: Vec<String>,
    signature: String,
    schema_version: i64,
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn schedule_error(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(message.into())
}

fn decode<'a, T: Deserialize<'a>>(payload: &'a str, label: &str) -> PyResult<T> {
    serde_json::from_str(payload).map_err(|error| value_error(format!("invalid {label}: {error}")))
}

fn digest(payload: &str) -> String {
    format!("{:x}", Sha256::digest(payload.as_bytes()))
}

#[derive(Serialize)]
struct StateLayoutIdBinding<'a> {
    resource: &'a str,
    layout_id: u32,
}

#[pyfunction]
fn intern_state_layout_ids(graph: &str) -> PyResult<String> {
    let graph_desc: ScheduleGraph = decode(graph, "ScheduleGraph")?;
    let mut resources = BTreeSet::new();
    let bindings: Vec<StateLayoutIdBinding<'_>> = graph_desc
        .states
        .iter()
        .enumerate()
        .map(|(index, state)| {
            if state.resource.is_empty() || !resources.insert(state.resource.as_str()) {
                return Err(value_error(
                    "state layout interner requires unique non-empty resources",
                ));
            }
            let layout_id = u32::try_from(index)
                .map_err(|_| value_error("state layout interner exhausted u32 IDs"))?;
            Ok(StateLayoutIdBinding {
                resource: state.resource.as_str(),
                layout_id,
            })
        })
        .collect::<PyResult<_>>()?;
    canonical_ascii_json(&bindings).map_err(value_error)
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn workload_compilation_wire_cache_key(graph: &str, timeline: &str, options: &str) -> String {
    digest(&format!(
        "workload-compilation-v7\0{graph}\0{timeline}\0{options}"
    ))
}

// Python schedule contracts use ASCII JSON with lexicographically ordered
// object keys. `serde_json::Value` gives us the same stable key ordering;
// normalize non-ASCII scalars explicitly so this remains a closed FFI wire.
fn canonical_ascii_json<T: Serialize>(value: &T) -> Result<String, String> {
    let value = serde_json::to_value(value)
        .map_err(|source| format!("failed to encode canonical JSON: {source}"))?;
    let encoded = serde_json::to_string(&value)
        .map_err(|source| format!("failed to serialize canonical JSON: {source}"))?;
    let mut canonical = String::with_capacity(encoded.len());
    for character in encoded.chars() {
        if character.is_ascii() {
            canonical.push(character);
            continue;
        }
        let scalar = character as u32;
        if scalar <= 0xffff {
            write!(&mut canonical, "\\u{scalar:04x}").expect("writing to String cannot fail");
            continue;
        }
        let surrogate = scalar - 0x1_0000;
        let high = 0xd800 + (surrogate >> 10);
        let low = 0xdc00 + (surrogate & 0x3ff);
        write!(&mut canonical, "\\u{high:04x}\\u{low:04x}").expect("writing to String cannot fail");
    }
    Ok(canonical)
}

fn parse_static_backend_binding_manifest(
    payload: &str,
    label: &str,
) -> PyResult<Vec<StaticBackendBinding>> {
    let bindings: Vec<StaticBackendBinding> = decode(payload, label)?;
    let mut previous: Option<(&str, &str)> = None;
    for binding in &bindings {
        if binding.workload_id.is_empty() || binding.factory.is_empty() {
            return Err(value_error(
                "static backend bindings require non-empty workload_id and factory",
            ));
        }
        if !binding.config.is_object() {
            return Err(value_error(
                "static backend binding config must be a JSON object",
            ));
        }
        let identity = (binding.workload_id.as_str(), binding.factory.as_str());
        if previous.is_some_and(|prior| prior >= identity) {
            return Err(value_error(
                "static backend bindings must be sorted with unique workload/factory identities",
            ));
        }
        previous = Some(identity);
        let mut specialization_names = BTreeSet::new();
        let mut prior_specialization_name: Option<&str> = None;
        for (name, value) in &binding.specialization_key {
            if name.is_empty()
                || value.is_empty()
                || !specialization_names.insert(name.as_str())
                || prior_specialization_name.is_some_and(|prior| prior >= name.as_str())
            {
                return Err(value_error(
                    "static backend binding specialization_key requires sorted unique non-empty entries",
                ));
            }
            prior_specialization_name = Some(name);
        }
        let mut implementation_ids = BTreeSet::new();
        for implementation_id in &binding.backend_region_implementation_ids {
            if implementation_id.is_empty()
                || !implementation_ids.insert(implementation_id.as_str())
            {
                return Err(value_error(
                    "static backend binding implementation IDs require sorted unique non-empty entries",
                ));
            }
        }
        if implementation_ids.iter().copied().collect::<Vec<_>>()
            != binding
                .backend_region_implementation_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        {
            return Err(value_error(
                "static backend binding implementation IDs require sorted unique non-empty entries",
            ));
        }
    }
    Ok(bindings)
}

fn validate_static_backend_binding_manifest(
    options: &CompileOptions,
) -> PyResult<Vec<StaticBackendBinding>> {
    parse_static_backend_binding_manifest(
        &options.static_binding_manifest,
        "CompileOptions static_binding_manifest",
    )
}

/// Validate the closed portions of CompileOptions which do not require a
/// schedule graph or backend timeline.  Cold specialization calls this before
/// any Python resource callback, so malformed signed options cannot allocate
/// local state before the compiler rejects them.
fn validate_compile_options_preflight(options: &CompileOptions) -> PyResult<()> {
    if !matches!(options.schema_version, 1 | 2) {
        return Err(value_error("unsupported schedule schema version"));
    }
    if options.schema_version == 1
        && options
            .packed_hidden_resource_manifest
            .as_ref()
            .is_some_and(|manifest| !manifest.is_empty())
    {
        return Err(value_error(
            "CompileOptions schema v1 requires an empty packed_hidden_resource_manifest",
        ));
    }
    if options.schema_version == 2 && options.packed_hidden_resource_manifest.is_none() {
        return Err(value_error(
            "CompileOptions schema v2 requires an explicit packed_hidden_resource_manifest",
        ));
    }
    validate_static_backend_binding_manifest(options)?;
    if !matches!(options.compile_policy.as_str(), "torch_compile" | "eager")
        || !matches!(
            options.capture_policy.as_str(),
            "auto" | "disabled" | "require"
        )
    {
        return Err(value_error("unsupported Region compile or capture policy"));
    }
    if !options.compiler_fingerprint.is_empty()
        && (options.compiler_fingerprint.len() != 64
            || options
                .compiler_fingerprint
                .chars()
                .any(|character| !character.is_ascii_hexdigit() || character.is_ascii_uppercase()))
    {
        return Err(value_error(
            "compiler fingerprint must be lowercase SHA-256",
        ));
    }
    let shape_keys: Vec<&str> = options
        .shape_specialization
        .iter()
        .map(|(key, _value)| key.as_str())
        .collect();
    if shape_keys.windows(2).any(|pair| pair[0] >= pair[1])
        || options
            .shape_specialization
            .iter()
            .any(|(key, value)| key.is_empty() || value.is_empty())
    {
        return Err(value_error(
            "shape specialization requires sorted unique non-empty keys",
        ));
    }
    for binding in &options.residual_resource_manifest {
        if binding.module_path.is_empty()
            || binding.site == HiddenSite::None
            || binding.phase == HiddenPhase::None
        {
            return Err(value_error(
                "residual resource manifest entries require a typed locator and module_path",
            ));
        }
    }
    let locator_keys: Vec<_> = options
        .residual_resource_manifest
        .iter()
        .map(ResidualResourceBinding::locator_key)
        .collect();
    if locator_keys.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(value_error(
            "residual resource manifest requires sorted unique locators",
        ));
    }
    let packed_manifest = options
        .packed_hidden_resource_manifest
        .as_deref()
        .unwrap_or(&[]);
    let mut packed_locator_keys = Vec::with_capacity(packed_manifest.len());
    for binding in packed_manifest {
        let key = packed_hidden_locator_key(&binding.layer, binding.layer_kind, binding.site)?;
        if binding.module_path.is_empty() {
            return Err(value_error(
                "packed hidden resource manifest entries require a non-empty module_path",
            ));
        }
        packed_locator_keys.push(key);
    }
    if packed_locator_keys
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(value_error(
            "packed hidden resource manifest requires sorted unique locators",
        ));
    }
    Ok(())
}

fn effective_budget(options: &CompileOptions, timeline: &Timeline) -> u64 {
    match (options.memory_budget_bytes, timeline.static_headroom_bytes) {
        (0, 0) => u64::MAX,
        (0, headroom) => headroom,
        (configured, 0) => configured,
        (configured, headroom) => configured.min(headroom),
    }
}

fn is_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    matches!(characters.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn is_bound_atom(value: &str) -> bool {
    value.parse::<i64>().is_ok() || is_identifier(value)
}

fn validate_affine_task(task: &ScheduleTask) -> PyResult<()> {
    let domain = &task.domain;
    if domain.indices.is_empty() {
        if !domain.bounds.is_empty() || !domain.parameters.is_empty() || !task.accesses.is_empty() {
            return Err(value_error(
                "scalar affine domains cannot declare bounds, parameters, or accesses",
            ));
        }
        return Ok(());
    }
    if domain.name.is_empty() || !is_identifier(&domain.name) {
        return Err(value_error("affine domain names must be identifiers"));
    }
    let indices: BTreeSet<&str> = domain.indices.iter().map(String::as_str).collect();
    let parameters: BTreeSet<&str> = domain.parameters.iter().map(String::as_str).collect();
    if indices.len() != domain.indices.len()
        || parameters.len() != domain.parameters.len()
        || domain.indices.iter().any(|value| !is_identifier(value))
        || domain.parameters.iter().any(|value| !is_identifier(value))
        || !indices.is_disjoint(&parameters)
    {
        return Err(value_error(
            "affine domain indices and parameters must be unique disjoint identifiers",
        ));
    }
    let mut bounded = BTreeSet::new();
    for (index, lower, upper) in &domain.bounds {
        if !indices.contains(index.as_str())
            || !bounded.insert(index.as_str())
            || !is_bound_atom(lower)
            || !is_bound_atom(upper)
        {
            return Err(value_error(
                "affine domains require one simple Presburger bound per index",
            ));
        }
    }
    if bounded != indices {
        return Err(value_error(
            "affine domains require one bound for every index",
        ));
    }
    for access in &task.accesses {
        if access.value.is_empty()
            || !matches!(access.mode.as_str(), "read" | "write" | "read_write")
            || access.indices.len() != domain.indices.len()
        {
            return Err(value_error("invalid affine access descriptor"));
        }
        for expression in &access.indices {
            let mut names = BTreeSet::new();
            for (name, coefficient) in &expression.coefficients {
                if *coefficient == 0
                    || !names.insert(name.as_str())
                    || (!indices.contains(name.as_str()) && !parameters.contains(name.as_str()))
                {
                    return Err(value_error(
                        "affine expressions require unique non-zero known coefficients",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn affine_components(
    graph: &ScheduleGraph,
    task_index: &HashMap<String, usize>,
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![BTreeSet::new(); graph.tasks.len()];
    for edge in &graph.edges {
        let source = task_index[&edge.source];
        let target = task_index[&edge.target];
        adjacency[source].insert(target);
        adjacency[target].insert(source);
    }
    let mut tasks_by_value: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, task) in graph.tasks.iter().enumerate() {
        for access in &task.accesses {
            let tasks = tasks_by_value.entry(access.value.as_str()).or_default();
            if tasks.last().copied() != Some(index) {
                tasks.push(index);
            }
        }
    }
    for tasks in tasks_by_value.values() {
        for left in tasks {
            for right in tasks {
                if left != right {
                    adjacency[*left].insert(*right);
                }
            }
        }
    }

    let mut visited = vec![false; graph.tasks.len()];
    let mut components = Vec::new();
    for root in 0..graph.tasks.len() {
        if visited[root] {
            continue;
        }
        let mut queue = VecDeque::from([root]);
        visited[root] = true;
        let mut component = Vec::new();
        while let Some(index) = queue.pop_front() {
            component.push(index);
            for neighbor in adjacency[index].iter().copied() {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        component.sort_unstable();
        if component
            .iter()
            .all(|index| !graph.tasks[*index].domain.indices.is_empty())
        {
            components.push(component);
        }
    }
    components
}

fn affine_expression(expression: &AffineExpr, variables: &HashMap<&str, String>) -> String {
    let mut terms = Vec::new();
    for (name, coefficient) in &expression.coefficients {
        let variable = variables
            .get(name.as_str())
            .cloned()
            .unwrap_or_else(|| name.clone());
        terms.push(format!("({coefficient} * {variable})"));
    }
    if expression.constant != 0 || terms.is_empty() {
        terms.push(expression.constant.to_string());
    }
    terms.join(" + ")
}

fn relation_tuple(task_index: usize, variables: &[String]) -> String {
    format!("T{task_index}[{}]", variables.join(", "))
}

fn relation_variables<'a>(
    task: &'a ScheduleTask,
    prefix: &str,
) -> (Vec<String>, HashMap<&'a str, String>) {
    let variables: Vec<String> = (0..task.domain.indices.len())
        .map(|ordinal| format!("{prefix}{ordinal}"))
        .collect();
    let mapping = task
        .domain
        .indices
        .iter()
        .map(String::as_str)
        .zip(variables.iter().cloned())
        .collect();
    (variables, mapping)
}

/// Derive value-flow dependences with isl's must-source analysis.
///
/// Reads become the sink, writes the must-source, and the schedule orders
/// statements by graph task index and instances lexicographically (zero
/// padded to a common rank).  `compute_flow` then keeps only true
/// must-source dependences: a write killed by a later write before any read
/// contributes nothing, which the syntactic all-pairs pairing cannot see.
/// Returns the dependence relations in isl's normalized text, or `None` when
/// the accesses are not Presburger-representable for the analysis (the
/// caller then falls back to the syntactic pairing).
fn affine_flow_dependences(
    graph: &ScheduleGraph,
    component: &[usize],
    prefix: &str,
) -> Option<String> {
    let mut reads: Vec<String> = Vec::new();
    let mut writes: Vec<String> = Vec::new();
    let mut schedule_entries: Vec<String> = Vec::new();
    let max_rank = component
        .iter()
        .map(|index| graph.tasks[*index].domain.indices.len())
        .max()
        .unwrap_or(0);
    for &index in component {
        let task = &graph.tasks[index];
        let bounds_by_index: HashMap<&str, (&str, &str)> = task
            .domain
            .bounds
            .iter()
            .map(|(name, lower, upper)| (name.as_str(), (lower.as_str(), upper.as_str())))
            .collect();
        let variables: Vec<String> = task.domain.indices.clone();
        let mapping: HashMap<&str, String> = task
            .domain
            .indices
            .iter()
            .map(|name| (name.as_str(), name.clone()))
            .collect();
        let mut constraints = Vec::new();
        for name in &task.domain.indices {
            let (lower, upper) = bounds_by_index[name.as_str()];
            constraints.push(format!("{lower} <= {name}"));
            constraints.push(format!("{name} < {upper}"));
        }
        let domain_condition = constraints.join(" and ");
        // isl map text carries constraints at the end: `{ S[i] -> A[i] : c }`.
        let suffix = if domain_condition.is_empty() {
            String::new()
        } else {
            format!(" : {domain_condition}")
        };
        let tuple = relation_tuple(index, &variables);
        let mut scatter: Vec<String> = vec![index.to_string()];
        scatter.extend(variables.iter().cloned());
        while scatter.len() < 1 + max_rank {
            scatter.push("0".to_owned());
        }
        schedule_entries.push(format!("{tuple} -> [{}]{suffix}", scatter.join(", ")));
        for access in &task.accesses {
            let elements = access
                .indices
                .iter()
                .map(|expression| affine_expression(expression, &mapping))
                .collect::<Vec<_>>()
                .join(", ");
            let entry = format!("{tuple} -> {}[{elements}]{suffix}", access.value);
            match access.mode.as_str() {
                "read" => reads.push(entry),
                "write" => writes.push(entry),
                "read_write" => {
                    reads.push(entry.clone());
                    writes.push(entry);
                }
                _ => {}
            }
        }
    }
    if reads.is_empty() || writes.is_empty() {
        // With no producer/consumer pair the syntactic pairing is empty too.
        return Some(String::new());
    }
    let context = Context::alloc();
    let sink =
        UnionMap::read_from_str(&context, &format!("{prefix}{{ {} }}", reads.join("; "))).ok()?;
    let must_source =
        UnionMap::read_from_str(&context, &format!("{prefix}{{ {} }}", writes.join("; "))).ok()?;
    let schedule_map = UnionMap::read_from_str(
        &context,
        &format!("{prefix}{{ {} }}", schedule_entries.join("; ")),
    )
    .ok()?;
    let flow = UnionAccessInfo::from_sink(sink)
        .ok()?
        .set_must_source(must_source)
        .ok()?
        .set_schedule_map(schedule_map)
        .ok()?
        .compute_flow()
        .ok()?;
    let must_dependence = flow.get_must_dependence().ok()?;
    let text = must_dependence.to_str().ok()?.to_owned();
    let open = text.find('{')?;
    let inner = text[open + 1..]
        .trim()
        .strip_suffix('}')
        .unwrap_or(text[open + 1..].trim())
        .trim()
        .to_owned();
    Some(inner)
}

fn affine_problem_strings(
    graph: &ScheduleGraph,
    component: &[usize],
) -> PyResult<(String, String)> {
    let component_set: BTreeSet<usize> = component.iter().copied().collect();
    let mut parameters = BTreeSet::new();
    let mut domains = Vec::new();
    for index in component {
        let task = &graph.tasks[*index];
        parameters.extend(task.domain.parameters.iter().cloned());
        let bounds_by_index: HashMap<&str, (&str, &str)> = task
            .domain
            .bounds
            .iter()
            .map(|(name, lower, upper)| (name.as_str(), (lower.as_str(), upper.as_str())))
            .collect();
        let mut constraints = Vec::new();
        for name in &task.domain.indices {
            let (lower, upper) = bounds_by_index[name.as_str()];
            if is_identifier(lower) && !task.domain.indices.iter().any(|index| index == lower) {
                parameters.insert(lower.to_owned());
            }
            if is_identifier(upper) && !task.domain.indices.iter().any(|index| index == upper) {
                parameters.insert(upper.to_owned());
            }
            constraints.push(format!("{lower} <= {name}"));
            constraints.push(format!("{name} < {upper}"));
        }
        domains.push(format!(
            "T{index}[{}] : {}",
            task.domain.indices.join(", "),
            constraints.join(" and ")
        ));
    }
    let prefix = if parameters.is_empty() {
        String::new()
    } else {
        format!(
            "[{}] -> ",
            parameters.into_iter().collect::<Vec<_>>().join(", ")
        )
    };
    let domain = format!("{prefix}{{ {} }}", domains.join("; "));

    let mut relations = BTreeSet::new();
    let index_by_id: HashMap<&str, usize> = graph
        .tasks
        .iter()
        .enumerate()
        .map(|(index, task)| (task.task_id.as_str(), index))
        .collect();
    for edge in &graph.edges {
        let source = index_by_id[edge.source.as_str()];
        let target = index_by_id[edge.target.as_str()];
        if !component_set.contains(&source) || !component_set.contains(&target) {
            continue;
        }
        let source_task = &graph.tasks[source];
        let target_task = &graph.tasks[target];
        let (source_variables, source_mapping) = relation_variables(source_task, "s");
        let (target_variables, target_mapping) = relation_variables(target_task, "d");
        let constraints: Vec<String> = source_task
            .domain
            .indices
            .iter()
            .filter_map(|name| {
                target_mapping.get(name.as_str()).map(|target_variable| {
                    format!("{} = {target_variable}", source_mapping[name.as_str()])
                })
            })
            .collect();
        let condition = if constraints.is_empty() {
            String::new()
        } else {
            format!(" : {}", constraints.join(" and "))
        };
        relations.insert(format!(
            "{} -> {}{}",
            relation_tuple(source, &source_variables),
            relation_tuple(target, &target_variables),
            condition
        ));
    }

    match affine_flow_dependences(graph, component, &prefix) {
        Some(inner) => {
            // isl must-source flow analysis derived the value dependences.
            if !inner.is_empty() {
                relations.insert(inner);
            }
        }
        None => {
            // Fallback: syntactic all-pairs write/read pairing for accesses
            // the flow analysis cannot represent.
            for source in component {
                let source_task = &graph.tasks[*source];
                for source_access in source_task
                    .accesses
                    .iter()
                    .filter(|access| matches!(access.mode.as_str(), "write" | "read_write"))
                {
                    for target in component {
                        let target_task = &graph.tasks[*target];
                        for target_access in target_task.accesses.iter().filter(|access| {
                            access.value == source_access.value
                                && matches!(access.mode.as_str(), "read" | "read_write")
                        }) {
                            if source_access.indices.len() != target_access.indices.len() {
                                return Err(value_error(format!(
                                    "affine value {:?} has inconsistent access rank",
                                    source_access.value
                                )));
                            }
                            let (source_variables, source_mapping) =
                                relation_variables(source_task, "s");
                            let (target_variables, target_mapping) =
                                relation_variables(target_task, "d");
                            let constraints: Vec<String> = source_access
                                .indices
                                .iter()
                                .zip(&target_access.indices)
                                .map(|(source_expression, target_expression)| {
                                    format!(
                                        "{} = {}",
                                        affine_expression(source_expression, &source_mapping),
                                        affine_expression(target_expression, &target_mapping)
                                    )
                                })
                                .collect();
                            relations.insert(format!(
                                "{} -> {} : {}",
                                relation_tuple(*source, &source_variables),
                                relation_tuple(*target, &target_variables),
                                constraints.join(" and ")
                            ));
                        }
                    }
                }
            }
        }
    }
    let dependences = format!(
        "{prefix}{{ {} }}",
        relations.into_iter().collect::<Vec<_>>().join("; ")
    );
    Ok((domain, dependences))
}

fn isl_schedule_error(label: &str, error: impl std::fmt::Display) -> PyErr {
    schedule_error(format!("{label}: {error}"))
}

fn compile_affine_regions(
    graph: &ScheduleGraph,
    task_index: &HashMap<String, usize>,
) -> PyResult<Vec<AffineRegionSchedule>> {
    let mut regions = Vec::new();
    for component in affine_components(graph, task_index) {
        let (domain_text, dependence_text) = affine_problem_strings(graph, &component)?;
        let context = Context::alloc();
        let domain = UnionSet::read_from_str(&context, &domain_text)
            .map_err(|error| isl_schedule_error("invalid affine domain", error))?;
        let dependences = UnionMap::read_from_str(&context, &dependence_text)
            .map_err(|error| isl_schedule_error("invalid affine dependences", error))?;
        let constraints =
            ScheduleConstraints::on_domain(domain)
                .map_err(|error| isl_schedule_error("failed to create isl constraints", error))?
                .set_validity(dependences.copy().map_err(|error| {
                    isl_schedule_error("failed to copy affine dependences", error)
                })?)
                .map_err(|error| isl_schedule_error("failed to set affine validity", error))?
                .set_coincidence(dependences.copy().map_err(|error| {
                    isl_schedule_error("failed to copy affine dependences", error)
                })?)
                .map_err(|error| isl_schedule_error("failed to set affine coincidence", error))?
                .set_proximity(dependences.copy().map_err(|error| {
                    isl_schedule_error("failed to copy affine dependences", error)
                })?)
                .map_err(|error| isl_schedule_error("failed to set affine proximity", error))?;
        let schedule = constraints
            .compute_schedule()
            .map_err(|error| isl_schedule_error("no legal affine schedule", error))?;
        regions.push(AffineRegionSchedule {
            tasks: component
                .iter()
                .map(|index| graph.tasks[*index].task_id.clone())
                .collect(),
            schedule: schedule
                .to_str()
                .map_err(|error| isl_schedule_error("failed to serialize isl schedule", error))?
                .to_owned(),
            dependences: dependences
                .to_str()
                .map_err(|error| {
                    isl_schedule_error("failed to serialize affine dependences", error)
                })?
                .to_owned(),
        });
    }
    Ok(regions)
}

fn validate_affine_regions(
    graph: &ScheduleGraph,
    task_index: &HashMap<String, usize>,
    tree: &ScheduleTree,
) -> PyResult<()> {
    let components = affine_components(graph, task_index);
    if tree.affine_regions.len() != components.len()
        || (tree.affine_regions.is_empty() && tree.kind != "sequence")
        || (!tree.affine_regions.is_empty() && tree.kind != "hybrid")
    {
        return Err(schedule_error(
            "ScheduleDecision affine region coverage is inconsistent",
        ));
    }
    for (region, component) in tree.affine_regions.iter().zip(components) {
        let expected_tasks: Vec<String> = component
            .iter()
            .map(|index| graph.tasks[*index].task_id.clone())
            .collect();
        if region.tasks != expected_tasks {
            return Err(schedule_error(
                "ScheduleDecision affine region task membership is inconsistent",
            ));
        }
        let (domain_text, dependence_text) = affine_problem_strings(graph, &component)?;
        let context = Context::alloc();
        let expected_domain = UnionSet::read_from_str(&context, &domain_text)
            .map_err(|error| isl_schedule_error("invalid affine domain", error))?;
        let expected_dependences = UnionMap::read_from_str(&context, &dependence_text)
            .map_err(|error| isl_schedule_error("invalid affine dependences", error))?;
        let supplied_dependences = UnionMap::read_from_str(&context, &region.dependences)
            .map_err(|error| isl_schedule_error("invalid decision dependences", error))?;
        if !supplied_dependences
            .is_equal(&expected_dependences)
            .map_err(|error| isl_schedule_error("failed to compare affine dependences", error))?
        {
            return Err(schedule_error(
                "ScheduleDecision affine dependences do not match ScheduleGraph accesses",
            ));
        }
        let supplied_schedule = Schedule::read_from_str(&context, &region.schedule)
            .map_err(|error| isl_schedule_error("invalid decision affine schedule", error))?;
        let supplied_domain = supplied_schedule.get_domain().map_err(|error| {
            isl_schedule_error("failed to inspect affine schedule domain", error)
        })?;
        if !supplied_domain
            .is_equal(&expected_domain)
            .map_err(|error| isl_schedule_error("failed to compare affine domains", error))?
        {
            return Err(schedule_error(
                "ScheduleDecision affine schedule has the wrong domain",
            ));
        }
        let schedule_map = supplied_schedule
            .get_map()
            .map_err(|error| isl_schedule_error("failed to inspect affine schedule map", error))?;
        let lexicographically_before = schedule_map
            .copy()
            .map_err(|error| isl_schedule_error("failed to copy affine schedule map", error))?
            .lex_lt_union_map(
                schedule_map.copy().map_err(|error| {
                    isl_schedule_error("failed to copy affine schedule map", error)
                })?,
            )
            .map_err(|error| isl_schedule_error("failed to order affine schedule map", error))?;
        if !expected_dependences
            .is_subset(&lexicographically_before)
            .map_err(|error| isl_schedule_error("failed to validate affine schedule", error))?
        {
            return Err(schedule_error(
                "ScheduleDecision affine schedule violates a Presburger dependence",
            ));
        }
    }
    Ok(())
}

const TORCH_EXPORT_CAPABILITY: &str = "torch.export";
const SIDE_STREAM_SAFE_CAPABILITY: &str = "runtime.side_stream_safe";
const ASYNC_HOST_SAFE_CAPABILITY: &str = "runtime.async_host_safe";
const PACKED_REQUEST_ROWS_CAPABILITY: &str = "runtime/packed_phase_rows@1";
const PACKED_SAMPLER_ROWS_CAPABILITY: &str = "runtime/packed_sampler_rows@1";
const PACKED_HIDDEN_CAPABILITY: &str = "hidden/block_output/packed@1";
const CAPTURE_STABLE_STATE_TRANSITION: &str = "optimizer";

/// CUDA graph replay requires every captured state read to retain the same
/// storage address.  The native optimizer contract updates parameters in
/// place; replacement, publication, and custom transition contracts do not
/// provide that guarantee.  Request slots are selected per invocation, so
/// their binding is deliberately never capture-stable.
fn state_has_capture_stable_storage(graph: &ScheduleGraph, resource: &str) -> bool {
    graph.states.iter().any(|state| {
        state.resource == resource
            && state.scope != StateScope::Request
            && state.storage == Some(ScheduleStateStorage::OptimizerStable)
            && state.transition == CAPTURE_STABLE_STATE_TRANSITION
    })
}

fn capture_effects_have_stable_storage(graph: &ScheduleGraph, effects: &[ScheduleEffect]) -> bool {
    effects.iter().all(|effect| {
        effect.kind == ScheduleEffectKind::StateRead
            && state_has_capture_stable_storage(graph, &effect.resource)
    })
}

fn validate_optimizer_transition_task(graph: &ScheduleGraph, task: &ScheduleTask) -> PyResult<()> {
    let optimizer_candidates = task
        .implementation_kinds
        .iter()
        .filter(|kind| **kind == TaskImplementationKind::OptimizerTransition)
        .count();
    if optimizer_candidates == 0 {
        return Ok(());
    }
    let stable_state = task.effects.first().and_then(|effect| {
        graph
            .states
            .iter()
            .find(|state| state.resource == effect.resource)
    });
    let capabilities_are_closed =
        task.implementation_capabilities
            .first()
            .is_some_and(|capabilities| {
                capabilities
                    == &[
                        ASYNC_HOST_SAFE_CAPABILITY.to_owned(),
                        SIDE_STREAM_SAFE_CAPABILITY.to_owned(),
                    ]
            });
    if !matches!(graph.schema_version, 3 | 4 | 5)
        || optimizer_candidates != 1
        || task.implementation_ids.len() != 1
        || task.implementation_kinds.len() != 1
        || !capabilities_are_closed
        || task.capture_safe
        || task.effects.len() != 1
        || task.effects[0].kind != ScheduleEffectKind::StateTransition
        || stable_state.is_none_or(|state| {
            state.scope == StateScope::Request
                || state.storage != Some(ScheduleStateStorage::OptimizerStable)
                || state.transition != CAPTURE_STABLE_STATE_TRANSITION
        })
        || task.inputs.len() != 1
        || task.inputs[0].kind != ScheduleInputKind::Result
        || !task.outputs.is_empty()
        || !task.domain.indices.is_empty()
        || !task.domain.bounds.is_empty()
        || !task.domain.parameters.is_empty()
        || !task.accesses.is_empty()
    {
        return Err(schedule_error(format!(
            "task {:?} optimizer_transition implementation requires one strict optimizer-stable state transition, one Tensor result input, no outputs, and the closed async/side capability evidence",
            task.task_id
        )));
    }
    Ok(())
}

fn backend_capabilities(capabilities: &[String]) -> BTreeSet<String> {
    capabilities
        .iter()
        .filter(|capability| {
            !matches!(
                capability.as_str(),
                TORCH_EXPORT_CAPABILITY | SIDE_STREAM_SAFE_CAPABILITY | ASYNC_HOST_SAFE_CAPABILITY
            )
        })
        .cloned()
        .collect()
}

fn packed_input_capabilities(task: &ScheduleTask) -> BTreeSet<String> {
    let mut required = BTreeSet::new();
    if task
        .inputs
        .iter()
        .any(|input| input.kind == ScheduleInputKind::PackedRequestRows)
    {
        required.insert(PACKED_REQUEST_ROWS_CAPABILITY.to_owned());
    }
    if task
        .inputs
        .iter()
        .any(|input| input.kind == ScheduleInputKind::PackedSamplerRows)
    {
        required.insert(PACKED_SAMPLER_ROWS_CAPABILITY.to_owned());
    }
    if task
        .inputs
        .iter()
        .any(|input| input.kind == ScheduleInputKind::PackedHidden)
    {
        required.insert(PACKED_HIDDEN_CAPABILITY.to_owned());
    }
    required
}

/// Select the first declaration-order implementation that the signed
/// timeline can actually provide.  A candidate's `torch.export` marker is
/// compiler evidence, not a backend requirement; every other capability must
/// be supplied by the timeline.  Schema v1 did not carry per-candidate facts,
/// so it intentionally preserves the historical first-candidate behaviour.
fn select_task_implementations(
    graph: &ScheduleGraph,
    timeline: &Timeline,
) -> PyResult<Vec<SelectedTaskImplementation>> {
    let available: BTreeSet<&str> = timeline.capabilities.iter().map(String::as_str).collect();
    graph
        .tasks
        .iter()
        .map(|task| {
            let input_requirements = packed_input_capabilities(task);
            let has_packed_inputs = task.inputs.iter().any(|input| {
                matches!(
                    input.kind,
                    ScheduleInputKind::PackedRequestRows
                        | ScheduleInputKind::PackedSamplerRows
                        | ScheduleInputKind::PackedHidden
                )
            });
            let candidates = if graph.schema_version == 1 {
                vec![(
                    0usize,
                    task.implementation_kinds[0],
                    task.implementation_ids[0].clone(),
                    task.required_capabilities.clone(),
                    true,
                )]
            } else {
                task.implementation_ids
                    .iter()
                    .enumerate()
                    .map(|(index, implementation_id)| {
                        let capabilities = &task.implementation_capabilities[index];
                        (
                            index,
                            task.implementation_kinds[index],
                            implementation_id.clone(),
                            capabilities.clone(),
                            capabilities
                                .iter()
                                .any(|capability| capability == TORCH_EXPORT_CAPABILITY),
                        )
                    })
                    .collect()
            };
            let unsupported = candidates
                .iter()
                .map(|(_, _, implementation_id, capabilities, _)| {
                    let missing: Vec<String> = backend_capabilities(capabilities)
                        .union(&input_requirements)
                        .cloned()
                        .into_iter()
                        .filter(|capability| !available.contains(capability.as_str()))
                        .collect();
                    format!("{implementation_id:?} requires unavailable capabilities {missing:?}")
                })
                .collect::<Vec<_>>()
                .join("; ");
            candidates
                .into_iter()
                .find_map(
                    |(_candidate_index, kind, implementation_id, capabilities, export_safe)| {
                        let required_capabilities: BTreeSet<String> =
                            backend_capabilities(&capabilities)
                                .union(&input_requirements)
                                .cloned()
                                .collect();
                        required_capabilities
                            .iter()
                            .all(|capability| available.contains(capability.as_str()))
                            .then(|| {
                                let compile_safe = !has_packed_inputs
                                    && kind == TaskImplementationKind::Captured
                                    && (graph.schema_version == 1 || export_safe);
                                let async_host_safe = !has_packed_inputs
                                    && graph.schema_version >= 2
                                    && capabilities
                                        .iter()
                                        .any(|capability| capability == ASYNC_HOST_SAFE_CAPABILITY);
                                SelectedTaskImplementation {
                                    kind,
                                    implementation_id,
                                    required_capabilities: required_capabilities
                                        .into_iter()
                                        .collect(),
                                    capture_safe: compile_safe
                                        && !async_host_safe
                                        && capture_effects_have_stable_storage(
                                            graph,
                                            &task.effects,
                                        ),
                                    compile_safe,
                                    side_stream_safe: !has_packed_inputs
                                        && (compile_safe
                                            || capabilities.iter().any(|capability| {
                                                capability == SIDE_STREAM_SAFE_CAPABILITY
                                            })),
                                    async_host_safe,
                                }
                            })
                    },
                )
                .ok_or_else(|| {
                    schedule_error(format!(
                        "task {:?} has no implementation candidate supported by the timeline: {}",
                        task.task_id, unsupported
                    ))
                })
        })
        .collect()
}

/// Recompute the graph facts that Python historically derived while lowering
/// a workload declaration.  V2 retains those fields on the wire only as
/// signed, inspectable evidence; the scheduler refuses to use them unless
/// they agree with this Rust-owned derivation.
fn validate_derived_graph_semantics(graph: &ScheduleGraph) -> PyResult<()> {
    if matches!(graph.schema_version, 3 | 4 | 5)
        && graph.states.iter().any(|state| state.storage.is_none())
    {
        return Err(schedule_error(
            "ScheduleGraph schema v3+ requires explicit state storage evidence",
        ));
    }
    if graph.schema_version < 3 && graph.states.iter().any(|state| state.storage.is_some()) {
        return Err(schedule_error(
            "ScheduleGraph schema before v3 forbids state storage evidence",
        ));
    }
    if graph.schema_version == 1 {
        // Historical payloads did not carry per-implementation capabilities,
        // so re-derivation would be fabricated.  New lowering always emits
        // schema v2 and is therefore fail-closed below.
        return Ok(());
    }

    let mut expected_effect_edges = BTreeSet::<(String, String)>::new();
    let mut previous_effects = BTreeMap::<String, (String, ScheduleEffectKind)>::new();
    for task in &graph.tasks {
        if task.implementation_capabilities.len() != task.implementation_ids.len() {
            return Err(schedule_error(format!(
                "task {:?} schema v2+ requires per-implementation capabilities",
                task.task_id
            )));
        }
        validate_optimizer_transition_task(graph, task)?;
        let mut required_by_all_candidates: Option<BTreeSet<String>> = None;
        for capabilities in &task.implementation_capabilities {
            let mut seen = BTreeSet::<&str>::new();
            for capability in capabilities {
                if capability.is_empty() || !seen.insert(capability.as_str()) {
                    return Err(schedule_error(format!(
                        "task {:?} has invalid implementation capability evidence",
                        task.task_id
                    )));
                }
            }
            let canonical: Vec<&str> = capabilities.iter().map(String::as_str).collect();
            if canonical.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(schedule_error(format!(
                    "task {:?} implementation capability evidence is not sorted",
                    task.task_id
                )));
            }
            let candidate_requirements = backend_capabilities(capabilities);
            required_by_all_candidates = Some(match required_by_all_candidates {
                Some(required) => required
                    .intersection(&candidate_requirements)
                    .cloned()
                    .collect(),
                None => candidate_requirements,
            });
        }
        // A graph-level field can only express requirements shared by every
        // candidate.  Candidate-specific requirements are selected above and
        // emitted only on the resulting ExecutionRegion.
        let mut expected_capability_set = required_by_all_candidates.unwrap_or_default();
        expected_capability_set.extend(packed_input_capabilities(task));
        let expected_capabilities: Vec<String> = expected_capability_set.into_iter().collect();
        if task.required_capabilities != expected_capabilities {
            return Err(schedule_error(format!(
                "task {:?} derived required_capabilities disagree with implementation evidence",
                task.task_id
            )));
        }
        let has_packed_inputs = task.inputs.iter().any(|input| {
            matches!(
                input.kind,
                ScheduleInputKind::PackedRequestRows
                    | ScheduleInputKind::PackedSamplerRows
                    | ScheduleInputKind::PackedHidden
            )
        });
        let expected_capture_safe = !has_packed_inputs
            && capture_effects_have_stable_storage(graph, &task.effects)
            && task
                .implementation_kinds
                .iter()
                .all(|kind| *kind == TaskImplementationKind::Captured)
            && task.implementation_capabilities.iter().all(|capabilities| {
                !capabilities
                    .iter()
                    .any(|capability| capability == ASYNC_HOST_SAFE_CAPABILITY)
            });
        if task.capture_safe != expected_capture_safe {
            return Err(schedule_error(format!(
                "task {:?} derived capture_safe disagrees with implementation, effect, or stable-state evidence",
                task.task_id
            )));
        }

        let mut task_resources = BTreeSet::<&str>::new();
        for effect in &task.effects {
            if !task_resources.insert(effect.resource.as_str()) {
                return Err(schedule_error(format!(
                    "task {:?} has multiple effects for one resource",
                    task.task_id
                )));
            }
            if let Some((previous_task, previous_kind)) = previous_effects.get(&effect.resource) {
                // transitions (strict or conditional) and extern channels
                // serialize against every adjacent access to the same
                // resource: the signed graph has no ordering proof for
                // either.  Reads may commute.
                if matches!(
                    effect.kind,
                    ScheduleEffectKind::StateTransition
                        | ScheduleEffectKind::StateTransitionMaybe
                        | ScheduleEffectKind::ExternIo
                ) || matches!(
                    *previous_kind,
                    ScheduleEffectKind::StateTransition
                        | ScheduleEffectKind::StateTransitionMaybe
                        | ScheduleEffectKind::ExternIo
                ) {
                    expected_effect_edges.insert((previous_task.clone(), task.task_id.clone()));
                }
            }
            previous_effects.insert(effect.resource.clone(), (task.task_id.clone(), effect.kind));
        }
    }

    let supplied_effect_edges: Vec<(String, String)> = graph
        .edges
        .iter()
        .filter(|edge| edge.kind == ScheduleEdgeKind::Effect)
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect();
    let supplied_effect_set: BTreeSet<(String, String)> =
        supplied_effect_edges.iter().cloned().collect();
    if supplied_effect_edges.len() != supplied_effect_set.len()
        || supplied_effect_set != expected_effect_edges
    {
        return Err(schedule_error(
            "derived effect edges disagree with typed state effects",
        ));
    }
    Ok(())
}

/// Verify that every INDEX hidden input has exactly one signed process-local
/// module binding.  The graph owns locator demand; the manifest merely binds
/// those typed locators to opaque backend module paths.
fn validate_residual_resource_manifest(
    graph: &ScheduleGraph,
    manifest: &[ResidualResourceBinding],
) -> PyResult<()> {
    let mut supplied = BTreeMap::<(i64, &'static str, &'static str), &str>::new();
    let mut previous: Option<(i64, &'static str, &'static str)> = None;
    for binding in manifest {
        let key = binding.locator_key();
        if binding.module_path.is_empty()
            || binding.site == HiddenSite::None
            || binding.phase == HiddenPhase::None
        {
            return Err(value_error(
                "residual resource manifest entries require a typed locator and module_path",
            ));
        }
        if previous.is_some_and(|prior| prior >= key)
            || supplied.insert(key, &binding.module_path).is_some()
        {
            return Err(value_error(
                "residual resource manifest requires sorted unique locators",
            ));
        }
        previous = Some(key);
    }

    let mut required = BTreeSet::<(i64, &'static str, &'static str)>::new();
    for task in &graph.tasks {
        for input in &task.inputs {
            if input.kind != ScheduleInputKind::Hidden || input.layer_kind != HiddenLayerKind::Index
            {
                continue;
            }
            let layer_index = input.layer.parse::<i64>().map_err(|_| {
                value_error("indexed hidden schedule input requires a signed layer index")
            })?;
            required.insert((layer_index, input.site.as_wire(), input.phase.as_wire()));
        }
    }

    let supplied_keys: BTreeSet<(i64, &'static str, &'static str)> =
        supplied.keys().copied().collect();
    let missing: Vec<_> = required.difference(&supplied_keys).collect();
    let unexpected: Vec<_> = supplied_keys.difference(&required).collect();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(schedule_error(format!(
            "residual resource manifest coverage mismatch: missing={missing:?}, unexpected={unexpected:?}",
        )));
    }
    Ok(())
}

/// Require a distinct backend module binding for every packed hidden locator.
/// Ordinary hidden inputs remain exclusively covered by the residual manifest.
fn validate_packed_hidden_resource_manifest(
    graph: &ScheduleGraph,
    manifest: &[PackedHiddenResourceBinding],
) -> PyResult<()> {
    let mut supplied = BTreeMap::<(&str, &'static str, &'static str), &str>::new();
    let mut previous: Option<(&str, &'static str, &'static str)> = None;
    for binding in manifest {
        let key = packed_hidden_locator_key(&binding.layer, binding.layer_kind, binding.site)?;
        if binding.module_path.is_empty() {
            return Err(value_error(
                "packed hidden resource manifest entries require a non-empty module_path",
            ));
        }
        if previous.is_some_and(|prior| prior >= key)
            || supplied.insert(key, &binding.module_path).is_some()
        {
            return Err(value_error(
                "packed hidden resource manifest requires sorted unique locators",
            ));
        }
        previous = Some(key);
    }

    let mut required = BTreeSet::<(&str, &'static str, &'static str)>::new();
    for task in &graph.tasks {
        for input in &task.inputs {
            if input.kind != ScheduleInputKind::PackedHidden {
                continue;
            }
            required.insert(packed_hidden_locator_key(
                &input.layer,
                input.layer_kind,
                input.site,
            )?);
        }
    }

    let supplied_keys: BTreeSet<(&str, &'static str, &'static str)> =
        supplied.keys().copied().collect();
    let missing: Vec<_> = required.difference(&supplied_keys).collect();
    let unexpected: Vec<_> = supplied_keys.difference(&required).collect();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(schedule_error(format!(
            "packed hidden resource manifest coverage mismatch: missing={missing:?}, unexpected={unexpected:?}",
        )));
    }
    Ok(())
}

fn validate_inputs(
    graph: &ScheduleGraph,
    timeline: &Timeline,
    options: &CompileOptions,
) -> PyResult<(HashMap<String, usize>, HashMap<String, usize>)> {
    if !matches!(graph.schema_version, 1 | 2 | 3 | 4 | 5)
        || !matches!(timeline.schema_version, 1 | 2)
    {
        return Err(value_error("unsupported schedule schema version"));
    }
    validate_derived_graph_semantics(graph)?;
    validate_compile_options_preflight(options)?;
    if graph.workload_key.is_empty() || timeline.backend_id.is_empty() {
        return Err(value_error("schedule identities must be non-empty"));
    }
    for (position, entry) in graph.workload_approximations.iter().enumerate() {
        if entry.0.is_empty() || entry.1.is_empty() || entry.1 == "exact" {
            return Err(value_error(
                "schedule graph workload_approximations require non-exact policy entries",
            ));
        }
        if position > 0 && graph.workload_approximations[position - 1] >= *entry {
            return Err(value_error(
                "schedule graph workload_approximations must be sorted unique",
            ));
        }
    }
    if timeline.backend_version.is_empty() {
        return Err(value_error("timeline backend_version must be non-empty"));
    }
    if timeline.runner_family != "mrv1" && timeline.runner_family != "mrv2" {
        return Err(value_error("timeline runner_family must be mrv1 or mrv2"));
    }
    if timeline.graph_mode != "eager"
        && timeline.graph_mode != "piecewise"
        && timeline.graph_mode != "full"
    {
        return Err(value_error("unsupported timeline graph_mode"));
    }
    let mut anchor_index = HashMap::new();
    for (index, anchor) in timeline.anchors.iter().enumerate() {
        if anchor.anchor_id.is_empty() || anchor.ordinal != index {
            return Err(value_error(
                "timeline anchors must have contiguous canonical ordinals",
            ));
        }
        if anchor.graph_boundary && !anchor.observable {
            return Err(value_error("timeline graph boundaries must be observable"));
        }
        if anchor_index
            .insert(anchor.anchor_id.clone(), index)
            .is_some()
        {
            return Err(value_error("timeline anchor ids must be unique"));
        }
        if timeline.schema_version == 2 && anchor.backend_events.is_none() {
            return Err(value_error(
                "timeline schema v2 anchors require explicit backend_events",
            ));
        }
        if let Some(events) = &anchor.backend_events {
            let mut seen = BTreeSet::new();
            if events.iter().any(|event| !seen.insert(*event)) {
                return Err(value_error(
                    "timeline anchor backend_events must not contain duplicates",
                ));
            }
        }
        let _ = &anchor.visible_values;
    }
    for region in &timeline.regions {
        let start = anchor_index.get(&region.start).copied();
        let end = anchor_index.get(&region.end).copied();
        if !region.pinned || start.is_none() || end.is_none() || start >= end {
            return Err(value_error(format!(
                "main workload region {:?} must be pinned with a valid interval",
                region.region_id
            )));
        }
        let _ = (
            region.reservation.memory_bytes,
            region.reservation.compute_units,
            region.reservation.bandwidth_units,
            &region.capture_region,
        );
    }

    let capabilities: BTreeSet<&str> = timeline.capabilities.iter().map(String::as_str).collect();
    let mut state_resources = BTreeSet::new();
    for state in &graph.states {
        let request_key_fields: BTreeSet<RequestKeyField> =
            state.request_key.iter().copied().collect();
        if state.resource.is_empty()
            || state.transition.is_empty()
            || (state.scope == StateScope::Request) != !state.request_key.is_empty()
            || request_key_fields.len() != state.request_key.len()
            || (matches!(graph.schema_version, 3 | 4 | 5) && state.storage.is_none())
            || (graph.schema_version < 3 && state.storage.is_some())
            || !state_resources.insert(state.resource.as_str())
        {
            return Err(value_error("invalid or duplicate typed schedule state"));
        }
    }
    let mut task_index = HashMap::new();
    let mut packed_state_phases: HashMap<&str, HiddenPhase> = HashMap::new();
    for (index, task) in graph.tasks.iter().enumerate() {
        if task.task_id.is_empty() || task.implementation_ids.is_empty() {
            return Err(value_error(
                "schedule tasks need ids and implementation candidates",
            ));
        }
        if task.implementation_ids.len() != task.implementation_kinds.len() {
            return Err(value_error(
                "schedule task implementation ids and kinds are inconsistent",
            ));
        }
        if graph.schema_version == 5
            && task
                .inputs
                .iter()
                .any(|input| input.packed_rows_port.is_none())
        {
            return Err(value_error(
                "schema v5 schedule inputs require an explicit packed_rows_port field",
            ));
        }
        if graph.schema_version < 5
            && task
                .inputs
                .iter()
                .any(|input| input.packed_rows_port.is_some())
        {
            return Err(value_error(
                "schedule schemas before v5 forbid packed_rows_port fields",
            ));
        }
        let packed_rows_inputs: Vec<&ScheduleInput> = task
            .inputs
            .iter()
            .filter(|input| input.kind == ScheduleInputKind::PackedRequestRows)
            .collect();
        let packed_hidden_inputs: Vec<&ScheduleInput> = task
            .inputs
            .iter()
            .filter(|input| input.kind == ScheduleInputKind::PackedHidden)
            .collect();
        let packed_sampler_rows_inputs: Vec<&ScheduleInput> = task
            .inputs
            .iter()
            .filter(|input| input.kind == ScheduleInputKind::PackedSamplerRows)
            .collect();
        let has_packed_relation = task.inputs.iter().any(|input| {
            input
                .packed_rows_port
                .as_deref()
                .is_some_and(|port| !port.is_empty())
        });
        if graph.schema_version < 5
            && (!packed_rows_inputs.is_empty()
                || !packed_hidden_inputs.is_empty()
                || !packed_sampler_rows_inputs.is_empty())
        {
            return Err(value_error(
                "schedule schemas before v5 forbid packed inputs",
            ));
        }
        if graph.schema_version == 5 {
            if packed_rows_inputs.len() > 1 {
                return Err(value_error(
                    "one task may declare only one packed_request_rows input",
                ));
            }
            if (!packed_hidden_inputs.is_empty()
                || !packed_sampler_rows_inputs.is_empty()
                || has_packed_relation)
                && packed_rows_inputs.is_empty()
            {
                return Err(value_error(
                    "packed_rows_port requires one packed_request_rows input",
                ));
            }
            if !packed_rows_inputs.is_empty()
                && task
                    .inputs
                    .iter()
                    .any(|input| input.kind == ScheduleInputKind::Hidden)
            {
                return Err(value_error(
                    "packed task cannot mix packed inputs with ordinary hidden inputs",
                ));
            }
            if let Some(rows) = packed_rows_inputs.first() {
                for input in &task.inputs {
                    let relation = input.packed_rows_port.as_deref().unwrap_or("");
                    if input.kind == ScheduleInputKind::PackedHidden && relation != rows.name {
                        return Err(value_error(
                            "packed_hidden packed_rows_port must name the task's packed_request_rows input",
                        ));
                    }
                    if input.kind == ScheduleInputKind::PackedSamplerRows && relation != rows.name {
                        return Err(value_error(
                            "packed_sampler_rows packed_rows_port must name the task's packed_request_rows input",
                        ));
                    }
                    if input.kind == ScheduleInputKind::StateSelection && relation != rows.name {
                        return Err(value_error(
                            "state_selection packed_rows_port must name the task's packed_request_rows input",
                        ));
                    }
                }
            }
        }
        for effect in &task.effects {
            if effect.resource.is_empty() {
                return Err(value_error("invalid normalized schedule effect"));
            }
            // extern_io names an external channel, not a typed state; every
            // other effect kind must reference a declared state when the
            // graph declares any.
            if effect.kind != ScheduleEffectKind::ExternIo
                && !graph.states.is_empty()
                && !state_resources.contains(effect.resource.as_str())
            {
                return Err(value_error(
                    "normalized schedule effect references an unknown typed state",
                ));
            }
        }
        let mut input_names = BTreeSet::new();
        for input in &task.inputs {
            if input.name.is_empty()
                || input.source.is_empty()
                || !input_names.insert(input.name.as_str())
            {
                return Err(value_error("invalid or duplicate typed schedule input"));
            }
            match input.kind {
                ScheduleInputKind::Hidden
                    if input.layer.is_empty()
                        || input.layer_kind == HiddenLayerKind::None
                        || input.site == HiddenSite::None
                        || input.phase == HiddenPhase::None =>
                {
                    return Err(value_error(
                        "hidden schedule input requires a typed locator",
                    ));
                }
                ScheduleInputKind::PackedHidden
                    if input.layer.is_empty()
                        || !matches!(
                            input.layer_kind,
                            HiddenLayerKind::Index | HiddenLayerKind::Path
                        )
                        || input.site != HiddenSite::BlockOutput
                        || input.phase != HiddenPhase::None =>
                {
                    return Err(value_error(
                        "packed_hidden schedule input requires a block-output locator with phase NONE",
                    ));
                }
                ScheduleInputKind::StateSelection
                    if input.layer_kind != HiddenLayerKind::None
                        || input.site != HiddenSite::None
                        || (input
                            .packed_rows_port
                            .as_deref()
                            .is_some_and(|port| !port.is_empty())
                            != matches!(
                                input.phase,
                                HiddenPhase::Prefill | HiddenPhase::Decode
                            )) =>
                {
                    return Err(value_error(
                        "packed state_selection requires exactly one prefill/decode phase",
                    ));
                }
                ScheduleInputKind::Result
                | ScheduleInputKind::PackedRequestRows
                | ScheduleInputKind::PackedSamplerRows
                | ScheduleInputKind::Provided
                | ScheduleInputKind::Staged
                | ScheduleInputKind::SamplerRequestIds
                | ScheduleInputKind::SamplerSampleIdxs
                | ScheduleInputKind::SamplerSampleIdxTensor
                | ScheduleInputKind::SamplerLogits
                | ScheduleInputKind::SampledTokens
                | ScheduleInputKind::StepId
                    if input.layer_kind != HiddenLayerKind::None
                        || input.site != HiddenSite::None
                        || input.phase != HiddenPhase::None =>
                {
                    return Err(value_error(
                        "only hidden schedule input may declare locator metadata",
                    ));
                }
                ScheduleInputKind::Result if input.output.is_empty() => {
                    return Err(value_error("result schedule input requires an output name"));
                }
                ScheduleInputKind::Provided
                | ScheduleInputKind::PackedRequestRows
                | ScheduleInputKind::PackedSamplerRows
                | ScheduleInputKind::StateSelection
                | ScheduleInputKind::Staged
                | ScheduleInputKind::SamplerRequestIds
                | ScheduleInputKind::SamplerSampleIdxs
                | ScheduleInputKind::SamplerSampleIdxTensor
                | ScheduleInputKind::SamplerLogits
                | ScheduleInputKind::SampledTokens
                | ScheduleInputKind::StepId
                    if !input.output.is_empty() || !input.layer.is_empty() =>
                {
                    return Err(value_error(
                        "provided schedule input contains unrelated locator metadata",
                    ));
                }
                ScheduleInputKind::StepId if input.source != "step_id" => {
                    return Err(value_error(
                        "step_id schedule input requires the built-in step_id source",
                    ));
                }
                ScheduleInputKind::PackedRequestRows
                    if input.source != "runtime.packed_phase_rows" =>
                {
                    return Err(value_error(
                        "packed_request_rows input requires runtime.packed_phase_rows",
                    ));
                }
                ScheduleInputKind::PackedSamplerRows
                    if input.source != "runtime.packed_sampler_rows" =>
                {
                    return Err(value_error(
                        "packed_sampler_rows input requires runtime.packed_sampler_rows",
                    ));
                }
                _ => {}
            }
            let relation = input.packed_rows_port.as_deref().unwrap_or("");
            if input.kind == ScheduleInputKind::PackedRequestRows && !relation.is_empty() {
                return Err(value_error(
                    "packed_request_rows input must not declare packed_rows_port",
                ));
            }
            if !matches!(
                input.kind,
                ScheduleInputKind::PackedHidden
                    | ScheduleInputKind::PackedSamplerRows
                    | ScheduleInputKind::StateSelection
            ) && !relation.is_empty()
            {
                return Err(value_error(
                    "only packed_hidden, packed_sampler_rows, or state_selection may declare packed_rows_port",
                ));
            }
            if input.kind == ScheduleInputKind::StateSelection && !relation.is_empty() {
                if packed_state_phases
                    .get(input.source.as_str())
                    .is_some_and(|phase| *phase != input.phase)
                {
                    return Err(value_error(
                        "one packed request state cannot mix row phases",
                    ));
                }
                packed_state_phases.insert(input.source.as_str(), input.phase);
            }
            if input.kind == ScheduleInputKind::Hidden {
                let capability = hidden_locator_capability(input.site, input.phase)
                    .ok_or_else(|| value_error("hidden schedule input requires a typed locator"))?;
                if !capabilities.contains(capability) {
                    return Err(schedule_error(format!(
                        "hidden schedule input requires unavailable timeline capability {:?}",
                        capability
                    )));
                }
            }
            if input.kind == ScheduleInputKind::PackedRequestRows
                && !capabilities.contains(PACKED_REQUEST_ROWS_CAPABILITY)
            {
                return Err(schedule_error(
                    "packed_request_rows input requires runtime/packed_phase_rows@1",
                ));
            }
            if input.kind == ScheduleInputKind::PackedHidden
                && !capabilities.contains(PACKED_HIDDEN_CAPABILITY)
            {
                return Err(schedule_error(
                    "packed_hidden input requires hidden/block_output/packed@1",
                ));
            }
            if input.kind == ScheduleInputKind::PackedSamplerRows
                && !capabilities.contains(PACKED_SAMPLER_ROWS_CAPABILITY)
            {
                return Err(schedule_error(
                    "packed_sampler_rows input requires runtime/packed_sampler_rows@1",
                ));
            }
        }
        if matches!(graph.schema_version, 4 | 5) {
            let mut request_effect_resources = BTreeSet::new();
            for effect in &task.effects {
                let selection_port = effect.selection_port.as_deref().ok_or_else(|| {
                    value_error(
                        "schema v4+ schedule effects require an explicit selection_port field",
                    )
                })?;
                if effect.kind == ScheduleEffectKind::ExternIo {
                    if !selection_port.is_empty() {
                        return Err(value_error(
                            "extern_io schedule effects forbid a state selection port",
                        ));
                    }
                    continue;
                }
                let state = graph
                    .states
                    .iter()
                    .find(|state| state.resource == effect.resource)
                    .ok_or_else(|| {
                        value_error("schema v4+ state effect references an unknown typed state")
                    })?;
                if state.scope == StateScope::Request {
                    if !request_effect_resources.insert(effect.resource.as_str()) {
                        return Err(value_error(
                            "one task cannot carry multiple effects on one request state",
                        ));
                    }
                    if !matches!(
                        effect.kind,
                        ScheduleEffectKind::StateRead | ScheduleEffectKind::StateTransition
                    ) {
                        return Err(value_error(
                            "request-scoped schedule effects support only read and strict transition",
                        ));
                    }
                    let selection = task
                        .inputs
                        .iter()
                        .find(|input| input.name == selection_port);
                    if selection.is_none_or(|input| {
                        input.kind != ScheduleInputKind::StateSelection
                            || input.source != effect.resource
                    }) {
                        return Err(value_error(
                            "request-scoped schedule effect requires its exact state_selection task input",
                        ));
                    }
                } else if !selection_port.is_empty() {
                    return Err(value_error(
                        "non-request schedule effects forbid a state selection port",
                    ));
                }
            }
            for input in task
                .inputs
                .iter()
                .filter(|input| input.kind == ScheduleInputKind::StateSelection)
            {
                let state = graph
                    .states
                    .iter()
                    .find(|state| state.resource == input.source)
                    .ok_or_else(|| {
                        value_error("state_selection input references an unknown typed state")
                    })?;
                if state.scope != StateScope::Request
                    || task
                        .effects
                        .iter()
                        .filter(|effect| {
                            effect.kind != ScheduleEffectKind::ExternIo
                                && effect.resource == input.source
                                && effect.selection_port.as_deref() == Some(input.name.as_str())
                        })
                        .count()
                        != 1
                {
                    return Err(value_error(
                        "state_selection input must bind exactly one request-scoped schedule effect",
                    ));
                }
            }
        } else if task
            .effects
            .iter()
            .any(|effect| effect.selection_port.is_some())
        {
            return Err(value_error(
                "schedule schemas before v4 forbid state selection ports",
            ));
        } else if task
            .inputs
            .iter()
            .any(|input| input.kind == ScheduleInputKind::StateSelection)
        {
            return Err(value_error(
                "schedule schemas before v4 forbid state_selection inputs",
            ));
        }
        let mut output_names = BTreeSet::new();
        for output in &task.outputs {
            if output.name.is_empty()
                || output.target.is_empty()
                || !output_names.insert(output.name.as_str())
            {
                return Err(value_error("invalid or duplicate typed schedule output"));
            }
        }
        if task_index.insert(task.task_id.clone(), index).is_some() {
            return Err(value_error("schedule task ids must be unique"));
        }
        let after = anchor_index.get(&task.timing.after).copied();
        let before = anchor_index.get(&task.timing.before).copied();
        if after.is_none() || before.is_none() || after >= before {
            return Err(schedule_error(format!(
                "task {:?} has no legal timeline interval",
                task.task_id
            )));
        }
        if task
            .inputs
            .iter()
            .any(|input| input.kind == ScheduleInputKind::StateSelection)
        {
            let ready = state_selection_ready_index(timeline).ok_or_else(|| {
                schedule_error(
                    "state_selection input requires an observable prepare_inputs.post event",
                )
            })?;
            if ready >= before.expect("validated task deadline") {
                return Err(schedule_error(format!(
                    "task {:?} has no legal interval after request identities become available",
                    task.task_id
                )));
            }
        }
        // V2 records only capability requirements common to all candidates;
        // candidate-specific requirements are checked by the Rust selector
        // after the whole graph has been validated.  V1 has no candidate
        // evidence, so retain its historical task-level capability gate.
        if graph.schema_version == 1 {
            for capability in &task.required_capabilities {
                if !capabilities.contains(capability.as_str()) {
                    return Err(schedule_error(format!(
                        "task {:?} requires unavailable capability {:?}",
                        task.task_id, capability
                    )));
                }
            }
        }
        validate_affine_task(task)?;
    }
    // Validate resource coverage only after every indexed locator has passed
    // the backend-capability gate, preserving the more specific diagnostic.
    validate_residual_resource_manifest(graph, &options.residual_resource_manifest)?;
    validate_packed_hidden_resource_manifest(
        graph,
        options
            .packed_hidden_resource_manifest
            .as_deref()
            .unwrap_or(&[]),
    )?;
    for edge in &graph.edges {
        if !task_index.contains_key(&edge.source) || !task_index.contains_key(&edge.target) {
            return Err(value_error("schedule edge references an unknown task"));
        }
    }
    let mut values_by_output = BTreeMap::new();
    for value in &graph.values {
        if value.value_id.is_empty() || value.version < 1 {
            return Err(value_error(
                "schedule values require an id and positive version",
            ));
        }
        if !value.producer.is_empty() && !task_index.contains_key(&value.producer) {
            return Err(value_error("schedule value references an unknown producer"));
        }
        if value
            .consumers
            .iter()
            .any(|consumer| !task_index.contains_key(consumer))
        {
            return Err(value_error("schedule value references an unknown consumer"));
        }
        if !value.producer.is_empty()
            && !value.producer_output.is_empty()
            && values_by_output
                .insert(
                    (value.producer.as_str(), value.producer_output.as_str()),
                    value,
                )
                .is_some()
        {
            return Err(value_error(
                "schedule values contain duplicate producer output identities",
            ));
        }
        let _ = (&value.alias_set, value.size_bytes);
    }
    for task in &graph.tasks {
        for input in &task.inputs {
            if input.kind != ScheduleInputKind::Result {
                continue;
            }
            let Some(value) = values_by_output.get(&(input.source.as_str(), input.output.as_str()))
            else {
                return Err(value_error(
                    "result schedule input does not resolve to a typed value",
                ));
            };
            if !value.consumers.contains(&task.task_id) {
                return Err(value_error(
                    "result schedule input is missing its typed value consumer edge",
                ));
            }
        }
    }
    Ok((anchor_index, task_index))
}

fn dependency_pairs(graph: &ScheduleGraph) -> BTreeMap<(String, String), ScheduleEdgeKind> {
    let mut dependencies = BTreeMap::new();
    for edge in &graph.edges {
        let key = (edge.source.clone(), edge.target.clone());
        let kind = dependencies.entry(key).or_insert_with(|| edge.kind.clone());
        if edge.kind == ScheduleEdgeKind::Effect {
            *kind = edge.kind.clone();
        }
    }
    for value in &graph.values {
        if value.producer.is_empty() {
            continue;
        }
        for consumer in &value.consumers {
            dependencies
                .entry((value.producer.clone(), consumer.clone()))
                .or_insert(ScheduleEdgeKind::Data);
        }
    }
    dependencies
}

fn topological_order(
    graph: &ScheduleGraph,
    task_index: &HashMap<String, usize>,
) -> PyResult<Vec<usize>> {
    let count = graph.tasks.len();
    let mut indegree = vec![0usize; count];
    let mut successors = vec![BTreeSet::<usize>::new(); count];
    for ((source_id, target_id), _kind) in dependency_pairs(graph) {
        let source = task_index[&source_id];
        let target = task_index[&target_id];
        if successors[source].insert(target) {
            indegree[target] += 1;
        }
    }
    let mut ready = BTreeSet::new();
    for (index, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            ready.insert(index);
        }
    }
    let mut order = Vec::with_capacity(count);
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for successor in successors[index].iter().copied() {
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready.insert(successor);
            }
        }
    }
    if order.len() != count {
        return Err(schedule_error("ScheduleGraph contains a dependency cycle"));
    }
    Ok(order)
}

fn peak_memory(
    graph: &ScheduleGraph,
    timeline: &Timeline,
    slots: &[usize],
    task_index: &HashMap<String, usize>,
) -> u64 {
    let mut slot_temporary = vec![0u64; timeline.anchors.len()];
    for (index, task) in graph.tasks.iter().enumerate() {
        slot_temporary[slots[index]] = slot_temporary[slots[index]].max(task.temporary_bytes);
    }
    let mut slot_values = vec![0u64; timeline.anchors.len()];
    let mut alias_slot_bytes: HashMap<(String, usize), u64> = HashMap::new();
    for value in &graph.values {
        if value.size_bytes == 0 {
            continue;
        }
        let live_start = if value.producer.is_empty() {
            0
        } else {
            slots[task_index[&value.producer]]
        };
        let live_end = value
            .consumers
            .iter()
            .map(|consumer| slots[task_index[consumer]])
            .max()
            .unwrap_or(live_start);
        for slot in live_start..=live_end {
            if value.alias_set.is_empty() {
                slot_values[slot] = slot_values[slot].saturating_add(value.size_bytes);
            } else {
                let entry = alias_slot_bytes
                    .entry((value.alias_set.clone(), slot))
                    .or_insert(0);
                *entry = (*entry).max(value.size_bytes);
            }
        }
    }
    for ((_alias, slot), size) in alias_slot_bytes {
        slot_values[slot] = slot_values[slot].saturating_add(size);
    }
    let mut slot_reservations = vec![0u64; timeline.anchors.len()];
    let anchor_index: HashMap<&str, usize> = timeline
        .anchors
        .iter()
        .enumerate()
        .map(|(index, anchor)| (anchor.anchor_id.as_str(), index))
        .collect();
    for region in &timeline.regions {
        let start = anchor_index[region.start.as_str()];
        let end = anchor_index[region.end.as_str()];
        for reservation in slot_reservations.iter_mut().take(end).skip(start) {
            *reservation = reservation.saturating_add(region.reservation.memory_bytes);
        }
    }
    slot_temporary
        .iter()
        .zip(slot_values.iter())
        .zip(slot_reservations.iter())
        .map(|((temporary, values), reservation)| {
            graph
                .persistent_state_bytes
                .saturating_add(*temporary)
                .saturating_add(*values)
                .saturating_add(*reservation)
        })
        .max()
        .unwrap_or(graph.persistent_state_bytes)
}

#[derive(Clone, Debug)]
struct RegionDraft {
    tasks: Vec<usize>,
    earliest: usize,
    before: usize,
    slot: usize,
    serial_fallback: bool,
}

fn region_identity(graph: &ScheduleGraph, draft: &RegionDraft) -> String {
    let mut task_ids: Vec<&str> = draft
        .tasks
        .iter()
        .map(|index| graph.tasks[*index].task_id.as_str())
        .collect();
    task_ids.sort_unstable();
    task_ids.join("\0")
}

fn effect_is_mutating(effect: &ScheduleEffect) -> bool {
    // A conditional transition is a *potential* mutation: merge legality and
    // ordering must treat it as mutating because the signed graph cannot
    // prove at plan time whether the update fires.
    matches!(
        effect.kind,
        ScheduleEffectKind::StateWrite
            | ScheduleEffectKind::StateTransition
            | ScheduleEffectKind::StateTransitionMaybe
            | ScheduleEffectKind::VersionPublish
    )
}

fn typed_effect_reachability(
    graph: &ScheduleGraph,
    task_index: &HashMap<String, usize>,
) -> Vec<BTreeSet<usize>> {
    let mut successors = vec![BTreeSet::new(); graph.tasks.len()];
    for edge in graph
        .edges
        .iter()
        .filter(|edge| edge.kind == ScheduleEdgeKind::Effect)
    {
        successors[task_index[&edge.source]].insert(task_index[&edge.target]);
    }
    let mut reachability = vec![BTreeSet::new(); graph.tasks.len()];
    for source in 0..graph.tasks.len() {
        let mut pending: Vec<usize> = successors[source].iter().copied().collect();
        while let Some(current) = pending.pop() {
            if !reachability[source].insert(current) {
                continue;
            }
            pending.extend(successors[current].iter().copied());
        }
    }
    reachability
}

/// An effectful Region is legal only when it preserves the typed state
/// ordering already present in the typed effect graph.  A state mutation can coexist with
/// pure work, and an ordered read/transition/publish chain can remain inside
/// one compiled callable.  `ExternIo`, or unordered mutations of one state,
/// must remain a Region boundary because the signed graph has no ordering
/// proof to internalize.
fn effects_can_be_internalized(
    graph: &ScheduleGraph,
    tasks: &[usize],
    effect_reachability: &[BTreeSet<usize>],
) -> bool {
    let mut effects_by_resource: BTreeMap<&str, Vec<(usize, &ScheduleEffect)>> = BTreeMap::new();
    for task_index in tasks {
        for effect in &graph.tasks[*task_index].effects {
            if effect.kind == ScheduleEffectKind::ExternIo {
                return false;
            }
            effects_by_resource
                .entry(effect.resource.as_str())
                .or_default()
                .push((*task_index, effect));
        }
    }
    for (resource, effects) in &effects_by_resource {
        if effects.len() > 1
            && graph
                .states
                .iter()
                .any(|state| state.resource == *resource && state.scope == StateScope::Request)
        {
            return false;
        }
        for (position, (left_task, left_effect)) in effects.iter().enumerate() {
            for (right_task, right_effect) in effects.iter().skip(position + 1) {
                if !(effect_is_mutating(left_effect) || effect_is_mutating(right_effect)) {
                    continue;
                }
                if !effect_reachability[*left_task].contains(right_task)
                    && !effect_reachability[*right_task].contains(left_task)
                {
                    return false;
                }
            }
        }
    }
    true
}

fn draft_window(
    graph: &ScheduleGraph,
    anchor_index: &HashMap<String, usize>,
    state_selection_ready: Option<usize>,
    tasks: &[usize],
) -> (usize, usize) {
    let earliest = tasks
        .iter()
        .map(|index| {
            let task = &graph.tasks[*index];
            let declared = anchor_index[&task.timing.after];
            if task
                .inputs
                .iter()
                .any(|input| input.kind == ScheduleInputKind::StateSelection)
            {
                declared.max(
                    state_selection_ready
                        .expect("validated state_selection input requires prepare_inputs.post"),
                )
            } else {
                declared
            }
        })
        .max()
        .expect("ExecutionRegion drafts are non-empty");
    let before = tasks
        .iter()
        .map(|index| anchor_index[&graph.tasks[*index].timing.before])
        .min()
        .expect("ExecutionRegion drafts are non-empty");
    (earliest, before)
}

fn state_selection_ready_index(timeline: &Timeline) -> Option<usize> {
    timeline.anchors.iter().position(|anchor| {
        anchor
            .backend_events
            .as_ref()
            .is_some_and(|events| events.contains(&BackendObservableEvent::PrepareInputsPost))
    })
}

fn task_order_rank(order: &[usize], task_count: usize) -> Vec<usize> {
    let mut rank = vec![0usize; task_count];
    for (position, task) in order.iter().enumerate() {
        rank[*task] = position;
    }
    rank
}

fn drafts_topologically_ordered(
    graph: &ScheduleGraph,
    task_index: &HashMap<String, usize>,
    drafts: &mut [RegionDraft],
    order_rank: &[usize],
) -> bool {
    let mut task_regions = vec![0usize; graph.tasks.len()];
    for (region, draft) in drafts.iter().enumerate() {
        for task in &draft.tasks {
            task_regions[*task] = region;
        }
    }
    let mut indegree = vec![0usize; drafts.len()];
    let mut successors = vec![BTreeSet::new(); drafts.len()];
    for ((source, target), _kind) in dependency_pairs(graph) {
        let source_region = task_regions[task_index[&source]];
        let target_region = task_regions[task_index[&target]];
        if source_region != target_region && successors[source_region].insert(target_region) {
            indegree[target_region] += 1;
        }
    }
    let identities: Vec<String> = drafts
        .iter()
        .map(|draft| region_identity(graph, draft))
        .collect();
    // The deterministic task topological order is retained for independent
    // Region ordering so an otherwise equivalent partition does not perturb
    // execution order; task identity is the secondary tie-breaker.
    let stable_rank: Vec<usize> = drafts
        .iter()
        .map(|draft| {
            draft
                .tasks
                .iter()
                .min_by_key(|task| order_rank[**task])
                .map(|task| order_rank[*task])
                .expect("drafts are non-empty")
        })
        .collect();
    let mut ready: BTreeSet<(usize, String, usize)> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| {
            (*degree == 0).then_some((stable_rank[index], identities[index].clone(), index))
        })
        .collect();
    let mut order = Vec::with_capacity(drafts.len());
    while let Some((_, _, region)) = ready.pop_first() {
        order.push(region);
        for successor in successors[region].iter().copied() {
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready.insert((
                    stable_rank[successor],
                    identities[successor].clone(),
                    successor,
                ));
            }
        }
    }
    if order.len() != drafts.len() {
        return false;
    }
    let original = drafts.to_vec();
    for (destination, source) in order.into_iter().enumerate() {
        drafts[destination] = original[source].clone();
    }
    true
}

fn task_approximation_policy<'a>(graph: &'a ScheduleGraph, task_id: &str) -> &'a str {
    // Multi-workload graphs prefix task ids with `<workload_id>/` (the same
    // convention Python lowering and the parity gate use); single-workload
    // graphs are policy-uniform by construction, so the absent-entry fallback
    // never hides a mix.
    if let Some((workload_id, _)) = task_id.split_once('/') {
        if let Ok(position) = graph
            .workload_approximations
            .binary_search_by(|entry| entry.0.as_str().cmp(workload_id))
        {
            return graph.workload_approximations[position].1.as_str();
        }
    }
    "exact"
}

fn drafts_can_merge(
    graph: &ScheduleGraph,
    selected: &[SelectedTaskImplementation],
    timeline: &Timeline,
    anchor_index: &HashMap<String, usize>,
    left: &RegionDraft,
    right: &RegionDraft,
    effect_reachability: &[BTreeSet<usize>],
    order_rank: &[usize],
) -> Option<RegionDraft> {
    let mut tasks = left.tasks.clone();
    tasks.extend(right.tasks.iter().copied());
    tasks.sort_by_key(|index| order_rank[*index]);
    let (earliest, before) = draft_window(
        graph,
        anchor_index,
        state_selection_ready_index(timeline),
        &tasks,
    );
    let implementation_kind = selected[tasks[0]].kind;
    // Merging is per-policy: a shared Region would otherwise let a non-exact
    // workload's parity tolerance wash over an exact workload's outputs.
    let first_policy = task_approximation_policy(graph, graph.tasks[tasks[0]].task_id.as_str());
    let mixes_policies = tasks.iter().skip(1).any(|index| {
        task_approximation_policy(graph, graph.tasks[*index].task_id.as_str()) != first_policy
    });
    if earliest >= before
        || matches!(
            implementation_kind,
            TaskImplementationKind::Extern | TaskImplementationKind::OptimizerTransition
        )
        || tasks
            .iter()
            .any(|index| selected[*index].kind != implementation_kind)
        || mixes_policies
        || !effects_can_be_internalized(graph, &tasks, effect_reachability)
    {
        return None;
    }
    Some(RegionDraft {
        tasks,
        earliest,
        before,
        slot: earliest,
        serial_fallback: left.serial_fallback || right.serial_fallback,
    })
}

const EXACT_REGION_PARTITION_STATE_LIMIT: usize = 100_000;

type DraftPartitionScore = (u64, u64, u64, u64, u64, u64, u64, String);

fn draft_partition_identity(graph: &ScheduleGraph, drafts: &[RegionDraft]) -> String {
    format!(
        "{:?}",
        drafts
            .iter()
            .map(|draft| {
                draft
                    .tasks
                    .iter()
                    .map(|task| graph.tasks[*task].task_id.as_str())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    )
}

fn explore_region_partitions(
    graph: &ScheduleGraph,
    selected: &[SelectedTaskImplementation],
    timeline: &Timeline,
    anchor_index: &HashMap<String, usize>,
    task_index: &HashMap<String, usize>,
    order_rank: &[usize],
    effect_reachability: &[BTreeSet<usize>],
    capture_cost_enabled: bool,
    budget: u64,
    drafts: Vec<RegionDraft>,
    visited: &mut BTreeSet<Vec<Vec<usize>>>,
    best: &mut Option<(DraftPartitionScore, Vec<RegionDraft>)>,
) -> PyResult<()> {
    let state: Vec<Vec<usize>> = drafts.iter().map(|draft| draft.tasks.clone()).collect();
    if !visited.insert(state) {
        return Ok(());
    }
    if visited.len() > EXACT_REGION_PARTITION_STATE_LIMIT {
        return Err(schedule_error(format!(
            "ExecutionRegion exact partition search exceeded {} states; tile the graph before compilation",
            EXACT_REGION_PARTITION_STATE_LIMIT,
        )));
    }

    let mut scored = drafts.clone();
    if let Ok(slots) = assign_region_slots(graph, task_index, &mut scored) {
        let peak = peak_memory(graph, timeline, &slots, task_index);
        if peak <= budget {
            let (materialization, capture_boundaries, fences, critical_path_extension) =
                draft_partition_cost(graph, selected, task_index, &scored, capture_cost_enabled);
            let score = (
                materialization,
                capture_boundaries,
                peak,
                scored.len() as u64,
                fences,
                critical_path_extension,
                graph.tasks.len() as u64,
                draft_partition_identity(graph, &scored),
            );
            if best.as_ref().is_none_or(|(current, _)| score < *current) {
                *best = Some((score, scored));
            }
        }
    }

    for left in 0..drafts.len() {
        for right in left + 1..drafts.len() {
            let Some(merged) = drafts_can_merge(
                graph,
                selected,
                timeline,
                anchor_index,
                &drafts[left],
                &drafts[right],
                effect_reachability,
                order_rank,
            ) else {
                continue;
            };
            let mut candidate = drafts.clone();
            candidate[left] = merged;
            candidate.remove(right);
            if !drafts_topologically_ordered(graph, task_index, &mut candidate, order_rank) {
                continue;
            }
            explore_region_partitions(
                graph,
                selected,
                timeline,
                anchor_index,
                task_index,
                order_rank,
                effect_reachability,
                capture_cost_enabled,
                budget,
                candidate,
                visited,
                best,
            )?;
        }
    }
    Ok(())
}

fn make_region_drafts(
    graph: &ScheduleGraph,
    selected: &[SelectedTaskImplementation],
    timeline: &Timeline,
    anchor_index: &HashMap<String, usize>,
    task_index: &HashMap<String, usize>,
    order: &[usize],
    force_singletons: bool,
    capture_cost_enabled: bool,
    budget: u64,
) -> PyResult<Vec<RegionDraft>> {
    let order_rank = task_order_rank(order, graph.tasks.len());
    let state_selection_ready = state_selection_ready_index(timeline);
    let drafts: Vec<RegionDraft> = order
        .iter()
        .map(|index| {
            let (earliest, before) =
                draft_window(graph, anchor_index, state_selection_ready, &[*index]);
            RegionDraft {
                tasks: vec![*index],
                earliest,
                before,
                slot: earliest,
                serial_fallback: force_singletons,
            }
        })
        .collect();
    if force_singletons {
        return Ok(drafts);
    }

    let effect_reachability = typed_effect_reachability(graph, task_index);
    let mut visited = BTreeSet::new();
    let mut best = None;
    explore_region_partitions(
        graph,
        selected,
        timeline,
        anchor_index,
        task_index,
        &order_rank,
        &effect_reachability,
        capture_cost_enabled,
        budget,
        drafts,
        &mut visited,
        &mut best,
    )?;
    best.map(|(_score, selected)| selected).ok_or_else(|| {
        schedule_error("ExecutionRegion planner found no partition within the memory budget")
    })
}

/// Mirror the final Region capture eligibility before the executor bindings
/// exist.  Result inputs from another draft become materialized bindings;
/// hidden inputs remain hidden bindings, and the sampler's device-side sample
/// index remains a Tensor binding.  Provided, staged, host-tuple, and step-id
/// bindings make capture ineligible, and a Region with no external Tensor
/// contract is not capturable.  This keeps partition-search cost equal to the
/// final signed `capture_mode` semantics instead of charging eager/disabled
/// timelines.
fn draft_capture_safe(
    graph: &ScheduleGraph,
    selected: &[SelectedTaskImplementation],
    task_index: &HashMap<String, usize>,
    draft: &RegionDraft,
) -> bool {
    if !draft.tasks.iter().all(|task| {
        selected[*task].capture_safe
            && capture_effects_have_stable_storage(graph, &graph.tasks[*task].effects)
    }) {
        return false;
    }
    let members: BTreeSet<usize> = draft.tasks.iter().copied().collect();
    let mut external_inputs = 0usize;
    for task_index_in_draft in &draft.tasks {
        for input in &graph.tasks[*task_index_in_draft].inputs {
            match input.kind {
                ScheduleInputKind::Hidden | ScheduleInputKind::SamplerSampleIdxTensor => {
                    external_inputs += 1
                }
                ScheduleInputKind::Result => {
                    let Some(source) = task_index.get(&input.source).copied() else {
                        return false;
                    };
                    if !members.contains(&source) {
                        external_inputs += 1;
                    }
                }
                ScheduleInputKind::Provided
                | ScheduleInputKind::PackedRequestRows
                | ScheduleInputKind::PackedSamplerRows
                | ScheduleInputKind::PackedHidden
                | ScheduleInputKind::StateSelection
                | ScheduleInputKind::Staged
                | ScheduleInputKind::SamplerRequestIds
                | ScheduleInputKind::SamplerSampleIdxs
                | ScheduleInputKind::SamplerLogits
                | ScheduleInputKind::SampledTokens
                | ScheduleInputKind::StepId => return false,
            }
        }
    }
    external_inputs > 0
}

fn draft_partition_cost(
    graph: &ScheduleGraph,
    selected: &[SelectedTaskImplementation],
    task_index: &HashMap<String, usize>,
    drafts: &[RegionDraft],
    capture_cost_enabled: bool,
) -> (u64, u64, u64, u64) {
    let mut task_regions = vec![0usize; graph.tasks.len()];
    for (region, draft) in drafts.iter().enumerate() {
        for task in &draft.tasks {
            task_regions[*task] = region;
        }
    }
    let materialization_bytes = graph
        .values
        .iter()
        .filter(|value| !value.producer.is_empty())
        .map(|value| {
            let source = task_regions[task_index[&value.producer]];
            let consumers: BTreeSet<usize> = value
                .consumers
                .iter()
                .map(|consumer| task_regions[task_index[consumer]])
                .filter(|consumer| *consumer != source)
                .collect();
            value.size_bytes.saturating_mul(consumers.len() as u64)
        })
        .fold(0u64, u64::saturating_add);
    let capture_boundaries = capture_cost_enabled
        .then(|| {
            drafts
                .iter()
                .filter(|draft| draft_capture_safe(graph, selected, task_index, draft))
                .count() as u64
        })
        .unwrap_or(0);
    let cross_region_fences: BTreeSet<(usize, usize)> = dependency_pairs(graph)
        .into_keys()
        .filter_map(|(source, target)| {
            let source_region = task_regions[task_index[&source]];
            let target_region = task_regions[task_index[&target]];
            (source_region != target_region).then_some((source_region, target_region))
        })
        .collect();
    let mut successors = vec![BTreeSet::new(); drafts.len()];
    let mut indegree = vec![0usize; drafts.len()];
    for &(source, target) in &cross_region_fences {
        if successors[source].insert(target) {
            indegree[target] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(region, degree)| (*degree == 0).then_some(region))
        .collect();
    let mut critical_path = vec![0u64; drafts.len()];
    while let Some(source) = ready.pop_first() {
        for &target in &successors[source] {
            critical_path[target] =
                critical_path[target].max(critical_path[source].saturating_add(1));
            indegree[target] -= 1;
            if indegree[target] == 0 {
                ready.insert(target);
            }
        }
    }
    (
        materialization_bytes,
        capture_boundaries,
        cross_region_fences.len() as u64,
        critical_path.into_iter().max().unwrap_or(0),
    )
}

fn assign_region_slots(
    graph: &ScheduleGraph,
    task_index: &HashMap<String, usize>,
    drafts: &mut [RegionDraft],
) -> PyResult<Vec<usize>> {
    let mut task_regions = vec![0usize; graph.tasks.len()];
    for (region, draft) in drafts.iter().enumerate() {
        for task in &draft.tasks {
            task_regions[*task] = region;
        }
    }
    let mut successors = vec![BTreeSet::new(); drafts.len()];
    for ((source, target), _kind) in dependency_pairs(graph) {
        let source_region = task_regions[task_index[&source]];
        let target_region = task_regions[task_index[&target]];
        if source_region != target_region {
            successors[source_region].insert(target_region);
        }
    }
    for region in (0..drafts.len()).rev() {
        let own_latest = drafts[region].before - 1;
        let successor_latest = successors[region]
            .iter()
            .map(|successor| drafts[*successor].slot)
            .min()
            .unwrap_or(own_latest);
        let slot = own_latest.min(successor_latest);
        if slot < drafts[region].earliest {
            return Err(schedule_error(
                "ExecutionRegion dependency has no legal anchor-window intersection",
            ));
        }
        drafts[region].slot = slot;
    }
    let mut slots = vec![0usize; graph.tasks.len()];
    for draft in drafts {
        for task in &draft.tasks {
            slots[*task] = draft.slot;
        }
    }
    Ok(slots)
}

fn region_artifact_key(
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    region: &ExecutionRegion,
) -> PyResult<String> {
    let mut unsigned = region.clone();
    unsigned.artifact_key.clear();
    let region_json = serde_json::to_string(&unsigned)
        .map_err(|error| schedule_error(format!("failed to serialize ExecutionRegion: {error}")))?;
    Ok(digest(&format!(
        "execution-region-v2\0{graph_json}\0{timeline_json}\0{options_json}\0{region_json}"
    )))
}

fn compile_command_for_region(
    options: &CompileOptions,
    region: &ExecutionRegion,
) -> Option<RegionCompileCommand> {
    (region.implementation_kind == RegionImplementationKind::Compiled).then(|| {
        let adapter = if region.compile_safe && options.compile_policy == "torch_compile" {
            RegionCompilerAdapter::TorchCompile
        } else {
            // This is an explicit Rust-selected eager adapter, never a
            // Python fallback after a failed torch.compile invocation.
            RegionCompilerAdapter::Eager
        };
        RegionCompileCommand {
            adapter,
            fullgraph: adapter == RegionCompilerAdapter::TorchCompile,
            dynamic: false,
            shape_specialization: options.shape_specialization.clone(),
            autograd_policy: RegionAutogradPolicy::EnableGrad,
            fallback_policy: RegionFallbackPolicy::Reject,
        }
    })
}

fn capture_mode_for_region(
    timeline: &Timeline,
    options: &CompileOptions,
    region: &ExecutionRegion,
) -> PyResult<RegionCaptureMode> {
    match options.capture_policy.as_str() {
        "disabled" => Ok(RegionCaptureMode::Disabled),
        "auto" if timeline.graph_mode == "eager" || !region.capture_safe => {
            Ok(RegionCaptureMode::Disabled)
        }
        "auto" => Ok(RegionCaptureMode::Require),
        "require" if timeline.graph_mode == "eager" => Err(schedule_error(
            "required Region capture is unavailable on an eager timeline",
        )),
        "require" if !region.capture_safe => Err(schedule_error(format!(
            "ExecutionRegion {:?} cannot satisfy required capture policy",
            region.region_id
        ))),
        "require" => Ok(RegionCaptureMode::Require),
        _ => Err(value_error("unsupported Region capture policy")),
    }
}

fn side_stream_eligible(graph_schema_version: i64, region: &ExecutionRegion) -> bool {
    let inputs_materialized = !region.input_bindings.is_empty()
        && region
            .input_bindings
            .iter()
            .all(|binding| matches!(binding.kind, RegionInputKind::Materialized));
    let pure_outputs_safe = region.output_bindings.iter().all(|binding| {
        matches!(
            binding.kind,
            RegionOutputKind::Materialized | RegionOutputKind::Sampler
        )
    });
    let transition_outputs_safe = region
        .output_bindings
        .iter()
        .all(|binding| matches!(binding.kind, RegionOutputKind::Materialized));
    let ordered_effects = region.effects.iter().all(|effect| {
        matches!(
            effect.kind,
            ScheduleEffectKind::StateRead
                | ScheduleEffectKind::StateTransition
                | ScheduleEffectKind::StateTransitionMaybe
        )
    });
    let has_ordered_effect = !region.effects.is_empty();
    let pure = region.effects.is_empty()
        && region.capture_safe
        && region.side_stream_safe
        && inputs_materialized
        && pure_outputs_safe;
    let ordered_state_tail = graph_schema_version >= 2
        && region.side_stream_safe
        && has_ordered_effect
        && ordered_effects
        && inputs_materialized
        && transition_outputs_safe;
    pure || ordered_state_tail
}

fn async_host_eligible(
    graph_schema_version: i64,
    region: &ExecutionRegion,
    selected_async_host_safe: bool,
) -> bool {
    let inputs_materialized = !region.input_bindings.is_empty()
        && region
            .input_bindings
            .iter()
            .all(|binding| binding.kind == RegionInputKind::Materialized);
    let outputs_materialized = region
        .output_bindings
        .iter()
        .all(|binding| binding.kind == RegionOutputKind::Materialized);
    let effects_safe = !region.effects.is_empty()
        && region.effects.iter().all(|effect| {
            matches!(
                effect.kind,
                ScheduleEffectKind::StateRead
                    | ScheduleEffectKind::StateTransition
                    | ScheduleEffectKind::StateTransitionMaybe
            )
        });
    graph_schema_version >= 2
        && region.stream == Some(RegionStreamClass::SideForward)
        && selected_async_host_safe
        && inputs_materialized
        && outputs_materialized
        && effects_safe
}

const ASYNC_SERIAL_SUBMISSION_ID: &str = "async-serial-0";

fn append_async_submission(commands: &mut Vec<RuntimeHostCommand>, region_ids: &mut Vec<String>) {
    debug_assert!(!region_ids.is_empty());
    commands.push(RuntimeHostCommand {
        kind: HostCommandKind::WaitAsync,
        submission_id: ASYNC_SERIAL_SUBMISSION_ID.to_owned(),
        region_ids: Vec::new(),
        async_capture_mode: AsyncBatchCaptureMode::Disabled,
        async_artifact_key: String::new(),
    });
    commands.push(RuntimeHostCommand {
        kind: HostCommandKind::SubmitAsync,
        submission_id: ASYNC_SERIAL_SUBMISSION_ID.to_owned(),
        region_ids: std::mem::take(region_ids),
        async_capture_mode: AsyncBatchCaptureMode::Disabled,
        async_artifact_key: String::new(),
    });
}

fn validate_host_commands(
    region_ids: &[String],
    commands: &[RuntimeHostCommand],
    regions_by_id: &HashMap<&str, &ExecutionRegion>,
) -> PyResult<()> {
    let mut flattened = Vec::new();
    for (index, command) in commands.iter().enumerate() {
        match command.kind {
            HostCommandKind::WaitAsync => {
                if command.submission_id != ASYNC_SERIAL_SUBMISSION_ID
                    || !command.region_ids.is_empty()
                    || command.async_capture_mode != AsyncBatchCaptureMode::Disabled
                    || !command.async_artifact_key.is_empty()
                {
                    return Err(schedule_error(
                        "wait_async requires the fixed async-serial-0 submission, no regions, and disabled async capture",
                    ));
                }
            }
            HostCommandKind::ExecuteInline => {
                if !command.submission_id.is_empty()
                    || command.region_ids.is_empty()
                    || command.async_capture_mode != AsyncBatchCaptureMode::Disabled
                    || !command.async_artifact_key.is_empty()
                {
                    return Err(schedule_error(
                        "execute_inline requires an empty submission_id, non-empty regions, and disabled async capture",
                    ));
                }
                for region_id in &command.region_ids {
                    if regions_by_id
                        .get(region_id.as_str())
                        .is_none_or(|region| region.host_submission != RegionHostSubmission::Inline)
                    {
                        return Err(schedule_error(
                            "execute_inline references a non-inline ExecutionRegion",
                        ));
                    }
                }
                flattened.extend(command.region_ids.iter().cloned());
            }
            HostCommandKind::SubmitAsync => {
                if command.submission_id != ASYNC_SERIAL_SUBMISSION_ID
                    || command.region_ids.is_empty()
                    || index == 0
                    || commands[index - 1].kind != HostCommandKind::WaitAsync
                    || commands[index - 1].submission_id != command.submission_id
                {
                    return Err(schedule_error(
                        "submit_async requires non-empty regions and an immediately preceding wait on async-serial-0",
                    ));
                }
                match command.async_capture_mode {
                    AsyncBatchCaptureMode::Disabled if !command.async_artifact_key.is_empty() => {
                        return Err(schedule_error(
                            "disabled submit_async capture forbids an async artifact key",
                        ));
                    }
                    AsyncBatchCaptureMode::OptimizerTransition
                        if !is_lowercase_sha256(&command.async_artifact_key) =>
                    {
                        return Err(schedule_error(
                            "optimizer_transition submit_async capture requires a lowercase SHA-256 artifact key",
                        ));
                    }
                    AsyncBatchCaptureMode::Disabled
                    | AsyncBatchCaptureMode::OptimizerTransition => {}
                }
                for region_id in &command.region_ids {
                    if regions_by_id.get(region_id.as_str()).is_none_or(|region| {
                        region.host_submission != RegionHostSubmission::AsyncSerial
                    }) {
                        return Err(schedule_error(
                            "submit_async references a non-async ExecutionRegion",
                        ));
                    }
                }
                flattened.extend(command.region_ids.iter().cloned());
            }
        }
    }
    if flattened != region_ids {
        return Err(schedule_error(
            "runtime host commands must flatten execute_inline and submit_async to region_ids exactly",
        ));
    }
    Ok(())
}

fn runtime_advance_plan(
    timeline: &Timeline,
    execution_regions: &[ExecutionRegion],
) -> PyResult<(Vec<RuntimeAdvancePlan>, String)> {
    let anchor_index: HashMap<&str, usize> = timeline
        .anchors
        .iter()
        .enumerate()
        .map(|(index, anchor)| (anchor.anchor_id.as_str(), index))
        .collect();
    let mut region_ids_by_anchor: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for region in execution_regions {
        let Some(anchor) = timeline
            .anchors
            .iter()
            .find(|anchor| anchor.anchor_id == region.placement_anchor)
        else {
            return Err(schedule_error(
                "ExecutionRegion placement references an unknown timeline anchor",
            ));
        };
        if timeline.schema_version == 2 && anchor.backend_events.as_ref().is_none_or(Vec::is_empty)
        {
            return Err(schedule_error(format!(
                "ExecutionRegion {:?} is placed at an anchor without a backend event",
                region.region_id
            )));
        }
        region_ids_by_anchor
            .entry(anchor.anchor_id.as_str())
            .or_default()
            .push(region.region_id.clone());
    }

    let regions_by_id: HashMap<&str, &ExecutionRegion> = execution_regions
        .iter()
        .map(|region| (region.region_id.as_str(), region))
        .collect();
    let mut side_access_resources = BTreeMap::<&str, bool>::new();
    for effect in execution_regions
        .iter()
        .filter(|region| region.stream == Some(RegionStreamClass::SideForward))
        .flat_map(|region| region.effects.iter())
    {
        let mutates = matches!(
            effect.kind,
            ScheduleEffectKind::StateTransition | ScheduleEffectKind::StateTransitionMaybe
        );
        side_access_resources
            .entry(effect.resource.as_str())
            .and_modify(|existing| *existing |= mutates)
            .or_insert(mutates);
    }
    let has_side_sampler_output = execution_regions.iter().any(|region| {
        region.stream == Some(RegionStreamClass::SideForward)
            && region
                .output_bindings
                .iter()
                .any(|binding| binding.kind == RegionOutputKind::Sampler)
    });
    let async_region_ids: BTreeSet<&str> = execution_regions
        .iter()
        .filter(|region| region.host_submission == RegionHostSubmission::AsyncSerial)
        .map(|region| region.region_id.as_str())
        .collect();
    let mut async_access_resources = BTreeMap::<&str, bool>::new();
    for effect in execution_regions
        .iter()
        .filter(|region| region.host_submission == RegionHostSubmission::AsyncSerial)
        .flat_map(|region| region.effects.iter())
    {
        let mutates = matches!(
            effect.kind,
            ScheduleEffectKind::StateTransition | ScheduleEffectKind::StateTransitionMaybe
        );
        async_access_resources
            .entry(effect.resource.as_str())
            .and_modify(|existing| *existing |= mutates)
            .or_insert(mutates);
    }
    let mut staged_binder_identities = BTreeSet::<(BackendObservableEvent, String)>::new();
    let mut advances = Vec::new();
    for anchor in &timeline.anchors {
        for backend_event in anchor.backend_events.as_deref().unwrap_or_default() {
            let region_ids = region_ids_by_anchor
                .get(anchor.anchor_id.as_str())
                .cloned()
                .unwrap_or_default();
            let wait_for_prior_side_work = region_ids.iter().any(|region_id| {
                regions_by_id.get(region_id.as_str()).is_some_and(|region| {
                    region.stream != Some(RegionStreamClass::SideForward)
                        && region.effects.iter().any(|effect| {
                            side_access_resources
                                .get(effect.resource.as_str())
                                .is_some_and(|side_mutates| {
                                    *side_mutates || effect.kind != ScheduleEffectKind::StateRead
                                })
                        })
                })
            });
            let mut host_commands = Vec::new();
            let mut inline_batch = Vec::new();
            let mut async_batch = Vec::new();
            for region_id in &region_ids {
                let region = regions_by_id.get(region_id.as_str()).ok_or_else(|| {
                    schedule_error("runtime advance references an unknown ExecutionRegion")
                })?;
                if region.host_submission == RegionHostSubmission::AsyncSerial {
                    if !inline_batch.is_empty() {
                        host_commands.push(RuntimeHostCommand {
                            kind: HostCommandKind::ExecuteInline,
                            submission_id: String::new(),
                            region_ids: std::mem::take(&mut inline_batch),
                            async_capture_mode: AsyncBatchCaptureMode::Disabled,
                            async_artifact_key: String::new(),
                        });
                    }
                    async_batch.push(region_id.clone());
                    continue;
                }

                if !async_batch.is_empty() {
                    append_async_submission(&mut host_commands, &mut async_batch);
                    // PhysicalExecutor currently reserves one host submission
                    // per logical step.  A later inline Region in that same
                    // step therefore receives an explicit Rust-owned wait;
                    // Python never infers this capacity edge.
                    host_commands.push(RuntimeHostCommand {
                        kind: HostCommandKind::WaitAsync,
                        submission_id: ASYNC_SERIAL_SUBMISSION_ID.to_owned(),
                        region_ids: Vec::new(),
                        async_capture_mode: AsyncBatchCaptureMode::Disabled,
                        async_artifact_key: String::new(),
                    });
                }
                let waits_for_async_dependency = region
                    .dependencies
                    .iter()
                    .any(|dependency| async_region_ids.contains(dependency.as_str()));
                let waits_for_async_state = region.effects.iter().any(|effect| {
                    async_access_resources
                        .get(effect.resource.as_str())
                        .is_some_and(|async_mutates| {
                            *async_mutates || effect.kind != ScheduleEffectKind::StateRead
                        })
                });
                if waits_for_async_dependency || waits_for_async_state {
                    if !inline_batch.is_empty() {
                        host_commands.push(RuntimeHostCommand {
                            kind: HostCommandKind::ExecuteInline,
                            submission_id: String::new(),
                            region_ids: std::mem::take(&mut inline_batch),
                            async_capture_mode: AsyncBatchCaptureMode::Disabled,
                            async_artifact_key: String::new(),
                        });
                    }
                    if host_commands
                        .last()
                        .is_none_or(|command| command.kind != HostCommandKind::WaitAsync)
                    {
                        host_commands.push(RuntimeHostCommand {
                            kind: HostCommandKind::WaitAsync,
                            submission_id: ASYNC_SERIAL_SUBMISSION_ID.to_owned(),
                            region_ids: Vec::new(),
                            async_capture_mode: AsyncBatchCaptureMode::Disabled,
                            async_artifact_key: String::new(),
                        });
                    }
                }
                inline_batch.push(region_id.clone());
            }
            if !async_batch.is_empty() {
                append_async_submission(&mut host_commands, &mut async_batch);
            }
            if !inline_batch.is_empty() {
                host_commands.push(RuntimeHostCommand {
                    kind: HostCommandKind::ExecuteInline,
                    submission_id: String::new(),
                    region_ids: inline_batch,
                    async_capture_mode: AsyncBatchCaptureMode::Disabled,
                    async_artifact_key: String::new(),
                });
            }
            validate_host_commands(&region_ids, &host_commands, &regions_by_id)?;
            let mut backend_region_binder_stages = Vec::new();
            for region_id in &region_ids {
                let region = regions_by_id.get(region_id.as_str()).ok_or_else(|| {
                    schedule_error("runtime advance references an unknown ExecutionRegion")
                })?;
                if region.implementation_kind != RegionImplementationKind::Extern
                    || !region
                        .required_capabilities
                        .iter()
                        .any(|capability| capability == "backend/region_binder")
                {
                    continue;
                }
                for implementation_id in &region.implementation_ids {
                    if !staged_binder_identities.insert((*backend_event, implementation_id.clone()))
                    {
                        return Err(schedule_error(format!(
                            "signed backend Region binder implementation {:?} is duplicated for backend event {:?}",
                            implementation_id, backend_event
                        )));
                    }
                    backend_region_binder_stages.push(BackendRegionBinderStage {
                        backend_event: *backend_event,
                        region_id: region_id.clone(),
                        implementation_id: implementation_id.clone(),
                    });
                }
            }
            advances.push(RuntimeAdvancePlan {
                backend_event: *backend_event,
                anchor_id: anchor.anchor_id.clone(),
                region_ids,
                host_commands,
                backend_region_binder_stages,
                wait_for_prior_side_work,
                wait_for_side_outputs: *backend_event == BackendObservableEvent::SamplerPreDraw
                    && has_side_sampler_output,
            });
        }
    }

    let referenced_as_dependency: BTreeSet<&str> = execution_regions
        .iter()
        .flat_map(|region| region.dependencies.iter().map(String::as_str))
        .collect();
    let completion_anchor = execution_regions
        .iter()
        .filter(|region| !referenced_as_dependency.contains(region.region_id.as_str()))
        .max_by_key(|region| anchor_index[region.placement_anchor.as_str()])
        .map(|region| region.placement_anchor.clone())
        .unwrap_or_else(|| timeline.anchors.last().unwrap().anchor_id.clone());
    Ok((advances, completion_anchor))
}

fn async_optimizer_capture_eligible(
    graph: &ScheduleGraph,
    timeline: &Timeline,
    options: &CompileOptions,
    selected: &[SelectedTaskImplementation],
    task_index: &HashMap<String, usize>,
    execution_regions: &[ExecutionRegion],
    materializations: &[Materialization],
    region_ids: &[String],
) -> bool {
    if !matches!(graph.schema_version, 3 | 4 | 5)
        || options.capture_policy == "disabled"
        || timeline.graph_mode == "eager"
        || region_ids.len() < 2
    {
        return false;
    }
    let region_position: HashMap<&str, usize> = execution_regions
        .iter()
        .enumerate()
        .map(|(position, region)| (region.region_id.as_str(), position))
        .collect();
    let Some(regions) = region_ids
        .iter()
        .map(|region_id| {
            region_position
                .get(region_id.as_str())
                .map(|position| &execution_regions[*position])
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    if regions.windows(2).any(|pair| {
        region_position[pair[1].region_id.as_str()]
            != region_position[pair[0].region_id.as_str()] + 1
    }) || regions.iter().any(|region| {
        region.placement_anchor != regions[0].placement_anchor
            || region.stream != Some(RegionStreamClass::SideForward)
            || region.host_submission != RegionHostSubmission::AsyncSerial
            || region.capture_mode != RegionCaptureMode::Disabled
            || region.input_bindings.is_empty()
            || region
                .input_bindings
                .iter()
                .any(|binding| binding.kind != RegionInputKind::Materialized)
            || region
                .output_bindings
                .iter()
                .any(|binding| binding.kind != RegionOutputKind::Materialized)
    }) {
        return false;
    }

    let mut optimizer_task = None;
    for (region_ordinal, region) in regions.iter().enumerate() {
        for task_id in &region.tasks {
            let implementation = &selected[task_index[task_id]];
            if !implementation.async_host_safe || !implementation.side_stream_safe {
                return false;
            }
            match implementation.kind {
                TaskImplementationKind::Captured
                    if optimizer_task.is_none() && implementation.compile_safe => {}
                TaskImplementationKind::OptimizerTransition
                    if optimizer_task.is_none()
                        && region_ordinal + 1 == regions.len()
                        && region.tasks.len() == 1 =>
                {
                    optimizer_task = Some(task_id.as_str());
                }
                TaskImplementationKind::Captured
                | TaskImplementationKind::Extern
                | TaskImplementationKind::OptimizerTransition => return false,
            }
        }
    }
    let Some(optimizer_task) = optimizer_task else {
        return false;
    };

    let mut mutation_resource = None;
    let mut read_resources = BTreeSet::new();
    for region in &regions {
        for effect in &region.effects {
            match effect.kind {
                ScheduleEffectKind::StateRead => {
                    read_resources.insert(effect.resource.as_str());
                }
                ScheduleEffectKind::StateTransition
                    if region.tasks == [optimizer_task] && mutation_resource.is_none() =>
                {
                    mutation_resource = Some(effect.resource.as_str());
                }
                ScheduleEffectKind::StateWrite
                | ScheduleEffectKind::StateTransition
                | ScheduleEffectKind::StateTransitionMaybe
                | ScheduleEffectKind::VersionPublish
                | ScheduleEffectKind::ExternIo => return false,
            }
        }
    }
    let Some(mutation_resource) = mutation_resource else {
        return false;
    };
    if !read_resources.is_empty() && read_resources != BTreeSet::from([mutation_resource]) {
        return false;
    }
    if graph
        .states
        .iter()
        .find(|state| state.resource == mutation_resource)
        .is_none_or(|state| {
            state.scope == StateScope::Request
                || state.storage != Some(ScheduleStateStorage::OptimizerStable)
                || state.transition != CAPTURE_STABLE_STATE_TRANSITION
        })
    {
        return false;
    }

    let members: BTreeSet<&str> = region_ids.iter().map(String::as_str).collect();
    let materializations_by_id: HashMap<&str, &Materialization> = materializations
        .iter()
        .map(|value| (value.materialization_id.as_str(), value))
        .collect();
    let mut external_inputs = 0usize;
    let mut adjacent_edges = BTreeSet::<(&str, &str)>::new();
    for (region_ordinal, region) in regions.iter().enumerate() {
        for binding in &region.input_bindings {
            let Some(materialization) = materializations_by_id.get(binding.binding_id.as_str())
            else {
                return false;
            };
            if materialization.consumer_region != region.region_id {
                return false;
            }
            if members.contains(materialization.producer_region.as_str()) {
                let producer_position = region_position[materialization.producer_region.as_str()];
                if producer_position >= region_position[region.region_id.as_str()] {
                    return false;
                }
                adjacent_edges.insert((
                    materialization.producer_region.as_str(),
                    materialization.consumer_region.as_str(),
                ));
            } else {
                if region_ordinal != 0
                    || region_position
                        .get(materialization.producer_region.as_str())
                        .is_none_or(|position| {
                            *position >= region_position[regions[0].region_id.as_str()]
                                || execution_regions[*position].placement_anchor
                                    == regions[0].placement_anchor
                        })
                {
                    return false;
                }
                external_inputs += 1;
            }
        }
        for binding in &region.output_bindings {
            let Some(materialization) = materializations_by_id.get(binding.binding_id.as_str())
            else {
                return false;
            };
            if materialization.producer_region != region.region_id
                || !members.contains(materialization.consumer_region.as_str())
            {
                return false;
            }
        }
    }
    external_inputs > 0
        && regions.windows(2).all(|pair| {
            adjacent_edges.contains(&(pair[0].region_id.as_str(), pair[1].region_id.as_str()))
                && pair[1]
                    .dependencies
                    .iter()
                    .any(|dependency| dependency == &pair[0].region_id)
        })
}

fn async_capture_artifact_key(
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    advance: &RuntimeAdvancePlan,
    command: &RuntimeHostCommand,
    execution_regions: &[ExecutionRegion],
) -> PyResult<String> {
    let regions_by_id: HashMap<&str, &ExecutionRegion> = execution_regions
        .iter()
        .map(|region| (region.region_id.as_str(), region))
        .collect();
    let members = command
        .region_ids
        .iter()
        .map(|region_id| {
            regions_by_id
                .get(region_id.as_str())
                .map(|region| (region_id.as_str(), region.artifact_key.as_str()))
                .ok_or_else(|| schedule_error("async capture references an unknown Region"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let batch_json = serde_json::to_string(&(
        advance.backend_event,
        advance.anchor_id.as_str(),
        AsyncBatchCaptureMode::OptimizerTransition,
        members,
    ))
    .map_err(|error| schedule_error(format!("failed to serialize async capture batch: {error}")))?;
    Ok(digest(&format!(
        "async-optimizer-transition-v1\0{graph_json}\0{timeline_json}\0{options_json}\0{batch_json}"
    )))
}

fn sign_async_capture_commands(
    graph: &ScheduleGraph,
    graph_json: &str,
    timeline: &Timeline,
    timeline_json: &str,
    options: &CompileOptions,
    options_json: &str,
    selected: &[SelectedTaskImplementation],
    task_index: &HashMap<String, usize>,
    execution_regions: &[ExecutionRegion],
    materializations: &[Materialization],
    advances: &mut [RuntimeAdvancePlan],
) -> PyResult<()> {
    let regions_by_id: HashMap<&str, &ExecutionRegion> = execution_regions
        .iter()
        .map(|region| (region.region_id.as_str(), region))
        .collect();
    for advance in advances {
        let advance_view = advance.clone();
        for command in &mut advance.host_commands {
            if command.kind == HostCommandKind::SubmitAsync
                && async_optimizer_capture_eligible(
                    graph,
                    timeline,
                    options,
                    selected,
                    task_index,
                    execution_regions,
                    materializations,
                    &command.region_ids,
                )
            {
                command.async_capture_mode = AsyncBatchCaptureMode::OptimizerTransition;
                command.async_artifact_key = async_capture_artifact_key(
                    graph_json,
                    timeline_json,
                    options_json,
                    &advance_view,
                    command,
                    execution_regions,
                )?;
            }
        }
        validate_host_commands(&advance.region_ids, &advance.host_commands, &regions_by_id)?;
    }
    Ok(())
}

/// A capture Region must be able to inspect its complete Tensor contract
/// before its backend-event group submits any Region. A materialization made
/// by another Region at the same placement anchor only exists after that
/// producer executes, so accepting this shape would permit a later graph miss
/// after an earlier effect/sink has already committed.
fn validate_capture_preflightability(
    execution_regions: &[ExecutionRegion],
    materializations: &[Materialization],
) -> PyResult<()> {
    let regions = execution_regions
        .iter()
        .map(|region| (region.region_id.as_str(), region))
        .collect::<HashMap<_, _>>();
    for materialization in materializations {
        let producer = regions
            .get(materialization.producer_region.as_str())
            .ok_or_else(|| schedule_error("materialization producer Region is missing"))?;
        let consumer = regions
            .get(materialization.consumer_region.as_str())
            .ok_or_else(|| schedule_error("materialization consumer Region is missing"))?;
        if consumer.capture_mode == RegionCaptureMode::Require
            && producer.placement_anchor == consumer.placement_anchor
        {
            return Err(schedule_error(format!(
                "capture Region {:?} consumes materialization {:?} from {:?} in the same backend-event group; shape preflight is impossible",
                consumer.region_id,
                materialization.materialization_id,
                producer.region_id,
            )));
        }
    }
    Ok(())
}

/// Materialization records are the only signed authority for executable
/// cross-Region values.  Region port bindings may transport those values, but
/// may not create an additional boundary or invent a lifetime locally.
fn validate_materialization_bindings(
    graph: &ScheduleGraph,
    execution_regions: &[ExecutionRegion],
    materializations: &[Materialization],
) -> PyResult<()> {
    let regions = execution_regions
        .iter()
        .map(|region| (region.region_id.as_str(), region))
        .collect::<HashMap<_, _>>();
    let values = graph
        .values
        .iter()
        .map(|value| ((value.value_id.as_str(), value.version), value))
        .collect::<HashMap<_, _>>();
    let mut materialization_ids = BTreeSet::new();
    let mut expected_inputs = BTreeSet::new();
    let mut expected_outputs = BTreeSet::new();

    for materialization in materializations {
        if !materialization_ids.insert(materialization.materialization_id.as_str()) {
            return Err(schedule_error("materialization ids must be unique"));
        }
        let producer = regions
            .get(materialization.producer_region.as_str())
            .ok_or_else(|| schedule_error("materialization producer Region is missing"))?;
        let consumer = regions
            .get(materialization.consumer_region.as_str())
            .ok_or_else(|| schedule_error("materialization consumer Region is missing"))?;
        if producer.region_id == consumer.region_id {
            return Err(schedule_error(
                "Region-internal values cannot be materialized",
            ));
        }
        if materialization.live_start != producer.placement_anchor
            || materialization.live_end != consumer.placement_anchor
        {
            return Err(schedule_error(
                "materialization lifetime must match its producer and consumer Region anchors",
            ));
        }
        let value = values
            .get(&(materialization.value_id.as_str(), materialization.version))
            .ok_or_else(|| schedule_error("materialization value or version is unknown"))?;
        if value.producer.is_empty() || value.producer_output.is_empty() {
            return Err(schedule_error(
                "materialization value has no executable producer output",
            ));
        }
        let output_matches: Vec<&RegionOutputBinding> = producer
            .output_bindings
            .iter()
            .filter(|binding| {
                binding.kind == RegionOutputKind::Materialized
                    && binding.binding_id == materialization.materialization_id
            })
            .collect();
        if output_matches.len() != 1
            || output_matches[0].target != materialization.value_id
            || output_matches[0].producer.task_id != value.producer
            || output_matches[0].producer.port_name != value.producer_output
        {
            return Err(schedule_error(
                "materialization has a missing or mismatched producer binding",
            ));
        }
        let input_matches: Vec<&RegionInputBinding> = consumer
            .input_bindings
            .iter()
            .filter(|binding| {
                binding.kind == RegionInputKind::Materialized
                    && binding.binding_id == materialization.materialization_id
            })
            .collect();
        if input_matches.len() != 1 || input_matches[0].source != materialization.value_id {
            return Err(schedule_error(
                "materialization has a missing or mismatched consumer binding",
            ));
        }
        let expected_ports: BTreeSet<(&str, &str)> = consumer
            .tasks
            .iter()
            .filter_map(|task_id| {
                let task = graph.tasks.iter().find(|task| task.task_id == *task_id)?;
                Some(
                    task.inputs
                        .iter()
                        .filter(move |input| {
                            input.kind == ScheduleInputKind::Result
                                && input.source == value.producer
                                && input.output == value.producer_output
                        })
                        .map(move |input| (task_id.as_str(), input.name.as_str())),
                )
            })
            .flatten()
            .collect();
        let actual_ports: BTreeSet<(&str, &str)> = input_matches[0]
            .consumers
            .iter()
            .map(|port| (port.task_id.as_str(), port.port_name.as_str()))
            .collect();
        if expected_ports.is_empty() || actual_ports != expected_ports {
            return Err(schedule_error(
                "materialization consumer ports do not match the signed value",
            ));
        }
        expected_inputs.insert((
            consumer.region_id.as_str(),
            materialization.materialization_id.as_str(),
        ));
        expected_outputs.insert((
            producer.region_id.as_str(),
            materialization.materialization_id.as_str(),
        ));
    }

    let actual_inputs: BTreeSet<(&str, &str)> = execution_regions
        .iter()
        .flat_map(|region| {
            region.input_bindings.iter().filter_map(move |binding| {
                (binding.kind == RegionInputKind::Materialized)
                    .then_some((region.region_id.as_str(), binding.binding_id.as_str()))
            })
        })
        .collect();
    let actual_outputs: BTreeSet<(&str, &str)> = execution_regions
        .iter()
        .flat_map(|region| {
            region.output_bindings.iter().filter_map(move |binding| {
                (binding.kind == RegionOutputKind::Materialized)
                    .then_some((region.region_id.as_str(), binding.binding_id.as_str()))
            })
        })
        .collect();
    if actual_inputs != expected_inputs {
        return Err(schedule_error(
            "materialized Region input bindings must exactly match materialization records",
        ));
    }
    if actual_outputs != expected_outputs {
        return Err(schedule_error(
            "materialized Region output bindings must exactly match materialization records",
        ));
    }
    for region in execution_regions {
        let declared_inputs: BTreeSet<&str> = region
            .materialized_inputs
            .iter()
            .map(String::as_str)
            .collect();
        let declared_outputs: BTreeSet<&str> = region
            .materialized_outputs
            .iter()
            .map(String::as_str)
            .collect();
        let bound_inputs: BTreeSet<&str> = region
            .input_bindings
            .iter()
            .filter(|binding| binding.kind == RegionInputKind::Materialized)
            .map(|binding| binding.binding_id.as_str())
            .collect();
        let bound_outputs: BTreeSet<&str> = region
            .output_bindings
            .iter()
            .filter(|binding| binding.kind == RegionOutputKind::Materialized)
            .map(|binding| binding.binding_id.as_str())
            .collect();
        if declared_inputs != bound_inputs || declared_outputs != bound_outputs {
            return Err(schedule_error(
                "ExecutionRegion declared materializations must match typed bindings",
            ));
        }
    }
    Ok(())
}

fn compile_impl(
    graph: &ScheduleGraph,
    graph_json: &str,
    timeline: &Timeline,
    timeline_json: &str,
    options: &CompileOptions,
    options_json: &str,
) -> PyResult<Decision> {
    let (anchor_index, task_index) = validate_inputs(graph, timeline, options)?;
    let selected = select_task_implementations(graph, timeline)?;
    let order = topological_order(graph, &task_index)?;
    let affine_regions = compile_affine_regions(graph, &task_index)?;
    let budget = effective_budget(options, timeline);

    let mut drafts = make_region_drafts(
        graph,
        &selected,
        timeline,
        &anchor_index,
        &task_index,
        &order,
        false,
        options.capture_policy != "disabled" && timeline.graph_mode != "eager",
        budget,
    )?;
    let slots = assign_region_slots(graph, &task_index, &mut drafts)?;
    let peak = peak_memory(graph, timeline, &slots, &task_index);
    if peak > budget {
        return Err(schedule_error(format!(
            "ExecutionRegion exact partition planner violated its memory hard constraint: peak={peak} budget={budget}"
        )));
    }

    let execution_order: Vec<String> = drafts
        .iter()
        .flat_map(|draft| draft.tasks.iter())
        .map(|index| graph.tasks[*index].task_id.clone())
        .collect();
    let mut task_regions = vec![0usize; graph.tasks.len()];
    for (ordinal, draft) in drafts.iter().enumerate() {
        for task in &draft.tasks {
            task_regions[*task] = ordinal;
        }
    }
    let mut execution_regions = Vec::with_capacity(drafts.len());
    for (ordinal, draft) in drafts.iter().enumerate() {
        let tasks: Vec<String> = draft
            .tasks
            .iter()
            .map(|index| graph.tasks[*index].task_id.clone())
            .collect();
        let identity = format!(
            "{}\0{}\0{}",
            tasks.join("\0"),
            timeline.anchors[draft.earliest].anchor_id,
            timeline.anchors[draft.before].anchor_id
        );
        let region_id = format!("region-{ordinal:04}-{}", &digest(&identity)[..16]);
        let implementation_kind = if matches!(
            selected[draft.tasks[0]].kind,
            TaskImplementationKind::Extern | TaskImplementationKind::OptimizerTransition
        ) {
            RegionImplementationKind::Extern
        } else {
            RegionImplementationKind::Compiled
        };
        let required_capabilities: BTreeSet<String> = draft
            .tasks
            .iter()
            .flat_map(|index| selected[*index].required_capabilities.iter().cloned())
            .collect();
        let region_tasks: BTreeSet<&str> = tasks.iter().map(String::as_str).collect();
        let affine_schedule = affine_regions
            .iter()
            .find(|region| {
                region.tasks.len() == tasks.len()
                    && region
                        .tasks
                        .iter()
                        .all(|task| region_tasks.contains(task.as_str()))
            })
            .map(|region| region.schedule.clone())
            .unwrap_or_default();
        // This is selected implementation evidence, not a compiler policy.
        // An explicitly signed eager command does not change it.
        let compile_safe = implementation_kind == RegionImplementationKind::Compiled
            && draft
                .tasks
                .iter()
                .all(|index| selected[*index].compile_safe)
            && draft.tasks.iter().all(|index| {
                graph.tasks[*index]
                    .inputs
                    .iter()
                    .all(|input| input.kind != ScheduleInputKind::StateSelection)
            });
        let capture_safe = compile_safe
            && draft
                .tasks
                .iter()
                .all(|index| selected[*index].capture_safe);
        let side_stream_safe = draft
            .tasks
            .iter()
            .all(|index| selected[*index].side_stream_safe);
        // The stream class is assigned after input bindings refine
        // capture safety (see below): side placement requires the final
        // signed capture_safe, not the draft-time value.
        let stream = RegionStreamClass::Default;
        execution_regions.push(ExecutionRegion {
            region_id,
            tasks,
            after_anchor: timeline.anchors[draft.earliest].anchor_id.clone(),
            before_anchor: timeline.anchors[draft.before].anchor_id.clone(),
            placement_anchor: timeline.anchors[draft.slot].anchor_id.clone(),
            implementation_kind,
            implementation_ids: draft
                .tasks
                .iter()
                .map(|index| selected[*index].implementation_id.clone())
                .collect(),
            compile_command: None,
            required_capabilities: required_capabilities.into_iter().collect(),
            dependencies: Vec::new(),
            materialized_inputs: Vec::new(),
            materialized_outputs: Vec::new(),
            input_bindings: Vec::new(),
            output_bindings: Vec::new(),
            task_invocations: Vec::new(),
            effects: Vec::new(),
            compile_safe,
            capture_safe,
            side_stream_safe,
            capture_mode: RegionCaptureMode::Disabled,
            stream: Some(stream),
            host_submission: RegionHostSubmission::Inline,
            fallback_policy: Some(RegionFallbackPolicy::Reject),
            artifact_key: String::new(),
            serial_fallback: draft.serial_fallback,
            affine_schedule,
        });
    }

    let dependencies = dependency_pairs(graph);
    let mut region_dependencies = vec![BTreeSet::new(); execution_regions.len()];
    let mut fence_kinds: BTreeMap<(usize, usize), String> = BTreeMap::new();
    for ((source, target), kind) in &dependencies {
        let source_region = task_regions[task_index[source]];
        let target_region = task_regions[task_index[target]];
        if source_region == target_region {
            continue;
        }
        region_dependencies[target_region]
            .insert(execution_regions[source_region].region_id.clone());
        let fence_kind = fence_kinds
            .entry((source_region, target_region))
            .or_insert_with(|| "ready".to_owned());
        if *kind == ScheduleEdgeKind::Effect {
            *fence_kind = "completion".to_owned();
        }
    }

    let mut materializations = Vec::new();
    // A Region input is bound to the full value identity.  ``value_id`` alone
    // is deliberately insufficient: the ScheduleGraph permits versions of
    // one logical value, and two versions can cross into the same Region.
    let mut materialization_by_value_region: BTreeMap<
        (String, i64, String, String, usize),
        String,
    > = BTreeMap::new();
    let mut materialized_inputs = vec![BTreeSet::new(); execution_regions.len()];
    let mut materialized_outputs = vec![BTreeSet::new(); execution_regions.len()];
    for value in &graph.values {
        if value.producer.is_empty() {
            continue;
        }
        let producer_region = task_regions[task_index[&value.producer]];
        let consumer_regions: BTreeSet<usize> = value
            .consumers
            .iter()
            .map(|consumer| task_regions[task_index[consumer]])
            .filter(|consumer_region| *consumer_region != producer_region)
            .collect();
        for consumer_region in consumer_regions {
            let materialization_id = format!(
                "materialization-{}",
                digest(&format!(
                    "{}\0{}\0{}\0{}\0{}\0{}",
                    value.value_id,
                    value.version,
                    value.producer,
                    value.producer_output,
                    execution_regions[producer_region].region_id,
                    execution_regions[consumer_region].region_id,
                ))
            );
            materialized_outputs[producer_region].insert(materialization_id.clone());
            materialized_inputs[consumer_region].insert(materialization_id.clone());
            materialization_by_value_region.insert(
                (
                    value.value_id.clone(),
                    value.version,
                    value.producer.clone(),
                    value.producer_output.clone(),
                    consumer_region,
                ),
                materialization_id.clone(),
            );
            materializations.push(Materialization {
                materialization_id,
                value_id: value.value_id.clone(),
                version: value.version,
                producer_region: execution_regions[producer_region].region_id.clone(),
                consumer_region: execution_regions[consumer_region].region_id.clone(),
                size_bytes: value.size_bytes,
                live_start: execution_regions[producer_region].placement_anchor.clone(),
                live_end: execution_regions[consumer_region].placement_anchor.clone(),
            });
        }
    }

    let values_by_output: BTreeMap<(String, String), &ScheduleValue> = graph
        .values
        .iter()
        .filter(|value| !value.producer.is_empty() && !value.producer_output.is_empty())
        .map(|value| {
            (
                (value.producer.clone(), value.producer_output.clone()),
                value,
            )
        })
        .collect();
    let mut region_inputs = vec![Vec::<RegionInputBinding>::new(); execution_regions.len()];
    let mut region_outputs = vec![Vec::<RegionOutputBinding>::new(); execution_regions.len()];
    let mut region_invocations = vec![Vec::<RegionTaskInvocation>::new(); execution_regions.len()];
    for (region_index, region) in execution_regions.iter().enumerate() {
        let region_tasks: BTreeSet<&str> = region.tasks.iter().map(String::as_str).collect();
        let mut external_input_positions = BTreeMap::<String, usize>::new();
        for task_id in &region.tasks {
            let task = &graph.tasks[task_index[task_id]];
            let mut invocation_inputs = Vec::with_capacity(task.inputs.len());
            let mut invocation_outputs = Vec::with_capacity(task.outputs.len());
            for input in &task.inputs {
                let port = RegionPort {
                    task_id: task_id.clone(),
                    port_name: input.name.clone(),
                };
                let (
                    identity,
                    binding_id,
                    kind,
                    source,
                    layer,
                    layer_index,
                    layer_kind,
                    site,
                    phase,
                ) = if input.kind == ScheduleInputKind::Result {
                    if region_tasks.contains(input.source.as_str()) {
                        invocation_inputs.push(RegionTaskPortBinding {
                            port_name: input.name.clone(),
                            value_binding_id: format!(
                                "value-{}",
                                &digest(&format!(
                                    "{}\0{}\0{}",
                                    region.region_id, input.source, input.output
                                ))[..16]
                            ),
                        });
                        continue;
                    }
                    let value = values_by_output
                        .get(&(input.source.clone(), input.output.clone()))
                        .ok_or_else(|| {
                            schedule_error("typed Region result input has no ScheduleValue")
                        })?;
                    let materialization_id = materialization_by_value_region
                        .get(&(
                            value.value_id.clone(),
                            value.version,
                            value.producer.clone(),
                            value.producer_output.clone(),
                            region_index,
                        ))
                        .ok_or_else(|| {
                            schedule_error("typed Region result input has no materialization")
                        })?
                        .clone();
                    (
                        format!("materialized\0{materialization_id}"),
                        materialization_id,
                        RegionInputKind::Materialized,
                        value.value_id.clone(),
                        String::new(),
                        None,
                        HiddenLayerKind::None,
                        HiddenSite::None,
                        HiddenPhase::None,
                    )
                } else {
                    let region_kind = match input.kind {
                        ScheduleInputKind::Hidden => RegionInputKind::Hidden,
                        ScheduleInputKind::PackedRequestRows => RegionInputKind::PackedRequestRows,
                        ScheduleInputKind::PackedSamplerRows => RegionInputKind::PackedSamplerRows,
                        ScheduleInputKind::PackedHidden => RegionInputKind::PackedHidden,
                        ScheduleInputKind::StateSelection => RegionInputKind::StateSelection,
                        ScheduleInputKind::Provided => RegionInputKind::Provided,
                        ScheduleInputKind::Staged => RegionInputKind::Staged,
                        ScheduleInputKind::SamplerRequestIds => RegionInputKind::SamplerRequestIds,
                        ScheduleInputKind::SamplerSampleIdxs => RegionInputKind::SamplerSampleIdxs,
                        ScheduleInputKind::SamplerSampleIdxTensor => {
                            RegionInputKind::SamplerSampleIdxTensor
                        }
                        ScheduleInputKind::SamplerLogits => RegionInputKind::SamplerLogits,
                        ScheduleInputKind::SampledTokens => RegionInputKind::SampledTokens,
                        ScheduleInputKind::StepId => RegionInputKind::StepId,
                        ScheduleInputKind::Result => {
                            unreachable!("result inputs are handled before external bindings")
                        }
                    };
                    let identity = format!(
                        "{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}",
                        region_kind.as_wire(),
                        input.source,
                        input.layer,
                        input.layer_kind.as_wire(),
                        input.site.as_wire(),
                        input.phase.as_wire(),
                        input.packed_rows_port.as_deref().unwrap_or(""),
                        if input
                            .packed_rows_port
                            .as_deref()
                            .is_some_and(|port| !port.is_empty())
                        {
                            task_id.as_str()
                        } else {
                            ""
                        }
                    );
                    let layer_index = match input.layer_kind {
                        HiddenLayerKind::Index => {
                            let index = input.layer.parse::<i64>().map_err(|_| {
                                schedule_error("typed hidden layer index is not a signed integer")
                            })?;
                            Some(index)
                        }
                        HiddenLayerKind::None
                        | HiddenLayerKind::Path
                        | HiddenLayerKind::Wildcard => None,
                    };
                    (
                        identity.clone(),
                        format!(
                            "input-{}",
                            &digest(&format!("{}\0{}", region.region_id, identity))[..16]
                        ),
                        region_kind,
                        input.source.clone(),
                        input.layer.clone(),
                        layer_index,
                        input.layer_kind,
                        input.site,
                        input.phase,
                    )
                };
                invocation_inputs.push(RegionTaskPortBinding {
                    port_name: input.name.clone(),
                    value_binding_id: binding_id.clone(),
                });
                if let Some(position) = external_input_positions.get(&identity).copied() {
                    region_inputs[region_index][position].consumers.push(port);
                } else {
                    external_input_positions.insert(identity, region_inputs[region_index].len());
                    region_inputs[region_index].push(RegionInputBinding {
                        binding_id,
                        kind,
                        source,
                        consumers: vec![port],
                        layer,
                        layer_index,
                        layer_kind,
                        site,
                        phase,
                        packed_rows_binding_id: Some(String::new()),
                    });
                }
            }
            for output in &task.outputs {
                let value_binding_id = format!(
                    "value-{}",
                    &digest(&format!(
                        "{}\0{}\0{}",
                        region.region_id, task_id, output.name
                    ))[..16]
                );
                invocation_outputs.push(RegionTaskPortBinding {
                    port_name: output.name.clone(),
                    value_binding_id: value_binding_id.clone(),
                });
                let binding_kind = match output.kind {
                    ScheduleOutputKind::Sink => RegionOutputKind::Sink,
                    ScheduleOutputKind::Host => RegionOutputKind::Host,
                    ScheduleOutputKind::Sampler => RegionOutputKind::Sampler,
                    ScheduleOutputKind::Temporary => continue,
                };
                let identity = format!(
                    "{}\0{}\0{}\0{}",
                    region.region_id, task_id, output.name, output.target
                );
                region_outputs[region_index].push(RegionOutputBinding {
                    binding_id: format!("sink-{}", &digest(&identity)[..16]),
                    kind: binding_kind,
                    target: output.target.clone(),
                    producer: RegionPort {
                        task_id: task_id.clone(),
                        port_name: output.name.clone(),
                    },
                    value_binding_id,
                });
            }
            region_invocations[region_index].push(RegionTaskInvocation {
                task_id: task_id.clone(),
                inputs: invocation_inputs,
                outputs: invocation_outputs,
            });
        }
        let binding_id_by_port: BTreeMap<(String, String), String> = region_inputs[region_index]
            .iter()
            .flat_map(|binding| {
                binding.consumers.iter().map(|consumer| {
                    (
                        (consumer.task_id.clone(), consumer.port_name.clone()),
                        binding.binding_id.clone(),
                    )
                })
            })
            .collect();
        for binding in &mut region_inputs[region_index] {
            let mut related_rows = BTreeSet::new();
            for consumer in &binding.consumers {
                let task = &graph.tasks[task_index[&consumer.task_id]];
                let input = task
                    .inputs
                    .iter()
                    .find(|input| input.name == consumer.port_name)
                    .ok_or_else(|| schedule_error("typed Region input consumer port is missing"))?;
                let packed_rows_port = input.packed_rows_port.as_deref().unwrap_or("");
                if packed_rows_port.is_empty() {
                    continue;
                }
                let rows_binding_id = binding_id_by_port
                    .get(&(consumer.task_id.clone(), packed_rows_port.to_owned()))
                    .ok_or_else(|| {
                        schedule_error("packed input has no Region packed rows binding")
                    })?;
                related_rows.insert(rows_binding_id.clone());
            }
            if related_rows.len() > 1 {
                return Err(schedule_error(
                    "one Region input cannot reference multiple packed rows bindings",
                ));
            }
            binding.packed_rows_binding_id =
                Some(related_rows.into_iter().next().unwrap_or_default());
        }
    }
    for materialization in &materializations {
        let producer_region = *execution_regions
            .iter()
            .position(|region| region.region_id == materialization.producer_region)
            .as_ref()
            .ok_or_else(|| schedule_error("materialization producer Region is missing"))?;
        let value = graph
            .values
            .iter()
            .find(|value| {
                value.value_id == materialization.value_id
                    && value.version == materialization.version
            })
            .ok_or_else(|| schedule_error("materialization value is missing"))?;
        region_outputs[producer_region].push(RegionOutputBinding {
            binding_id: materialization.materialization_id.clone(),
            kind: RegionOutputKind::Materialized,
            target: materialization.value_id.clone(),
            producer: RegionPort {
                task_id: value.producer.clone(),
                port_name: value.producer_output.clone(),
            },
            value_binding_id: format!(
                "value-{}",
                &digest(&format!(
                    "{}\0{}\0{}",
                    execution_regions[producer_region].region_id,
                    value.producer,
                    value.producer_output
                ))[..16]
            ),
        });
    }
    for (index, region) in execution_regions.iter_mut().enumerate() {
        region.dependencies = region_dependencies[index].iter().cloned().collect();
        region.materialized_inputs = materialized_inputs[index].iter().cloned().collect();
        region.materialized_outputs = materialized_outputs[index].iter().cloned().collect();
        region.input_bindings = std::mem::take(&mut region_inputs[index]);
        region.output_bindings = std::mem::take(&mut region_outputs[index]);
        region.task_invocations = std::mem::take(&mut region_invocations[index]);
        let mut effects = Vec::new();
        for task_id in &region.tasks {
            let task = &graph.tasks[task_index[task_id]];
            for effect in &task.effects {
                let selection_port = effect.selection_port.as_deref().unwrap_or("");
                let selection_binding_id = if selection_port.is_empty() {
                    String::new()
                } else {
                    region
                        .input_bindings
                        .iter()
                        .find(|binding| {
                            binding.kind == RegionInputKind::StateSelection
                                && binding.source == effect.resource
                                && binding.consumers.iter().any(|port| {
                                    port.task_id == *task_id
                                        && port.port_name == selection_port
                                })
                        })
                        .map(|binding| binding.binding_id.clone())
                        .ok_or_else(|| {
                            schedule_error(
                                "request-scoped effect has no canonical Region state selection binding",
                            )
                        })?
                };
                effects.push(RegionEffect {
                    kind: effect.kind,
                    resource: effect.resource.clone(),
                    selection_binding_id,
                });
            }
        }
        region.effects = effects;
        region.capture_safe = region.capture_safe
            && region.effects.iter().all(|effect| {
                effect.kind == ScheduleEffectKind::StateRead
                    && state_has_capture_stable_storage(graph, &effect.resource)
            })
            && !region.input_bindings.is_empty()
            && region.input_bindings.iter().all(|binding| {
                matches!(
                    binding.kind,
                    RegionInputKind::Hidden
                        | RegionInputKind::Materialized
                        | RegionInputKind::SamplerSampleIdxTensor
                )
            });
        // Side-stream placement only exposes legal parallelism to the GPU
        // hardware scheduler; every cross-stream dependency remains an
        // explicit fence. Hidden/provided inputs and host/sink outputs stay on
        // the default stream; sampler outputs carry a signed after-wait.
        // Fully materialized training tails may carry only ordered state
        // effects: the runtime joins them before the next signed state reader,
        // allowing overlap with intervening main model work without a
        // workload-owned lane policy.
        if options.allow_side_stream && side_stream_eligible(graph.schema_version, region) {
            region.stream = Some(RegionStreamClass::SideForward);
        }
        let selected_async_host_safe = region
            .tasks
            .iter()
            .all(|task_id| selected[task_index[task_id]].async_host_safe);
        if async_host_eligible(graph.schema_version, region, selected_async_host_safe) {
            region.host_submission = RegionHostSubmission::AsyncSerial;
        }
        region.capture_mode = capture_mode_for_region(timeline, options, region)?;
        region.compile_command = compile_command_for_region(options, region);
        region.artifact_key = region_artifact_key(graph_json, timeline_json, options_json, region)?;
    }
    validate_materialization_bindings(graph, &execution_regions, &materializations)?;
    validate_capture_preflightability(&execution_regions, &materializations)?;

    let fences: Vec<Fence> = fence_kinds
        .iter()
        .enumerate()
        .map(|(ordinal, ((source, target), kind))| Fence {
            fence_id: format!("region-dependency/{ordinal}"),
            source: execution_regions[*source].region_id.clone(),
            target: execution_regions[*target].region_id.clone(),
            kind: kind.clone(),
        })
        .collect();
    let placements: Vec<Placement> = drafts
        .iter()
        .flat_map(|draft| draft.tasks.iter())
        .map(|index| {
            let region = &execution_regions[task_regions[*index]];
            Placement {
                task_id: graph.tasks[*index].task_id.clone(),
                after_anchor: region.placement_anchor.clone(),
                implementation_id: selected[*index].implementation_id.clone(),
                stream: "region".to_owned(),
            }
        })
        .collect();
    let allocations: Vec<Allocation> = drafts
        .iter()
        .flat_map(|draft| draft.tasks.iter())
        .filter_map(|index| {
            let task = &graph.tasks[*index];
            if task.temporary_bytes == 0 {
                return None;
            }
            let anchor = timeline.anchors[slots[*index]].anchor_id.clone();
            Some(Allocation {
                task_id: task.task_id.clone(),
                size_bytes: task.temporary_bytes,
                live_start: anchor.clone(),
                live_end: anchor,
            })
        })
        .collect();
    let materialization_bytes = materializations
        .iter()
        .map(|value| value.size_bytes)
        .fold(0u64, u64::saturating_add);
    let serial_fallback_count = execution_regions
        .iter()
        .filter(|region| region.serial_fallback)
        .count();
    let (mut runtime_advance_plan, completion_anchor) =
        runtime_advance_plan(timeline, &execution_regions)?;
    sign_async_capture_commands(
        graph,
        graph_json,
        timeline,
        timeline_json,
        options,
        options_json,
        &selected,
        &task_index,
        &execution_regions,
        &materializations,
        &mut runtime_advance_plan,
    )?;
    let capture_boundary_count = execution_regions
        .iter()
        .filter(|region| region.capture_mode == RegionCaptureMode::Require)
        .count() as u64
        + runtime_advance_plan
            .iter()
            .flat_map(|advance| advance.host_commands.iter())
            .filter(|command| {
                command.async_capture_mode == AsyncBatchCaptureMode::OptimizerTransition
            })
            .count() as u64;
    let mut decision = Decision {
        graph_key: digest(graph_json),
        timeline_key: digest(timeline_json),
        execution_order: execution_order.clone(),
        placements,
        schedule_tree: ScheduleTree {
            kind: if affine_regions.is_empty() {
                "sequence".to_owned()
            } else {
                "hybrid".to_owned()
            },
            tasks: execution_order,
            affine_regions,
        },
        memory_plan: MemoryPlan {
            budget_bytes: budget,
            persistent_bytes: graph.persistent_state_bytes,
            peak_bytes: peak,
            allocations,
        },
        execution_regions,
        runtime_advance_plan: Some(runtime_advance_plan),
        completion_anchor: Some(completion_anchor),
        materializations,
        moves: Vec::new(),
        fences,
        capture_partitions: Vec::new(),
        fallback: "explicit_regions".to_owned(),
        cost: Cost {
            peak_memory_bytes: peak,
            transfer_bytes: 0,
            main_path_extension_ns: 0,
            compute_cost: graph.tasks.len() as u64,
            materialization_bytes,
            capture_boundary_count,
            launch_group_count: drafts.len() as u64,
            cross_region_fence_count: fence_kinds.len() as u64,
            critical_path_extension: draft_partition_cost(
                graph,
                &selected,
                &task_index,
                &drafts,
                false,
            )
            .3,
        },
        explanations: {
            let mut explanations = vec![
                "selected deterministic maximal legal ExecutionRegion partition".to_owned(),
                format!("materialized {materialization_bytes} bytes across Region boundaries"),
                format!("selected {} launch groups", drafts.len()),
                format!(
                    "selected structural critical-path extension {}",
                    draft_partition_cost(graph, &selected, &task_index, &drafts, false).3
                ),
                format!("selected {serial_fallback_count} explicit serial fallback regions"),
                format!("effective memory budget is {budget} bytes"),
            ];
            for (workload_id, policy) in &graph.workload_approximations {
                explanations.push(format!(
                    "workload {workload_id} signs approximation policy {policy}: Region parity uses declaration-supplied criteria"
                ));
            }
            explanations
        },
        signature: String::new(),
        schema_version: 7,
    };
    let unsigned = serde_json::to_string(&decision)
        .map_err(|error| schedule_error(format!("failed to serialize decision: {error}")))?;
    decision.signature = digest(&unsigned);
    Ok(decision)
}

#[pyfunction]
fn compile_schedule(graph: &str, timeline: &str, options: &str) -> PyResult<String> {
    let graph_desc: ScheduleGraph = decode(graph, "ScheduleGraph")?;
    let timeline_desc: Timeline = decode(timeline, "MainWorkloadTimeline")?;
    let options_desc: CompileOptions = decode(options, "CompileOptions")?;
    let decision = compile_impl(
        &graph_desc,
        graph,
        &timeline_desc,
        timeline,
        &options_desc,
        options,
    )?;
    serde_json::to_string(&decision)
        .map_err(|error| schedule_error(format!("failed to serialize decision: {error}")))
}

#[pyfunction]
fn validate_schedule(graph: &str, timeline: &str, options: &str, decision: &str) -> PyResult<()> {
    let graph_desc: ScheduleGraph = decode(graph, "ScheduleGraph")?;
    let timeline_desc: Timeline = decode(timeline, "MainWorkloadTimeline")?;
    let options_desc: CompileOptions = decode(options, "CompileOptions")?;
    let supplied: Decision = decode(decision, "ScheduleDecision")?;
    let (anchor_index, task_index) = validate_inputs(&graph_desc, &timeline_desc, &options_desc)?;
    let selected = select_task_implementations(&graph_desc, &timeline_desc)?;
    if supplied.schema_version != 7
        || supplied.graph_key != digest(graph)
        || supplied.timeline_key != digest(timeline)
    {
        return Err(schedule_error(
            "ScheduleDecision identity or schema does not match its inputs",
        ));
    }
    if supplied
        .execution_regions
        .iter()
        .any(|region| region.stream.is_none() || region.fallback_policy.is_none())
    {
        return Err(schedule_error(
            "ScheduleDecision v7 ExecutionRegion requires explicit stream and fallback_policy",
        ));
    }
    if supplied.execution_regions.iter().any(|region| {
        (region.implementation_kind == RegionImplementationKind::Compiled)
            != region.compile_command.is_some()
    }) {
        return Err(schedule_error(
            "ScheduleDecision v7 compiled ExecutionRegion requires a compile_command and extern Region forbids one",
        ));
    }
    let mut unsigned = supplied.clone();
    let signature = std::mem::take(&mut unsigned.signature);
    let unsigned_json = serde_json::to_string(&unsigned)
        .map_err(|error| schedule_error(format!("failed to serialize decision: {error}")))?;
    if signature != digest(&unsigned_json) {
        return Err(schedule_error("ScheduleDecision signature is invalid"));
    }
    if supplied.fallback != "explicit_regions" || !supplied.capture_partitions.is_empty() {
        return Err(schedule_error(
            "ScheduleDecision v7 requires explicit regions and forbids capture_partitions",
        ));
    }
    if supplied.execution_order.len() != graph_desc.tasks.len()
        || supplied.placements.len() != graph_desc.tasks.len()
        || supplied.schedule_tree.tasks != supplied.execution_order
    {
        return Err(schedule_error(
            "ScheduleDecision does not cover every task exactly once",
        ));
    }
    validate_affine_regions(&graph_desc, &task_index, &supplied.schedule_tree)?;
    let mut order_position = HashMap::new();
    for (position, task_id) in supplied.execution_order.iter().enumerate() {
        if !task_index.contains_key(task_id)
            || order_position.insert(task_id.clone(), position).is_some()
        {
            return Err(schedule_error(
                "ScheduleDecision execution order contains an unknown or duplicate task",
            ));
        }
    }
    let mut region_position = HashMap::new();
    let mut task_region = HashMap::new();
    let mut flattened = Vec::new();
    for (position, region) in supplied.execution_regions.iter().enumerate() {
        if region.region_id.is_empty()
            || region.tasks.is_empty()
            || region_position
                .insert(region.region_id.as_str(), position)
                .is_some()
        {
            return Err(schedule_error(
                "ScheduleDecision contains an invalid or duplicate ExecutionRegion",
            ));
        }
        let (Some(after), Some(before), Some(placement)) = (
            anchor_index.get(&region.after_anchor).copied(),
            anchor_index.get(&region.before_anchor).copied(),
            anchor_index.get(&region.placement_anchor).copied(),
        ) else {
            return Err(schedule_error(
                "ExecutionRegion has an invalid anchor window or placement",
            ));
        };
        if after > placement || placement >= before {
            return Err(schedule_error(
                "ExecutionRegion has an invalid anchor window or placement",
            ));
        }
        if region.artifact_key.len() != 64
            || region
                .artifact_key
                .chars()
                .any(|character| !character.is_ascii_hexdigit() || character.is_ascii_uppercase())
        {
            return Err(schedule_error(
                "ExecutionRegion artifact key is not lowercase SHA-256",
            ));
        }
        for task in &region.tasks {
            if !task_index.contains_key(task)
                || task_region
                    .insert(task.as_str(), region.region_id.as_str())
                    .is_some()
            {
                return Err(schedule_error(
                    "ExecutionRegion task membership is unknown or duplicated",
                ));
            }
            flattened.push(task.clone());
        }
    }
    if flattened != supplied.execution_order {
        return Err(schedule_error(
            "ExecutionRegion partition does not cover execution_order exactly once",
        ));
    }
    for (position, region) in supplied.execution_regions.iter().enumerate() {
        let dependencies: BTreeSet<&str> = region.dependencies.iter().map(String::as_str).collect();
        if dependencies.len() != region.dependencies.len()
            || dependencies.iter().any(|dependency| {
                region_position
                    .get(dependency)
                    .is_none_or(|dependency_position| *dependency_position >= position)
            })
        {
            return Err(schedule_error(
                "ExecutionRegion dependencies must reference unique earlier regions",
            ));
        }
        if region.serial_fallback && !options_desc.allow_serial_fallback {
            return Err(schedule_error(
                "serial fallback region is disabled by CompileOptions",
            ));
        }
        if region.stream == Some(RegionStreamClass::SideForward) {
            if !options_desc.allow_side_stream
                || !side_stream_eligible(graph_desc.schema_version, region)
            {
                return Err(schedule_error(
                    "side-stream ExecutionRegion requires scheduler enablement, stream-safe implementation evidence, typed materializations, and ordered effects",
                ));
            }
        }
        let selected_async_host_safe = region
            .tasks
            .iter()
            .all(|task_id| selected[task_index[task_id]].async_host_safe);
        let expected_host_submission =
            if async_host_eligible(graph_desc.schema_version, region, selected_async_host_safe) {
                RegionHostSubmission::AsyncSerial
            } else {
                RegionHostSubmission::Inline
            };
        if region.host_submission != expected_host_submission {
            return Err(schedule_error(
                "ExecutionRegion host_submission is inconsistent with selected async-host evidence and typed Region contracts",
            ));
        }
        if let Some(command) = &region.compile_command {
            let expected_adapter =
                if region.compile_safe && options_desc.compile_policy == "torch_compile" {
                    RegionCompilerAdapter::TorchCompile
                } else {
                    RegionCompilerAdapter::Eager
                };
            if command.adapter != expected_adapter
                || command.fullgraph != (command.adapter == RegionCompilerAdapter::TorchCompile)
                || command.dynamic
                || command.shape_specialization != options_desc.shape_specialization
                || command.autograd_policy != RegionAutogradPolicy::EnableGrad
                || command.fallback_policy != RegionFallbackPolicy::Reject
                || region.fallback_policy != Some(command.fallback_policy)
            {
                return Err(schedule_error(
                    "ExecutionRegion compile_command is inconsistent with the signed Rust compile policy",
                ));
            }
        }
        let region_tasks: BTreeSet<&str> = region.tasks.iter().map(String::as_str).collect();
        let materialized_inputs: BTreeSet<&str> = region
            .materialized_inputs
            .iter()
            .map(String::as_str)
            .collect();
        let materialized_outputs: BTreeSet<&str> = region
            .materialized_outputs
            .iter()
            .map(String::as_str)
            .collect();
        let mut binding_ids = BTreeSet::new();
        let mut covered_inputs = BTreeSet::new();
        let mut input_binding_by_port = BTreeMap::new();
        for binding in &region.input_bindings {
            let packed_rows_binding_id =
                binding.packed_rows_binding_id.as_deref().ok_or_else(|| {
                    schedule_error(
                        "ScheduleDecision v7 input bindings require packed_rows_binding_id",
                    )
                })?;
            if binding.binding_id.is_empty()
                || binding.source.is_empty()
                || binding.consumers.is_empty()
                || !binding_ids.insert(binding.binding_id.as_str())
                || (binding.kind == RegionInputKind::Materialized
                    && !materialized_inputs.contains(binding.binding_id.as_str()))
                || (binding.kind == RegionInputKind::Hidden
                    && (binding.layer_kind == HiddenLayerKind::None
                        || binding.site == HiddenSite::None
                        || binding.phase == HiddenPhase::None))
                || (binding.kind == RegionInputKind::PackedHidden
                    && (!matches!(
                        binding.layer_kind,
                        HiddenLayerKind::Index | HiddenLayerKind::Path
                    ) || binding.site != HiddenSite::BlockOutput
                        || binding.phase != HiddenPhase::None
                        || packed_rows_binding_id.is_empty()))
                || (binding.kind == RegionInputKind::PackedRequestRows
                    && (binding.source != "runtime.packed_phase_rows"
                        || !packed_rows_binding_id.is_empty()))
                || (binding.kind == RegionInputKind::PackedSamplerRows
                    && (binding.source != "runtime.packed_sampler_rows"
                        || packed_rows_binding_id.is_empty()))
                || (binding.kind == RegionInputKind::StateSelection
                    && (binding.layer_kind != HiddenLayerKind::None
                        || binding.site != HiddenSite::None
                        || (packed_rows_binding_id.is_empty()
                            == matches!(
                                binding.phase,
                                HiddenPhase::Prefill | HiddenPhase::Decode
                            ))))
                || (binding.layer_kind == HiddenLayerKind::Index && binding.layer_index.is_none())
                || (binding.layer_kind != HiddenLayerKind::Index && binding.layer_index.is_some())
                || (!matches!(
                    binding.kind,
                    RegionInputKind::Hidden
                        | RegionInputKind::PackedHidden
                        | RegionInputKind::StateSelection
                ) && (binding.layer_kind != HiddenLayerKind::None
                    || binding.site != HiddenSite::None
                    || binding.phase != HiddenPhase::None))
                || (!matches!(
                    binding.kind,
                    RegionInputKind::PackedHidden
                        | RegionInputKind::PackedSamplerRows
                        | RegionInputKind::StateSelection
                ) && !packed_rows_binding_id.is_empty())
            {
                return Err(schedule_error(
                    "ExecutionRegion has an invalid typed input binding",
                ));
            }
            for port in &binding.consumers {
                if !region_tasks.contains(port.task_id.as_str())
                    || !covered_inputs.insert((port.task_id.as_str(), port.port_name.as_str()))
                {
                    return Err(schedule_error(
                        "ExecutionRegion typed input ports are unknown or duplicated",
                    ));
                }
                let task = &graph_desc.tasks[task_index[&port.task_id]];
                let input = task
                    .inputs
                    .iter()
                    .find(|input| input.name == port.port_name);
                let Some(input) = input else {
                    return Err(schedule_error(
                        "ExecutionRegion typed input references an undeclared task port",
                    ));
                };
                if (input.kind == ScheduleInputKind::StateSelection)
                    != (binding.kind == RegionInputKind::StateSelection)
                    || (input.kind == ScheduleInputKind::StateSelection
                        && (input.source != binding.source || input.phase != binding.phase))
                {
                    return Err(schedule_error(
                        "ExecutionRegion state selection binding disagrees with its task input",
                    ));
                }
                if (input.kind == ScheduleInputKind::PackedRequestRows)
                    != (binding.kind == RegionInputKind::PackedRequestRows)
                    || (input.kind == ScheduleInputKind::PackedSamplerRows)
                        != (binding.kind == RegionInputKind::PackedSamplerRows)
                    || (input.kind == ScheduleInputKind::PackedHidden)
                        != (binding.kind == RegionInputKind::PackedHidden)
                {
                    return Err(schedule_error(
                        "ExecutionRegion packed input binding disagrees with its task input",
                    ));
                }
                input_binding_by_port
                    .insert((port.task_id.as_str(), port.port_name.as_str()), binding);
            }
        }
        let packed_rows_bindings: BTreeMap<&str, &RegionInputBinding> = region
            .input_bindings
            .iter()
            .filter(|binding| binding.kind == RegionInputKind::PackedRequestRows)
            .map(|binding| (binding.binding_id.as_str(), binding))
            .collect();
        for binding in &region.input_bindings {
            let relation = binding
                .packed_rows_binding_id
                .as_deref()
                .expect("v7 field presence was validated");
            if !relation.is_empty() {
                let rows_binding = packed_rows_bindings.get(relation).ok_or_else(|| {
                    schedule_error(
                        "ExecutionRegion packed rows binding does not name packed_request_rows",
                    )
                })?;
                let binding_tasks: BTreeSet<&str> = binding
                    .consumers
                    .iter()
                    .map(|port| port.task_id.as_str())
                    .collect();
                let rows_tasks: BTreeSet<&str> = rows_binding
                    .consumers
                    .iter()
                    .map(|port| port.task_id.as_str())
                    .collect();
                if binding_tasks.is_disjoint(&rows_tasks) {
                    return Err(schedule_error(
                        "ExecutionRegion packed rows binding has no shared consumer task",
                    ));
                }
            }
            for consumer in &binding.consumers {
                let task = &graph_desc.tasks[task_index[&consumer.task_id]];
                let input = task
                    .inputs
                    .iter()
                    .find(|input| input.name == consumer.port_name)
                    .expect("consumer input presence was validated");
                let rows_port = input.packed_rows_port.as_deref().unwrap_or("");
                let expected_relation = if rows_port.is_empty() {
                    ""
                } else {
                    input_binding_by_port
                        .get(&(consumer.task_id.as_str(), rows_port))
                        .ok_or_else(|| {
                            schedule_error(
                                "ExecutionRegion packed input has no same-task rows binding",
                            )
                        })?
                        .binding_id
                        .as_str()
                };
                if relation != expected_relation {
                    return Err(schedule_error(
                        "ExecutionRegion packed rows binding disagrees with its task relation",
                    ));
                }
            }
        }
        let mut expected_effects = Vec::new();
        for task_id in &region.tasks {
            let task = &graph_desc.tasks[task_index[task_id]];
            for effect in &task.effects {
                let selection_port = effect.selection_port.as_deref().unwrap_or("");
                let selection_binding_id = if selection_port.is_empty() {
                    String::new()
                } else {
                    let binding = input_binding_by_port
                        .get(&(task_id.as_str(), selection_port))
                        .ok_or_else(|| {
                            schedule_error(
                                "ExecutionRegion request effect selection port is not bound",
                            )
                        })?;
                    if binding.kind != RegionInputKind::StateSelection
                        || binding.source != effect.resource
                    {
                        return Err(schedule_error(
                            "ExecutionRegion request effect selection binding has the wrong source or kind",
                        ));
                    }
                    binding.binding_id.clone()
                };
                expected_effects.push(RegionEffect {
                    kind: effect.kind,
                    resource: effect.resource.clone(),
                    selection_binding_id,
                });
            }
        }
        if region.effects != expected_effects {
            return Err(schedule_error(
                "ExecutionRegion effects disagree with typed task effects and state selection bindings",
            ));
        }
        if region.task_invocations.len() != region.tasks.len() {
            return Err(schedule_error(
                "ExecutionRegion task invocations do not cover Region tasks",
            ));
        }
        let mut available_values: BTreeSet<&str> = region
            .input_bindings
            .iter()
            .map(|binding| binding.binding_id.as_str())
            .collect();
        let mut task_output_values = BTreeMap::<(&str, &str), &str>::new();
        for (invocation_position, invocation) in region.task_invocations.iter().enumerate() {
            if invocation.task_id != region.tasks[invocation_position] {
                return Err(schedule_error(
                    "ExecutionRegion task invocation order differs from task order",
                ));
            }
            let task = &graph_desc.tasks[task_index[&invocation.task_id]];
            if invocation.inputs.len() != task.inputs.len()
                || invocation.outputs.len() != task.outputs.len()
                || invocation
                    .inputs
                    .iter()
                    .zip(task.inputs.iter())
                    .any(|(binding, input)| {
                        binding.port_name != input.name
                            || binding.value_binding_id.is_empty()
                            || !available_values.contains(binding.value_binding_id.as_str())
                    })
                || invocation
                    .outputs
                    .iter()
                    .zip(task.outputs.iter())
                    .any(|(binding, output)| {
                        binding.port_name != output.name || binding.value_binding_id.is_empty()
                    })
            {
                return Err(schedule_error(
                    "ExecutionRegion has an invalid signed task invocation",
                ));
            }
            for output in &invocation.outputs {
                if !available_values.insert(output.value_binding_id.as_str())
                    || task_output_values
                        .insert(
                            (invocation.task_id.as_str(), output.port_name.as_str()),
                            output.value_binding_id.as_str(),
                        )
                        .is_some()
                {
                    return Err(schedule_error(
                        "ExecutionRegion task output value bindings are duplicated",
                    ));
                }
            }
        }
        let mut covered_outputs = BTreeSet::new();
        for binding in &region.output_bindings {
            if binding.binding_id.is_empty()
                || binding.target.is_empty()
                || !binding_ids.insert(binding.binding_id.as_str())
                || binding.value_binding_id.is_empty()
                || !region_tasks.contains(binding.producer.task_id.as_str())
                || !covered_outputs.insert((
                    binding.producer.task_id.as_str(),
                    binding.producer.port_name.as_str(),
                    binding.kind,
                    binding.binding_id.as_str(),
                ))
                || (binding.kind == RegionOutputKind::Materialized
                    && !materialized_outputs.contains(binding.binding_id.as_str()))
                || task_output_values
                    .get(&(
                        binding.producer.task_id.as_str(),
                        binding.producer.port_name.as_str(),
                    ))
                    .is_none_or(|value_id| *value_id != binding.value_binding_id)
            {
                return Err(schedule_error(
                    "ExecutionRegion has an invalid typed output binding",
                ));
            }
            let task = &graph_desc.tasks[task_index[&binding.producer.task_id]];
            if !task
                .outputs
                .iter()
                .any(|output| output.name == binding.producer.port_name)
            {
                return Err(schedule_error(
                    "ExecutionRegion typed output references an undeclared task port",
                ));
            }
        }
    }

    let supplied_runtime_plan = supplied.runtime_advance_plan.as_ref().ok_or_else(|| {
        schedule_error("ScheduleDecision v7 requires an explicit runtime_advance_plan")
    })?;
    let supplied_completion_anchor = supplied.completion_anchor.as_deref().ok_or_else(|| {
        schedule_error("ScheduleDecision v7 requires an explicit completion_anchor")
    })?;
    let supplied_regions_by_id: HashMap<&str, &ExecutionRegion> = supplied
        .execution_regions
        .iter()
        .map(|region| (region.region_id.as_str(), region))
        .collect();
    for advance in supplied_runtime_plan {
        validate_host_commands(
            &advance.region_ids,
            &advance.host_commands,
            &supplied_regions_by_id,
        )?;
    }
    if supplied_completion_anchor.is_empty() {
        return Err(schedule_error(
            "ScheduleDecision completion_anchor must be non-empty",
        ));
    }
    let (mut canonical_runtime_plan, canonical_completion_anchor) =
        runtime_advance_plan(&timeline_desc, &supplied.execution_regions)?;
    sign_async_capture_commands(
        &graph_desc,
        graph,
        &timeline_desc,
        timeline,
        &options_desc,
        options,
        &selected,
        &task_index,
        &supplied.execution_regions,
        &supplied.materializations,
        &mut canonical_runtime_plan,
    )?;
    if supplied_runtime_plan != &canonical_runtime_plan
        || supplied_completion_anchor != canonical_completion_anchor
    {
        return Err(schedule_error(
            "ScheduleDecision runtime advancement or completion metadata is inconsistent",
        ));
    }

    let mut placement_by_task: HashMap<&str, &Placement> = HashMap::new();
    for placement in &supplied.placements {
        let task_position = match task_index.get(&placement.task_id) {
            Some(position) => *position,
            None => {
                return Err(schedule_error(
                    "ScheduleDecision placement references an unknown task",
                ))
            }
        };
        if placement_by_task
            .insert(placement.task_id.as_str(), placement)
            .is_some()
        {
            return Err(schedule_error(
                "ScheduleDecision contains duplicate placements",
            ));
        }
        let task = &graph_desc.tasks[task_position];
        if !task
            .implementation_ids
            .contains(&placement.implementation_id)
        {
            return Err(schedule_error(format!(
                "task {:?} selected an undeclared implementation",
                task.task_id
            )));
        }
        if placement.stream != "region" {
            return Err(schedule_error(
                "schema v2 task placements must identify Region execution",
            ));
        }
        let slot = match anchor_index.get(&placement.after_anchor) {
            Some(slot) => *slot,
            None => return Err(schedule_error("placement references an unknown anchor")),
        };
        let declared_earliest = anchor_index[&task.timing.after];
        let earliest = if task
            .inputs
            .iter()
            .any(|input| input.kind == ScheduleInputKind::StateSelection)
        {
            declared_earliest.max(state_selection_ready_index(&timeline_desc).ok_or_else(|| {
                schedule_error(
                    "state_selection input requires prepare_inputs.post placement evidence",
                )
            })?)
        } else {
            declared_earliest
        };
        let latest = anchor_index[&task.timing.before] - 1;
        if slot < earliest || slot > latest {
            return Err(schedule_error(format!(
                "task {:?} placement violates its timing window",
                task.task_id
            )));
        }
        let region_id = task_region[placement.task_id.as_str()];
        let region = &supplied.execution_regions[region_position[region_id]];
        if placement.after_anchor != region.placement_anchor {
            return Err(schedule_error(
                "task placement does not match its owning ExecutionRegion",
            ));
        }
    }

    let expected = compile_impl(
        &graph_desc,
        graph,
        &timeline_desc,
        timeline,
        &options_desc,
        options,
    )?;
    if supplied.execution_regions != expected.execution_regions {
        return Err(schedule_error(
            "ScheduleDecision ExecutionRegion partition or artifact metadata is inconsistent",
        ));
    }
    if supplied.materializations != expected.materializations {
        return Err(schedule_error(
            "ScheduleDecision cross-Region materialization or lifetime is inconsistent",
        ));
    }
    if supplied.fences != expected.fences {
        return Err(schedule_error(
            "ScheduleDecision fences must cover only cross-Region dependencies",
        ));
    }
    if supplied.placements != expected.placements
        || supplied.schedule_tree != expected.schedule_tree
        || supplied.memory_plan != expected.memory_plan
        || supplied.moves != expected.moves
        || supplied.cost != expected.cost
        || supplied.explanations != expected.explanations
        || supplied.signature != expected.signature
    {
        return Err(schedule_error(
            "ScheduleDecision metadata is inconsistent with canonical Region lowering",
        ));
    }
    Ok(())
}

/// Validate an unparsed coordinator-to-worker compilation envelope.
///
/// This deliberately consumes the original JSON strings.  Python object
/// decoding may normalize optional fields and would otherwise make unknown or
/// duplicate JSON fields disappear before the signed schedule validator sees
/// them.
#[pyfunction]
fn validate_workload_compilation_wire(
    cache_key: &str,
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    decision_json: &str,
    plan_epoch: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if !plan_epoch.is_exact_instance_of::<PyInt>() {
        return Err(value_error("plan_epoch must be a positive integer"));
    }
    let plan_epoch: u64 = plan_epoch
        .extract()
        .map_err(|_| value_error("plan_epoch must be a positive integer"))?;
    if plan_epoch == 0 {
        return Err(value_error("plan_epoch must be a positive integer"));
    }
    validate_workload_compilation_wire_inputs(
        cache_key,
        graph_json,
        timeline_json,
        options_json,
        decision_json,
        plan_epoch,
    )
}

fn validate_workload_compilation_wire_inputs(
    cache_key: &str,
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    decision_json: &str,
    plan_epoch: u64,
) -> PyResult<()> {
    if plan_epoch == 0 {
        return Err(value_error("plan_epoch must be a positive integer"));
    }
    if !is_lowercase_sha256(cache_key) {
        return Err(value_error("cache_key must be a lowercase SHA-256 digest"));
    }

    // `validate_schedule` decodes the raw payloads with strict serde records,
    // verifies the decision signature, and compares the canonical Region plan.
    validate_schedule(graph_json, timeline_json, options_json, decision_json)?;

    if cache_key != workload_compilation_wire_cache_key(graph_json, timeline_json, options_json) {
        return Err(value_error(
            "workload compilation cache key does not match wire inputs",
        ));
    }
    Ok(())
}

fn worker_spec_projection<'a, I>(entries: I, label: &str) -> PyResult<Vec<(String, String)>>
where
    I: IntoIterator<Item = (&'a str, &'a serde_json::Value)>,
{
    let mut projection = Vec::new();
    for (factory, config) in entries {
        if factory.is_empty() || !config.is_object() {
            return Err(value_error(format!(
                "{label} require a non-empty factory and JSON object config"
            )));
        }
        projection.push((
            factory.to_owned(),
            canonical_ascii_json(config).map_err(value_error)?,
        ));
    }
    projection.sort();
    Ok(projection)
}

/// Validate executable worker import recipes before Python resolves a module
/// or invokes a factory. The full compilation wire is revalidated in the same
/// Rust call so the static manifest cannot be detached from its cache/signature
/// identity.
#[pyfunction]
fn validate_worker_reconstruction_specs(
    cache_key: &str,
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    decision_json: &str,
    plan_epoch: &Bound<'_, PyAny>,
    reconstruction_specs_json: &str,
) -> PyResult<()> {
    if !plan_epoch.is_exact_instance_of::<PyInt>() {
        return Err(value_error("plan_epoch must be a positive integer"));
    }
    let plan_epoch: u64 = plan_epoch
        .extract()
        .map_err(|_| value_error("plan_epoch must be a positive integer"))?;
    validate_workload_compilation_wire_inputs(
        cache_key,
        graph_json,
        timeline_json,
        options_json,
        decision_json,
        plan_epoch,
    )?;

    let options: CompileOptions = decode(options_json, "CompileOptions")?;
    let signed = validate_static_backend_binding_manifest(&options)?;
    let submitted: Vec<WorkerReconstructionSpec> =
        decode(reconstruction_specs_json, "worker reconstruction specs")?;
    let signed_projection = worker_spec_projection(
        signed
            .iter()
            .map(|binding| (binding.factory.as_str(), &binding.config)),
        "signed worker reconstruction specs",
    )?;
    let submitted_projection = worker_spec_projection(
        submitted
            .iter()
            .map(|spec| (spec.factory.as_str(), &spec.config)),
        "worker reconstruction specs",
    )?;
    if submitted_projection != signed_projection {
        return Err(schedule_error(
            "worker reconstruction specs do not match the parent-signed static binding manifest",
        ));
    }
    Ok(())
}

/// Runtime facts observed by a worker before it binds process-local resources.
///
/// The JSON object is deliberately closed and nullable at the FFI boundary.
/// `null` means the worker has no runtime specialization observation.  A
/// partial object is never meaningful because the base facts jointly identify
/// the compiled Region artifact.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSpecializationFacts {
    device: String,
    dtype: String,
    hidden_size: u64,
    rows: u64,
    #[serde(default)]
    max_prefill_tokens: Option<u64>,
    #[serde(default)]
    model_layer_count: Option<u64>,
}

impl RuntimeSpecializationFacts {
    fn validate(&self) -> PyResult<()> {
        if self.device.is_empty()
            || self.dtype.is_empty()
            || self.hidden_size == 0
            || self.rows == 0
            || self.max_prefill_tokens == Some(0)
            || self.model_layer_count == Some(0)
        {
            return Err(value_error(
                "runtime specialization facts require non-empty device/dtype and positive rows/hidden_size/max_prefill_tokens",
            ));
        }
        Ok(())
    }

    fn as_signed_entries(&self) -> Vec<(&'static str, String)> {
        let mut entries = vec![
            ("runtime/device", self.device.clone()),
            ("runtime/dtype", self.dtype.clone()),
            ("runtime/hidden_size", self.hidden_size.to_string()),
            ("runtime/rows", self.rows.to_string()),
        ];
        if let Some(model_layer_count) = self.model_layer_count {
            entries.push(("runtime/model_layer_count", model_layer_count.to_string()));
        }
        if let Some(max_prefill_tokens) = self.max_prefill_tokens {
            entries.push(("runtime/max_prefill_tokens", max_prefill_tokens.to_string()));
        }
        entries
    }
}

fn validate_runtime_specialization_facts(
    options: &CompileOptions,
    facts_json: &str,
) -> PyResult<()> {
    let facts: Option<RuntimeSpecializationFacts> =
        decode(facts_json, "nullable RuntimeSpecializationFacts")?;
    if let Some(facts) = &facts {
        facts.validate()?;
    }

    const REQUIRED_KEYS: [&str; 4] = [
        "runtime/device",
        "runtime/dtype",
        "runtime/hidden_size",
        "runtime/rows",
    ];
    let signed: BTreeMap<&str, &str> = options
        .shape_specialization
        .iter()
        .filter_map(|(key, value)| {
            (REQUIRED_KEYS.contains(&key.as_str())
                || key == "runtime/model_layer_count"
                || key == "runtime/max_prefill_tokens")
                .then_some((key.as_str(), value.as_str()))
        })
        .collect();

    if signed.is_empty() {
        if facts.is_some() {
            return Err(schedule_error(
                "worker runtime specialization facts are unsigned by CompileOptions",
            ));
        }
        return Ok(());
    }
    if !REQUIRED_KEYS.iter().all(|key| signed.contains_key(key)) {
        return Err(schedule_error(
            "CompileOptions runtime specialization must contain all signed facts",
        ));
    }
    let facts = facts.ok_or_else(|| {
        schedule_error("worker runtime specialization facts are required by CompileOptions")
    })?;
    let observed: BTreeMap<&str, String> = facts.as_signed_entries().into_iter().collect();
    for (key, expected) in signed {
        if observed.get(key).map(String::as_str) != Some(expected) {
            return Err(schedule_error(
                "worker specialization does not match the parent-signed CompileOptions",
            ));
        }
    }
    Ok(())
}

fn normalize_runtime_specialization_options_wire(
    options_json: &str,
    facts_json: &str,
) -> PyResult<String> {
    let mut options: CompileOptions = decode(options_json, "CompileOptions")?;
    validate_compile_options_preflight(&options)?;
    let facts: Option<RuntimeSpecializationFacts> =
        decode(facts_json, "nullable RuntimeSpecializationFacts")?;
    if let Some(facts) = &facts {
        facts.validate()?;
    }

    const REQUIRED_KEYS: [&str; 4] = [
        "runtime/device",
        "runtime/dtype",
        "runtime/hidden_size",
        "runtime/rows",
    ];
    let mut entries: BTreeMap<String, String> = BTreeMap::new();
    for (key, value) in options.shape_specialization {
        if key.is_empty() || value.is_empty() || entries.insert(key.clone(), value).is_some() {
            return Err(value_error(
                "shape specialization requires sorted unique non-empty keys",
            ));
        }
    }
    let signed: BTreeMap<&str, &str> = entries
        .iter()
        .filter_map(|(key, value)| {
            (REQUIRED_KEYS.contains(&key.as_str())
                || key == "runtime/model_layer_count"
                || key == "runtime/max_prefill_tokens")
                .then_some((key.as_str(), value.as_str()))
        })
        .collect();
    if !signed.is_empty() && !REQUIRED_KEYS.iter().all(|key| signed.contains_key(key)) {
        return Err(schedule_error(
            "CompileOptions runtime specialization must contain all signed facts",
        ));
    }

    match (facts, signed.is_empty()) {
        (None, true) => {}
        (None, false) => {
            return Err(schedule_error(
                "worker runtime specialization facts are required by CompileOptions",
            ));
        }
        (Some(facts), true) => {
            for (key, value) in facts.as_signed_entries() {
                entries.insert(key.to_owned(), value);
            }
        }
        (Some(facts), false) => {
            let observed: BTreeMap<&str, String> = facts.as_signed_entries().into_iter().collect();
            for (key, expected) in signed {
                if observed.get(key).map(String::as_str) != Some(expected) {
                    return Err(schedule_error(
                        "worker specialization does not match the parent-signed CompileOptions",
                    ));
                }
            }
        }
    }
    options.shape_specialization = entries.into_iter().collect();
    canonical_ascii_json(&options).map_err(value_error)
}

/// Normalize nullable worker observations into the signed CompileOptions
/// identity. Python transports raw JSON and never infers runtime keys.
#[pyfunction]
fn normalize_runtime_specialization_options(
    options_json: &str,
    facts_json: &str,
) -> PyResult<String> {
    normalize_runtime_specialization_options_wire(options_json, facts_json)
}

fn validate_reconstructed_workload_wire_inputs(
    cache_key: &str,
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    decision_json: &str,
    plan_epoch: u64,
    reconstructed_graph_json: &str,
    reconstructed_residual_resource_manifest_json: &str,
    reconstructed_packed_hidden_resource_manifest_json: &str,
    reconstructed_static_binding_manifest_json: &str,
) -> PyResult<CompileOptions> {
    validate_workload_compilation_wire_inputs(
        cache_key,
        graph_json,
        timeline_json,
        options_json,
        decision_json,
        plan_epoch,
    )?;

    let original: ScheduleGraph = decode(graph_json, "ScheduleGraph")?;
    let reconstructed: ScheduleGraph =
        decode(reconstructed_graph_json, "reconstructed ScheduleGraph")?;
    let timeline: Timeline = decode(timeline_json, "MainWorkloadTimeline")?;
    let options: CompileOptions = decode(options_json, "CompileOptions")?;

    validate_inputs(&reconstructed, &timeline, &options)?;
    let original_value: serde_json::Value = decode(graph_json, "ScheduleGraph")?;
    let reconstructed_value: serde_json::Value =
        decode(reconstructed_graph_json, "reconstructed ScheduleGraph")?;
    if canonical_ascii_json(&original_value).map_err(value_error)?
        != canonical_ascii_json(&reconstructed_value).map_err(value_error)?
    {
        return Err(schedule_error(
            "worker reconstruction does not match the signed ScheduleGraph",
        ));
    }
    validate_inputs(&original, &timeline, &options)?;

    let reconstructed_manifest: Vec<ResidualResourceBinding> = decode(
        reconstructed_residual_resource_manifest_json,
        "reconstructed residual resource manifest",
    )?;
    validate_residual_resource_manifest(&reconstructed, &reconstructed_manifest)?;
    if reconstructed_manifest != options.residual_resource_manifest {
        return Err(schedule_error(
            "worker residual resource manifest does not match the parent-signed CompileOptions",
        ));
    }
    let reconstructed_packed_manifest: Vec<PackedHiddenResourceBinding> = decode(
        reconstructed_packed_hidden_resource_manifest_json,
        "reconstructed packed hidden resource manifest",
    )?;
    validate_packed_hidden_resource_manifest(&reconstructed, &reconstructed_packed_manifest)?;
    if reconstructed_packed_manifest
        != options
            .packed_hidden_resource_manifest
            .as_deref()
            .unwrap_or(&[])
    {
        return Err(schedule_error(
            "worker packed hidden resource manifest does not match the parent-signed CompileOptions",
        ));
    }
    let signed_static_bindings = validate_static_backend_binding_manifest(&options)?;
    let reconstructed_static_bindings = parse_static_backend_binding_manifest(
        reconstructed_static_binding_manifest_json,
        "reconstructed static backend binding manifest",
    )?;
    if reconstructed_static_bindings != signed_static_bindings {
        return Err(schedule_error(
            "worker static backend binding manifest does not match the parent-signed CompileOptions",
        ));
    }
    Ok(options)
}

/// Validate local in-process bindings before any Region artifact or hook is
/// materialized. It intentionally does not require a worker observation.
#[pyfunction]
fn validate_local_execution_binding_wire(
    cache_key: &str,
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    decision_json: &str,
    plan_epoch: &Bound<'_, PyAny>,
    reconstructed_graph_json: &str,
    reconstructed_residual_resource_manifest_json: &str,
    reconstructed_packed_hidden_resource_manifest_json: &str,
    reconstructed_static_binding_manifest_json: &str,
) -> PyResult<()> {
    if !plan_epoch.is_exact_instance_of::<PyInt>() {
        return Err(value_error("plan_epoch must be a positive integer"));
    }
    let plan_epoch: u64 = plan_epoch
        .extract()
        .map_err(|_| value_error("plan_epoch must be a positive integer"))?;
    validate_reconstructed_workload_wire_inputs(
        cache_key,
        graph_json,
        timeline_json,
        options_json,
        decision_json,
        plan_epoch,
        reconstructed_graph_json,
        reconstructed_residual_resource_manifest_json,
        reconstructed_packed_hidden_resource_manifest_json,
        reconstructed_static_binding_manifest_json,
    )?;
    Ok(())
}

/// Validate one worker's reconstruction before Python binds any callbacks or
/// resources.  This owns semantic graph equality and the signed runtime fact
/// contract; Python only creates the local declaration and transports JSON.
#[pyfunction]
fn validate_worker_reconstruction_wire(
    cache_key: &str,
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    decision_json: &str,
    plan_epoch: &Bound<'_, PyAny>,
    reconstructed_graph_json: &str,
    runtime_specialization_facts_json: &str,
    reconstructed_residual_resource_manifest_json: &str,
    reconstructed_packed_hidden_resource_manifest_json: &str,
    reconstructed_static_binding_manifest_json: &str,
) -> PyResult<()> {
    if !plan_epoch.is_exact_instance_of::<PyInt>() {
        return Err(value_error("plan_epoch must be a positive integer"));
    }
    let plan_epoch: u64 = plan_epoch
        .extract()
        .map_err(|_| value_error("plan_epoch must be a positive integer"))?;

    let options = validate_reconstructed_workload_wire_inputs(
        cache_key,
        graph_json,
        timeline_json,
        options_json,
        decision_json,
        plan_epoch,
        reconstructed_graph_json,
        reconstructed_residual_resource_manifest_json,
        reconstructed_packed_hidden_resource_manifest_json,
        reconstructed_static_binding_manifest_json,
    )?;
    validate_runtime_specialization_facts(&options, runtime_specialization_facts_json)?;
    Ok(())
}

/// Apply cold Region shape observations to an active signed compilation.
///
/// This is deliberately a pure JSON transformation.  It validates the
/// original coordinator wire before decoding it, then owns all identity,
/// Region-port coverage, duplicate/conflict, and deterministic merge rules.
/// Python only transports the requests and binds the returned options to a
/// subsequent cold compilation.
fn apply_region_shape_specializations_wire(
    cache_key: &str,
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    decision_json: &str,
    plan_epoch: u64,
    requests_json: &str,
) -> PyResult<String> {
    if plan_epoch == 0 {
        return Err(value_error("plan_epoch must be a positive integer"));
    }
    if !is_lowercase_sha256(cache_key) {
        return Err(value_error("cache_key must be a lowercase SHA-256 digest"));
    }

    // Keep this first: it consumes unparsed JSON and verifies the cache key,
    // decision signature, and canonical Region plan before anything can be
    // normalized by a host-language object model.
    validate_schedule(graph_json, timeline_json, options_json, decision_json)?;
    if cache_key != workload_compilation_wire_cache_key(graph_json, timeline_json, options_json) {
        return Err(value_error(
            "workload compilation cache key does not match wire inputs",
        ));
    }

    let graph: ScheduleGraph = decode(graph_json, "ScheduleGraph")?;
    let timeline: Timeline = decode(timeline_json, "MainWorkloadTimeline")?;
    let mut options: CompileOptions = decode(options_json, "CompileOptions")?;
    let decision: Decision = decode(decision_json, "ScheduleDecision")?;
    let requests: Vec<crate::shape_specialization::RegionShapeSpecializationRequest> =
        decode(requests_json, "RegionShapeSpecializationRequest array")?;
    if requests.is_empty() {
        return Err(value_error(
            "shape specialization requests must be non-empty",
        ));
    }

    let mut requested_entries = BTreeMap::<String, String>::new();
    for request in &requests {
        crate::shape_specialization::validate(request).map_err(value_error)?;
        if request.base_cache_key != cache_key
            || request.base_signature != decision.signature
            || request.observed_plan_epoch != plan_epoch
        {
            return Err(value_error(
                "shape specialization request does not match the active signed epoch",
            ));
        }
        let region = decision
            .execution_regions
            .iter()
            .find(|region| region.region_id == request.region_id)
            .ok_or_else(|| {
                value_error("shape specialization request does not match a signed Region artifact")
            })?;
        if region.artifact_key != request.artifact_key {
            return Err(value_error(
                "shape specialization request does not match a signed Region artifact",
            ));
        }
        let mut expected_bindings: Vec<&str> = region
            .input_bindings
            .iter()
            .map(|binding| binding.binding_id.as_str())
            .collect();
        expected_bindings.sort_unstable();
        let observed_bindings: Vec<&str> = request
            .inputs
            .iter()
            .map(|input| input.binding_id.as_str())
            .collect();
        if observed_bindings != expected_bindings {
            return Err(value_error(
                "shape specialization request does not cover signed Region inputs",
            ));
        }
        for input in &request.inputs {
            let key = format!(
                "runtime/region-shape/{}/{}",
                request.region_id, input.binding_id
            );
            let value = canonical_ascii_json(input).map_err(value_error)?;
            if let Some(previous) = requested_entries.insert(key, value.clone()) {
                if previous != value {
                    return Err(value_error(
                        "shape specialization requests disagree for one Region input",
                    ));
                }
            }
        }
    }

    let mut merged: BTreeMap<String, String> = options.shape_specialization.into_iter().collect();
    for (key, value) in requested_entries {
        if merged.contains_key(&key) {
            return Err(value_error(
                "shape specialization request would overwrite an active Region input",
            ));
        }
        merged.insert(key, value);
    }
    options.shape_specialization = merged.into_iter().collect();

    // The returned options remain usable only if the same Rust scheduler
    // accepts them for this graph/timeline pair.
    validate_inputs(&graph, &timeline, &options)?;
    canonical_ascii_json(&options).map_err(value_error)
}

/// PyO3 boundary for cold Region shape specialization.
#[pyfunction]
fn apply_region_shape_specializations(
    cache_key: &str,
    graph_json: &str,
    timeline_json: &str,
    options_json: &str,
    decision_json: &str,
    plan_epoch: &Bound<'_, PyAny>,
    requests_json: &str,
) -> PyResult<String> {
    if !plan_epoch.is_exact_instance_of::<PyInt>() {
        return Err(value_error("plan_epoch must be a positive integer"));
    }
    let plan_epoch: u64 = plan_epoch
        .extract()
        .map_err(|_| value_error("plan_epoch must be a positive integer"))?;
    apply_region_shape_specializations_wire(
        cache_key,
        graph_json,
        timeline_json,
        options_json,
        decision_json,
        plan_epoch,
        requests_json,
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(intern_state_layout_ids, module)?)?;
    module.add_function(wrap_pyfunction!(compile_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_workload_compilation_wire,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_worker_reconstruction_specs,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_worker_reconstruction_wire,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_local_execution_binding_wire,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        normalize_runtime_specialization_options,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        apply_region_shape_specializations,
        module
    )?)?;
    Ok(())
}

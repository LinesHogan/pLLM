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
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ScheduleInputKind {
    Hidden,
    Result,
    Provided,
    StepId,
    Staged,
    SamplerRequestIds,
    SamplerSampleIdxs,
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

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum RegionInputKind {
    Hidden,
    Provided,
    Materialized,
    StepId,
    Staged,
    SamplerRequestIds,
    SamplerSampleIdxs,
    SamplerLogits,
    SampledTokens,
}

impl RegionInputKind {
    fn as_wire(self) -> &'static str {
        match self {
            Self::Hidden => "hidden",
            Self::Provided => "provided",
            Self::Materialized => "materialized",
            Self::StepId => "step_id",
            Self::Staged => "staged",
            Self::SamplerRequestIds => "sampler_request_ids",
            Self::SamplerSampleIdxs => "sampler_sample_idxs",
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
    request_key: Vec<String>,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompileOptions {
    memory_budget_bytes: u64,
    allow_serial_fallback: bool,
    #[serde(default)]
    shape_specialization: Vec<(String, String)>,
    #[serde(default)]
    residual_resource_manifest: Vec<ResidualResourceBinding>,
    #[serde(default = "default_static_binding_manifest")]
    static_binding_manifest: String,
    #[serde(default)]
    compiler_fingerprint: String,
    #[serde(default = "default_compile_policy")]
    compile_policy: String,
    #[serde(default = "default_capture_policy")]
    capture_policy: String,
    /// Permit the planner to place effect-free Regions on a side stream so
    /// they overlap with the main workload.  Cross-stream ordering stays on
    /// explicit fences; the planner never predicts GPU timing.
    #[serde(default)]
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
    effects: Vec<ScheduleEffect>,
    compile_safe: bool,
    capture_safe: bool,
    capture_mode: RegionCaptureMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stream: Option<RegionStreamClass>,
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
    backend_region_binder_stages: Vec<BackendRegionBinderStage>,
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

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn workload_compilation_wire_cache_key(graph: &str, timeline: &str, options: &str) -> String {
    digest(&format!(
        "workload-compilation-v2\0{graph}\0{timeline}\0{options}"
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
    if options.schema_version != 1 {
        return Err(value_error("unsupported schedule schema version"));
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

fn backend_capabilities(capabilities: &[String]) -> BTreeSet<String> {
    capabilities
        .iter()
        .filter(|capability| capability.as_str() != TORCH_EXPORT_CAPABILITY)
        .cloned()
        .collect()
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
                        let required_capabilities = backend_capabilities(&capabilities);
                        required_capabilities
                            .iter()
                            .all(|capability| available.contains(capability.as_str()))
                            .then(|| {
                                let compile_safe = kind == TaskImplementationKind::Captured
                                    && (graph.schema_version == 1 || export_safe);
                                SelectedTaskImplementation {
                                    kind,
                                    implementation_id,
                                    required_capabilities: required_capabilities
                                        .into_iter()
                                        .collect(),
                                    capture_safe: compile_safe && task.effects.is_empty(),
                                    compile_safe,
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
                "task {:?} schema v2 requires per-implementation capabilities",
                task.task_id
            )));
        }
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
        let expected_capabilities: Vec<String> = required_by_all_candidates
            .unwrap_or_default()
            .into_iter()
            .collect();
        if task.required_capabilities != expected_capabilities {
            return Err(schedule_error(format!(
                "task {:?} derived required_capabilities disagree with implementation evidence",
                task.task_id
            )));
        }
        let expected_capture_safe = task.effects.is_empty()
            && task
                .implementation_kinds
                .iter()
                .all(|kind| *kind == TaskImplementationKind::Captured);
        if task.capture_safe != expected_capture_safe {
            return Err(schedule_error(format!(
                "task {:?} derived capture_safe disagrees with effects or implementation kinds",
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

fn validate_inputs(
    graph: &ScheduleGraph,
    timeline: &Timeline,
    options: &CompileOptions,
) -> PyResult<(HashMap<String, usize>, HashMap<String, usize>)> {
    if !matches!(graph.schema_version, 1 | 2) || !matches!(timeline.schema_version, 1 | 2) {
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
        if state.resource.is_empty()
            || state.transition.is_empty()
            || (state.scope == StateScope::Request) != !state.request_key.is_empty()
            || !state_resources.insert(state.resource.as_str())
        {
            return Err(value_error("invalid or duplicate typed schedule state"));
        }
    }
    let mut task_index = HashMap::new();
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
                ScheduleInputKind::Result
                | ScheduleInputKind::Provided
                | ScheduleInputKind::Staged
                | ScheduleInputKind::SamplerRequestIds
                | ScheduleInputKind::SamplerSampleIdxs
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
                | ScheduleInputKind::Staged
                | ScheduleInputKind::SamplerRequestIds
                | ScheduleInputKind::SamplerSampleIdxs
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
                _ => {}
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
    for effects in effects_by_resource.values() {
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
    tasks: &[usize],
) -> (usize, usize) {
    let earliest = tasks
        .iter()
        .map(|index| anchor_index[&graph.tasks[*index].timing.after])
        .max()
        .expect("ExecutionRegion drafts are non-empty");
    let before = tasks
        .iter()
        .map(|index| anchor_index[&graph.tasks[*index].timing.before])
        .min()
        .expect("ExecutionRegion drafts are non-empty");
    (earliest, before)
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
    anchor_index: &HashMap<String, usize>,
    left: &RegionDraft,
    right: &RegionDraft,
    effect_reachability: &[BTreeSet<usize>],
    order_rank: &[usize],
) -> Option<RegionDraft> {
    let mut tasks = left.tasks.clone();
    tasks.extend(right.tasks.iter().copied());
    tasks.sort_by_key(|index| order_rank[*index]);
    let (earliest, before) = draft_window(graph, anchor_index, &tasks);
    let implementation_kind = selected[tasks[0]].kind;
    // Merging is per-policy: a shared Region would otherwise let a non-exact
    // workload's parity tolerance wash over an exact workload's outputs.
    let first_policy = task_approximation_policy(graph, graph.tasks[tasks[0]].task_id.as_str());
    let mixes_policies = tasks.iter().skip(1).any(|index| {
        task_approximation_policy(graph, graph.tasks[*index].task_id.as_str()) != first_policy
    });
    if earliest >= before
        || implementation_kind == TaskImplementationKind::Extern
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
    let drafts: Vec<RegionDraft> = order
        .iter()
        .map(|index| {
            let (earliest, before) = draft_window(graph, anchor_index, &[*index]);
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
/// hidden inputs remain hidden bindings.  Provided and step-id bindings make
/// capture ineligible, and a Region with no external tensor contract is not
/// capturable.  This keeps partition-search cost equal to the final signed
/// `capture_mode` semantics instead of charging eager/disabled timelines.
fn draft_capture_safe(
    graph: &ScheduleGraph,
    selected: &[SelectedTaskImplementation],
    task_index: &HashMap<String, usize>,
    draft: &RegionDraft,
) -> bool {
    if !draft.tasks.iter().all(|task| selected[*task].capture_safe) {
        return false;
    }
    let members: BTreeSet<usize> = draft.tasks.iter().copied().collect();
    let mut external_inputs = 0usize;
    for task_index_in_draft in &draft.tasks {
        for input in &graph.tasks[*task_index_in_draft].inputs {
            match input.kind {
                ScheduleInputKind::Hidden => external_inputs += 1,
                ScheduleInputKind::Result => {
                    let Some(source) = task_index.get(&input.source).copied() else {
                        return false;
                    };
                    if !members.contains(&source) {
                        external_inputs += 1;
                    }
                }
                ScheduleInputKind::Provided
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
    let mut staged_binder_identities = BTreeSet::<(BackendObservableEvent, String)>::new();
    let mut advances = Vec::new();
    for anchor in &timeline.anchors {
        for backend_event in anchor.backend_events.as_deref().unwrap_or_default() {
            let region_ids = region_ids_by_anchor
                .get(anchor.anchor_id.as_str())
                .cloned()
                .unwrap_or_default();
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
                backend_region_binder_stages,
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
        let implementation_kind = if selected[draft.tasks[0]].kind == TaskImplementationKind::Extern
        {
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
                .all(|index| selected[*index].compile_safe);
        let capture_safe = compile_safe
            && draft
                .tasks
                .iter()
                .all(|index| selected[*index].capture_safe);
        let effects: Vec<ScheduleEffect> = draft
            .tasks
            .iter()
            .flat_map(|index| graph.tasks[*index].effects.iter().cloned())
            .collect();
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
            effects,
            compile_safe,
            capture_safe,
            capture_mode: RegionCaptureMode::Disabled,
            stream: Some(stream),
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
                        ScheduleInputKind::Provided => RegionInputKind::Provided,
                        ScheduleInputKind::Staged => RegionInputKind::Staged,
                        ScheduleInputKind::SamplerRequestIds => RegionInputKind::SamplerRequestIds,
                        ScheduleInputKind::SamplerSampleIdxs => RegionInputKind::SamplerSampleIdxs,
                        ScheduleInputKind::SamplerLogits => RegionInputKind::SamplerLogits,
                        ScheduleInputKind::SampledTokens => RegionInputKind::SampledTokens,
                        ScheduleInputKind::StepId => RegionInputKind::StepId,
                        ScheduleInputKind::Result => {
                            unreachable!("result inputs are handled before external bindings")
                        }
                    };
                    let identity = format!(
                        "{}\0{}\0{}\0{}\0{}\0{}",
                        region_kind.as_wire(),
                        input.source,
                        input.layer,
                        input.layer_kind.as_wire(),
                        input.site.as_wire(),
                        input.phase.as_wire()
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
        region.capture_safe = region.capture_safe
            && !region.input_bindings.is_empty()
            && region.input_bindings.iter().all(|binding| {
                matches!(
                    binding.kind,
                    RegionInputKind::Hidden | RegionInputKind::Materialized
                )
            });
        // Side-stream placement only exposes legal parallelism to the GPU
        // hardware scheduler; every cross-stream dependency remains an
        // explicit fence.  A Region may leave the default stream only when
        // the signed graph proves the placement safe:
        // - zero state effects, so version ordering never depends on
        //   hardware timing;
        // - the final signed capture_safe, so the compiled callable has no
        //   host-side hazard the stream context could invalidate;
        // - every input a signed materialization (fence-covered cross-region
        //   value).  Hidden taps and provided/step-id values are excluded:
        //   the model cannot prove the main snapshot event covers their
        //   production.
        if options.allow_side_stream
            && region.effects.is_empty()
            && region.capture_safe
            && region
                .input_bindings
                .iter()
                .all(|binding| matches!(binding.kind, RegionInputKind::Materialized))
        {
            region.stream = Some(RegionStreamClass::SideForward);
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
    let capture_boundary_count = execution_regions
        .iter()
        .filter(|region| region.capture_mode == RegionCaptureMode::Require)
        .count() as u64;
    let serial_fallback_count = execution_regions
        .iter()
        .filter(|region| region.serial_fallback)
        .count();
    let (runtime_advance_plan, completion_anchor) =
        runtime_advance_plan(timeline, &execution_regions)?;
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
        schema_version: 2,
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
    if supplied.schema_version != 2
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
            "ScheduleDecision v2 ExecutionRegion requires explicit stream and fallback_policy",
        ));
    }
    if supplied.execution_regions.iter().any(|region| {
        (region.implementation_kind == RegionImplementationKind::Compiled)
            != region.compile_command.is_some()
    }) {
        return Err(schedule_error(
            "ScheduleDecision v2 compiled ExecutionRegion requires a compile_command and extern Region forbids one",
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
            "ScheduleDecision v2 requires explicit regions and forbids capture_partitions",
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
            let inputs_materialized = region
                .input_bindings
                .iter()
                .all(|binding| matches!(binding.kind, RegionInputKind::Materialized));
            if !options_desc.allow_side_stream
                || !region.effects.is_empty()
                || !region.capture_safe
                || !inputs_materialized
            {
                return Err(schedule_error(
                    "side-stream ExecutionRegion requires allow_side_stream, zero state effects, capture safety, and fully materialized inputs",
                ));
            }
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
        for binding in &region.input_bindings {
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
                || (binding.layer_kind == HiddenLayerKind::Index && binding.layer_index.is_none())
                || (binding.layer_kind != HiddenLayerKind::Index && binding.layer_index.is_some())
                || (binding.kind != RegionInputKind::Hidden
                    && (binding.layer_kind != HiddenLayerKind::None
                        || binding.site != HiddenSite::None
                        || binding.phase != HiddenPhase::None))
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
                if !task.inputs.iter().any(|input| input.name == port.port_name) {
                    return Err(schedule_error(
                        "ExecutionRegion typed input references an undeclared task port",
                    ));
                }
            }
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
        schedule_error("ScheduleDecision v2 requires an explicit runtime_advance_plan")
    })?;
    let supplied_completion_anchor = supplied.completion_anchor.as_deref().ok_or_else(|| {
        schedule_error("ScheduleDecision v2 requires an explicit completion_anchor")
    })?;
    if supplied_completion_anchor.is_empty() {
        return Err(schedule_error(
            "ScheduleDecision completion_anchor must be non-empty",
        ));
    }
    let (canonical_runtime_plan, canonical_completion_anchor) =
        runtime_advance_plan(&timeline_desc, &supplied.execution_regions)?;
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
        let earliest = anchor_index[&task.timing.after];
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

#[cfg(test)]
mod tests {
    use super::{
        apply_region_shape_specializations_wire, compile_schedule, digest,
        normalize_runtime_specialization_options_wire, validate_runtime_specialization_facts,
        validate_schedule, workload_compilation_wire_cache_key, CompileOptions, Decision,
        ExecutionRegion, HiddenPhase, HiddenSite, RegionCaptureMode, RegionFallbackPolicy,
        RegionInputBinding, RegionInputKind, RegionOutputKind, RegionStreamClass,
        ScheduleEffectKind, ScheduleInput,
    };
    use isl_rs::{Context, ScheduleConstraints, UnionAccessInfo, UnionMap, UnionSet};

    #[test]
    fn worker_runtime_facts_are_closed_complete_and_exact() {
        let signed: CompileOptions = serde_json::from_str(
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"shape_specialization":[["runtime/device","cpu"],["runtime/dtype","torch.float32"],["runtime/hidden_size","4"],["runtime/rows","2"]],"schema_version":1}"#,
        )
        .unwrap();
        let facts = r#"{"device":"cpu","dtype":"torch.float32","hidden_size":4,"rows":2}"#;
        validate_runtime_specialization_facts(&signed, facts).unwrap();
        assert!(validate_runtime_specialization_facts(&signed, "null")
            .unwrap_err()
            .to_string()
            .contains("are required"));
        assert!(validate_runtime_specialization_facts(
            &signed,
            r#"{"device":"cpu","dtype":"torch.float16","hidden_size":4,"rows":2}"#,
        )
        .unwrap_err()
        .to_string()
        .contains("specialization does not match"));
        assert!(validate_runtime_specialization_facts(
            &signed,
            r#"{"device":"cpu","device":"cpu","dtype":"torch.float32","hidden_size":4,"rows":2}"#,
        )
        .unwrap_err()
        .to_string()
        .contains("duplicate field"));

        let unsigned: CompileOptions = serde_json::from_str(
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#,
        )
        .unwrap();
        validate_runtime_specialization_facts(&unsigned, "null").unwrap();
        assert!(validate_runtime_specialization_facts(&unsigned, facts)
            .unwrap_err()
            .to_string()
            .contains("unsigned"));
        assert!(validate_runtime_specialization_facts(
            &signed,
            r#"{"device":"cpu","dtype":"torch.float32","hidden_size":0,"rows":2}"#,
        )
        .unwrap_err()
        .to_string()
        .contains("positive rows/hidden_size"));
        assert!(validate_runtime_specialization_facts(
            &signed,
            r#"{"device":"cpu","dtype":"torch.float32","hidden_size":"4","rows":2}"#,
        )
        .unwrap_err()
        .to_string()
        .contains("invalid nullable RuntimeSpecializationFacts"));
    }

    #[test]
    fn runtime_specialization_options_are_rust_normalized_before_identity() {
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;
        let facts = r#"{"device":"cpu","dtype":"torch.float32","hidden_size":4,"rows":2}"#;
        let normalized = normalize_runtime_specialization_options_wire(options, facts).unwrap();
        let value: serde_json::Value = serde_json::from_str(&normalized).unwrap();
        assert_eq!(
            value["shape_specialization"],
            serde_json::json!([
                ["runtime/device", "cpu"],
                ["runtime/dtype", "torch.float32"],
                ["runtime/hidden_size", "4"],
                ["runtime/rows", "2"],
            ])
        );
        assert_eq!(
            normalize_runtime_specialization_options_wire(&normalized, facts).unwrap(),
            normalized
        );
        assert!(normalize_runtime_specialization_options_wire(
            &normalized,
            r#"{"device":"cpu","dtype":"torch.float32","hidden_size":4,"rows":3}"#,
        )
        .unwrap_err()
        .to_string()
        .contains("does not match"));
        assert!(
            normalize_runtime_specialization_options_wire(&normalized, "null")
                .unwrap_err()
                .to_string()
                .contains("are required")
        );
        assert!(normalize_runtime_specialization_options_wire(
            options,
            r#"{"device":"cpu","dtype":"torch.float32","hidden_size":"4","rows":2}"#,
        )
        .unwrap_err()
        .to_string()
        .contains("invalid nullable RuntimeSpecializationFacts"));

        let prefill_facts = r#"{"device":"cpu","dtype":"torch.float32","hidden_size":4,"max_prefill_tokens":16,"rows":2}"#;
        let prefill_normalized =
            normalize_runtime_specialization_options_wire(options, prefill_facts).unwrap();
        let prefill_value: serde_json::Value = serde_json::from_str(&prefill_normalized).unwrap();
        assert!(prefill_value["shape_specialization"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!(["runtime/max_prefill_tokens", "16"])));
        assert!(normalize_runtime_specialization_options_wire(
            &prefill_normalized,
            r#"{"device":"cpu","dtype":"torch.float32","hidden_size":4,"max_prefill_tokens":32,"rows":2}"#,
        )
        .unwrap_err()
        .to_string()
        .contains("does not match"));
    }

    #[test]
    fn runtime_specialization_preflight_rejects_malformed_static_bindings() {
        pyo3::prepare_freethreaded_python();
        let options = serde_json::json!({
            "memory_budget_bytes": 64,
            "allow_serial_fallback": true,
            "static_binding_manifest": r#"[{"workload_id":"fixture","factory":"factory","config":[],"specialization_key":[],"backend_region_implementation_ids":[]}]"#,
            "schema_version": 1,
        })
        .to_string();
        let facts = r#"{"device":"cpu","dtype":"torch.float32","hidden_size":4,"rows":2}"#;

        assert!(
            normalize_runtime_specialization_options_wire(&options, facts)
                .unwrap_err()
                .to_string()
                .contains("static backend binding config must be a JSON object")
        );
    }

    #[test]
    fn schema_v2_signs_backend_event_advances_and_completion_anchor() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"event-plan","tasks":[{"task_id":"early","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false},{"task_id":"late","timing":{"after":"token.commit","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[],"backend_events":["execute_model.pre"]},{"anchor_id":"token.commit","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[],"backend_events":["sample_tokens.post"]},{"anchor_id":"step.end","ordinal":2,"observable":true,"graph_boundary":true,"visible_values":[],"backend_events":["sample_tokens.post"]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":2}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled: serde_json::Value =
            serde_json::from_str(&compile_schedule(graph, timeline, options).unwrap()).unwrap();
        let advances = compiled["runtime_advance_plan"].as_array().unwrap();
        assert_eq!(advances.len(), 3);
        assert_eq!(advances[0]["backend_event"], "execute_model.pre");
        assert_eq!(advances[0]["anchor_id"], "model.execute");
        assert_eq!(advances[1]["backend_event"], "sample_tokens.post");
        assert_eq!(advances[2]["anchor_id"], "step.end");
        assert_eq!(compiled["completion_anchor"], "token.commit");
        validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&compiled).unwrap(),
        )
        .unwrap();

        let mut legacy_shaped: Decision = serde_json::from_value(compiled).unwrap();
        legacy_shaped.runtime_advance_plan = None;
        legacy_shaped.completion_anchor = None;
        legacy_shaped.signature.clear();
        legacy_shaped.signature = digest(&serde_json::to_string(&legacy_shaped).unwrap());
        let error = validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&legacy_shaped).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("requires an explicit runtime_advance_plan"));

        let missing_mapping =
            timeline.replacen(r#","backend_events":["execute_model.pre"]"#, "", 1);
        let error = compile_schedule(graph, &missing_mapping, options).unwrap_err();
        assert!(error
            .to_string()
            .contains("anchors require explicit backend_events"));

        let uncovered_placement = timeline.replacen(
            r#"backend_events":["sample_tokens.post"]"#,
            r#"backend_events":[]"#,
            1,
        );
        let error = compile_schedule(graph, &uncovered_placement, options).unwrap_err();
        assert!(error
            .to_string()
            .contains("placed at an anchor without a backend event"));
    }

    #[test]
    fn schema_v2_emits_explicit_backend_binder_stages_and_rejects_duplicates() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"binder-stage","tasks":[{"task_id":"bind","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["test.backend:bind@1"],"implementation_kinds":["extern"],"implementation_capabilities":[["backend/region_binder"]],"required_capabilities":["backend/region_binder"],"capture_safe":false}],"edges":[],"values":[],"states":[],"persistent_state_bytes":0,"schema_version":2}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[],"backend_events":["prepare_inputs.post"]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[],"backend_events":["sample_tokens.post"]}],"regions":[],"static_headroom_bytes":64,"capabilities":["backend/region_binder"],"schema_version":2}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled: serde_json::Value =
            serde_json::from_str(&compile_schedule(graph, timeline, options).unwrap()).unwrap();
        let advances = compiled["runtime_advance_plan"].as_array().unwrap();
        let stages = advances[0]["backend_region_binder_stages"]
            .as_array()
            .unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0]["backend_event"], "prepare_inputs.post");
        assert_eq!(stages[0]["implementation_id"], "test.backend:bind@1");
        assert!(stages[0]["region_id"]
            .as_str()
            .is_some_and(|id| !id.is_empty()));
        validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&compiled).unwrap(),
        )
        .unwrap();

        let mut duplicate_graph: serde_json::Value = serde_json::from_str(graph).unwrap();
        let tasks = duplicate_graph["tasks"].as_array_mut().unwrap();
        let mut second = tasks[0].clone();
        second["task_id"] = serde_json::Value::String("bind_again".to_owned());
        tasks.push(second);
        let duplicate_graph = serde_json::to_string(&duplicate_graph).unwrap();
        assert!(compile_schedule(&duplicate_graph, timeline, options)
            .unwrap_err()
            .to_string()
            .contains("binder implementation"));
    }

    #[test]
    fn schema_v2_selects_first_timeline_supported_implementation() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"candidate-selection","tasks":[{"task_id":"select","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["backend:extern@1","python:compiled@1"],"implementation_kinds":["extern","captured"],"implementation_capabilities":[["backend/alternate"],["backend/default","torch.export"]],"required_capabilities":[],"capture_safe":false}],"edges":[],"values":[],"states":[],"persistent_state_bytes":0,"schema_version":2}"#;
        let timeline = r#"{"backend_id":"test","backend_version":"1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"begin","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":["backend/default"],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled: serde_json::Value =
            serde_json::from_str(&compile_schedule(graph, timeline, options).unwrap()).unwrap();
        let region = &compiled["execution_regions"][0];
        assert_eq!(
            region["implementation_ids"],
            serde_json::json!(["python:compiled@1"])
        );
        assert_eq!(region["implementation_kind"], "compiled");
        assert_eq!(
            region["required_capabilities"],
            serde_json::json!(["backend/default"])
        );
        assert_eq!(
            compiled["placements"][0]["implementation_id"],
            "python:compiled@1"
        );

        let alternate_timeline = timeline.replace(
            r#"["backend/default"]"#,
            r#"["backend/alternate","backend/default"]"#,
        );
        let extern_decision: serde_json::Value =
            serde_json::from_str(&compile_schedule(graph, &alternate_timeline, options).unwrap())
                .unwrap();
        assert_eq!(
            extern_decision["execution_regions"][0]["implementation_ids"],
            serde_json::json!(["backend:extern@1"])
        );
        assert_eq!(
            extern_decision["execution_regions"][0]["implementation_kind"],
            "extern"
        );

        let unavailable_timeline = timeline.replace(r#"["backend/default"]"#, "[]");
        assert!(compile_schedule(graph, &unavailable_timeline, options)
            .unwrap_err()
            .to_string()
            .contains("no implementation candidate"));
    }

    #[test]
    fn validator_rejects_a_resigned_illegal_timing_placement() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"abc","tasks":[{"task_id":"predict","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"sampler.pre_draw","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":2,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"next.prepare","ordinal":3,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled = compile_schedule(graph, timeline, options).unwrap();
        let mut decision: Decision = serde_json::from_str(&compiled).unwrap();
        decision.placements[0].after_anchor = "step.end".to_owned();
        decision.signature.clear();
        let unsigned = serde_json::to_string(&decision).unwrap();
        decision.signature = digest(&unsigned);
        let resigned = serde_json::to_string(&decision).unwrap();

        let error = validate_schedule(graph, timeline, options, &resigned).unwrap_err();
        assert!(error.to_string().contains("violates its timing window"));
    }

    #[test]
    fn region_wire_schema_rejects_unknown_or_omitted_semantic_fields() {
        let unknown_region = r#"{
            "region_id":"region-0",
            "tasks":["task"],
            "after_anchor":"model.execute",
            "before_anchor":"step.end",
            "placement_anchor":"model.execute",
            "implementation_kind":"compiled",
            "implementation_ids":["python:task@1"],
            "compile_command":null,
            "required_capabilities":[],
            "dependencies":[],
            "materialized_inputs":[],
            "materialized_outputs":[],
            "input_bindings":[],
            "output_bindings":[],
            "task_invocations":[],
            "effects":[],
            "compile_safe":true,
            "capture_safe":false,
            "capture_mode":"disabled",
            "stream":"default",
            "fallback_policy":"reject",
            "artifact_key":"0000000000000000000000000000000000000000000000000000000000000000",
            "serial_fallback":false,
            "affine_schedule":"",
            "unexpected":true
        }"#;
        let error = serde_json::from_str::<ExecutionRegion>(unknown_region).unwrap_err();
        assert!(error.to_string().contains("unknown field `unexpected`"));
        let error = serde_json::from_str::<RegionCaptureMode>(r#""prefer""#).unwrap_err();
        assert!(error.to_string().contains("unknown variant `prefer`"));
        assert_eq!(
            serde_json::from_str::<RegionStreamClass>(r#""side_forward""#).unwrap(),
            RegionStreamClass::SideForward
        );
        let error = serde_json::from_str::<RegionStreamClass>(r#""side_train""#).unwrap_err();
        assert!(error.to_string().contains("unknown variant `side_train`"));
        let error = serde_json::from_str::<RegionFallbackPolicy>(r#""eager""#).unwrap_err();
        assert!(error.to_string().contains("unknown variant `eager`"));

        let mut missing_command: serde_json::Value = serde_json::from_str(unknown_region).unwrap();
        missing_command
            .as_object_mut()
            .unwrap()
            .remove("compile_command");
        missing_command
            .as_object_mut()
            .unwrap()
            .remove("unexpected");
        let error = serde_json::from_value::<ExecutionRegion>(missing_command).unwrap_err();
        assert!(error
            .to_string()
            .contains("missing field `compile_command`"));

        let mut legacy_region: serde_json::Value = serde_json::from_str(unknown_region).unwrap();
        let legacy_fields = legacy_region.as_object_mut().unwrap();
        legacy_fields.remove("stream");
        legacy_fields.remove("fallback_policy");
        legacy_fields.remove("unexpected");
        let legacy_region: ExecutionRegion = serde_json::from_value(legacy_region).unwrap();
        assert_eq!(legacy_region.stream, None);
        assert_eq!(legacy_region.fallback_policy, None);

        let omitted_binding_field = r#"{
            "binding_id":"input-0",
            "kind":"hidden",
            "source":"hidden:0:post:decode",
            "consumers":[{"task_id":"task","port_name":"hidden"}],
            "layer":"0",
            "layer_index":0,
            "site":"post",
            "phase":"decode"
        }"#;
        let error = serde_json::from_str::<RegionInputBinding>(omitted_binding_field).unwrap_err();
        assert!(error.to_string().contains("missing field `layer_kind`"));
    }

    #[test]
    fn hidden_locator_wire_is_closed_and_requires_a_signed_timeline_capability() {
        let unknown_site = r#"{"name":"hidden","kind":"hidden","source":"hidden:0:middle:decode","layer":"0","layer_kind":"index","site":"middle","phase":"decode"}"#;
        let error = serde_json::from_str::<ScheduleInput>(unknown_site).unwrap_err();
        assert!(error.to_string().contains("unknown variant `middle`"));

        let unknown_phase = r#"{"name":"hidden","kind":"hidden","source":"hidden:0:block_output:train","layer":"0","layer_kind":"index","site":"block_output","phase":"train"}"#;
        let error = serde_json::from_str::<ScheduleInput>(unknown_phase).unwrap_err();
        assert!(error.to_string().contains("unknown variant `train`"));

        let legacy_non_hidden =
            r#"{"name":"step","kind":"step_id","source":"step_id","site":"","phase":""}"#;
        let legacy_non_hidden: ScheduleInput = serde_json::from_str(legacy_non_hidden).unwrap();
        assert_eq!(legacy_non_hidden.site, HiddenSite::None);
        assert_eq!(legacy_non_hidden.phase, HiddenPhase::None);

        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"hidden-capability","tasks":[{"task_id":"run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"hidden","kind":"hidden","source":"hidden:0:block_output:decode","layer":"0","layer_kind":"index","site":"block_output","phase":"decode"}]}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options = r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"residual_resource_manifest":[{"layer_index":0,"site":"block_output","phase":"decode","module_path":"model.layers.0"}],"schema_version":1}"#;
        let error = compile_schedule(graph, timeline, options).unwrap_err();
        assert!(error.to_string().contains("hidden/block_output/decode"));

        let capable_timeline = timeline.replace(
            r#""capabilities":[]"#,
            r#""capabilities":["hidden/block_output/decode"]"#,
        );
        let compiled = compile_schedule(graph, &capable_timeline, options).unwrap();
        let mut decision: Decision = serde_json::from_str(&compiled).unwrap();
        assert_eq!(
            decision.execution_regions[0].input_bindings[0].site,
            HiddenSite::BlockOutput
        );
        assert_eq!(
            decision.execution_regions[0].input_bindings[0].phase,
            HiddenPhase::Decode
        );

        decision.execution_regions[0].input_bindings[0].site = HiddenSite::Post;
        decision.signature.clear();
        decision.signature = digest(&serde_json::to_string(&decision).unwrap());
        let error = validate_schedule(
            graph,
            &capable_timeline,
            options,
            &serde_json::to_string(&decision).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("artifact metadata"));
    }

    #[test]
    fn validator_rejects_resigned_region_artifact_and_lifetime_tampering() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"split","tasks":[{"task_id":"produce","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false,"outputs":[{"name":"hidden","kind":"temporary","target":"hidden"}]},{"task_id":"consume","timing":{"after":"step.end","before":"next.prepare"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"hidden","kind":"result","source":"produce","output":"hidden"}]}],"edges":[{"source":"produce","target":"consume","kind":"data"}],"values":[{"value_id":"hidden","version":1,"producer":"produce","producer_output":"hidden","consumers":["consume"],"size_bytes":16,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"sampler.pre_draw","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":2,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"next.prepare","ordinal":3,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled = compile_schedule(graph, timeline, options).unwrap();
        let original: Decision = serde_json::from_str(&compiled).unwrap();

        let mut artifact_tampered = original.clone();
        artifact_tampered.execution_regions[0].artifact_key = "0".repeat(64);
        artifact_tampered.signature.clear();
        artifact_tampered.signature = digest(&serde_json::to_string(&artifact_tampered).unwrap());
        let error = validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&artifact_tampered).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("artifact metadata"));

        let mut lifetime_tampered = original;
        lifetime_tampered.materializations[0].live_end = "sampler.pre_draw".to_owned();
        lifetime_tampered.signature.clear();
        lifetime_tampered.signature = digest(&serde_json::to_string(&lifetime_tampered).unwrap());
        let error = validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&lifetime_tampered).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("materialization or lifetime"));
    }

    #[test]
    fn rust_emits_signed_closed_region_capture_modes_after_binding() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"capture-mode","tasks":[{"task_id":"captured","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":true,"inputs":[{"name":"hidden","kind":"hidden","source":"hidden:0:post:decode","layer":"0","layer_kind":"index","site":"post","phase":"decode"}]}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":1}"#;
        let eager = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":["hidden/post/decode"],"schema_version":1}"#;
        let piecewise = eager.replace("\"graph_mode\":\"eager\"", "\"graph_mode\":\"piecewise\"");
        let auto = r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"residual_resource_manifest":[{"layer_index":0,"site":"post","phase":"decode","module_path":"model.layers.0"}],"schema_version":1}"#;

        let first: Decision =
            serde_json::from_str(&compile_schedule(graph, &piecewise, auto).unwrap()).unwrap();
        let second: Decision =
            serde_json::from_str(&compile_schedule(graph, &piecewise, auto).unwrap()).unwrap();
        assert!(first.execution_regions[0].capture_safe);
        assert_eq!(
            first.execution_regions[0].capture_mode,
            RegionCaptureMode::Require
        );
        assert_eq!(first.cost.capture_boundary_count, 1);
        assert_eq!(
            first.execution_regions[0].stream,
            Some(RegionStreamClass::Default)
        );
        assert_eq!(
            first.execution_regions[0].fallback_policy,
            Some(RegionFallbackPolicy::Reject)
        );
        assert_eq!(
            first.execution_regions[0].artifact_key,
            second.execution_regions[0].artifact_key
        );
        assert_eq!(first.signature, second.signature);

        let disabled = r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"residual_resource_manifest":[{"layer_index":0,"site":"post","phase":"decode","module_path":"model.layers.0"}],"capture_policy":"disabled","schema_version":1}"#;
        let disabled: Decision =
            serde_json::from_str(&compile_schedule(graph, &piecewise, disabled).unwrap()).unwrap();
        assert!(disabled.execution_regions[0].capture_safe);
        assert_eq!(
            disabled.execution_regions[0].capture_mode,
            RegionCaptureMode::Disabled
        );
        assert_eq!(disabled.cost.capture_boundary_count, 0);

        let eager_auto: Decision =
            serde_json::from_str(&compile_schedule(graph, eager, auto).unwrap()).unwrap();
        assert_eq!(
            eager_auto.execution_regions[0].capture_mode,
            RegionCaptureMode::Disabled
        );
        assert_eq!(eager_auto.cost.capture_boundary_count, 0);
        let require = r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"residual_resource_manifest":[{"layer_index":0,"site":"post","phase":"decode","module_path":"model.layers.0"}],"capture_policy":"require","schema_version":1}"#;
        assert!(compile_schedule(graph, eager, require)
            .unwrap_err()
            .to_string()
            .contains("required Region capture"));

        let mut missing_commands = first.clone();
        missing_commands.execution_regions[0].stream = None;
        missing_commands.execution_regions[0].fallback_policy = None;
        missing_commands.signature.clear();
        missing_commands.signature = digest(&serde_json::to_string(&missing_commands).unwrap());
        let error = validate_schedule(
            graph,
            &piecewise,
            auto,
            &serde_json::to_string(&missing_commands).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("requires explicit stream and fallback_policy"));

        let mut tampered = first;
        tampered.execution_regions[0].capture_mode = RegionCaptureMode::Disabled;
        tampered.signature.clear();
        tampered.signature = digest(&serde_json::to_string(&tampered).unwrap());
        let error = validate_schedule(
            graph,
            &piecewise,
            auto,
            &serde_json::to_string(&tampered).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("artifact metadata"));
    }

    #[test]
    fn side_stream_placement_is_signed_and_option_gated() {
        pyo3::prepare_freethreaded_python();
        // Side-eligible: pure, capture-safe, and every Region input a signed
        // materialization.  The two tasks sit in disjoint windows so the
        // planner cannot merge them: `consume` reads `produce` through a
        // fence-covered cross-region value.
        let graph = r#"{"workload_key":"side-stream","tasks":[{"task_id":"produce","timing":{"after":"model.execute","before":"sampler.pre_draw"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":true,"outputs":[{"name":"value","kind":"temporary","target":"value"}]},{"task_id":"consume","timing":{"after":"sampler.pre_draw","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":true,"inputs":[{"name":"value","kind":"result","source":"produce","output":"value"}],"outputs":[{"name":"answer","kind":"sink","target":"answer"}]}],"edges":[{"source":"produce","target":"consume","kind":"data"}],"values":[{"value_id":"side:produce:value","version":1,"producer":"produce","producer_output":"value","consumers":["consume"],"size_bytes":0,"alias_set":""},{"value_id":"side:consume:answer","version":1,"producer":"consume","producer_output":"answer","consumers":[],"size_bytes":0,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"sampler.pre_draw","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":2,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let serial_options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;
        let side_options = r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"allow_side_stream":true,"schema_version":1}"#;

        let serial: Decision =
            serde_json::from_str(&compile_schedule(graph, timeline, serial_options).unwrap())
                .unwrap();
        assert!(serial
            .execution_regions
            .iter()
            .all(|region| region.stream == Some(RegionStreamClass::Default)));

        let side: Decision =
            serde_json::from_str(&compile_schedule(graph, timeline, side_options).unwrap())
                .unwrap();
        // `produce` has no signed inputs (never capture-safe), `consume` is
        // the side-eligible Region.
        assert_eq!(side.execution_regions.len(), 2);
        let consume = side
            .execution_regions
            .iter()
            .find(|region| region.tasks.iter().any(|task| task == "consume"))
            .unwrap();
        assert_eq!(consume.stream, Some(RegionStreamClass::SideForward));
        let produce = side
            .execution_regions
            .iter()
            .find(|region| region.tasks.iter().any(|task| task == "produce"))
            .unwrap();
        assert_eq!(produce.stream, Some(RegionStreamClass::Default));
        validate_schedule(
            graph,
            timeline,
            side_options,
            &serde_json::to_string(&side).unwrap(),
        )
        .unwrap();

        // A side-stream decision is not valid under options without the flag.
        let error = validate_schedule(
            graph,
            timeline,
            serial_options,
            &serde_json::to_string(&side).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("side-stream"));

        // Provided/hidden inputs are not signed materializations: the model
        // cannot prove the main snapshot covers their production, so the
        // Region stays on the default stream.
        let provided_graph = r#"{"workload_key":"side-stream-provided","tasks":[{"task_id":"run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":true,"inputs":[{"name":"value","kind":"provided","source":"value"}],"outputs":[{"name":"answer","kind":"sink","target":"answer"}]}],"edges":[],"values":[{"value_id":"side:run:answer","version":1,"producer":"run","producer_output":"answer","consumers":[],"size_bytes":0,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1}"#;
        let provided: Decision = serde_json::from_str(
            &compile_schedule(provided_graph, timeline, side_options).unwrap(),
        )
        .unwrap();
        assert_eq!(
            provided.execution_regions[0].stream,
            Some(RegionStreamClass::Default)
        );

        // Without capture safety the compiled callable may carry host-side
        // hazards the stream context could invalidate: default stream.
        let unsafe_graph = r#"{"workload_key":"side-stream-unsafe","tasks":[{"task_id":"run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false,"outputs":[{"name":"answer","kind":"sink","target":"answer"}]}],"edges":[],"values":[{"value_id":"side:run:answer","version":1,"producer":"run","producer_output":"answer","consumers":[],"size_bytes":0,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1}"#;
        let unsafe_decision: Decision =
            serde_json::from_str(&compile_schedule(unsafe_graph, timeline, side_options).unwrap())
                .unwrap();
        assert_eq!(
            unsafe_decision.execution_regions[0].stream,
            Some(RegionStreamClass::Default)
        );
    }

    #[test]
    fn cross_policy_region_merge_is_forbidden() {
        pyo3::prepare_freethreaded_python();
        // Review reproduction: an exact and a jacobi workload each contribute
        // one task in the same anchor window.  Without a policy check the
        // planner merges them and the non-exact tolerance washes over the
        // exact output; the merge must be rejected so each task keeps its own
        // Region (and its own parity tolerance).
        let graph = r#"{"workload_key":"mixed-policy","tasks":[{"task_id":"audit.exact/run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":true,"outputs":[{"name":"answer","kind":"sink","target":"exact"}]},{"task_id":"audit.approx/run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":true,"outputs":[{"name":"answer","kind":"sink","target":"approx"}]}],"edges":[],"values":[{"value_id":"mix:audit.exact/run:answer","version":1,"producer":"audit.exact/run","producer_output":"answer","consumers":[],"size_bytes":0,"alias_set":""},{"value_id":"mix:audit.approx/run:answer","version":1,"producer":"audit.approx/run","producer_output":"answer","consumers":[],"size_bytes":0,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1,"workload_approximations":[["audit.approx","jacobi"]]}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let decision: Decision =
            serde_json::from_str(&compile_schedule(graph, timeline, options).unwrap()).unwrap();
        assert_eq!(decision.execution_regions.len(), 2);
        assert!(decision
            .execution_regions
            .iter()
            .all(|region| region.tasks.len() == 1));
        validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&decision).unwrap(),
        )
        .unwrap();

        // Uniform non-exact policies still merge: the restriction only bites
        // across *different* policies.
        let uniform_graph = graph.replace(
            r#""workload_approximations":[["audit.approx","jacobi"]]"#,
            r#""workload_approximations":[["audit.approx","jacobi"],["audit.exact","jacobi"]]"#,
        );
        let uniform: Decision =
            serde_json::from_str(&compile_schedule(&uniform_graph, timeline, options).unwrap())
                .unwrap();
        assert_eq!(uniform.execution_regions.len(), 1);
    }

    #[test]
    fn rust_emits_and_validates_typed_region_ports() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"ports","tasks":[{"task_id":"run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"value","kind":"provided","source":"value"}],"outputs":[{"name":"answer","kind":"sink","target":"answer"}]}],"edges":[],"values":[{"value_id":"ports:run:answer","version":1,"producer":"run","producer_output":"answer","consumers":[],"size_bytes":0,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled = compile_schedule(graph, timeline, options).unwrap();
        let mut decision: Decision = serde_json::from_str(&compiled).unwrap();
        assert_eq!(decision.execution_regions[0].input_bindings.len(), 1);
        assert_eq!(
            decision.execution_regions[0].input_bindings[0].kind,
            RegionInputKind::Provided
        );
        assert_eq!(decision.execution_regions[0].output_bindings.len(), 1);
        assert_eq!(
            decision.execution_regions[0].output_bindings[0].kind,
            RegionOutputKind::Sink
        );
        assert_eq!(decision.execution_regions[0].task_invocations.len(), 1);
        assert_eq!(
            decision.execution_regions[0].task_invocations[0].inputs[0].value_binding_id,
            decision.execution_regions[0].input_bindings[0].binding_id
        );
        assert_eq!(
            decision.execution_regions[0].task_invocations[0].outputs[0].value_binding_id,
            decision.execution_regions[0].output_bindings[0].value_binding_id
        );

        let mut invocation_tampered = decision.clone();
        invocation_tampered.execution_regions[0].task_invocations[0].outputs[0].value_binding_id =
            "value-tampered".to_owned();
        invocation_tampered.signature.clear();
        invocation_tampered.signature =
            digest(&serde_json::to_string(&invocation_tampered).unwrap());
        let error = validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&invocation_tampered).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("typed output binding"));

        decision.execution_regions[0].input_bindings[0].source = "other".to_owned();
        decision.signature.clear();
        decision.signature = digest(&serde_json::to_string(&decision).unwrap());
        let error = validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&decision).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("artifact metadata"));
    }

    #[test]
    fn region_shape_specialization_merge_is_signed_and_deterministic() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"shape-ports","tasks":[{"task_id":"run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"value","kind":"provided","source":"value"}]}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options = r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"shape_specialization":[["workload/static","x"]],"schema_version":1}"#;
        let decision_json = compile_schedule(graph, timeline, options).unwrap();
        let decision: Decision = serde_json::from_str(&decision_json).unwrap();
        let region = &decision.execution_regions[0];
        let binding_id = &region.input_bindings[0].binding_id;
        let cache_key = workload_compilation_wire_cache_key(graph, timeline, options);
        let request = serde_json::json!({
            "version": 1,
            "base_cache_key": cache_key,
            "base_signature": decision.signature,
            "observed_plan_epoch": 7,
            "region_id": region.region_id,
            "artifact_key": region.artifact_key,
            "inputs": [{
                "binding_id": binding_id,
                "shape": [2, 4],
                "dtype": "torch.float32",
                "device": "cuda:0"
            }]
        });
        let requests = serde_json::to_string(&vec![request.clone(), request.clone()]).unwrap();
        let normalized_json = apply_region_shape_specializations_wire(
            &cache_key,
            graph,
            timeline,
            options,
            &decision_json,
            7,
            &requests,
        )
        .unwrap();
        let normalized: serde_json::Value = serde_json::from_str(&normalized_json).unwrap();
        let specialization = normalized["shape_specialization"].as_array().unwrap();
        assert_eq!(specialization.len(), 2);
        assert_eq!(
            specialization[0][0],
            "runtime/region-shape/".to_owned() + &region.region_id + "/" + binding_id
        );
        assert_eq!(
            specialization[1],
            serde_json::json!(["workload/static", "x"])
        );

        // A request is an immutable transition to a fresh epoch.  Even the
        // exact same observation cannot mutate the active specialization.
        let occupied_decision_json = compile_schedule(graph, timeline, &normalized_json).unwrap();
        let occupied_decision: Decision = serde_json::from_str(&occupied_decision_json).unwrap();
        let occupied_region = &occupied_decision.execution_regions[0];
        assert_eq!(occupied_region.region_id, region.region_id);
        let occupied_request = serde_json::json!({
            "version": 1,
            "base_cache_key": workload_compilation_wire_cache_key(graph, timeline, &normalized_json),
            "base_signature": occupied_decision.signature,
            "observed_plan_epoch": 7,
            "region_id": occupied_region.region_id,
            "artifact_key": occupied_region.artifact_key,
            "inputs": [{
                "binding_id": occupied_region.input_bindings[0].binding_id,
                "shape": [2, 4],
                "dtype": "torch.float32",
                "device": "cuda:0"
            }]
        });
        let occupied_cache_key =
            workload_compilation_wire_cache_key(graph, timeline, &normalized_json);
        let error = apply_region_shape_specializations_wire(
            &occupied_cache_key,
            graph,
            timeline,
            &normalized_json,
            &occupied_decision_json,
            7,
            &serde_json::to_string(&vec![occupied_request.clone()]).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("would overwrite an active Region input"));

        let mut conflicting_options: serde_json::Value =
            serde_json::from_str(&normalized_json).unwrap();
        conflicting_options["shape_specialization"][0][1] = serde_json::json!("conflict");
        let conflicting_options_json = serde_json::to_string(&conflicting_options).unwrap();
        let conflicting_decision_json =
            compile_schedule(graph, timeline, &conflicting_options_json).unwrap();
        let conflicting_decision: Decision =
            serde_json::from_str(&conflicting_decision_json).unwrap();
        let conflicting_region = &conflicting_decision.execution_regions[0];
        let conflicting_cache_key =
            workload_compilation_wire_cache_key(graph, timeline, &conflicting_options_json);
        let mut conflicting_occupied_request = occupied_request;
        conflicting_occupied_request["base_cache_key"] = serde_json::json!(conflicting_cache_key);
        conflicting_occupied_request["base_signature"] =
            serde_json::json!(conflicting_decision.signature);
        conflicting_occupied_request["region_id"] = serde_json::json!(conflicting_region.region_id);
        conflicting_occupied_request["artifact_key"] =
            serde_json::json!(conflicting_region.artifact_key);
        let error = apply_region_shape_specializations_wire(
            &conflicting_cache_key,
            graph,
            timeline,
            &conflicting_options_json,
            &conflicting_decision_json,
            7,
            &serde_json::to_string(&vec![conflicting_occupied_request]).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("would overwrite an active Region input"));

        let mut conflicting = request.clone();
        conflicting["inputs"][0]["shape"] = serde_json::json!([4, 4]);
        let error = apply_region_shape_specializations_wire(
            &cache_key,
            graph,
            timeline,
            options,
            &decision_json,
            7,
            &serde_json::to_string(&vec![request.clone(), conflicting]).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("requests disagree for one Region input"));

        let mut uncovered = request.clone();
        uncovered["inputs"][0]["binding_id"] = serde_json::json!("unknown");
        let error = apply_region_shape_specializations_wire(
            &cache_key,
            graph,
            timeline,
            options,
            &decision_json,
            7,
            &serde_json::to_string(&vec![uncovered]).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("does not cover signed Region inputs"));

        let error = apply_region_shape_specializations_wire(
            &cache_key,
            graph,
            timeline,
            options,
            &decision_json,
            8,
            &serde_json::to_string(&vec![request]).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match the active signed epoch"));
    }

    #[test]
    fn rust_distinguishes_builtin_step_id_from_a_named_provider() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"typed-step","tasks":[{"task_id":"run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"builtin","kind":"step_id","source":"step_id"},{"name":"named","kind":"provided","source":"step_id"}]}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled = compile_schedule(graph, timeline, options).unwrap();
        let decision: Decision = serde_json::from_str(&compiled).unwrap();
        assert_eq!(
            decision.execution_regions[0]
                .input_bindings
                .iter()
                .map(|binding| binding.kind)
                .collect::<Vec<_>>(),
            vec![RegionInputKind::StepId, RegionInputKind::Provided]
        );
    }

    #[test]
    fn graph_v2_rederives_task_capabilities_capture_and_effect_edges() {
        let graph = r#"{"workload_key":"v2-derived","tasks":[{"task_id":"captured","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["backend/test","torch.export"]],"required_capabilities":["backend/test"],"capture_safe":true},{"task_id":"read","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_read","resource":"state"}]},{"task_id":"transition","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_transition","resource":"state"}]},{"task_id":"after","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_read","resource":"state"}]}],"edges":[{"source":"read","target":"transition","kind":"effect"},{"source":"transition","target":"after","kind":"effect"}],"values":[],"states":[{"resource":"state","scope":"global","transition":"optimizer"}],"persistent_state_bytes":0,"schema_version":2}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":["backend/test"],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        compile_schedule(graph, timeline, options).unwrap();
        let mut tampered: serde_json::Value = serde_json::from_str(graph).unwrap();
        tampered["tasks"][0]["required_capabilities"] = serde_json::json!([]);
        assert!(compile_schedule(&tampered.to_string(), timeline, options)
            .unwrap_err()
            .to_string()
            .contains("derived required_capabilities"));

        let mut tampered: serde_json::Value = serde_json::from_str(graph).unwrap();
        tampered["tasks"][0]["capture_safe"] = serde_json::json!(false);
        assert!(compile_schedule(&tampered.to_string(), timeline, options)
            .unwrap_err()
            .to_string()
            .contains("derived capture_safe"));

        let mut tampered: serde_json::Value = serde_json::from_str(graph).unwrap();
        tampered["edges"] = serde_json::json!([
            {"source":"read","target":"transition","kind":"effect"}
        ]);
        assert!(compile_schedule(&tampered.to_string(), timeline, options)
            .unwrap_err()
            .to_string()
            .contains("derived effect edges"));
    }

    #[test]
    fn graph_v2_conditional_transition_serializes_like_a_strict_one() {
        // A conditional transition orders against adjacent accesses exactly
        // like a strict transition: the signed graph cannot prove the update
        // will not fire, so the ordering contract is identical.
        let graph = r#"{"workload_key":"v2-maybe","tasks":[{"task_id":"read","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_read","resource":"state"}]},{"task_id":"maybe","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_transition_maybe","resource":"state"}]},{"task_id":"after","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_read","resource":"state"}]}],"edges":[{"source":"read","target":"maybe","kind":"effect"},{"source":"maybe","target":"after","kind":"effect"}],"values":[],"states":[{"resource":"state","scope":"global","transition":"optimizer"}],"persistent_state_bytes":0,"schema_version":2}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled = compile_schedule(graph, timeline, options).unwrap();
        let decision: Decision = serde_json::from_str(&compiled).unwrap();
        // The ordered read/maybe/read chain internalizes into one Region,
        // exactly as a strict transition chain would.
        assert_eq!(decision.execution_regions.len(), 1);
        assert!(decision.execution_regions[0]
            .effects
            .iter()
            .any(|effect| effect.kind == ScheduleEffectKind::StateTransitionMaybe));

        let mut tampered: serde_json::Value = serde_json::from_str(graph).unwrap();
        tampered["edges"] = serde_json::json!([
            {"source":"read","target":"maybe","kind":"effect"}
        ]);
        assert!(compile_schedule(&tampered.to_string(), timeline, options)
            .unwrap_err()
            .to_string()
            .contains("derived effect edges"));
    }

    #[test]
    fn non_exact_workload_policies_are_explained_and_validated() {
        pyo3::prepare_freethreaded_python();
        let graph = r#"{"workload_key":"approx","tasks":[{"task_id":"run","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:capture@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":true}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":2,"workload_approximations":[["w.a","jacobi"]]}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled = compile_schedule(graph, timeline, options).unwrap();
        let decision: Decision = serde_json::from_str(&compiled).unwrap();
        assert!(decision
            .explanations
            .iter()
            .any(|line| line.contains("w.a") && line.contains("jacobi")));

        // "exact" entries and unsorted entries are rejected.
        let mut tampered: serde_json::Value = serde_json::from_str(graph).unwrap();
        tampered["workload_approximations"] = serde_json::json!([["w.a", "exact"]]);
        assert!(compile_schedule(&tampered.to_string(), timeline, options)
            .unwrap_err()
            .to_string()
            .contains("workload_approximations"));

        let mut tampered: serde_json::Value = serde_json::from_str(graph).unwrap();
        tampered["workload_approximations"] =
            serde_json::json!([["w.b", "jacobi"], ["w.a", "jacobi"]]);
        assert!(compile_schedule(&tampered.to_string(), timeline, options)
            .unwrap_err()
            .to_string()
            .contains("sorted unique"));
    }

    #[test]
    fn region_planner_merges_across_an_independent_topological_task() {
        // `middle` is deliberately between `produce` and `consume` in the
        // stable task order, but is an independent backend boundary. An
        // append-only planner incorrectly creates three Regions; the legal
        // maximal partition keeps the data value internal across it.
        let graph = r#"{"workload_key":"non-adjacent","tasks":[{"task_id":"produce","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:produce@1"],"required_capabilities":[],"capture_safe":false,"outputs":[{"name":"value","kind":"temporary","target":"value"}]},{"task_id":"middle","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["backend:middle@1"],"implementation_kinds":["extern"],"required_capabilities":[],"capture_safe":false},{"task_id":"consume","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:consume@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"value","kind":"result","source":"produce","output":"value"}]}],"edges":[{"source":"produce","target":"consume","kind":"data"}],"values":[{"value_id":"value","version":1,"producer":"produce","producer_output":"value","consumers":["consume"],"size_bytes":32,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"test","backend_version":"1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"begin","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let decision: Decision =
            serde_json::from_str(&compile_schedule(graph, timeline, options).unwrap()).unwrap();
        assert_eq!(
            decision
                .execution_regions
                .iter()
                .map(|region| region.tasks.clone())
                .collect::<Vec<_>>(),
            vec![
                vec!["produce".to_owned(), "consume".to_owned()],
                vec!["middle".to_owned()]
            ]
        );
        assert!(decision.materializations.is_empty());
        assert_eq!(
            decision.execution_order,
            vec![
                "produce".to_owned(),
                "consume".to_owned(),
                "middle".to_owned()
            ]
        );
        validate_schedule(
            graph,
            timeline,
            options,
            &serde_json::to_string(&decision).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn region_partition_ties_follow_topological_order_not_declaration_order() {
        // `late` is declared first but depends on `source`, so the canonical
        // topological order is independent, source, late. This catches a
        // partition tie-break based on raw declaration indices.
        let graph = r#"{"workload_key":"topological-tie","tasks":[{"task_id":"late","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:late@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"value","kind":"result","source":"source","output":"value"}]},{"task_id":"independent","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["backend:independent@1"],"implementation_kinds":["extern"],"required_capabilities":[],"capture_safe":false},{"task_id":"source","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:source@1"],"required_capabilities":[],"capture_safe":false,"outputs":[{"name":"value","kind":"temporary","target":"value"}]}],"edges":[{"source":"source","target":"late","kind":"data"}],"values":[{"value_id":"value","version":1,"producer":"source","producer_output":"value","consumers":["late"],"size_bytes":8,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"test","backend_version":"1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"begin","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let decision: Decision =
            serde_json::from_str(&compile_schedule(graph, timeline, options).unwrap()).unwrap();
        assert_eq!(
            decision.execution_order,
            vec![
                "independent".to_owned(),
                "source".to_owned(),
                "late".to_owned(),
            ]
        );
        assert_eq!(
            decision
                .execution_regions
                .iter()
                .map(|region| region.tasks.clone())
                .collect::<Vec<_>>(),
            vec![
                vec!["independent".to_owned()],
                vec!["source".to_owned(), "late".to_owned()],
            ]
        );
    }

    #[test]
    fn exact_partition_search_beats_the_locally_best_merge() {
        // A-B, B-C, and C-D can each merge, but their windows make those
        // choices mutually exclusive. A local materialization-only merge
        // picks B-C (10 bytes) and leaves 12 bytes. The global partition
        // A-B | C-D leaves only 10 bytes, so this is a regression guard
        // against replacing exact search with pairwise greedy contraction.
        let graph = r#"{"workload_key":"exact-partition","tasks":[{"task_id":"a","timing":{"after":"zero","before":"two"},"temporary_bytes":0,"implementation_ids":["python:a@1"],"required_capabilities":[],"capture_safe":false,"outputs":[{"name":"ab","kind":"temporary","target":"ab"}]},{"task_id":"b","timing":{"after":"one","before":"three"},"temporary_bytes":0,"implementation_ids":["python:b@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"ab","kind":"result","source":"a","output":"ab"}],"outputs":[{"name":"bc","kind":"temporary","target":"bc"}]},{"task_id":"c","timing":{"after":"two","before":"four"},"temporary_bytes":0,"implementation_ids":["python:c@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"bc","kind":"result","source":"b","output":"bc"}],"outputs":[{"name":"cd","kind":"temporary","target":"cd"}]},{"task_id":"d","timing":{"after":"three","before":"five"},"temporary_bytes":0,"implementation_ids":["python:d@1"],"required_capabilities":[],"capture_safe":false,"inputs":[{"name":"cd","kind":"result","source":"c","output":"cd"}]}],"edges":[{"source":"a","target":"b","kind":"data"},{"source":"b","target":"c","kind":"data"},{"source":"c","target":"d","kind":"data"}],"values":[{"value_id":"ab","version":1,"producer":"a","producer_output":"ab","consumers":["b"],"size_bytes":6,"alias_set":""},{"value_id":"bc","version":1,"producer":"b","producer_output":"bc","consumers":["c"],"size_bytes":10,"alias_set":""},{"value_id":"cd","version":1,"producer":"c","producer_output":"cd","consumers":["d"],"size_bytes":6,"alias_set":""}],"persistent_state_bytes":0,"schema_version":1}"#;
        let timeline = r#"{"backend_id":"test","backend_version":"1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"zero","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"one","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"two","ordinal":2,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"three","ordinal":3,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"four","ordinal":4,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"five","ordinal":5,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let decision: Decision =
            serde_json::from_str(&compile_schedule(graph, timeline, options).unwrap()).unwrap();
        assert_eq!(
            decision
                .execution_regions
                .iter()
                .map(|region| region.tasks.clone())
                .collect::<Vec<_>>(),
            vec![
                vec!["a".to_owned(), "b".to_owned()],
                vec!["c".to_owned(), "d".to_owned()],
            ]
        );
        assert_eq!(decision.cost.materialization_bytes, 10);
        assert_eq!(decision.cost.compute_cost, 4);
        assert_eq!(decision.cost.critical_path_extension, 1);
    }

    #[test]
    fn region_planner_internalizes_only_ordered_state_effects() {
        let timeline = r#"{"backend_id":"test","backend_version":"1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"begin","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;
        let ordered = r#"{"workload_key":"ordered-effects","tasks":[{"task_id":"read","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:read@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_read","resource":"state"}]},{"task_id":"transition","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:transition@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_transition","resource":"state"}]},{"task_id":"publish","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:publish@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"version_publish","resource":"state"}]}],"edges":[{"source":"read","target":"transition","kind":"effect"},{"source":"transition","target":"publish","kind":"effect"}],"values":[],"states":[{"resource":"state","scope":"global","transition":"optimizer"}],"persistent_state_bytes":0,"schema_version":2}"#;
        let decision: Decision =
            serde_json::from_str(&compile_schedule(ordered, timeline, options).unwrap()).unwrap();
        assert_eq!(decision.execution_regions.len(), 1);
        assert_eq!(
            decision.execution_regions[0].tasks,
            vec![
                "read".to_owned(),
                "transition".to_owned(),
                "publish".to_owned()
            ]
        );

        let unordered = r#"{"workload_key":"unordered-effects","tasks":[{"task_id":"write","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:write@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"state_write","resource":"state"}]},{"task_id":"publish","timing":{"after":"begin","before":"end"},"temporary_bytes":0,"implementation_ids":["python:publish@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":false,"effects":[{"kind":"version_publish","resource":"state"}]}],"edges":[],"values":[],"states":[{"resource":"state","scope":"global","transition":"optimizer"}],"persistent_state_bytes":0,"schema_version":2}"#;
        let decision: Decision =
            serde_json::from_str(&compile_schedule(unordered, timeline, options).unwrap()).unwrap();
        assert_eq!(decision.execution_regions.len(), 2);
    }

    #[test]
    fn isl_schedule_and_flow_apis_are_linked_and_deterministic() {
        let context = Context::alloc();
        let domain = UnionSet::read_from_str(
            &context,
            "{ Produce[i] : 0 <= i < 4; Consume[i] : 0 <= i < 4 }",
        )
        .unwrap();
        let validity = UnionMap::read_from_str(&context, "{ Produce[i] -> Consume[i] }").unwrap();
        let constraints = ScheduleConstraints::on_domain(domain)
            .unwrap()
            .set_validity(validity)
            .unwrap();
        let schedule = constraints.compute_schedule().unwrap();
        let first = schedule.to_str().unwrap().to_owned();
        let second = schedule.to_str().unwrap().to_owned();
        assert_eq!(first, second);

        let sink = UnionMap::read_from_str(&context, "{ Consume[i] -> Buffer[i] }").unwrap();
        let source = UnionMap::read_from_str(&context, "{ Produce[i] -> Buffer[i] }").unwrap();
        let schedule_map = schedule.get_map().unwrap();
        let flow = UnionAccessInfo::from_sink(sink)
            .unwrap()
            .set_must_source(source)
            .unwrap()
            .set_schedule_map(schedule_map)
            .unwrap()
            .compute_flow()
            .unwrap();
        assert!(!flow
            .get_must_dependence()
            .unwrap()
            .to_str()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn flow_dependences_drop_a_killed_write() {
        pyo3::prepare_freethreaded_python();
        // w1 and w2 both write A[i]; r reads A[i].  The syntactic pairing
        // emits w1 -> r and w2 -> r, while must-source flow keeps only
        // w2 -> r: w1's write is killed before any read.
        let graph = r#"{"workload_key":"flow-kill","tasks":[{"task_id":"w1","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:w1@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":true,"domain":{"name":"affine","indices":["i"],"bounds":[["i","0","N"]],"parameters":["N"]},"accesses":[{"value":"A","indices":[{"coefficients":[["i",1]],"constant":0}],"mode":"write"}]},{"task_id":"w2","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:w2@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":true,"domain":{"name":"affine","indices":["i"],"bounds":[["i","0","N"]],"parameters":["N"]},"accesses":[{"value":"A","indices":[{"coefficients":[["i",1]],"constant":0}],"mode":"write"}]},{"task_id":"r","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:r@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":true,"domain":{"name":"affine","indices":["i"],"bounds":[["i","0","N"]],"parameters":["N"]},"accesses":[{"value":"A","indices":[{"coefficients":[["i",1]],"constant":0}],"mode":"read"}]}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":2}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled = compile_schedule(graph, timeline, options).unwrap();
        let decision: Decision = serde_json::from_str(&compiled).unwrap();
        assert_eq!(decision.schedule_tree.affine_regions.len(), 1);
        let dependences = &decision.schedule_tree.affine_regions[0].dependences;
        assert!(
            dependences.contains("T1") && dependences.contains("T2"),
            "must-source dependence w2 -> r is required: {dependences}"
        );
        assert!(
            !dependences.contains("T0"),
            "killed write w1 must not reach the reader: {dependences}"
        );
    }

    #[test]
    fn flow_dependences_keep_a_self_referential_recurrence() {
        pyo3::prepare_freethreaded_python();
        // recurrence pattern: one task writes kv[t, l] and reads
        // kv[t - 1, l]; the must-source flow relates instance t-1 to t.
        let graph = r#"{"workload_key":"flow-recurrence","tasks":[{"task_id":"rw","timing":{"after":"model.execute","before":"step.end"},"temporary_bytes":0,"implementation_ids":["python:rw@1"],"implementation_kinds":["captured"],"implementation_capabilities":[["torch.export"]],"required_capabilities":[],"capture_safe":true,"domain":{"name":"affine","indices":["t","l"],"bounds":[["t","0","T"],["l","0","L"]],"parameters":["T","L"]},"accesses":[{"value":"kv","indices":[{"coefficients":[["t",1]],"constant":0},{"coefficients":[["l",1]],"constant":0}],"mode":"write"},{"value":"kv","indices":[{"coefficients":[["t",1]],"constant":-1},{"coefficients":[["l",1]],"constant":0}],"mode":"read"}]}],"edges":[],"values":[],"persistent_state_bytes":0,"schema_version":2}"#;
        let timeline = r#"{"backend_id":"vllm","backend_version":"0.27.1","runner_family":"mrv2","graph_mode":"eager","anchors":[{"anchor_id":"model.execute","ordinal":0,"observable":true,"graph_boundary":true,"visible_values":[]},{"anchor_id":"step.end","ordinal":1,"observable":true,"graph_boundary":true,"visible_values":[]}],"regions":[],"static_headroom_bytes":64,"capabilities":[],"schema_version":1}"#;
        let options =
            r#"{"memory_budget_bytes":64,"allow_serial_fallback":true,"schema_version":1}"#;

        let compiled = compile_schedule(graph, timeline, options).unwrap();
        let decision: Decision = serde_json::from_str(&compiled).unwrap();
        assert_eq!(decision.schedule_tree.affine_regions.len(), 1);
        let dependences = &decision.schedule_tree.affine_regions[0].dependences;
        assert!(
            dependences.contains("T0"),
            "the recurrence dependence must survive flow analysis: {dependences}"
        );
        // The flow relates instance t-1 to t and the domain restricts sources
        // to t >= 0, so the boundary read at t = 0 has no in-domain source:
        // the shift is present without the syntactic pairing's blanket row.
        assert!(
            dependences.contains("1 + "),
            "flow dependence must carry the t-1 -> t shift: {dependences}"
        );
    }
}

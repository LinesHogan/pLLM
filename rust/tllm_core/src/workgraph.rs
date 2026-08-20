//! Pure descriptor validation for the Phase 1a WorkGraph IR.
//!
//! The validator consumes the typed, metadata-only descriptors produced by
//! ``tllm.workgraph.ir``. Tensor/CUDA objects, placement decisions, and
//! runtime resources never cross this boundary.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value as JsonValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;

const SCHEMA_VERSION: i64 = 1;
const VERSION_PREFIX: &str = "0.27.";
const GRAPH_MODES: [&str; 2] = ["eager", "piecewise"];
const RUNNER_FAMILIES: [&str; 3] = ["mrv1", "mrv2", "unresolved"];
const MOBILITIES: [&str; 4] = ["pinned", "deferrable", "advanceable", "free"];
const OBSERVATIONS: [&str; 3] = ["at_declared_window", "latest_at_execution", "stale_ok"];
const ACCESS_MODES: [&str; 3] = ["read", "write", "read_write"];
const EDGE_KINDS: [&str; 6] = ["data", "control", "effect", "lease", "collective", "anchor"];
const NODE_KINDS: [&str; 7] = [
    "anchor",
    "compute",
    "provider",
    "transfer",
    "collective",
    "barrier",
    "state",
];
const GRAPH_SAFETY_KINDS: [&str; 3] = ["eager_only", "capture_safe", "replay_stateful"];
const TIMING_WINDOWS: [&str; 4] = ["background", "same_step", "next_step", "out_of_band"];
const TIMING_DEADLINES: [&str; 3] = ["step_end", "next_prepare", "generate_end"];
const BOUNDARY_KINDS: [&str; 1] = ["legacy_boundary"];
const ANCHOR_HOOKS: [&str; 7] = [
    "prepare_inputs.pre",
    "prepare_inputs.post",
    "execute_model.pre",
    "execute_model.post",
    "sample_tokens.pre",
    "sampler.pre_draw",
    "sample_tokens.post",
];

fn quote(value: &str) -> String {
    format!("'{}'", value)
}

fn validation_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct NodeKeyDesc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    owner: String,
    #[pyo3(attribute)]
    ordinal: i64,
}

impl NodeKeyDesc {
    fn text(&self) -> String {
        if self.namespace == "anchor" {
            format!("{}/{}", self.namespace, self.owner)
        } else {
            format!("{}/{}/{}", self.namespace, self.owner, self.ordinal)
        }
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PortIdDesc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    version: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct LocatorFieldDesc {
    #[pyo3(attribute)]
    path: Vec<String>,
    #[pyo3(attribute)]
    value_type: String,
    #[pyo3(attribute)]
    value: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PortAddressDesc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    locator_type: String,
    #[pyo3(attribute)]
    locator_fields: Vec<LocatorFieldDesc>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum PortRefDesc {
    Id(PortIdDesc),
    Address(PortAddressDesc),
}

impl<'py> FromPyObject<'py> for PortRefDesc {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(port_id) = ob.extract::<PortIdDesc>() {
            return Ok(Self::Id(port_id));
        }
        if let Ok(address) = ob.extract::<PortAddressDesc>() {
            return Ok(Self::Address(address));
        }
        Err(PyValueError::new_err(
            "port reference must be a PortId or PortAddressDesc",
        ))
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ValueDesc {
    #[pyo3(attribute)]
    value_id: String,
    #[pyo3(attribute)]
    version: i64,
    #[pyo3(attribute)]
    port: Option<PortRefDesc>,
    #[pyo3(attribute)]
    shape: Vec<String>,
    #[pyo3(attribute)]
    dtype: String,
    #[pyo3(attribute)]
    layout: String,
    #[pyo3(attribute)]
    alias_set: String,
}

impl ValueDesc {
    fn key(&self) -> String {
        format!("{}@{}", self.value_id, self.version)
    }
}

impl PortRefDesc {
    fn label(&self) -> String {
        match self {
            Self::Id(port) => format!("{}:{}@{}", port.namespace, port.name, port.version),
            Self::Address(address) => {
                let fields = address
                    .locator_fields
                    .iter()
                    .map(|field| {
                        format!(
                            "{}:{}={}",
                            field.path.join("."),
                            field.value_type,
                            field.value
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{}|{}|{}", address.kind, address.locator_type, fields)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct AccessDesc {
    #[pyo3(attribute)]
    node_key: NodeKeyDesc,
    #[pyo3(attribute)]
    value: ValueDesc,
    #[pyo3(attribute)]
    mode: String,
    #[pyo3(attribute)]
    port: Option<PortRefDesc>,
    #[pyo3(attribute)]
    subset: Vec<String>,
    #[pyo3(attribute)]
    consistency: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct TimingDesc {
    #[pyo3(attribute)]
    window: String,
    #[pyo3(attribute)]
    mobility: String,
    #[pyo3(attribute)]
    deadline: String,
    #[pyo3(attribute)]
    max_defer_steps: i64,
    #[pyo3(attribute)]
    observes_version: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct AnchorDesc {
    #[pyo3(attribute)]
    backend: String,
    #[pyo3(attribute)]
    runner: String,
    #[pyo3(attribute)]
    graph_mode: String,
    #[pyo3(attribute)]
    hook: String,
    #[pyo3(attribute)]
    externally_owned: bool,
    #[pyo3(attribute)]
    mobility: String,
    #[pyo3(attribute)]
    service_class: String,
}

impl AnchorDesc {
    fn key(&self) -> String {
        format!(
            "anchor/{}/{}/{}/{}",
            self.backend, self.runner, self.graph_mode, self.hook
        )
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct BackendDesc {
    #[pyo3(attribute)]
    backend_id: String,
    #[pyo3(attribute)]
    runner_family: String,
    #[pyo3(attribute)]
    graph_mode: String,
    #[pyo3(attribute)]
    backend_version: String,
    #[pyo3(attribute)]
    capabilities: Vec<String>,
    #[pyo3(attribute)]
    device: String,
    #[pyo3(attribute)]
    rank: i64,
    #[pyo3(attribute)]
    tp_size: i64,
    #[pyo3(attribute)]
    pp_size: i64,
    #[pyo3(attribute)]
    dp_size: i64,
    #[pyo3(attribute)]
    topology: Vec<String>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct SemanticNodeDesc {
    #[pyo3(attribute)]
    node_key: NodeKeyDesc,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    executor: Option<ExecutorKeyDesc>,
    #[pyo3(attribute)]
    reads: Vec<AccessDesc>,
    #[pyo3(attribute)]
    writes: Vec<AccessDesc>,
    #[pyo3(attribute)]
    effects: Vec<String>,
    #[pyo3(attribute)]
    timing: TimingDesc,
    #[pyo3(attribute)]
    anchor: Option<AnchorDesc>,
    #[pyo3(attribute)]
    graph_safety: String,
    #[pyo3(attribute)]
    source: String,
    #[pyo3(attribute)]
    legacy_boundary: Option<String>,
    #[pyo3(attribute)]
    shape_guard: Vec<String>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ExecutorKeyDesc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    version: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ExecutionNodeDesc {
    #[pyo3(attribute)]
    node_key: NodeKeyDesc,
    #[pyo3(attribute)]
    semantic_key: Option<NodeKeyDesc>,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    executor: Option<ExecutorKeyDesc>,
    #[pyo3(attribute)]
    accesses: Vec<AccessDesc>,
    #[pyo3(attribute)]
    residence_keys: Vec<String>,
    #[pyo3(attribute)]
    move_keys: Vec<String>,
    #[pyo3(attribute)]
    lease_keys: Vec<String>,
    #[pyo3(attribute)]
    device: String,
    #[pyo3(attribute)]
    rank: i64,
    #[pyo3(attribute)]
    anchor: Option<AnchorDesc>,
    #[pyo3(attribute)]
    graph_safety: String,
    #[pyo3(attribute)]
    fusion_group: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ResidenceDesc {
    #[pyo3(attribute)]
    value: ValueDesc,
    #[pyo3(attribute)]
    device: String,
    #[pyo3(attribute)]
    rank: i64,
    #[pyo3(attribute)]
    memory_kind: String,
    #[pyo3(attribute)]
    valid_version: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct MoveDesc {
    #[pyo3(attribute)]
    value: ValueDesc,
    #[pyo3(attribute)]
    source: ResidenceDesc,
    #[pyo3(attribute)]
    destination: ResidenceDesc,
    #[pyo3(attribute)]
    subset: Vec<String>,
    #[pyo3(attribute)]
    layout: String,
    #[pyo3(attribute)]
    engine: String,
    #[pyo3(attribute)]
    asynchronous: bool,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct LeaseDesc {
    #[pyo3(attribute)]
    value: ValueDesc,
    #[pyo3(attribute)]
    storage: String,
    #[pyo3(attribute)]
    live_start: String,
    #[pyo3(attribute)]
    live_end: String,
    #[pyo3(attribute)]
    ready_fence: String,
    #[pyo3(attribute)]
    retire_fence: String,
    #[pyo3(attribute)]
    reader_count: i64,
    #[pyo3(attribute)]
    stable_address: bool,
    #[pyo3(attribute)]
    materialized: bool,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct SemanticGraphDesc {
    #[pyo3(attribute)]
    schema_version: i64,
    #[pyo3(attribute)]
    nodes: Vec<SemanticNodeDesc>,
    #[pyo3(attribute)]
    edges: Vec<(NodeKeyDesc, NodeKeyDesc, String)>,
    #[pyo3(attribute)]
    values: Vec<ValueDesc>,
    #[pyo3(attribute)]
    backend: Option<BackendDesc>,
    #[pyo3(attribute)]
    graph_mode: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct DataCentricDesc {
    #[pyo3(attribute)]
    schema_version: i64,
    #[pyo3(attribute)]
    nodes: Vec<ExecutionNodeDesc>,
    #[pyo3(attribute)]
    values: Vec<ValueDesc>,
    #[pyo3(attribute)]
    accesses: Vec<AccessDesc>,
    #[pyo3(attribute)]
    residences: Vec<ResidenceDesc>,
    #[pyo3(attribute)]
    moves: Vec<MoveDesc>,
    #[pyo3(attribute)]
    leases: Vec<LeaseDesc>,
    #[pyo3(attribute)]
    edges: Vec<(NodeKeyDesc, NodeKeyDesc, String)>,
    #[pyo3(attribute)]
    backend: Option<BackendDesc>,
    #[pyo3(attribute)]
    graph_mode: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ArtifactKeyV2Desc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    version: i64,
    #[pyo3(attribute)]
    digest: String,
}

impl ArtifactKeyV2Desc {
    fn text(&self) -> String {
        let suffix = if self.digest.is_empty() {
            String::new()
        } else {
            format!("#{}", self.digest)
        };
        format!(
            "{}:{}@{}{}",
            self.namespace, self.name, self.version, suffix
        )
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct RegionKeyV2Desc {
    #[pyo3(attribute)]
    artifact_key: ArtifactKeyV2Desc,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    ordinal: i64,
}

impl RegionKeyV2Desc {
    fn text(&self) -> String {
        format!(
            "{}:{}:{}",
            self.artifact_key.text(),
            self.kind,
            self.ordinal
        )
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct LeafOperatorV2Desc {
    #[pyo3(attribute)]
    ordinal: i64,
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    overload: String,
    #[pyo3(attribute)]
    inputs: Vec<i64>,
    #[pyo3(attribute)]
    outputs: Vec<String>,
    #[pyo3(attribute)]
    module_path: Vec<String>,
    #[pyo3(attribute)]
    operator_kind: String,
    #[pyo3(attribute)]
    effects: Vec<String>,
    #[pyo3(attribute)]
    capabilities: Vec<String>,
    #[pyo3(attribute)]
    attributes: Vec<OperatorAttributeV2Desc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct OperatorAttributeV2Desc {
    #[pyo3(attribute)]
    path: Vec<String>,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    value: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct LeafArtifactV2Desc {
    #[pyo3(attribute)]
    artifact_key: ArtifactKeyV2Desc,
    #[pyo3(attribute)]
    operators: Vec<LeafOperatorV2Desc>,
    #[pyo3(attribute)]
    input_values: Vec<String>,
    #[pyo3(attribute)]
    output_values: Vec<String>,
    #[pyo3(attribute)]
    terminal_ordinal: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct LeafRegionV2Desc {
    #[pyo3(attribute)]
    key: RegionKeyV2Desc,
    #[pyo3(attribute)]
    artifact_key: ArtifactKeyV2Desc,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    operator_ordinals: Vec<i64>,
    #[pyo3(attribute)]
    leaf_artifact: Option<LeafArtifactV2Desc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct TaskRegionV2Desc {
    #[pyo3(attribute)]
    key: RegionKeyV2Desc,
    #[pyo3(attribute)]
    node_key: NodeKeyDesc,
    #[pyo3(attribute)]
    executor: ExecutorKeyDesc,
    #[pyo3(attribute)]
    boundary: String,
    #[pyo3(attribute)]
    module_path: Vec<String>,
    #[pyo3(attribute)]
    operator_ordinals: Vec<i64>,
    #[pyo3(attribute)]
    input_values: Vec<String>,
    #[pyo3(attribute)]
    output_values: Vec<String>,
    #[pyo3(attribute)]
    effects: Vec<String>,
    #[pyo3(attribute)]
    capabilities: Vec<String>,
    #[pyo3(attribute)]
    dependencies: Vec<RegionKeyV2Desc>,
    #[pyo3(attribute)]
    children: Vec<LeafRegionV2Desc>,
    #[pyo3(attribute)]
    kind: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct SourceSpanV2Desc {
    #[pyo3(attribute)]
    file: String,
    #[pyo3(attribute)]
    start_line: i64,
    #[pyo3(attribute)]
    start_column: i64,
    #[pyo3(attribute)]
    end_line: i64,
    #[pyo3(attribute)]
    end_column: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct RootFlowV2Desc {
    #[pyo3(attribute)]
    key: RegionKeyV2Desc,
    #[pyo3(attribute)]
    artifact_key: ArtifactKeyV2Desc,
    #[pyo3(attribute)]
    source_span: SourceSpanV2Desc,
    #[pyo3(attribute)]
    status: String,
    #[pyo3(attribute)]
    shape_guard: ShapeGuardV2Desc,
    #[pyo3(attribute)]
    inputs: Vec<PortBindingV2Desc>,
    #[pyo3(attribute)]
    outputs: Vec<PortBindingV2Desc>,
    #[pyo3(attribute)]
    values: Vec<CapturedValueV2Desc>,
    #[pyo3(attribute)]
    effects: Vec<String>,
    #[pyo3(attribute)]
    diagnostics: Vec<FlowDiagnosticV2Desc>,
    #[pyo3(attribute)]
    children: Vec<TaskRegionV2Desc>,
    #[pyo3(attribute)]
    kind: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ShapeGuardExprV2Desc {
    #[pyo3(attribute)]
    value_path: Vec<i64>,
    #[pyo3(attribute)]
    dimension: i64,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    lower_bound: i64,
    #[pyo3(attribute)]
    upper_bound: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ShapeGuardV2Desc {
    #[pyo3(attribute)]
    expressions: Vec<ShapeGuardExprV2Desc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct TensorDTypeV2Desc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct TensorDeviceV2Desc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    index: Option<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PhysicalKeyDesc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    ordinal: i64,
}

impl PhysicalKeyDesc {
    fn text(&self) -> String {
        format!("{}:{}/{}", self.namespace, self.name, self.ordinal)
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PhysicalBucketDesc {
    #[pyo3(attribute)]
    key: PhysicalKeyDesc,
    #[pyo3(attribute)]
    shape: Vec<i64>,
    #[pyo3(attribute)]
    dtype: TensorDTypeV2Desc,
    #[pyo3(attribute)]
    device: TensorDeviceV2Desc,
    #[pyo3(attribute)]
    slots: i64,
    #[pyo3(attribute)]
    ownership: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PhysicalFenceDesc {
    #[pyo3(attribute)]
    fence: PhysicalKeyDesc,
    #[pyo3(attribute)]
    producer: NodeKeyDesc,
    #[pyo3(attribute)]
    consumer: NodeKeyDesc,
    #[pyo3(attribute)]
    kind: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PhysicalNodeDesc {
    #[pyo3(attribute)]
    node_key: NodeKeyDesc,
    #[pyo3(attribute)]
    executor: ExecutorKeyDesc,
    #[pyo3(attribute)]
    stream: String,
    #[pyo3(attribute)]
    fences: Vec<PhysicalFenceDesc>,
    #[pyo3(attribute)]
    artifact_key: Option<ArtifactKeyV2Desc>,
    #[pyo3(attribute)]
    buffer_bucket: Option<PhysicalKeyDesc>,
    #[pyo3(attribute)]
    capture_policy: String,
    #[pyo3(attribute)]
    fallback_policy: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, PartialEq)]
struct PhysicalFlowPlanDesc {
    #[pyo3(attribute)]
    plan_epoch: i64,
    #[pyo3(attribute)]
    nodes: Vec<PhysicalNodeDesc>,
    #[pyo3(attribute)]
    buffer_buckets: Vec<PhysicalBucketDesc>,
    #[pyo3(attribute)]
    schema_version: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PlacementV3Desc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    index: Option<i64>,
}

impl PlacementV3Desc {
    fn text(&self) -> String {
        match self.index {
            Some(index) => format!("{}:{}", self.kind, index),
            None => self.kind.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, PartialEq)]
struct DeviceTopologyV3Desc {
    #[pyo3(attribute)]
    peer_access: HashSet<(i64, i64)>,
    #[pyo3(attribute)]
    cuda_devices: Vec<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, Ord, PartialEq, PartialOrd)]
struct ValueKeyV3Desc {
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    version: i64,
}

impl ValueKeyV3Desc {
    fn text(&self) -> String {
        format!("{}@{}", self.name, self.version)
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ResidencyV3Desc {
    #[pyo3(attribute)]
    value: ValueKeyV3Desc,
    #[pyo3(attribute)]
    placement: PlacementV3Desc,
    #[pyo3(attribute)]
    memory: String,
}

impl ResidencyV3Desc {
    fn text(&self) -> String {
        format!(
            "{}|{}|{}",
            self.value.text(),
            self.placement.text(),
            self.memory
        )
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct DurationProfileV3Desc {
    #[pyo3(attribute)]
    microseconds: i64,
    #[pyo3(attribute)]
    samples: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct SemanticOracleKeyV3Desc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    version: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ImplementationSelectionV3Desc {
    #[pyo3(attribute)]
    node_key: NodeKeyDesc,
    #[pyo3(attribute)]
    logical_executor: ExecutorKeyDesc,
    #[pyo3(attribute)]
    key: ExecutorKeyDesc,
    #[pyo3(attribute)]
    placement: PlacementV3Desc,
    #[pyo3(attribute)]
    duration: DurationProfileV3Desc,
    #[pyo3(attribute)]
    oracle_key: Option<SemanticOracleKeyV3Desc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PlacedNodeV3Desc {
    #[pyo3(attribute)]
    node_key: NodeKeyDesc,
    #[pyo3(attribute)]
    implementation: ExecutorKeyDesc,
    #[pyo3(attribute)]
    placement: PlacementV3Desc,
    #[pyo3(attribute)]
    start_at: i64,
    #[pyo3(attribute)]
    end_at: i64,
    #[pyo3(attribute)]
    depends_on: Vec<NodeKeyDesc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, Ord, PartialEq, PartialOrd)]
struct MoveKeyV3Desc {
    #[pyo3(attribute)]
    value: ValueKeyV3Desc,
    #[pyo3(attribute)]
    ordinal: i64,
}

impl MoveKeyV3Desc {
    fn text(&self) -> String {
        format!("{}/move/{}", self.value.text(), self.ordinal)
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct MoveV3Desc {
    #[pyo3(attribute)]
    value: ValueKeyV3Desc,
    #[pyo3(attribute)]
    source: ResidencyV3Desc,
    #[pyo3(attribute)]
    destination: ResidencyV3Desc,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    byte_count: i64,
    #[pyo3(attribute)]
    asynchronous: bool,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ScheduledMoveV3Desc {
    #[pyo3(attribute)]
    key: MoveKeyV3Desc,
    #[pyo3(attribute)]
    r#move: MoveV3Desc,
    #[pyo3(attribute)]
    engine: String,
    #[pyo3(attribute)]
    start_at: i64,
    #[pyo3(attribute)]
    end_at: i64,
    #[pyo3(attribute)]
    producer: Option<NodeKeyDesc>,
    #[pyo3(attribute)]
    consumer: Option<NodeKeyDesc>,
    #[pyo3(attribute)]
    depends_on: Vec<MoveKeyV3Desc>,
    #[pyo3(attribute)]
    used_fallback: bool,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PlacementLeaseV3Desc {
    #[pyo3(attribute)]
    residency: ResidencyV3Desc,
    #[pyo3(attribute)]
    live_start: i64,
    #[pyo3(attribute)]
    live_end: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CopyEngineCapacityV3Desc {
    #[pyo3(attribute)]
    engine: String,
    #[pyo3(attribute)]
    capacity: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ValueFlowV3Desc {
    #[pyo3(attribute)]
    value: ValueKeyV3Desc,
    #[pyo3(attribute)]
    producer: NodeKeyDesc,
    #[pyo3(attribute)]
    consumer: NodeKeyDesc,
    #[pyo3(attribute)]
    byte_count: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, PartialEq)]
struct PlacementPlanV3Desc {
    #[pyo3(attribute)]
    plan_epoch: i64,
    #[pyo3(attribute)]
    selections: Vec<ImplementationSelectionV3Desc>,
    #[pyo3(attribute)]
    schedule: Vec<PlacedNodeV3Desc>,
    #[pyo3(attribute)]
    topology: DeviceTopologyV3Desc,
    #[pyo3(attribute)]
    value_flows: Vec<ValueFlowV3Desc>,
    #[pyo3(attribute)]
    residencies: Vec<ResidencyV3Desc>,
    #[pyo3(attribute)]
    moves: Vec<ScheduledMoveV3Desc>,
    #[pyo3(attribute)]
    leases: Vec<PlacementLeaseV3Desc>,
    #[pyo3(attribute)]
    copy_engines: Vec<CopyEngineCapacityV3Desc>,
    #[pyo3(attribute)]
    schema_version: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, Ord, PartialEq, PartialOrd)]
struct StateKeyV4Desc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    version: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ShardSpecV4Desc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    partitions: i64,
    #[pyo3(attribute)]
    axis: Option<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PlacementAffinityV4Desc {
    #[pyo3(attribute)]
    ranks: Vec<i64>,
    #[pyo3(attribute)]
    node_ids: Vec<String>,
    #[pyo3(attribute)]
    role: Option<String>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct StateRequirementV4Desc {
    #[pyo3(attribute)]
    key: StateKeyV4Desc,
    #[pyo3(attribute)]
    scope: String,
    #[pyo3(attribute)]
    shard: ShardSpecV4Desc,
    #[pyo3(attribute)]
    affinity: PlacementAffinityV4Desc,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct TopologyRankV4Desc {
    #[pyo3(attribute)]
    rank: i64,
    #[pyo3(attribute)]
    node_id: String,
    #[pyo3(attribute)]
    local_rank: i64,
    #[pyo3(attribute)]
    placement: PlacementV3Desc,
    #[pyo3(attribute)]
    role: String,
    #[pyo3(attribute)]
    tp_rank: Option<i64>,
    #[pyo3(attribute)]
    pp_rank: Option<i64>,
    #[pyo3(attribute)]
    dp_rank: Option<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, PartialEq)]
struct DistributedTopologyV4Desc {
    #[pyo3(attribute)]
    ranks: Vec<TopologyRankV4Desc>,
    #[pyo3(attribute)]
    tp_size: i64,
    #[pyo3(attribute)]
    pp_size: i64,
    #[pyo3(attribute)]
    dp_size: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct StateResidenceV4Desc {
    #[pyo3(attribute)]
    key: StateKeyV4Desc,
    #[pyo3(attribute)]
    rank: i64,
    #[pyo3(attribute)]
    placement: PlacementV3Desc,
    #[pyo3(attribute)]
    shard_index: i64,
    #[pyo3(attribute)]
    replica_index: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, Ord, PartialEq, PartialOrd)]
struct CommunicatorKeyV4Desc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    group_index: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CommunicatorV4Desc {
    #[pyo3(attribute)]
    key: CommunicatorKeyV4Desc,
    #[pyo3(attribute)]
    ranks: Vec<i64>,
    #[pyo3(attribute)]
    backend: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, Ord, PartialEq, PartialOrd)]
struct CollectiveKeyV4Desc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    ordinal: i64,
}

impl CollectiveKeyV4Desc {
    fn text(&self) -> String {
        format!("{}:{}/{}", self.namespace, self.name, self.ordinal)
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CollectiveRequestV4Desc {
    #[pyo3(attribute)]
    key: CollectiveKeyV4Desc,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    group: String,
    #[pyo3(attribute)]
    group_index: i64,
    #[pyo3(attribute)]
    backend: String,
    #[pyo3(attribute)]
    window: String,
    #[pyo3(attribute)]
    value: Option<ValueKeyV3Desc>,
    #[pyo3(attribute)]
    depends_on: Vec<CollectiveKeyV4Desc>,
    #[pyo3(attribute)]
    root_rank: Option<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CollectiveV4Desc {
    #[pyo3(attribute)]
    key: CollectiveKeyV4Desc,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    communicator: CommunicatorKeyV4Desc,
    #[pyo3(attribute)]
    sequence: i64,
    #[pyo3(attribute)]
    window: String,
    #[pyo3(attribute)]
    value: Option<ValueKeyV3Desc>,
    #[pyo3(attribute)]
    depends_on: Vec<CollectiveKeyV4Desc>,
    #[pyo3(attribute)]
    root_rank: Option<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct LocalCollectiveScheduleV4Desc {
    #[pyo3(attribute)]
    rank: i64,
    #[pyo3(attribute)]
    collectives: Vec<CollectiveKeyV4Desc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, PartialEq)]
struct DistributedFlowPlanV4Desc {
    #[pyo3(attribute)]
    plan_epoch: i64,
    #[pyo3(attribute)]
    topology: DistributedTopologyV4Desc,
    #[pyo3(attribute)]
    state_requirements: Vec<StateRequirementV4Desc>,
    #[pyo3(attribute)]
    state_residences: Vec<StateResidenceV4Desc>,
    #[pyo3(attribute)]
    collective_requests: Vec<CollectiveRequestV4Desc>,
    #[pyo3(attribute)]
    communicators: Vec<CommunicatorV4Desc>,
    #[pyo3(attribute)]
    collectives: Vec<CollectiveV4Desc>,
    #[pyo3(attribute)]
    local_schedules: Vec<LocalCollectiveScheduleV4Desc>,
    #[pyo3(attribute)]
    schema_version: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, Ord, PartialEq, PartialOrd)]
struct VersionKeyV5Desc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    version: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, Ord, PartialEq, PartialOrd)]
struct OrdinalKeyV5Desc {
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    ordinal: i64,
}

impl OrdinalKeyV5Desc {
    fn text(&self) -> String {
        format!("{}:{}/{}", self.namespace, self.name, self.ordinal)
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct FixedRatioV5Desc {
    #[pyo3(attribute)]
    numerator: i64,
    #[pyo3(attribute)]
    denominator: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct IntegerEwmaProfileV5Desc {
    #[pyo3(attribute)]
    key: VersionKeyV5Desc,
    #[pyo3(attribute)]
    estimate: i64,
    #[pyo3(attribute)]
    alpha: FixedRatioV5Desc,
    #[pyo3(attribute)]
    samples: i64,
    #[pyo3(attribute)]
    minimum: i64,
    #[pyo3(attribute)]
    maximum: Option<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct ShapeBucketV5Desc {
    #[pyo3(attribute)]
    key: VersionKeyV5Desc,
    #[pyo3(attribute)]
    shape: Vec<i64>,
    #[pyo3(attribute)]
    lower_bounds: Vec<i64>,
    #[pyo3(attribute)]
    upper_bounds: Vec<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct WorkloadBucketV5Desc {
    #[pyo3(attribute)]
    key: VersionKeyV5Desc,
    #[pyo3(attribute)]
    shape_buckets: Vec<VersionKeyV5Desc>,
    #[pyo3(attribute)]
    required_by_default: bool,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct InterferenceBudgetV5Desc {
    #[pyo3(attribute)]
    key: VersionKeyV5Desc,
    #[pyo3(attribute)]
    capacity: i64,
    #[pyo3(attribute)]
    profile: Option<VersionKeyV5Desc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct BudgetUseV5Desc {
    #[pyo3(attribute)]
    key: VersionKeyV5Desc,
    #[pyo3(attribute)]
    amount: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, PartialEq)]
struct CandidateMoveSetV5Desc {
    #[pyo3(attribute)]
    key: OrdinalKeyV5Desc,
    #[pyo3(attribute)]
    moves: Vec<ScheduledMoveV3Desc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CandidateActionV5Desc {
    #[pyo3(attribute)]
    key: OrdinalKeyV5Desc,
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    semantic_key: String,
    #[pyo3(attribute)]
    workload_bucket: Option<VersionKeyV5Desc>,
    #[pyo3(attribute)]
    shape_bucket: Option<VersionKeyV5Desc>,
    #[pyo3(attribute)]
    move_set: Option<OrdinalKeyV5Desc>,
    #[pyo3(attribute)]
    fallback_key: Option<OrdinalKeyV5Desc>,
    #[pyo3(attribute)]
    deadline: i64,
    #[pyo3(attribute)]
    ready_at: i64,
    #[pyo3(attribute)]
    duration: i64,
    #[pyo3(attribute)]
    buffer_credits: i64,
    #[pyo3(attribute)]
    budget_uses: Vec<BudgetUseV5Desc>,
    #[pyo3(attribute)]
    priority: i64,
    #[pyo3(attribute)]
    utility: i64,
    #[pyo3(attribute)]
    required: bool,
    #[pyo3(attribute)]
    unknown_bucket: bool,
    #[pyo3(attribute)]
    authoritative: bool,
}

#[derive(Clone, Debug, Eq, FromPyObject, PartialEq)]
struct AdaptiveFlowPlanV5Desc {
    #[pyo3(attribute)]
    plan_epoch: i64,
    #[pyo3(attribute)]
    placement_plan: PlacementPlanV3Desc,
    #[pyo3(attribute)]
    scheduled_moves: Vec<ScheduledMoveV3Desc>,
    #[pyo3(attribute)]
    move_sets: Vec<CandidateMoveSetV5Desc>,
    #[pyo3(attribute)]
    workload_buckets: Vec<WorkloadBucketV5Desc>,
    #[pyo3(attribute)]
    shape_buckets: Vec<ShapeBucketV5Desc>,
    #[pyo3(attribute)]
    budgets: Vec<InterferenceBudgetV5Desc>,
    #[pyo3(attribute)]
    actions: Vec<CandidateActionV5Desc>,
    #[pyo3(attribute)]
    unknown_bucket_fallback: OrdinalKeyV5Desc,
    #[pyo3(attribute)]
    buffer_capacity: i64,
    #[pyo3(attribute)]
    mode: String,
    #[pyo3(attribute)]
    schema_version: i64,
    #[pyo3(attribute)]
    profiles: Vec<IntegerEwmaProfileV5Desc>,
    #[pyo3(attribute)]
    placement_signature: String,
    #[pyo3(attribute)]
    densification_limit: i64,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct UseDefV2Desc {
    #[pyo3(attribute)]
    producer: i64,
    #[pyo3(attribute)]
    inputs: Vec<i64>,
    #[pyo3(attribute)]
    users: Vec<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CapturedValueV2Desc {
    #[pyo3(attribute)]
    value_id: String,
    #[pyo3(attribute)]
    version: i64,
    #[pyo3(attribute)]
    shape: Vec<String>,
    #[pyo3(attribute)]
    dtype: TensorDTypeV2Desc,
    #[pyo3(attribute)]
    device: TensorDeviceV2Desc,
    #[pyo3(attribute)]
    layout: String,
    #[pyo3(attribute)]
    stride: Vec<String>,
    #[pyo3(attribute)]
    use_def: UseDefV2Desc,
    #[pyo3(attribute)]
    alias_set: String,
    #[pyo3(attribute)]
    effect: String,
}

impl CapturedValueV2Desc {
    fn key(&self) -> String {
        format!("{}@{}", self.value_id, self.version)
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct FlowDiagnosticV2Desc {
    #[pyo3(attribute)]
    code: String,
    #[pyo3(attribute)]
    message: String,
    #[pyo3(attribute)]
    source_span: SourceSpanV2Desc,
    #[pyo3(attribute)]
    detail: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PortBindingV2Desc {
    #[pyo3(attribute)]
    port: PortRefDesc,
    #[pyo3(attribute)]
    direction: String,
    #[pyo3(attribute)]
    value_path: Vec<i64>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CaptureAttachmentV2Desc {
    #[pyo3(attribute)]
    node_key: NodeKeyDesc,
    #[pyo3(attribute)]
    artifact_key: ArtifactKeyV2Desc,
    #[pyo3(attribute)]
    bindings: Vec<PortBindingV2Desc>,
    #[pyo3(attribute)]
    anchor_node_key: Option<NodeKeyDesc>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CaptureLeaseV2Desc {
    #[pyo3(attribute)]
    value_id: String,
    #[pyo3(attribute)]
    live_start: NodeKeyDesc,
    #[pyo3(attribute)]
    live_end: NodeKeyDesc,
    #[pyo3(attribute)]
    reader_count: i64,
    #[pyo3(attribute)]
    materialized: bool,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CaptureDataCentricV2Desc {
    #[pyo3(attribute)]
    leases: Vec<CaptureLeaseV2Desc>,
    #[pyo3(attribute)]
    moves: Vec<String>,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct CaptureHierarchyV2Desc {
    #[pyo3(attribute)]
    roots: Vec<RootFlowV2Desc>,
    #[pyo3(attribute)]
    attachments: Vec<CaptureAttachmentV2Desc>,
    #[pyo3(attribute)]
    data_centric: CaptureDataCentricV2Desc,
    #[pyo3(attribute)]
    schema_version: i64,
}

#[pyclass]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkGraphValidation {
    #[pyo3(get)]
    pub topological_order: Vec<String>,
}

fn validate_unique<T>(
    items: &[T],
    kind: &str,
    key: impl Fn(&T) -> String,
) -> Result<HashSet<String>, String> {
    let mut keys = HashSet::with_capacity(items.len());
    for item in items {
        let value = key(item);
        if value.is_empty() {
            return Err(format!("{} key must not be empty", kind));
        }
        if !keys.insert(value.clone()) {
            return Err(format!("duplicate {} key {}", kind, quote(&value)));
        }
    }
    Ok(keys)
}

fn validate_port_ref(port: &PortRefDesc, layer: &str) -> Result<(), String> {
    match port {
        PortRefDesc::Id(port_id) => {
            if port_id.namespace.is_empty()
                || port_id.namespace.contains('.')
                || port_id.namespace.contains(':')
                || port_id.name.is_empty()
                || port_id.name.contains(':')
                || port_id.version < 1
            {
                return Err(format!(
                    "{} declares an invalid PortId {}",
                    layer,
                    quote(&port.label())
                ));
            }
        }
        PortRefDesc::Address(address) => {
            if address.kind.is_empty()
                || address.locator_type.is_empty() && !address.locator_fields.is_empty()
            {
                return Err(format!(
                    "{} declares an invalid PortAddressDesc {}",
                    layer,
                    quote(&port.label())
                ));
            }
            for field in &address.locator_fields {
                if field.path.is_empty()
                    || field.path.iter().any(|segment| segment.is_empty())
                    || field.value_type.is_empty()
                {
                    return Err(format!(
                        "{} declares an invalid locator field in {}",
                        layer,
                        quote(&port.label())
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_values(values: &[ValueDesc], kind: &str) -> Result<HashSet<String>, String> {
    let keys = validate_unique(values, kind, ValueDesc::key)?;
    for value in values {
        if value.value_id.is_empty() || value.version < 1 {
            return Err(format!(
                "{} {} has invalid version",
                kind,
                quote(&value.key())
            ));
        }
        if let Some(port) = &value.port {
            validate_port_ref(port, &format!("{} value {}", kind, quote(&value.key())))?;
        }
    }
    Ok(keys)
}

fn validate_node_key(node_key: &NodeKeyDesc, kind: &str) -> Result<(), String> {
    if node_key.namespace.is_empty() || node_key.owner.is_empty() {
        return Err(format!(
            "{} key must have non-empty namespace and owner",
            kind
        ));
    }
    if node_key.namespace.contains('/') {
        return Err(format!("{} key namespace must not contain '/'", kind));
    }
    if node_key.namespace == "anchor" {
        let segments: Vec<&str> = node_key.owner.split('/').collect();
        if segments.len() != 4
            || segments
                .iter()
                .any(|segment| segment.is_empty() || segment.contains('\\'))
        {
            return Err(format!(
                "{} anchor owner must contain exactly four non-empty segments",
                kind
            ));
        }
    } else if node_key.owner.contains('/') || node_key.owner.contains('\\') {
        return Err(format!("{} key owner must not contain '/' or '\\'", kind));
    }
    if node_key.ordinal < 0 {
        return Err(format!("{} key has negative ordinal", kind));
    }
    Ok(())
}

fn validate_executor(executor: Option<&ExecutorKeyDesc>, layer: &str) -> Result<(), String> {
    let Some(executor) = executor else {
        return Ok(());
    };
    if executor.namespace.is_empty()
        || executor.name.is_empty()
        || executor.version.is_empty()
        || executor.namespace.contains('/')
        || executor.name.contains('/')
    {
        return Err(format!("{} node has an invalid executor key", layer));
    }
    Ok(())
}

fn validate_node_domain(
    kind: &str,
    graph_safety: &str,
    legacy_boundary: Option<&String>,
    executor: Option<&ExecutorKeyDesc>,
    layer: &str,
) -> Result<(), String> {
    if !NODE_KINDS.contains(&kind) {
        return Err(format!(
            "{} node declares unsupported kind {}",
            layer,
            quote(kind)
        ));
    }
    if !GRAPH_SAFETY_KINDS.contains(&graph_safety) {
        return Err(format!(
            "{} node declares unsupported graph safety {}",
            layer,
            quote(graph_safety)
        ));
    }
    if let Some(boundary) = legacy_boundary {
        if !BOUNDARY_KINDS.contains(&boundary.as_str()) {
            return Err(format!(
                "{} node declares unsupported legacy boundary {}",
                layer,
                quote(boundary)
            ));
        }
    }
    validate_executor(executor, layer)
}

fn validate_backend(backend: &BackendDesc, graph_mode: &str) -> Result<(), String> {
    if backend.backend_id.is_empty()
        || backend.backend_id.contains('/')
        || backend.backend_id.contains('\\')
    {
        return Err("backend snapshot has an invalid backend_id".to_string());
    }
    if backend.backend_version.is_empty() || !backend.backend_version.starts_with(VERSION_PREFIX) {
        return Err(format!(
            "unsupported backend version family {}; expected vLLM 0.27.x",
            quote(&backend.backend_version)
        ));
    }
    if !RUNNER_FAMILIES.contains(&backend.runner_family.as_str()) {
        return Err(format!(
            "unsupported runner family {}; expected one of [{}]",
            quote(&backend.runner_family),
            RUNNER_FAMILIES
                .iter()
                .map(|value| quote(value))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !GRAPH_MODES.contains(&backend.graph_mode.as_str()) || backend.graph_mode != graph_mode {
        return Err(format!(
            "unsupported or inconsistent graph mode {}; expected eager or piecewise",
            quote(graph_mode)
        ));
    }
    if backend.device.is_empty()
        || backend.rank < 0
        || backend.tp_size < 1
        || backend.pp_size < 1
        || backend.dp_size < 1
    {
        return Err("backend snapshot has invalid placement dimensions".to_string());
    }
    Ok(())
}

fn validate_backend_parity(left: &BackendDesc, right: &BackendDesc) -> Result<(), String> {
    if left != right {
        return Err(format!(
            "semantic/data backend snapshot mismatch: {} vs {}",
            quote(&left.backend_id),
            quote(&right.backend_id)
        ));
    }
    Ok(())
}

fn validate_timing(node_key: &str, timing: &TimingDesc, legacy: bool) -> Result<(), String> {
    if !TIMING_WINDOWS.contains(&timing.window.as_str()) {
        return Err(format!(
            "node {} declares unsupported timing window {}",
            quote(node_key),
            quote(&timing.window)
        ));
    }
    if !TIMING_DEADLINES.contains(&timing.deadline.as_str()) {
        return Err(format!(
            "node {} declares unsupported deadline {}",
            quote(node_key),
            quote(&timing.deadline)
        ));
    }
    if !MOBILITIES.contains(&timing.mobility.as_str()) {
        return Err(format!(
            "node {} declares unsupported mobility {}",
            quote(node_key),
            quote(&timing.mobility)
        ));
    }
    if !OBSERVATIONS.contains(&timing.observes_version.as_str()) {
        return Err(format!(
            "node {} declares unsupported version observation {}",
            quote(node_key),
            quote(&timing.observes_version)
        ));
    }
    if timing.max_defer_steps < 0 {
        return Err(format!(
            "node {} declares negative max_defer_steps",
            quote(node_key)
        ));
    }
    if !legacy && timing.mobility != "pinned" {
        return Err(format!(
            "WorkGraph-native node {} must be pinned",
            quote(node_key)
        ));
    }
    // Native WorkGraph nodes are executable by this phase and therefore must
    // observe the version captured at their declared window. Legacy
    // boundaries remain owned by the existing runtime plan; retain their
    // authoritative timing contract after validating the enum values above.
    if !legacy && timing.observes_version != "at_declared_window" {
        return Err(format!(
            "WorkGraph-native node {} must observe at_declared_window",
            quote(node_key)
        ));
    }
    Ok(())
}

fn validate_anchor(
    anchor: &AnchorDesc,
    backend: &BackendDesc,
    graph_mode: &str,
    anchors: &mut HashMap<String, AnchorDesc>,
) -> Result<String, String> {
    let key = anchor.key();
    if anchor.backend != backend.backend_id {
        return Err(format!(
            "anchor {} backend {} differs from backend snapshot {}",
            quote(&key),
            quote(&anchor.backend),
            quote(&backend.backend_id)
        ));
    }
    if anchor.runner != backend.runner_family {
        return Err(format!(
            "anchor {} runner {} differs from backend snapshot {}",
            quote(&key),
            quote(&anchor.runner),
            quote(&backend.runner_family)
        ));
    }
    if anchor.graph_mode != graph_mode {
        return Err(format!(
            "anchor {} graph mode {} differs from graph mode {}",
            quote(&key),
            quote(&anchor.graph_mode),
            quote(graph_mode)
        ));
    }
    if !ANCHOR_HOOKS.contains(&anchor.hook.as_str()) {
        return Err(format!(
            "anchor {} declares unsupported hook {}; layer-owned hooks are legacy boundaries",
            quote(&key),
            quote(&anchor.hook)
        ));
    }
    if !anchor.externally_owned || anchor.mobility != "pinned" {
        return Err(format!(
            "anchor {} must be externally_owned and pinned",
            quote(&key)
        ));
    }
    if anchor.service_class.is_empty() {
        return Err(format!("anchor {} has an empty service_class", quote(&key)));
    }
    if let Some(existing) = anchors.get(&key) {
        if existing != anchor {
            return Err(format!("conflicting duplicate anchor {}", quote(&key)));
        }
    } else {
        anchors.insert(key.clone(), anchor.clone());
    }
    Ok(key)
}

fn validate_explicit_anchor(
    node_key: &NodeKeyDesc,
    kind: &str,
    anchor: Option<&AnchorDesc>,
    layer: &str,
) -> Result<(), String> {
    match anchor {
        Some(anchor) => {
            if kind != "anchor" {
                return Err(format!(
                    "{} node {} with an AnchorDesc must have kind 'anchor'",
                    layer,
                    quote(&node_key.text())
                ));
            }
            if node_key.text() != anchor.key() {
                return Err(format!(
                    "{} anchor node {} does not match AnchorDesc {}",
                    layer,
                    quote(&node_key.text()),
                    quote(&anchor.key())
                ));
            }
        }
        None if kind == "anchor" => {
            return Err(format!(
                "{} anchor node {} requires an AnchorDesc",
                layer,
                quote(&node_key.text())
            ));
        }
        None => {}
    }
    Ok(())
}

fn validate_anchor_parity(
    semantic: &SemanticGraphDesc,
    data: &DataCentricDesc,
) -> Result<(), String> {
    for semantic_node in &semantic.nodes {
        let key = semantic_node.node_key.text();
        let execution_node = data
            .nodes
            .iter()
            .find(|node| node.node_key.text() == key)
            .expect("semantic/data node parity was validated before anchor parity");
        if semantic_node.anchor != execution_node.anchor {
            return Err(format!(
                "semantic/data anchor descriptors differ for node {}",
                quote(&key)
            ));
        }
    }
    Ok(())
}

fn validate_access(
    access: &AccessDesc,
    owner: Option<&str>,
    values: &HashSet<String>,
    layer: &str,
) -> Result<(), String> {
    let value_key = access.value.key();
    if !values.contains(&value_key) {
        return Err(format!(
            "{} access references unknown value {}",
            layer,
            quote(&value_key)
        ));
    }
    if !ACCESS_MODES.contains(&access.mode.as_str()) {
        return Err(format!(
            "{} access declares unsupported mode {}",
            layer,
            quote(&access.mode)
        ));
    }
    if access.consistency.is_empty() {
        return Err(format!("{} access has an empty consistency", layer));
    }
    if let Some(port) = &access.port {
        validate_port_ref(port, layer)?;
        if let Some(value_port) = &access.value.port {
            if port != value_port {
                return Err(format!(
                    "{} access port {} does not match value {} port {}",
                    layer,
                    quote(&port.label()),
                    quote(&value_key),
                    quote(&value_port.label())
                ));
            }
        }
    }
    if let Some(owner) = owner {
        let access_node = access.node_key.text();
        if access_node != owner {
            return Err(format!(
                "{} access node {} does not match owner node {}",
                layer,
                quote(&access_node),
                quote(owner)
            ));
        }
    }
    Ok(())
}

fn validate_edges(
    nodes: &HashSet<String>,
    edges: &[(NodeKeyDesc, NodeKeyDesc, String)],
) -> Result<Vec<String>, String> {
    let mut successors: HashMap<String, BTreeSet<String>> = nodes
        .iter()
        .map(|key| (key.clone(), BTreeSet::new()))
        .collect();
    let mut indegree: HashMap<String, usize> = nodes.iter().map(|key| (key.clone(), 0)).collect();
    let mut seen_edges: HashSet<(String, String, String)> = HashSet::new();
    for (source_key, target_key, kind) in edges {
        let source = source_key.text();
        let target = target_key.text();
        if !nodes.contains(&source) {
            return Err(format!(
                "edge references unknown source node {}",
                quote(&source)
            ));
        }
        if !nodes.contains(&target) {
            return Err(format!(
                "edge references unknown target node {}",
                quote(&target)
            ));
        }
        if !EDGE_KINDS.contains(&kind.as_str()) {
            return Err(format!(
                "edge {} -> {} declares unsupported kind {}; expected one of [{}]",
                quote(&source),
                quote(&target),
                quote(kind),
                EDGE_KINDS
                    .iter()
                    .map(|value| quote(value))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !seen_edges.insert((source.clone(), target.clone(), kind.clone())) {
            return Err(format!(
                "duplicate graph edge {} -> {} ({})",
                quote(&source),
                quote(&target),
                quote(kind)
            ));
        }
        if source == target {
            return Err(format!(
                "edge creates a self-cycle at node {}",
                quote(&source)
            ));
        }
        let targets = successors.get_mut(&source).expect("validated source");
        if targets.insert(target.clone()) {
            *indegree.get_mut(&target).expect("validated target") += 1;
        }
    }
    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter_map(|(key, degree)| (*degree == 0).then_some(key.clone()))
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(key) = ready.pop_first() {
        order.push(key.clone());
        for successor in successors.get(&key).expect("all nodes have successors") {
            let degree = indegree.get_mut(successor).expect("validated successor");
            *degree -= 1;
            if *degree == 0 {
                ready.insert(successor.clone());
            }
        }
    }
    if order.len() != nodes.len() {
        return Err("workgraph dependency cycle detected".to_string());
    }
    Ok(order)
}

fn validate_anchor_edges(
    node_kinds: &HashMap<String, String>,
    edges: &[(NodeKeyDesc, NodeKeyDesc, String)],
    layer: &str,
) -> Result<(), String> {
    let declared_anchors: HashSet<String> = node_kinds
        .iter()
        .filter_map(|(key, kind)| (kind == "anchor").then_some(key.clone()))
        .collect();
    let mut attached_anchors = HashSet::new();
    for (source_key, target_key, kind) in edges {
        if kind != "anchor" {
            continue;
        }
        let source = source_key.text();
        let target = target_key.text();
        if node_kinds.get(&source).map(String::as_str) != Some("anchor") {
            return Err(format!(
                "{} anchor edge source {} is not an AnchorNode",
                layer,
                quote(&source)
            ));
        }
        if node_kinds.get(&target).map(String::as_str) == Some("anchor") {
            return Err(format!(
                "{} anchor edge target {} must not be an AnchorNode",
                layer,
                quote(&target)
            ));
        }
        attached_anchors.insert(source);
    }
    if declared_anchors != attached_anchors {
        return Err(format!(
            "{} anchor nodes must each have an Anchor edge to a non-anchor node",
            layer
        ));
    }
    Ok(())
}

fn edge_set(edges: &[(NodeKeyDesc, NodeKeyDesc, String)]) -> HashSet<(String, String, String)> {
    edges
        .iter()
        .map(|(source, target, kind)| (source.text(), target.text(), kind.clone()))
        .collect()
}

fn access_counts<'a>(accesses: impl Iterator<Item = &'a AccessDesc>) -> HashMap<AccessDesc, usize> {
    let mut counts = HashMap::new();
    for access in accesses {
        *counts.entry(access.clone()).or_insert(0) += 1;
    }
    counts
}

fn validate_access_parity(
    semantic: &SemanticGraphDesc,
    data: &DataCentricDesc,
) -> Result<(), String> {
    let semantic_global = access_counts(
        semantic
            .nodes
            .iter()
            .flat_map(|node| node.reads.iter().chain(node.writes.iter())),
    );
    let execution_global = access_counts(data.nodes.iter().flat_map(|node| node.accesses.iter()));
    if semantic_global != execution_global {
        return Err(
            "semantic/execution access sets differ; execution nodes must preserve every semantic access"
                .to_string(),
        );
    }

    let global_accesses = access_counts(data.accesses.iter());
    if semantic_global != global_accesses {
        return Err(
            "semantic/data-global access sets differ; data-centric accesses must preserve every semantic access"
                .to_string(),
        );
    }

    for semantic_node in &semantic.nodes {
        let key = semantic_node.node_key.text();
        let semantic_accesses = access_counts(
            semantic_node
                .reads
                .iter()
                .chain(semantic_node.writes.iter()),
        );
        let execution_node = data
            .nodes
            .iter()
            .find(|node| node.node_key.text() == key)
            .expect("semantic/data node parity was validated before access parity");
        let execution_accesses = access_counts(execution_node.accesses.iter());
        if semantic_accesses != execution_accesses {
            return Err(format!(
                "semantic/execution accesses differ for node {}",
                quote(&key)
            ));
        }
        let global_node_accesses = access_counts(
            data.accesses
                .iter()
                .filter(|access| access.node_key.text() == key),
        );
        if semantic_accesses != global_node_accesses {
            return Err(format!(
                "semantic/data-global accesses differ for node {}",
                quote(&key)
            ));
        }
    }
    Ok(())
}

fn validate_node_projection_parity(
    semantic: &SemanticGraphDesc,
    data: &DataCentricDesc,
) -> Result<(), String> {
    for semantic_node in &semantic.nodes {
        let key = semantic_node.node_key.text();
        let execution_node = data
            .nodes
            .iter()
            .find(|node| node.node_key.text() == key)
            .expect("semantic/data node parity was validated before node projection parity");
        if semantic_node.kind != execution_node.kind
            || semantic_node.executor != execution_node.executor
            || semantic_node.graph_safety != execution_node.graph_safety
        {
            return Err(format!(
                "semantic/execution node descriptors differ for node {}",
                quote(&key)
            ));
        }
    }
    Ok(())
}

fn validate_semantic(
    semantic: &SemanticGraphDesc,
) -> Result<
    (
        HashSet<String>,
        HashSet<String>,
        HashMap<String, AnchorDesc>,
    ),
    String,
> {
    if semantic.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported semantic schema_version {}; expected {}",
            semantic.schema_version, SCHEMA_VERSION
        ));
    }
    let backend = semantic
        .backend
        .as_ref()
        .ok_or_else(|| "semantic graph requires a backend snapshot".to_string())?;
    if !GRAPH_MODES.contains(&semantic.graph_mode.as_str()) {
        return Err(format!(
            "unsupported graph mode {}",
            quote(&semantic.graph_mode)
        ));
    }
    validate_backend(backend, &semantic.graph_mode)?;
    let node_keys = validate_unique(&semantic.nodes, "node", |node| node.node_key.text())?;
    let value_keys = validate_values(&semantic.values, "semantic value")?;
    let mut anchors = HashMap::new();
    for node in &semantic.nodes {
        let key = node.node_key.text();
        validate_node_key(&node.node_key, "node")?;
        validate_node_domain(
            &node.kind,
            &node.graph_safety,
            node.legacy_boundary.as_ref(),
            node.executor.as_ref(),
            "semantic",
        )?;
        let legacy = node.legacy_boundary.is_some();
        validate_timing(&key, &node.timing, legacy)?;
        validate_explicit_anchor(&node.node_key, &node.kind, node.anchor.as_ref(), "semantic")?;
        if let Some(anchor) = &node.anchor {
            validate_anchor(anchor, backend, &semantic.graph_mode, &mut anchors)?;
        }
        for access in node.reads.iter().chain(node.writes.iter()) {
            validate_access(access, Some(&key), &value_keys, "semantic")?;
        }
    }
    let _ = validate_edges(&node_keys, &semantic.edges)?;
    let node_kinds = semantic
        .nodes
        .iter()
        .map(|node| (node.node_key.text(), node.kind.clone()))
        .collect();
    validate_anchor_edges(&node_kinds, &semantic.edges, "semantic")?;
    Ok((node_keys, value_keys, anchors))
}

fn validate_data(
    data: &DataCentricDesc,
    graph_mode: &str,
    backend: &BackendDesc,
    anchors: &mut HashMap<String, AnchorDesc>,
) -> Result<(HashSet<String>, HashSet<String>), String> {
    if data.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported data-centric schema_version {}; expected {}",
            data.schema_version, SCHEMA_VERSION
        ));
    }
    if data.graph_mode != graph_mode || !GRAPH_MODES.contains(&data.graph_mode.as_str()) {
        return Err(format!(
            "data-centric graph mode {} differs from semantic graph",
            quote(&data.graph_mode)
        ));
    }
    let data_backend = data
        .backend
        .as_ref()
        .ok_or_else(|| "data-centric graph requires a backend snapshot".to_string())?;
    validate_backend(data_backend, graph_mode)?;
    validate_backend_parity(backend, data_backend)?;
    let node_keys = validate_unique(&data.nodes, "execution node", |node| node.node_key.text())?;
    let value_keys = validate_values(&data.values, "data-centric value")?;
    let mut residence_keys = HashSet::new();
    let mut lease_keys = HashSet::new();
    for node in &data.nodes {
        let key = node.node_key.text();
        validate_node_key(&node.node_key, "execution node")?;
        validate_node_domain(
            &node.kind,
            &node.graph_safety,
            None,
            node.executor.as_ref(),
            "execution",
        )?;
        if node.semantic_key.as_ref() != Some(&node.node_key) {
            return Err(format!(
                "execution node {} semantic_key must equal its node_key",
                quote(&key)
            ));
        }
        validate_explicit_anchor(
            &node.node_key,
            &node.kind,
            node.anchor.as_ref(),
            "execution",
        )?;
        for access in &node.accesses {
            validate_access(access, Some(&key), &value_keys, "execution")?;
        }
        if !node.move_keys.is_empty() {
            return Err(format!(
                "execution node {} references unsupported Move descriptors",
                quote(&key)
            ));
        }
        if node.device.is_empty()
            || node.rank < 0
            || node.device != backend.device
            || node.rank != backend.rank
        {
            return Err(format!(
                "execution node {} has invalid placement",
                quote(&key)
            ));
        }
        if let Some(anchor) = &node.anchor {
            validate_anchor(anchor, backend, graph_mode, anchors)?;
        }
    }
    for access in &data.accesses {
        validate_access(access, None, &value_keys, "data-centric")?;
        if !node_keys.contains(&access.node_key.text()) {
            return Err(format!(
                "data-centric access references unknown node {}",
                quote(&access.node_key.text())
            ));
        }
    }
    for residence in &data.residences {
        if !value_keys.contains(&residence.value.key()) {
            return Err(format!(
                "residence references unknown value {}",
                quote(&residence.value.key())
            ));
        }
        if residence.device.is_empty()
            || residence.memory_kind.is_empty()
            || residence.rank < 0
            || residence.device != backend.device
            || residence.rank != backend.rank
        {
            return Err(format!(
                "residence for value {} has invalid location",
                quote(&residence.value.key())
            ));
        }
        if residence.valid_version < 1 {
            return Err(format!(
                "residence for value {} has invalid version",
                quote(&residence.value.key())
            ));
        }
        let residence_key = residence.value.key();
        if !residence_keys.insert(residence_key.clone()) {
            return Err(format!("duplicate residence key {}", quote(&residence_key)));
        }
    }
    for lease in &data.leases {
        if !value_keys.contains(&lease.value.key()) {
            return Err(format!(
                "lease references unknown value {}",
                quote(&lease.value.key())
            ));
        }
        if lease.storage.is_empty() || lease.reader_count < 0 {
            return Err(format!(
                "lease for value {} has invalid metadata",
                quote(&lease.value.key())
            ));
        }
        if lease.materialized {
            return Err(format!(
                "lease for value {} is materialized",
                quote(&lease.value.key())
            ));
        }
        let lease_key = lease.value.key();
        if !lease_keys.insert(lease_key.clone()) {
            return Err(format!("duplicate lease key {}", quote(&lease_key)));
        }
    }
    for node in &data.nodes {
        let key = node.node_key.text();
        for residence_key in &node.residence_keys {
            if !residence_keys.contains(residence_key) {
                return Err(format!(
                    "execution node {} references unknown residence {}",
                    quote(&key),
                    quote(residence_key)
                ));
            }
        }
        for lease_key in &node.lease_keys {
            if !lease_keys.contains(lease_key) {
                return Err(format!(
                    "execution node {} references unknown lease {}",
                    quote(&key),
                    quote(lease_key)
                ));
            }
        }
    }
    if !data.moves.is_empty() {
        return Err(format!(
            "WorkGraph Phase 1a does not support Move descriptors; found {}",
            data.moves.len()
        ));
    }
    let _ = validate_edges(&node_keys, &data.edges)?;
    let node_kinds = data
        .nodes
        .iter()
        .map(|node| (node.node_key.text(), node.kind.clone()))
        .collect();
    validate_anchor_edges(&node_kinds, &data.edges, "execution")?;
    Ok((node_keys, value_keys))
}

fn validate_artifact_key_v2(key: &ArtifactKeyV2Desc) -> Result<(), String> {
    if key.namespace.is_empty()
        || key.name.is_empty()
        || key.version < 1
        || key.namespace.contains('/')
        || key.namespace.contains('\\')
        || key.name.contains('/')
        || key.name.contains('\\')
        || key.digest.len() != 64
        || key
            .digest
            .chars()
            .any(|character| !character.is_ascii_hexdigit() || character.is_ascii_uppercase())
    {
        return Err(format!(
            "invalid capture artifact key {}",
            quote(&key.text())
        ));
    }
    Ok(())
}

fn validate_region_key_v2(
    key: &RegionKeyV2Desc,
    expected_kind: &str,
    artifact: &ArtifactKeyV2Desc,
) -> Result<(), String> {
    validate_artifact_key_v2(&key.artifact_key)?;
    if key.kind != expected_kind || &key.artifact_key != artifact || key.ordinal < 0 {
        return Err(format!(
            "capture region key {} does not match its descriptor",
            quote(&key.text())
        ));
    }
    Ok(())
}

fn validate_leaf_operator_v2(operator: &LeafOperatorV2Desc) -> Result<(), String> {
    if operator.ordinal < 0 || operator.name.is_empty() || operator.namespace.is_empty() {
        return Err("capture leaf operator has an invalid identity".to_string());
    }
    if operator.operator_kind == "aten" && operator.namespace != "aten" {
        return Err("ATen capture leaf operator must use the aten namespace".to_string());
    }
    if operator.operator_kind == "custom" && operator.namespace == "aten" {
        return Err("custom capture leaf operator must not use the aten namespace".to_string());
    }
    if !["aten", "custom"].contains(&operator.operator_kind.as_str()) {
        return Err("capture leaf operator has an unsupported typed kind".to_string());
    }
    if operator.effects.is_empty()
        || operator.effects.iter().any(|effect| {
            ![
                "pure",
                "read_state",
                "write_state",
                "rng",
                "host_io",
                "opaque",
            ]
            .contains(&effect.as_str())
        })
    {
        return Err("capture leaf operator has invalid effect facts".to_string());
    }
    if operator.capabilities.is_empty()
        || operator.capabilities.iter().any(|capability| {
            !["eager_only", "capture_safe", "replay_safe"].contains(&capability.as_str())
        })
    {
        return Err("capture leaf operator has invalid capability facts".to_string());
    }
    if operator
        .module_path
        .iter()
        .any(|segment| segment.is_empty() || segment.contains('/') || segment.contains('\\'))
    {
        return Err("capture leaf operator has an invalid module path".to_string());
    }
    if operator.inputs.iter().any(|ordinal| *ordinal < 0)
        || operator.outputs.iter().any(|value| value.is_empty())
    {
        return Err("capture leaf operator has invalid SSA facts".to_string());
    }
    let attribute_kinds = [
        "none",
        "bool",
        "int",
        "float",
        "string",
        "tuple",
        "list",
        "mapping",
        "slice",
        "ellipsis",
        "dtype",
        "device",
        "layout",
        "memory_format",
    ];
    let mut attribute_paths = HashSet::new();
    for attribute in &operator.attributes {
        if attribute.path.is_empty()
            || attribute.path.iter().any(|segment| segment.is_empty())
            || !attribute_kinds.contains(&attribute.kind.as_str())
            || !attribute_paths.insert(attribute.path.clone())
        {
            return Err("capture leaf operator has invalid typed attributes".to_string());
        }
        let _ = &attribute.value;
    }
    let _ = &operator.overload;
    Ok(())
}

fn validate_leaf_region_v2(
    leaf: &LeafRegionV2Desc,
    artifact: &ArtifactKeyV2Desc,
) -> Result<HashSet<i64>, String> {
    if &leaf.artifact_key != artifact {
        return Err("capture leaf belongs to another artifact".to_string());
    }
    validate_region_key_v2(&leaf.key, &leaf.kind, artifact)?;
    let ordinals: HashSet<i64> = leaf.operator_ordinals.iter().copied().collect();
    if ordinals.len() != leaf.operator_ordinals.len()
        || leaf.operator_ordinals.iter().any(|ordinal| *ordinal < 0)
    {
        return Err("capture leaf has duplicate or invalid operator ordinals".to_string());
    }
    match leaf.kind.as_str() {
        "opaque" => {
            if leaf.leaf_artifact.is_some() || !ordinals.is_empty() {
                return Err("opaque capture leaf retained a partial export graph".to_string());
            }
        }
        "leaf" => {
            let leaf_artifact = leaf
                .leaf_artifact
                .as_ref()
                .ok_or_else(|| "exported capture leaf is missing its artifact".to_string())?;
            if &leaf_artifact.artifact_key != artifact {
                return Err("capture leaf artifact identity differs from its region".to_string());
            }
            let known: HashSet<i64> = leaf_artifact
                .operators
                .iter()
                .map(|operator| operator.ordinal)
                .collect();
            if known.len() != leaf_artifact.operators.len() || !ordinals.is_subset(&known) {
                return Err("capture leaf has invalid operator coverage".to_string());
            }
            if ordinals.is_empty() && !known.is_empty() {
                return Err("capture leaf omitted operator coverage".to_string());
            }
            for operator in &leaf_artifact.operators {
                validate_leaf_operator_v2(operator)?;
            }
            if leaf_artifact
                .input_values
                .iter()
                .chain(leaf_artifact.output_values.iter())
                .any(|value| value.is_empty())
            {
                return Err("capture leaf has an empty input/output value identity".to_string());
            }
            if leaf_artifact.terminal_ordinal < 0 {
                return Err("capture leaf has an invalid terminal ordinal".to_string());
            }
        }
        _ => return Err("capture leaf has an unsupported region kind".to_string()),
    }
    Ok(ordinals)
}

fn partition_boundary_v2(
    previous: Option<&LeafOperatorV2Desc>,
    current: Option<&LeafOperatorV2Desc>,
) -> &'static str {
    let Some(current) = current else {
        return "root";
    };
    if current.operator_kind == "custom" {
        return "custom_op";
    }
    let Some(previous) = previous else {
        return "root";
    };
    if previous.module_path != current.module_path {
        "module"
    } else if previous.effects != current.effects {
        "effect"
    } else if previous.capabilities != current.capabilities {
        "capability"
    } else {
        "root"
    }
}

fn valid_capture_effect(effect: &str) -> bool {
    [
        "pure",
        "read_state",
        "write_state",
        "rng",
        "host_io",
        "opaque",
    ]
    .contains(&effect)
}

fn validate_root_specialization_v2(root: &RootFlowV2Desc) -> Result<(), String> {
    for expression in &root.shape_guard.expressions {
        if expression.value_path.iter().any(|part| *part < 0)
            || expression.dimension < 0
            || expression.lower_bound < 0
            || expression.upper_bound < expression.lower_bound
            || !["static", "range"].contains(&expression.kind.as_str())
            || expression.kind == "static" && expression.lower_bound != expression.upper_bound
        {
            return Err("capture root has an invalid shape guard".to_string());
        }
    }
    if root.effects.is_empty()
        || root
            .effects
            .iter()
            .any(|effect| !valid_capture_effect(effect))
    {
        return Err("capture root has invalid effect facts".to_string());
    }
    let mut bindings = HashSet::new();
    for binding in root.inputs.iter().chain(root.outputs.iter()) {
        validate_port_ref(&binding.port, "capture root")?;
        if !["input", "output"].contains(&binding.direction.as_str())
            || binding.value_path.iter().any(|part| *part < 0)
            || !bindings.insert(binding.clone())
        {
            return Err("capture root has an invalid or duplicate binding".to_string());
        }
    }
    if root
        .inputs
        .iter()
        .any(|binding| binding.direction != "input")
        || root
            .outputs
            .iter()
            .any(|binding| binding.direction != "output")
    {
        return Err("capture root binding direction differs from its collection".to_string());
    }
    let mut value_keys = HashSet::new();
    let mut value_ordinals = HashSet::new();
    for value in &root.values {
        if value.value_id.is_empty()
            || value.version < 1
            || value.shape.is_empty()
            || value.stride.is_empty()
            || value.dtype.namespace.is_empty()
            || value.dtype.name.is_empty()
            || !["cpu", "cuda"].contains(&value.device.kind.as_str())
            || value.device.index.is_some_and(|index| index < 0)
            || value.layout.is_empty()
            || value.alias_set.is_empty()
            || !valid_capture_effect(&value.effect)
            || value.use_def.producer < 0
            || value.use_def.inputs.iter().any(|ordinal| *ordinal < 0)
            || value.use_def.users.iter().any(|ordinal| *ordinal < 0)
            || !value_keys.insert(value.key())
            || !value_ordinals.insert(value.use_def.producer)
        {
            return Err("capture root has invalid inferred value facts".to_string());
        }
    }
    for diagnostic in &root.diagnostics {
        if diagnostic.code.is_empty()
            || diagnostic.message.is_empty()
            || diagnostic.source_span.file.is_empty()
        {
            return Err("capture root has an invalid typed diagnostic".to_string());
        }
        let _ = &diagnostic.detail;
    }
    if root.status == "captured" {
        if root.values.is_empty() || !root.diagnostics.is_empty() {
            return Err("captured root requires inferred values and no diagnostics".to_string());
        }
    } else if root.status == "opaque_barrier"
        && (!root.values.is_empty()
            || root.diagnostics.is_empty()
            || !root.effects.iter().any(|effect| effect == "opaque"))
    {
        return Err("opaque root retained partial values or lost diagnostics/effect".to_string());
    }
    Ok(())
}

fn validate_capture_root_v2(root: &RootFlowV2Desc) -> Result<Vec<String>, String> {
    validate_artifact_key_v2(&root.artifact_key)?;
    validate_region_key_v2(&root.key, "root", &root.artifact_key)?;
    if root.kind != "root" || root.children.is_empty() {
        return Err("capture root must contain task regions".to_string());
    }
    if root.source_span.file.is_empty()
        || root.source_span.start_line < 1
        || root.source_span.end_line < root.source_span.start_line
        || root.source_span.start_column < 0
        || root.source_span.end_column < 0
    {
        return Err("capture root has an invalid source span".to_string());
    }
    validate_root_specialization_v2(root)?;
    let mut task_positions = HashMap::new();
    for (position, task) in root.children.iter().enumerate() {
        validate_region_key_v2(&task.key, "task", &root.artifact_key)?;
        if task.kind != "task" || task.children.len() != 1 {
            return Err("capture task must contain exactly one leaf".to_string());
        }
        if task_positions.insert(task.key.clone(), position).is_some() {
            return Err("capture root contains duplicate task keys".to_string());
        }
    }
    let mut covered = HashSet::new();
    let mut expected: Option<HashSet<i64>> = None;
    let mut previous_operator: Option<LeafOperatorV2Desc> = None;
    let mut order = Vec::with_capacity(root.children.len());
    for (position, task) in root.children.iter().enumerate() {
        validate_node_key(&task.node_key, "capture task")?;
        validate_executor(Some(&task.executor), "capture task")?;
        if task.node_key.namespace != "capture_task" {
            return Err("capture task must use the capture_task node namespace".to_string());
        }
        if ![
            "root",
            "module",
            "custom_op",
            "effect",
            "capability",
            "opaque",
        ]
        .contains(&task.boundary.as_str())
        {
            return Err("capture task has an unsupported partition boundary".to_string());
        }
        if task.effects.is_empty() || task.capabilities.is_empty() {
            return Err("capture task requires effect and capability facts".to_string());
        }
        if task.effects.iter().any(|effect| {
            ![
                "pure",
                "read_state",
                "write_state",
                "rng",
                "host_io",
                "opaque",
            ]
            .contains(&effect.as_str())
        }) || task.capabilities.iter().any(|capability| {
            !["eager_only", "capture_safe", "replay_safe"].contains(&capability.as_str())
        }) {
            return Err("capture task has invalid effect or capability facts".to_string());
        }
        if task
            .module_path
            .iter()
            .any(|segment| segment.is_empty() || segment.contains('/') || segment.contains('\\'))
            || task.input_values.iter().any(|value| value.is_empty())
            || task.output_values.iter().any(|value| value.is_empty())
        {
            return Err("capture task has invalid structural facts".to_string());
        }
        for dependency in &task.dependencies {
            let dependency_position = task_positions
                .get(dependency)
                .ok_or_else(|| "capture task dependency references another root".to_string())?;
            if *dependency_position >= position {
                return Err("capture task dependency violates topological order".to_string());
            }
        }
        let task_ordinals: HashSet<i64> = task.operator_ordinals.iter().copied().collect();
        if task_ordinals.len() != task.operator_ordinals.len() {
            return Err("capture task has duplicate operator coverage".to_string());
        }
        let leaf_ordinals = validate_leaf_region_v2(&task.children[0], &root.artifact_key)?;
        if leaf_ordinals != task_ordinals || !covered.is_disjoint(&task_ordinals) {
            return Err("capture task/leaf operator coverage differs or overlaps".to_string());
        }
        covered.extend(task_ordinals);
        if let Some(leaf_artifact) = &task.children[0].leaf_artifact {
            let selected: Vec<&LeafOperatorV2Desc> = task
                .operator_ordinals
                .iter()
                .map(|ordinal| {
                    leaf_artifact
                        .operators
                        .iter()
                        .find(|operator| operator.ordinal == *ordinal)
                        .expect("validated leaf coverage")
                })
                .collect();
            if selected.iter().any(|operator| {
                operator.module_path != task.module_path
                    || operator.effects != task.effects
                    || operator.capabilities != task.capabilities
            }) {
                return Err("capture task facts differ from its leaf operators".to_string());
            }
            if task.boundary
                != partition_boundary_v2(previous_operator.as_ref(), selected.first().copied())
            {
                return Err("capture task boundary differs from its leaf facts".to_string());
            }
            if let Some(operator) = selected.last() {
                previous_operator = Some((*operator).clone());
            }
            let known = leaf_artifact
                .operators
                .iter()
                .map(|operator| operator.ordinal)
                .collect();
            if let Some(previous) = &expected {
                if previous != &known {
                    return Err("capture tasks disagree on the leaf artifact".to_string());
                }
            } else {
                expected = Some(known);
            }
        }
        order.push(task.node_key.text());
    }
    match root.status.as_str() {
        "captured" => {
            if covered != expected.unwrap_or_default() {
                return Err(
                    "capture root does not cover every leaf operator exactly once".to_string(),
                );
            }
            let leaf_artifact = root.children[0].children[0]
                .leaf_artifact
                .as_ref()
                .expect("captured root has a validated leaf artifact");
            let value_keys: HashSet<String> =
                root.values.iter().map(CapturedValueV2Desc::key).collect();
            let value_ordinals: HashSet<i64> = root
                .values
                .iter()
                .map(|value| value.use_def.producer)
                .collect();
            let operator_ordinals: HashSet<i64> = leaf_artifact
                .operators
                .iter()
                .map(|operator| operator.ordinal)
                .collect();
            let expected_effects: Vec<String> = if leaf_artifact.operators.is_empty() {
                vec!["pure".to_string()]
            } else {
                let mut effects = Vec::new();
                for operator in &leaf_artifact.operators {
                    for effect in &operator.effects {
                        if !effects.contains(effect) {
                            effects.push(effect.clone());
                        }
                    }
                }
                effects
            };
            if root.effects != expected_effects {
                return Err("capture root effects differ from its leaf operators".to_string());
            }
            if !leaf_artifact
                .input_values
                .iter()
                .chain(leaf_artifact.output_values.iter())
                .all(|value| value_keys.contains(value))
            {
                return Err("capture leaf input/output references an unknown value".to_string());
            }
            let leaf_outputs: HashSet<String> =
                leaf_artifact.output_values.iter().cloned().collect();
            for value in &root.values {
                if !value
                    .use_def
                    .inputs
                    .iter()
                    .all(|ordinal| value_ordinals.contains(ordinal))
                {
                    return Err("capture value references an unknown input".to_string());
                }
                let unknown_users: HashSet<i64> = value
                    .use_def
                    .users
                    .iter()
                    .filter(|ordinal| {
                        !value_ordinals.contains(ordinal) && !operator_ordinals.contains(ordinal)
                    })
                    .copied()
                    .collect();
                if !unknown_users.is_empty()
                    && (unknown_users != HashSet::from([leaf_artifact.terminal_ordinal])
                        || !leaf_outputs.contains(&value.key()))
                {
                    return Err("capture value references an unknown user".to_string());
                }
            }
            for operator in &leaf_artifact.operators {
                if !operator
                    .inputs
                    .iter()
                    .all(|ordinal| value_ordinals.contains(ordinal))
                    || !operator
                        .outputs
                        .iter()
                        .all(|value| value_keys.contains(value))
                {
                    return Err("capture operator has dangling SSA facts".to_string());
                }
            }
        }
        "opaque_barrier" => {
            if root.children.len() != 1
                || root.children[0].boundary != "opaque"
                || root.children[0].children[0].kind != "opaque"
            {
                return Err("opaque capture root must contain one opaque task".to_string());
            }
        }
        _ => return Err("capture root has an unsupported status".to_string()),
    }
    Ok(order)
}

fn validate_capture_hierarchy_v2(
    hierarchy: &CaptureHierarchyV2Desc,
    semantic: &SemanticGraphDesc,
) -> Result<Vec<String>, String> {
    if hierarchy.schema_version != 2 {
        return Err("capture hierarchy schema version must be 2".to_string());
    }
    if hierarchy.roots.is_empty() {
        return Err("capture hierarchy requires at least one root".to_string());
    }
    let mut root_artifacts = HashSet::new();
    let mut root_bindings: HashMap<ArtifactKeyV2Desc, HashSet<PortBindingV2Desc>> = HashMap::new();
    let mut order = Vec::new();
    for root in &hierarchy.roots {
        if !root_artifacts.insert(root.artifact_key.clone()) {
            return Err("capture hierarchy contains duplicate root artifacts".to_string());
        }
        root_bindings.insert(
            root.artifact_key.clone(),
            root.inputs
                .iter()
                .chain(root.outputs.iter())
                .cloned()
                .collect(),
        );
        order.extend(validate_capture_root_v2(root)?);
    }
    let semantic_nodes: HashSet<NodeKeyDesc> = semantic
        .nodes
        .iter()
        .map(|node| node.node_key.clone())
        .collect();
    let semantic_anchors: HashSet<NodeKeyDesc> = semantic
        .nodes
        .iter()
        .filter(|node| node.kind == "anchor")
        .map(|node| node.node_key.clone())
        .collect();
    let semantic_ports: HashSet<PortRefDesc> = semantic
        .values
        .iter()
        .filter_map(|value| value.port.clone())
        .collect();
    let mut attachment_artifacts = HashSet::new();
    for attachment in &hierarchy.attachments {
        validate_artifact_key_v2(&attachment.artifact_key)?;
        validate_node_key(&attachment.node_key, "capture attachment")?;
        if !root_artifacts.contains(&attachment.artifact_key)
            || !attachment_artifacts.insert(attachment.artifact_key.clone())
        {
            return Err("capture attachment has an unknown or duplicate artifact".to_string());
        }
        if !semantic_nodes.contains(&attachment.node_key) {
            return Err("capture attachment references an unknown semantic node".to_string());
        }
        let target_node = semantic
            .nodes
            .iter()
            .find(|node| node.node_key == attachment.node_key)
            .expect("attachment target membership validated");
        if attachment.node_key.namespace != "consumer_flow" || target_node.kind != "compute" {
            return Err("capture attachment must target a consumer flow node".to_string());
        }
        let mut bindings = HashSet::new();
        for binding in &attachment.bindings {
            validate_port_ref(&binding.port, "capture attachment")?;
            if !["input", "output"].contains(&binding.direction.as_str())
                || binding.value_path.iter().any(|part| *part < 0)
                || !bindings.insert(binding.clone())
            {
                return Err("capture attachment has an invalid or duplicate binding".to_string());
            }
            match &binding.port {
                PortRefDesc::Id(port) => {
                    if !semantic_ports.contains(&PortRefDesc::Id(port.clone())) {
                        return Err("capture attachment references an unknown PortId".to_string());
                    }
                }
                PortRefDesc::Address(address) => {
                    let accesses = if binding.direction == "input" {
                        &target_node.reads
                    } else {
                        &target_node.writes
                    };
                    if !accesses.iter().any(|access| {
                        access.port.as_ref() == Some(&PortRefDesc::Address(address.clone()))
                    }) {
                        return Err(
                            "capture attachment address is not declared by its target".to_string()
                        );
                    }
                }
            }
        }
        if root_bindings[&attachment.artifact_key] != bindings {
            return Err("capture attachment does not bind every captured port exactly".to_string());
        }
        if let Some(anchor) = &attachment.anchor_node_key {
            if !semantic_anchors.contains(anchor)
                || !semantic.edges.iter().any(|(source, target, kind)| {
                    source == anchor && target == &attachment.node_key && kind == "anchor"
                })
            {
                return Err("capture attachment references a mismatched anchor".to_string());
            }
        }
    }
    if root_artifacts != attachment_artifacts {
        return Err("every capture root requires one explicit attachment".to_string());
    }
    if !hierarchy.data_centric.moves.is_empty() {
        return Err("Phase 1 capture hierarchy must not contain Move".to_string());
    }
    let mut lease_values = HashSet::new();
    let task_positions: HashMap<String, usize> = order
        .iter()
        .enumerate()
        .map(|(position, key)| (key.clone(), position))
        .collect();
    for lease in &hierarchy.data_centric.leases {
        validate_node_key(&lease.live_start, "capture lease start")?;
        validate_node_key(&lease.live_end, "capture lease end")?;
        if lease.value_id.is_empty()
            || lease.reader_count < 0
            || lease.materialized
            || !lease_values.insert(lease.value_id.clone())
            || !task_positions.contains_key(&lease.live_start.text())
            || !task_positions.contains_key(&lease.live_end.text())
            || task_positions[&lease.live_start.text()] > task_positions[&lease.live_end.text()]
        {
            return Err("capture hierarchy has an invalid logical lease".to_string());
        }
    }
    if semantic
        .nodes
        .iter()
        .any(|node| node.node_key.namespace == "capture_operator")
    {
        return Err("ATen leaf operators must not become semantic schedule nodes".to_string());
    }
    Ok(order)
}

/// Validate semantic and data-centric WorkGraph descriptors and return the
/// deterministic topological order of semantic nodes.
#[pyfunction]
#[pyo3(signature = (semantic, data))]
fn validate_workgraph(
    semantic: SemanticGraphDesc,
    data: DataCentricDesc,
) -> PyResult<WorkGraphValidation> {
    let (semantic_nodes, _semantic_values, mut _anchors) =
        validate_semantic(&semantic).map_err(validation_error)?;
    let (data_nodes, data_values) = validate_data(
        &data,
        &semantic.graph_mode,
        semantic.backend.as_ref().expect("validated backend"),
        &mut _anchors,
    )
    .map_err(validation_error)?;
    if semantic_nodes != data_nodes {
        return Err(validation_error(
            "semantic/data node key sets differ; execution IR must preserve node identity",
        ));
    }
    if _semantic_values != data_values {
        return Err(validation_error(
            "semantic/data value key sets differ; execution IR must preserve value identity",
        ));
    }
    let semantic_value_descriptors: HashSet<ValueDesc> = semantic.values.iter().cloned().collect();
    let data_value_descriptors: HashSet<ValueDesc> = data.values.iter().cloned().collect();
    if semantic_value_descriptors != data_value_descriptors {
        return Err(validation_error(
            "semantic/data value descriptor sets differ; execution IR must preserve value metadata",
        ));
    }
    if edge_set(&semantic.edges) != edge_set(&data.edges) {
        return Err(validation_error(
            "semantic/data edge sets differ; execution IR must preserve graph dependencies",
        ));
    }
    validate_anchor_parity(&semantic, &data).map_err(validation_error)?;
    validate_node_projection_parity(&semantic, &data).map_err(validation_error)?;
    validate_access_parity(&semantic, &data).map_err(validation_error)?;
    let topological_order =
        validate_edges(&semantic_nodes, &semantic.edges).map_err(validation_error)?;
    Ok(WorkGraphValidation { topological_order })
}

#[pyfunction]
#[pyo3(signature = (semantic, data))]
fn validate_workgraph_ir(
    semantic: SemanticGraphDesc,
    data: DataCentricDesc,
) -> PyResult<WorkGraphValidation> {
    validate_workgraph(semantic, data)
}

/// Validate the frozen Phase 1 v2 capture hierarchy against the existing
/// semantic schedule projection. The sidecar remains metadata-only.
#[pyfunction]
#[pyo3(signature = (hierarchy, semantic))]
fn validate_capture_hierarchy(
    hierarchy: CaptureHierarchyV2Desc,
    semantic: SemanticGraphDesc,
) -> PyResult<WorkGraphValidation> {
    let topological_order =
        validate_capture_hierarchy_v2(&hierarchy, &semantic).map_err(validation_error)?;
    Ok(WorkGraphValidation { topological_order })
}

fn contains_runtime_text(value: &str) -> bool {
    value.to_ascii_lowercase().contains("0x") || value.contains('<') || value.contains('>')
}

fn validate_physical_key(key: &PhysicalKeyDesc, layer: &str) -> Result<(), String> {
    if key.namespace.is_empty()
        || key.name.is_empty()
        || key.ordinal < 0
        || key.namespace.contains('/')
        || key.namespace.contains('\\')
        || key.name.contains('/')
        || key.name.contains('\\')
        || contains_runtime_text(&key.namespace)
        || contains_runtime_text(&key.name)
    {
        return Err(format!("{} has an invalid stable key", layer));
    }
    Ok(())
}

fn validate_physical_plan(plan: &PhysicalFlowPlanDesc) -> Result<Vec<String>, String> {
    if plan.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported PhysicalFlowPlan schema version {}",
            plan.schema_version
        ));
    }
    if plan.plan_epoch < 1 {
        return Err("PhysicalFlowPlan plan_epoch must be >= 1".to_string());
    }

    let mut bucket_keys = HashSet::new();
    for bucket in &plan.buffer_buckets {
        validate_physical_key(&bucket.key, "physical buffer bucket")?;
        if !bucket_keys.insert(bucket.key.text()) {
            return Err("PhysicalFlowPlan contains duplicate buffer bucket keys".to_string());
        }
        if bucket.shape.is_empty()
            || bucket.shape.iter().any(|dimension| *dimension < 1)
            || bucket.slots < 1
            || bucket.dtype.namespace.is_empty()
            || bucket.dtype.name.is_empty()
            || contains_runtime_text(&bucket.dtype.namespace)
            || contains_runtime_text(&bucket.dtype.name)
            || !["borrowed", "managed"].contains(&bucket.ownership.as_str())
            || !["cpu", "cuda"].contains(&bucket.device.kind.as_str())
            || bucket.device.index.is_some_and(|index| index < 0)
            || bucket.device.kind == "cpu" && bucket.device.index.is_some()
        {
            return Err(format!(
                "physical buffer bucket {} has invalid typed facts",
                quote(&bucket.key.text())
            ));
        }
    }

    let mut node_keys = HashSet::new();
    let mut positions = HashMap::new();
    for (position, node) in plan.nodes.iter().enumerate() {
        validate_node_key(&node.node_key, "physical node")?;
        if contains_runtime_text(&node.node_key.namespace)
            || contains_runtime_text(&node.node_key.owner)
        {
            return Err("physical node key contains runtime identity text".to_string());
        }
        validate_executor(Some(&node.executor), "physical")?;
        if contains_runtime_text(&node.executor.namespace)
            || contains_runtime_text(&node.executor.name)
            || contains_runtime_text(&node.executor.version)
        {
            return Err("physical node executor contains runtime identity text".to_string());
        }
        let node_text = node.node_key.text();
        if !node_keys.insert(node_text.clone()) {
            return Err("PhysicalFlowPlan contains duplicate node keys".to_string());
        }
        positions.insert(node_text, position);
        if !["default", "side_forward", "side_train"].contains(&node.stream.as_str()) {
            return Err("physical node declares an unsupported stream class".to_string());
        }
        if !["disabled", "prefer", "require"].contains(&node.capture_policy.as_str()) {
            return Err("physical node declares an unsupported capture policy".to_string());
        }
        if !["eager", "reject"].contains(&node.fallback_policy.as_str()) {
            return Err("physical node declares an unsupported fallback policy".to_string());
        }
        if node.capture_policy == "require" && node.fallback_policy == "eager" {
            return Err("required physical capture cannot use eager fallback".to_string());
        }
        if node.capture_policy != "disabled" && node.artifact_key.is_none() {
            return Err("enabled physical capture requires an ArtifactKey".to_string());
        }
        if let Some(artifact) = &node.artifact_key {
            validate_artifact_key_v2(artifact)?;
        }
        if let Some(bucket) = &node.buffer_bucket {
            validate_physical_key(bucket, "physical node buffer bucket")?;
            if !bucket_keys.contains(&bucket.text()) {
                return Err("physical node references an unknown buffer bucket".to_string());
            }
        }
    }

    let mut fence_keys = HashSet::new();
    for node in &plan.nodes {
        for fence in &node.fences {
            validate_physical_key(&fence.fence, "physical fence")?;
            validate_node_key(&fence.producer, "physical fence producer")?;
            validate_node_key(&fence.consumer, "physical fence consumer")?;
            if !fence_keys.insert(fence.fence.text()) {
                return Err("PhysicalFlowPlan contains duplicate fence keys".to_string());
            }
            if fence.consumer != node.node_key {
                return Err("physical fence must target its owning node".to_string());
            }
            if fence.producer == fence.consumer {
                return Err("physical fence cannot depend on its own node".to_string());
            }
            if !["ready", "completion", "retire"].contains(&fence.kind.as_str()) {
                return Err("physical fence declares an unsupported kind".to_string());
            }
            let producer_position = positions
                .get(&fence.producer.text())
                .ok_or_else(|| "physical fence references an unknown producer".to_string())?;
            let consumer_position = positions
                .get(&fence.consumer.text())
                .ok_or_else(|| "physical fence references an unknown consumer".to_string())?;
            if producer_position >= consumer_position {
                return Err(
                    "physical nodes must be in dependency order and fences must be acyclic"
                        .to_string(),
                );
            }
        }
    }

    Ok(plan.nodes.iter().map(|node| node.node_key.text()).collect())
}

fn validate_v3_placement(placement: &PlacementV3Desc, layer: &str) -> Result<(), String> {
    if contains_runtime_text(&placement.kind) {
        return Err(format!("{} contains runtime identity text", layer));
    }
    match (placement.kind.as_str(), placement.index) {
        ("cpu", None) => Ok(()),
        ("cuda", Some(index)) if index >= 0 => Ok(()),
        _ => Err(format!("{} has an invalid typed placement", layer)),
    }
}

fn validate_v3_node_key(node_key: &NodeKeyDesc, layer: &str) -> Result<(), String> {
    validate_node_key(node_key, layer)?;
    if contains_runtime_text(&node_key.namespace) || contains_runtime_text(&node_key.owner) {
        return Err(format!("{} key contains runtime identity text", layer));
    }
    Ok(())
}

fn validate_v3_executor(executor: &ExecutorKeyDesc, layer: &str) -> Result<(), String> {
    validate_executor(Some(executor), layer)?;
    let components = [&executor.namespace, &executor.name, &executor.version];
    if components.iter().any(|component| {
        component.contains(':') || component.contains('@') || contains_runtime_text(component)
    }) {
        return Err(format!(
            "{} has an ambiguous or runtime executor key",
            layer
        ));
    }
    Ok(())
}

fn validate_v3_value(value: &ValueKeyV3Desc, layer: &str) -> Result<(), String> {
    if value.name.is_empty()
        || value.version < 1
        || contains_runtime_text(&value.name)
        || value.name.contains('/')
        || value.name.contains('\\')
    {
        return Err(format!("{} has an invalid stable value key", layer));
    }
    Ok(())
}

fn validate_v3_topology(topology: &DeviceTopologyV3Desc) -> Result<HashSet<i64>, String> {
    let mut devices = HashSet::new();
    for device in &topology.cuda_devices {
        if *device < 0 || !devices.insert(*device) {
            return Err("PlacementPlan topology has an invalid CUDA inventory".to_string());
        }
    }
    for (source, destination) in &topology.peer_access {
        if source == destination || !devices.contains(source) || !devices.contains(destination) {
            return Err("PlacementPlan topology has an invalid peer-access edge".to_string());
        }
    }
    Ok(devices)
}

fn topology_contains_v3(placement: &PlacementV3Desc, cuda_devices: &HashSet<i64>) -> bool {
    match (placement.kind.as_str(), placement.index) {
        ("cpu", None) => true,
        ("cuda", Some(index)) => cuda_devices.contains(&index),
        _ => false,
    }
}

fn validate_v3_residency(
    residency: &ResidencyV3Desc,
    layer: &str,
    cuda_devices: &HashSet<i64>,
) -> Result<(), String> {
    validate_v3_value(&residency.value, layer)?;
    validate_v3_placement(&residency.placement, layer)?;
    if !topology_contains_v3(&residency.placement, cuda_devices) {
        return Err(format!("{} references a placement outside topology", layer));
    }
    let valid_memory = match residency.placement.kind.as_str() {
        "cpu" => ["host", "pinned_host"].contains(&residency.memory.as_str()),
        "cuda" => residency.memory == "device",
        _ => false,
    };
    if !valid_memory || contains_runtime_text(&residency.memory) {
        return Err(format!("{} has an invalid typed memory kind", layer));
    }
    Ok(())
}

fn validate_placement_plan_v3(plan: &PlacementPlanV3Desc) -> Result<Vec<String>, String> {
    if plan.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported PlacementPlan schema version {}",
            plan.schema_version
        ));
    }
    if plan.plan_epoch < 1 {
        return Err("PlacementPlan plan_epoch must be >= 1".to_string());
    }
    let topology_devices = validate_v3_topology(&plan.topology)?;

    let mut selections = HashMap::new();
    let mut logical_oracles: HashMap<
        String,
        (
            HashSet<PlacementV3Desc>,
            HashSet<Option<SemanticOracleKeyV3Desc>>,
        ),
    > = HashMap::new();
    for selection in &plan.selections {
        validate_v3_node_key(&selection.node_key, "placement selection")?;
        validate_v3_executor(&selection.logical_executor, "placement logical")?;
        validate_v3_executor(&selection.key, "placement implementation")?;
        validate_v3_placement(&selection.placement, "placement selection")?;
        if !topology_contains_v3(&selection.placement, &topology_devices) {
            return Err("placement selection is outside topology".to_string());
        }
        if selection.duration.microseconds < 1 || selection.duration.samples < 1 {
            return Err("placement selection has an invalid duration profile".to_string());
        }
        if let Some(oracle) = &selection.oracle_key {
            if oracle.namespace.is_empty()
                || oracle.name.is_empty()
                || oracle.version.is_empty()
                || contains_runtime_text(&oracle.namespace)
                || contains_runtime_text(&oracle.name)
                || contains_runtime_text(&oracle.version)
            {
                return Err("placement selection has an invalid semantic oracle key".to_string());
            }
        }
        let logical = format!(
            "{}:{}@{}",
            selection.logical_executor.namespace,
            selection.logical_executor.name,
            selection.logical_executor.version
        );
        let (placements, oracles) = logical_oracles
            .entry(logical)
            .or_insert_with(|| (HashSet::new(), HashSet::new()));
        placements.insert(selection.placement.clone());
        oracles.insert(selection.oracle_key.clone());
        let node = selection.node_key.text();
        if selections.insert(node, selection).is_some() {
            return Err("PlacementPlan contains duplicate selections".to_string());
        }
    }
    if logical_oracles.values().any(|(placements, oracles)| {
        placements.len() > 1 && (oracles.len() != 1 || oracles.contains(&None))
    }) {
        return Err("cross-placement selections require one semantic oracle key".to_string());
    }

    let mut schedule_positions = HashMap::new();
    for (position, item) in plan.schedule.iter().enumerate() {
        validate_v3_node_key(&item.node_key, "placed node")?;
        validate_v3_executor(&item.implementation, "placed implementation")?;
        validate_v3_placement(&item.placement, "placed node")?;
        if !topology_contains_v3(&item.placement, &topology_devices) {
            return Err("placed node is outside topology".to_string());
        }
        if item.start_at < 0 || item.end_at <= item.start_at {
            return Err("placed node has an invalid schedule interval".to_string());
        }
        let node = item.node_key.text();
        if schedule_positions.insert(node.clone(), position).is_some() {
            return Err("PlacementPlan contains duplicate scheduled nodes".to_string());
        }
        let selection = selections
            .get(&node)
            .ok_or_else(|| "placed node has no implementation selection".to_string())?;
        if selection.key != item.implementation || selection.placement != item.placement {
            return Err("placed node differs from its implementation selection".to_string());
        }
        if item.end_at - item.start_at != selection.duration.microseconds {
            return Err("placed node duration differs from its selected profile".to_string());
        }
    }
    if selections.len() != schedule_positions.len() {
        return Err("PlacementPlan selections and schedule must cover the same nodes".to_string());
    }
    for (position, item) in plan.schedule.iter().enumerate() {
        let mut dependencies = HashSet::new();
        for dependency in &item.depends_on {
            validate_v3_node_key(dependency, "placed node dependency")?;
            let text = dependency.text();
            if !dependencies.insert(text.clone()) {
                return Err("placed node dependencies must be unique".to_string());
            }
            let dependency_position = schedule_positions
                .get(&text)
                .ok_or_else(|| "placed node references an unknown dependency".to_string())?;
            if *dependency_position >= position
                || plan.schedule[*dependency_position].end_at > item.start_at
            {
                return Err("placed node dependencies must be ordered before launch".to_string());
            }
        }
        if plan.schedule[..position].iter().any(|other| {
            other.placement == item.placement
                && other.start_at < item.end_at
                && other.end_at > item.start_at
        }) {
            return Err(
                "PlacementPlan compute intervals must not overlap on one placement".to_string(),
            );
        }
    }

    let mut flow_keys = HashSet::new();
    let mut flow_producers: HashMap<ValueKeyV3Desc, NodeKeyDesc> = HashMap::new();
    let mut expected_routes: HashSet<(NodeKeyDesc, NodeKeyDesc, ValueKeyV3Desc)> = HashSet::new();
    let mut required_flow_residencies = HashSet::new();
    for flow in &plan.value_flows {
        validate_v3_value(&flow.value, "placement value flow")?;
        validate_v3_node_key(&flow.producer, "placement value-flow producer")?;
        validate_v3_node_key(&flow.consumer, "placement value-flow consumer")?;
        if flow.byte_count < 1
            || !flow_keys.insert((
                flow.value.clone(),
                flow.producer.clone(),
                flow.consumer.clone(),
            ))
        {
            return Err("PlacementPlan contains an invalid or duplicate value flow".to_string());
        }
        let producer_position = schedule_positions
            .get(&flow.producer.text())
            .ok_or_else(|| "value flow references an unknown producer".to_string())?;
        let consumer_position = schedule_positions
            .get(&flow.consumer.text())
            .ok_or_else(|| "value flow references an unknown consumer".to_string())?;
        if producer_position >= consumer_position {
            return Err("PlacementPlan value flow must follow its schedule".to_string());
        }
        if let Some(previous) = flow_producers.insert(flow.value.clone(), flow.producer.clone()) {
            if previous != flow.producer {
                return Err("PlacementPlan ValueKey must have one producer".to_string());
            }
        }
        let producer_selection = selections[&flow.producer.text()];
        let consumer_selection = selections[&flow.consumer.text()];
        for placement in [&producer_selection.placement, &consumer_selection.placement] {
            let memory = if placement.kind == "cpu" {
                "host"
            } else {
                "device"
            };
            required_flow_residencies.insert(format!(
                "{}|{}|{}",
                flow.value.text(),
                placement.text(),
                memory
            ));
        }
        if producer_selection.placement != consumer_selection.placement {
            expected_routes.insert((
                flow.producer.clone(),
                flow.consumer.clone(),
                flow.value.clone(),
            ));
        }
    }

    let mut residencies = HashSet::new();
    for residency in &plan.residencies {
        validate_v3_residency(residency, "placement residency", &topology_devices)?;
        if !residencies.insert(residency.text()) {
            return Err("PlacementPlan contains duplicate residencies".to_string());
        }
    }
    if !required_flow_residencies.is_subset(&residencies) {
        return Err("PlacementPlan value flow is missing a required residency".to_string());
    }

    let mut engine_capacities = HashMap::new();
    for descriptor in &plan.copy_engines {
        if !["h2d", "d2h", "p2p"].contains(&descriptor.engine.as_str())
            || descriptor.capacity < 1
            || contains_runtime_text(&descriptor.engine)
        {
            return Err("PlacementPlan has an invalid copy-engine capacity".to_string());
        }
        if engine_capacities
            .insert(descriptor.engine.clone(), descriptor.capacity)
            .is_some()
        {
            return Err("PlacementPlan contains duplicate copy-engine capacities".to_string());
        }
    }

    let mut move_positions = HashMap::new();
    let mut routes: HashMap<(NodeKeyDesc, NodeKeyDesc, ValueKeyV3Desc), Vec<usize>> =
        HashMap::new();
    for (position, item) in plan.moves.iter().enumerate() {
        validate_v3_value(&item.key.value, "scheduled move key")?;
        if item.key.ordinal < 0 || item.key.value != item.r#move.value {
            return Err("scheduled move has an invalid stable key".to_string());
        }
        validate_v3_value(&item.r#move.value, "scheduled move value")?;
        validate_v3_residency(
            &item.r#move.source,
            "scheduled move source",
            &topology_devices,
        )?;
        validate_v3_residency(
            &item.r#move.destination,
            "scheduled move destination",
            &topology_devices,
        )?;
        if item.r#move.source.value != item.r#move.value
            || item.r#move.destination.value != item.r#move.value
            || item.r#move.source.placement == item.r#move.destination.placement
            || item.r#move.byte_count < 0
            || !item.r#move.asynchronous
        {
            return Err("scheduled move has inconsistent typed values".to_string());
        }
        let expected_kind = match (
            item.r#move.source.placement.kind.as_str(),
            item.r#move.destination.placement.kind.as_str(),
        ) {
            ("cpu", "cuda") => "h2d",
            ("cuda", "cpu") => "d2h",
            ("cuda", "cuda") => "p2p",
            _ => "",
        };
        if item.r#move.kind != expected_kind || item.engine != expected_kind {
            return Err("scheduled move kind/engine differs from its placements".to_string());
        }
        if item.r#move.kind == "p2p" {
            let source = item
                .r#move
                .source
                .placement
                .index
                .expect("validated CUDA source");
            let destination = item
                .r#move
                .destination
                .placement
                .index
                .expect("validated CUDA destination");
            if !plan.topology.peer_access.contains(&(source, destination)) {
                return Err("scheduled P2P move is outside topology".to_string());
            }
        }
        if item.start_at < 0 || item.end_at <= item.start_at {
            return Err("scheduled move has an invalid interval".to_string());
        }
        let producer = item
            .producer
            .as_ref()
            .ok_or_else(|| "scheduled move requires a producer node".to_string())?;
        let consumer = item
            .consumer
            .as_ref()
            .ok_or_else(|| "scheduled move requires a consumer node".to_string())?;
        validate_v3_node_key(producer, "scheduled move producer")?;
        validate_v3_node_key(consumer, "scheduled move consumer")?;
        routes
            .entry((
                producer.clone(),
                consumer.clone(),
                item.r#move.value.clone(),
            ))
            .or_default()
            .push(position);
        let producer_position = schedule_positions
            .get(&producer.text())
            .ok_or_else(|| "scheduled move references an unknown producer".to_string())?;
        let consumer_position = schedule_positions
            .get(&consumer.text())
            .ok_or_else(|| "scheduled move references an unknown consumer".to_string())?;
        if producer_position >= consumer_position
            || item.start_at < plan.schedule[*producer_position].end_at
            || item.end_at > plan.schedule[*consumer_position].start_at
        {
            return Err(
                "scheduled move must complete between its producer and consumer".to_string(),
            );
        }
        if !engine_capacities.contains_key(&item.engine) {
            return Err("scheduled move references an undeclared copy engine".to_string());
        }
        if !residencies.contains(&item.r#move.source.text())
            || !residencies.contains(&item.r#move.destination.text())
        {
            return Err("scheduled move references an unknown residency".to_string());
        }
        if move_positions.insert(item.key.text(), position).is_some() {
            return Err("PlacementPlan contains duplicate scheduled move keys".to_string());
        }
        let _ = item.used_fallback;
    }
    let mut expected_residencies = required_flow_residencies.clone();
    for item in &plan.moves {
        expected_residencies.insert(item.r#move.source.text());
        expected_residencies.insert(item.r#move.destination.text());
    }
    if residencies != expected_residencies {
        return Err(
            "PlacementPlan residencies must exactly cover value flows and routes".to_string(),
        );
    }
    for ((producer, consumer, _value), positions) in &routes {
        let route: Vec<&ScheduledMoveV3Desc> = positions
            .iter()
            .map(|position| &plan.moves[*position])
            .collect();
        let producer_selection = selections
            .get(&producer.text())
            .ok_or_else(|| "transfer route references an unknown producer".to_string())?;
        let consumer_selection = selections
            .get(&consumer.text())
            .ok_or_else(|| "transfer route references an unknown consumer".to_string())?;
        if route[0].r#move.source.placement != producer_selection.placement {
            return Err("transfer route source differs from its producer".to_string());
        }
        if route[route.len() - 1].r#move.destination.placement != consumer_selection.placement {
            return Err("transfer route destination differs from its consumer".to_string());
        }
        if !route[0].depends_on.is_empty() {
            return Err("transfer route head must not depend on another move".to_string());
        }
        let source = &producer_selection.placement;
        let destination = &consumer_selection.placement;
        let expected_kinds: &[&str] = if source == destination {
            &[]
        } else if source.kind == "cpu" {
            &["h2d"]
        } else if destination.kind == "cpu" {
            &["d2h"]
        } else {
            let source_index = source.index.expect("validated CUDA source");
            let destination_index = destination.index.expect("validated CUDA destination");
            if plan
                .topology
                .peer_access
                .contains(&(source_index, destination_index))
            {
                &["p2p"]
            } else {
                &["d2h", "h2d"]
            }
        };
        if route
            .iter()
            .map(|item| item.r#move.kind.as_str())
            .collect::<Vec<_>>()
            != expected_kinds
        {
            return Err("transfer route differs from its topology matrix".to_string());
        }
        if route.len() == 2
            && (route[0].r#move.destination != route[1].r#move.source
                || route[0].r#move.destination.memory != "pinned_host"
                || route[1].depends_on.as_slice() != [route[0].key.clone()])
        {
            return Err("staged transfer route must use one continuous pinned chain".to_string());
        }
    }
    if routes.keys().cloned().collect::<HashSet<_>>() != expected_routes {
        return Err(
            "PlacementPlan routes must exactly cover cross-placement value flows".to_string(),
        );
    }
    for (position, item) in plan.moves.iter().enumerate() {
        let mut dependencies = HashSet::new();
        for dependency in &item.depends_on {
            let text = dependency.text();
            if !dependencies.insert(text.clone()) {
                return Err("scheduled move dependencies must be unique".to_string());
            }
            let dependency_position = move_positions
                .get(&text)
                .ok_or_else(|| "scheduled move references an unknown dependency".to_string())?;
            if *dependency_position >= position
                || plan.moves[*dependency_position].end_at > item.start_at
            {
                return Err("scheduled move dependencies must complete before launch".to_string());
            }
        }
        if item.r#move.source.memory == "pinned_host" {
            let has_staging_dependency = item.depends_on.iter().any(|dependency| {
                let dependency_position = move_positions[&dependency.text()];
                plan.moves[dependency_position].r#move.destination == item.r#move.source
            });
            if !has_staging_dependency {
                return Err("pinned-host H2D requires its D2H dependency".to_string());
            }
        }
    }
    for (engine, capacity) in &engine_capacities {
        for item in plan.moves.iter().filter(|item| item.engine == *engine) {
            let overlap = plan
                .moves
                .iter()
                .filter(|other| {
                    other.engine == *engine
                        && other.start_at < item.end_at
                        && other.end_at > item.start_at
                })
                .count() as i64;
            if overlap > *capacity {
                return Err("scheduled moves exceed copy-engine capacity".to_string());
            }
        }
    }

    let mut leases = HashMap::new();
    for lease in &plan.leases {
        validate_v3_residency(&lease.residency, "placement lease", &topology_devices)?;
        if lease.live_start < 0 || lease.live_end < lease.live_start {
            return Err("placement lease has an invalid live range".to_string());
        }
        let residency = lease.residency.text();
        if !residencies.contains(&residency) {
            return Err("placement lease references an unknown residency".to_string());
        }
        if leases.insert(residency, lease).is_some() {
            return Err("PlacementPlan contains duplicate residency leases".to_string());
        }
    }
    if leases.len() != residencies.len() {
        return Err("every PlacementPlan residency requires one lease".to_string());
    }
    for flow in &plan.value_flows {
        let producer_position = schedule_positions[&flow.producer.text()];
        let consumer_position = schedule_positions[&flow.consumer.text()];
        let producer_end = plan.schedule[producer_position].end_at;
        let consumer_end = plan.schedule[consumer_position].end_at;
        let producer_placement = &selections[&flow.producer.text()].placement;
        let consumer_placement = &selections[&flow.consumer.text()].placement;
        for placement in [producer_placement, consumer_placement] {
            let memory = if placement.kind == "cpu" {
                "host"
            } else {
                "device"
            };
            let residency = format!("{}|{}|{}", flow.value.text(), placement.text(), memory);
            let lease = leases[&residency];
            if lease.live_start > producer_end || lease.live_end < consumer_end {
                return Err(
                    "PlacementPlan value-flow lease must cover producer to consumer".to_string(),
                );
            }
        }
    }
    for item in &plan.moves {
        for residency in [&item.r#move.source, &item.r#move.destination] {
            let lease = leases[&residency.text()];
            if lease.live_start > item.start_at || lease.live_end < item.end_at {
                return Err("placement lease does not cover its scheduled move".to_string());
            }
        }
    }

    Ok(plan
        .schedule
        .iter()
        .map(|item| item.node_key.text())
        .collect())
}

fn validate_v4_segment(value: &str, layer: &str) -> Result<(), String> {
    if value.is_empty()
        || value.contains('/')
        || value.contains('\\')
        || contains_runtime_text(value)
    {
        return Err(format!("{} has an invalid stable segment", layer));
    }
    Ok(())
}

fn validate_v4_key_segment(value: &str, layer: &str, reserved: &[char]) -> Result<(), String> {
    validate_v4_segment(value, layer)?;
    if value.chars().any(|character| reserved.contains(&character)) {
        return Err(format!("{} contains a reserved key delimiter", layer));
    }
    Ok(())
}

fn distributed_groups(
    topology: &DistributedTopologyV4Desc,
    kind: &str,
) -> Result<Vec<Vec<i64>>, String> {
    let model: Vec<&TopologyRankV4Desc> = topology
        .ranks
        .iter()
        .filter(|rank| rank.role == "model")
        .collect();
    if kind == "global" {
        return Ok(vec![model.iter().map(|rank| rank.rank).collect()]);
    }
    if kind == "helper" {
        let helpers: Vec<i64> = topology
            .ranks
            .iter()
            .filter(|rank| rank.role == "helper")
            .map(|rank| rank.rank)
            .collect();
        return Ok(if helpers.is_empty() {
            Vec::new()
        } else {
            vec![helpers]
        });
    }
    let mut grouped: HashMap<(i64, i64), Vec<&TopologyRankV4Desc>> = HashMap::new();
    for rank in model {
        let (group_key, order) = match kind {
            "tensor_parallel" => (
                (
                    rank.dp_rank.expect("model coordinates validated"),
                    rank.pp_rank.expect("model coordinates validated"),
                ),
                rank.tp_rank.expect("model coordinates validated"),
            ),
            "pipeline_parallel" => (
                (
                    rank.dp_rank.expect("model coordinates validated"),
                    rank.tp_rank.expect("model coordinates validated"),
                ),
                rank.pp_rank.expect("model coordinates validated"),
            ),
            "data_parallel" => (
                (
                    rank.pp_rank.expect("model coordinates validated"),
                    rank.tp_rank.expect("model coordinates validated"),
                ),
                rank.dp_rank.expect("model coordinates validated"),
            ),
            _ => return Err("distributed plan has an unsupported group kind".to_string()),
        };
        let values = grouped.entry(group_key).or_default();
        values.push(rank);
        values.sort_by_key(|candidate| {
            let candidate_order = match kind {
                "tensor_parallel" => candidate.tp_rank,
                "pipeline_parallel" => candidate.pp_rank,
                _ => candidate.dp_rank,
            }
            .expect("model coordinates validated");
            (candidate_order, candidate.rank)
        });
        let _ = order;
    }
    let mut keys: Vec<(i64, i64)> = grouped.keys().copied().collect();
    keys.sort_unstable();
    Ok(keys
        .iter()
        .map(|key| grouped[key].iter().map(|rank| rank.rank).collect())
        .collect())
}

fn validate_distributed_topology_v4(topology: &DistributedTopologyV4Desc) -> Result<(), String> {
    if topology.ranks.is_empty()
        || topology.tp_size < 1
        || topology.pp_size < 1
        || topology.dp_size < 1
    {
        return Err("DistributedTopology has invalid dimensions".to_string());
    }
    let mut local_devices = HashSet::new();
    let mut coordinates = HashSet::new();
    let mut model_count = 0_i64;
    for (position, rank) in topology.ranks.iter().enumerate() {
        if rank.rank != position as i64 || rank.local_rank < 0 {
            return Err("DistributedTopology ranks must be contiguous".to_string());
        }
        validate_v4_segment(&rank.node_id, "distributed node id")?;
        validate_v3_placement(&rank.placement, "distributed rank placement")?;
        if !local_devices.insert((rank.node_id.clone(), rank.local_rank)) {
            return Err("DistributedTopology local rank identities must be unique".to_string());
        }
        match rank.role.as_str() {
            "model" => {
                model_count += 1;
                let coordinate = match (rank.tp_rank, rank.pp_rank, rank.dp_rank) {
                    (Some(tp), Some(pp), Some(dp))
                        if tp >= 0
                            && tp < topology.tp_size
                            && pp >= 0
                            && pp < topology.pp_size
                            && dp >= 0
                            && dp < topology.dp_size =>
                    {
                        (tp, pp, dp)
                    }
                    _ => return Err("model rank has invalid TP/PP/DP coordinates".to_string()),
                };
                if !coordinates.insert(coordinate) {
                    return Err("DistributedTopology has duplicate model coordinates".to_string());
                }
            }
            "helper" => {
                if rank.tp_rank.is_some() || rank.pp_rank.is_some() || rank.dp_rank.is_some() {
                    return Err("helper ranks must not declare model coordinates".to_string());
                }
            }
            _ => return Err("DistributedTopology has an unsupported rank role".to_string()),
        }
    }
    if model_count != topology.tp_size * topology.pp_size * topology.dp_size {
        return Err("DistributedTopology model mesh is incomplete".to_string());
    }
    for dp in 0..topology.dp_size {
        for pp in 0..topology.pp_size {
            for tp in 0..topology.tp_size {
                if !coordinates.contains(&(tp, pp, dp)) {
                    return Err("DistributedTopology model mesh is incomplete".to_string());
                }
            }
        }
    }
    Ok(())
}

fn v4_shard_coordinates(
    rank: &TopologyRankV4Desc,
    kind: &str,
) -> Result<(i64, (i64, i64)), String> {
    let tp = rank
        .tp_rank
        .ok_or_else(|| "sharded state requires model-rank affinity".to_string())?;
    let pp = rank
        .pp_rank
        .ok_or_else(|| "sharded state requires model-rank affinity".to_string())?;
    let dp = rank
        .dp_rank
        .ok_or_else(|| "sharded state requires model-rank affinity".to_string())?;
    match kind {
        "tensor" => Ok((tp, (pp, dp))),
        "pipeline" => Ok((pp, (tp, dp))),
        "data" => Ok((dp, (tp, pp))),
        _ => Err("state does not declare a sharded topology axis".to_string()),
    }
}

fn v4_collective_window_order(window: &str) -> Result<i64, String> {
    match window {
        "prepare" => Ok(0),
        "before_compute" => Ok(1),
        "after_compute" => Ok(2),
        "step_end" => Ok(3),
        _ => Err("distributed collective has an unsupported lifecycle window".to_string()),
    }
}

fn validate_distributed_plan_v4(plan: &DistributedFlowPlanV4Desc) -> Result<Vec<String>, String> {
    if plan.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported DistributedFlowPlan schema version {}",
            plan.schema_version
        ));
    }
    if plan.plan_epoch < 1 {
        return Err("DistributedFlowPlan plan_epoch must be >= 1".to_string());
    }
    validate_distributed_topology_v4(&plan.topology)?;

    let mut requirement_keys = HashSet::new();
    let mut expected_residences = Vec::new();
    let mut last_state_key: Option<StateKeyV4Desc> = None;
    for requirement in &plan.state_requirements {
        validate_v4_key_segment(
            &requirement.key.namespace,
            "distributed state namespace",
            &[':', '@'],
        )?;
        validate_v4_key_segment(&requirement.key.name, "distributed state name", &[':', '@'])?;
        if requirement.key.version < 1 || !requirement_keys.insert(requirement.key.clone()) {
            return Err(
                "distributed state requirement has an invalid or duplicate key".to_string(),
            );
        }
        let state_key = requirement.key.clone();
        if last_state_key.as_ref().is_some_and(|key| key > &state_key) {
            return Err("distributed state requirements must use canonical order".to_string());
        }
        last_state_key = Some(state_key);
        if !["replica", "shard", "global"].contains(&requirement.scope.as_str()) {
            return Err("distributed state has an unsupported scope".to_string());
        }
        let expected_partitions = match requirement.shard.kind.as_str() {
            "replicated" => 1,
            "tensor" => plan.topology.tp_size,
            "pipeline" => plan.topology.pp_size,
            "data" => plan.topology.dp_size,
            _ => return Err("distributed state has an unsupported shard kind".to_string()),
        };
        if requirement.shard.partitions != expected_partitions
            || requirement.shard.partitions < 1
            || matches!(requirement.shard.kind.as_str(), "tensor" | "data")
                && requirement.shard.axis.is_none_or(|axis| axis < 0)
            || requirement.shard.kind == "pipeline" && requirement.shard.axis.is_some()
            || requirement.shard.kind == "replicated"
                && (requirement.shard.axis.is_some() || requirement.scope == "shard")
        {
            return Err("distributed state shard does not match topology or scope".to_string());
        }
        let role = requirement.affinity.role.as_deref().unwrap_or("model");
        if !["model", "helper"].contains(&role) {
            return Err("distributed state affinity has an unsupported role".to_string());
        }
        let rank_filter: HashSet<i64> = requirement.affinity.ranks.iter().copied().collect();
        let node_filter: HashSet<&str> = requirement
            .affinity
            .node_ids
            .iter()
            .map(String::as_str)
            .collect();
        for node_id in &requirement.affinity.node_ids {
            validate_v4_segment(node_id, "distributed state affinity node id")?;
        }
        if rank_filter.len() != requirement.affinity.ranks.len()
            || node_filter.len() != requirement.affinity.node_ids.len()
            || requirement
                .affinity
                .ranks
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || requirement
                .affinity
                .node_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || rank_filter
                .iter()
                .any(|rank| *rank < 0 || *rank >= plan.topology.ranks.len() as i64)
        {
            return Err("distributed state affinity is invalid".to_string());
        }
        let eligible: Vec<&TopologyRankV4Desc> = plan
            .topology
            .ranks
            .iter()
            .filter(|rank| {
                (rank_filter.is_empty() || rank_filter.contains(&rank.rank))
                    && (node_filter.is_empty() || node_filter.contains(rank.node_id.as_str()))
                    && rank.role == role
            })
            .collect();
        if eligible.is_empty() {
            return Err("distributed state affinity has no eligible rank".to_string());
        }
        if requirement.shard.kind == "replicated" {
            for (ordinal, rank) in eligible.iter().enumerate() {
                expected_residences.push(StateResidenceV4Desc {
                    key: requirement.key.clone(),
                    rank: rank.rank,
                    placement: rank.placement.clone(),
                    shard_index: 0,
                    replica_index: ordinal as i64,
                });
            }
            continue;
        }
        let mut shard_groups: HashMap<(i64, i64), HashSet<i64>> = HashMap::new();
        for rank in &eligible {
            let (shard_index, replica_key) = v4_shard_coordinates(rank, &requirement.shard.kind)?;
            shard_groups
                .entry(replica_key)
                .or_default()
                .insert(shard_index);
        }
        let expected_shards: HashSet<i64> = (0..expected_partitions).collect();
        if shard_groups
            .values()
            .any(|shards| shards != &expected_shards)
            || eligible.len() != shard_groups.len() * expected_partitions as usize
        {
            return Err(
                "distributed state affinity must exactly cover each topology shard axis"
                    .to_string(),
            );
        }
        let mut replica_keys: Vec<(i64, i64)> = shard_groups.keys().copied().collect();
        replica_keys.sort_unstable();
        let replica_indices: HashMap<(i64, i64), i64> = replica_keys
            .iter()
            .enumerate()
            .map(|(index, key)| (*key, index as i64))
            .collect();
        for rank in eligible {
            let (shard_index, replica_key) = v4_shard_coordinates(rank, &requirement.shard.kind)?;
            expected_residences.push(StateResidenceV4Desc {
                key: requirement.key.clone(),
                rank: rank.rank,
                placement: rank.placement.clone(),
                shard_index,
                replica_index: replica_indices[&replica_key],
            });
        }
    }
    if plan.state_residences != expected_residences {
        return Err(
            "DistributedFlowPlan state residences must exactly match requirements".to_string(),
        );
    }

    let mut request_by_key = HashMap::new();
    for request in &plan.collective_requests {
        validate_v4_key_segment(
            &request.key.namespace,
            "collective namespace",
            &[':', '/', '@'],
        )?;
        validate_v4_key_segment(&request.key.name, "collective name", &[':', '/', '@'])?;
        if request.key.ordinal < 0
            || request_by_key
                .insert(request.key.clone(), request)
                .is_some()
        {
            return Err("distributed plan has duplicate or invalid collective keys".to_string());
        }
        if ![
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "broadcast",
            "send_recv",
            "barrier",
        ]
        .contains(&request.kind.as_str())
            || !["nccl", "gloo"].contains(&request.backend.as_str())
            || !["prepare", "before_compute", "after_compute", "step_end"]
                .contains(&request.window.as_str())
            || request.group_index < 0
        {
            return Err("distributed collective request has invalid typed facts".to_string());
        }
        if request
            .value
            .as_ref()
            .is_some_and(|value| validate_v3_value(value, "collective value").is_err())
        {
            return Err("distributed collective request has an invalid value".to_string());
        }
        let groups = distributed_groups(&plan.topology, &request.group)?;
        let ranks = groups
            .get(request.group_index as usize)
            .ok_or_else(|| "collective group index is outside topology".to_string())?;
        if request.kind == "send_recv" && ranks.len() != 2 {
            return Err("send_recv requires an exact two-rank communicator".to_string());
        }
        match (request.kind.as_str(), request.root_rank) {
            ("broadcast" | "send_recv", Some(root)) if ranks.contains(&root) => {}
            ("broadcast" | "send_recv", _) => {
                return Err("broadcast/send_recv requires a root in its communicator".to_string())
            }
            (_, None) => {}
            _ => return Err("only broadcast/send_recv may declare a root".to_string()),
        }
        if request.backend == "nccl"
            && ranks
                .iter()
                .any(|rank| plan.topology.ranks[*rank as usize].placement.kind != "cuda")
        {
            return Err("NCCL communicators require CUDA ranks".to_string());
        }
        let dependency_set: HashSet<&CollectiveKeyV4Desc> = request.depends_on.iter().collect();
        if dependency_set.len() != request.depends_on.len() || dependency_set.contains(&request.key)
        {
            return Err("collective dependencies must be unique and non-self".to_string());
        }
    }
    if plan.collective_requests.iter().any(|request| {
        request
            .depends_on
            .iter()
            .any(|dependency| !request_by_key.contains_key(dependency))
    }) {
        return Err("collective request references an unknown dependency".to_string());
    }
    for request in &plan.collective_requests {
        let request_window = v4_collective_window_order(&request.window)?;
        if request.depends_on.iter().any(|dependency| {
            v4_collective_window_order(&request_by_key[dependency].window)
                .is_ok_and(|dependency_window| dependency_window > request_window)
        }) {
            return Err(
                "collective dependency cannot move backwards in lifecycle time".to_string(),
            );
        }
    }
    let mut remaining: HashMap<CollectiveKeyV4Desc, HashSet<CollectiveKeyV4Desc>> = plan
        .collective_requests
        .iter()
        .map(|request| {
            (
                request.key.clone(),
                request.depends_on.iter().cloned().collect(),
            )
        })
        .collect();
    let mut ordered_keys = Vec::new();
    while !remaining.is_empty() {
        let mut ready: Vec<CollectiveKeyV4Desc> = remaining
            .iter()
            .filter(|(_, dependencies)| dependencies.is_empty())
            .map(|(key, _)| key.clone())
            .collect();
        ready.sort_by_key(|key| {
            (
                v4_collective_window_order(&request_by_key[key].window)
                    .expect("request window validated"),
                key.clone(),
            )
        });
        let key = ready
            .first()
            .ok_or_else(|| "collective dependencies contain a cycle".to_string())?
            .clone();
        remaining.remove(&key);
        for dependencies in remaining.values_mut() {
            dependencies.remove(&key);
        }
        ordered_keys.push(key);
    }
    let declared_keys: Vec<CollectiveKeyV4Desc> = plan
        .collective_requests
        .iter()
        .map(|request| request.key.clone())
        .collect();
    if declared_keys != ordered_keys {
        return Err("collective requests must use canonical global order".to_string());
    }

    let mut expected_communicators: HashMap<CommunicatorKeyV4Desc, CommunicatorV4Desc> =
        HashMap::new();
    for key in &ordered_keys {
        let request = request_by_key[key];
        let communicator_key = CommunicatorKeyV4Desc {
            kind: request.group.clone(),
            group_index: request.group_index,
        };
        let ranks = distributed_groups(&plan.topology, &request.group)?
            [request.group_index as usize]
            .clone();
        let descriptor = CommunicatorV4Desc {
            key: communicator_key.clone(),
            ranks,
            backend: request.backend.clone(),
        };
        if expected_communicators
            .insert(communicator_key.clone(), descriptor.clone())
            .is_some_and(|previous| previous != descriptor)
        {
            return Err("one communicator cannot use conflicting backends".to_string());
        }
    }
    let actual_communicators: HashMap<CommunicatorKeyV4Desc, CommunicatorV4Desc> = plan
        .communicators
        .iter()
        .map(|communicator| (communicator.key.clone(), communicator.clone()))
        .collect();
    let mut expected_communicator_order: Vec<CommunicatorV4Desc> =
        expected_communicators.values().cloned().collect();
    expected_communicator_order.sort_by(|left, right| left.key.cmp(&right.key));
    if actual_communicators.len() != plan.communicators.len()
        || actual_communicators != expected_communicators
        || plan.communicators != expected_communicator_order
    {
        return Err("DistributedFlowPlan communicators must exactly match demand".to_string());
    }

    if plan.collectives.len() != ordered_keys.len() {
        return Err("DistributedFlowPlan collective count differs from demand".to_string());
    }
    for (sequence, (key, collective)) in ordered_keys.iter().zip(&plan.collectives).enumerate() {
        let request = request_by_key[key];
        let expected_communicator = CommunicatorKeyV4Desc {
            kind: request.group.clone(),
            group_index: request.group_index,
        };
        if collective.key != *key
            || collective.kind != request.kind
            || collective.communicator != expected_communicator
            || collective.sequence != sequence as i64
            || collective.window != request.window
            || collective.value != request.value
            || collective.depends_on != request.depends_on
            || collective.root_rank != request.root_rank
        {
            return Err(
                "DistributedFlowPlan collectives must use canonical global order".to_string(),
            );
        }
    }

    if plan.local_schedules.len() != plan.topology.ranks.len() {
        return Err("DistributedFlowPlan requires one schedule per rank".to_string());
    }
    for (rank, schedule) in plan.local_schedules.iter().enumerate() {
        let expected: Vec<CollectiveKeyV4Desc> = plan
            .collectives
            .iter()
            .filter(|collective| {
                actual_communicators[&collective.communicator]
                    .ranks
                    .contains(&(rank as i64))
            })
            .map(|collective| collective.key.clone())
            .collect();
        if schedule.rank != rank as i64 || schedule.collectives != expected {
            return Err("rank-local schedule must project the global collective order".to_string());
        }
    }
    Ok(ordered_keys.iter().map(CollectiveKeyV4Desc::text).collect())
}

fn node_key_json_v5(value: &NodeKeyDesc) -> JsonValue {
    json!({
        "namespace": value.namespace,
        "owner": value.owner,
        "ordinal": value.ordinal,
    })
}

fn executor_key_json_v5(value: &ExecutorKeyDesc) -> JsonValue {
    json!({
        "namespace": value.namespace,
        "name": value.name,
        "version": value.version,
    })
}

fn placement_json_v5(value: &PlacementV3Desc) -> JsonValue {
    json!({"kind": value.kind, "index": value.index})
}

fn value_key_json_v5(value: &ValueKeyV3Desc) -> JsonValue {
    json!({"name": value.name, "version": value.version})
}

fn residency_json_v5(value: &ResidencyV3Desc) -> JsonValue {
    json!({
        "value": value_key_json_v5(&value.value),
        "placement": placement_json_v5(&value.placement),
        "memory": value.memory,
    })
}

fn move_key_json_v5(value: &MoveKeyV3Desc) -> JsonValue {
    json!({
        "value": value_key_json_v5(&value.value),
        "ordinal": value.ordinal,
    })
}

fn scheduled_move_json_v5(value: &ScheduledMoveV3Desc) -> JsonValue {
    json!({
        "key": move_key_json_v5(&value.key),
        "move": {
            "value": value_key_json_v5(&value.r#move.value),
            "source": residency_json_v5(&value.r#move.source),
            "destination": residency_json_v5(&value.r#move.destination),
            "kind": value.r#move.kind,
            "byte_count": value.r#move.byte_count,
            "asynchronous": value.r#move.asynchronous,
        },
        "engine": value.engine,
        "start_at": value.start_at,
        "end_at": value.end_at,
        "producer": value.producer.as_ref().map(node_key_json_v5),
        "consumer": value.consumer.as_ref().map(node_key_json_v5),
        "depends_on": value.depends_on.iter().map(move_key_json_v5).collect::<Vec<_>>(),
        "used_fallback": value.used_fallback,
    })
}

fn python_ascii_json_v5(value: &JsonValue) -> Result<Vec<u8>, String> {
    let encoded = serde_json::to_string(value)
        .map_err(|error| format!("failed to serialize PlacementPlan signature: {error}"))?;
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
    Ok(canonical.into_bytes())
}

fn placement_signature_v5(plan: &PlacementPlanV3Desc) -> Result<String, String> {
    let mut peers: Vec<(i64, i64)> = plan.topology.peer_access.iter().copied().collect();
    peers.sort_unstable();
    let selections: Vec<JsonValue> = plan
        .selections
        .iter()
        .map(|selection| {
            json!({
                "node_key": node_key_json_v5(&selection.node_key),
                "logical_executor": executor_key_json_v5(&selection.logical_executor),
                "key": executor_key_json_v5(&selection.key),
                "placement": placement_json_v5(&selection.placement),
                "duration": {
                    "microseconds": selection.duration.microseconds,
                    "samples": selection.duration.samples,
                },
                "oracle_key": selection.oracle_key.as_ref().map(|key| json!({
                    "namespace": key.namespace,
                    "name": key.name,
                    "version": key.version,
                })),
            })
        })
        .collect();
    let schedule: Vec<JsonValue> = plan
        .schedule
        .iter()
        .map(|node| {
            json!({
                "node_key": node_key_json_v5(&node.node_key),
                "implementation": executor_key_json_v5(&node.implementation),
                "placement": placement_json_v5(&node.placement),
                "start_at": node.start_at,
                "end_at": node.end_at,
                "depends_on": node.depends_on.iter().map(node_key_json_v5).collect::<Vec<_>>(),
            })
        })
        .collect();
    let payload = json!({
        "plan_epoch": plan.plan_epoch,
        "selections": selections,
        "schedule": schedule,
        "topology": {
            "peer_access": peers,
            "cuda_devices": plan.topology.cuda_devices,
        },
        "value_flows": plan.value_flows.iter().map(|flow| json!({
            "value": value_key_json_v5(&flow.value),
            "producer": node_key_json_v5(&flow.producer),
            "consumer": node_key_json_v5(&flow.consumer),
            "byte_count": flow.byte_count,
        })).collect::<Vec<_>>(),
        "residencies": plan.residencies.iter().map(residency_json_v5).collect::<Vec<_>>(),
        "moves": plan.moves.iter().map(scheduled_move_json_v5).collect::<Vec<_>>(),
        "leases": plan.leases.iter().map(|lease| json!({
            "residency": residency_json_v5(&lease.residency),
            "live_start": lease.live_start,
            "live_end": lease.live_end,
        })).collect::<Vec<_>>(),
        "copy_engines": plan.copy_engines.iter().map(|engine| json!({
            "engine": engine.engine,
            "capacity": engine.capacity,
        })).collect::<Vec<_>>(),
        "schema_version": plan.schema_version,
    });
    let canonical = python_ascii_json_v5(&payload)?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

fn validate_v5_version_key(key: &VersionKeyV5Desc, layer: &str) -> Result<(), String> {
    validate_v4_key_segment(&key.namespace, layer, &[':', '@', '/'])?;
    validate_v4_key_segment(&key.name, layer, &[':', '@', '/'])?;
    if key.version < 1 {
        return Err(format!("{layer} version must be >= 1"));
    }
    Ok(())
}

fn validate_v5_ordinal_key(key: &OrdinalKeyV5Desc, layer: &str) -> Result<(), String> {
    validate_v4_key_segment(&key.namespace, layer, &[':', '@', '/'])?;
    validate_v4_key_segment(&key.name, layer, &[':', '@', '/'])?;
    if key.ordinal < 0 {
        return Err(format!("{layer} ordinal must be non-negative"));
    }
    Ok(())
}

fn validate_adaptive_plan_v5(plan: &AdaptiveFlowPlanV5Desc) -> Result<Vec<String>, String> {
    if plan.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported AdaptiveFlowPlan schema version {}",
            plan.schema_version
        ));
    }
    if plan.plan_epoch < 1 || plan.plan_epoch != plan.placement_plan.plan_epoch {
        return Err("AdaptiveFlowPlan epoch must match its PlacementPlan".to_string());
    }
    validate_placement_plan_v3(&plan.placement_plan)?;
    if plan.scheduled_moves != plan.placement_plan.moves {
        return Err("AdaptiveFlowPlan moves must exactly match its PlacementPlan".to_string());
    }
    if plan.placement_signature.len() != 64
        || !plan
            .placement_signature
            .bytes()
            .all(|value| value.is_ascii_hexdigit() && !value.is_ascii_uppercase())
    {
        return Err("AdaptiveFlowPlan placement_signature must be lowercase SHA-256".to_string());
    }
    if plan.placement_signature != placement_signature_v5(&plan.placement_plan)? {
        return Err(
            "AdaptiveFlowPlan placement_signature differs from its PlacementPlan".to_string(),
        );
    }
    if plan.densification_limit < 1
        || plan.move_sets.len() > plan.densification_limit as usize
        || plan.actions.len() > plan.densification_limit as usize
    {
        return Err("AdaptiveFlowPlan exceeds its densification limit".to_string());
    }
    if plan.buffer_capacity < 0 || !matches!(plan.mode.as_str(), "static" | "adaptive") {
        return Err("AdaptiveFlowPlan has invalid capacity or mode".to_string());
    }

    let move_universe: HashMap<MoveKeyV3Desc, &ScheduledMoveV3Desc> = plan
        .scheduled_moves
        .iter()
        .map(|item| (item.key.clone(), item))
        .collect();
    if move_universe.len() != plan.scheduled_moves.len() {
        return Err("AdaptiveFlowPlan scheduled Move keys must be unique".to_string());
    }
    let mut move_set_keys = HashSet::new();
    let mut previous_move_set: Option<&OrdinalKeyV5Desc> = None;
    for move_set in &plan.move_sets {
        validate_v5_ordinal_key(&move_set.key, "adaptive Move-set key")?;
        if previous_move_set.is_some_and(|previous| previous >= &move_set.key)
            || !move_set_keys.insert(move_set.key.clone())
        {
            return Err("AdaptiveFlowPlan Move sets must use canonical unique keys".to_string());
        }
        previous_move_set = Some(&move_set.key);
        let selected: HashSet<MoveKeyV3Desc> =
            move_set.moves.iter().map(|item| item.key.clone()).collect();
        if selected.len() != move_set.moves.len() {
            return Err("adaptive Move set contains duplicate Moves".to_string());
        }
        let mut previous_move_key: Option<&MoveKeyV3Desc> = None;
        for scheduled in &move_set.moves {
            let expected = move_universe
                .get(&scheduled.key)
                .ok_or_else(|| "adaptive Move set invents a Phase 3 Move".to_string())?;
            if *expected != scheduled {
                return Err("adaptive Move set rewrites a Phase 3 Move".to_string());
            }
            if scheduled
                .depends_on
                .iter()
                .any(|dependency| !selected.contains(dependency))
            {
                return Err("adaptive Move set must be dependency-closed".to_string());
            }
            if previous_move_key.is_some_and(|previous| previous >= &scheduled.key) {
                return Err("adaptive Move sets must use canonical Move order".to_string());
            }
            previous_move_key = Some(&scheduled.key);
        }
    }

    let mut shape_keys = HashSet::new();
    let mut previous_shape: Option<&VersionKeyV5Desc> = None;
    for bucket in &plan.shape_buckets {
        validate_v5_version_key(&bucket.key, "adaptive shape-bucket key")?;
        if previous_shape.is_some_and(|previous| previous >= &bucket.key)
            || !shape_keys.insert(bucket.key.clone())
        {
            return Err("adaptive shape buckets must use canonical unique keys".to_string());
        }
        previous_shape = Some(&bucket.key);
        let exact = !bucket.shape.is_empty();
        let bounded = !bucket.lower_bounds.is_empty() && !bucket.upper_bounds.is_empty();
        if exact == bounded
            || bucket.shape.iter().any(|value| *value < 1)
            || bucket.lower_bounds.len() != bucket.upper_bounds.len()
            || bucket.lower_bounds.iter().any(|value| *value < 1)
            || bucket.upper_bounds.iter().any(|value| *value < 1)
            || bucket
                .lower_bounds
                .iter()
                .zip(&bucket.upper_bounds)
                .any(|(lower, upper)| lower > upper)
        {
            return Err("adaptive shape bucket has invalid exact/bounded dimensions".to_string());
        }
    }

    let mut workload_keys = HashSet::new();
    let mut previous_workload: Option<&VersionKeyV5Desc> = None;
    for bucket in &plan.workload_buckets {
        validate_v5_version_key(&bucket.key, "adaptive workload-bucket key")?;
        if previous_workload.is_some_and(|previous| previous >= &bucket.key)
            || !workload_keys.insert(bucket.key.clone())
        {
            return Err("adaptive workload buckets must use canonical unique keys".to_string());
        }
        previous_workload = Some(&bucket.key);
        if bucket
            .shape_buckets
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || bucket
                .shape_buckets
                .iter()
                .any(|key| !shape_keys.contains(key))
        {
            return Err("adaptive workload bucket has invalid shape references".to_string());
        }
    }

    let mut profile_keys = HashSet::new();
    let mut previous_profile: Option<&VersionKeyV5Desc> = None;
    for profile in &plan.profiles {
        validate_v5_version_key(&profile.key, "adaptive EWMA-profile key")?;
        if previous_profile.is_some_and(|previous| previous >= &profile.key)
            || !profile_keys.insert(profile.key.clone())
        {
            return Err("adaptive EWMA profiles must use canonical unique keys".to_string());
        }
        previous_profile = Some(&profile.key);
        if profile.alpha.numerator < 1
            || profile.alpha.denominator < 1
            || profile.alpha.numerator > profile.alpha.denominator
            || profile.samples < 0
            || profile.minimum < 0
            || profile.estimate < profile.minimum
            || profile
                .maximum
                .is_some_and(|maximum| maximum < profile.minimum || profile.estimate > maximum)
        {
            return Err("adaptive EWMA profile has invalid integer bounds".to_string());
        }
    }

    let mut budget_keys = HashSet::new();
    let mut previous_budget: Option<&VersionKeyV5Desc> = None;
    for budget in &plan.budgets {
        validate_v5_version_key(&budget.key, "adaptive interference-budget key")?;
        if previous_budget.is_some_and(|previous| previous >= &budget.key)
            || !budget_keys.insert(budget.key.clone())
            || budget.capacity < 0
            || budget
                .profile
                .as_ref()
                .is_some_and(|key| !profile_keys.contains(key))
        {
            return Err("adaptive interference budget is invalid".to_string());
        }
        previous_budget = Some(&budget.key);
    }

    let mut action_keys = HashSet::new();
    let mut previous_action: Option<&OrdinalKeyV5Desc> = None;
    for action in &plan.actions {
        validate_v5_ordinal_key(&action.key, "adaptive action key")?;
        validate_v4_key_segment(
            &action.semantic_key,
            "adaptive semantic key",
            &[':', '@', '/'],
        )?;
        if previous_action.is_some_and(|previous| previous >= &action.key)
            || !action_keys.insert(action.key.clone())
        {
            return Err("adaptive actions must use canonical unique keys".to_string());
        }
        previous_action = Some(&action.key);
    }
    let action_by_key: HashMap<OrdinalKeyV5Desc, &CandidateActionV5Desc> = plan
        .actions
        .iter()
        .map(|action| (action.key.clone(), action))
        .collect();
    let unknown_fallback = action_by_key
        .get(&plan.unknown_bucket_fallback)
        .ok_or_else(|| "unknown-bucket fallback is not an approved action".to_string())?;
    if unknown_fallback.kind != "authoritative_fallback" || !unknown_fallback.unknown_bucket {
        return Err("unknown-bucket fallback must be authoritative".to_string());
    }
    for action in &plan.actions {
        if !matches!(
            action.kind.as_str(),
            "execute" | "authoritative_fallback" | "shed"
        ) || action.ready_at < 0
            || action.deadline < 1
            || action.buffer_credits < 0
            || action.priority < 0
            || action.utility < 0
            || action.buffer_credits > plan.buffer_capacity
            || action
                .workload_bucket
                .as_ref()
                .is_some_and(|key| !workload_keys.contains(key))
            || action
                .shape_bucket
                .as_ref()
                .is_some_and(|key| !shape_keys.contains(key))
            || action
                .move_set
                .as_ref()
                .is_some_and(|key| !move_set_keys.contains(key))
        {
            return Err("adaptive action contains an invalid field or reference".to_string());
        }
        if action
            .budget_uses
            .windows(2)
            .any(|pair| pair[0].key >= pair[1].key)
            || action
                .budget_uses
                .iter()
                .any(|usage| usage.amount < 1 || !budget_keys.contains(&usage.key))
        {
            return Err("adaptive action has invalid interference-budget use".to_string());
        }
        match action.kind.as_str() {
            "shed" => {
                if action.required
                    || action.authoritative
                    || action.move_set.is_some()
                    || action.duration != 0
                    || action.buffer_credits != 0
                    || !action.budget_uses.is_empty()
                {
                    return Err("shedding action cannot represent required work".to_string());
                }
            }
            "authoritative_fallback" => {
                if !action.authoritative || action.duration < 1 || action.fallback_key.is_some() {
                    return Err("authoritative fallback action is invalid".to_string());
                }
            }
            "execute" => {
                if action.authoritative || action.duration < 1 || action.fallback_key.is_none() {
                    return Err("execute action requires an authoritative fallback".to_string());
                }
            }
            _ => unreachable!(),
        }
        if action.unknown_bucket
            && (action.kind != "authoritative_fallback"
                || action.workload_bucket.is_some()
                || action.shape_bucket.is_some())
        {
            return Err("unknown-bucket action must be an unscoped fallback".to_string());
        }
        if let Some(fallback_key) = &action.fallback_key {
            let fallback = action_by_key
                .get(fallback_key)
                .ok_or_else(|| "adaptive action fallback is unknown".to_string())?;
            if fallback.kind != "authoritative_fallback"
                || fallback.semantic_key != action.semantic_key
            {
                return Err("adaptive fallback changes action semantics".to_string());
            }
        }
        if let (Some(workload_key), Some(shape_key)) =
            (&action.workload_bucket, &action.shape_bucket)
        {
            let workload = plan
                .workload_buckets
                .iter()
                .find(|bucket| &bucket.key == workload_key)
                .expect("workload reference was validated");
            if !workload.shape_buckets.is_empty() && !workload.shape_buckets.contains(shape_key) {
                return Err("adaptive action shape is outside its workload bucket".to_string());
            }
        }
    }
    Ok(plan
        .actions
        .iter()
        .map(|action| action.key.text())
        .collect())
}

/// Validate one frozen, metadata-only physical execution plan. Runtime Tensor,
/// stream, event, graph, callback, and lease objects remain in Python.
#[pyfunction]
#[pyo3(signature = (plan))]
fn validate_physical_flow_plan(plan: PhysicalFlowPlanDesc) -> PyResult<WorkGraphValidation> {
    let topological_order = validate_physical_plan(&plan).map_err(validation_error)?;
    Ok(WorkGraphValidation { topological_order })
}

/// Validate one frozen Phase 3 local placement plan. Runtime implementations,
/// tensors, streams, events, pinned allocations, and transfer handles stay in Python.
#[pyfunction]
#[pyo3(signature = (plan))]
fn validate_placement_plan(plan: PlacementPlanV3Desc) -> PyResult<WorkGraphValidation> {
    let topological_order = validate_placement_plan_v3(&plan).map_err(validation_error)?;
    Ok(WorkGraphValidation { topological_order })
}

/// Validate one frozen Phase 4 global topology/state/collective plan. Runtime
/// process groups, tensors, streams, events, and async work handles stay in Python.
#[pyfunction]
#[pyo3(signature = (plan))]
fn validate_distributed_flow_plan(
    plan: DistributedFlowPlanV4Desc,
) -> PyResult<WorkGraphValidation> {
    let topological_order = validate_distributed_plan_v4(&plan).map_err(validation_error)?;
    Ok(WorkGraphValidation { topological_order })
}

/// Validate one frozen Phase 5 adaptive plan. Mutable EWMAs, observations,
/// admission reservations, queues, callbacks, and runtime payloads stay in Python.
#[pyfunction]
#[pyo3(signature = (plan))]
fn validate_adaptive_flow_plan(plan: AdaptiveFlowPlanV5Desc) -> PyResult<WorkGraphValidation> {
    let topological_order = validate_adaptive_plan_v5(&plan).map_err(validation_error)?;
    Ok(WorkGraphValidation { topological_order })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_workgraph, m)?)?;
    m.add_function(wrap_pyfunction!(validate_workgraph_ir, m)?)?;
    m.add_function(wrap_pyfunction!(validate_capture_hierarchy, m)?)?;
    m.add_function(wrap_pyfunction!(validate_physical_flow_plan, m)?)?;
    m.add_function(wrap_pyfunction!(validate_placement_plan, m)?)?;
    m.add_function(wrap_pyfunction!(validate_distributed_flow_plan, m)?)?;
    m.add_function(wrap_pyfunction!(validate_adaptive_flow_plan, m)?)?;
    m.add_class::<WorkGraphValidation>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(owner: &str) -> NodeKeyDesc {
        NodeKeyDesc {
            namespace: "consumer_flow".into(),
            owner: owner.into(),
            ordinal: 0,
        }
    }

    fn timing(mobility: &str) -> TimingDesc {
        TimingDesc {
            window: "same_step".into(),
            mobility: mobility.into(),
            deadline: "step_end".into(),
            max_defer_steps: 0,
            observes_version: "at_declared_window".into(),
        }
    }

    fn node(owner: &str, kind: &str, mobility: &str) -> SemanticNodeDesc {
        SemanticNodeDesc {
            node_key: key(owner),
            kind: kind.into(),
            executor: None,
            reads: Vec::new(),
            writes: Vec::new(),
            effects: Vec::new(),
            timing: timing(mobility),
            anchor: None,
            graph_safety: "eager_only".into(),
            source: owner.into(),
            legacy_boundary: None,
            shape_guard: Vec::new(),
        }
    }

    fn backend() -> BackendDesc {
        BackendDesc {
            backend_id: "vllm".into(),
            runner_family: "mrv1".into(),
            graph_mode: "eager".into(),
            backend_version: "0.27.1".into(),
            capabilities: Vec::new(),
            device: "cuda".into(),
            rank: 0,
            tp_size: 1,
            pp_size: 1,
            dp_size: 1,
            topology: Vec::new(),
        }
    }

    fn semantic(
        nodes: Vec<SemanticNodeDesc>,
        edges: Vec<(NodeKeyDesc, NodeKeyDesc, String)>,
    ) -> SemanticGraphDesc {
        SemanticGraphDesc {
            schema_version: 1,
            nodes,
            edges,
            values: Vec::new(),
            backend: Some(backend()),
            graph_mode: "eager".into(),
        }
    }

    fn data() -> DataCentricDesc {
        DataCentricDesc {
            schema_version: 1,
            nodes: Vec::new(),
            values: Vec::new(),
            accesses: Vec::new(),
            residences: Vec::new(),
            moves: Vec::new(),
            leases: Vec::new(),
            edges: Vec::new(),
            backend: Some(backend()),
            graph_mode: "eager".into(),
        }
    }

    #[test]
    fn deterministic_topological_order() {
        let graph = semantic(
            vec![
                node("b", "compute", "pinned"),
                node("a", "compute", "pinned"),
            ],
            vec![(key("a"), key("b"), "control".into())],
        );
        let keys = validate_semantic(&graph).expect("valid graph").0;
        assert_eq!(
            validate_edges(&keys, &graph.edges).unwrap(),
            vec!["consumer_flow/a/0", "consumer_flow/b/0"]
        );
    }

    #[test]
    fn native_legacy_cycle_and_move_contracts() {
        let native = semantic(vec![node("native", "compute", "deferrable")], Vec::new());
        assert!(validate_semantic(&native)
            .unwrap_err()
            .contains("WorkGraph-native"));
        let mut legacy_node = node("legacy", "compute", "deferrable");
        legacy_node.legacy_boundary = Some("legacy_boundary".into());
        let legacy = semantic(vec![legacy_node], Vec::new());
        assert!(validate_semantic(&legacy).is_ok());
        let mut legacy_latest_node = node("legacy_latest", "compute", "deferrable");
        legacy_latest_node.legacy_boundary = Some("legacy_boundary".into());
        legacy_latest_node.timing.observes_version = "latest_at_execution".into();
        let legacy_latest = semantic(vec![legacy_latest_node], Vec::new());
        assert!(validate_semantic(&legacy_latest).is_ok());
        let mut native_latest_node = node("native_latest", "compute", "pinned");
        native_latest_node.timing.observes_version = "latest_at_execution".into();
        let native_latest = semantic(vec![native_latest_node], Vec::new());
        assert!(validate_semantic(&native_latest)
            .unwrap_err()
            .contains("WorkGraph-native"));
        let cycle = semantic(
            vec![
                node("a", "compute", "pinned"),
                node("b", "compute", "pinned"),
            ],
            vec![
                (key("a"), key("b"), "control".into()),
                (key("b"), key("a"), "control".into()),
            ],
        );
        assert!(validate_semantic(&cycle).unwrap_err().contains("cycle"));
        let mut moved = data();
        moved.moves.push(MoveDesc {
            value: ValueDesc {
                value_id: "v".into(),
                version: 1,
                port: None,
                shape: Vec::new(),
                dtype: String::new(),
                layout: String::new(),
                alias_set: String::new(),
            },
            source: ResidenceDesc {
                value: ValueDesc {
                    value_id: "v".into(),
                    version: 1,
                    port: None,
                    shape: Vec::new(),
                    dtype: String::new(),
                    layout: String::new(),
                    alias_set: String::new(),
                },
                device: "cuda".into(),
                rank: 0,
                memory_kind: "backend_owned".into(),
                valid_version: 1,
            },
            destination: ResidenceDesc {
                value: ValueDesc {
                    value_id: "v".into(),
                    version: 1,
                    port: None,
                    shape: Vec::new(),
                    dtype: String::new(),
                    layout: String::new(),
                    alias_set: String::new(),
                },
                device: "cpu".into(),
                rank: 0,
                memory_kind: "pinned".into(),
                valid_version: 1,
            },
            subset: Vec::new(),
            layout: String::new(),
            engine: String::new(),
            asynchronous: false,
        });
        let backend = backend();
        let mut anchors = HashMap::new();
        assert!(validate_data(&moved, "eager", &backend, &mut anchors)
            .unwrap_err()
            .contains("Move descriptors"));
    }
}

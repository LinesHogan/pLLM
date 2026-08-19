//! Pure descriptor validation for the Phase 1a WorkGraph IR.
//!
//! The validator consumes the typed, metadata-only descriptors produced by
//! ``tllm.workgraph.ir``. Tensor/CUDA objects, placement decisions, and
//! runtime resources never cross this boundary.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};

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

/// Validate one frozen, metadata-only physical execution plan. Runtime Tensor,
/// stream, event, graph, callback, and lease objects remain in Python.
#[pyfunction]
#[pyo3(signature = (plan))]
fn validate_physical_flow_plan(plan: PhysicalFlowPlanDesc) -> PyResult<WorkGraphValidation> {
    let topological_order = validate_physical_plan(&plan).map_err(validation_error)?;
    Ok(WorkGraphValidation { topological_order })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_workgraph, m)?)?;
    m.add_function(wrap_pyfunction!(validate_workgraph_ir, m)?)?;
    m.add_function(wrap_pyfunction!(validate_capture_hierarchy, m)?)?;
    m.add_function(wrap_pyfunction!(validate_physical_flow_plan, m)?)?;
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

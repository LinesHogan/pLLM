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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_workgraph, m)?)?;
    m.add_function(wrap_pyfunction!(validate_workgraph_ir, m)?)?;
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

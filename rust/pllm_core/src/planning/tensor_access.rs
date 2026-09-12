//! Closed tensor access import boundary. No ATen evaluation or scheduling policy.

use super::{aer, semantic_graph::canonical_ascii_json};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum AccessKind { View, Dense, Unknown }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Storage {
    pub device: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StorageView {
    pub name: String,
    pub storage: usize,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
    pub offset: u64,
    pub dtype: String,
}

impl StorageView {
    fn width(&self) -> Result<u64, String> {
        match self.dtype.as_str() {
            "torch.bool" | "torch.uint8" | "torch.int8" => Ok(1),
            "torch.int16" | "torch.float16" | "torch.bfloat16" => Ok(2),
            "torch.int32" | "torch.float32" => Ok(4),
            "torch.int64" | "torch.float64" | "torch.complex64" => Ok(8),
            "torch.complex128" => Ok(16),
            _ => Err("tensor access model needs a supported dtype width".into()),
        }
    }

    pub fn logical_bytes(&self) -> Result<u64, String> {
        self.shape.iter().try_fold(self.width()?, |n, &d| n.checked_mul(d)
            .ok_or_else(|| "tensor logical size overflows u64".into()))
    }

    fn span(&self) -> Result<u64, String> {
        if self.shape.len() != self.stride.len() {
            return Err("tensor view rank and strides differ".into());
        }
        if self.shape.contains(&0) { return Ok(0); }
        let last = self.shape.iter().zip(&self.stride).try_fold(self.offset, |n, (&d, &s)|
            (d - 1).checked_mul(s).and_then(|span| n.checked_add(span)))
            .and_then(|last| last.checked_add(1));
        last.and_then(|n| n.checked_mul(self.width().ok()?))
            .ok_or_else(|| "tensor storage span overflows u64".into())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Access {
    pub node: String,
    pub operator: String,
    pub kind: AccessKind,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct AccessGraph {
    pub source_digest: String,
    pub graph_digest: String,
    pub storages: Vec<Storage>,
    pub values: Vec<StorageView>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub operations: Vec<Access>,
}

pub(super) fn validate(graph: &AccessGraph) -> Result<BTreeMap<&str, &StorageView>, String> {
    for digest in [&graph.source_digest, &graph.graph_digest] {
        if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err("tensor access import requires source and graph digests".into());
        }
    }
    let mut values = BTreeMap::new();
    let mut storage_ids = BTreeSet::new();
    for value in &graph.values {
        if value.name.is_empty() || values.insert(value.name.as_str(), value).is_some() {
            return Err("tensor access values require unique names".into());
        }
        let storage = graph.storages.get(value.storage).ok_or("tensor view references absent storage")?;
        if storage.device.is_empty() || value.span()? > storage.bytes {
            return Err("tensor view is outside its imported storage".into());
        }
        value.logical_bytes()?;
        storage_ids.insert(value.storage);
    }
    if storage_ids.len() != graph.storages.len() {
        return Err("tensor access import has unused storages".into());
    }
    let mut defined = BTreeSet::new();
    let mut allocated = BTreeSet::new();
    for name in &graph.inputs {
        let value = values.get(name.as_str()).ok_or("unknown tensor input")?;
        if !defined.insert(name.as_str()) || !allocated.insert(value.storage) {
            return Err("tensor inputs require distinct storage and names".into());
        }
    }
    for op in &graph.operations {
        if op.operator.is_empty() || op.inputs.iter().collect::<BTreeSet<_>>().len() != op.inputs.len()
            || op.inputs.iter().any(|name| !defined.contains(name.as_str()))
        {
            return Err("tensor access needs unique, already defined operands".into());
        }
        let value = values.get(op.node.as_str()).ok_or("unknown tensor operation result")?;
        if !defined.insert(op.node.as_str()) {
            return Err("tensor value is defined more than once".into());
        }
        let aliases = op.inputs.iter().any(|name| values[name.as_str()].storage == value.storage);
        if (matches!(op.kind, AccessKind::View) && !aliases)
            || (matches!(op.kind, AccessKind::Dense) && allocated.contains(&value.storage))
            || (allocated.contains(&value.storage) && !aliases)
        {
            return Err("tensor access kind contradicts its storage aliases".into());
        }
        allocated.insert(value.storage);
    }
    if defined.len() != values.len() || graph.outputs.iter().any(|name| !defined.contains(name.as_str())) {
        return Err("tensor access values or results are not completely defined".into());
    }
    Ok(values)
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct MemoryStep {
    pub node: String,
    pub live_storages: Vec<usize>,
    pub release_storages: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct DeviceMemory {
    pub device: String,
    pub input_bytes: u64,
    pub result_bytes: u64,
    pub peak_live_tensor_bytes: u64,
    pub peak_new_tensor_bytes: u64,
    pub logical_read_bytes: u64,
    pub logical_write_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct MemoryProfile {
    pub access_digest: String,
    pub devices: Vec<DeviceMemory>,
    pub steps: Vec<MemoryStep>,
    pub unknown_accesses: Vec<String>,
    pub signature: String,
}

#[pyfunction]
fn profile_tensor_memory(graph_json: &str) -> PyResult<String> {
    let run = || -> Result<String, String> {
        let graph: AccessGraph = serde_json::from_str(graph_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&aer::tensor_memory(&graph)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_tensor_memory(graph_json: &str, profile_json: &str) -> PyResult<()> {
    let run = || -> Result<(), String> {
        let graph: AccessGraph = serde_json::from_str(graph_json).map_err(|e| e.to_string())?;
        let supplied: MemoryProfile = serde_json::from_str(profile_json).map_err(|e| e.to_string())?;
        if supplied != aer::tensor_memory(&graph)? {
            return Err("tensor memory witness differs from source access facts".into());
        }
        Ok(())
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(profile_tensor_memory, module)?)?;
    module.add_function(wrap_pyfunction!(validate_tensor_memory, module)?)?;
    Ok(())
}

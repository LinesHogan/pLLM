//! AER composition of tensor payload and original backend retention scopes.

use std::collections::BTreeMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::{aer, region_storage, region_retention};
use super::region_capacity::Capacity;
use super::region_imports::TensorImport;
use super::region_program::Program;
use super::region_retention::{Allocation, Retention};
use super::semantic_graph::{canonical_ascii_json, digest};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Device {
    pub device: String,
    tensor_bytes_upper_bound: u64,
    retained_bytes_upper_bound: u64,
    pub peak_bytes_upper_bound: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Payload {
    pub capacity: Capacity,
    retention: Retention,
    pub devices: Vec<Device>,
    signature: String,
}

pub(super) fn derive(program: &Program, schedule: &str, tensors: &[TensorImport], retained: &[Allocation]) -> Result<Payload> {
    let storage = region_storage::derive(program, schedule, tensors)?;
    let capacity = aer::region_tensor_memory(program, &storage, tensors)?;
    let retention = if retained.is_empty() {
        // Storage/capacity already checked the source and schedule. With no
        // retained allocations there are no endpoint relations to derive.
        region_retention::empty(capacity.program_digest.clone(), capacity.schedule_map.clone())?
    } else { region_retention::derive(program, schedule, retained)? };
    compose(capacity, retention)
}

pub(super) fn compose(capacity: Capacity, retention: Retention) -> Result<Payload> {
    if capacity.program_digest != retention.program_digest || capacity.schedule_map != retention.schedule_map {
        return Err("tensor capacity and backend retention belong to different source schedules".into());
    }
    let mut devices = BTreeMap::new();
    for device in &capacity.devices {
        devices.insert(device.device.clone(), Device { device: device.device.clone(),
            tensor_bytes_upper_bound: device.peak_tensor_bytes_upper_bound,
            retained_bytes_upper_bound: 0, peak_bytes_upper_bound: device.peak_tensor_bytes_upper_bound });
    }
    for allocation in &retention.allocations {
        let device = devices.entry(allocation.device.clone()).or_insert_with(|| Device {
            device: allocation.device.clone(), tensor_bytes_upper_bound: 0,
            retained_bytes_upper_bound: 0, peak_bytes_upper_bound: 0 });
        device.retained_bytes_upper_bound = device.retained_bytes_upper_bound.checked_add(allocation.peak_bytes_upper_bound)
            .ok_or("backend retention sum overflows u64")?;
        device.peak_bytes_upper_bound = device.tensor_bytes_upper_bound.checked_add(device.retained_bytes_upper_bound)
            .ok_or("combined region payload overflows u64")?;
    }
    let mut result = Payload { capacity, retention, devices: devices.into_values().collect(), signature: String::new() };
    result.signature = digest(&canonical_ascii_json(&result)?);
    Ok(result)
}

#[pyfunction]
fn analyze_region_payload(program_json: &str, schedule: &str, tensor_imports_json: &str, retention_imports_json: &str) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let tensors: Vec<TensorImport> = serde_json::from_str(tensor_imports_json).map_err(|e| e.to_string())?;
        let retained: Vec<Allocation> = serde_json::from_str(retention_imports_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(&program, schedule, &tensors, &retained)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_payload(program_json: &str, tensor_imports_json: &str, retention_imports_json: &str, payload_json: &str) -> PyResult<()> {
    let run = || -> Result<()> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let tensors: Vec<TensorImport> = serde_json::from_str(tensor_imports_json).map_err(|e| e.to_string())?;
        let retained: Vec<Allocation> = serde_json::from_str(retention_imports_json).map_err(|e| e.to_string())?;
        let supplied: Payload = serde_json::from_str(payload_json).map_err(|e| e.to_string())?;
        if supplied != derive(&program, &supplied.capacity.schedule_map, &tensors, &retained)? {
            return Err("region payload differs from source tensor and backend retention lifetimes".into());
        }
        Ok(())
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze_region_payload, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_payload, module)?)?;
    Ok(())
}

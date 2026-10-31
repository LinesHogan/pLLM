//! Canonical interning for frozen tensor-layout descriptors.
//!
//! Resource names identify bindings only.  Dense layout IDs are derived from
//! canonical descriptor bytes, so equal layouts share an ID independently of
//! request or graph ordering.

use super::semantic_graph::canonical_ascii_json;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) enum TensorDType {
    #[serde(rename = "bool")]
    Bool,
    #[serde(rename = "uint8")]
    Uint8,
    #[serde(rename = "int32")]
    Int32,
    #[serde(rename = "int64")]
    Int64,
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    Bfloat16,
    #[serde(rename = "float32")]
    Float32,
    #[serde(rename = "float64")]
    Float64,
}

// Layout specs stay closed: their re-serialization is the identity that dense IDs intern, so a field
// Rust did not decode would silently merge distinct layouts.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorFieldSpec {
    pub(crate) field_id: u64,
    pub(crate) shape: Vec<u64>,
    pub(crate) dtype: TensorDType,
    alignment_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorLayoutSpec {
    pub(crate) namespace: String,
    pub(crate) fields: Vec<TensorFieldSpec>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct StateTensorLayout {
    pub(crate) resource: String,
    pub(crate) layout: TensorLayoutSpec,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct TensorLayoutRequest {
    pub(crate) layouts: Vec<StateTensorLayout>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TensorLayoutBinding {
    pub(crate) resource: String,
    pub(crate) layout_id: u32,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TensorLayoutDecision {
    pub(crate) bindings: Vec<TensorLayoutBinding>,
}

fn decode_request(payload: &str) -> Result<TensorLayoutRequest, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid TensorLayoutRequest: {source}"))
}

/// Check the layout facts a user declares on InstanceState/RequestState; field IDs and order are pllm's.
fn validate_request(request: &TensorLayoutRequest) -> Result<(), String> {
    for item in &request.layouts {
        if item.layout.fields.is_empty() {
            return Err(format!(
                "state {:?} declares no tensor fields; give it at least one field",
                item.resource
            ));
        }
        if !(item.layout.fields.iter()).all(|field| field.alignment_bytes.is_power_of_two()) {
            return Err(format!(
                "state {:?}: set alignment= to a power of two",
                item.resource
            ));
        }
    }
    Ok(())
}

fn compile(request: &TensorLayoutRequest) -> Result<TensorLayoutDecision, String> {
    validate_request(request)?;
    let descriptors = request
        .layouts
        .iter()
        .map(|item| canonical_ascii_json(&item.layout))
        .collect::<Result<Vec<_>, String>>()?;
    let distinct: BTreeSet<_> = descriptors.iter().collect();
    let bindings = request
        .layouts
        .iter()
        .zip(&descriptors)
        .map(|(item, descriptor)| TensorLayoutBinding {
            resource: item.resource.clone(),
            layout_id: distinct.iter().position(|d| *d == descriptor).unwrap() as u32,
        })
        .collect();
    Ok(TensorLayoutDecision { bindings })
}

#[pyfunction]
fn intern_tensor_layouts(request_json: &str) -> PyResult<String> {
    let intern = || canonical_ascii_json(&compile(&decode_request(request_json)?)?);
    intern().map_err(PyValueError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(intern_tensor_layouts, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compile, TensorLayoutRequest};

    fn request_json() -> String {
        r#"{"layouts":[{"resource":"alpha","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16}]}},{"resource":"beta","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16}]}},{"resource":"gamma","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[1],"dtype":"float32","alignment_bytes":16},{"field_id":1,"shape":[],"dtype":"float32","alignment_bytes":16}]}}]}"#.to_owned()
    }

    #[test]
    fn equal_descriptors_share_dense_ids() {
        let request: TensorLayoutRequest = serde_json::from_str(&request_json()).unwrap();
        let decision = compile(&request).unwrap();
        let base_request: TensorLayoutRequest = serde_json::from_str(
            r#"{"layouts":[{"resource":"alpha","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16}]}}]}"#,
        )
        .unwrap();
        let base_decision = compile(&base_request).unwrap();
        assert_eq!(
            decision.bindings[0].layout_id,
            decision.bindings[1].layout_id
        );
        assert_ne!(
            decision.bindings[0].layout_id,
            decision.bindings[2].layout_id
        );
        assert_eq!(
            decision
                .bindings
                .iter()
                .map(|binding| binding.layout_id)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1])
        );
        assert_ne!(
            decision.bindings[0].layout_id,
            base_decision.bindings[0].layout_id
        );
    }
}

//! Canonical interning for frozen tensor-layout descriptors.
//!
//! Resource names identify bindings only.  Dense layout IDs are derived from
//! canonical descriptor bytes, so equal layouts share an ID independently of
//! request or graph ordering.

use super::semantic_graph::{canonical_ascii_json, require_text};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const SCHEMA_VERSION: u64 = 1;

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
    schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct StateTensorLayout {
    pub(crate) resource: String,
    pub(crate) layout: TensorLayoutSpec,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorLayoutRequest {
    pub(crate) layouts: Vec<StateTensorLayout>,
    schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorLayoutBinding {
    pub(crate) resource: String,
    pub(crate) layout_id: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorLayoutDecision {
    pub(crate) bindings: Vec<TensorLayoutBinding>,
    schema_version: u64,
}

fn decode_request(payload: &str) -> Result<TensorLayoutRequest, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid TensorLayoutRequest: {source}"))
}

fn validate_layout(layout: &TensorLayoutSpec) -> Result<(), String> {
    if layout.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported tensor layout schema_version: {}",
            layout.schema_version
        ));
    }
    require_text(&layout.namespace, "layout.namespace")?;
    if layout.fields.is_empty() {
        return Err("tensor layout fields must be non-empty".to_owned());
    }
    for (expected_id, field) in layout.fields.iter().enumerate() {
        if field.field_id != expected_id as u64 {
            return Err("tensor layout field IDs must be dense and ordered".to_owned());
        }
        if field.alignment_bytes == 0 || !field.alignment_bytes.is_power_of_two() {
            return Err("tensor field alignment must be a power of two".to_owned());
        }
    }
    Ok(())
}

fn validate_request(request: &TensorLayoutRequest) -> Result<(), String> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported TensorLayoutRequest schema_version: {}",
            request.schema_version
        ));
    }
    for item in &request.layouts {
        require_text(&item.resource, "layouts.resource")?;
        validate_layout(&item.layout)?;
    }
    if request
        .layouts
        .windows(2)
        .any(|pair| pair[0].resource >= pair[1].resource)
    {
        return Err("tensor layout request resources must be sorted unique".to_owned());
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
    Ok(TensorLayoutDecision {
        bindings,
        schema_version: SCHEMA_VERSION,
    })
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
        r#"{"layouts":[{"resource":"alpha","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16}],"schema_version":1}},{"resource":"beta","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16}],"schema_version":1}},{"resource":"gamma","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[1],"dtype":"float32","alignment_bytes":16},{"field_id":1,"shape":[],"dtype":"float32","alignment_bytes":16}],"schema_version":1}}],"schema_version":1}"#.to_owned()
    }

    #[test]
    fn equal_descriptors_share_dense_ids() {
        let request: TensorLayoutRequest = serde_json::from_str(&request_json()).unwrap();
        let decision = compile(&request).unwrap();
        let base_request: TensorLayoutRequest = serde_json::from_str(
            r#"{"layouts":[{"resource":"alpha","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16}],"schema_version":1}}],"schema_version":1}"#,
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

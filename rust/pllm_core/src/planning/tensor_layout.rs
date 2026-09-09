//! Canonical interning for frozen tensor-layout descriptors.
//!
//! Resource names identify bindings only.  Dense layout IDs are derived from
//! canonical descriptor bytes, so equal layouts share an ID independently of
//! request or graph ordering.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write;

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
#[serde(rename_all = "snake_case")]
pub(crate) enum MutationGranularity {
    WholeTensor,
    Row,
    Page,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorFieldSpec {
    pub(crate) field_id: u64,
    shape: Vec<u64>,
    dtype: TensorDType,
    alignment_bytes: u64,
    mutation_granularity: MutationGranularity,
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
    pub(crate) layout_digest: String,
    pub(crate) layout_namespace: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorLayoutDecision {
    request_digest: String,
    pub(crate) bindings: Vec<TensorLayoutBinding>,
    layout_namespace: String,
    signature: String,
    schema_version: u64,
}

fn error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn decode_request(payload: &str) -> Result<TensorLayoutRequest, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid TensorLayoutRequest: {source}"))
}

fn decode_decision(payload: &str) -> Result<TensorLayoutDecision, String> {
    serde_json::from_str(payload)
        .map_err(|source| format!("invalid TensorLayoutDecision: {source}"))
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{label} must be a non-empty string"));
    }
    Ok(())
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
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

// serde_json maps retain lexicographic object-key ordering.  Normalize to the
// ASCII JSON convention used by the Python contract before hashing or output.
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

fn digest(payload: &str) -> String {
    format!("{:x}", Sha256::digest(payload.as_bytes()))
}

fn decision_unsigned(decision: &TensorLayoutDecision) -> TensorLayoutDecision {
    let mut unsigned = decision.clone();
    unsigned.signature.clear();
    unsigned
}

fn decision_signature(decision: &TensorLayoutDecision) -> Result<String, String> {
    Ok(digest(&canonical_ascii_json(&decision_unsigned(decision))?))
}

fn compile(request: &TensorLayoutRequest) -> Result<TensorLayoutDecision, String> {
    validate_request(request)?;
    let mut descriptors = BTreeMap::<String, TensorLayoutSpec>::new();
    let mut resource_descriptors = Vec::with_capacity(request.layouts.len());
    for item in &request.layouts {
        let descriptor = canonical_ascii_json(&item.layout)?;
        descriptors
            .entry(descriptor.clone())
            .or_insert_with(|| item.layout.clone());
        resource_descriptors.push((item.resource.as_str(), descriptor));
    }
    let layout_namespace = digest(&canonical_ascii_json(
        &descriptors.values().cloned().collect::<Vec<_>>(),
    )?);
    let layout_ids = descriptors
        .keys()
        .enumerate()
        .map(|(position, descriptor)| {
            let layout_id = u32::try_from(position)
                .map_err(|_| "canonical tensor layout IDs exhausted u32".to_owned())?;
            Ok((descriptor.as_str(), layout_id))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let bindings = resource_descriptors
        .into_iter()
        .map(|(resource, descriptor)| {
            let layout_id = *layout_ids
                .get(descriptor.as_str())
                .expect("canonical descriptor must have an interned layout ID");
            TensorLayoutBinding {
                resource: resource.to_owned(),
                layout_id,
                layout_digest: digest(&descriptor),
                layout_namespace: layout_namespace.clone(),
            }
        })
        .collect();
    let mut decision = TensorLayoutDecision {
        request_digest: digest(&canonical_ascii_json(request)?),
        bindings,
        layout_namespace,
        signature: String::new(),
        schema_version: SCHEMA_VERSION,
    };
    decision.signature = decision_signature(&decision)?;
    Ok(decision)
}

fn validate_decision_shape(decision: &TensorLayoutDecision) -> Result<(), String> {
    if decision.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported TensorLayoutDecision schema_version: {}",
            decision.schema_version
        ));
    }
    require_digest(&decision.request_digest, "decision.request_digest")?;
    require_digest(&decision.layout_namespace, "decision.layout_namespace")?;
    require_digest(&decision.signature, "decision.signature")?;
    for binding in &decision.bindings {
        require_text(&binding.resource, "bindings.resource")?;
        require_digest(&binding.layout_digest, "bindings.layout_digest")?;
        require_digest(&binding.layout_namespace, "bindings.layout_namespace")?;
        if binding.layout_namespace != decision.layout_namespace {
            return Err("tensor layout binding namespace differs from its decision".to_owned());
        }
    }
    if decision
        .bindings
        .windows(2)
        .any(|pair| pair[0].resource >= pair[1].resource)
    {
        return Err("tensor layout decision bindings must be sorted unique".to_owned());
    }
    Ok(())
}

pub(crate) fn validate_decision(
    request: &TensorLayoutRequest,
    decision: &TensorLayoutDecision,
) -> Result<(), String> {
    validate_decision_shape(decision)?;
    let expected = compile(request)?;
    if decision != &expected {
        return Err("tensor layout decision differs from Rust recomputation".to_owned());
    }
    Ok(())
}

#[pyfunction]
fn intern_tensor_layouts(request_json: &str) -> PyResult<String> {
    let request = decode_request(request_json).map_err(error)?;
    let decision = compile(&request).map_err(error)?;
    canonical_ascii_json(&decision).map_err(error)
}

#[pyfunction]
fn validate_tensor_layouts(request_json: &str, decision_json: &str) -> PyResult<()> {
    let request = decode_request(request_json).map_err(error)?;
    let decision = decode_decision(decision_json).map_err(error)?;
    validate_decision(&request, &decision).map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(intern_tensor_layouts, module)?)?;
    module.add_function(wrap_pyfunction!(validate_tensor_layouts, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compile, validate_decision, TensorLayoutDecision, TensorLayoutRequest};

    fn request_json() -> String {
        r#"{"layouts":[{"resource":"alpha","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16,"mutation_granularity":"row"}],"schema_version":1}},{"resource":"beta","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16,"mutation_granularity":"row"}],"schema_version":1}},{"resource":"gamma","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[1],"dtype":"float32","alignment_bytes":16,"mutation_granularity":"page"},{"field_id":1,"shape":[],"dtype":"float32","alignment_bytes":16,"mutation_granularity":"whole_tensor"}],"schema_version":1}}],"schema_version":1}"#.to_owned()
    }

    #[test]
    fn equal_descriptors_share_dense_ids_and_tampering_is_rejected() {
        let request: TensorLayoutRequest = serde_json::from_str(&request_json()).unwrap();
        let decision = compile(&request).unwrap();
        let base_request: TensorLayoutRequest = serde_json::from_str(
            r#"{"layouts":[{"resource":"alpha","layout":{"namespace":"adapter","fields":[{"field_id":0,"shape":[2,4],"dtype":"float32","alignment_bytes":16,"mutation_granularity":"row"}],"schema_version":1}}],"schema_version":1}"#,
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
        assert_ne!(decision.layout_namespace, base_decision.layout_namespace);
        assert_ne!(
            decision.bindings[0].layout_id,
            base_decision.bindings[0].layout_id
        );
        assert_eq!(
            decision.bindings[0].layout_digest,
            base_decision.bindings[0].layout_digest
        );
        validate_decision(&request, &decision).unwrap();

        let mut tampered: TensorLayoutDecision = decision.clone();
        tampered.bindings[0].layout_id = 99;
        assert!(validate_decision(&request, &tampered).is_err());
    }
}

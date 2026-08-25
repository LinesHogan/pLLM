//! Closed JSON contracts for cold ExecutionRegion shape specialization.
//!
//! The coordinator and worker exchange only this payload.  Tensor values and
//! backend runtime types must never cross this FFI boundary.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Write;

const REQUEST_VERSION: u64 = 1;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RegionShapeSpecializationInput {
    pub(crate) binding_id: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: String,
    pub(crate) device: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RegionShapeSpecializationRequest {
    pub(crate) version: u64,
    pub(crate) base_cache_key: String,
    pub(crate) base_signature: String,
    pub(crate) observed_plan_epoch: u64,
    pub(crate) region_id: String,
    pub(crate) artifact_key: String,
    pub(crate) inputs: Vec<RegionShapeSpecializationInput>,
}

fn error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

pub(crate) fn decode(payload: &str) -> PyResult<RegionShapeSpecializationRequest> {
    serde_json::from_str(payload).map_err(|source| {
        error(format!(
            "invalid RegionShapeSpecializationRequest: {source}"
        ))
    })
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{label} must be a non-empty string"));
    }
    Ok(())
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    require_text(value, label)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

pub(crate) fn validate(request: &RegionShapeSpecializationRequest) -> Result<(), String> {
    if request.version != REQUEST_VERSION {
        return Err(format!(
            "unsupported Region shape specialization version: {}",
            request.version
        ));
    }
    require_digest(&request.base_cache_key, "base_cache_key")?;
    require_digest(&request.base_signature, "base_signature")?;
    if request.observed_plan_epoch == 0 {
        return Err("observed_plan_epoch must be positive".to_owned());
    }
    require_text(&request.region_id, "region_id")?;
    require_digest(&request.artifact_key, "artifact_key")?;
    if request.inputs.is_empty() {
        return Err("inputs must be non-empty".to_owned());
    }

    let mut previous_binding_id: Option<&str> = None;
    for input in &request.inputs {
        require_text(&input.binding_id, "inputs.binding_id")?;
        require_text(&input.dtype, "inputs.dtype")?;
        require_text(&input.device, "inputs.device")?;
        if input.shape.iter().any(|dimension| *dimension < 0) {
            return Err("inputs.shape must contain non-negative integers".to_owned());
        }
        if let Some(previous) = previous_binding_id {
            if previous >= input.binding_id.as_str() {
                return Err("inputs must be uniquely ordered by binding_id".to_owned());
            }
        }
        previous_binding_id = Some(input.binding_id.as_str());
    }
    Ok(())
}

// serde_json intentionally retains Unicode scalars.  The Python wire format
// is ASCII JSON, so normalize output to that shared representation.
fn ascii_json(request: &RegionShapeSpecializationRequest) -> Result<String, String> {
    let value = serde_json::to_value(request)
        .map_err(|source| format!("failed to encode RegionShapeSpecializationRequest: {source}"))?;
    let encoded = serde_json::to_string(&value).map_err(|source| {
        format!("failed to serialize RegionShapeSpecializationRequest: {source}")
    })?;
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

/// Parse and validate the closed JSON contract without changing process state.
#[pyfunction]
fn validate_region_shape_specialization_request(payload: &str) -> PyResult<()> {
    let request = decode(payload)?;
    validate(&request).map_err(error)
}

/// Return the unique ASCII JSON representation of a validated request.
#[pyfunction]
fn normalize_region_shape_specialization_request(payload: &str) -> PyResult<String> {
    let request = decode(payload)?;
    validate(&request).map_err(error)?;
    ascii_json(&request).map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        validate_region_shape_specialization_request,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        normalize_region_shape_specialization_request,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ascii_json, decode, validate};

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn valid_json() -> String {
        format!(
            r#"{{"version":1,"base_cache_key":"{A}","base_signature":"{B}","observed_plan_epoch":4,"region_id":"esamp.forward","artifact_key":"{C}","inputs":[{{"binding_id":"hidden","shape":[8,896],"dtype":"torch.bfloat16","device":"cuda:0"}}]}}"#
        )
    }

    #[test]
    fn specialization_request_normalizes_to_canonical_ascii_json() {
        let request = decode(&valid_json()).unwrap();
        validate(&request).unwrap();
        assert_eq!(
            ascii_json(&request).unwrap(),
            format!(
                r#"{{"artifact_key":"{C}","base_cache_key":"{A}","base_signature":"{B}","inputs":[{{"binding_id":"hidden","device":"cuda:0","dtype":"torch.bfloat16","shape":[8,896]}}],"observed_plan_epoch":4,"region_id":"esamp.forward","version":1}}"#
            )
        );
    }

    #[test]
    fn specialization_request_rejects_unknown_fields() {
        let payload = valid_json().replace("\"version\":1", "\"version\":1,\"unknown\":true");
        assert!(decode(&payload).is_err());
    }

    #[test]
    fn specialization_request_rejects_invalid_closed_values() {
        let cases = [
            valid_json().replace("\"observed_plan_epoch\":4", "\"observed_plan_epoch\":0"),
            valid_json().replace("\"shape\":[8,896]", "\"shape\":[8,-1]"),
            valid_json().replace("\"region_id\":\"esamp.forward\"", "\"region_id\":\"\""),
            valid_json().replace(
                A,
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            ),
            valid_json().replace("\"binding_id\":\"hidden\"", "\"binding_id\":\"\""),
        ];
        for payload in cases {
            let request = decode(&payload).unwrap();
            assert!(
                validate(&request).is_err(),
                "payload should be rejected: {payload}"
            );
        }
    }

    #[test]
    fn specialization_request_rejects_unsorted_or_duplicate_bindings() {
        let payload = valid_json().replace(
            "]}",
            ",{\"binding_id\":\"alpha\",\"shape\":[],\"dtype\":\"torch.float32\",\"device\":\"cpu\"}]}",
        );
        let request = decode(&payload).unwrap();
        assert!(validate(&request).is_err());
    }
}

//! Python boundary for closed backend attachment compiler inputs and outputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::super::semantic_graph::canonical_ascii_json;
use super::signing::{compile, lower, validate_decision, validate_physical};
use super::types::{
    AttachmentCompilerInputs, BackendAttachmentDecision, BackendAttachmentPhysicalPlan,
};

fn error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn decode<T: DeserializeOwned>(payload: &str) -> PyResult<T> {
    serde_json::from_str(payload).map_err(|source| error(source.to_string()))
}

fn encode<T: Serialize>(value: &T) -> PyResult<String> {
    canonical_ascii_json(value).map_err(error)
}

#[pyfunction]
fn compile_backend_attachment(inputs_json: &str) -> PyResult<String> {
    let inputs = decode::<AttachmentCompilerInputs>(inputs_json)?;
    encode(&compile(&inputs).map_err(error)?)
}

#[pyfunction]
fn validate_backend_attachment(inputs_json: &str, decision_json: &str) -> PyResult<()> {
    let inputs = decode::<AttachmentCompilerInputs>(inputs_json)?;
    let decision = decode::<BackendAttachmentDecision>(decision_json)?;
    validate_decision(&inputs, &decision).map_err(error)
}

#[pyfunction]
fn lower_backend_attachment(inputs_json: &str, decision_json: &str) -> PyResult<String> {
    let inputs = decode::<AttachmentCompilerInputs>(inputs_json)?;
    let decision = decode::<BackendAttachmentDecision>(decision_json)?;
    encode(&lower(&inputs, &decision).map_err(error)?)
}

#[pyfunction]
fn validate_backend_attachment_physical(
    inputs_json: &str,
    decision_json: &str,
    physical_json: &str,
) -> PyResult<()> {
    let inputs = decode::<AttachmentCompilerInputs>(inputs_json)?;
    let decision = decode::<BackendAttachmentDecision>(decision_json)?;
    let physical = decode::<BackendAttachmentPhysicalPlan>(physical_json)?;
    validate_physical(&inputs, &decision, &physical).map_err(error)
}

#[pyfunction]
fn validate_backend_attachment_reconstruction(
    inputs_json: &str,
    decision_json: &str,
    physical_json: &str,
    local_inputs_json: &str,
    local_physical_json: &str,
) -> PyResult<()> {
    let inputs = decode::<AttachmentCompilerInputs>(inputs_json)?;
    let decision = decode::<BackendAttachmentDecision>(decision_json)?;
    let physical = decode::<BackendAttachmentPhysicalPlan>(physical_json)?;
    let local_inputs = decode::<AttachmentCompilerInputs>(local_inputs_json)?;
    let local_physical = decode::<BackendAttachmentPhysicalPlan>(local_physical_json)?;
    validate_physical(&inputs, &decision, &physical).map_err(error)?;
    validate_physical(&local_inputs, &decision, &local_physical).map_err(error)?;
    if local_inputs != inputs || local_physical != physical {
        return Err(error(
            "local backend attachment differs from parent-signed facts",
        ));
    }
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_backend_attachment, module)?)?;
    module.add_function(wrap_pyfunction!(validate_backend_attachment, module)?)?;
    module.add_function(wrap_pyfunction!(lower_backend_attachment, module)?)?;
    module.add_function(wrap_pyfunction!(
        validate_backend_attachment_physical,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_backend_attachment_reconstruction,
        module
    )?)?;
    Ok(())
}

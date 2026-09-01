use super::super::semantic_graph::canonical_ascii_json;
use super::signing::{compile, lower, validate_decision, validate_physical};
use super::types::{Decision, Inputs, Physical};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{de::DeserializeOwned, Serialize};
fn decode<T: DeserializeOwned>(value: &str) -> PyResult<T> {
    serde_json::from_str(value).map_err(|error| PyValueError::new_err(error.to_string()))
}
fn encode<T: Serialize>(value: &T) -> PyResult<String> {
    canonical_ascii_json(value).map_err(PyValueError::new_err)
}
#[pyfunction]
fn compile_parameter_overlay_attachment(inputs_json: &str) -> PyResult<String> {
    encode(&compile(&decode::<Inputs>(inputs_json)?).map_err(PyValueError::new_err)?)
}
#[pyfunction]
fn validate_parameter_overlay_attachment(inputs_json: &str, decision_json: &str) -> PyResult<()> {
    validate_decision(
        &decode::<Inputs>(inputs_json)?,
        &decode::<Decision>(decision_json)?,
    )
    .map_err(PyValueError::new_err)
}
#[pyfunction]
fn lower_parameter_overlay_attachment(inputs_json: &str, decision_json: &str) -> PyResult<String> {
    encode(
        &lower(
            &decode::<Inputs>(inputs_json)?,
            &decode::<Decision>(decision_json)?,
        )
        .map_err(PyValueError::new_err)?,
    )
}
#[pyfunction]
fn validate_parameter_overlay_attachment_reconstruction(
    inputs_json: &str,
    decision_json: &str,
    physical_json: &str,
    local_inputs_json: &str,
    local_physical_json: &str,
) -> PyResult<()> {
    let inputs = decode::<Inputs>(inputs_json)?;
    let decision = decode::<Decision>(decision_json)?;
    let physical = decode::<Physical>(physical_json)?;
    let local_inputs = decode::<Inputs>(local_inputs_json)?;
    let local_physical = decode::<Physical>(local_physical_json)?;
    validate_physical(&inputs, &decision, &physical).map_err(PyValueError::new_err)?;
    validate_physical(&local_inputs, &decision, &local_physical).map_err(PyValueError::new_err)?;
    if inputs != local_inputs || physical != local_physical {
        return Err(PyValueError::new_err(
            "local parameter-overlay facts differ from parent signature",
        ));
    }
    Ok(())
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        compile_parameter_overlay_attachment,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_parameter_overlay_attachment,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        lower_parameter_overlay_attachment,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        validate_parameter_overlay_attachment_reconstruction,
        module
    )?)?;
    Ok(())
}

//! Python boundary for checking closed backend attachment compiler inputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::types::AttachmentCompilerInputs;
use super::validation::validate_inputs;

#[pyfunction]
fn check_backend_attachment(inputs_json: &str) -> PyResult<()> {
    let inputs: AttachmentCompilerInputs = serde_json::from_str(inputs_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    validate_inputs(&inputs).map_err(PyValueError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(check_backend_attachment, module)?)?;
    Ok(())
}

//! Python boundary for checking closed backend attachment facts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::types::BackendAttachment;
use super::validation::validate_attachment;

#[pyfunction]
fn check_backend_attachment(attachment_json: &str) -> PyResult<()> {
    let attachment: BackendAttachment = serde_json::from_str(attachment_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    validate_attachment(&attachment).map_err(PyValueError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(check_backend_attachment, module)?)?;
    Ok(())
}

//! pllm._core: pure-data planning and validation for PLLM.
//!
//! Tensor, CUDA, IPC, and plugin instances remain in the Python runtime.

use pyo3::prelude::*;

mod planning;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    planning::register(m)?;
    planning::scheduler::register(m)?;
    planning::tensor_layout::register(m)?;
    Ok(())
}

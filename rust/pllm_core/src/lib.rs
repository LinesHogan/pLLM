//! pllm._core: planning, validation and packed-row preparation for PLLM.
//!
//! Tensor, CUDA, IPC, and plugin instances remain in the Python runtime.

use pyo3::prelude::*;

mod packed_rows;
mod planning;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    packed_rows::register(m)?;
    planning::register(m)?;
    planning::scheduler::register(m)?;
    planning::tensor_layout::register(m)?;
    Ok(())
}

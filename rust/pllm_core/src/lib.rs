//! pllm._core: pure-data planning and validation for PLLM.
//!
//! Tensor, CUDA, IPC, and plugin instances remain in the Python runtime.

use pyo3::prelude::*;

mod planning;
mod prefill_row_plan;
mod workload_frontend;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    planning::register(m)?;
    planning::backend_attachment::register(m)?;
    planning::scheduler::register(m)?;
    planning::tensor_layout::register(m)?;
    prefill_row_plan::register(m)?;
    workload_frontend::register(m)?;
    Ok(())
}

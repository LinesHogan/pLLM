//! tllm._core: pure-data planning and validation for tLLM.
//!
//! Tensor, CUDA, IPC, and plugin instances remain in the Python runtime.

use pyo3::prelude::*;

mod canonical;
mod prefill_row_plan;
mod workload_frontend;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    canonical::backend_attachment::register(m)?;
    canonical::scheduler::register(m)?;
    canonical::state_representation::register(m)?;
    canonical::tensor_layout::register(m)?;
    prefill_row_plan::register(m)?;
    workload_frontend::register(m)?;
    Ok(())
}

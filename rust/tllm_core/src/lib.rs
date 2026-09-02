//! tllm._core: pure-data planning and validation for tLLM.
//!
//! Tensor, CUDA, IPC, and plugin instances remain in the Python runtime.

use pyo3::prelude::*;

mod canonical;
mod decode_localization;
mod execution_plan;
mod placement;
mod prefill_row_plan;
mod recurrence;
mod sampler_plan;
mod schedule;
mod shape_specialization;
mod workload_frontend;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    canonical::backend_attachment::register(m)?;
    canonical::scheduler::register(m)?;
    canonical::state_representation::register(m)?;
    canonical::tensor_layout::register(m)?;
    decode_localization::register(m)?;
    sampler_plan::register(m)?;
    shape_specialization::register(m)?;
    execution_plan::register(m)?;
    placement::register(m)?;
    prefill_row_plan::register(m)?;
    recurrence::register(m)?;
    schedule::register(m)?;
    workload_frontend::register(m)?;
    Ok(())
}

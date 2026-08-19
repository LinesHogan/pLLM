//! tllm._core: pure-data planning and validation for tLLM.
//!
//! Tensor, CUDA, IPC, and plugin instances remain in the Python runtime.

use pyo3::prelude::*;

mod decode_localization;
mod execution_plan;
mod placement;
mod sampler_plan;
mod workgraph;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    decode_localization::register(m)?;
    sampler_plan::register(m)?;
    execution_plan::register(m)?;
    placement::register(m)?;
    workgraph::register(m)?;
    Ok(())
}

//! Canonical closed planning contracts shared across Python and Rust.

use pyo3::prelude::*;

pub(crate) mod scheduler;
pub(crate) mod semantic_graph;
pub(crate) mod state_representation;
pub(crate) mod tensor_layout;
mod tensor_policy;
mod version_hazards;

#[pyfunction]
fn compile_state_schedule(
    graph_json: &str,
    timeline_json: &str,
    request_json: &str,
) -> PyResult<(String, String)> {
    let compile = || -> Result<_, String> {
        let graph = semantic_graph::decode_graph(graph_json)?;
        let timeline = semantic_graph::decode_timeline(timeline_json)?;
        let request = state_representation::decode_request(request_json)?;
        let (decision, schedule) = state_representation::compile(&request, &graph, &timeline)?;
        Ok((
            semantic_graph::canonical_ascii_json(&decision)?,
            semantic_graph::canonical_ascii_json(&schedule)?,
        ))
    };
    compile().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_state_schedule, module)?)?;
    Ok(())
}

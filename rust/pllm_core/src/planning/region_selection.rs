//! Closed measured-cost inputs and decisions for the original AER.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use super::{aer, region_payload::Payload, region_retention::Allocation, region_imports::TensorImport, region_program::Program};
use super::semantic_graph::canonical_ascii_json;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct PayloadLimit { pub device: String, pub bytes: u64 }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Candidate {
    pub schedule_map: String,
    pub imports: Vec<TensorImport>,
    pub elapsed_ns: Vec<u64>,
    #[serde(default)]
    pub outlines: Vec<super::region_outlining::CallOutlineEvidence>,
    #[serde(default)]
    pub stream_plan: Option<super::region_streams::StreamPlan>,
    #[serde(default)]
    pub affine_scan: Option<super::affine_scan::Evidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct TuningSize { pub name: String, pub value: u64 }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub source_imports: Vec<TensorImport>,
    pub candidates: Vec<Candidate>,
    pub payload_limits: Vec<PayloadLimit>,
    pub allow_numerical_transforms: bool,
    pub measurement_context: String,
    #[serde(default)]
    pub tuning_sizes: Vec<TuningSize>,
    #[serde(default)]
    pub retention_imports: Vec<Allocation>,
}

pub(super) fn validate_tuning_sizes(program: &Program, request: &Request) -> Result<(), String> {
    if request.tuning_sizes.len() != program.parameters.len()
        || request.tuning_sizes.windows(2).any(|p| p[0].name >= p[1].name)
    {
        return Err("region tuning sizes must be complete and sorted by unique parameter name".into());
    }
    for binding in &request.tuning_sizes {
        let parameter = program.parameters.iter().find(|p| p.name == binding.name)
            .ok_or("region tuning size has no source parameter")?;
        if binding.value < parameter.minimum || binding.value > parameter.maximum {
            return Err("region tuning size is outside its declared bounds".into());
        }
    }
    for constraint in &program.constraints {
        let mut value = 0_i128;
        for term in &constraint.terms {
            let size = request.tuning_sizes.iter().find(|s| s.name == term.parameter)
                .ok_or("size constraint has no tuning binding")?;
            value = value.checked_add(i128::from(term.coefficient) * i128::from(size.value))
                .ok_or("tuning size constraint overflows i128")?;
        }
        if value > i128::from(constraint.upper) {
            return Err("region tuning sizes violate a declared relation".into());
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Assessment {
    pub payload: Payload,
    pub peak_bytes: Vec<PayloadLimit>,
    pub median_ns: Option<u64>,
    pub feasible: bool,
    pub changed_functions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Selection {
    pub program_digest: String,
    pub request_digest: String,
    pub candidates: Vec<Assessment>,
    pub selected: usize,
    pub signature: String,
}

#[pyfunction]
fn assess_region_candidates(program_json: &str, request_json: &str) -> PyResult<String> {
    let run = || -> Result<String, String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let request: Request = serde_json::from_str(request_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&aer::assess_region_candidates(&program, &request)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn select_region_candidate(program_json: &str, request_json: &str) -> PyResult<String> {
    let run = || -> Result<String, String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let request: Request = serde_json::from_str(request_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&aer::select_region_candidate(&program, &request)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_selection(program_json: &str, request_json: &str, selection_json: &str) -> PyResult<()> {
    let run = || -> Result<(), String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let request: Request = serde_json::from_str(request_json).map_err(|e| e.to_string())?;
        let supplied: Selection = serde_json::from_str(selection_json).map_err(|e| e.to_string())?;
        if supplied != aer::select_region_candidate(&program, &request)? {
            return Err("region selection differs from source, imports, policy or measured costs".into());
        }
        Ok(())
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(assess_region_candidates, module)?)?;
    module.add_function(wrap_pyfunction!(select_region_candidate, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_selection, module)?)?;
    Ok(())
}

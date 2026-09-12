//! Source-to-source call extraction. No callable or claimed replacement body
//! is trusted; both the frontier and every surviving SSA result are derived.

use std::collections::{BTreeMap, BTreeSet};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use super::region_program::{self, Block, Function, Operation, Program, Region, Type, Value};
use super::semantic_graph::{canonical_ascii_json, digest};
use super::region_imports::TensorImport;
use super::tensor_access::AccessGraph;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CallOutline {
    pub region_path: Vec<u32>,
    pub first: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CallOutlineEvidence {
    pub site: CallOutline,
    pub access: AccessGraph,
}

pub(super) fn apply_evidence(program: &Program, imports: &[TensorImport], evidence: &[CallOutlineEvidence])
    -> Result<(Program, Vec<TensorImport>), String>
{
    let mut current = program.clone();
    let mut baseline = imports.to_vec();
    for record in evidence {
        let outlined = derive(&current, &record.site)?;
        let import = TensorImport::Exported { access: record.access.clone() };
        if import.name() != outlined.function.name {
            return Err("outlined tensor import has a different source identity".into());
        }
        import.validate(&outlined.function)?;
        baseline.push(import);
        current = outlined.program;
    }
    baseline.sort_by(|a, b| a.name().cmp(b.name()));
    Ok((current, baseline))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct OutlinedProgram {
    pub program: Program,
    pub function: Function,
    pub body: Block,
}

fn target<'a>(region: &'a mut Region, path: &[u32]) -> Result<&'a mut Block, String> {
    if path.is_empty() { return Ok(&mut region.block); }
    if path.len() % 2 != 0 {
        return Err("outline region path must contain operation/child pairs".into());
    }
    let op = region.block.operations.get_mut(path[0] as usize)
        .ok_or("outline region path is outside the source block")?;
    let child = match op {
        Operation::Scan { body, .. } | Operation::Map { body, .. } if path[1] == 0 => body,
        Operation::If { then_region, .. } if path[1] == 0 => then_region,
        Operation::If { else_region, .. } if path[1] == 1 => else_region,
        Operation::While { condition, .. } if path[1] == 0 => condition,
        Operation::While { body, .. } if path[1] == 1 => body,
        _ => return Err("outline path does not identify a source child region".into()),
    };
    target(child, &path[2..])
}

pub(super) fn derive(program: &Program, site: &CallOutline) -> Result<OutlinedProgram, String> {
    region_program::verify(program)?;
    if site.count < 2 {
        return Err("outlining needs at least two adjacent calls".into());
    }
    let name = digest(&canonical_ascii_json(&("region-call-outline@1", program, site))?);
    let functions: BTreeMap<_, _> = program.functions.iter().map(|f| (f.name.as_str(), f)).collect();
    if functions.contains_key(name.as_str()) {
        return Err("outlined function identity must be new".into());
    }
    let mut result = program.clone();
    let block = target(&mut result.entry, &site.region_path)?;
    let first = site.first as usize;
    let end = first.checked_add(site.count as usize).ok_or("outline range overflows")?;
    if end > block.operations.len() {
        return Err("outline range is outside the source block".into());
    }
    let calls = &block.operations[first..end];
    let available: BTreeMap<_, _> = block.arguments.iter()
        .chain(block.operations[..first].iter().flat_map(|op| op.ports().1))
        .map(|v| (v.name.as_str(), v)).collect();
    let mut produced = BTreeSet::new();
    let mut frontier_names = BTreeSet::new();
    let mut frontier = Vec::new();
    let mut values: Vec<Value> = Vec::new();
    for op in calls {
        let Operation::Call { inputs, outputs, function } = op else {
            return Err("outline range must contain only tensor calls".into());
        };
        let declaration = functions[function.as_str()];
        if !declaration.resources.is_empty()
            || declaration.inputs.iter().chain(&declaration.outputs).any(|t| !matches!(t, Type::Tensor { .. }))
        {
            return Err("outlining requires pure all-tensor function declarations".into());
        }
        for input in inputs {
            if !produced.contains(input.as_str()) && frontier_names.insert(input.as_str()) {
                frontier.push(available[input.as_str()].clone());
            }
        }
        for output in outputs {
            produced.insert(output.name.as_str());
            values.push(output.clone());
        }
    }
    let used: BTreeSet<_> = block.yields.iter()
        .chain(block.operations[end..].iter().flat_map(|op| op.ports().0)).collect();
    let outputs: Vec<_> = values.into_iter().filter(|v| used.contains(&v.name)).collect();
    let body = Block { arguments: frontier.clone(), operations: calls.to_vec(),
        yields: outputs.iter().map(|v| v.name.clone()).collect() };
    let function = Function { name: name.clone(), inputs: frontier.iter().map(|v| v.type_.clone()).collect(),
        outputs: outputs.iter().map(|v| v.type_.clone()).collect(), resources: vec![], accesses: vec![] };
    let call = Operation::Call { inputs: frontier.into_iter().map(|v| v.name).collect(), outputs, function: name };
    block.operations.splice(first..end, [call]);
    result.functions.push(function.clone());
    region_program::verify(&result)?;
    Ok(OutlinedProgram { program: result, function, body })
}

#[pyfunction]
fn outline_region_program(program_json: &str, outline_json: &str) -> PyResult<String> {
    let run = || -> Result<String, String> {
        let program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let site = serde_json::from_str(outline_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(&program, &site)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_outlining(program_json: &str, outline_json: &str, outlined_json: &str) -> PyResult<()> {
    let run = || -> Result<(), String> {
        let program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let site = serde_json::from_str(outline_json).map_err(|e| e.to_string())?;
        let supplied: OutlinedProgram = serde_json::from_str(outlined_json).map_err(|e| e.to_string())?;
        if supplied != derive(&program, &site)? {
            return Err("outlined program differs from original calls, frontier or escaping values".into());
        }
        Ok(())
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(outline_region_program, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_outlining, module)?)?;
    Ok(())
}

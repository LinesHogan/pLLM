//! Closed scan windows: source closure, bounded placement and engine admission facts.

use std::collections::BTreeSet;
use isl_rs::UnionMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_dependence::{analyze, context, isl, Result};
use super::region_imports::{native_copy_bytes, TensorImport};
use super::region_payload::{self, Payload};
use super::region_program::{Extent, Operation, Program, Resource};
use super::region_selection::{Policy, SchedulePolicy};
use super::region_streams::{self, StateWindow, StreamPlan};
use super::semantic_graph::{canonical_ascii_json, digest};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub counts: Vec<String>,
    pub quantum: u64,
    pub token_budget: u64,
    pub visibility: super::visibility::StateVisibility,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Proposal {
    pub payload: Payload,
    pub stream_plan: Option<StreamPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct ClosedWindow {
    pub source_digest: String,
    pub request: Request,
    pub chunks: u64,
    pub program: Program,
    pub proposal: Proposal,
    pub state_windows: Vec<StateWindow>,
    pub workspace_bytes: u64,
    pub diagnostics: Vec<String>,
    pub signature: String,
}

fn check_closure(program: &Program, request: &Request) -> Result<String> {
    if request.quantum == 0 || request.counts.is_empty() || request.token_budget < request.quantum
        || request.counts.iter().collect::<BTreeSet<_>>().len() != request.counts.len() {
        return Err("increase the engine token budget to fit a complete logical chunk".into());
    }
    let arguments: BTreeSet<_> = program.entry.block.arguments.iter().map(|a| &a.name).collect();
    let mut counts = BTreeSet::new();
    for operation in &program.entry.block.operations {
        let Operation::Scan { inputs, count: Extent::Parameter(name), invariants, .. } = operation else {
            return Err("closed engine windows require outer scans with explicit input banks".into());
        };
        if *invariants != inputs.len() || inputs.iter().any(|input| !arguments.contains(input)) {
            return Err("closed engine cuts cannot retain SSA carry between scans or chunks".into());
        }
        counts.insert(name);
    }
    if request.counts.iter().any(|name| !counts.contains(name)) {
        return Err("engine window counts must name original outer scans".into());
    }
    let source = analyze(program, false)?;
    if program.functions.iter().any(|function| function.resources.iter().any(|name|
        program.resources.iter().any(|resource| matches!(resource, Resource::State { name: state, .. } if state == name)))) {
        return Err("expose state accesses as source snapshots and publications before closing engine windows".into());
    }
    let ctx = context()?;
    let pieces = source.statements.iter().map(|statement| {
        let axes = (0..statement.extents.len()).map(|i| format!("i{i}")).collect::<Vec<_>>();
        if axes.is_empty() { return Err("closed window statement is outside a source scan".into()); }
        Ok(format!("S{}[{}] -> U[{}, i0]", statement.id, axes.join(","), statement.path[0]))
    }).collect::<Result<Vec<_>>>()?;
    let units = isl(UnionMap::read_from_str(&ctx, &format!("{{ {} }}", pieces.join(";"))))?;
    let dependencies = isl(UnionMap::read_from_str(&ctx, &source.dependencies))?;
    let unit_edges = isl(isl(isl(isl(units.copy())?.reverse())?.apply_range(dependencies))?.apply_range(units))?;
    let permitted = isl(UnionMap::read_from_str(&ctx, "{ U[s,c] -> U[t,d] : s=t and c<=d }"))?;
    if !isl(unit_edges.is_subset(&permitted))? {
        return Err("closed engine cuts need independent scans and forward chunk dependencies".into());
    }
    Ok(source.program_digest)
}

pub(super) fn derive(program: &Program, imports: &[TensorImport], request: &Request,
                     policy: &Policy, parallel: bool, budget: Option<u64>) -> Result<ClosedWindow> {
    super::visibility::validate(&request.visibility)?;
    if request.visibility.source.scope != super::visibility::Scope::ModelScan
        || request.visibility.source.chunk_size != request.quantum
        || request.visibility.source.resources.iter().any(|name| !program.resources.iter().any(|resource|
            matches!(resource, Resource::State { name: declared, .. } if declared == name))) {
        return Err("closed source visibility must match its target resources and chunk domain".into());
    }
    let source_digest = check_closure(program, request).map_err(|e| format!("closed source: {e}"))?;
    let maximum = request.counts.iter().map(|name| program.parameters.iter()
        .find(|p| &p.name == name).map(|p| p.maximum).ok_or_else(|| "missing window count parameter".to_owned()))
        .collect::<Result<Vec<_>>>()?.into_iter().min().unwrap().min(request.token_budget / request.quantum);
    let mut sizes = Vec::new();
    for size in [4, 2, 1] {
        let size = size.min(maximum);
        if size > 0 && !sizes.contains(&size) { sizes.push(size); }
    }
    let auto = policy.schedule_policy == SchedulePolicy::Auto;
    let arrangements: &[bool] = if auto && parallel { &[true, false] } else { &[false] };
    let mut diagnostics = Vec::new();
    for &streams in arrangements {
        for &chunks in &sizes {
            let mut window = program.clone();
            for parameter in &mut window.parameters {
                if request.counts.contains(&parameter.name) { parameter.maximum = chunks; }
            }
            let schedule = if auto {
                super::region_tiling::frontier(&window, 4096).map_err(|e| format!("window frontier: {e}"))?.schedule_map
            } else { analyze(&window, false)?.reference_schedule };
            let stream_plan = if streams {
                match region_streams::derive(&window, &schedule, 1, 2, 2, imports, &[]) {
                    Ok(plan) => Some(plan),
                    Err(error) => { diagnostics.push(format!("{chunks}-chunk two-stream candidate: {error}")); continue; },
                }
            } else { None };
            let payload = match &stream_plan {
                Some(plan) => plan.payload.clone(),
                None => region_payload::derive(&window, &schedule, imports, &[])?,
            };
            let peaks = stream_plan.as_ref().map_or_else(
                || payload.devices.iter().map(|d| d.peak_bytes_upper_bound).collect::<Vec<_>>(),
                |plan| plan.devices.iter().map(|d| d.peak_bytes_upper_bound).collect());
            let peak = peaks.into_iter().try_fold(0_u64, |n, bytes| n.checked_add(bytes))
                .ok_or("window device bytes overflow")?;
            // Original embedding/padded-input preparation and final concatenation
            // live outside the shared frame, but within this task's reservation.
            let preparation = payload.capacity.devices.iter().try_fold(0_u64, |n, d|
                n.checked_add(d.entry_copy_bytes_upper_bound.checked_mul(2)?)).ok_or("window input bytes overflow")?;
            let results = super::region_program::verify(&window)?.iter().try_fold(0_u64, |n, t| {
                n.checked_add(native_copy_bytes(t, &window.parameters).ok()?.map_or(0, |(_, bytes)| bytes))
            }).ok_or("window output bytes overflow")?;
            let workspace_bytes = peak.checked_add(preparation).and_then(|n| n.checked_add(results))
                .ok_or("closed window bytes overflow")?;
            if budget.is_some_and(|limit| workspace_bytes > limit) {
                diagnostics.push(format!("{chunks}-chunk candidate needs {workspace_bytes} workspace bytes"));
                continue;
            }
            let analysis = analyze(&window, false)?;
            let actions = super::region_control::statements(&window, &analysis.statements)?;
            let state_windows = program.resources.iter().filter_map(|resource| match resource {
                Resource::State { name, .. } => Some(StateWindow { resource: name.clone(),
                    private_roots: if actions.iter().any(|action| region_streams::publishes(action, name)) { 2 } else { 0 } }),
                _ => None,
            }).collect();
            let mut result = ClosedWindow { source_digest: source_digest.clone(), request: request.clone(), chunks,
                program: window, proposal: Proposal { payload, stream_plan }, state_windows, workspace_bytes,
                diagnostics, signature: String::new() };
            result.signature = digest(&canonical_ascii_json(&result)?);
            return Ok(result);
        }
    }
    Err("increase the memory budget to fit a complete closed engine window".into())
}

#[pyfunction]
#[pyo3(signature = (program_json, imports_json, request_json, policy_json, parallel, budget=None))]
fn plan_closed_region_window(program_json: &str, imports_json: &str, request_json: &str,
                             policy_json: &str, parallel: bool, budget: Option<u64>) -> PyResult<String> {
    let run = || -> Result<String> {
        let program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let request = serde_json::from_str(request_json).map_err(|e| e.to_string())?;
        let policy = serde_json::from_str(policy_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(&program, &imports, &request, &policy, parallel, budget)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(plan_closed_region_window, module)?)?;
    Ok(())
}

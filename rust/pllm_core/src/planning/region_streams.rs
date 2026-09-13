//! Bounded concurrent phases, derived from a verified source schedule.
//! Placement and completion are physical products, never source annotations.

use isl_rs::{DimType, UnionMap, UnionSet};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::VecDeque;

use super::region_bindings::{tables, AccessTable};
use super::region_control::{self, Action};
use super::region_dependence::{analyze, canonical_schedule, context, isl, legal, map_text, read_schedule, Result};
use super::region_imports::TensorImport;
use super::region_payload::{self, Payload};
use super::region_program::{Operation, Program, Resource};
use super::region_retention::Allocation;
use super::semantic_graph::{canonical_ascii_json, digest};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StreamDevice {
    pub device: String,
    concurrent_scratch_bytes_upper_bound: u64,
    pending_phase_bytes_upper_bound: u64,
    pub peak_bytes_upper_bound: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StreamWaits {
    lane: u32,
    predecessors: Vec<AccessTable>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StreamPlan {
    version: u32,
    program_digest: String,
    schedule_map: String,
    axis: usize,
    lanes: u32,
    pipeline_depth: u32,
    pub phase_schedule: String,
    placements: Vec<AccessTable>,
    waits: Vec<StreamWaits>,
    host_fences: Vec<usize>,
    // The phase payload covers live values and backend retention. Its ordinary
    // per-call scratch bound is multiplied below for concurrently active lanes.
    pub payload: Payload,
    pub devices: Vec<StreamDevice>,
    signature: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Exploration {
    program_digest: String,
    candidates: Vec<StreamPlan>,
    diagnostics: Vec<String>,
}

fn concurrent_action(action: &Action, program: &Program, imports: &[TensorImport], retained: &[Allocation]) -> bool {
    match action {
        Action::Primitive { operation: Operation::Call { function, .. } } => {
            // Acquiring/releasing backend retention is a host ownership event,
            // even when the endpoint also has a device-kernel implementation.
            !retained.iter().any(|allocation| allocation.is_endpoint(function))
            && imports.iter().any(|import| import.name() == function && match import {
                TensorImport::Exported { .. } | TensorImport::Compiled { .. }
                    | TensorImport::FunctionalKernel { .. } | TensorImport::Borrowed { .. } => true,
                TensorImport::NativeKernel { declaration, .. } => declaration.resources.iter().all(|name|
                    program.resources.iter().any(|r| matches!(r, Resource::Borrowed { name: known, .. } if known == name))
                    && declaration.accesses.iter().any(|a| &a.resource == name)),
                TensorImport::Native { .. } => false,
            })
        },
        Action::Primitive { operation } => matches!(operation, Operation::Constant { .. } | Operation::Parameter { .. }
            | Operation::Coordinate { .. } | Operation::Scalar { .. } | Operation::Select { .. } | Operation::Update { .. }),
        Action::If { then_block, else_block } => then_block.steps.iter().chain(&else_block.steps)
            .all(|step| concurrent_action(&step.action, program, imports, retained)),
        Action::While { condition, body, .. } => condition.steps.iter().chain(&body.steps)
            .all(|step| concurrent_action(&step.action, program, imports, retained)),
        Action::Scan { body, .. } | Action::Map { body, .. } => body.steps.iter()
            .all(|step| concurrent_action(&step.action, program, imports, retained)),
    }
}

fn derive(program: &Program, schedule: &str, axis: usize, lanes: u32, pipeline_depth: u32,
          imports: &[TensorImport], retained: &[Allocation]) -> Result<StreamPlan> {
    thread_local! {
        static PLANS: RefCell<VecDeque<(String, StreamPlan)>> = const { RefCell::new(VecDeque::new()) };
    }
    let schedule = canonical_schedule(schedule)?;
    let key = digest(&canonical_ascii_json(&(program, &schedule, axis, lanes, pipeline_depth, imports, retained))?);
    super::region_cache::memoized(&PLANS, key, || derive_plan(program, &schedule, axis, lanes, pipeline_depth, imports, retained))
}

fn derive_plan(program: &Program, schedule: &str, axis: usize, lanes: u32, pipeline_depth: u32,
               imports: &[TensorImport], retained: &[Allocation]) -> Result<StreamPlan> {
    if !(2..=8).contains(&lanes) {
        return Err("concurrent lowering requires between two and eight borrowed streams".into());
    }
    if !(1..=8).contains(&pipeline_depth) {
        return Err("stream pipeline requires between one and eight pending phases".into());
    }
    let source = analyze(program, false)?;
    let ctx = context()?;
    let domain = isl(UnionSet::read_from_str(&ctx, &source.domain))?;
    let dependencies = isl(UnionMap::read_from_str(&ctx, &source.dependencies))?;
    let schedule = read_schedule(&ctx, schedule)?;
    legal(&domain, &dependencies, &schedule)?;
    let pieces = super::region_capacity::maps(isl(schedule.copy())?)?;
    let width = isl(pieces.first().ok_or("concurrent lowering requires a nonempty schedule")?.dim(DimType::Out))? as usize;
    if axis >= width { return Err("concurrent phase axis exceeds the schedule rank".into()); }
    let coordinates = (0..width).map(|i| format!("t{i}")).collect::<Vec<_>>();
    let mut phases = coordinates.clone();
    phases[axis] = format!("floor(t{axis}/{lanes})");
    let projection = isl(UnionMap::read_from_str(&ctx, &format!("{{ [{}] -> [{}] }}",
        coordinates.join(","), phases.join(","))))?;
    let phase = isl(isl(schedule.copy())?.apply_range(projection))?;
    // Every dependency must cross a phase boundary, including RAW/WAR/WAW,
    // order tokens, and opaque effects. A serially legal input is insufficient.
    legal(&domain, &dependencies, &phase)?;
    let actions = region_control::statements(program, &source.statements)?;
    let eligible = actions.iter().map(|action| concurrent_action(action, program, imports, retained)).collect::<Vec<_>>();
    let mut placements = Vec::new();
    let mut routing = isl(UnionMap::read_from_str(&ctx, "{ }"))?;
    for piece in pieces {
        let id = isl(piece.get_tuple_name(DimType::In))?.strip_prefix('S')
            .ok_or("unexpected statement tuple")?.parse::<usize>().map_err(|e| e.to_string())?;
        let lane = if eligible[id] { format!("t{axis} - {lanes}*floor(t{axis}/{lanes})") }
            else { "0".into() };
        let projection = isl(UnionMap::read_from_str(&ctx, &format!("{{ [{}] -> [{lane}] }}", coordinates.join(","))))?;
        let map = isl(isl(UnionMap::from_map(piece))?.apply_range(projection))?;
        routing = isl(routing.union(map))?;
    }
    // Demand actual symbolic parallel opportunity, not an unused second lane.
    // Equal-phase pairs on distinct lanes are independent by the legality
    // proof above. This checks the complete domain without sampling a cursor.
    let peers = isl(isl(phase.copy())?.apply_range(isl(isl(phase.copy())?.reverse())?))?;
    let different_lanes = isl(isl(routing.copy())?.lex_lt_union_map(isl(routing.copy())?))?;
    if isl(isl(peers.intersect(different_lanes))?.is_empty())? {
        return Err("physical phases expose no independent work on distinct streams".into());
    }
    let mut waits = Vec::new();
    if pipeline_depth > 1 {
        for lane in 0..lanes {
            let lane_domain = isl(isl(isl(routing.copy())?.intersect_range(
                isl(UnionSet::read_from_str(&ctx, &format!("{{ [{lane}] }}")))?))?.domain())?;
            let producer_phases = isl(isl(phase.copy())?.intersect_domain(lane_domain))?;
            // A lane issues phases in increasing order. Its last dependency
            // phase covers all earlier producers on that lane, including ties
            // within a phase; no assumption about ISL's statement tie order.
            let frontier = isl(isl(isl(isl(dependencies.copy())?.reverse())?
                .apply_range(producer_phases))?.lexmax())?;
            waits.push(StreamWaits { lane, predecessors: tables(frontier, &ctx, 'S', false)? });
        }
    }
    placements.extend(tables(routing, &ctx, 'S', false)?);
    placements.sort_by_key(|table| table.input_id);
    let phase_schedule = map_text(&phase)?;
    let payload = region_payload::derive(program, &phase_schedule, imports, retained)?;
    let mut devices = Vec::new();
    for device in &payload.devices {
        let scratch = payload.capacity.devices.iter().find(|d| d.device == device.device)
            .map_or(0, |d| d.call_scratch_bytes_upper_bound);
        let concurrent = scratch.checked_mul(u64::from(lanes)).ok_or("concurrent scratch overflows u64")?;
        let phase_peak = device.peak_bytes_upper_bound.checked_add(concurrent - scratch)
            .ok_or("concurrent payload overflows u64")?;
        // Keep the existing phase proof intact. At most depth-1 completed-in-
        // issue-order phases retain their references while the active phase
        // executes. Charge their entire envelopes, including scratch/retention.
        let pending = phase_peak.checked_mul(u64::from(pipeline_depth - 1))
            .ok_or("pending phase payload overflows u64")?;
        let peak = phase_peak.checked_add(pending).ok_or("pipeline payload overflows u64")?;
        devices.push(StreamDevice { device: device.device.clone(), concurrent_scratch_bytes_upper_bound: concurrent,
            pending_phase_bytes_upper_bound: pending, peak_bytes_upper_bound: peak });
    }
    let mut result = StreamPlan { version: 2, program_digest: source.program_digest,
        schedule_map: map_text(&schedule)?, axis, lanes, pipeline_depth, phase_schedule, placements, waits,
        host_fences: eligible.iter().enumerate().filter_map(|(id, eligible)| (!eligible).then_some(id)).collect(), payload, devices,
        signature: String::new() };
    result.signature = digest(&canonical_ascii_json(&result)?);
    Ok(result)
}

pub(super) fn validate(program: &Program, imports: &[TensorImport], retained: &[Allocation], supplied: &StreamPlan) -> Result<()> {
    if *supplied != derive(program, &supplied.schedule_map, supplied.axis, supplied.lanes, supplied.pipeline_depth, imports, retained)? {
        return Err("stream plan differs from source dependencies, imports or concurrent lifetimes".into());
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (program_json, schedule, axis, lanes, imports_json, retained_json, pipeline_depth=1))]
fn lower_region_streams(program_json: &str, schedule: &str, axis: usize, lanes: u32,
                        imports_json: &str, retained_json: &str, pipeline_depth: u32) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let retained: Vec<Allocation> = serde_json::from_str(retained_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(&program, schedule, axis, lanes, pipeline_depth, &imports, &retained)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_streams(program_json: &str, imports_json: &str, retained_json: &str, plan_json: &str) -> PyResult<()> {
    let run = || -> Result<()> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let retained: Vec<Allocation> = serde_json::from_str(retained_json).map_err(|e| e.to_string())?;
        let supplied: StreamPlan = serde_json::from_str(plan_json).map_err(|e| e.to_string())?;
        validate(&program, &imports, &retained, &supplied)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (program_json, imports_json, retained_json, lanes, max_candidates, max_pipeline_depth=1))]
fn explore_region_streams(program_json: &str, imports_json: &str, retained_json: &str,
                          lanes: u32, max_candidates: usize, max_pipeline_depth: u32) -> PyResult<String> {
    let run = || -> Result<String> {
        if !(2..=8).contains(&lanes) || !(1..=8).contains(&max_pipeline_depth) || max_candidates > 8 {
            return Err("stream exploration requires bounded lanes, pipeline depth and at most eight candidates".into());
        }
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let retained: Vec<Allocation> = serde_json::from_str(retained_json).map_err(|e| e.to_string())?;
        let source = super::region_tiling::explore_schedules(&program, 8)?;
        let mut result = Exploration { program_digest: source.program_digest, candidates: vec![], diagnostics: source.diagnostics };
        let ctx = context()?;
        let mut seen = std::collections::BTreeSet::new();
        for candidate in source.candidates {
            let pieces = super::region_capacity::maps(read_schedule(&ctx, &candidate.schedule_map)?)?;
            let Some(first) = pieces.first() else { continue; };
            let width = isl(first.dim(DimType::Out))? as usize;
            for axis in 0..width {
                if result.candidates.len() == max_candidates { return canonical_ascii_json(&result); }
                match derive(&program, &candidate.schedule_map, axis, lanes, 1, &imports, &retained) {
                    Ok(plan) => {
                        let identity = canonical_ascii_json(&(&plan.phase_schedule, &plan.placements))?;
                        if seen.insert(identity) {
                            result.candidates.push(plan);
                            for depth in 2..=max_pipeline_depth {
                                if result.candidates.len() == max_candidates { return canonical_ascii_json(&result); }
                                match derive(&program, &candidate.schedule_map, axis, lanes, depth, &imports, &retained) {
                                    Ok(pipeline) => result.candidates.push(pipeline),
                                    Err(error) => result.diagnostics.push(format!("{} axis {axis} depth {depth}: {error}", candidate.source)),
                                }
                            }
                        }
                    },
                    Err(error) => result.diagnostics.push(format!("{} axis {axis}: {error}", candidate.source)),
                }
            }
        }
        canonical_ascii_json(&result)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lower_region_streams, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_streams, module)?)?;
    module.add_function(wrap_pyfunction!(explore_region_streams, module)?)?;
    Ok(())
}

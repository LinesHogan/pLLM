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
use super::region_imports::{TensorImport, native_copy_bytes, tensor_bytes};
use super::region_payload::{self, Payload};
use super::region_program::{Operation, Program, Resource, Type};
use super::region_retention::Allocation;
use super::semantic_graph::{canonical_ascii_json, digest};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StreamDevice {
    pub device: String,
    concurrent_scratch_bytes_upper_bound: u64,
    window_storage_bytes_upper_bound: u64,
    entry_copy_bytes_upper_bound: u64,
    result_copy_bytes_upper_bound: u64,
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
    slots: Vec<WindowSlot>,
    state_windows: Vec<StateWindow>,
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct WindowSlot { value_id: usize, shape: Vec<u64>, dtype: String, device: String, count: u64, bytes: u64, stride_bytes: u64 }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StateWindow { pub resource: String, pub private_roots: u64 }

fn tensor_leaf(type_: &Type) -> Option<Type> {
    match type_ {
        Type::Tensor { .. } => Some(type_.clone()),
        Type::Bank { element, .. } => tensor_leaf(element),
        _ => None,
    }
}

fn queue_only(action: &Action, program: &Program, imports: &[TensorImport]) -> bool {
    // Retention endpoints remain on the owner, but an audited device-only
    // implementation can acquire/release ownership without observing the GPU.
    if concurrent_action(action, program, imports, &[]) { return true; }
    match action {
        Action::Primitive { operation } => matches!(operation, Operation::Snapshot { .. } | Operation::Publish { .. }),
        Action::If { then_block, else_block } => then_block.steps.iter().chain(&else_block.steps)
            .all(|step| queue_only(&step.action, program, imports)),
        Action::Scan { body, .. } | Action::Map { body, .. } => body.steps.iter().all(|step| queue_only(&step.action, program, imports)),
        Action::While { condition, body, .. } => condition.steps.iter().chain(&body.steps).all(|step| queue_only(&step.action, program, imports)),
    }
}

pub(super) fn publishes(action: &Action, resource: &str) -> bool {
    match action {
        Action::Primitive { operation } => matches!(operation, Operation::Publish { resource: name, .. } if name == resource),
        Action::If { then_block, else_block } => then_block.steps.iter().chain(&else_block.steps).any(|step| publishes(&step.action, resource)),
        Action::While { condition, body, .. } => condition.steps.iter().chain(&body.steps).any(|step| publishes(&step.action, resource)),
        Action::Scan { body, .. } | Action::Map { body, .. } => body.steps.iter().any(|step| publishes(&step.action, resource)),
    }
}

fn concurrent_action(action: &Action, program: &Program, imports: &[TensorImport], retained: &[Allocation]) -> bool {
    match action {
        Action::Primitive { operation: Operation::Call { function, .. } } => {
            // Acquiring/releasing backend retention is a host ownership event,
            // even when the endpoint also has a device-kernel implementation.
            !retained.iter().any(|allocation| allocation.is_endpoint(function))
            && imports.iter().any(|import| import.name() == function && match import {
                TensorImport::Exported { access } => access.operations.iter().all(|op| op.kind != super::tensor_access::AccessKind::Unknown),
                TensorImport::Compiled { .. } | TensorImport::FunctionalKernel { .. } | TensorImport::Borrowed { .. } => true,
                TensorImport::NativeKernel { declaration, host_reads, tensor_sources, .. } => !host_reads
                    && tensor_sources.iter().all(|graph| graph.operations.iter().all(|op| op.kind != super::tensor_access::AccessKind::Unknown))
                    && declaration.resources.iter().all(|name|
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

pub(super) fn derive(program: &Program, schedule: &str, axis: usize, lanes: u32, pipeline_depth: u32,
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
    if actions.iter().any(|action| !queue_only(action, program, imports)) {
        return Err("asynchronous windows require imported device-only endpoints and queue-only page actions".into());
    }
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
    {
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
    // Private materialization removes output aliases at the physical window
    // boundary. Keep actual imports for native scratch; use the copy ABI only
    // to derive the independent value lifetimes and slot counts.
    let mut owned_imports = program.functions.iter().map(|function| {
        imports.iter().find(|import| import.name() == function.name && matches!(import, TensorImport::Borrowed { .. }))
            .cloned().unwrap_or(TensorImport::Native { declaration: function.clone() })
    }).collect::<Vec<_>>();
    owned_imports.sort_by(|a, b| a.name().cmp(b.name()));
    let storage = super::region_storage::derive(program, &phase_schedule, &owned_imports)?;
    let backing = super::region_storage::derive(program, &phase_schedule, imports)?;
    let capacity = super::aer::region_tensor_memory(program, &storage, &owned_imports)?;
    let mut borrowed = std::collections::BTreeSet::new();
    for statement in &source.statements {
        if let Operation::Call { function, .. } = super::region_storage::operation(program, &statement.path)? {
            if imports.iter().any(|import| import.name() == function && matches!(import, TensorImport::Borrowed { .. })) {
                borrowed.extend(statement.outputs.iter().copied());
            }
        }
    }
    let mut footprints = std::collections::BTreeMap::new();
    for statement in &source.statements {
        if let Operation::Call { function, .. } = super::region_storage::operation(program, &statement.path)? {
            let graph = imports.iter().find_map(|import| match import {
                TensorImport::Exported { access } if import.name() == function => Some(access),
                TensorImport::Compiled { source_access, .. } if import.name() == function => Some(source_access),
                _ => None,
            });
            if let Some(graph) = graph {
                let values = super::tensor_access::validate(graph)?;
                for (name, &id) in graph.outputs.iter().zip(&statement.outputs) {
                    let view = values[name.as_str()];
                    let width = tensor_bytes(&Type::Tensor { shape: vec![1], dtype: view.dtype.clone(), device: String::new() })?;
                    let span = if view.shape.contains(&0) { 0 } else { view.shape.iter().zip(&view.stride)
                        .try_fold(1_u64, |n, (&size, &stride)| n.checked_add((size - 1).checked_mul(stride)?))
                        .ok_or("window view span overflows")? };
                    footprints.insert(id, span.checked_mul(width).ok_or("window view bytes overflow")?);
                }
            }
        }
    }
    let slots = capacity.roots.iter().filter(|root| !borrowed.contains(&root.value_id) && root.live_instances_upper_bound > 0)
        .map(|root| {
            let Type::Tensor { shape, dtype, device } = tensor_leaf(&source.values[root.value_id].type_).ok_or("window root must be a tensor")? else { unreachable!() };
            let bytes = footprints.get(&root.value_id).copied().or_else(|| backing.roots.iter().find(|r| r.value_id == root.value_id).map(|r| r.bytes))
                .unwrap_or(tensor_bytes(&Type::Tensor { shape: shape.clone(), dtype: dtype.clone(), device: device.clone() })?);
            let width = tensor_bytes(&Type::Tensor { shape: vec![1], dtype: dtype.clone(), device: device.clone() })?;
            let bytes = bytes.checked_add(width - 1).ok_or("window element alignment overflows")? / width * width;
            let alignment = if device.starts_with("cuda:") { 512 } else { 64 };
            let stride_bytes = bytes.checked_add(alignment - 1).ok_or("window slot alignment overflows")? / alignment * alignment;
            Ok(WindowSlot { value_id: root.value_id, shape, dtype, device, bytes, stride_bytes,
                count: root.live_instances_upper_bound.checked_mul(u64::from(pipeline_depth)).ok_or("window slot count overflows")? })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut materialization = std::collections::BTreeMap::<String, u64>::new();
    for statement in &source.statements {
        let mut raw = std::collections::BTreeMap::<String, u64>::new();
        for &id in &statement.outputs {
            if let Some(root) = backing.roots.iter().find(|root| root.value_id == id && !borrowed.contains(&id)) {
                let mut leaves = 1_u64;
                let mut type_ = &source.values[id].type_;
                while let Type::Bank { element, size } = type_ {
                    leaves = leaves.checked_mul(size.maximum(program)).ok_or("window result count overflows")?;
                    type_ = element;
                }
                let n = raw.entry(root.device.clone()).or_default();
                *n = n.checked_add(root.bytes.checked_mul(leaves).ok_or("raw window output overflows")?).ok_or("raw window output sum overflows")?;
            }
        }
        for (device, bytes) in raw { let n = materialization.entry(device).or_default(); *n = (*n).max(bytes); }
    }
    let mut devices = Vec::new();
    for device in &payload.devices {
        let owned = capacity.devices.iter().find(|d| d.device == device.device).ok_or("missing window device capacity")?;
        let scratch = payload.capacity.devices.iter().find(|d| d.device == device.device).map_or(0, |d| d.call_scratch_bytes_upper_bound);
        let concurrent = scratch.checked_add(materialization.get(&device.device).copied().unwrap_or(0))
            .and_then(|n| n.checked_mul(u64::from(lanes))).ok_or("window scratch overflows")?;
        let resident = slots.iter().filter(|slot| slot.device == device.device)
            .try_fold(0_u64, |n, slot| n.checked_add(slot.stride_bytes.checked_mul(slot.count)?)).ok_or("window residency overflows")?;
        let mut result_bytes = 0_u64;
        for type_ in super::region_program::verify(program)? {
            if let Some((name, bytes)) = native_copy_bytes(&type_, &program.parameters)? {
                if name == device.device { result_bytes = result_bytes.checked_add(bytes).ok_or("window results overflow")?; }
            }
        }
        let peak = resident.checked_add(concurrent).and_then(|n| n.checked_add(result_bytes))
            .and_then(|n| n.checked_add(owned.entry_copy_bytes_upper_bound))
            .and_then(|n| n.checked_add(device.retained_bytes_upper_bound)).ok_or("window peak overflows")?;
        devices.push(StreamDevice { device: device.device.clone(), concurrent_scratch_bytes_upper_bound: concurrent,
            window_storage_bytes_upper_bound: resident, entry_copy_bytes_upper_bound: owned.entry_copy_bytes_upper_bound, result_copy_bytes_upper_bound: result_bytes, peak_bytes_upper_bound: peak });
    }
    let state_windows = program.resources.iter().filter_map(|resource| match resource {
        Resource::State { name, .. } => Some(StateWindow { resource: name.clone(), private_roots: if actions.iter().any(|action| publishes(action, name)) { 2 } else { 0 } }),
        _ => None,
    }).collect();
    let mut result = StreamPlan { version: 3, program_digest: source.program_digest,
        schedule_map: map_text(&schedule)?, axis, lanes, pipeline_depth, phase_schedule, placements, waits,
        slots, state_windows, payload, devices, signature: String::new() };
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
#[pyo3(signature = (program_json, imports_json, retained_json, schedule, limits_json=None))]
fn explore_region_streams(program_json: &str, imports_json: &str, retained_json: &str,
                          schedule: &str, limits_json: Option<&str>) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let retained: Vec<Allocation> = serde_json::from_str(retained_json).map_err(|e| e.to_string())?;
        let limits: Option<Vec<super::region_selection::PayloadLimit>> = limits_json
            .map(|text| serde_json::from_str(text).map_err(|e| e.to_string())).transpose()?;
        // Invalid source/imports are errors, not failed parallel opportunities.
        region_payload::derive(&program, schedule, &imports, &retained)?;
        let source = analyze(&program, false)?;
        let mut result = Exploration { program_digest: source.program_digest, candidates: vec![], diagnostics: source.diagnostics };
        let ctx = context()?;
        let pieces = super::region_capacity::maps(read_schedule(&ctx, schedule)?)?;
        if let Some(first) = pieces.first() {
            for axis in 0..isl(first.dim(DimType::Out))? as usize {
                match derive(&program, schedule, axis, 2, 2, &imports, &retained) {
                    Ok(plan) => {
                        if limits.as_ref().is_some_and(|limits| plan.devices.iter().any(|device|
                            !limits.iter().any(|limit| limit.device == device.device && limit.bytes >= device.peak_bytes_upper_bound))) {
                            continue;
                        }
                        result.candidates.push(plan);
                        break;
                    },
                    Err(error) => result.diagnostics.push(format!("axis {axis}: {error}")),
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

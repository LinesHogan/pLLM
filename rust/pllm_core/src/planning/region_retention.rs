//! Imported backend retention scopes over source-derived scalar identities.
//! This is a payload bound. Release calls are not GPU completion evidence.

use std::cell::RefCell;
use std::collections::{BTreeSet, VecDeque};
use isl_rs::{UnionMap, UnionSet};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_dependence::{self, analyze, canonical_schedule, context, isl, legal, map_text, read_schedule, Analysis};
use super::region_program::{Operation, Program, Region, ScalarType, Type};
use super::region_storage::operation;
use super::region_capacity::{fiber_bound, maps};
use super::semantic_graph::{canonical_ascii_json, digest};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Endpoint { function: String, argument: usize }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Allocation { name: String, acquire: Endpoint, release: Endpoint, device: String, bytes: u64 }

impl Allocation {
    pub(super) fn is_endpoint(&self, function: &str) -> bool {
        self.acquire.function == function || self.release.function == function
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct RetentionBound {
    name: String,
    pub device: String,
    bytes_per_instance: u64,
    acquisitions: String,
    releases: String,
    live_instances: String,
    peak_instances_upper_bound: u64,
    pub peak_bytes_upper_bound: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Retention {
    pub program_digest: String,
    pub schedule_map: String,
    imports_digest: String,
    pub allocations: Vec<RetentionBound>,
    signature: String,
}

/// The caller has already validated the source and canonical schedule while
/// deriving capacity. Empty imports need no further lifetime query.
pub(super) fn empty(program_digest: String, schedule_map: String) -> Result<Retention> {
    let mut result = Retention { program_digest, schedule_map,
        imports_digest: digest(&canonical_ascii_json(&Vec::<Allocation>::new())?), allocations: vec![], signature: String::new() };
    result.signature = digest(&canonical_ascii_json(&result)?);
    Ok(result)
}

fn calls(region: &Region, name: &str) -> usize {
    region.block.operations.iter().map(|op| match op {
        Operation::Call { function, .. } => usize::from(function == name),
        Operation::Scan { body, .. } | Operation::Map { body, .. } => calls(body, name),
        Operation::If { then_region, else_region, .. } => calls(then_region, name) + calls(else_region, name),
        Operation::While { condition, body, .. } => calls(condition, name) + calls(body, name),
        _ => 0,
    }).sum()
}

fn endpoint(program: &Program, analysis: &Analysis, endpoint: &Endpoint) -> Result<String> {
    let declaration = program.functions.iter().find(|f| f.name == endpoint.function)
        .ok_or("retention endpoint has no source function")?;
    if declaration.inputs.get(endpoint.argument) != Some(&Type::Scalar { name: ScalarType::Index }) {
        return Err("retention identity requires an index data operand".into());
    }
    let mut mappings = Vec::new();
    for statement in &analysis.statements {
        if matches!(operation(program, &statement.path)?, Operation::Call { function, .. } if function == &endpoint.function) {
            let id = statement.inputs[endpoint.argument];
            let value = &analysis.values[id];
            let coordinates = (0..statement.extents.len()).map(|i| format!("i{i}")).collect::<Vec<_>>();
            if value.extents.len() > coordinates.len() { return Err("retention operand escapes its lexical scope".into()); }
            mappings.push(format!("S{}[{}] -> V{id}[{}]", statement.id, coordinates.join(","),
                coordinates[..value.extents.len()].join(",")));
        }
    }
    if mappings.is_empty() || mappings.len() != calls(&program.entry, &endpoint.function) {
        return Err("retention endpoints must be visible affine calls, outside dynamic control".into());
    }
    Ok(format!("{}{{ {} }}", region_dependence::parameter_prefix(program), mappings.join("; ")))
}

pub(super) fn derive(program: &Program, schedule_map: &str, imports: &[Allocation]) -> Result<Retention> {
    thread_local! {
        static RETENTION: RefCell<VecDeque<(String, Retention)>> = const { RefCell::new(VecDeque::new()) };
    }
    let schedule = canonical_schedule(schedule_map)?;
    let key = digest(&canonical_ascii_json(&(program, &schedule, imports))?);
    super::region_cache::memoized(&RETENTION, key, || derive_retention(program, &schedule, imports))
}

fn derive_retention(program: &Program, schedule_map: &str, imports: &[Allocation]) -> Result<Retention> {
    let analysis = analyze(program, false)?;
    let ctx = context()?;
    let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain))?;
    let dependencies = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies))?;
    let schedule = read_schedule(&ctx, schedule_map)?;
    legal(&domain, &dependencies, &schedule)?;
    let scalars = isl(isl(UnionMap::read_from_str(&ctx, &analysis.value_sources))?.apply_range(
        isl(UnionMap::read_from_str(&ctx, &analysis.scalar_values))?))?;
    let mut names = BTreeSet::new();
    let mut allocations = Vec::new();
    for (ordinal, import) in imports.iter().enumerate() {
        if import.name.is_empty() || !names.insert(&import.name) || import.device.is_empty() || import.bytes == 0 {
            return Err("retained allocations require unique names, a device and positive bytes".into());
        }
        let acquire = isl(isl(UnionMap::read_from_str(&ctx, &endpoint(program, &analysis, &import.acquire)?))?
            .intersect_domain(isl(domain.copy())?))?;
        let release = isl(isl(UnionMap::read_from_str(&ctx, &endpoint(program, &analysis, &import.release)?))?
            .intersect_domain(isl(domain.copy())?))?;
        let identities = isl(UnionMap::read_from_str(&ctx, &format!("{{ [key] -> A{ordinal}[key] }}")))?;
        let acquire_keys = isl(isl(isl(acquire.copy())?.apply_range(isl(scalars.copy())?))?.apply_range(isl(identities.copy())?))?;
        let release_keys = isl(isl(isl(release.copy())?.apply_range(isl(scalars.copy())?))?.apply_range(identities))?;
        for (calls, keys) in [(&acquire, &acquire_keys), (&release, &release_keys)] {
            if !isl(keys.is_single_valued())? || !isl(keys.is_injective())?
                || !isl(isl(isl(calls.copy())?.domain())?.is_equal(&isl(isl(keys.copy())?.domain())?))?
            {
                return Err("retention identities must resolve uniquely for every endpoint instance".into());
            }
        }
        if !isl(isl(isl(acquire_keys.copy())?.range())?.is_equal(&isl(isl(release_keys.copy())?.range())?))? {
            return Err("retention acquisitions and releases must cover the same identities".into());
        }
        let paired = isl(isl(acquire_keys.copy())?.apply_range(isl(isl(release_keys.copy())?.reverse())?))?;
        let reference = read_schedule(&ctx, &analysis.reference_schedule)?;
        let source_before = isl(isl(reference.copy())?.lex_lt_union_map(reference))?;
        if !isl(paired.is_subset(&source_before))? {
            return Err("source must acquire retained allocation before its release".into());
        }
        let before = isl(isl(schedule.copy())?.lex_lt_union_map(isl(schedule.copy())?))?;
        if !isl(paired.is_subset(&before))? {
            return Err("retained allocation must be acquired before its release".into());
        }
        let births = isl(isl(acquire_keys.reverse())?.apply_range(isl(schedule.copy())?))?;
        let ends = isl(isl(release_keys.reverse())?.apply_range(isl(schedule.copy())?))?;
        let live = isl(isl(isl(births.copy())?.lex_le_union_map(isl(schedule.copy())?))?.intersect(
            isl(isl(ends.copy())?.lex_ge_union_map(isl(schedule.copy())?))?))?;
        let mut peak = 0;
        for fiber in maps(isl(isl(live.copy())?.reverse())?)? {
            peak = peak.max(fiber_bound(&fiber, None)?);
        }
        allocations.push(RetentionBound {
            name: import.name.clone(), device: import.device.clone(), bytes_per_instance: import.bytes,
            acquisitions: map_text(&births)?, releases: map_text(&ends)?, live_instances: map_text(&live)?,
            peak_instances_upper_bound: peak,
            peak_bytes_upper_bound: peak.checked_mul(import.bytes).ok_or("retained payload bound overflows u64")?,
        });
    }
    let mut result = Retention { program_digest: analysis.program_digest, schedule_map: map_text(&schedule)?,
        imports_digest: digest(&canonical_ascii_json(&imports)?), allocations, signature: String::new() };
    result.signature = digest(&canonical_ascii_json(&result)?);
    Ok(result)
}

#[pyfunction]
fn analyze_region_retention(program_json: &str, schedule_map: &str, imports_json: &str) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<Allocation> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(&program, schedule_map, &imports)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_retention(program_json: &str, imports_json: &str, retention_json: &str) -> PyResult<()> {
    let run = || -> Result<()> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<Allocation> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let supplied: Retention = serde_json::from_str(retention_json).map_err(|e| e.to_string())?;
        if supplied != derive(&program, &supplied.schedule_map, &imports)? {
            return Err("retention bound differs from source calls and imported allocation scopes".into());
        }
        Ok(())
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze_region_retention, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_retention, module)?)?;
    Ok(())
}

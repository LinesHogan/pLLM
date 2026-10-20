//! Symbolic storage attribution for imported calls and atomic control results.
//! These are derived facts for AER, never a second state store or runtime plan.

use isl_rs::{UnionMap, UnionSet};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};

use super::region_bindings::{tables, AccessTable};
use super::region_dependence::{analyze, canonical_schedule, closure, context, isl, legal, map_text, read_schedule, set_text, ValueSite};
use super::region_program::{Operation, Program, Type};
use super::region_imports::{tensor_bytes, TensorImport};
use super::semantic_graph::{canonical_ascii_json, digest};
use super::tensor_access::{self, StorageView};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StorageRoot { pub value_id: usize, pub bytes: u64, pub device: String }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StorageAnalysis {
    pub program_digest: String,
    pub schedule_map: String,
    pub imports_digest: String,
    pub roots: Vec<StorageRoot>,
    pub input_storages: String,
    pub borrowed_storages: String,
    pub live_times: String,
    pub allocation_times: String,
    pub retained_storages: String,
    value_storage: String,
    storage_consumers: String,
    routes: Vec<AccessTable>,
    births: Vec<AccessTable>,
    last_uses: Vec<AccessTable>,
    retained: Vec<AccessTable>,
    pub signature: String,
}

pub(super) fn tensor_leaf(mut type_: &Type) -> Option<&Type> {
    while let Type::Bank { element, .. } = type_ { type_ = element; }
    matches!(type_, Type::Tensor { .. }).then_some(type_)
}

/// Conservative backing sizes for results of atomic structured actions. Such
/// results can alias an input or an internal exported view; logical numel alone
/// cannot bound the retained storage. Count each result separately at this cap.
pub(super) fn backing_bounds(program: &Program, imports: &[TensorImport]) -> Result<BTreeMap<String, u64>> {
    let mut bounds = BTreeMap::<String, u64>::new();
    let mut include = |device: &str, bytes: u64| {
        bounds.entry(device.to_owned()).and_modify(|n| *n = (*n).max(bytes)).or_insert(bytes);
    };
    for type_ in program.entry.block.arguments.iter().map(|v| &v.type_)
        .chain(program.functions.iter().flat_map(|f| f.inputs.iter().chain(&f.outputs)))
        .chain(program.resources.iter().filter_map(|r| r.payload()).flatten())
    {
        if let Some(type_ @ Type::Tensor { device, .. }) = tensor_leaf(type_) {
            include(device, tensor_bytes(type_)?);
        }
    }
    for import in imports {
        let graph = match import {
            TensorImport::Exported { access } => access,
            TensorImport::Compiled { source_access, .. } => source_access,
            TensorImport::Native { .. } | TensorImport::NativeKernel { .. }
                | TensorImport::FunctionalKernel { .. } | TensorImport::AdditiveScan { .. } | TensorImport::Borrowed { .. } => continue,
        };
        for storage in &graph.storages { include(&storage.device, storage.bytes); }
    }
    Ok(bounds)
}

fn tuple(site: &ValueSite) -> String {
    format!("V{}[{}]", site.id, (0..site.extents.len() + site.bank_shape.len())
        .map(|i| format!("i{i}")).collect::<Vec<_>>().join(","))
}

fn union(parts: &[String]) -> String { format!("{{ {} }}", parts.join("; ")) }

pub(super) fn operation<'a>(program: &'a Program, path: &[u32]) -> Result<&'a Operation> {
    let mut region = &program.entry;
    for pair in path[..path.len() - 1].chunks_exact(2) {
        region = match &region.block.operations[pair[0] as usize] {
            Operation::Scan { body, .. } | Operation::Map { body, .. } if pair[1] == 0 => body,
            _ => return Err("storage analysis needs an affine statement path".into()),
        };
    }
    Ok(&region.block.operations[*path.last().unwrap() as usize])
}

pub(super) fn derive(program: &Program, schedule_map: &str, imports: &[TensorImport]) -> Result<StorageAnalysis> {
    thread_local! {
        static STORAGE: RefCell<VecDeque<(String, StorageAnalysis)>> = const { RefCell::new(VecDeque::new()) };
    }
    let schedule = canonical_schedule(schedule_map)?;
    let key = digest(&canonical_ascii_json(&(program, &schedule, imports))?);
    super::region_cache::memoized(&STORAGE, key, || derive_storage(program, &schedule, imports))
}

/// Storage attribution facts that never mention a schedule. Candidate
/// schedules over the same source share one provenance derivation; only the
/// live-range envelope at the end is specialized per schedule.
#[derive(Clone)]
struct Provenance {
    domain: String,
    dependencies: String,
    program_digest: String,
    imports_digest: String,
    roots: Vec<StorageRoot>,
    borrowed_storages: String,
    input_storages: String,
    retained_storages: String,
    value_storage: String,
    storage_consumers: String,
    routes_tables: Vec<AccessTable>,
    retained_tables: Vec<AccessTable>,
    consumers: String,
    retained: String,
    inputs: String,
    births: String,
}

fn provenance(program: &Program, imports: &[TensorImport]) -> Result<Provenance> {
    thread_local! {
        static PROVENANCE: RefCell<VecDeque<(String, Provenance)>> = const { RefCell::new(VecDeque::new()) };
    }
    let key = digest(&canonical_ascii_json(&(program, imports))?);
    super::region_cache::memoized(&PROVENANCE, key, || derive_provenance(program, imports))
}

fn derive_provenance(program: &Program, imports: &[TensorImport]) -> Result<Provenance> {
    let analysis = analyze(program, false)?;
    if !analysis.exact_provenance {
        return Err("storage composition requires exact tensor value provenance".into());
    }
    if imports.len() != program.functions.len()
        || imports.windows(2).any(|p| p[0].name() >= p[1].name())
    {
        return Err("region storage imports must be complete and sorted by unique source identity".into());
    }
    let mut functions = BTreeMap::new();
    for import in imports {
        let function = program.functions.iter().find(|f| f.name == import.name())
            .ok_or("storage import has no source function")?;
        import.validate(function)?;
        if matches!(import, TensorImport::Borrowed { .. }) && !program.resources.iter().any(|resource|
            matches!(resource, super::region_program::Resource::Borrowed { name, .. } if name == &function.resources[0]))
        { return Err("tensor borrowing requires an original borrowed resource family".into()); }
        functions.insert(function.name.as_str(), import);
    }
    let ctx = context()?;
    let sources = isl(UnionMap::read_from_str(&ctx, &analysis.value_sources))?;
    let tensor_names = analysis.values.iter().filter(|v| tensor_leaf(&v.type_).is_some())
        .map(tuple).collect::<Vec<_>>();
    let tensor_domains = isl(isl(UnionSet::read_from_str(&ctx, &union(&tensor_names)))?
        .intersect(isl(isl(sources.copy())?.domain())?))?;
    let mut roots = BTreeMap::new();
    let mut inputs = Vec::new();
    let mut borrowed = Vec::new();
    let mut edges = Vec::new();
    let mut births = Vec::new();
    let backing = backing_bounds(program, imports)?;
    for &id in &analysis.arguments {
        let site = &analysis.values[id];
        if let Some(Type::Tensor { shape, dtype, device }) = tensor_leaf(&site.type_) {
            // RegionValueFrame privately clones each entry tensor leaf. Dense
            // clone storage is numel * itemsize, regardless of caller base size.
            let view = StorageView { name: String::new(), storage: 0, shape: shape.clone(),
                stride: vec![], offset: 0, dtype: dtype.clone() };
            roots.insert(id, StorageRoot { value_id: id, bytes: view.logical_bytes()?, device: device.clone() });
            inputs.push(tuple(site));
        }
    }
    for statement in &analysis.statements {
        let op = operation(program, &statement.path)?;
        let function = match op {
            Operation::Call { function, .. } => function,
            Operation::If { .. } | Operation::While { .. } | Operation::Snapshot { .. } => {
                for &id in &statement.outputs {
                    if let Some(type_ @ Type::Tensor { device, .. }) = tensor_leaf(&analysis.values[id].type_) {
                        // Snapshot clones are independent SSA allocations, not
                        // PTTS page roots. Atomic control may still return a view.
                        let bytes = if matches!(op, Operation::Snapshot { .. }) { tensor_bytes(type_)? }
                            else { *backing.get(device).ok_or("structured tensor result has no bounded backing storage")? };
                        roots.insert(id, StorageRoot { value_id: id, bytes, device: device.clone() });
                        births.push(format!("{} -> S{}[{}]", tuple(&analysis.values[id]), statement.id,
                            (0..statement.extents.len()).map(|i| format!("i{i}")).collect::<Vec<_>>().join(",")));
                    }
                }
                continue;
            },
            Operation::Scalar { .. } | Operation::Select { .. } | Operation::Update { .. }
                | Operation::Publish { .. } => continue,
            _ => return Err("storage composition needs tensor or structured actions; PTTS payloads retain their original budget".into()),
        };
        let import = functions.get(function.as_str()).ok_or("missing tensor access import")?;
        let graph = match import {
            TensorImport::Native { .. } | TensorImport::NativeKernel { .. } | TensorImport::FunctionalKernel { .. } | TensorImport::AdditiveScan { .. }
                | TensorImport::Compiled { .. } | TensorImport::Borrowed { .. } => {
                // Native/compiled boundaries clone each result independently.
                // Functional kernels return their own distinct dense backings.
                // A borrowed result instead names original resource storage;
                // retain its size and routes but distinguish it from new roots.
                let declaration = program.functions.iter().find(|f| f.name == *function).ok_or("missing native source declaration")?;
                for (type_, &id) in declaration.outputs.iter().zip(&statement.outputs) {
                    let Some(type_ @ Type::Tensor { device, .. }) = tensor_leaf(type_) else { continue; };
                    roots.insert(id, StorageRoot { value_id: id, bytes: tensor_bytes(type_)?, device: device.clone() });
                    if matches!(import, TensorImport::Borrowed { .. }) { borrowed.push(tuple(&analysis.values[id])); }
                    births.push(format!("{} -> S{}[{}]", tuple(&analysis.values[id]), statement.id,
                        (0..statement.extents.len()).map(|i| format!("i{i}")).collect::<Vec<_>>().join(",")));
                }
                continue;
            },
            TensorImport::Exported { access } => access,
        };
        let values = tensor_access::validate(graph)?;
        let mut storage_values = BTreeMap::new();
        for (name, &id) in graph.inputs.iter().zip(&statement.inputs) {
            storage_values.insert(values[name.as_str()].storage, id);
        }
        for (name, &id) in graph.outputs.iter().zip(&statement.outputs) {
            let storage = values[name.as_str()].storage;
            let site = &analysis.values[id];
            if let Some(&origin) = storage_values.get(&storage) {
                edges.push(format!("{} -> {}", tuple(site), tuple(&analysis.values[origin])));
            } else {
                storage_values.insert(storage, id);
                let facts = &graph.storages[storage];
                roots.insert(id, StorageRoot { value_id: id, bytes: facts.bytes, device: facts.device.clone() });
                births.push(format!("{} -> S{}[{}]", tuple(site), statement.id,
                    (0..statement.extents.len()).map(|i| format!("i{i}")).collect::<Vec<_>>().join(",")));
            }
        }
    }
    let root_names = roots.keys().map(|&id| tuple(&analysis.values[id])).collect::<Vec<_>>();
    let root_domains = isl(isl(UnionSet::read_from_str(&ctx, &union(&root_names)))?
        .intersect(isl(tensor_domains.copy())?))?;
    let borrowed_storages = map_text(&isl(isl(isl(UnionSet::read_from_str(&ctx, &union(&borrowed)))?
        .intersect(isl(root_domains.copy())?))?.identity())?)?;
    let aliases = isl(isl(isl(UnionMap::read_from_str(&ctx, &union(&edges)))?
        .intersect_domain(isl(tensor_domains.copy())?))?.intersect_range(isl(tensor_domains.copy())?))?;
    let relation = isl(isl(sources.intersect_domain(isl(tensor_domains.copy())?))?.union(aliases))?;
    let (paths, exact) = closure(relation)?;
    let routes = isl(paths.intersect_range(isl(root_domains.copy())?))?;
    if !exact || !isl(routes.is_single_valued())?
        || !isl(isl(isl(routes.copy())?.domain())?.is_equal(&tensor_domains))?
    { return Err("storage charges do not give every tensor leaf one bounded attribution".into()); }
    let consumers = isl(isl(isl(routes.copy())?.reverse())?.apply_range(
        isl(UnionMap::read_from_str(&ctx, &analysis.value_consumers))?))?;
    let result_names = analysis.results.iter().filter(|&&id| tensor_leaf(&analysis.values[id].type_).is_some())
        .map(|&id| tuple(&analysis.values[id])).collect::<Vec<_>>();
    let retained = isl(isl(isl(isl(routes.copy())?.intersect_domain(
        isl(UnionSet::read_from_str(&ctx, &union(&result_names)))?))?.range())?.identity())?;
    let inputs = isl(isl(UnionSet::read_from_str(&ctx, &union(&inputs)))?.intersect(tensor_domains))?;
    let input_storages = map_text(&isl(isl(inputs.copy())?.identity())?)?;
    // A call can return a bank whose element coordinates do not occur in the
    // producing statement's iteration tuple. Bound those coordinates by the
    // original root domain before assigning its allocation time.
    let births = isl(isl(UnionMap::read_from_str(&ctx, &union(&births)))?
        .intersect_domain(root_domains))?;
    let retained_text = map_text(&retained)?;
    let consumers_text = map_text(&consumers)?;
    Ok(Provenance {
        domain: analysis.domain, dependencies: analysis.dependencies,
        program_digest: analysis.program_digest,
        imports_digest: digest(&canonical_ascii_json(&imports)?),
        roots: roots.into_values().collect(),
        borrowed_storages,
        input_storages,
        retained_storages: retained_text.clone(),
        value_storage: map_text(&routes)?,
        storage_consumers: consumers_text.clone(),
        routes_tables: tables(routes, &ctx, 'V', true)?,
        retained_tables: tables(isl(retained.copy())?, &ctx, 'V', true)?,
        consumers: consumers_text,
        retained: retained_text,
        inputs: set_text(&inputs)?,
        births: map_text(&births)?,
    })
}

fn derive_storage(program: &Program, schedule_map: &str, imports: &[TensorImport]) -> Result<StorageAnalysis> {
    let provenance = provenance(program, imports).map_err(|e| format!("storage provenance: {e}"))?;
    let ctx = context()?;
    let domain = isl(UnionSet::read_from_str(&ctx, &provenance.domain))?;
    let dependencies = isl(UnionMap::read_from_str(&ctx, &provenance.dependencies))?;
    let schedule = read_schedule(&ctx, schedule_map)?;
    legal(&domain, &dependencies, &schedule).map_err(|e| format!("storage schedule: {e}"))?;
    ctx.reset_operations();
    let last_uses = isl(isl(isl(UnionMap::read_from_str(&ctx, &provenance.consumers))?
        .apply_range(isl(schedule.copy())?))?.lexmax())?;
    let retained = isl(UnionMap::read_from_str(&ctx, &provenance.retained))?;
    let inputs = isl(UnionSet::read_from_str(&ctx, &provenance.inputs))?;
    let births = isl(isl(UnionMap::read_from_str(&ctx, &provenance.births))?
        .apply_range(isl(schedule.copy())?))?;
    // Compare value times against the universe of the time space, not the
    // schedule's image: comparison pieces stay free of image constraints.
    // Joining the bounded times back through the schedule still restricts the
    // envelope to times that actually occur, so the live relation is unchanged.
    let times = isl(isl(isl(schedule.copy())?.range())?.universe())?;
    let after_birth = isl(isl(isl(births.copy())?.lex_le_union_map(isl(isl(times.copy())?.identity())?))?
        .union(isl(UnionMap::from_domain_and_range(inputs, isl(times.copy())?))?))?;
    let before_release = isl(isl(isl(last_uses.copy())?.lex_ge_union_map(isl(isl(times.copy())?.identity())?))?
        .union(isl(UnionMap::from_domain_and_range(isl(isl(retained.copy())?.domain())?, times))?))?;
    // Include even unused outputs during their producing call. Equal-time
    // calls share a conservative envelope, matching strict-< frame releases.
    let live_t = isl(after_birth.intersect(before_release)).map_err(|e| format!("storage live times: {e}"))?;
    ctx.reset_operations();
    // Capacity counts roots at a time, not at every statement sharing that
    // time. Birth/last-use intervals also cover gaps between real times;
    // gaps introduce no new allocations and cannot increase peak liveness.
    // Retained roots remain live after the final statement as before.
    let live = isl(live_t.union(isl(births.copy())?))?;
    ctx.reset_operations();
    let mut result = StorageAnalysis {
        program_digest: provenance.program_digest, schedule_map: map_text(&schedule)?,
        imports_digest: provenance.imports_digest, roots: provenance.roots,
        input_storages: provenance.input_storages,
        borrowed_storages: provenance.borrowed_storages,
        live_times: map_text(&live)?, allocation_times: map_text(&births)?,
        retained_storages: provenance.retained_storages,
        value_storage: provenance.value_storage, storage_consumers: provenance.storage_consumers,
        routes: provenance.routes_tables, births: tables(births, &ctx, 'V', false)?,
        last_uses: tables(last_uses, &ctx, 'V', false)?, retained: provenance.retained_tables,
        signature: String::new(),
    };
    result.signature = digest(&canonical_ascii_json(&result)?);
    Ok(result)
}

#[pyfunction]
fn analyze_region_storage(program_json: &str, schedule_map: &str, imports_json: &str) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(&program, schedule_map, &imports)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_storage(program_json: &str, imports_json: &str, storage_json: &str) -> PyResult<()> {
    let run = || -> Result<()> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let supplied: StorageAnalysis = serde_json::from_str(storage_json).map_err(|e| e.to_string())?;
        if supplied != derive(&program, &supplied.schedule_map, &imports)? {
            return Err("region storage witness differs from source and imported tensor accesses".into());
        }
        Ok(())
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze_region_storage, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_storage, module)?)?;
    Ok(())
}

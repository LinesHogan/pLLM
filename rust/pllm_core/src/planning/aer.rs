//! Exact small-graph partition + placement over the original semantic graph.
//!
//! The memory witness is an anchor envelope: boundary values remain live for
//! the whole producer-through-last-consumer anchor interval. Regions execute
//! serially at an anchor, so their private scratch combines by maximum.
//! Cold task allocations (including captured graph pools) remain resident at
//! every anchor and sum across tasks, independently of region fusion.

use super::scheduler::{GraphIndex, PhysicalTask};
use super::semantic_graph::{
    ImplementationCandidate, InputSource, OutputKind, SemanticGraph, Timeline,
};
use super::state_representation::StatePhysicalPlan;
use super::tensor_access::{self, AccessGraph, AccessKind, DeviceMemory, MemoryProfile, MemoryStep};
use super::semantic_graph::{canonical_ascii_json, digest};
use super::region_capacity::{fiber_bound, maps, Capacity, DeviceCapacity, StorageBound};
use super::region_dependence::{analyze, context, isl};
use super::region_program::{Operation, Program};
use super::region_imports::{native_copy_bytes, tensor_bytes, TensorImport};
use super::region_storage::{self, StorageAnalysis};
use super::region_selection::{Request as RegionRequest, Selection as RegionSelection, Assessment};
use isl_rs::{DimType, Map, Set, UnionMap};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

thread_local! {
    // Only locally derived plain-data witnesses enter this bounded cache.
    // No caller-supplied capacity, ISL context or device allocation is retained.
    static REGION_CAPACITIES: RefCell<VecDeque<(String, Capacity)>> = const { RefCell::new(VecDeque::new()) };
}

/// Derive the tensor part of a candidate's resource envelope from its access
/// graph. This is also used by the region compilation boundary. Backend scratch,
/// allocator rounding, outer frames and PTTS residency remain separate facts;
/// a tensor-only profile must never certify a complete device memory budget.
pub(super) fn tensor_memory(graph: &AccessGraph) -> Result<MemoryProfile, String> {
    let values = tensor_access::validate(graph)?;
    let end = graph.operations.len();
    let input_storage: BTreeSet<_> = graph.inputs.iter().map(|n| values[n.as_str()].storage).collect();
    let result_storage: BTreeSet<_> = graph.outputs.iter().map(|n| values[n.as_str()].storage).collect();
    let mut last = vec![0; graph.storages.len()];
    let mut used = BTreeSet::new();
    for (step, op) in graph.operations.iter().enumerate() {
        last[values[op.node.as_str()].storage] = step;
        for name in &op.inputs {
            used.insert(name.as_str());
            last[values[name.as_str()].storage] = step;
        }
    }
    used.extend(graph.outputs.iter().map(String::as_str));
    // FX codegen does not clear an unused local result at a last consumer.
    // Keep that storage to return, rather than inventing dead-code elimination.
    for value in &graph.values {
        if input_storage.contains(&value.storage) || result_storage.contains(&value.storage)
            || !used.contains(value.name.as_str())
        {
            last[value.storage] = end;
        }
    }
    let mut devices = BTreeMap::new();
    for storage in &graph.storages {
        devices.entry(storage.device.clone()).or_insert_with(|| DeviceMemory {
            device: storage.device.clone(), input_bytes: 0, result_bytes: 0,
            peak_live_tensor_bytes: 0, peak_new_tensor_bytes: 0,
            logical_read_bytes: 0, logical_write_bytes: 0,
        });
    }
    fn add(total: &mut u64, bytes: u64) -> Result<(), String> {
        *total = total.checked_add(bytes).ok_or("tensor memory envelope overflows u64")?;
        Ok(())
    }
    for &id in &input_storage {
        let storage = &graph.storages[id];
        let device = devices.get_mut(&storage.device).unwrap();
        add(&mut device.input_bytes, storage.bytes)?;
        device.peak_live_tensor_bytes = device.input_bytes;
    }
    for &id in &result_storage {
        let storage = &graph.storages[id];
        add(&mut devices.get_mut(&storage.device).unwrap().result_bytes, storage.bytes)?;
    }
    let mut live = input_storage.clone();
    let mut steps = Vec::new();
    let mut unknown_accesses = Vec::new();
    for (step, op) in graph.operations.iter().enumerate() {
        let output = values[op.node.as_str()];
        live.insert(output.storage);
        match op.kind {
            AccessKind::View => {},
            AccessKind::Dense => {
                for name in &op.inputs {
                    let input = values[name.as_str()];
                    let device = &graph.storages[input.storage].device;
                    add(&mut devices.get_mut(device).unwrap().logical_read_bytes, input.logical_bytes()?)?;
                }
                let device = &graph.storages[output.storage].device;
                add(&mut devices.get_mut(device).unwrap().logical_write_bytes, output.logical_bytes()?)?;
            },
            AccessKind::Unknown => unknown_accesses.push(op.node.clone()),
        }
        let mut totals = BTreeMap::<&str, u64>::new();
        for &id in &live {
            let storage = &graph.storages[id];
            add(totals.entry(&storage.device).or_default(), storage.bytes)?;
        }
        for (device, bytes) in totals {
            let facts = devices.get_mut(device).unwrap();
            facts.peak_live_tensor_bytes = facts.peak_live_tensor_bytes.max(bytes);
            facts.peak_new_tensor_bytes = facts.peak_new_tensor_bytes.max(bytes - facts.input_bytes);
        }
        let release: Vec<_> = live.iter().copied().filter(|&id| last[id] == step).collect();
        steps.push(MemoryStep { node: op.node.clone(), live_storages: live.iter().copied().collect(),
            release_storages: release.clone() });
        for id in release { live.remove(&id); }
    }
    let mut profile = MemoryProfile { access_digest: digest(&canonical_ascii_json(graph)?),
        devices: devices.into_values().collect(), steps, unknown_accesses, signature: String::new() };
    profile.signature = digest(&canonical_ascii_json(&profile)?);
    Ok(profile)
}

/// Imported call costs exclude the source frame's boundary roots. For native
/// calls only wrapper copies are known: never invent an ATen graph or count
/// opaque backend internals as zero-cost operations.
#[derive(Clone, Default)]
pub(super) struct CallDeviceMemory {
    pub scratch_bytes: u64,
    pub logical_read_bytes: u64,
    pub logical_write_bytes: u64,
}

pub(super) struct CallMemory {
    pub devices: BTreeMap<String, CallDeviceMemory>,
    pub unknown: bool,
}

fn call_memory(import: &TensorImport, program: &Program) -> Result<CallMemory, String> {
    fn add(total: &mut u64, bytes: u64) -> Result<(), String> {
        *total = total.checked_add(bytes).ok_or("call memory envelope overflows u64")?;
        Ok(())
    }
    let mut devices = BTreeMap::<String, CallDeviceMemory>::new();
    match import {
        TensorImport::Borrowed { .. } => Ok(CallMemory { devices, unknown: false }),
        TensorImport::FunctionalKernel { declaration, allocations, device, .. } => {
            let mut allocated = 0_u64;
            for allocation in allocations { add(&mut allocated, allocation.bytes)?; }
            let mut results = 0_u64;
            for type_ in &declaration.outputs { add(&mut results, tensor_bytes(type_)?)?; }
            // Owned results are already outer storage roots. Only the other
            // functionalization/kernel allocations are call-local scratch.
            let scratch_bytes = allocated.checked_sub(results)
                .ok_or("functional kernel allocations do not cover owned results")?;
            devices.insert(device.clone(), CallDeviceMemory { scratch_bytes, ..Default::default() });
            // There are no wrapper copies. Internal kernel/functionalization
            // traffic remains unknown, not an assertion of zero memory access.
            Ok(CallMemory { devices, unknown: true })
        },
        TensorImport::Compiled { source_access, allocations, .. } => {
            let values = tensor_access::validate(source_access)?;
            // Compiler-proven read-only inputs borrow the source frame's live
            // roots. No private-input copies or alignment allocations occur.
            for name in &source_access.outputs {
                let value = values[name.as_str()];
                let bytes = value.logical_bytes()?;
                let facts = devices.entry(source_access.storages[value.storage].device.clone()).or_default();
                add(&mut facts.logical_read_bytes, bytes)?;
                add(&mut facts.logical_write_bytes, bytes)?;
            }
            for allocation in allocations {
                add(&mut devices.entry(allocation.device.clone()).or_default().scratch_bytes, allocation.bytes)?;
            }
            // Kernel allocation layouts are known. Its fused access traffic
            // is not inferred from unfused ATen operands or wrapper copy bytes.
            Ok(CallMemory { devices, unknown: true })
        },
        TensorImport::Native { declaration } | TensorImport::NativeKernel { declaration, .. } => {
            for type_ in &declaration.inputs {
                let Some((device, bytes)) = native_copy_bytes(type_, &program.parameters)? else { continue; };
                let facts = devices.entry(device.to_owned()).or_default();
                // Private arguments remain live until every output clone has
                // been made. Output clones themselves are outer storage roots.
                add(&mut facts.scratch_bytes, bytes)?;
                add(&mut facts.logical_read_bytes, bytes)?;
                add(&mut facts.logical_write_bytes, bytes)?;
            }
            for type_ in &declaration.outputs {
                let Some((device, bytes)) = native_copy_bytes(type_, &program.parameters)? else { continue; };
                let facts = devices.entry(device.to_owned()).or_default();
                add(&mut facts.logical_read_bytes, bytes)?;
                add(&mut facts.logical_write_bytes, bytes)?;
            }
            if let TensorImport::NativeKernel { allocations, device, .. } = import {
                devices.entry(device.clone()).or_default();
                for allocation in allocations {
                    add(&mut devices.entry(allocation.device.clone()).or_default().scratch_bytes, allocation.bytes)?;
                }
            }
            // Native kernel storage is explicit when supplied. Internal
            // logical traffic is still unknown, just as for compiled imports.
            Ok(CallMemory { devices, unknown: true })
        },
        TensorImport::Exported { access: graph } => {
            let profile = tensor_memory(graph)?;
            let values = tensor_access::validate(graph)?;
            let boundary: BTreeSet<_> = graph.inputs.iter().chain(&graph.outputs)
                .map(|n| values[n.as_str()].storage).collect();
            for facts in &profile.devices {
                devices.insert(facts.device.clone(), CallDeviceMemory { scratch_bytes: 0,
                    logical_read_bytes: facts.logical_read_bytes, logical_write_bytes: facts.logical_write_bytes });
            }
            for step in &profile.steps {
                let mut bytes = BTreeMap::<&str, u64>::new();
                for &id in &step.live_storages {
                    if !boundary.contains(&id) {
                        let buffer = &graph.storages[id];
                        add(bytes.entry(&buffer.device).or_default(), buffer.bytes)?;
                    }
                }
                for (device, bytes) in bytes {
                    let facts = devices.get_mut(device).unwrap();
                    facts.scratch_bytes = facts.scratch_bytes.max(bytes);
                }
            }
            Ok(CallMemory { devices, unknown: !profile.unknown_accesses.is_empty() })
        },
    }
}

/// Tensor payload envelope for one fixed region schedule. The source's
/// parameter context stays inside every cardinality bound; no loop expansion
/// or max-shape sampling is used. This is not a backend/allocator budget.
pub(super) fn region_tensor_memory(program: &Program, storage: &StorageAnalysis,
                                  imports: &[TensorImport]) -> Result<Capacity, String> {
    let key = digest(&canonical_ascii_json(&(program, storage, imports))?);
    super::region_cache::memoized(&REGION_CAPACITIES, key,
        || derive_region_tensor_memory(program, storage, imports))
}

fn derive_region_tensor_memory(program: &Program, storage: &StorageAnalysis,
                              imports: &[TensorImport]) -> Result<Capacity, String> {
    let ctx = context()?;
    let source = analyze(program, false)?;
    let mut devices = BTreeMap::new();
    let mut roots = BTreeMap::new();
    for root in &storage.roots {
        roots.insert(root.value_id, root);
        devices.entry(root.device.clone()).or_insert_with(|| DeviceCapacity {
            device: root.device.clone(), entry_copy_bytes_upper_bound: 0, result_copy_bytes_upper_bound: 0, live_tensor_bytes_upper_bound: 0,
            call_scratch_bytes_upper_bound: 0, peak_tensor_bytes_upper_bound: 0,
            logical_read_bytes_upper_bound: 0, logical_write_bytes_upper_bound: 0,
        });
    }
    fn id(map: &Map, dim: DimType, prefix: &str) -> Result<usize, String> {
        isl(map.get_tuple_name(dim))?.strip_prefix(prefix).ok_or("unexpected capacity tuple")?
            .parse().map_err(|_| "invalid capacity tuple identity".into())
    }
    fn add(total: &mut u64, count: u64, bytes: u64) -> Result<(), String> {
        *total = count.checked_mul(bytes).and_then(|n| total.checked_add(n))
            .ok_or("region capacity envelope overflows u64")?;
        Ok(())
    }
    let borrowed: BTreeSet<_> = maps(isl(UnionMap::read_from_str(&ctx, &storage.borrowed_storages))?)?
        .iter().map(|map| id(map, DimType::In, "V")).collect::<Result<_, _>>()?;
    let times = maps(isl(UnionMap::read_from_str(&ctx, &storage.allocation_times))?)?
        .into_iter().map(|map| Ok((id(&map, DimType::In, "V")?, map)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let live = isl(isl(UnionMap::read_from_str(&ctx, &storage.live_storages))?.reverse())?;
    let retained = isl(isl(UnionMap::read_from_str(&ctx, &storage.retained_storages))?.domain())?;
    let partitions = [isl(live.copy())?, isl(isl(live.copy())?.intersect_range(isl(retained.copy())?))?,
        isl(live.subtract_range(retained))?];
    let mut bounds = Vec::new();
    let mut fibers = BTreeMap::<String, u64>::new();
    for partition in partitions {
        let mut counts = BTreeMap::<usize, u64>::new();
        for map in maps(partition)? {
            let root = id(&map, DimType::Out, "V")?;
            let key = super::region_dependence::map_text(&isl(UnionMap::from_map(isl(map.copy())?))?)?;
            let count = if let Some(&count) = fibers.get(&key) { count } else {
                // Each fiber is an independent bounded optimization problem.
                // Reuse identical retained/transient queries, and apply the
                // existing ISL quota per query rather than accumulating it
                // across unrelated roots in a larger source program.
                ctx.reset_operations();
                let count = fiber_bound(&map, times.get(&root))?;
                fibers.insert(key, count);
                count
            };
            counts.entry(root).and_modify(|n| *n = (*n).max(count)).or_insert(count);
        }
        bounds.push(counts);
    }
    let mut root_bounds = Vec::new();
    for root in &storage.roots {
        let split = bounds[1].get(&root.value_id).copied().unwrap_or(0)
            .checked_add(bounds[2].get(&root.value_id).copied().unwrap_or(0))
            .ok_or("region retained/transient count overflows")?;
        let count = bounds[0].get(&root.value_id).copied().unwrap_or(0).min(split);
        // Borrowed roots describe original resource references, not new model
        // copies. Their residency and lifetime remain with the original owner.
        if !borrowed.contains(&root.value_id) {
            add(&mut devices.get_mut(&root.device).unwrap().live_tensor_bytes_upper_bound, count, root.bytes)?;
        }
        root_bounds.push(StorageBound { value_id: root.value_id, live_instances_upper_bound: count });
    }
    for map in maps(isl(UnionMap::read_from_str(&ctx, &storage.input_storages))?)? {
        ctx.reset_operations();
        let root = roots[&id(&map, DimType::In, "V")?];
        let set = isl(map.domain())?;
        let fiber = isl(Map::from_domain_and_range(isl(Set::read_from_str(&ctx, "{ [] }"))?, set))?;
        let count = fiber_bound(&fiber, None)?;
        let device = devices.get_mut(&root.device).unwrap();
        add(&mut device.entry_copy_bytes_upper_bound, count, root.bytes)?;
        add(&mut device.logical_read_bytes_upper_bound, count, root.bytes)?;
        add(&mut device.logical_write_bytes_upper_bound, count, root.bytes)?;
    }
    let profiles = imports.iter().map(|import| Ok((import.name(), call_memory(import, program)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let backings = region_storage::backing_bounds(program, imports)?;
    let scratch = super::region_scratch::Context { program, backings: &backings, calls: &profiles };
    let domains = isl(isl_rs::UnionSet::read_from_str(&ctx, &source.domain))?;
    let list = isl(domains.get_set_list())?;
    let mut unknown_functions = BTreeSet::new();
    for i in 0..isl(list.n_set())? {
        ctx.reset_operations();
        let domain = isl(list.get_at(i))?;
        let fiber = isl(Map::from_domain_and_range(isl(Set::read_from_str(&ctx, "{ [] }"))?, domain))?;
        let statement = &source.statements[id(&fiber, DimType::Out, "S")?];
        let count = fiber_bound(&fiber, None)?;
        if count == 0 { continue; }
        let action = super::region_control::action(region_storage::operation(program, &statement.path)?)?;
        let profile = scratch.action(&action)?;
        unknown_functions.extend(profile.unknown_functions);
        for (device_name, facts) in &profile.devices {
            // A cell can allocate a temporary on a device with no returned
            // storage roots. Keep that device in the envelope as well.
            let device = devices.entry(device_name.clone()).or_insert_with(|| DeviceCapacity {
                device: device_name.clone(), entry_copy_bytes_upper_bound: 0, result_copy_bytes_upper_bound: 0, live_tensor_bytes_upper_bound: 0,
                call_scratch_bytes_upper_bound: 0, peak_tensor_bytes_upper_bound: 0,
                logical_read_bytes_upper_bound: 0, logical_write_bytes_upper_bound: 0,
            });
            add(&mut device.logical_read_bytes_upper_bound, count, facts.logical_read_bytes)?;
            add(&mut device.logical_write_bytes_upper_bound, count, facts.logical_write_bytes)?;
            device.call_scratch_bytes_upper_bound = device.call_scratch_bytes_upper_bound.max(
                facts.scratch_bytes);
        }
    }
    if imports.iter().any(|import| matches!(import, TensorImport::Borrowed { .. })) {
        // Any result, including a view returned through atomic control or a
        // collected bank, might expose the original resource backing. The
        // executor materializes every result leaf under this common ABI.
        for type_ in super::region_program::verify(program)? {
            if let Some((name, bytes)) = native_copy_bytes(&type_, &program.parameters)? {
                let device = devices.get_mut(name).ok_or("borrowed result has no device capacity")?;
                add(&mut device.result_copy_bytes_upper_bound, 1, bytes)?;
                add(&mut device.logical_read_bytes_upper_bound, 1, bytes)?;
                add(&mut device.logical_write_bytes_upper_bound, 1, bytes)?;
            }
        }
    }
    for device in devices.values_mut() {
        device.peak_tensor_bytes_upper_bound = device.entry_copy_bytes_upper_bound.max(
            device.live_tensor_bytes_upper_bound.checked_add(device.call_scratch_bytes_upper_bound)
                .and_then(|bytes| bytes.checked_add(device.result_copy_bytes_upper_bound))
                .ok_or("region peak tensor envelope overflows")?);
    }
    let mut result = Capacity { program_digest: storage.program_digest.clone(), schedule_map: storage.schedule_map.clone(),
        imports_digest: storage.imports_digest.clone(), storage_signature: storage.signature.clone(),
        roots: root_bounds, devices: devices.into_values().collect(),
        unknown_functions: unknown_functions.into_iter().collect(), signature: String::new() };
    result.signature = digest(&canonical_ascii_json(&result)?);
    Ok(result)
}

fn reachable_region_functions(program: &Program, source: &super::region_dependence::Analysis) -> Result<BTreeSet<String>, String> {
    // Numerical policy concerns functions that can execute anywhere in the
    // source domain, not merely in the timing example (which may be zero-trip).
    let ctx = context()?;
    let domain = isl(isl_rs::UnionSet::read_from_str(&ctx, &source.domain))?;
    let mut reachable = BTreeSet::new();
    fn visit(program: &Program, op: &Operation, found: &mut BTreeSet<String>) {
        let regions = match op {
            Operation::Call { function, .. } => { found.insert(function.clone()); return; },
            Operation::If { then_region, else_region, .. } => vec![then_region, else_region],
            Operation::While { limit, condition, body, .. } if *limit > 0 => vec![condition, body],
            Operation::Scan { count, body, .. } | Operation::Map { count, body, .. } => {
                let used = match count {
                    super::region_program::Extent::Fixed(n) => *n > 0,
                    super::region_program::Extent::Parameter(name) => program.parameters.iter().any(|p| &p.name == name && p.maximum > 0),
                };
                if used { vec![body] } else { vec![] }
            },
            _ => vec![],
        };
        for region in regions { for operation in &region.block.operations { visit(program, operation, found); } }
    }
    for statement in &source.statements {
        let coordinates = (0..statement.extents.len()).map(|i| format!("i{i}")).collect::<Vec<_>>().join(",");
        let instances = isl(isl_rs::UnionSet::read_from_str(&ctx, &format!("{{ S{}[{coordinates}] }}", statement.id)))?;
        if !isl(isl(instances.intersect(isl(domain.copy())?))?.is_empty())? {
            visit(program, region_storage::operation(program, &statement.path)?, &mut reachable);
        }
    }
    Ok(reachable)
}

/// Derive each candidate's exact outlined source before legality and capacity.
/// Tensor payload is a hard partial-resource constraint; elapsed time ranks
/// feasible candidates. No FLOP/byte heuristic stands in for execution time.
pub(super) fn assess_region_candidates(program: &Program, request: &RegionRequest) -> Result<Vec<Assessment>, String> {
    let source = analyze(program, false)?;
    super::region_selection::validate_tuning_sizes(program, request)?;
    let reference = super::region_dependence::canonical_schedule(&source.reference_schedule)?;
    let source_reachable = reachable_region_functions(program, &source)?;
    if request.candidates.is_empty() || request.measurement_context.is_empty() {
        return Err("region selection requires reference execution and a measurement context".into());
    }
    let mut limits = BTreeMap::new();
    for limit in &request.payload_limits {
        if limit.device.is_empty() || limits.insert(limit.device.as_str(), limit.bytes).is_some() {
            return Err("region payload limits require unique device identities".into());
        }
    }
    let mut assessments = Vec::new();
    for (index, candidate) in request.candidates.iter().enumerate() {
        if index == 0 && (!candidate.outlines.is_empty() || candidate.stream_plan.is_some()) {
            return Err("first region candidate must be the original reference execution".into());
        }
        let (derived, baseline) = super::region_outlining::apply_evidence(program, &request.source_imports, &candidate.outlines)?;
        let payload = if let Some(plan) = &candidate.stream_plan {
            // Validation derives the source payload as part of the complete
            // stream plan. Reuse it only after that full comparison succeeds.
            super::region_streams::validate(&derived, &candidate.imports, &request.retention_imports, plan)?;
            if super::region_dependence::canonical_schedule(&candidate.schedule_map)? != plan.payload.capacity.schedule_map {
                return Err("parallel candidate and stream plan describe different phase lifetimes".into());
            }
            plan.payload.clone()
        } else {
            super::region_payload::derive(&derived, &candidate.schedule_map, &candidate.imports, &request.retention_imports)?
        };
        // Candidate zero's payload derivation validates the original imports;
        // a separate reference storage analysis would repeat that same work.
        if index == 0 && (payload.capacity.schedule_map != reference || candidate.imports != request.source_imports) {
            return Err("first region candidate must be the original reference execution".into());
        }
        let reachable = if candidate.outlines.is_empty() { source_reachable.clone() }
            else { reachable_region_functions(&derived, &analyze(&derived, false)?)? };
        let changed_functions = candidate.imports.iter().zip(&baseline)
            .filter(|(a, b)| a != b && reachable.contains(a.name()))
            .map(|(import, _)| import.name().to_owned()).collect::<Vec<_>>();
        let peak_bytes = if let Some(plan) = &candidate.stream_plan {
            plan.devices.iter().map(|d| super::region_selection::PayloadLimit {
                device: d.device.clone(), bytes: d.peak_bytes_upper_bound }).collect::<Vec<_>>()
        } else {
            payload.devices.iter().map(|d| super::region_selection::PayloadLimit {
                device: d.device.clone(), bytes: d.peak_bytes_upper_bound }).collect::<Vec<_>>()
        };
        let fits = peak_bytes.iter().all(|d| limits.get(d.device.as_str())
            .is_some_and(|&limit| d.bytes <= limit));
        let feasible = fits && (request.allow_numerical_transforms || changed_functions.is_empty());
        let mut samples = candidate.elapsed_ns.clone();
        if samples.contains(&0) {
            return Err("region candidates require positive measured execution samples".into());
        }
        samples.sort_unstable();
        let median_ns = samples.get(samples.len() / 2).copied();
        assessments.push(Assessment { payload, peak_bytes, median_ns, feasible, changed_functions });
    }
    Ok(assessments)
}

pub(super) fn select_region_candidate(program: &Program, request: &RegionRequest) -> Result<RegionSelection, String> {
    let assessments = assess_region_candidates(program, request)?;
    let mut best = None;
    for (index, assessment) in assessments.iter().enumerate() {
        if !assessment.feasible { continue; }
        let median_ns = assessment.median_ns.ok_or("feasible region candidates require execution measurements")?;
        // Source-first ties avoid numerical changes without lower measured time.
        let score = (median_ns, index);
        if best.is_none_or(|previous| score < previous) { best = Some(score); }
    }
    let (_, selected) = best.ok_or("no region candidate satisfies policy and tensor payload limits")?;
    let mut result = RegionSelection { program_digest: assessments[0].payload.capacity.program_digest.clone(),
        request_digest: digest(&canonical_ascii_json(request)?), candidates: assessments,
        selected, signature: String::new() };
    result.signature = digest(&canonical_ascii_json(&result)?);
    Ok(result)
}

struct ValueUse {
    producer: usize,
    users: BTreeSet<usize>,
    bytes: u64,
    exported: bool,
}

pub(super) struct Solution {
    pub(super) regions: Vec<(Vec<usize>, u32)>,
    pub(super) materialized_values: Vec<u32>,
    pub(super) score: [u64; 4],
    pub(super) release_values: Vec<Vec<u32>>,
    slots: Vec<u32>,
}

struct Search<'a> {
    graph: &'a SemanticGraph,
    timeline: &'a Timeline,
    index: &'a GraphIndex,
    implementations: &'a [&'a ImplementationCandidate],
    state: Option<&'a StatePhysicalPlan>,
    values: Vec<ValueUse>,
    workspace: Vec<u64>,
    best: Option<Solution>,
}

fn prepare<'a>(
    graph: &'a SemanticGraph,
    timeline: &'a Timeline,
    index: &'a GraphIndex,
    implementations: &'a [&'a ImplementationCandidate],
    state: Option<&'a StatePhysicalPlan>,
) -> Result<Search<'a>, String> {
    if (!timeline.reserved_bytes.is_empty()
        && timeline.reserved_bytes.len() != timeline.anchors.len())
        || timeline
            .task_resources
            .windows(2)
            .any(|pair| pair[0].task >= pair[1].task)
        || timeline
            .value_resources
            .windows(2)
            .any(|pair| (&pair[0].task, &pair[0].output) >= (&pair[1].task, &pair[1].output))
    {
        return Err(
            "AER resource facts require exact anchor coverage and sorted unique identities".into(),
        );
    }
    let mut workspace = vec![0; graph.tasks.len()];
    for fact in &timeline.task_resources {
        let task = graph
            .tasks
            .binary_search_by(|task| task.symbol.cmp(&fact.task))
            .map_err(|_| "AER workspace references an unknown task")?;
        workspace[task] = fact.workspace_bytes;
    }
    let mut values = Vec::new();
    for (producer, task) in graph.tasks.iter().enumerate() {
        for output in &task.outputs {
            let fact = timeline
                .value_resources
                .iter()
                .find(|fact| fact.task == task.symbol && fact.output == output.name);
            if timeline.memory_budget_bytes.is_some() && fact.is_none() {
                return Err("memory-constrained AER requires every output size, including explicit zero-byte values".into());
            }
            let mut users: BTreeSet<_> = graph.tasks.iter().enumerate().filter_map(|(user, task)|
                task.inputs.iter().any(|input| matches!(&input.source,
                    InputSource::Result { task, output: name } if task == &graph.tasks[producer].symbol && name == &output.name))
                    .then_some(user)).collect();
            // Autograd may save a consumer's inputs until its downstream gradient sink.
            for &user in &index.order {
                if graph.tasks[user].math.as_ref().is_some_and(|math|
                    graph.tasks[user].inputs.iter().any(|input| math.input_gradients.contains(&input.name)
                        && matches!(&input.source, InputSource::Result { task, .. }
                            if users.iter().any(|&parent| &graph.tasks[parent].symbol == task)))) {
                    users.insert(user);
                }
            }
            values.push(ValueUse {
                producer,
                users,
                bytes: fact.map_or(0, |fact| fact.bytes),
                exported: output.kind != OutputKind::Temporary,
            });
        }
    }
    if timeline.memory_budget_bytes.is_some() && timeline.task_resources.len() != graph.tasks.len() {
        return Err("memory-constrained AER requires every task workspace size".into());
    }
    for fact in &timeline.value_resources {
        if !graph.tasks.iter().any(|task| {
            task.symbol == fact.task && task.outputs.iter().any(|output| output.name == fact.output)
        }) {
            return Err("AER value size references an unknown output".into());
        }
    }
    values
        .iter()
        .map(|value| value.bytes)
        .chain(workspace.iter().copied())
        .chain(timeline.task_resources.iter().map(|fact| fact.resident_bytes))
        .chain(timeline.reserved_bytes.iter().copied())
        .chain(state.map(|plan| plan.reserved_bytes))
        .try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or("AER resource envelope overflows u64")?;
    Ok(Search {
        graph,
        timeline,
        index,
        implementations,
        state,
        values,
        workspace,
        best: None,
    })
}

pub(super) fn solve(
    graph: &SemanticGraph,
    timeline: &Timeline,
    index: &GraphIndex,
    implementations: &[&ImplementationCandidate],
    state: Option<&StatePhysicalPlan>,
) -> Result<Solution, String> {
    let mut search = prepare(graph, timeline, index, implementations, state)?;
    search.partition(0, &mut Vec::new(), &mut vec![usize::MAX; graph.tasks.len()]);
    search.best.ok_or_else(|| {
        "no AER partition and placement fits dependencies, windows and memory envelope".into()
    })
}

pub(super) fn check_placement(
    graph: &SemanticGraph,
    timeline: &Timeline,
    index: &GraphIndex,
    implementations: &[&ImplementationCandidate],
    state: Option<&StatePhysicalPlan>,
    tasks: &[PhysicalTask],
    region_ends: &[u32],
) -> Result<Solution, String> {
    if tasks.len() != graph.tasks.len() || region_ends.last().copied() != Some(tasks.len() as u32) {
        return Err("AER placement does not cover every task".into());
    }
    let mut search = prepare(graph, timeline, index, implementations, state)?;
    let mut groups = Vec::new();
    let mut membership = vec![usize::MAX; tasks.len()];
    let mut positions = vec![usize::MAX; tasks.len()];
    let mut task_slots = vec![0; tasks.len()];
    let mut slots = Vec::new();
    let mut start = 0;
    for &end in region_ends {
        let end = end as usize;
        if end <= start || end > tasks.len() {
            return Err("AER regions must be nonempty contiguous ranges".into());
        }
        let slot = tasks[start].anchor_id;
        let mut members = Vec::new();
        for (position, task) in tasks.iter().enumerate().take(end).skip(start) {
            let id = task.task_id as usize;
            if id >= tasks.len() || membership[id] != usize::MAX || task.anchor_id != slot
                || slot < index.windows[id].0 || slot >= index.windows[id].1
            {
                return Err("AER task identity or timing differs from its declared region".into());
            }
            if end - start > 1 && (graph.tasks[id].replay.is_some()
                || implementations[id].execution_kind != implementations[tasks[start].task_id as usize].execution_kind)
            {
                return Err("AER region crosses an execution or replay boundary".into());
            }
            membership[id] = groups.len();
            positions[id] = position;
            task_slots[id] = slot;
            members.push(id);
        }
        slots.push(slot);
        groups.push(members);
        start = end;
    }
    let mut predecessors = vec![BTreeSet::new(); groups.len()];
    for (task, parents) in index.predecessors.iter().enumerate() {
        for &parent in parents {
            if positions[parent] >= positions[task] || task_slots[parent] > task_slots[task] {
                return Err("AER placement violates an exact-version or result dependency".into());
            }
            if membership[parent] != membership[task] {
                predecessors[membership[task]].insert(membership[parent]);
            }
        }
    }
    if slots.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("AER regions are not in execution order".into());
    }
    search.evaluate(&groups, &membership, &(0..groups.len()).collect::<Vec<_>>(), &predecessors, &slots);
    search.best.ok_or_else(|| "selected AER placement exceeds its memory envelope".into())
}

impl Search<'_> {
    fn partition(
        &mut self,
        next: usize,
        groups: &mut Vec<Vec<usize>>,
        membership: &mut Vec<usize>,
    ) {
        let cut_lower_bound: u64 = self
            .values
            .iter()
            .filter(|value| {
                value.exported
                    || (value.producer < next
                        && value.users.iter().any(|&user| {
                            user < next && membership[user] != membership[value.producer]
                        }))
            })
            .map(|value| value.bytes)
            .sum();
        if self.best.as_ref().is_some_and(|best| {
            cut_lower_bound > best.score[0]
        }) {
            return;
        }
        if next == self.graph.tasks.len() {
            self.place_partition(groups, membership);
            return;
        }
        // Restricted-growth membership enumerates each partition once. Conditional
        // tasks are closed control regions and cannot absorb unconditional work.
        for group in 0..groups.len() {
            let first = groups[group][0];
            if self.graph.tasks[next].replay.is_some()
                || self.graph.tasks[first].replay.is_some()
                || self.implementations[next].execution_kind
                    != self.implementations[first].execution_kind
            {
                continue;
            }
            let lo = groups[group]
                .iter()
                .map(|&task| self.index.windows[task].0)
                .max()
                .unwrap()
                .max(self.index.windows[next].0);
            let hi = groups[group]
                .iter()
                .map(|&task| self.index.windows[task].1)
                .min()
                .unwrap()
                .min(self.index.windows[next].1);
            if lo >= hi {
                continue;
            }
            groups[group].push(next);
            membership[next] = group;
            self.partition(next + 1, groups, membership);
            groups[group].pop();
        }
        membership[next] = groups.len();
        groups.push(vec![next]);
        self.partition(next + 1, groups, membership);
        groups.pop();
        membership[next] = usize::MAX;
    }

    fn place_partition(&mut self, groups: &[Vec<usize>], membership: &[usize]) {
        let mut predecessors = vec![BTreeSet::new(); groups.len()];
        for (task, parents) in self.index.predecessors.iter().enumerate() {
            for &parent in parents {
                if membership[parent] != membership[task] {
                    predecessors[membership[task]].insert(membership[parent]);
                }
            }
        }
        let mut order = Vec::new();
        let mut pending: BTreeSet<_> = (0..groups.len()).collect();
        while let Some(group) = pending
            .iter()
            .find(|&&group| {
                predecessors[group]
                    .iter()
                    .all(|parent| !pending.contains(parent))
            })
            .copied()
        {
            pending.remove(&group);
            order.push(group);
        }
        if !pending.is_empty() {
            return;
        }
        let bounds: Vec<_> = groups
            .iter()
            .map(|members| {
                (
                    members
                        .iter()
                        .map(|&task| self.index.windows[task].0)
                        .max()
                        .unwrap(),
                    members
                        .iter()
                        .map(|&task| self.index.windows[task].1)
                        .min()
                        .unwrap(),
                )
            })
            .collect();
        self.placements(
            groups,
            membership,
            &order,
            &predecessors,
            &bounds,
            &mut vec![0; groups.len()],
            0,
        );
    }

    fn placements(
        &mut self,
        groups: &[Vec<usize>],
        membership: &[usize],
        order: &[usize],
        predecessors: &[BTreeSet<usize>],
        bounds: &[(u32, u32)],
        slots: &mut [u32],
        next: usize,
    ) {
        if next != order.len() {
            let group = order[next];
            let after = bounds[group].0.max(
                predecessors[group]
                    .iter()
                    .map(|&parent| slots[parent])
                    .max()
                    .unwrap_or(0),
            );
            for anchor in after..bounds[group].1 {
                slots[group] = anchor;
                self.placements(groups, membership, order, predecessors, bounds, slots, next + 1);
            }
            return;
        }
        self.evaluate(groups, membership, order, predecessors, slots);
    }

    fn evaluate(
        &mut self,
        groups: &[Vec<usize>],
        membership: &[usize],
        order: &[usize],
        predecessors: &[BTreeSet<usize>],
        slots: &[u32],
    ) {
        let anchors = self.timeline.anchors.len();
        let mut internal = vec![0_u128; groups.len()];
        let mut live = vec![0_u128; anchors];
        let mut materialized_values = Vec::new();
        let mut materialization = 0_u128;
        let mut placed_order = order.to_vec();
        placed_order.sort_by_key(|&group| slots[group]);
        let execution_order: Vec<_> = placed_order.iter().flat_map(|&group|
            self.index.order.iter().copied().filter(move |&task| membership[task] == group)).collect();
        let mut positions = vec![0; membership.len()];
        for (position, &task) in execution_order.iter().enumerate() { positions[task] = position; }
        let mut release_values = vec![Vec::new(); membership.len()];
        for (value_id, value) in self.values.iter().enumerate() {
            let source = membership[value.producer];
            if !value.exported {
                let last = value.users.iter().copied().chain(std::iter::once(value.producer))
                    .max_by_key(|&task| positions[task]).unwrap();
                release_values[last].push(value_id as u32);
            }
            if value.exported || value.users.iter().any(|&user| membership[user] != source) {
                materialized_values.push(value_id as u32);
                materialization += value.bytes as u128;
                let last = if value.exported {
                    anchors as u32 - 1
                } else {
                    value
                        .users
                        .iter()
                        .map(|&user| slots[membership[user]])
                        .max()
                        .unwrap_or(slots[source])
                };
                for bytes in &mut live[slots[source] as usize..=last as usize] {
                    *bytes += value.bytes as u128;
                }
            } else {
                internal[source] += value.bytes as u128;
            }
        }
        let mut scratch = vec![0_u128; anchors];
        for (group, members) in groups.iter().enumerate() {
            let bytes = internal[group]
                + members
                    .iter()
                    .map(|&task| self.workspace[task])
                    .max()
                    .unwrap() as u128;
            let slot = &mut scratch[slots[group] as usize];
            *slot = (*slot).max(bytes);
        }
        let mut resident = 0_u128;
        for fact in &self.timeline.task_resources {
            resident += fact.resident_bytes as u128;
        }
        if let Some(state) = self.state {
            for provider in &state.representations {
                resident += provider.reserved_bytes as u128;
            }
        }
        let mut peak = 0_u128;
        for anchor in 0..anchors {
            let total = self.timeline.reserved_bytes.get(anchor).copied().unwrap_or(0) as u128
                + live[anchor] + scratch[anchor] + resident;
            peak = peak.max(total);
        }
        if self.timeline.memory_budget_bytes.is_some_and(|budget| peak > budget as u128) { return; }
        let (Ok(materialization), Ok(peak)) =
            (u64::try_from(materialization), u64::try_from(peak)) else { return; };
        let score = [
            materialization,
            groups.len() as u64,
            peak,
            predecessors.iter().map(|p| p.len() as u64).sum::<u64>(),
        ];
        let task_slots: Vec<_> = membership.iter().map(|&group| slots[group]).collect();
        if self
            .best
            .as_ref()
            .is_some_and(|best| (best.score, &best.slots) <= (score, &task_slots))
        {
            return;
        }
        let regions = placed_order
            .iter()
            .map(|&group| {
                (
                    self.index
                        .order
                        .iter()
                        .copied()
                        .filter(|&task| membership[task] == group)
                        .collect(),
                    slots[group],
                )
            })
            .collect();
        self.best = Some(Solution {
            regions,
            materialized_values,
            score,
            slots: task_slots,
            release_values,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::scheduler::compile;
    use super::super::semantic_graph::{decode_graph, decode_timeline};
    use serde_json::json;
    use std::collections::BTreeSet;

    #[test]
    fn aer_matches_bruteforce_partition_oracle() {
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let windows = [(0, 1), (0, 3), (1, 3)];
        let bytes = [3_u64, 5, 7];
        let workspace = [2_u64, 4, 1];
        for mask in 0..8 {
            let arcs: Vec<_> = [(0, 1), (0, 2), (1, 2)]
                .into_iter()
                .enumerate()
                .filter_map(|(bit, arc)| (mask & (1 << bit) != 0).then_some(arc))
                .collect();
            for window_code in 0..27 {
                let bounds = [
                    windows[window_code % 3],
                    windows[window_code / 3 % 3],
                    windows[window_code / 9],
                ];
                for mixed_kind in [false, true] {
                    for budget in [8_u64, 16, 40] {
                        let reserved = [0_u64, 9, 1, 0];
                        let resident = if mixed_kind { [2_u64, 5, 1] } else { [0; 3] };
                        let graph = decode_graph(&json!({
                "workload_id": "aer_oracle", "execution_parameters_digest": "a".repeat(64),
                "schema_version": 1, "input_groups": [], "edges": [], "states": [],
                "tasks": (0..3).map(|task| json!({
                    "symbol": format!("t{task}"),
                    "timing": {"after": format!("a{}", bounds[task].0), "before": format!("a{}", bounds[task].1)},
                    "inputs": arcs.iter().filter(|(_, target)| *target == task).map(|(source, _)| json!({
                        "name": format!("from{source}"), "source": {"kind": "result", "task": format!("t{source}"), "output": "value"}
                    })).collect::<Vec<_>>(),
                    "outputs": [{"name": "value", "kind": "temporary", "schema": ""}], "effects": [],
                    "implementations": [{"implementation_id": "kernel", "execution_kind":
                        if mixed_kind && task == 1 {"host_service"} else {"native_task"}, "required_capabilities": []}]
                })).collect::<Vec<_>>()
              }).to_string()).unwrap();
                        let timeline = decode_timeline(&json!({
                "schema_version": 1, "capabilities": [],
                "anchors": (0..4).map(|anchor| json!({"symbol": format!("a{anchor}"), "ordinal": anchor})).collect::<Vec<_>>(),
                "reserved_bytes": reserved, "memory_budget_bytes": budget,
                "value_resources": (0..3).map(|task| json!({"task": format!("t{task}"), "output": "value", "bytes": bytes[task]})).collect::<Vec<_>>(),
                "task_resources": (0..3).map(|task| json!({"task": format!("t{task}"), "workspace_bytes": workspace[task],
                    "resident_bytes": resident[task]})).collect::<Vec<_>>()
              }).to_string()).unwrap();
                        let mut expected = None;
                        // Deliberately enumerate labelled partitions and per-task placements,
                        // not the production restricted-growth or region-placement recursion.
                        for labels in 0..27 {
                            let group = [labels % 3, labels / 3 % 3, labels / 9];
                            let count = group.iter().copied().collect::<BTreeSet<_>>().len();
                            if (0..3).any(|a| {
                                (0..3).any(|b| {
                                    group[a] == group[b] && mixed_kind && (a == 1) != (b == 1)
                                })
                            }) {
                                continue;
                            }
                            let legal_order = permutations.iter().any(|order| {
                                let rank =
                                    |task| order.iter().position(|&known| known == task).unwrap();
                                arcs.iter().all(|&(a, b)| rank(a) < rank(b))
                                    && !(group[order[0]] == group[order[2]]
                                        && group[order[0]] != group[order[1]])
                            });
                            if !legal_order {
                                continue;
                            }
                            for assignment in 0..27 {
                                let at = [assignment % 3, assignment / 3 % 3, assignment / 9];
                                if (0..3).any(|task| {
                                    at[task] < bounds[task].0 || at[task] >= bounds[task].1
                                }) || arcs.iter().any(|&(a, b)| at[a] > at[b])
                                    || (0..3).any(|a| {
                                        (0..3).any(|b| group[a] == group[b] && at[a] != at[b])
                                    })
                                {
                                    continue;
                                }
                                let cut: Vec<_> = (0..3)
                                    .filter(|&producer| {
                                        arcs.iter()
                                            .any(|&(a, b)| a == producer && group[a] != group[b])
                                    })
                                    .collect();
                                let mut peak = 0;
                                for anchor in 0..4 {
                                    let live: u64 = cut
                                        .iter()
                                        .filter(|&&producer| {
                                            at[producer] <= anchor
                                                && anchor
                                                    <= arcs
                                                        .iter()
                                                        .filter(|&&(a, _)| a == producer)
                                                        .map(|&(_, b)| at[b])
                                                        .max()
                                                        .unwrap()
                                        })
                                        .map(|&producer| bytes[producer])
                                        .sum();
                                    let scratch = (0..3)
                                        .filter(|&member| at[member] == anchor)
                                        .map(|member| {
                                            let internal: u64 = (0..3)
                                                .filter(|task| {
                                                    group[*task] == group[member]
                                                        && !cut.contains(task)
                                                })
                                                .map(|task| bytes[task])
                                                .sum();
                                            internal
                                                + (0..3)
                                                    .filter(|&task| group[task] == group[member])
                                                    .map(|task| workspace[task])
                                                    .max()
                                                    .unwrap()
                                        })
                                        .max()
                                        .unwrap_or(0);
                                    peak = peak.max(reserved[anchor as usize] + live + scratch + resident.iter().sum::<u64>());
                                }
                                if peak > budget {
                                    continue;
                                }
                                let fences = arcs
                                    .iter()
                                    .filter_map(|&(a, b)| {
                                        (group[a] != group[b]).then_some((group[a], group[b]))
                                    })
                                    .collect::<BTreeSet<_>>()
                                    .len() as u64;
                                let score = [
                                    cut.iter().map(|&producer| bytes[producer]).sum(),
                                    count as u64,
                                    peak,
                                    fences,
                                ];
                                let candidate = (score, at);
                                if expected.as_ref().is_none_or(|best| candidate < *best) {
                                    expected = Some(candidate);
                                }
                            }
                        }
                        let actual = compile(&graph, &timeline, None);
                        assert_eq!(
                            actual.is_ok(),
                            expected.is_some(),
                            "mask={mask}, bounds={bounds:?}, budget={budget}"
                        );
                        if let Some((score, slots)) = expected {
                            let wire = serde_json::to_value(actual.unwrap()).unwrap();
                            let mut actual_slots = [0; 3];
                            for task in wire["tasks"].as_array().unwrap() {
                                actual_slots[task["task_id"].as_u64().unwrap() as usize] =
                                    task["anchor_id"].as_u64().unwrap() as u32;
                            }
                            assert_eq!(actual_slots, slots);
                            assert_eq!(
                                [
                                    wire["materialization_bytes"].as_u64().unwrap(),
                                    wire["region_ends"].as_array().unwrap().len() as u64,
                                    wire["peak_memory_bytes"].as_u64().unwrap(),
                                    wire["fence_count"].as_u64().unwrap()
                                ],
                                score
                            );
                        }
                    }
                }
            }
        }
    }
}

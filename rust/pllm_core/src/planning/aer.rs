//! Tensor and region memory analysis used by region compilation.

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
        TensorImport::AffineScan { allocations, device, .. } => {
            let mut scratch_bytes = 0;
            for allocation in allocations { add(&mut scratch_bytes, allocation.bytes)?; }
            devices.insert(device.clone(), CallDeviceMemory { scratch_bytes, ..Default::default() });
            Ok(CallMemory { devices, unknown: true })
        },
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
            if let TensorImport::NativeKernel { allocations, device, tensor_sources, .. } = import {
                devices.entry(device.clone()).or_default();
                for allocation in allocations {
                    add(&mut devices.entry(allocation.device.clone()).or_default().scratch_bytes, allocation.bytes)?;
                }
                // A host-indexed dispatch executes exactly one captured graph.
                // Its private arguments are already counted above; all new
                // storage, including raw results, lives through wrapper copies.
                let mut peak = 0_u64;
                for source in tensor_sources {
                    for profile in tensor_memory(source)?.devices {
                        peak = peak.max(profile.peak_new_tensor_bytes);
                    }
                }
                add(&mut devices.entry(device.clone()).or_default().scratch_bytes, peak)?;
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
    let live = isl(isl(UnionMap::read_from_str(&ctx, &storage.live_times)).map_err(|e| format!("read storage liveness: {e}"))?.reverse())?;
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
                let count = fiber_bound(&map, times.get(&root)).map_err(|e| format!("root {root} fiber: {e}"))?;
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
        let count = fiber_bound(&fiber, None).map_err(|e| format!("input or statement fiber: {e}"))?;
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
        let count = fiber_bound(&fiber, None).map_err(|e| format!("input or statement fiber: {e}"))?;
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
        if index == 0 && (!candidate.outlines.is_empty() || candidate.stream_plan.is_some() || candidate.affine_scan.is_some()) {
            return Err("first region candidate must be the original reference execution".into());
        }
        let (mut derived, baseline) = super::region_outlining::apply_evidence(program, &request.source_imports, &candidate.outlines)?;
        let mut affine_name = None;
        if let Some(evidence) = &candidate.affine_scan {
            if !candidate.outlines.is_empty() || candidate.stream_plan.is_some() {
                return Err("affine scan must derive directly from its source region".into());
            }
            let plan = super::affine_scan::derive(program, &request.source_imports, evidence)?
                .ok_or("source does not prove the affine scan rewrite")?;
            let mut expected = request.source_imports.clone();
            affine_name = Some(plan.implementation.name().to_owned());
            expected.push(plan.implementation);
            expected.sort_by(|a, b| a.name().cmp(b.name()));
            if candidate.imports != expected { return Err("affine scan imports differ from the derived kernel".into()); }
            derived = plan.program;
        }
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
        let mut changed_functions = candidate.imports.iter().zip(&baseline)
            .filter(|(a, b)| a != b && reachable.contains(a.name()))
            .map(|(import, _)| import.name().to_owned()).collect::<Vec<_>>();
        if let Some(name) = affine_name { changed_functions = vec![name]; }
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

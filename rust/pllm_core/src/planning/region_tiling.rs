//! Bounded schedule-tree tiling over source-derived permutable ISL bands.

use std::cell::RefCell;
use std::collections::{BTreeSet, VecDeque};
use isl_rs::{Context, DimType, MultiUnionPwAff, MultiVal, Schedule, ScheduleNode, ScheduleNodeType, Set, UnionMap, UnionSet, Val};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_dependence::{analyze, context, equivalent_order, isl, legal, map_text, schedule_text, Analysis, Result, ScheduleCandidate};
use super::region_program::Program;
use super::semantic_graph::{canonical_ascii_json, digest};
use super::region_capacity::maps;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Exploration {
    pub program_digest: String,
    pub candidates: Vec<ScheduleCandidate>,
    pub diagnostics: Vec<String>,
}

/// Physical windows have finite bounds. ISL computes their ready frontiers
/// over the union of all admitted sizes; no model coordinates or wave formula
/// enter the construction. Recheck the result against the full symbolic domain.
pub(super) fn frontier(program: &Program, max_instances: u64) -> Result<ScheduleCandidate> {
    let source = analyze(program, false)?;
    let bound = source.statements.iter().try_fold(0_u64, |sum, statement| {
        let count = statement.extents.iter().try_fold(1_u64, |n, extent|
            n.checked_mul(extent.maximum(program))).ok_or("frontier domain overflows")?;
        sum.checked_add(count).ok_or("frontier domain overflows")
    })?;
    if bound > max_instances { return Err("reduce the physical window for bounded frontier planning".into()); }
    let ctx = context()?;
    let domain = isl(UnionSet::read_from_str(&ctx, &source.domain))?;
    let dependencies = isl(UnionMap::read_from_str(&ctx, &source.dependencies))?;
    let mut remaining = isl(isl(domain.copy())?.project_out_all_params())?;
    let mut edges = isl(isl(dependencies.copy())?.project_out_all_params())?;
    let mut times = isl(UnionMap::read_from_str(&ctx, "{ }"))?;
    let mut wave = 0_u64;
    while !isl(remaining.is_empty())? {
        let ready = isl(isl(remaining.copy())?.subtract(isl(isl(edges.copy())?.range())?))?;
        if isl(ready.is_empty())? {
            return Err("size-independent frontier would contain a dependence cycle".into());
        }
        times = isl(times.union(frontier_times(&ready, wave, &ctx)?))?;
        remaining = isl(remaining.subtract(ready))?;
        edges = isl(isl(edges.intersect_domain(isl(remaining.copy())?))?
            .intersect_range(isl(remaining.copy())?))?;
        wave += 1;
    }
    let times = isl(isl(times.coalesce())?.intersect_domain(isl(domain.copy())?))?;
    legal(&domain, &dependencies, &times)?;
    let tree = isl(isl(Schedule::from_domain(domain))?.insert_partial_schedule(
        isl(MultiUnionPwAff::from_union_map(isl(times.copy())?))?))?;
    Ok(ScheduleCandidate {
        source: "isl-bounded-frontier".into(),
        schedule_map: map_text(&times)?,
        schedule_tree: schedule_text(&tree)?,
    })
}

fn frontier_times(ready: &UnionSet, wave: u64, ctx: &Context) -> Result<UnionMap> {
    let list = isl(ready.get_set_list())?;
    let mut sets = (0..isl(list.n_set())?).map(|i| {
        let set = isl(list.get_at(i))?;
        Ok((isl(set.get_tuple_name())?.to_owned(), set))
    }).collect::<Result<Vec<_>>>()?;
    sets.sort_by(|a, b| a.0.cmp(&b.0));
    let mut result = isl(UnionMap::read_from_str(ctx, "{ }"))?;
    let mut lane_order = 0_u64;
    for (_, mut set) in sets {
        while !isl(set.is_empty())? {
            let point = isl(isl(isl(set.copy())?.lexmin())?.sample_point())?;
            let cell = isl(Set::from_point(point))?;
            set = isl(set.subtract(isl(cell.copy())?))?;
            let range = isl(UnionSet::read_from_str(ctx, &format!("{{ [{wave}, {lane_order}] }}")))?;
            result = isl(result.union(isl(UnionMap::from_domain_and_range(
                isl(UnionSet::from_set(cell))?, range))?))?;
            lane_order += 1;
        }
    }
    Ok(result)
}

#[pyfunction]
fn derive_region_frontier(program_json: &str, max_instances: u64) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&frontier(&program, max_instances)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

fn bands(node: &ScheduleNode, path: &mut Vec<i32>, found: &mut Vec<Vec<i32>>) -> Result<()> {
    if isl(node.get_type())? == ScheduleNodeType::Band
        && isl(node.band_get_permutable())? && isl(node.band_n_member())? >= 2
    {
        found.push(path.clone());
    }
    for i in 0..isl(node.n_children())? {
        path.push(i);
        bands(&isl(node.get_child(i))?, path, found)?;
        path.pop();
    }
    Ok(())
}

fn band_node(schedule: &Schedule, path: &[i32]) -> Result<ScheduleNode> {
    let mut node = isl(schedule.get_root())?;
    for &index in path { node = isl(node.child(index))?; }
    Ok(node)
}

fn shapes(rank: usize, factors: &[u64]) -> Vec<Vec<u64>> {
    let mut result = factors.iter().map(|&size| vec![size; rank]).collect::<Vec<_>>();
    // Also vary one axis between neighboring factors. Rectangular tiles can
    // expose different reuse/frontier tradeoffs without an exponential product.
    for pair in factors.windows(2) {
        for (base, other) in [(pair[0], pair[1]), (pair[1], pair[0])] {
            for axis in 0..rank {
                let mut shape = vec![base; rank];
                shape[axis] = other;
                if !result.contains(&shape) { result.push(shape); }
            }
        }
    }
    result
}

fn tile(schedule: &Schedule, path: &[i32], shape: &[u64], ctx: &Context) -> Result<Schedule> {
    let node = band_node(schedule, path)?;
    let mut sizes = isl(MultiVal::zero(isl(node.band_get_space())?))?;
    for (member, &size) in shape.iter().enumerate() {
        sizes = isl(sizes.set_val(member as i32, isl(Val::int_from_si(ctx, size as i64))?))?;
    }
    isl(isl(node.band_tile(sizes))?.get_schedule())
}

fn tile_candidates<'a>(source: &'a Analysis, tile_sizes: &[u64], max_candidates: u64, ctx: &'a Context)
    -> Result<impl Iterator<Item = Result<ScheduleCandidate>> + 'a>
{
    let domain = isl(UnionSet::read_from_str(ctx, &source.domain))?;
    let dependencies = isl(UnionMap::read_from_str(ctx, &source.dependencies))?;
    let mut schedules = Vec::new();
    let mut recipes = Vec::new();
    // Enumerate only paths and integer shapes here. Building a tiled map,
    // proving its legality and printing its tree wait until the next request.
    for base in &source.candidates {
        if base.schedule_tree.is_empty() { continue; }
        ctx.reset_operations();
        let schedule = isl(Schedule::read_from_str(ctx, &base.schedule_tree))?;
        let mut paths = Vec::new();
        bands(&isl(schedule.get_root())?, &mut Vec::new(), &mut paths)?;
        for (band, path) in paths.into_iter().enumerate() {
            let rank = isl(band_node(&schedule, &path)?.band_n_member())? as usize;
            for shape in shapes(rank, tile_sizes) {
                recipes.push((schedules.len(), band, path.clone(), shape));
            }
        }
        schedules.push((base, schedule));
    }
    let mut recipes = recipes.into_iter();
    let mut seen = source.candidates.iter().map(|c| c.schedule_map.clone()).collect::<BTreeSet<_>>();
    let mut emitted = 0;
    Ok(std::iter::from_fn(move || {
        if emitted >= max_candidates { return None; }
        loop {
            let (base_index, band, path, shape) = recipes.next()?;
            let (base, schedule) = &schedules[base_index];
            ctx.reset_operations();
            let candidate = (|| {
                let tiled = tile(schedule, &path, &shape, ctx)?;
                let map = isl(isl(tiled.get_map())?.intersect_domain(isl(domain.copy())?))?;
                // Band permutability never replaces the complete legality check.
                legal(&domain, &dependencies, &map)?;
                Ok::<_, String>(ScheduleCandidate {
                    source: format!("{}-tile-{}-band-{band}", base.source,
                        shape.iter().map(u64::to_string).collect::<Vec<_>>().join("x")),
                    schedule_map: map_text(&map)?, schedule_tree: schedule_text(&tiled)?,
                })
            })();
            match candidate {
                Ok(candidate) if seen.insert(candidate.schedule_map.clone()) => {
                    emitted += 1;
                    return Some(Ok(candidate));
                },
                Ok(_) => {},
                Err(error) => return Some(Err(format!("tile candidate discarded: {error}"))),
            }
        }
    }))
}

pub(super) fn explore(program: &Program, tile_sizes: &[u64], max_candidates: u64) -> Result<Exploration> {
    if tile_sizes.iter().any(|&size| size < 2 || size > i64::MAX as u64)
        || tile_sizes.iter().collect::<BTreeSet<_>>().len() != tile_sizes.len()
    {
        return Err("tile sizes must be distinct integers from 2 through i64::MAX".into());
    }
    let source = analyze(program, true)?;
    let mut result = Exploration { program_digest: source.program_digest.clone(),
        candidates: vec![], diagnostics: source.diagnostics.clone() };
    if tile_sizes.is_empty() || max_candidates == 0 { return Ok(result); }
    let ctx = context()?;
    for candidate in tile_candidates(&source, tile_sizes, max_candidates, &ctx)? {
        match candidate {
            Ok(candidate) => result.candidates.push(candidate),
            Err(error) => result.diagnostics.push(error),
        }
    }
    Ok(result)
}

#[pyfunction]
fn explore_region_tiling(program_json: &str, tile_sizes_json: &str, max_candidates: u64) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let sizes: Vec<u64> = serde_json::from_str(tile_sizes_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&explore(&program, &sizes, max_candidates)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Retain a coarse lexical prefix of the reference, then use the ISL order
/// inside each window. The reference alternates operation positions and loop
/// indices; depth d cuts its first 2*d coordinates at the last loop index.
/// This bounds movement, not memory: imported retention must still be checked.
fn window(reference: &UnionMap, candidate: &UnionMap, size: u64, depth: usize, ctx: &Context) -> Result<UnionMap> {
    let pieces = maps(isl(reference.copy())?)?;
    let width = isl(pieces.first().ok_or("window requires a nonempty source domain")?.dim(DimType::Out))? as usize;
    let prefix = depth.checked_mul(2).ok_or("window depth overflows")?;
    if prefix > width { return Err("window depth exceeds the reference schedule rank".into()); }
    let coordinates = (0..width).map(|i| format!("t{i}")).collect::<Vec<_>>();
    let mut targets = coordinates[..prefix].to_vec();
    targets[prefix - 1] = format!("floor({}/{size})", targets[prefix - 1]);
    let projection = isl(UnionMap::read_from_str(ctx, &format!("{{ [{}] -> [{}] }}", coordinates.join(","), targets.join(","))))?;
    let prefix = isl(isl(reference.copy())?.apply_range(projection))?;
    isl(isl(prefix.flat_range_product(isl(candidate.copy())?))?.coalesce())
}

fn window_candidates<'a>(source: &'a Analysis, sizes: &'a [u64], depth: usize,
                         max_candidates: u64, ctx: &'a Context)
    -> Result<impl Iterator<Item = Result<ScheduleCandidate>> + 'a>
{
    let domain = isl(UnionSet::read_from_str(ctx, &source.domain))?;
    let dependencies = isl(UnionMap::read_from_str(ctx, &source.dependencies))?;
    let reference = isl(UnionMap::read_from_str(ctx, &source.reference_schedule))?;
    let applicable = !isl(domain.is_empty())? && source.statements.iter().any(|s| s.extents.len() >= depth);
    let mut choices = sizes.iter().flat_map(move |&size|
        source.candidates.iter().skip(1).map(move |base| (size, base)));
    let mut seen = BTreeSet::new();
    let mut emitted = 0;
    Ok(std::iter::from_fn(move || {
        if !applicable || emitted >= max_candidates { return None; }
        loop {
            let (size, base) = choices.next()?;
            ctx.reset_operations();
            // Enumerate constructor-faithful windows. Semantic duplicates of the
            // reference or the base are the shared catalog merge's job: it holds
            // the exact (budgeted) order comparison with the reference first.
            let build = (|| {
                let candidate = isl(UnionMap::read_from_str(ctx, &base.schedule_map))?;
                let map = window(&reference, &candidate, size, depth, ctx)?;
                legal(&domain, &dependencies, &map)?;
                let schedule = isl(isl(Schedule::from_domain(isl(domain.copy())?))?
                    .insert_partial_schedule(isl(isl(map.copy())?.as_multi_union_pw_aff())?))?;
                Ok::<_, String>(ScheduleCandidate {
                    source: format!("{}-window-{size}-depth-{depth}", base.source),
                    schedule_map: map_text(&map)?, schedule_tree: schedule_text(&schedule)?,
                })
            })();
            match build {
                Ok(candidate) if seen.insert(candidate.schedule_map.clone()) => {
                    emitted += 1;
                    return Some(Ok(candidate));
                },
                Ok(_) => {},
                Err(error) => return Some(Err(format!("window candidate discarded: {error}"))),
            }
        }
    }))
}

pub(super) fn explore_windows(program: &Program, sizes: &[u64], depth: usize, max_candidates: u64) -> Result<Exploration> {
    if depth == 0 || sizes.iter().any(|&n| n == 0 || n > i64::MAX as u64)
        || sizes.iter().collect::<BTreeSet<_>>().len() != sizes.len()
    { return Err("windows require positive loop depth and distinct positive index sizes".into()); }
    let source = analyze(program, true)?;
    let mut result = Exploration { program_digest: source.program_digest.clone(),
        candidates: vec![], diagnostics: source.diagnostics.clone() };
    if sizes.is_empty() || max_candidates == 0 { return Ok(result); }
    let ctx = context()?;
    for candidate in window_candidates(&source, sizes, depth, max_candidates, &ctx)? {
        match candidate {
            Ok(candidate) => result.candidates.push(candidate),
            Err(error) => result.diagnostics.push(error),
        }
    }
    Ok(result)
}

#[pyfunction]
fn explore_region_windows(program_json: &str, sizes_json: &str, depth: usize, max_candidates: u64) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let sizes: Vec<u64> = serde_json::from_str(sizes_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&explore_windows(&program, &sizes, depth, max_candidates)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

/// One source-derived catalog for tensor and native model regions. Keep the
/// original execution first, then give global, bounded and tiled schedules a
/// share of the search. Capacity and completed timing decide between them.
pub(super) fn explore_schedules(program: &Program, max_candidates: u64) -> Result<Exploration> {
    thread_local! {
        static CATALOGS: RefCell<VecDeque<(String, Exploration)>> = const { RefCell::new(VecDeque::new()) };
    }
    // Serial and physical-stream exploration consume the same source catalog.
    // Keep owned compiler data under the full input key; supplied proposals
    // never populate this cache and still require independent verification.
    let key = digest(&canonical_ascii_json(&(program, max_candidates))?);
    super::region_cache::memoized(&CATALOGS, key, || derive_schedules(program, max_candidates))
}

fn derive_schedules(program: &Program, max_candidates: u64) -> Result<Exploration> {
    if max_candidates == 0 { return Err("schedule catalog requires a positive candidate limit".into()); }
    let source = analyze(program, max_candidates > 1)?;
    let mut result = Exploration { program_digest: source.program_digest.clone(),
        candidates: vec![source.candidates[0].clone()], diagnostics: source.diagnostics.clone() };
    if max_candidates == 1 { return Ok(result); }
    let ctx = context()?;
    let mut schedules = vec![isl(UnionMap::read_from_str(&ctx, &result.candidates[0].schedule_map))?];
    // Semantic uniqueness lives only in this merge. Text equality filters
    // identical maps for free; the exact order comparison is the fallback for
    // textually distinct survivors. Every family is emission-capped, so the
    // total number of comparisons is structurally bounded.
    let mut seen: BTreeSet<String> = result.candidates.iter().map(|c| c.schedule_map.clone()).collect();
    let mut global = source.candidates.iter().skip(1).cloned().map(Ok);
    let mut windows = window_candidates(&source, &[2, 4], 1, max_candidates - 1, &ctx)?;
    let mut tiles = tile_candidates(&source, &[2, 4, 8, 16], max_candidates - 1, &ctx)?;
    let mut families: [&mut dyn Iterator<Item = Result<ScheduleCandidate>>; 3] =
        [&mut global, &mut windows, &mut tiles];
    loop {
        let mut advanced = false;
        for family in &mut families {
            // A family's turn yields one distinct candidate: duplicates and
            // errors keep pulling without consuming the turn. This preserves
            // the catalog's round-robin merge order exactly.
            let candidate = loop {
                match family.next() {
                    Some(Err(error)) => {
                        if !result.diagnostics.contains(&error) { result.diagnostics.push(error); }
                    },
                    Some(Ok(candidate)) => {
                        if !seen.contains(&candidate.schedule_map) {
                            ctx.reset_operations();
                            let distinct = (|| {
                                let map = isl(UnionMap::read_from_str(&ctx, &candidate.schedule_map))?;
                                for previous in &schedules {
                                    if equivalent_order(&map, previous)? { return Ok(None); }
                                }
                                Ok::<_, String>(Some(map))
                            })();
                            match distinct {
                                Ok(Some(map)) => {
                                    seen.insert(candidate.schedule_map.clone());
                                    schedules.push(map);
                                    break Some(candidate);
                                },
                                Ok(None) => {},
                                Err(error) => result.diagnostics.push(
                                    format!("schedule catalog candidate discarded: {error}")),
                            }
                        }
                    },
                    None => break None,
                }
            };
            let Some(candidate) = candidate else { continue; };
            advanced = true;
            result.candidates.push(candidate);
            if result.candidates.len() as u64 >= max_candidates { return Ok(result); }
        }
        if !advanced { return Ok(result); }
    }
}

#[pyfunction]
fn explore_region_schedules(program_json: &str, max_candidates: u64) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&explore_schedules(&program, max_candidates)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(derive_region_frontier, module)?)?;
    module.add_function(wrap_pyfunction!(explore_region_tiling, module)?)?;
    module.add_function(wrap_pyfunction!(explore_region_windows, module)?)?;
    module.add_function(wrap_pyfunction!(explore_region_schedules, module)?)?;
    Ok(())
}

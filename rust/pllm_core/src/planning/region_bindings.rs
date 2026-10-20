//! Value routing and last-use times are compiler products, not hot ISL queries.
use std::cell::RefCell;
use std::collections::VecDeque;
use isl_rs::{Context, DimType, UnionMap, UnionSet};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_ast::{accessor, Expression};
use super::region_dependence::{
    analyze, canonical_schedule, context, isl, legal, map_text, read_schedule, Result, ValueSite,
};
use super::region_program::Program;
use super::semantic_graph::{canonical_ascii_json, digest};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AccessCase {
    target_id: usize,
    condition: Expression,
    coordinates: Vec<Expression>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct AccessTable {
    pub input_id: usize,
    rank: u32,
    cases: Vec<AccessCase>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RegionBindings {
    version: u32,
    program_digest: String,
    schedule_map: String,
    values: Vec<ValueSite>,
    arguments: Vec<usize>,
    results: Vec<usize>,
    sources: Vec<AccessTable>,
    scalars: Vec<AccessTable>,
    times: Vec<AccessTable>,
    last_uses: Vec<AccessTable>,
    retained: Vec<AccessTable>,
    resource_uses: Vec<AccessTable>,
    signature: String,
}

fn tuple_id(map: &isl_rs::Map, dim: DimType, prefix: char) -> Result<usize> {
    isl(map.get_tuple_name(dim))?
        .strip_prefix(prefix)
        .ok_or("unexpected routing tuple")?
        .parse::<usize>()
        .map_err(|e| e.to_string())
}

pub(super) fn tables(
    map: UnionMap,
    ctx: &Context,
    input_prefix: char,
    targets: bool,
) -> Result<Vec<AccessTable>> {
    let maps = isl(map.get_map_list())?;
    let mut result = std::collections::BTreeMap::<usize, AccessTable>::new();
    for i in 0..isl(maps.n_map())? {
        // Each accessor is an independent bounded compilation, like an AER root.
        ctx.reset_operations();
        let map = isl(maps.get_at(i))?;
        if isl(map.is_empty())? {
            continue;
        }
        if !isl(map.is_single_valued())? {
            return Err("runtime accessor is not single-valued".into());
        }
        let input_id = tuple_id(&map, DimType::In, input_prefix)?;
        let target_id = if targets {
            tuple_id(&map, DimType::Out, 'V')?
        } else {
            0
        };
        let rank = u32::try_from(isl(map.dim(DimType::In))?).map_err(|e| e.to_string())?;
        let (condition, coordinates) = accessor(map, ctx).map_err(|e| format!("accessor {input_prefix}{input_id}: {e}"))?;
        let table = result.entry(input_id).or_insert_with(|| AccessTable {
            input_id,
            rank,
            cases: vec![],
        });
        if table.rank != rank {
            return Err("inconsistent value accessor rank".into());
        }
        table.cases.push(AccessCase {
            target_id,
            condition,
            coordinates,
        });
    }
    for table in result.values_mut() {
        table.cases.sort_by_key(|c| c.target_id);
    }
    Ok(result.into_values().collect())
}

fn lower(program: &Program, schedule_map: &str) -> Result<RegionBindings> {
    thread_local! {
        static LOWERED: RefCell<VecDeque<(String, RegionBindings)>> = const { RefCell::new(VecDeque::new()) };
    }
    let schedule = canonical_schedule(schedule_map)?;
    let key = digest(&canonical_ascii_json(&(program, &schedule))?);
    super::region_cache::memoized(&LOWERED, key, || derive_lowering(program, &schedule))
}

fn derive_lowering(program: &Program, schedule_map: &str) -> Result<RegionBindings> {
    let analysis = analyze(program, false)?;
    if !analysis.exact_provenance {
        return Err("tensor binding requires exact value provenance".into());
    }
    let ctx = context()?;
    let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain))?;
    let dependencies = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies))?;
    let schedule = read_schedule(&ctx, schedule_map).map_err(|e| format!("bindings schedule: {e}"))?;
    legal(&domain, &dependencies, &schedule).map_err(|e| format!("bindings legality: {e}"))?;
    ctx.reset_operations();
    let sources = isl(UnionMap::read_from_str(&ctx, &analysis.value_sources))?;
    let consumers = isl(UnionMap::read_from_str(&ctx, &analysis.value_consumers))?;
    let last_uses = isl(isl(consumers.apply_range(isl(schedule.copy())?))?.lexmax())?;
    ctx.reset_operations();
    let result_domains = analysis
        .results
        .iter()
        .map(|&id| {
            let value = &analysis.values[id];
            format!(
                "V{id}[{}]",
                (0..value.bank_shape.len())
                    .map(|i| format!("b{i}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    let result_domains = isl(UnionSet::read_from_str(
        &ctx,
        &format!("{{ {result_domains} }}"),
    ))?;
    let retained = isl(isl(sources.copy())?.intersect_domain(result_domains))?;
    let retained = isl(isl(retained.range())?.identity())?;
    ctx.reset_operations();
    let reads = isl(UnionMap::read_from_str(&ctx, &analysis.resource_dependencies.reads))?;
    let writes = isl(UnionMap::read_from_str(&ctx, &analysis.resource_dependencies.writes))?;
    let state_domains = program.resources.iter().enumerate().filter_map(|(i, resource)| {
        matches!(resource, super::region_program::Resource::State { .. }).then(|| {
            format!("R{i}[{}]", (0..resource.domain().len()).map(|d| format!("r{d}")).collect::<Vec<_>>().join(","))
        })
    }).collect::<Vec<_>>().join(";");
    let used = isl(isl(isl(isl(reads.union(writes))?.intersect_domain(isl(domain.copy())?))?.range())?.intersect(
        isl(UnionSet::read_from_str(&ctx, &format!("{{ {state_domains} }}")))?))?;
    let mut bindings = RegionBindings {
        version: 2,
        program_digest: analysis.program_digest,
        schedule_map: map_text(&schedule)?,
        sources: tables(sources, &ctx, 'V', true)?,
        scalars: tables(
            isl(UnionMap::read_from_str(&ctx, &analysis.scalar_values))?,
            &ctx,
            'V',
            false,
        )?,
        times: tables(schedule, &ctx, 'S', false)?,
        last_uses: tables(last_uses, &ctx, 'V', false)?,
        retained: tables(retained, &ctx, 'V', true)?,
        resource_uses: tables(isl(used.identity())?, &ctx, 'R', false)?,
        values: analysis.values,
        arguments: analysis.arguments,
        results: analysis.results,
        signature: String::new(),
    };
    bindings.signature = digest(&canonical_ascii_json(&bindings)?);
    Ok(bindings)
}

#[pyfunction]
fn lower_region_bindings(program_json: &str, schedule_map: &str) -> PyResult<String> {
    let program: Program = serde_json::from_str(program_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    canonical_ascii_json(
        &lower(&program, schedule_map).map_err(pyo3::exceptions::PyValueError::new_err)?,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_bindings(program_json: &str, bindings_json: &str) -> PyResult<()> {
    let validate = || -> Result<()> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let supplied: RegionBindings =
            serde_json::from_str(bindings_json).map_err(|e| e.to_string())?;
        if supplied != lower(&program, &supplied.schedule_map)? {
            return Err(
                "value bindings differ from the original program and fixed schedule".into(),
            );
        }
        Ok(())
    };
    validate().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lower_region_bindings, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_bindings, module)?)?;
    Ok(())
}

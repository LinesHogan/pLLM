//! Parameter-preserving polyhedral fiber bounds for the original AER.

use isl_rs::{DimType, Map, UnionMap};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::{aer, region_storage};
use super::region_dependence::isl;
use super::region_program::Program;
use super::semantic_graph::canonical_ascii_json;
use super::region_imports::TensorImport;

type Result<T> = std::result::Result<T, String>;

/// Bound every fiber separately, then maximize its widths over time AND the
/// declared parameter context. Moving parameters into the input dimensions
/// prevents counting allocations from different invocation sizes together.
fn box_bound(map: &Map) -> Result<u64> {
    if isl(map.is_empty())? { return Ok(0); }
    let parameters = isl(map.dim(DimType::Param))? as u32;
    let inputs = isl(map.dim(DimType::In))? as u32;
    let map = if parameters == 0 { isl(map.copy())? } else {
        isl(isl(map.copy())?.move_dims(DimType::In, inputs, DimType::Param, 0, parameters))?
    };
    let mut count = 1_u64;
    for axis in 0..isl(map.dim(DimType::Out))? {
        let maximum = isl(isl(map.copy())?.dim_max(axis as i32))?;
        let minimum = isl(isl(map.copy())?.dim_min(axis as i32))?;
        // Widths of integer fibers are integral on their guarded domain, but
        // window floors can leave a rational spelling (i/2 on even i). Make
        // that integrality explicit before ILP optimization. Projecting the
        // entire range also works, but needlessly eliminates every input axis.
        let widths = isl(isl(maximum.sub(minimum))?.floor())?;
        let width = isl(widths.max_val())?;
        if !isl(width.is_int())? || isl(width.is_infty())? || isl(width.is_nan())?
            || isl(width.gt_si(i64::MAX))?
        {
            // isl_val_get_num_si truncates large integers. Check the range
            // while it is still an arbitrary-precision ISL value.
            return Err("region capacity requires finite integer fiber widths fitting i64".into());
        }
        let width = u64::try_from(isl(width.get_num_si())?).map_err(|_| "negative region fiber width")?;
        count = count.checked_mul(width.checked_add(1).ok_or("region fiber width overflows")?)
            .ok_or("region fiber volume overflows u64")?;
    }
    Ok(count)
}

/// A bounded set of coordinate projections. Dropping an axis is safe only
/// when the projected coordinate map still uniquely identifies every root.
fn coordinate_bounds(fiber: &Map, coordinates: &Map) -> Result<u64> {
    let mut coordinates = isl(coordinates.copy())?;
    let mut axis = isl(coordinates.dim(DimType::Out))?;
    while axis > 0 {
        axis -= 1;
        let projected = isl(isl(coordinates.copy())?.project_out(DimType::Out, axis as u32, 1))?;
        if isl(projected.is_injective())? {
            coordinates = projected;
        }
    }
    // Projection leaves every remaining coordinate's fiber extrema unchanged.
    // It only removes width factors >= 1, so the final injective projection
    // is at least as tight as every intermediate box. Solve its extrema once,
    // instead of repeating integer optimization after every dropped axis.
    box_bound(&isl(isl(fiber.copy())?.apply_range(coordinates))?)
}

pub(super) fn fiber_bound(fiber: &Map, allocation_times: Option<&Map>) -> Result<u64> {
    if isl(fiber.is_empty())? { return Ok(0); }
    // Time (and parameters) determine at most one live root. This relational
    // proof avoids repeated integer optimization of singleton temporaries.
    if isl(fiber.is_single_valued())? { return Ok(1); }
    let identity = isl(isl(isl(fiber.copy())?.range())?.identity())?;
    let mut bound = coordinate_bounds(fiber, &identity)?;
    if let Some(times) = allocation_times {
        if isl(times.is_injective())?
            && isl(isl(isl(fiber.copy())?.range())?.is_subset(&isl(isl(times.copy())?.domain())?))?
        {
            bound = bound.min(coordinate_bounds(fiber, times)?);
        }
    }
    Ok(bound)
}

pub(super) fn maps(map: UnionMap) -> Result<Vec<Map>> {
    let list = isl(map.get_map_list())?;
    (0..isl(list.n_map())?).map(|i| isl(list.get_at(i))).collect()
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StorageBound { pub value_id: usize, pub live_instances_upper_bound: u64 }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct DeviceCapacity {
    pub device: String,
    pub entry_copy_bytes_upper_bound: u64,
    pub result_copy_bytes_upper_bound: u64,
    pub live_tensor_bytes_upper_bound: u64,
    pub call_scratch_bytes_upper_bound: u64,
    pub peak_tensor_bytes_upper_bound: u64,
    pub logical_read_bytes_upper_bound: u64,
    pub logical_write_bytes_upper_bound: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Capacity {
    pub program_digest: String,
    pub schedule_map: String,
    pub imports_digest: String,
    pub storage_signature: String,
    pub roots: Vec<StorageBound>,
    pub devices: Vec<DeviceCapacity>,
    pub unknown_functions: Vec<String>,
    pub signature: String,
}

#[pyfunction]
fn analyze_region_capacity(program_json: &str, schedule_map: &str, imports_json: &str) -> PyResult<String> {
    let run = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let storage = region_storage::derive(&program, schedule_map, &imports)?;
        canonical_ascii_json(&aer::region_tensor_memory(&program, &storage, &imports)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_capacity(program_json: &str, imports_json: &str, capacity_json: &str) -> PyResult<()> {
    let run = || -> Result<()> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let supplied: Capacity = serde_json::from_str(capacity_json).map_err(|e| e.to_string())?;
        let storage = region_storage::derive(&program, &supplied.schedule_map, &imports)?;
        if supplied != aer::region_tensor_memory(&program, &storage, &imports)? {
            return Err("region capacity differs from the source storage envelope".into());
        }
        Ok(())
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze_region_capacity, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_capacity, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::region_dependence::context;

    #[test]
    fn parameter_values_are_not_counted_as_simultaneous_allocations() {
        let ctx = context().unwrap();
        for (text, expected) in [
            ("[N] -> { S[t] -> V[i] : 0 <= N <= 1000000 and 0 <= t < N and i = t }", 1),
            ("[N] -> { S[t] -> V[i] : 2 <= N <= 12 and -N <= t <= N and t - 2 <= i <= t }", 3),
            ("{ S[t] -> V[i] : false }", 0),
            ("{ S[t] -> V[] : 0 <= t < 10 }", 1),
        ] {
            let map = Map::read_from_str(&ctx, text).unwrap();
            assert_eq!(fiber_bound(&map, None).unwrap(), expected);
        }
    }

    #[test]
    fn wavefront_coordinates_tighten_bounds_only_when_they_cover_every_root() {
        let ctx = context().unwrap();
        let live = Map::read_from_str(&ctx,
            "{ S[t] -> V[i,j] : 0 <= t < 9 and 0 <= i <= t and j = t-i }").unwrap();
        let full = Map::read_from_str(&ctx, "{ V[i,j] -> [i+j,i] }").unwrap();
        let partial = Map::read_from_str(&ctx, "{ V[i,j] -> [i] : j = 0 }").unwrap();
        let noninjective = Map::read_from_str(&ctx, "{ V[i,j] -> [i+j] }").unwrap();
        assert_eq!(fiber_bound(&live, Some(&full)).unwrap(), 9);
        // Both alternatives would lose roots; neither may replace the valid box.
        assert_eq!(fiber_bound(&live, Some(&partial)).unwrap(), 81);
        assert_eq!(fiber_bound(&live, Some(&noninjective)).unwrap(), 81);
    }

    #[test]
    fn arbitrary_precision_widths_cannot_wrap_into_a_small_bound() {
        let ctx = context().unwrap();
        let fiber = Map::read_from_str(&ctx,
            "{ S[] -> V[i] : 0 <= i <= 18446744073709551617 }").unwrap();
        assert!(fiber_bound(&fiber, None).unwrap_err().contains("fitting i64"));
    }
}

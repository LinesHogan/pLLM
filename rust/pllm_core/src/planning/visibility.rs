//! Derive state readers' logical writers before model source partitioning.

use isl_rs::{Context, UnionMap};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_dependence::{isl, map_text};
use super::semantic_graph::canonical_ascii_json;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind { ForwardPinned, Exact }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Visibility { pub kind: Kind }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Scope { Overlay, ModelScan }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Tail { Transition, Include, Carry }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum KV { CausalHistory }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub policy: Visibility,
    pub scope: Scope,
    pub equations_digest: String,
    pub resources: Vec<String>,
    pub chunk_size: u64,
    pub tail: Tail,
    pub kv: KV,
    pub sweeps: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StateVisibility {
    pub source: Source,
    pub reads: String,
    pub dependencies: String,
}

fn source_domain(source: &Source) -> Result<String, String> {
    if source.sweeps != 1 {
        return Err("state visibility needs one sweep; no iteration preset is supported".into());
    }
    if source.equations_digest.len() != 64 || !source.equations_digest.bytes()
        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return Err("state visibility needs the source equations' SHA-256 digest".into());
    }
    if source.resources.is_empty() || source.resources.iter().any(|name| name.is_empty()) {
        return Err("state visibility needs a state resource for every target".into());
    }
    let extent = match (&source.scope, &source.policy.kind, &source.tail, source.chunk_size) {
        (Scope::Overlay, Kind::ForwardPinned, Tail::Transition, 0) => "C = 1 and N = 1",
        (Scope::ModelScan, Kind::Exact, Tail::Include, 1..) => "0 <= C and N = C",
        (Scope::Overlay, Kind::Exact, Tail::Carry, 1..) => "0 <= C and C <= N <= C+1",
        _ => return Err("state visibility must use its declaration's chunk and tail domain".into()),
    };
    Ok(format!("{extent} and R >= 1 and 0 <= r < R and 0 <= t < {}", source.resources.len()))
}

fn relation(source: &Source, ctx: &Context) -> Result<UnionMap, String> {
    let domain = source_domain(source)?;
    let mut parts = Vec::new();
    for (reader, count) in [("model", "N"), ("update", "C")] {
        let entry = if reader == "model" && source.policy.kind == Kind::ForwardPinned {
            format!("0 <= c < {count}")
        } else { "c = 0 and 0 < ".to_owned() + count };
        parts.push(format!("{reader}[r,t,c] -> entry[r,t] : {domain} and {entry}"));
        if source.policy.kind == Kind::Exact {
            parts.push(format!("{reader}[r,t,c] -> update[r,t,c-1] : {domain} and 0 < c < {count}"));
        }
    }
    if source.scope == Scope::ModelScan {
        parts.push(format!("decode[r,t] -> entry[r,t] : {domain} and C = 0"));
        parts.push(format!("decode[r,t] -> update[r,t,C-1] : {domain} and C > 0"));
    }
    isl(UnionMap::read_from_str(ctx, &format!("[C,N,R] -> {{ {} }}", parts.join("; "))))
}

fn derive(source: Source) -> Result<StateVisibility, String> {
    let ctx = Context::alloc();
    let reads = relation(&source, &ctx)?;
    let dependencies = isl(isl(reads.copy())?.reverse())?;
    Ok(StateVisibility { source, reads: map_text(&reads)?, dependencies: map_text(&dependencies)? })
}

pub(super) fn validate(value: &StateVisibility) -> Result<(), String> {
    let ctx = Context::alloc();
    let expected = relation(&value.source, &ctx)?;
    let reads = isl(UnionMap::read_from_str(&ctx, &value.reads))?;
    if !isl(reads.is_single_valued())?
        || !isl(isl(isl(reads.copy())?.domain())?.is_equal(&isl(isl(expected.copy())?.domain())?))? {
        return Err("state visibility must name exactly one writer for every model/update reader".into());
    }
    // Equality also excludes future/cyclic edges and reads from another request or target.
    if !isl(reads.is_equal(&expected))? {
        return Err("state visibility writers differ from the declared policy; restore the derived read map".into());
    }
    let dependencies = isl(UnionMap::read_from_str(&ctx, &value.dependencies))?;
    if !isl(dependencies.is_equal(&isl(isl(expected.copy())?.reverse())?))?
        || value.reads != map_text(&expected)? {
        return Err("state visibility needs the normalized read map and its derived writer dependencies".into());
    }
    Ok(())
}

#[pyfunction]
fn derive_state_visibility(source_json: &str) -> PyResult<String> {
    let run = || -> Result<String, String> {
        let source = serde_json::from_str(source_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(source)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_state_visibility(visibility_json: &str) -> PyResult<()> {
    let run = || -> Result<(), String> {
        let value = serde_json::from_str(visibility_json).map_err(|e| e.to_string())?;
        validate(&value)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(derive_state_visibility, module)?)?;
    module.add_function(wrap_pyfunction!(validate_state_visibility, module)?)?;
    Ok(())
}

//! Derive state readers' logical writers from a declaration's visibility source.

use isl_rs::{Context, UnionMap};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_dependence::{isl, map_text};

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
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub policy: Visibility,
    pub scope: Scope,
    pub equations_digest: String,
    pub resources: Vec<String>,
    pub chunk_size: u64,
    pub tail: Tail,
}

/// The read-from map: every model/update reader names exactly one logical writer.
fn relation(source: &Source, ctx: &Context) -> Result<UnionMap, String> {
    let extent = match (&source.scope, &source.policy.kind, &source.tail, source.chunk_size) {
        (Scope::Overlay, Kind::ForwardPinned, Tail::Transition, 0) => "C = 1 and N = 1",
        (Scope::ModelScan, Kind::Exact, Tail::Include, 1..) => "0 <= C and N = C",
        (Scope::Overlay, Kind::Exact, Tail::Carry, 0) => "0 <= C and C <= N <= C+1",
        _ => return Err("state visibility must use its declaration's chunk and tail domain".into()),
    };
    let domain = format!("{extent} and R >= 1 and 0 <= r < R and 0 <= t < {}", source.resources.len());
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

pub(super) fn validate(source: &Source) -> Result<(), String> {
    relation(source, &Context::alloc()).map(|_| ())
}

#[pyfunction]
fn explain_state_visibility(source_json: &str) -> PyResult<String> {
    let run = || -> Result<String, String> {
        let source: Source = serde_json::from_str(source_json).map_err(|e| e.to_string())?;
        map_text(&relation(&source, &Context::alloc())?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(explain_state_visibility, module)?)?;
    Ok(())
}

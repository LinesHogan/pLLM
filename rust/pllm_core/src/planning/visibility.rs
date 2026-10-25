//! Derive state readers' logical writers from a declaration's visibility source.

use isl_rs::{Context, UnionMap};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_dependence::{isl, map_text};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind { ForwardPinned, Exact, StepLag }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Visibility { pub kind: Kind }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Scope { Overlay, PostSampler }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Tail { Transition, Carry }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub policy: Visibility,
    pub scope: Scope,
    pub equations_digest: String,
    pub resources: Vec<String>,
    pub chunk_size: u64,
    pub tail: Tail,
    #[serde(default)]
    pub lag: u64,
}

/// The read-from map: every model/update reader names exactly one logical writer.
fn relation(source: &Source, ctx: &Context) -> Result<UnionMap, String> {
    if source.scope == Scope::PostSampler {
        if source.policy.kind != Kind::StepLag || source.chunk_size != 0 || source.tail != Tail::Transition || !source.resources.is_empty() {
            return Err("post-sampler visibility needs the step-input boundary domain".into());
        }
        let k = source.lag;
        // boundary[b,q] is E's logical boundary, never a numerical version ID.
        return isl(UnionMap::read_from_str(ctx, &format!("[side] -> {{
            reader[t,q] -> latest[t,q] : side = 0 and t >= 0 and q >= 0;
            reader[t,q] -> boundary[0,q] : side = 1 and 0 <= t < {k} and q >= 0;
            reader[t,q] -> boundary[t-{k},q] : side = 1 and t >= {k} and q >= 0;
            writer[t,q] -> latest[t,q] : 0 <= side <= 1 and t >= 0 and q >= 0 }}")));
    }
    if source.lag != 0 { return Err("step lag belongs to post-sampler visibility".into()); }
    let extent = match (&source.scope, &source.policy.kind, &source.tail, source.chunk_size) {
        (Scope::Overlay, Kind::ForwardPinned, Tail::Transition, 0) => "C = 1 and N = 1",
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
    isl(UnionMap::read_from_str(ctx, &format!("[C,N,R] -> {{ {} }}", parts.join("; "))))
}

pub(super) fn permits_overlap(source: &Source) -> Result<bool, String> {
    let ctx = Context::alloc();
    let allowed = isl(UnionMap::read_from_str(&ctx, "[side] -> {
        reader[t,q] -> latest[t,q] : side = 0 and t >= 0 and q >= 0;
        reader[t,q] -> boundary[b,q] : side = 1 and q >= 0 and ((t = 0 and b = 0) or (t > 0 and 0 <= b < t));
        writer[t,q] -> latest[t,q] : 0 <= side <= 1 and t >= 0 and q >= 0 }"))?;
    isl(relation(source, &ctx)?.is_subset(&allowed))
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

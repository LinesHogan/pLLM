//! Numeric state identities, resource projection and fixed-table preparation for tensor execution.

use std::collections::HashMap;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
};
use pyo3::prelude::*;

use crate::packed_rows::{
    column, IDENTITY, PACKED, PROMPT_INDEX_WORD, REQUEST_LENGTH_WORD, REQUEST_WORDS,
    SAMPLE_INDEX_WORD,
};

// Request-key fields a resource projects, as bits; RequestStateDirectory sends their sum.
const REQUEST_ID: i64 = 1;
const PROMPT_INDEX: i64 = 2;
const SAMPLE_INDEX: i64 = 4;
const ROUTE: i64 = 8;

type Array<'py> = Bound<'py, PyArray1<i64>>;
type Matrix<'py> = Bound<'py, PyArray2<i64>>;
type Identity = (i64, i64, i64);

fn positions(columns: [&[i64]; 3], selected: [&[i64]; 3]) -> Vec<i64> {
    let mut lookup = HashMap::<Identity, i64>::with_capacity(columns[0].len());
    // Preserve the first position, including when a pin contains duplicate refs.
    for i in 0..columns[0].len() {
        lookup
            .entry((columns[0][i], columns[1][i], columns[2][i]))
            .or_insert(i as i64);
    }
    (0..selected[0].len())
        .map(|i| {
            *lookup
                .get(&(selected[0][i], selected[1][i], selected[2][i]))
                .unwrap_or(&-1)
        })
        .collect()
}

/// Compare exact identity columns without constructing individual state refs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn refs_equal(
    py: Python<'_>,
    layouts: PyReadonlyArray1<'_, i64>,
    lineages: PyReadonlyArray1<'_, i64>,
    versions: PyReadonlyArray1<'_, i64>,
    other_layouts: PyReadonlyArray1<'_, i64>,
    other_lineages: PyReadonlyArray1<'_, i64>,
    other_versions: PyReadonlyArray1<'_, i64>,
) -> PyResult<bool> {
    let left = [column(&layouts)?, column(&lineages)?, column(&versions)?];
    let right = [
        column(&other_layouts)?,
        column(&other_lineages)?,
        column(&other_versions)?,
    ];
    Ok(py.allow_threads(|| left == right))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ref_positions<'py>(
    py: Python<'py>,
    layouts: PyReadonlyArray1<'py, i64>,
    lineages: PyReadonlyArray1<'py, i64>,
    versions: PyReadonlyArray1<'py, i64>,
    selected_layouts: PyReadonlyArray1<'py, i64>,
    selected_lineages: PyReadonlyArray1<'py, i64>,
    selected_versions: PyReadonlyArray1<'py, i64>,
) -> PyResult<Array<'py>> {
    let columns = [column(&layouts)?, column(&lineages)?, column(&versions)?];
    let selected = [
        column(&selected_layouts)?,
        column(&selected_lineages)?,
        column(&selected_versions)?,
    ];
    assert!(columns.iter().all(|c| c.len() == columns[0].len()));
    assert!(selected.iter().all(|c| c.len() == selected[0].len()));
    Ok(py
        .allow_threads(|| positions(columns, selected))
        .into_pyarray(py))
}

#[pyfunction]
fn replace_versions<'py>(
    py: Python<'py>,
    versions: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    after: PyReadonlyArray1<'py, i64>,
) -> PyResult<Array<'py>> {
    let (versions, positions, after) = (column(&versions)?, column(&positions)?, column(&after)?);
    assert_eq!(versions.len(), positions.len());
    Ok(py
        .allow_threads(|| {
            versions
                .iter()
                .zip(positions)
                .map(|(&v, &p)| if p < 0 { v } else { after[p as usize] })
                .collect::<Vec<_>>()
        })
        .into_pyarray(py))
}

/// Replace input versions and their numeric access owners together.
#[pyfunction]
fn read_versions<'py>(
    py: Python<'py>,
    versions: PyReadonlyArray1<'py, i64>,
    owners: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    after: PyReadonlyArray1<'py, i64>,
    owner: i64,
) -> PyResult<(Array<'py>, Array<'py>)> {
    let (versions, owners, positions, after) = (
        column(&versions)?,
        column(&owners)?,
        column(&positions)?,
        column(&after)?,
    );
    let (versions, owners): (Vec<_>, Vec<_>) = py.allow_threads(|| {
        versions
            .iter()
            .zip(owners)
            .zip(positions)
            .map(|((&v, &o), &p)| {
                if p < 0 {
                    (v, o)
                } else {
                    (after[p as usize], owner)
                }
            })
            .unzip()
    });
    Ok((versions.into_pyarray(py), owners.into_pyarray(py)))
}

#[derive(Clone, Copy)]
struct State {
    layout: i64,
    lineage: i64,
    row: i64,
}

struct Resource {
    fields: i64,
    bindings: HashMap<Vec<i64>, State>,
}

/// Admission-time resource keys; latest versions and page holds remain owned by the arena.
#[pyclass]
pub struct ResourceProjection {
    resources: Vec<Resource>,
}

type ProjectionOutput<'py> = Vec<(Matrix<'py>, Array<'py>)>;

#[pymethods]
impl ResourceProjection {
    #[new]
    fn new(fields: PyReadonlyArray1<'_, i64>) -> PyResult<Self> {
        Ok(Self {
            resources: column(&fields)?
                .iter()
                .map(|&fields| Resource {
                    fields,
                    bindings: HashMap::new(),
                })
                .collect(),
        })
    }

    fn bind(
        &mut self,
        words: PyReadonlyArray1<'_, i64>,
        resource: usize,
        layout: i64,
        lineage: i64,
        row: i64,
    ) -> PyResult<()> {
        let words = column(&words)?;
        let resource = &mut self.resources[resource];
        resource.bindings.insert(
            project_key(words, resource.fields),
            State {
                layout,
                lineage,
                row,
            },
        );
        Ok(())
    }

    fn retire(&mut self, resource: usize, lineage: i64) {
        self.resources[resource]
            .bindings
            .retain(|_, state| state.lineage != lineage);
    }

    fn project<'py>(
        &self,
        py: Python<'py>,
        words: PyReadonlyArray1<'py, i64>,
        offsets: PyReadonlyArray1<'py, i64>,
        requests: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<ProjectionOutput<'py>> {
        let (words, offsets, requests) = (column(&words)?, column(&offsets)?, column(&requests)?);
        let groups = py.allow_threads(|| self.project_rows(words, offsets, requests))?;
        Ok(groups
            .into_iter()
            .map(|(refs, indices)| {
                (
                    numpy::ndarray::Array2::from_shape_vec((3, refs.len() / 3), refs)
                        .unwrap()
                        .into_pyarray(py),
                    indices.into_pyarray(py),
                )
            })
            .collect())
    }
}

fn project_key(words: &[i64], fields: i64) -> Vec<i64> {
    if fields == 0 {
        return Vec::new();
    }
    let route = REQUEST_WORDS + words[REQUEST_LENGTH_WORD] as usize;
    let word = |field: i64, index: usize, absent: i64| {
        if fields & field != 0 {
            words[index]
        } else {
            absent
        }
    };
    let mut key = vec![
        word(PROMPT_INDEX, PROMPT_INDEX_WORD, -1),
        word(SAMPLE_INDEX, SAMPLE_INDEX_WORD, -1),
        word(REQUEST_ID, REQUEST_LENGTH_WORD, 0),
    ];
    if fields & REQUEST_ID != 0 {
        key.extend_from_slice(&words[REQUEST_WORDS..route]);
    }
    if fields & ROUTE != 0 {
        key.extend_from_slice(&words[route..]);
    }
    key
}

impl Resource {
    fn project(
        &self,
        words: &[i64],
        offsets: &[i64],
        requests: &[i64],
        offset: i64,
    ) -> PyResult<(Vec<i64>, Vec<i64>)> {
        let mut states = Vec::new();
        let mut positions = HashMap::new();
        if self.fields == 0 {
            let state = self.bindings[&Vec::new()];
            states.push(state);
            positions.insert(state.lineage, 0);
        }
        let mut indices = Vec::with_capacity(requests.len());
        for &request in requests {
            let state = if self.fields == 0 {
                states[0]
            } else {
                let r = request as usize;
                let key = project_key(
                    &words[offsets[r] as usize..offsets[r + 1] as usize],
                    self.fields,
                );
                *self.bindings.get(&key).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(
                        "admit each selected resource key before execution",
                    )
                })?
            };
            let n = positions.len();
            let position = *positions.entry(state.lineage).or_insert_with(|| {
                states.push(state);
                n
            });
            indices.push(offset + position as i64);
        }
        let refs = states
            .iter()
            .map(|s| s.layout)
            .chain(states.iter().map(|s| s.lineage))
            .chain(states.iter().map(|s| s.row))
            .collect();
        Ok((refs, indices))
    }
}

impl ResourceProjection {
    fn project_rows(
        &self,
        words: &[i64],
        offsets: &[i64],
        requests: &[i64],
    ) -> PyResult<Vec<(Vec<i64>, Vec<i64>)>> {
        let mut offset = 0;
        self.resources
            .iter()
            .map(|resource| {
                let (refs, indices) = resource.project(words, offsets, requests, offset)?;
                offset += (refs.len() / 3) as i64;
                Ok((refs, indices))
            })
            .collect()
    }
}

/// Resolve row columns and the last row of each state from borrowed numeric arrays.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn row_plan<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray2<'py, i64>,
    packed: PyReadonlyArray1<'py, i64>,
    links: PyReadonlyArray1<'py, i64>,
    identity_words: PyReadonlyArray1<'py, i64>,
    identity_offsets: PyReadonlyArray1<'py, i64>,
    selection_requests: PyReadonlyArray1<'py, i64>,
    state_indices: PyReadonlyArray1<'py, i64>,
    versions: PyReadonlyArray1<'py, i64>,
    instance: bool,
) -> PyResult<(Matrix<'py>, Array<'py>, Array<'py>)> {
    let rows = rows.as_array();
    let (packed, links, words, offsets, requests, states, versions) = (
        column(&packed)?,
        column(&links)?,
        column(&identity_words)?,
        column(&identity_offsets)?,
        column(&selection_requests)?,
        column(&state_indices)?,
        column(&versions)?,
    );
    let (columns, updated, last) = py.allow_threads(|| {
        let sampler: HashMap<_, _> = links
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i as i64))
            .collect();
        let selection: HashMap<_, _> = requests.iter().zip(states).map(|(&r, &s)| (r, s)).collect();
        let mut columns = vec![0; 5 * packed.len()];
        let mut last = std::collections::BTreeMap::new();
        for (position, &p) in packed.iter().enumerate() {
            let p = p as usize;
            let request = rows[[IDENTITY, p]];
            let state = if instance || versions.is_empty() {
                0
            } else {
                selection[&request]
            };
            for (c, v) in [
                *sampler.get(&rows[[PACKED, p]]).unwrap_or(&-1),
                words[offsets[request as usize] as usize + SAMPLE_INDEX_WORD],
                state,
                if versions.is_empty() {
                    state
                } else {
                    versions[state as usize]
                },
                rows[[PACKED, p]],
            ]
            .into_iter()
            .enumerate()
            {
                columns[c * packed.len() + position] = v;
            }
            last.insert(state, position as i64);
        }
        (
            columns,
            last.keys().copied().collect::<Vec<_>>(),
            last.values().copied().collect::<Vec<_>>(),
        )
    });
    Ok((
        numpy::ndarray::Array2::from_shape_vec((5, packed.len()), columns)
            .unwrap()
            .into_pyarray(py),
        updated.into_pyarray(py),
        last.into_pyarray(py),
    ))
}

/// Fill a borrowed pinned resource table, leaving H2D enqueue and stream ownership to Python.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fill_table(
    py: Python<'_>,
    mut host: PyReadwriteArray1<'_, i64>,
    columns: PyReadonlyArray2<'_, i64>,
    sources: PyReadonlyArray2<'_, i64>,
    updated: PyReadonlyArray1<'_, i64>,
    destinations: PyReadonlyArray2<'_, i64>,
    inputs: PyReadonlyArray2<'_, i64>,
    rows: usize,
    states: usize,
) -> PyResult<()> {
    let columns = columns.as_array();
    let sources = sources.as_array();
    let destinations = destinations.as_array();
    let inputs = inputs.as_array();
    let updated = column(&updated)?;
    let host = host.as_slice_mut()?;
    py.allow_threads(|| {
        let count = columns.shape()[1];
        let fields = sources.shape()[1];
        assert_eq!(
            host.len(),
            6 * rows + fields * (2 * states + 1) + inputs.len()
        );
        host.fill(0);
        host[2 * rows..3 * rows].fill(states as i64);
        for c in 0..5 {
            for r in 0..count {
                host[c * rows + r] = if c == 0 {
                    columns[[c, r]].max(0)
                } else {
                    columns[[c, r]]
                };
            }
        }
        host[5 * rows..5 * rows + count].fill(1);
        let start = 6 * rows;
        host[start..start + fields * (2 * states + 1)].fill(-1);
        for f in 0..fields {
            for s in 0..sources.shape()[0] {
                host[start + f * (states + 1) + s] = sources[[s, f]];
            }
            for (s, &position) in updated.iter().enumerate() {
                host[start + fields * (states + 1) + f * states + position as usize] =
                    destinations[[s, f]];
            }
        }
        let start = start + fields * (2 * states + 1);
        for (target, &value) in host[start..].iter_mut().zip(inputs.iter()) {
            *target = value;
        }
    });
    Ok(())
}

/// Select unique state positions for producer rows, preserving producer order.
#[pyfunction]
fn select_states<'py>(
    py: Python<'py>,
    requests: PyReadonlyArray1<'py, i64>,
    states: PyReadonlyArray1<'py, i64>,
    selected: PyReadonlyArray1<'py, i64>,
) -> PyResult<Array<'py>> {
    let (requests, states, selected) = (column(&requests)?, column(&states)?, column(&selected)?);
    let result = py.allow_threads(|| {
        let lookup: HashMap<_, _> = requests.iter().zip(states).map(|(&r, &s)| (r, s)).collect();
        let mut seen = std::collections::HashSet::new();
        selected
            .iter()
            .map(|request| lookup[request])
            .filter(|state| seen.insert(*state))
            .collect::<Vec<_>>()
    });
    Ok(result.into_pyarray(py))
}

/// Apply the declared lag history by lineage, with later history entries taking precedence.
#[pyfunction]
fn rewind_versions<'py>(
    py: Python<'py>,
    lineages: PyReadonlyArray1<'py, i64>,
    versions: PyReadonlyArray1<'py, i64>,
    history_lineages: Vec<PyReadonlyArray1<'py, i64>>,
    history_versions: Vec<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Array<'py>> {
    let (lineages, versions) = (column(&lineages)?, column(&versions)?);
    let history = history_lineages
        .iter()
        .zip(&history_versions)
        .map(|(l, v)| Ok((column(l)?, column(v)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let result = py.allow_threads(|| {
        let mut lookup = HashMap::new();
        for (lineages, versions) in history.into_iter().rev() {
            let mut seen = std::collections::HashSet::new();
            for (&lineage, &version) in lineages.iter().zip(versions) {
                if seen.insert(lineage) {
                    lookup.insert(lineage, version);
                }
            }
        }
        lineages
            .iter()
            .zip(versions)
            .map(|(lineage, version)| *lookup.get(lineage).unwrap_or(version))
            .collect::<Vec<_>>()
    });
    Ok(result.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ResourceProjection>()?;
    m.add_function(wrap_pyfunction!(select_states, m)?)?;
    m.add_function(wrap_pyfunction!(rewind_versions, m)?)?;
    m.add_function(wrap_pyfunction!(ref_positions, m)?)?;
    m.add_function(wrap_pyfunction!(refs_equal, m)?)?;
    m.add_function(wrap_pyfunction!(read_versions, m)?)?;
    m.add_function(wrap_pyfunction!(replace_versions, m)?)?;
    m.add_function(wrap_pyfunction!(row_plan, m)?)?;
    m.add_function(wrap_pyfunction!(fill_table, m)?)?;
    Ok(())
}

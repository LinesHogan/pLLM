//! Bulk packed-token labels and hidden-tap rank maps for the Python runners.
//!
//! Tap selections are latest decode rows, completed prompt endings, both, and every fresh token.

use numpy::{
    ndarray::Dimension, Element, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray,
    PyReadonlyArray1,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Request generations and retained-state frontiers, owned by one runner session.
#[pyclass]
#[derive(Default)]
pub struct RowLabeler {
    frontiers: Vec<Option<i64>>,
    free: Vec<usize>,
    identities: Vec<Vec<i64>>,
}

struct Labels {
    rows: Vec<i64>,
    links: Vec<i64>,
    selected: Vec<Vec<i64>>,
    ranks: Vec<Vec<i64>>,
}

type Array<'py> = Bound<'py, PyArray1<i64>>;
type Output<'py> = (
    Bound<'py, PyArray2<i64>>,
    Array<'py>,
    Vec<Array<'py>>,
    Vec<Array<'py>>,
    Array<'py>,
    Array<'py>,
);

#[pymethods]
impl RowLabeler {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn admit(&mut self, words: PyReadonlyArray1<'_, i64>) -> PyResult<usize> {
        let words = column(&words)?.to_vec();
        if let Some(id) = self.free.pop() {
            self.frontiers[id] = Some(0);
            self.identities[id] = words;
            return Ok(id);
        }
        let id = self.frontiers.len();
        self.frontiers.push(Some(0));
        self.identities.push(words);
        Ok(id)
    }

    fn retire(&mut self, id: usize) {
        self.frontiers[id] = None;
        self.free.push(id);
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare<'py>(
        &mut self,
        py: Python<'py>,
        scheduled: PyReadonlyArray1<'py, i32>,
        prompt: PyReadonlyArray1<'py, i32>,
        computed: PyReadonlyArray1<'py, i32>,
        ids: PyReadonlyArray1<'py, i64>,
        keep: bool,
    ) -> PyResult<Output<'py>> {
        let (scheduled, prompt, computed, ids) = (
            column(&scheduled)?,
            column(&prompt)?,
            column(&computed)?,
            column(&ids)?,
        );
        self.validate(scheduled, prompt, computed, ids)?;
        // numpy's readonly borrows stay alive until preparation finishes; no tensor or CUDA access.
        let labels = py.allow_threads(|| self.label(scheduled, prompt, computed, ids, keep));
        let rows = numpy::ndarray::Array2::from_shape_vec((7, labels.rows.len() / 7), labels.rows)
            .expect("label columns have equal lengths")
            .into_pyarray(py);
        let mut words = Vec::new();
        let mut offsets = vec![0];
        for &id in ids {
            words.extend_from_slice(&self.identities[id as usize]);
            offsets.push(words.len() as i64);
        }
        Ok((
            rows,
            labels.links.into_pyarray(py),
            labels
                .selected
                .into_iter()
                .map(|v| v.into_pyarray(py))
                .collect(),
            labels
                .ranks
                .into_iter()
                .map(|v| v.into_pyarray(py))
                .collect(),
            words.into_pyarray(py),
            offsets.into_pyarray(py),
        ))
    }
}

impl RowLabeler {
    fn validate(
        &self,
        scheduled: &[i32],
        prompt: &[i32],
        computed: &[i32],
        ids: &[i64],
    ) -> PyResult<()> {
        if [prompt.len(), computed.len(), ids.len()]
            .iter()
            .any(|&n| n != scheduled.len())
        {
            return Err(PyValueError::new_err(
                "provide equally sized scheduled, prompt, computed and request ID columns",
            ));
        }
        for (((&count, &length), &done), &id) in scheduled.iter().zip(prompt).zip(computed).zip(ids)
        {
            if count < 0 || length < 0 || done < 0 || done.checked_add(count).is_none() {
                return Err(PyValueError::new_err(
                    "use nonnegative scheduled counts and int32 token positions",
                ));
            }
            if id < 0 || self.frontiers.get(id as usize).is_none_or(Option::is_none) {
                return Err(PyValueError::new_err(
                    "admit each request before labeling its rows",
                ));
            }
        }
        Ok(())
    }

    fn label(
        &mut self,
        scheduled: &[i32],
        prompt: &[i32],
        computed: &[i32],
        ids: &[i64],
        keep: bool,
    ) -> Labels {
        let total: usize = scheduled.iter().map(|&n| n as usize).sum();
        let mut rows = vec![0; 7 * total];
        let mut links = Vec::with_capacity(ids.len());
        let (mut prefill, mut decode, mut fresh) = (Vec::new(), Vec::new(), Vec::new());
        let mut start = 0;
        for (request, (((&count, &length), &done), &id)) in scheduled
            .iter()
            .zip(prompt)
            .zip(computed)
            .zip(ids)
            .enumerate()
        {
            let frontier = self.frontiers[id as usize].unwrap();
            for token in 0..count {
                let packed = start + token as usize;
                let offset = i64::from(done) + i64::from(token);
                let replay = keep && offset < frontier;
                for (column, value) in [
                    packed as i64,
                    request as i64,
                    offset,
                    i64::from(offset < i64::from(length)),
                    request as i64,
                    i64::from(length),
                    i64::from(replay),
                ]
                .into_iter()
                .enumerate()
                {
                    rows[column * total + packed] = value;
                }
                if !replay {
                    fresh.push(packed as i64);
                }
                if !replay && offset == i64::from(length) - 1 {
                    prefill.push(packed as i64);
                }
                if !replay && offset >= i64::from(length) && token == count - 1 {
                    decode.push(packed as i64);
                }
            }
            start += count as usize;
            links.push(start as i64 - 1);
            if keep && count > 0 {
                self.frontiers[id as usize] =
                    Some(frontier.max(i64::from(done) + i64::from(count)));
            }
        }
        // HiddenPhase sorts decode before prefill; preserve that order for taps reading both.
        let both: Vec<i64> = decode.iter().chain(&prefill).copied().collect();
        let selected = vec![decode, prefill, both, fresh];
        let ranks = selected
            .iter()
            .map(|selection| {
                let mut mapping = vec![0; total];
                for (rank, &packed) in selection.iter().enumerate() {
                    mapping[packed as usize] = rank as i64;
                }
                mapping
            })
            .collect();
        Labels {
            rows,
            links,
            selected,
            ranks,
        }
    }
}

pub(crate) fn column<'a, T: Element, D: Dimension>(
    array: &'a PyReadonlyArray<'_, T, D>,
) -> PyResult<&'a [T]> {
    // rust-numpy 0.23 checks contiguity but does not check pointer alignment before making a slice.
    if array.data().is_null() || !array.data().is_aligned() {
        return Err(PyValueError::new_err(
            "use aligned NumPy columns; numpy.require(values, requirements=['C', 'A']) makes an aligned copy",
        ));
    }
    Ok(array.as_slice()?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RowLabeler>()
}

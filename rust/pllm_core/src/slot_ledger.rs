//! Numeric page holds, allocation generations and event fences for PTTS and hidden rows.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::packed_rows::column;

pyo3::create_exception!(_core, PageCapacity, pyo3::exceptions::PyRuntimeError);

pub(crate) type Array<'py> = Bound<'py, PyArray1<i64>>;

/// Each field has its own page count: older versions keep only the fields their readers pin.
#[derive(Default)]
pub(crate) struct Pool {
    pub width: usize,
    pub capacities: Vec<usize>,
    offsets: Vec<usize>,
    hands: Vec<usize>,
    holders: Vec<i64>,
    generations: Vec<i64>,
    fences: Vec<(i64, Vec<bool>)>,
}

impl Pool {
    fn new(capacities: Vec<usize>) -> Self {
        let offsets: Vec<usize> = capacities
            .iter()
            .scan(0, |start, &capacity| {
                let offset = *start;
                *start += capacity;
                Some(offset)
            })
            .collect();
        let pages = capacities.iter().sum();
        Self {
            width: capacities.len(),
            hands: vec![0; capacities.len()],
            capacities,
            offsets,
            holders: vec![0; pages],
            generations: vec![0; pages],
            fences: Vec::new(),
        }
    }

    fn index(&self, field: usize, slot: usize) -> usize {
        assert!(slot < self.capacities[field]);
        self.offsets[field] + slot
    }

    pub fn free_count(&self, field: usize) -> usize {
        self.holders[self.offsets[field]..self.offsets[field] + self.capacities[field]]
            .iter()
            .filter(|&&count| count == 0)
            .count()
    }

    pub fn collect(&mut self, completed: &[i64]) {
        self.fences.retain(|(event, _)| !completed.contains(event));
    }

    pub fn events(&self) -> Vec<i64> {
        self.fences.iter().map(|(event, _)| *event).collect()
    }

    pub fn allocate(
        &mut self,
        field: usize,
        count: usize,
    ) -> PyResult<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let capacity = self.capacities[field];
        if count == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let hand = self.hands[field];
        let offset = self.offsets[field];
        let mut ready = Vec::with_capacity(count);
        let mut reuse = Vec::new();
        for slot in (hand..capacity).chain(0..hand) {
            if ready.len() == count {
                break;
            }
            if self.holders[offset + slot] != 0 {
                continue;
            }
            if self
                .fences
                .iter()
                .any(|(_, pending)| pending[offset + slot])
            {
                reuse.push(slot);
            } else {
                ready.push(slot);
            }
        }
        if ready.len() + reuse.len() < count {
            return Err(PageCapacity::new_err(
                "Slot capacity is exhausted; release current holders before retrying",
            ));
        }
        let slots: Vec<_> = ready.into_iter().chain(reuse).take(count).collect();
        self.hands[field] = (slots[count - 1] + 1) % capacity;
        let mut waits = Vec::new();
        for (event, pending) in &mut self.fences {
            if slots.iter().any(|&slot| pending[offset + slot]) {
                waits.push(*event);
                for &slot in &slots {
                    pending[offset + slot] = false;
                }
            }
        }
        self.fences
            .retain(|(_, pending)| pending.iter().any(|&v| v));
        let generations = slots
            .iter()
            .map(|&slot| self.generations[offset + slot])
            .collect();
        for &slot in &slots {
            self.holders[offset + slot] = 1;
        }
        Ok((
            slots.into_iter().map(|v| v as i64).collect(),
            generations,
            waits,
        ))
    }

    pub fn share(&mut self, field: usize, slots: &[i64]) {
        for &slot in slots {
            let index = self.index(field, slot as usize);
            self.holders[index] += 1;
        }
    }

    /// Hold the `held` fields of a [rows, fields] slot matrix.
    pub fn hold(&mut self, slots: &[i64], held: &[bool]) {
        for (position, &slot) in slots.iter().enumerate() {
            if held[position % self.width] {
                let index = self.index(position % self.width, slot as usize);
                self.holders[index] += 1;
            }
        }
    }

    pub fn release(&mut self, slots: &[i64], held: &[bool], completion: i64) {
        let mut counts = vec![0; self.holders.len()];
        let mut touched = Vec::new();
        for (position, &slot) in slots.iter().enumerate() {
            if !held[position % self.width] {
                continue;
            }
            let index = self.index(position % self.width, slot as usize);
            if counts[index] == 0 {
                touched.push(index);
            }
            counts[index] += 1;
        }
        if completion != 0 && !touched.is_empty() {
            let fence = self
                .fences
                .iter()
                .position(|(event, _)| *event == completion)
                .unwrap_or_else(|| {
                    self.fences
                        .push((completion, vec![false; self.holders.len()]));
                    self.fences.len() - 1
                });
            for &index in &touched {
                self.fences[fence].1[index] = true;
            }
        }
        for index in touched {
            self.holders[index] -= counts[index];
            assert!(self.holders[index] >= 0);
            if self.holders[index] == 0 {
                self.generations[index] += 1;
            }
        }
    }

    fn check_page(&self, field: usize, slot: usize, generation: i64) {
        assert!(self.live_page(field, slot, generation));
    }

    /// Whether a page still holds the allocation `generation` names: held, and not reclaimed since.
    fn live_page(&self, field: usize, slot: usize, generation: i64) -> bool {
        let index = self.index(field, slot);
        self.holders[index] >= 1 && self.generations[index] == generation
    }
}

/// Single-owner numeric storage shared with an arena or used alone by hidden-row capture.
#[pyclass(unsendable)]
pub struct SlotLedger {
    pub(crate) pool: Pool,
}

type Allocation<'py> = (Array<'py>, Array<'py>, Array<'py>, Array<'py>);

#[pymethods]
impl SlotLedger {
    #[new]
    fn new(capacities: Vec<usize>) -> Self {
        Self {
            pool: Pool::new(capacities),
        }
    }

    fn free_count(&self, field: usize) -> usize {
        self.pool.free_count(field)
    }

    fn allocate<'py>(
        &mut self,
        py: Python<'py>,
        field: usize,
        count: usize,
        completed: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Allocation<'py>> {
        let completed = column(&completed)?;
        let pool = &mut self.pool;
        let (slots, generations, waits) = py.allow_threads(|| {
            pool.collect(completed);
            pool.allocate(field, count)
        })?;
        Ok((
            slots.into_pyarray(py),
            generations.into_pyarray(py),
            waits.into_pyarray(py),
            pool.events().into_pyarray(py),
        ))
    }

    fn hold(&mut self, slots: PyReadonlyArray2<'_, i64>) -> PyResult<()> {
        let slots = column(&slots)?;
        let pool = &mut self.pool;
        let held = vec![true; pool.width];
        pool.hold(slots, &held);
        Ok(())
    }

    fn release<'py>(
        &mut self,
        py: Python<'py>,
        slots: PyReadonlyArray2<'py, i64>,
        completion: i64,
        completed: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Array<'py>> {
        let (slots, completed) = (column(&slots)?, column(&completed)?);
        let pool = &mut self.pool;
        pool.collect(completed);
        let held = vec![true; pool.width];
        pool.release(slots, &held, completion);
        Ok(pool.events().into_pyarray(py))
    }

    fn check_page(&self, field: usize, slot: usize, generation: i64) {
        self.pool.check_page(field, slot, generation);
    }

    /// Per field of one version, whether its page is still live.
    fn live(
        &self,
        slots: PyReadonlyArray1<'_, i64>,
        generations: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Vec<bool>> {
        let (slots, generations) = (column(&slots)?, column(&generations)?);
        assert_eq!(slots.len(), self.pool.width);
        Ok((0..self.pool.width)
            .map(|field| self.pool.live_page(field, slots[field] as usize, generations[field]))
            .collect())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SlotLedger>()?;
    m.add("PageCapacity", m.py().get_type::<PageCapacity>())?;
    Ok(())
}

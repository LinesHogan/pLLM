//! Single-owner PTTS admission, transition, exact-pin and publication bookkeeping.
//!
//! Python retains tensors and completion events. This ledger owns every numeric fact and
//! returns only array snapshots, copy descriptors and event IDs for the owner to enqueue.

use std::collections::{BTreeMap, HashMap};

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;

use crate::packed_rows::column;
use crate::slot_ledger::{Array, PageCapacity, Pool, SlotLedger};

type Matrix<'py> = Bound<'py, PyArray2<i64>>;
type Mask<'py> = Bound<'py, PyArray2<bool>>;
type RefColumns<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    Option<PyReadonlyArray1<'py, i64>>,
);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StateRef {
    layout: i64,
    lineage: i64,
    version: i64,
}

struct Refs<'a> {
    layouts: &'a [i64],
    lineages: &'a [i64],
    versions: &'a [i64],
    rows: Option<&'a [i64]>,
}

impl<'a> Refs<'a> {
    fn read(columns: &'a RefColumns<'_>) -> PyResult<Self> {
        let result = Self {
            layouts: column(&columns.0)?,
            lineages: column(&columns.1)?,
            versions: column(&columns.2)?,
            rows: columns.3.as_ref().map(column).transpose()?,
        };
        assert_eq!(result.layouts.len(), result.lineages.len());
        assert_eq!(result.layouts.len(), result.versions.len());
        Ok(result)
    }

    fn get(&self, index: usize) -> StateRef {
        StateRef {
            layout: self.layouts[index],
            lineage: self.lineages[index],
            version: self.versions[index],
        }
    }
}

#[derive(Default)]
struct Row {
    lineage: i64,
    version: i64,
    slots: Vec<i64>,
    generations: Vec<i64>,
    readiness: i64,
    status: u8,
    owners: usize,
    writing: bool,
}

struct Table {
    layout: i64,
    max_lineages: usize,
    mutable: bool,
    // The fields a pin holds: older versions keep only what their readers read.
    pinned: Vec<bool>,
    rows: Vec<Row>,
    row_of: HashMap<i64, usize>,
}

impl Table {
    fn ref_at(&self, row: usize) -> StateRef {
        StateRef {
            layout: self.layout,
            lineage: self.rows[row].lineage,
            version: self.rows[row].version,
        }
    }

    fn reclaim(&mut self, rows: &[usize]) {
        for &index in rows {
            let row = &mut self.rows[index];
            if row.owners == 0 && row.status != 0 {
                self.row_of.remove(&row.lineage);
                row.status = 0;
                row.readiness = 0;
            }
        }
    }
}

#[derive(Clone)]
struct Pages {
    resource: usize,
    refs: Vec<StateRef>,
    rows: Vec<usize>,
    slots: Vec<i64>,
    generations: Vec<i64>,
    readiness: Vec<i64>,
    written: Option<Vec<bool>>,
    // The fields these pages hold; a pin holds only its table's pinned fields.
    held: Vec<bool>,
}

impl Pages {
    fn hold(&self, table: &mut Table, pool: &mut Pool) {
        pool.hold(&self.slots, &self.held);
        for &row in &self.rows {
            table.rows[row].owners += 1;
        }
    }

    fn release(&self, table: &mut Table, pool: &mut Pool, completion: i64) {
        pool.release(&self.slots, &self.held, completion);
        for &row in &self.rows {
            table.rows[row].owners -= 1;
        }
        table.reclaim(&self.rows);
    }

    fn columns(&self) -> Vec<i64> {
        self.refs
            .iter()
            .map(|r| r.layout)
            .chain(self.refs.iter().map(|r| r.lineage))
            .chain(self.refs.iter().map(|r| r.version))
            .chain(self.rows.iter().map(|&r| r as i64))
            .collect()
    }
}

struct Write {
    targets: Vec<Pages>,
    sources: Vec<Pages>,
}

struct Pin {
    pages: Vec<Pages>,
    stream: i64,
}

struct Release {
    pages: Vec<Pages>,
    completion: i64,
}

struct Copy {
    resource: usize,
    field: usize,
    source: Vec<i64>,
    target: Vec<i64>,
}

struct Prepared {
    id: i64,
    refs: Vec<i64>,
    copies: Vec<Copy>,
    waits: Vec<i64>,
}

struct Arena {
    tables: Vec<Table>,
    writes: BTreeMap<i64, Write>,
    pins: BTreeMap<i64, Pin>,
    releases: BTreeMap<i64, Release>,
    next_lineage: i64,
    next_operation: i64,
    next_pin: i64,
    next_release: i64,
    admission_failures: usize,
}

impl Arena {
    fn held_pages(&self) -> impl Iterator<Item = &Pages> {
        self.pins
            .values()
            .flat_map(|p| &p.pages)
            .chain(self.releases.values().flat_map(|r| &r.pages))
            .chain(self.writes.values().flat_map(|w| &w.sources))
    }

    fn select(&self, refs: &Refs<'_>, latest: bool) -> Vec<Pages> {
        self.tables
            .iter()
            .enumerate()
            .filter_map(|(resource, table)| {
                let indices: Vec<_> = (0..refs.layouts.len())
                    .filter(|&i| {
                        refs.layouts[i] == table.layout
                            && table.row_of.contains_key(&refs.lineages[i])
                    })
                    .collect();
                if indices.is_empty() {
                    return None;
                }
                let mut group = Pages {
                    resource,
                    refs: Vec::new(),
                    rows: Vec::new(),
                    slots: Vec::new(),
                    generations: Vec::new(),
                    readiness: Vec::new(),
                    written: None,
                    held: vec![true; table.pinned.len()],
                };
                for index in indices {
                    self.select_ref(&mut group, table, refs, index, latest);
                }
                Some(group)
            })
            .collect()
    }

    fn select_ref(
        &self,
        group: &mut Pages,
        table: &Table,
        refs: &Refs<'_>,
        index: usize,
        latest: bool,
    ) {
        let state = refs.get(index);
        let row = refs
            .rows
            .map_or_else(|| table.row_of[&state.lineage], |rows| rows[index] as usize);
        let entry = &table.rows[row];
        assert!(entry.status == 2 && entry.lineage == state.lineage);
        group.refs.push(state);
        group.rows.push(row);
        if entry.version == state.version {
            group.slots.extend_from_slice(&entry.slots);
            group.generations.extend_from_slice(&entry.generations);
            group.readiness.push(entry.readiness);
        } else {
            assert!(!latest);
            let (held, position) = self
                .held_pages()
                .filter(|p| p.resource == group.resource)
                .find_map(|p| p.refs.iter().position(|r| *r == state).map(|i| (p, i)))
                .expect("retain each lagged version until its reader is pinned");
            let width = entry.slots.len();
            group
                .slots
                .extend_from_slice(&held.slots[position * width..(position + 1) * width]);
            group
                .generations
                .extend_from_slice(&held.generations[position * width..(position + 1) * width]);
            group
                .readiness
                .push(held.readiness.get(position).copied().unwrap_or(0));
        }
    }

    fn events(&self, pools: &[&mut Pool]) -> Vec<i64> {
        let events = self
            .tables
            .iter()
            .flat_map(|t| t.row_of.values().map(|&i| t.rows[i].readiness))
            .chain(self.held_pages().flat_map(|p| p.readiness.iter().copied()))
            .chain(
                self.writes
                    .values()
                    .flat_map(|w| w.targets.iter().flat_map(|p| p.readiness.iter().copied())),
            )
            .chain(self.releases.values().map(|r| r.completion))
            .chain(pools.iter().flat_map(|p| p.events()));
        unique_events(events)
    }

    fn prepare_admission(
        &mut self,
        pools: &mut [&mut Pool],
        resources: &[i64],
    ) -> PyResult<Prepared> {
        self.check_admission(pools, resources)?;
        let refs: Vec<_> = resources
            .iter()
            .enumerate()
            .map(|(i, &resource)| StateRef {
                layout: self.tables[resource as usize].layout,
                lineage: self.next_lineage + i as i64,
                version: 0,
            })
            .collect();
        self.next_lineage += refs.len() as i64;
        let mut groups = Vec::new();
        let mut waits = Vec::new();
        for resource in resources.iter().copied().fold(Vec::new(), |mut v, r| {
            if !v.contains(&r) {
                v.push(r);
            }
            v
        }) {
            let r = resource as usize;
            let selected: Vec<_> = refs
                .iter()
                .zip(resources)
                .filter(|(_, &v)| v == resource)
                .map(|(&s, _)| s)
                .collect();
            groups.push(admit_pages(
                r,
                &mut self.tables[r],
                pools[r],
                selected,
                &mut waits,
            )?);
        }
        let columns = ref_columns(&refs, &groups);
        Ok(self.prepare(
            Write {
                targets: groups,
                sources: Vec::new(),
            },
            columns,
            None,
            waits,
        ))
    }

    fn check_admission(&mut self, pools: &[&mut Pool], resources: &[i64]) -> PyResult<()> {
        for (resource, table) in self.tables.iter().enumerate() {
            let count = resources
                .iter()
                .filter(|&&r| r as usize == resource)
                .count();
            let admitted = table
                .rows
                .iter()
                .filter(|r| r.status == 1 || r.status == 2)
                .count();
            if admitted + count > table.max_lineages {
                self.admission_failures += 1;
                return Err(PyRuntimeError::new_err(
                    "PageBundle admission capacity is exhausted",
                ));
            }
        }
        for (resource, table) in self.tables.iter().enumerate() {
            let count = resources
                .iter()
                .filter(|&&r| r as usize == resource)
                .count();
            if count == 0 {
                continue;
            }
            let headroom = if table.mutable {
                count
                    + table
                        .rows
                        .iter()
                        .filter(|r| r.status == 1 || (r.status == 2 && !r.writing))
                        .count()
            } else {
                0
            };
            if (0..pools[resource].width).any(|f| pools[resource].free_count(f) < count + headroom)
            {
                self.admission_failures += 1;
                return Err(PageCapacity::new_err(
                    "page capacity is held by live owners; retry admission asynchronously",
                ));
            }
        }
        Ok(())
    }

    fn prepare_transition(
        &mut self,
        pools: &mut [&mut Pool],
        refs: &Refs<'_>,
        fields: Option<(&[i64], &[i64])>,
        copy_fields: Option<&[i64]>,
    ) -> PyResult<Prepared> {
        let sources = self.select(refs, true);
        let masks = self.write_masks(&sources, refs, fields);
        for (group, mask) in sources.iter().zip(&masks) {
            let pool = &pools[group.resource];
            assert!(group
                .rows
                .iter()
                .all(|&row| !self.tables[group.resource].rows[row].writing));
            for field in 0..pool.width {
                let count = mask
                    .iter()
                    .skip(field)
                    .step_by(pool.width)
                    .filter(|&&v| v)
                    .count();
                if pool.free_count(field) < count {
                    return Err(PageCapacity::new_err(
                        "page capacity is unavailable without waiting",
                    ));
                }
            }
        }
        let mut targets = Vec::new();
        let mut waits = Vec::new();
        for (group, mask) in sources.iter().zip(masks) {
            let resource = group.resource;
            targets.push(transition_pages(
                group,
                mask,
                &mut self.tables[resource],
                pools[resource],
                &mut waits,
            )?);
        }
        let after: Vec<_> = (0..refs.layouts.len())
            .map(|i| StateRef {
                version: refs.versions[i] + 1,
                ..refs.get(i)
            })
            .collect();
        let columns = ref_columns(&after, &targets);
        Ok(self.prepare(Write { targets, sources }, columns, copy_fields, waits))
    }

    fn write_masks(
        &self,
        groups: &[Pages],
        refs: &Refs<'_>,
        fields: Option<(&[i64], &[i64])>,
    ) -> Vec<Vec<bool>> {
        groups
            .iter()
            .map(|group| {
                let width = group.slots.len() / group.refs.len();
                let mut mask = vec![fields.is_none(); group.slots.len()];
                if let Some((offsets, fields)) = fields {
                    for (row, state) in group.refs.iter().enumerate() {
                        let position = (0..refs.layouts.len())
                            .find(|&i| refs.get(i) == *state)
                            .unwrap();
                        for &field in
                            &fields[offsets[position] as usize..offsets[position + 1] as usize]
                        {
                            mask[row * width + field as usize] = true;
                        }
                    }
                }
                mask
            })
            .collect()
    }

    fn prepare(
        &mut self,
        write: Write,
        refs: Vec<i64>,
        copy_fields: Option<&[i64]>,
        mut waits: Vec<i64>,
    ) -> Prepared {
        waits.extend(write.sources.iter().flat_map(|p| &p.readiness));
        let copies = copy_descriptors(&write, copy_fields);
        let id = self.next_operation;
        self.next_operation += 1;
        self.writes.insert(id, write);
        Prepared {
            id,
            refs,
            copies,
            waits: unique_events(waits),
        }
    }

    fn publish(&mut self, pools: &mut [&mut Pool], id: i64, completion: i64) {
        let write = self
            .writes
            .remove(&id)
            .expect("publish each issued write once");
        for target in &write.targets {
            let table = &mut self.tables[target.resource];
            target.hold(table, pools[target.resource]);
            let width = pools[target.resource].width;
            for (i, &row) in target.rows.iter().enumerate() {
                let entry = &mut table.rows[row];
                entry.slots = target.slots[i * width..(i + 1) * width].to_vec();
                entry.generations = target.generations[i * width..(i + 1) * width].to_vec();
                entry.version = target.refs[i].version;
                entry.readiness = completion;
                entry.status = 2;
                entry.writing = false;
            }
        }
        for group in write
            .targets
            .iter()
            .chain(&write.sources)
            .chain(&write.sources)
        {
            group.release(
                &mut self.tables[group.resource],
                pools[group.resource],
                completion,
            );
        }
    }

    fn abort(&mut self, pools: &mut [&mut Pool], id: i64, completion: i64) {
        let write = self
            .writes
            .remove(&id)
            .expect("abort each issued write once");
        for group in write.targets.iter().chain(&write.sources) {
            for &row in &group.rows {
                self.tables[group.resource].rows[row].writing = false;
            }
            group.release(
                &mut self.tables[group.resource],
                pools[group.resource],
                completion,
            );
        }
    }

    fn pin(&mut self, pools: &mut [&mut Pool], refs: &Refs<'_>, stream: i64) -> (i64, Vec<i64>) {
        let mut groups = self.select(refs, false);
        for group in &mut groups {
            group.held.clone_from(&self.tables[group.resource].pinned);
        }
        let events = unique_events(groups.iter().flat_map(|p| p.readiness.iter().copied()));
        for group in &groups {
            group.hold(&mut self.tables[group.resource], pools[group.resource]);
        }
        let id = self.next_pin;
        self.next_pin += 1;
        self.pins.insert(
            id,
            Pin {
                pages: groups,
                stream,
            },
        );
        (id, events)
    }

    fn release_pin(
        &mut self,
        pools: &mut [&mut Pool],
        pin: i64,
        completion: i64,
        defer: bool,
        stream: i64,
    ) -> i64 {
        assert!(self.valid_access(pin, stream));
        let pages = self
            .pins
            .remove(&pin)
            .expect("release each issued pin once")
            .pages;
        let id = self.next_release;
        self.next_release += 1;
        if defer {
            self.releases.insert(id, Release { pages, completion });
        } else {
            self.release_pages(pools, &pages, completion);
        }
        id
    }

    fn release_pages(&mut self, pools: &mut [&mut Pool], pages: &[Pages], completion: i64) {
        for group in pages {
            group.release(
                &mut self.tables[group.resource],
                pools[group.resource],
                completion,
            );
        }
    }

    fn valid_access(&self, id: i64, stream: i64) -> bool {
        if id < 0 {
            self.writes.contains_key(&-id)
        } else {
            self.pins.get(&id).is_some_and(|p| p.stream == stream)
        }
    }

    fn access(&self, id: i64, stream: i64) -> &[Pages] {
        assert!(self.valid_access(id, stream));
        if id < 0 {
            &self.writes[&-id].targets
        } else {
            &self.pins[&id].pages
        }
    }

    fn slots(
        &self,
        access: &[i64],
        streams: &[i64],
        owners: &[i64],
        refs: &Refs<'_>,
    ) -> PyResult<(usize, Vec<i64>)> {
        if access.len() == 1 {
            for group in self.access(access[0], streams[0]) {
                if group.refs.len() == refs.layouts.len()
                    && group
                        .refs
                        .iter()
                        .enumerate()
                        .all(|(i, r)| *r == refs.get(i))
                {
                    return Ok((group.resource, group.slots.clone()));
                }
            }
        }
        let mut resource = None;
        let mut slots = Vec::new();
        for i in 0..refs.layouts.len() {
            let owner = if access.len() == 1 {
                0
            } else {
                owners[i] as usize
            };
            let state = refs.get(i);
            let (group, row) = self
                .access(access[owner], streams[owner])
                .iter()
                .find_map(|p| p.refs.iter().position(|r| *r == state).map(|i| (p, i)))
                .ok_or_else(|| PyKeyError::new_err((state.layout, state.lineage, state.version)))?;
            assert!(resource.is_none_or(|r| r == group.resource));
            resource = Some(group.resource);
            let width = group.slots.len() / group.refs.len();
            slots.extend_from_slice(&group.slots[row * width..(row + 1) * width]);
        }
        Ok((resource.unwrap_or(0), slots))
    }
}

fn unique_events(events: impl IntoIterator<Item = i64>) -> Vec<i64> {
    let mut result = Vec::new();
    for event in events {
        if event != 0 && !result.contains(&event) {
            result.push(event);
        }
    }
    result
}

fn admit_pages(
    resource: usize,
    table: &mut Table,
    pool: &mut Pool,
    refs: Vec<StateRef>,
    waits: &mut Vec<i64>,
) -> PyResult<Pages> {
    let rows: Vec<_> = table
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.status == 0)
        .map(|(i, _)| i)
        .take(refs.len())
        .collect();
    assert_eq!(rows.len(), refs.len());
    for (&row, state) in rows.iter().zip(&refs) {
        table.rows[row].lineage = state.lineage;
        table.rows[row].status = 1;
        table.rows[row].owners = 1;
        table.row_of.insert(state.lineage, row);
    }
    let mut group = Pages {
        resource,
        slots: vec![0; refs.len() * pool.width],
        generations: vec![0; refs.len() * pool.width],
        written: Some(vec![true; refs.len() * pool.width]),
        held: vec![true; pool.width],
        refs,
        rows,
        readiness: Vec::new(),
    };
    for field in 0..pool.width {
        let (slots, generations, events) = pool.allocate(field, group.rows.len())?;
        waits.extend(events);
        for i in 0..group.rows.len() {
            group.slots[i * pool.width + field] = slots[i];
            group.generations[i * pool.width + field] = generations[i];
        }
    }
    Ok(group)
}

fn transition_pages(
    source: &Pages,
    mask: Vec<bool>,
    table: &mut Table,
    pool: &mut Pool,
    waits: &mut Vec<i64>,
) -> PyResult<Pages> {
    let mut target = source.clone();
    target.readiness.clear();
    for state in &mut target.refs {
        state.version += 1;
    }
    for field in 0..pool.width {
        let indices: Vec<_> = (0..source.rows.len())
            .map(|i| i * pool.width + field)
            .filter(|&i| mask[i])
            .collect();
        let (slots, generations, events) = pool.allocate(field, indices.len())?;
        waits.extend(events);
        for ((&index, &slot), &generation) in indices.iter().zip(&slots).zip(&generations) {
            target.slots[index] = slot;
            target.generations[index] = generation;
        }
        let shared: Vec<_> = (0..source.rows.len())
            .map(|i| i * pool.width + field)
            .filter(|&i| !mask[i])
            .map(|i| source.slots[i])
            .collect();
        pool.share(field, &shared);
    }
    target.written = Some(mask);
    for &row in &source.rows {
        table.rows[row].owners += 1;
    }
    source.hold(table, pool);
    for &row in &source.rows {
        table.rows[row].writing = true;
    }
    Ok(target)
}

fn ref_columns(refs: &[StateRef], groups: &[Pages]) -> Vec<i64> {
    let rows: HashMap<_, _> = groups
        .iter()
        .flat_map(|p| p.refs.iter().zip(&p.rows).map(|(r, &row)| (r.lineage, row)))
        .collect();
    refs.iter()
        .map(|r| r.layout)
        .chain(refs.iter().map(|r| r.lineage))
        .chain(refs.iter().map(|r| r.version))
        .chain(refs.iter().map(|r| rows[&r.lineage] as i64))
        .collect()
}

fn copy_descriptors(write: &Write, copy_fields: Option<&[i64]>) -> Vec<Copy> {
    let mut copies = Vec::new();
    for (index, target) in write.targets.iter().enumerate() {
        let width = target.slots.len() / target.rows.len();
        let written = target.written.as_ref().unwrap();
        for field in 0..width {
            if !write.sources.is_empty()
                && copy_fields.is_some_and(|v| !v.contains(&(field as i64)))
            {
                continue;
            }
            let selected: Vec<_> = (0..target.rows.len())
                .map(|r| r * width + field)
                .filter(|&i| written[i])
                .collect();
            if selected.is_empty() {
                continue;
            }
            copies.push(Copy {
                resource: target.resource,
                field,
                source: write.sources.get(index).map_or_else(Vec::new, |source| {
                    selected.iter().map(|&i| source.slots[i]).collect()
                }),
                target: selected.iter().map(|&i| target.slots[i]).collect(),
            });
        }
    }
    copies
}

/// Thread-confined numeric arena; pool objects share ownership, never a second ledger.
#[pyclass(unsendable)]
pub struct PageLedger {
    arena: Arena,
    pools: Vec<Py<SlotLedger>>,
}

type CopyOutput<'py> = (usize, usize, Array<'py>, Array<'py>);
type PreparedOutput<'py> = (
    i64,
    Matrix<'py>,
    Vec<CopyOutput<'py>>,
    Array<'py>,
    Array<'py>,
);
type Snapshot<'py> = (
    usize,
    Matrix<'py>,
    Matrix<'py>,
    Matrix<'py>,
    Array<'py>,
    Option<Mask<'py>>,
);

fn matrix<'py>(py: Python<'py>, values: Vec<i64>, rows: usize, columns: usize) -> Matrix<'py> {
    numpy::ndarray::Array2::from_shape_vec((rows, columns), values)
        .unwrap()
        .into_pyarray(py)
}

fn snapshot<'py>(py: Python<'py>, pages: &Pages, width: usize) -> Snapshot<'py> {
    let count = pages.rows.len();
    (
        pages.resource,
        matrix(py, pages.columns(), 4, count),
        matrix(py, pages.slots.clone(), count, width),
        matrix(py, pages.generations.clone(), count, width),
        pages.readiness.clone().into_pyarray(py),
        pages.written.as_ref().map(|mask| {
            numpy::ndarray::Array2::from_shape_vec((count, width), mask.clone())
                .unwrap()
                .into_pyarray(py)
        }),
    )
}

impl PageLedger {
    fn with_pools<T>(
        &mut self,
        py: Python<'_>,
        action: impl FnOnce(&mut Arena, &mut [&mut Pool]) -> T,
    ) -> T {
        let mut guards: Vec<_> = self.pools.iter().map(|p| p.borrow_mut(py)).collect();
        let mut pools: Vec<_> = guards.iter_mut().map(|p| &mut p.pool).collect();
        let arena = &mut self.arena;
        // Short commits retain the GIL to avoid Python-thread scheduling between submissions.
        action(arena, &mut pools)
    }

    fn prepared<'py>(&mut self, py: Python<'py>, prepared: Prepared) -> PreparedOutput<'py> {
        let count = prepared.refs.len() / 4;
        let events = self.with_pools(py, |arena, pools| arena.events(pools));
        (
            prepared.id,
            matrix(py, prepared.refs, 4, count),
            prepared
                .copies
                .into_iter()
                .map(|c| {
                    (
                        c.resource,
                        c.field,
                        c.source.into_pyarray(py),
                        c.target.into_pyarray(py),
                    )
                })
                .collect(),
            prepared.waits.into_pyarray(py),
            events.into_pyarray(py),
        )
    }
}

#[pymethods]
impl PageLedger {
    #[new]
    fn new(
        py: Python<'_>,
        pools: Vec<Py<SlotLedger>>,
        specs: PyReadonlyArray2<'_, i64>,
        pinned: Vec<Vec<bool>>,
    ) -> PyResult<Self> {
        let specs = column(&specs)?;
        assert_eq!(specs.len(), pools.len() * 3);
        assert_eq!(pinned.len(), pools.len());
        let tables = pools
            .iter()
            .zip(pinned)
            .enumerate()
            .map(|(i, (p, pinned))| {
                let pool = &p.borrow(py).pool;
                assert_eq!(pinned.len(), pool.width);
                Table {
                    layout: specs[i * 3],
                    max_lineages: specs[i * 3 + 1] as usize,
                    mutable: specs[i * 3 + 2] != 0,
                    pinned,
                    // Retired lineages keep their row while a pin holds them; the largest field bounds both.
                    rows: (0..pool.capacities.iter().copied().max().unwrap_or(0))
                        .map(|_| Row::default())
                        .collect(),
                    row_of: HashMap::new(),
                }
            })
            .collect();
        Ok(Self {
            pools,
            arena: Arena {
                tables,
                writes: BTreeMap::new(),
                pins: BTreeMap::new(),
                releases: BTreeMap::new(),
                next_lineage: 1,
                next_operation: 1,
                next_pin: 1,
                next_release: 1,
                admission_failures: 0,
            },
        })
    }

    fn prepare_admission<'py>(
        &mut self,
        py: Python<'py>,
        resources: PyReadonlyArray1<'py, i64>,
        completed: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<PreparedOutput<'py>> {
        let (resources, completed) = (column(&resources)?, column(&completed)?);
        let prepared = self.with_pools(py, |arena, pools| {
            py.allow_threads(|| {
                for pool in pools.iter_mut() {
                    pool.collect(completed);
                }
                arena.prepare_admission(pools, resources)
            })
        })?;
        Ok(self.prepared(py, prepared))
    }

    #[pyo3(signature = (refs, fields, copy_fields, completed))]
    fn prepare_transition<'py>(
        &mut self,
        py: Python<'py>,
        refs: RefColumns<'py>,
        fields: Option<(PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>)>,
        copy_fields: Option<PyReadonlyArray1<'py, i64>>,
        completed: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<PreparedOutput<'py>> {
        let refs = Refs::read(&refs)?;
        let fields = fields
            .as_ref()
            .map(|(o, f)| Ok::<_, PyErr>((column(o)?, column(f)?)))
            .transpose()?;
        let copy_fields = copy_fields.as_ref().map(column).transpose()?;
        let completed = column(&completed)?;
        let prepared = self.with_pools(py, |arena, pools| {
            py.allow_threads(|| {
                for pool in pools.iter_mut() {
                    pool.collect(completed);
                }
                arena.prepare_transition(pools, &refs, fields, copy_fields)
            })
        })?;
        Ok(self.prepared(py, prepared))
    }

    fn publish<'py>(
        &mut self,
        py: Python<'py>,
        id: i64,
        completion: i64,
        completed: PyReadonlyArray1<'py, i64>,
        abort: bool,
    ) -> PyResult<Array<'py>> {
        let completed = column(&completed)?;
        let events = self.with_pools(py, |arena, pools| {
            for pool in pools.iter_mut() {
                pool.collect(completed);
            }
            if abort {
                arena.abort(pools, id, completion);
            } else {
                arena.publish(pools, id, completion);
            }
            arena.events(pools)
        });
        Ok(events.into_pyarray(py))
    }

    fn pin_exact<'py>(
        &mut self,
        py: Python<'py>,
        refs: RefColumns<'py>,
        stream: i64,
    ) -> PyResult<(i64, Array<'py>)> {
        let refs = Refs::read(&refs)?;
        let (id, waits) = self.with_pools(py, |arena, pools| {
            py.allow_threads(|| arena.pin(pools, &refs, stream))
        });
        Ok((id, waits.into_pyarray(py)))
    }

    fn release_pin<'py>(
        &mut self,
        py: Python<'py>,
        id: i64,
        completion: i64,
        defer: bool,
        stream: i64,
        completed: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(i64, Array<'py>)> {
        let completed = column(&completed)?;
        let (id, events) = self.with_pools(py, |arena, pools| {
            for pool in pools.iter_mut() {
                pool.collect(completed);
            }
            let id = arena.release_pin(pools, id, completion, defer, stream);
            (id, arena.events(pools))
        });
        Ok((id, events.into_pyarray(py)))
    }

    fn complete_release<'py>(
        &mut self,
        py: Python<'py>,
        id: i64,
        completed: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Array<'py>> {
        let completed = column(&completed)?;
        let events = self.with_pools(py, |arena, pools| {
            for pool in pools.iter_mut() {
                pool.collect(completed);
            }
            let release = arena
                .releases
                .remove(&id)
                .expect("complete each remote release once");
            arena.release_pages(pools, &release.pages, release.completion);
            arena.events(pools)
        });
        Ok(events.into_pyarray(py))
    }

    fn retire<'py>(&mut self, py: Python<'py>, refs: RefColumns<'py>) -> PyResult<Array<'py>> {
        let refs = Refs::read(&refs)?;
        let events = self.with_pools(py, |arena, pools| {
            let pages = arena.select(&refs, true);
            for group in &pages {
                let table = &mut arena.tables[group.resource];
                assert!(group.rows.iter().all(|&r| !table.rows[r].writing));
                for &row in &group.rows {
                    table.rows[row].status = 3;
                }
                group.release(table, pools[group.resource], 0);
            }
            arena.events(pools)
        });
        Ok(events.into_pyarray(py))
    }

    fn latest_batch<'py>(
        &self,
        py: Python<'py>,
        resource: usize,
        columns: PyReadonlyArray2<'py, i64>,
    ) -> PyResult<Array<'py>> {
        let columns = column(&columns)?;
        let count = columns.len() / 3;
        let table = &self.arena.tables[resource];
        Ok(py
            .allow_threads(|| {
                (0..count)
                    .map(|i| {
                        let row = &table.rows[columns[2 * count + i] as usize];
                        assert!(row.status == 2 && row.lineage == columns[count + i]);
                        row.version
                    })
                    .collect::<Vec<_>>()
            })
            .into_pyarray(py))
    }

    fn resolve_lineages<'py>(
        &self,
        py: Python<'py>,
        resource: usize,
        lineages: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Matrix<'py>> {
        let lineages = column(&lineages)?;
        let table = &self.arena.tables[resource];
        let rows: Vec<_> = lineages.iter().map(|l| table.row_of[l]).collect();
        assert!(rows.iter().all(|&r| table.rows[r].status == 2));
        let values = std::iter::repeat_n(table.layout, rows.len())
            .chain(lineages.iter().copied())
            .chain(rows.iter().map(|&r| table.rows[r].version))
            .chain(rows.iter().map(|&r| r as i64))
            .collect();
        Ok(matrix(py, values, 4, rows.len()))
    }

    fn latest(&self, lineage: i64) -> PyResult<(i64, i64, i64)> {
        for table in &self.arena.tables {
            if let Some(&row) = table.row_of.get(&lineage) {
                if table.rows[row].status == 2 {
                    let state = table.ref_at(row);
                    return Ok((state.layout, state.lineage, state.version));
                }
            }
        }
        Err(PyKeyError::new_err(lineage))
    }

    fn resource(&self, lineage: i64) -> PyResult<usize> {
        self.arena
            .tables
            .iter()
            .position(|t| t.row_of.contains_key(&lineage))
            .ok_or_else(|| PyKeyError::new_err(lineage))
    }

    fn valid_access(&self, access: i64, stream: i64) -> bool {
        self.arena.valid_access(access, stream)
    }

    fn move_pin(&mut self, id: i64, before_stream: i64, after_stream: i64) {
        assert!(self.arena.valid_access(id, before_stream));
        self.arena.pins.get_mut(&id).unwrap().stream = after_stream;
    }

    fn access_resource(&self, access: i64, stream: i64) -> usize {
        let pages = self.arena.access(access, stream);
        assert_eq!(pages.len(), 1);
        pages[0].resource
    }

    fn slots<'py>(
        &self,
        py: Python<'py>,
        access: PyReadonlyArray2<'py, i64>,
        owners: PyReadonlyArray1<'py, i64>,
        refs: RefColumns<'py>,
    ) -> PyResult<(usize, Matrix<'py>)> {
        let (access, owners, refs) = (column(&access)?, column(&owners)?, Refs::read(&refs)?);
        let (access, streams) = access.split_at(access.len() / 2);
        let count = refs.layouts.len();
        let arena = &self.arena;
        let (resource, slots) = py.allow_threads(|| arena.slots(access, streams, owners, &refs))?;
        let width = self
            .pools
            .get(resource)
            .map_or(0, |p| p.borrow(py).pool.width);
        Ok((resource, matrix(py, slots, count, width)))
    }

    fn access_pages<'py>(&self, py: Python<'py>, access: i64, stream: i64) -> Vec<Snapshot<'py>> {
        self.arena
            .access(access, stream)
            .iter()
            .map(|p| snapshot(py, p, self.pools[p.resource].borrow(py).pool.width))
            .collect()
    }

    fn inspect<'py>(&self, py: Python<'py>, category: &str) -> Vec<Snapshot<'py>> {
        let pages: Vec<_> = match category {
            "held" => self.arena.held_pages().cloned().collect(),
            "private" => self
                .arena
                .writes
                .values()
                .flat_map(|w| &w.targets)
                .cloned()
                .collect(),
            "live" => self
                .arena
                .tables
                .iter()
                .enumerate()
                .filter_map(|(resource, t)| {
                    let rows: Vec<_> = t
                        .rows
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.status == 2)
                        .map(|(i, _)| i)
                        .collect();
                    if rows.is_empty() {
                        return None;
                    }
                    Some(Pages {
                        resource,
                        refs: rows.iter().map(|&r| t.ref_at(r)).collect(),
                        slots: rows
                            .iter()
                            .flat_map(|&r| t.rows[r].slots.iter().copied())
                            .collect(),
                        generations: rows
                            .iter()
                            .flat_map(|&r| t.rows[r].generations.iter().copied())
                            .collect(),
                        readiness: rows.iter().map(|&r| t.rows[r].readiness).collect(),
                        rows,
                        written: None,
                        held: vec![true; t.pinned.len()],
                    })
                })
                .collect(),
            _ => unreachable!(),
        };
        pages
            .iter()
            .map(|p| snapshot(py, p, self.pools[p.resource].borrow(py).pool.width))
            .collect()
    }

    #[getter]
    fn active_states(&self) -> usize {
        self.arena
            .tables
            .iter()
            .flat_map(|t| &t.rows)
            .filter(|r| r.status >= 2)
            .count()
    }

    #[getter]
    fn admission_failures(&self) -> usize {
        self.arena.admission_failures
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PageLedger>()
}

//! Bounds for the same structured slots and releases executed by RegionActions.
//! Alias results are conservatively charged at their possible backing size.

use std::collections::{BTreeMap, BTreeSet};

use super::aer::{CallDeviceMemory, CallMemory};
use super::region_control::{Action, ControlBlock};
use super::region_program::{Extent, Operation, Program, Type};

type Result<T> = std::result::Result<T, String>;
type Bytes = BTreeMap<String, u64>;

#[derive(Clone, Default)]
pub(super) struct Memory {
    pub devices: BTreeMap<String, CallDeviceMemory>,
    pub unknown_functions: BTreeSet<String>,
}

fn plus(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| "structured memory bound overflows u64".into())
}

fn times(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| "structured memory bound overflows u64".into())
}

fn add_bytes(target: &mut Bytes, source: &Bytes) -> Result<()> {
    for (device, bytes) in source { *target.entry(device.clone()).or_default() = plus(target.get(device).copied().unwrap_or(0), *bytes)?; }
    Ok(())
}

pub(super) struct Context<'a> {
    pub program: &'a Program,
    pub backings: &'a Bytes,
    pub calls: &'a BTreeMap<&'a str, CallMemory>,
}

impl Context<'_> {
    fn extent(&self, extent: &Extent) -> Result<u64> {
        match extent {
            Extent::Fixed(n) => Ok(*n),
            Extent::Parameter(name) => self.program.parameters.iter().find(|p| &p.name == name)
                .map(|p| p.maximum).ok_or_else(|| "structured memory has an unknown size parameter".into()),
        }
    }

    fn value(&self, type_: &Type) -> Result<Bytes> {
        match type_ {
            Type::Tensor { device, .. } => Ok(BTreeMap::from([(device.clone(), *self.backings.get(device)
                .ok_or("structured tensor has no bounded backing storage")?)])),
            Type::Bank { element, size } => {
                let n = self.extent(size)?;
                self.value(element)?.into_iter().map(|(device, bytes)| Ok((device, times(n, bytes)?))).collect()
            },
            _ => Ok(Bytes::new()),
        }
    }

    fn slots(&self, block: &ControlBlock) -> Result<BTreeMap<usize, Bytes>> {
        block.arguments.iter().chain(block.steps.iter().flat_map(|s| &s.outputs))
            .map(|slot| Ok((slot.id, self.value(&slot.type_)?))).collect()
    }

    fn block(&self, block: &ControlBlock) -> Result<Memory> {
        let slots = self.slots(block)?;
        let mut live: BTreeSet<_> = block.arguments.iter().filter(|s| !block.discard_arguments.contains(&s.id)).map(|s| s.id).collect();
        let mut memory = Memory::default();
        for step in &block.steps {
            let action = self.action(&step.action)?;
            memory.sequence(&action)?;
            // Python's operands and result tuple coexist until _step returns.
            // Charge even discarded arguments if they are used by this action.
            live.extend(&step.inputs);
            live.extend(step.outputs.iter().map(|s| s.id));
            let mut bytes = Bytes::new();
            for id in &live { add_bytes(&mut bytes, &slots[id])?; }
            memory.peak_with(&bytes, &action)?;
            for id in &step.releases { live.remove(id); }
        }
        // Empty blocks can still return a borrowed carry or view.
        let mut bytes = Bytes::new();
        for id in &block.results { add_bytes(&mut bytes, &slots[id])?; }
        memory.peak_with(&bytes, &Memory::default())?;
        Ok(memory)
    }

    pub fn action(&self, action: &Action) -> Result<Memory> {
        match action {
            Action::Primitive { operation: Operation::Call { function, .. } } => {
                let call = self.calls.get(function.as_str()).ok_or("structured call has no storage import")?;
                Ok(Memory { devices: call.devices.clone(), unknown_functions: if call.unknown {
                    BTreeSet::from([function.clone()])
                } else { BTreeSet::new() } })
            },
            Action::Primitive { operation } => match operation {
                Operation::Snapshot { resource, .. } | Operation::Publish { resource, .. } => {
                    let payload = self.program.resources.iter().find(|r| r.name() == resource)
                        .and_then(|r| r.payload()).ok_or("page copy requires a declared state payload")?;
                    let mut memory = Memory::default();
                    for type_ in payload {
                        if let Some((device, bytes)) = super::region_imports::native_copy_bytes(type_, &self.program.parameters)? {
                            let facts = memory.devices.entry(device.to_owned()).or_default();
                            facts.logical_read_bytes = plus(facts.logical_read_bytes, bytes)?;
                            facts.logical_write_bytes = plus(facts.logical_write_bytes, bytes)?;
                        }
                    }
                    // Snapshot outputs are roots (or structured slots). A
                    // publication copies into the original arena's write token;
                    // its COW storage and reclaim remain in the PTTS budget.
                    Ok(memory)
                },
                Operation::Constant { .. } | Operation::Parameter { .. } | Operation::Scalar { .. }
                    | Operation::Select { .. } | Operation::Update { .. } => Ok(Memory::default()),
                _ => Err("structured storage requires imported tensor calls; PTTS payloads keep their original budget".into()),
            },
            Action::If { then_block, else_block } => {
                let mut memory = self.block(then_block)?;
                memory.alternative(&self.block(else_block)?);
                Ok(memory)
            },
            Action::While { limit, condition, body } => {
                if *limit == 0 { return Ok(Memory::default()); }
                let mut memory = self.block(condition)?;
                memory.sequence(&self.block(body)?)?;
                memory.repeat(*limit)?;
                // The original input tuple and the previous carry may both
                // remain referenced while the next carry is being produced.
                let mut carry = Bytes::new();
                for argument in &body.arguments { add_bytes(&mut carry, &self.value(&argument.type_)?)?; }
                memory.retain(&carry)?;
                Ok(memory)
            },
            Action::Scan { count, body, .. } | Action::Map { count, body } => {
                let n = self.extent(count)?;
                if n == 0 { return Ok(Memory::default()); }
                let mut memory = self.block(body)?;
                memory.repeat(n)?;
                let carry = if let Action::Scan { invariants, .. } = action {
                    body.arguments.len() - 1 - invariants
                } else { 0 };
                let slots = self.slots(body)?;
                let mut retained = Bytes::new();
                for id in &body.results[carry..] {
                    let scaled = slots[id].iter().map(|(d, b)| Ok((d.clone(), times(*b, n)?))).collect::<Result<Bytes>>()?;
                    add_bytes(&mut retained, &scaled)?;
                }
                // The Python loop also holds the previous result tuple until
                // the next iteration finishes; its carry is not a new store.
                for id in &body.results[..carry] { add_bytes(&mut retained, &slots[id])?; }
                memory.retain(&retained)?;
                Ok(memory)
            },
        }
    }
}

impl Memory {
    fn sequence(&mut self, other: &Self) -> Result<()> {
        self.unknown_functions.extend(other.unknown_functions.iter().cloned());
        for (name, facts) in &other.devices {
            let target = self.devices.entry(name.clone()).or_default();
            target.scratch_bytes = target.scratch_bytes.max(facts.scratch_bytes);
            target.logical_read_bytes = plus(target.logical_read_bytes, facts.logical_read_bytes)?;
            target.logical_write_bytes = plus(target.logical_write_bytes, facts.logical_write_bytes)?;
        }
        Ok(())
    }

    fn alternative(&mut self, other: &Self) {
        self.unknown_functions.extend(other.unknown_functions.iter().cloned());
        for (name, facts) in &other.devices {
            let target = self.devices.entry(name.clone()).or_default();
            target.scratch_bytes = target.scratch_bytes.max(facts.scratch_bytes);
            target.logical_read_bytes = target.logical_read_bytes.max(facts.logical_read_bytes);
            target.logical_write_bytes = target.logical_write_bytes.max(facts.logical_write_bytes);
        }
    }

    fn peak_with(&mut self, live: &Bytes, action: &Self) -> Result<()> {
        let devices: BTreeSet<_> = live.keys().chain(action.devices.keys()).collect();
        for name in devices {
            let bytes = plus(live.get(name).copied().unwrap_or(0), action.devices.get(name).map_or(0, |d| d.scratch_bytes))?;
            let facts = self.devices.entry(name.clone()).or_default();
            facts.scratch_bytes = facts.scratch_bytes.max(bytes);
        }
        Ok(())
    }

    fn retain(&mut self, bytes: &Bytes) -> Result<()> {
        for (name, bytes) in bytes {
            let facts = self.devices.entry(name.clone()).or_default();
            facts.scratch_bytes = plus(facts.scratch_bytes, *bytes)?;
        }
        Ok(())
    }

    fn repeat(&mut self, count: u64) -> Result<()> {
        for facts in self.devices.values_mut() {
            facts.logical_read_bytes = times(facts.logical_read_bytes, count)?;
            facts.logical_write_bytes = times(facts.logical_write_bytes, count)?;
        }
        Ok(())
    }
}

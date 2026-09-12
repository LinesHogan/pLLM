//! Structural regions lowered to data slots. Effects keep source order inside
//! each opaque boundary; loop trips and branch decisions stay at runtime.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::region_dependence::{Result, Statement};
use super::region_program::{Extent, Operation, Program, Region, Type, Value};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Slot {
    pub id: usize,
    #[serde(rename = "type")]
    pub type_: Type,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Action {
    Primitive {
        operation: Operation,
    },
    If {
        then_block: ControlBlock,
        else_block: ControlBlock,
    },
    While {
        limit: u64,
        condition: ControlBlock,
        body: ControlBlock,
    },
    Scan {
        count: Extent,
        body: ControlBlock,
        #[serde(default)]
        invariants: usize,
    },
    Map {
        count: Extent,
        body: ControlBlock,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Step {
    pub inputs: Vec<usize>,
    pub outputs: Vec<Slot>,
    pub action: Action,
    pub releases: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct ControlBlock {
    pub arguments: Vec<Slot>,
    pub steps: Vec<Step>,
    pub results: Vec<usize>,
    pub discard_arguments: Vec<usize>,
}

pub(super) fn action(operation: &Operation) -> Result<Action> {
    Ok(match operation {
        Operation::If {
            then_region,
            else_region,
            ..
        } => Action::If {
            then_block: block(then_region)?,
            else_block: block(else_region)?,
        },
        Operation::While {
            limit,
            condition,
            body,
            ..
        } => Action::While {
            limit: *limit,
            condition: block(condition)?,
            body: block(body)?,
        },
        Operation::Scan { count, body, invariants, .. } => Action::Scan {
            count: count.clone(),
            body: block(body)?,
            invariants: *invariants,
        },
        Operation::Map { count, body, .. } => Action::Map {
            count: count.clone(),
            body: block(body)?,
        },
        _ => Action::Primitive {
            operation: operation.clone(),
        },
    })
}

fn define<'a>(
    values: &'a [Value],
    env: &mut BTreeMap<&'a str, usize>,
    next: &mut usize,
) -> Vec<Slot> {
    let mut slots = Vec::new();
    for value in values {
        let id = *next;
        *next += 1;
        env.insert(&value.name, id);
        slots.push(Slot { id, type_: value.type_.clone() });
    }
    slots
}

fn inputs(names: &[String], env: &BTreeMap<&str, usize>) -> Result<Vec<usize>> {
    names.iter().map(|name| env.get(name.as_str()).copied()
        .ok_or_else(|| "undefined structural operand".into())).collect()
}

fn block(region: &Region) -> Result<ControlBlock> {
    let mut env = BTreeMap::new();
    let mut next = 0;
    let arguments = define(&region.block.arguments, &mut env, &mut next);
    let mut steps = Vec::new();
    let mut last_use = BTreeMap::new();
    for (i, op) in region.block.operations.iter().enumerate() {
        let (ins, outs) = op.ports();
        let ins = inputs(ins, &env)?;
        let outputs = define(outs, &mut env, &mut next);
        for &slot in &ins {
            last_use.insert(slot, i);
        }
        // Unused outputs still execute (and may fail), then die immediately.
        for slot in &outputs {
            last_use.insert(slot.id, i);
        }
        steps.push(Step {
            inputs: ins,
            outputs,
            action: action(op)?,
            releases: vec![],
        });
    }
    let results = inputs(&region.block.yields, &env)?;
    for &slot in &results {
        last_use.insert(slot, steps.len());
    }
    let discard_arguments = arguments
        .iter()
        .filter(|s| !last_use.contains_key(&s.id))
        .map(|s| s.id)
        .collect();
    for (slot, step) in last_use {
        if let Some(step) = steps.get_mut(step) {
            step.releases.push(slot);
        }
    }
    Ok(ControlBlock {
        arguments,
        steps,
        results,
        discard_arguments,
    })
}

pub(super) fn statements(program: &Program, statements: &[Statement]) -> Result<Vec<Action>> {
    statements
        .iter()
        .map(|statement| {
            let path = &statement.path;
            if path.len() % 2 != 1 {
                return Err("source statement path must identify an operation".into());
            }
            let mut region = &program.entry;
            for pair in path[..path.len() - 1].chunks(2) {
                let op = region
                    .block
                    .operations
                    .get(pair[0] as usize)
                    .ok_or("invalid source statement path")?;
                region = match op {
                    Operation::Scan { body, .. } | Operation::Map { body, .. } if pair[1] == 0 => {
                        body
                    }
                    _ => return Err("source statement path must descend through a loop".into()),
                };
            }
            action(
                region
                    .block
                    .operations
                    .get(path[path.len() - 1] as usize)
                    .ok_or("invalid source statement path")?,
            )
        })
        .collect()
}

//! Prove an identity-carry matrix update and derive its ordered, windowed kernel.
//! Region selection replays this rule from the source imports before admission.

use std::collections::BTreeMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use super::region_imports::{KernelAllocation, TensorImport, tensor_bytes};
use super::region_program::{self, Function, Operation, Program, Type};
use super::semantic_graph::{canonical_ascii_json, digest};
use super::tensor_access::{Access, AccessGraph};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Evidence { pub window: u64, pub workspace: u64 }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    pub program: Program,
    pub implementation: TensorImport,
    pub scale: f64,
    pub window: u64,
}

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Input(usize),
    Add(Box<Expr>, Box<Expr>),
    Transpose(Box<Expr>),
    Matmul(Box<Expr>, Box<Expr>),
    Scale(Box<Expr>, f64),
}

fn expression(op: &Access, values: &BTreeMap<String, Expr>) -> Option<Expr> {
    let args = &op.arguments;
    let operand = |i: usize| values.get(args.get(i)?.strip_prefix('$')?).cloned();
    if !op.keywords.is_empty() { return None; }
    match op.operator.as_str() {
        "aten.add.Tensor" if args.len() == 2 => Some(Expr::Add(Box::new(operand(0)?), Box::new(operand(1)?))),
        "aten.mm.default" | "aten.matmul.default" if args.len() == 2 => Some(Expr::Matmul(Box::new(operand(0)?), Box::new(operand(1)?))),
        "aten.permute.default" if args.len() == 2 && args[1] == "[1, 0]" => Some(Expr::Transpose(Box::new(operand(0)?))),
        "aten.t.default" if args.len() == 1 => Some(Expr::Transpose(Box::new(operand(0)?))),
        "aten.mul.Tensor" if args.len() == 2 => {
            let scale = args[1].parse::<f64>().ok()?;
            if scale.is_finite() && (scale as f32).is_finite() { Some(Expr::Scale(Box::new(operand(0)?), scale)) } else { None }
        },
        _ => None,
    }
}

fn inline(access: &AccessGraph, inputs: &[String], outputs: &[super::region_program::Value],
          values: &mut BTreeMap<String, Expr>, dtype: &str) -> Option<()> {
    // All literals come from the same export boundary as access/alias facts.
    // Unknown ops, RNG, downcasts and detach never enter this closed rule.
    if access.inputs.len() != inputs.len() || access.outputs.len() != outputs.len()
        || access.values.iter().any(|v| v.dtype != dtype) { return None; }
    let mut local = BTreeMap::new();
    for (port, name) in access.inputs.iter().zip(inputs) {
        if values.get(name)? == &Expr::Input(0) {
            let view = access.values.iter().find(|v| &v.name == port)?;
            if view.shape.len() != 2 || view.stride != vec![view.shape[1], 1] { return None; }
        }
        local.insert(port.clone(), values.get(name)?.clone());
    }
    for op in &access.operations {
        let result = expression(op, &local)?;
        local.insert(op.node.clone(), result);
    }
    for (port, output) in access.outputs.iter().zip(outputs) {
        values.insert(output.name.clone(), local.get(port)?.clone());
    }
    Some(())
}

fn recognize(program: &Program, imports: &[TensorImport]) -> Option<f64> {
    let block = &program.entry.block;
    if block.arguments.len() != 4 || block.operations.len() != 1 || !program.resources.is_empty() { return None; }
    let Operation::Scan { inputs, outputs, body, invariants: 3, .. } = &block.operations[0] else { return None; };
    if inputs != &block.arguments.iter().map(|v| v.name.clone()).collect::<Vec<_>>()
        || outputs.len() != 2 || body.block.arguments.len() != 5
        || block.yields != outputs.iter().map(|v| v.name.clone()).collect::<Vec<_>>() { return None; }
    let Type::Tensor { dtype, .. } = &block.arguments[0].type_ else { return None; };
    let mut values = BTreeMap::from([
        (body.block.arguments[1].name.clone(), Expr::Input(0)),
        (body.block.arguments[2].name.clone(), Expr::Input(1)),
    ]);
    for op in &body.block.operations {
        match op {
            Operation::Select { inputs, outputs } if inputs.len() == 2 && outputs.len() == 1
                && inputs[1] == body.block.arguments[0].name => {
                let port = (3..5).find(|&i| inputs[0] == body.block.arguments[i].name)?;
                values.insert(outputs[0].name.clone(), Expr::Input(port - 1));
            },
            Operation::Call { inputs, outputs, function } => {
                let TensorImport::Exported { access } = imports.iter().find(|i| i.name() == function)? else { return None; };
                inline(access, inputs, outputs, &mut values, dtype)?;
            },
            _ => return None,
        }
    }
    if body.block.yields.len() != 2 { return None; }
    let Expr::Add(carry, increment) = values.get(&body.block.yields[0])? else { return None; };
    let Expr::Scale(product, scale) = increment.as_ref() else { return None; };
    let input = |i| Box::new(Expr::Input(i));
    let transposed = |x| Box::new(Expr::Transpose(x));
    let expected_increment = Expr::Matmul(transposed(input(3)), input(2));
    let expected_output = Expr::Matmul(input(2), transposed(Box::new(Expr::Add(input(1), input(0)))));
    (carry.as_ref() == &Expr::Input(0) && product.as_ref() == &expected_increment
        && values.get(&body.block.yields[1])? == &expected_output).then_some(*scale)
}

fn total(parts: &[u64]) -> Result<u64, String> {
    parts.iter().try_fold(0_u64, |n, x| n.checked_add(*x).ok_or("additive scan allocation overflows".into()))
}
fn times(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b).ok_or("additive scan allocation overflows".into())
}

pub(super) fn derive(program: &Program, imports: &[TensorImport], evidence: &Evidence) -> Result<Option<Plan>, String> {
    let block = &program.entry.block;
    let Some(carry) = block.arguments.first() else { return Ok(None); };
    let Type::Tensor { shape, dtype, device } = &carry.type_ else { return Ok(None); };
    if shape.len() != 2 || !device.starts_with("cuda") || !matches!(dtype.as_str(), "torch.float32" | "torch.bfloat16")
        || shape.contains(&0) { return Ok(None); }
    region_program::verify(program)?;
    if evidence.window < 2 || evidence.window > 64 { return Err("additive scan window must be between 2 and 64".into()); }
    for function in &program.functions {
        imports.iter().find(|i| i.name() == function.name).ok_or("missing additive source import")?.validate(function)?;
    }
    let Some(scale) = recognize(program, imports) else { return Ok(None); };
    let Operation::Scan { inputs, outputs, count, .. } = &block.operations[0] else { unreachable!() };
    if block.arguments[1].type_ != block.arguments[0].type_ { return Ok(None); }
    let Type::Bank { element: x, size: xs } = &block.arguments[2].type_ else { return Ok(None); };
    let Type::Bank { element: y, size: ys } = &block.arguments[3].type_ else { return Ok(None); };
    let (Type::Tensor { shape: sx, dtype: dx, device: vx }, Type::Tensor { shape: sy, dtype: dy, device: vy }) = (x.as_ref(), y.as_ref()) else { return Ok(None); };
    if outputs[0].type_ != block.arguments[0].type_ || outputs[1].type_ != block.arguments[3].type_ { return Ok(None); }
    if sx.len() != 2 || sy != &vec![sx[0], shape[0]] || sx[1] != shape[1] || sx[0] == 0
        || xs != count || ys != count || dx != dtype || dy != dtype || vx != device || vy != device { return Ok(None); }
    let capacity = count.maximum(program);
    if capacity < evidence.window { return Ok(None); }
    let window = evidence.window;
    let carry_bytes = tensor_bytes(&block.arguments[0].type_)?;
    let packed_inputs = times(window, total(&[tensor_bytes(x)?, tensor_bytes(y)?])?)?;
    // Outputs are independent bank leaves. Raw batched outputs live until each
    // leaf is copied; the frame separately charges the returned leaves.
    let packed_outputs = times(window, tensor_bytes(y)?)?;
    let scratch = total(&[packed_inputs, times(2 * window + 2, carry_bytes)?, packed_outputs, evidence.workspace])?;
    let function = Function {
        name: digest(&canonical_ascii_json(&("ordered-additive-scan@1", program, imports, evidence))?),
        inputs: block.arguments.iter().map(|v| v.type_.clone()).collect(),
        outputs: outputs.iter().map(|v| v.type_.clone()).collect(), resources: vec![], accesses: vec![],
    };
    let implementation = TensorImport::OrderedScan {
        declaration: function.clone(), device: device.clone(),
        allocations: vec![KernelAllocation { name: "ordered-window".into(), device: device.clone(), bytes: scratch }],
    };
    let mut result = program.clone();
    result.entry.block.operations = vec![Operation::Call { inputs: inputs.clone(), outputs: outputs.clone(), function: function.name.clone() }];
    result.functions.push(function);
    region_program::verify(&result)?;
    Ok(Some(Plan { program: result, implementation, scale, window }))
}

#[pyfunction]
fn derive_additive_scan(program_json: &str, imports_json: &str, evidence_json: &str) -> PyResult<String> {
    let run = || -> Result<String, String> {
        let program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let evidence = serde_json::from_str(evidence_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(&program, &imports, &evidence)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(derive_additive_scan, module)?)?;
    Ok(())
}

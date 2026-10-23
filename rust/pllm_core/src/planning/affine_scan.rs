//! Prove identity and right-matrix carry rewrites for the region selector.

use std::collections::BTreeMap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use super::region_imports::{KernelAllocation, TensorImport, tensor_bytes};
use super::region_program::{self, Function, Operation, Program, Type};
use super::semantic_graph::{canonical_ascii_json, digest};
use super::tensor_access::{Access, AccessGraph};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Evidence { pub window: u64, pub workspace: u64, pub form: Form }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Form { Ordered, HillisSteele }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Action { Identity, RightMatrix }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    pub program: Program,
    pub implementation: TensorImport,
    pub scale: f64,
    pub decay: f64,
    pub window: u64,
    pub chunk: u64,
}

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Input(usize),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Permute(Box<Expr>, Vec<usize>),
    Matmul(Box<Expr>, Box<Expr>),
    Scale(Box<Expr>, f64),
}

fn expression(op: &Access, graph: &AccessGraph, values: &BTreeMap<String, Expr>) -> Option<Expr> {
    let args = &op.arguments;
    let operand = |i: usize| values.get(args.get(i)?.strip_prefix('$')?).cloned();
    if !op.keywords.is_empty() { return None; }
    match op.operator.as_str() {
        "aten.add.Tensor" if args.len() == 2 => Some(Expr::Add(Box::new(operand(0)?), Box::new(operand(1)?))),
        "aten.sub.Tensor" if args.len() == 2 => Some(Expr::Sub(Box::new(operand(0)?), Box::new(operand(1)?))),
        "aten.mm.default" | "aten.bmm.default" | "aten.matmul.default" if args.len() == 2 => Some(Expr::Matmul(Box::new(operand(0)?), Box::new(operand(1)?))),
        "aten.permute.default" if args.len() == 2 => Some(Expr::Permute(Box::new(operand(0)?), serde_json::from_str(&args[1]).ok()?)),
        "aten.t.default" if args.len() == 1 => Some(Expr::Permute(Box::new(operand(0)?), vec![1, 0])),
        "aten.transpose.int" if args.len() == 3 => {
            let rank = graph.values.iter().find(|v| Some(v.name.as_str()) == args[0].strip_prefix('$'))?.shape.len();
            let mut axes: Vec<_> = (0..rank).collect();
            let axis = |i: usize| -> Option<usize> {
                let n = args[i].parse::<i64>().ok()?;
                (n >= -(rank as i64) && n < rank as i64).then(|| n.rem_euclid(rank as i64) as usize)
            };
            axes.swap(axis(1)?, axis(2)?);
            Some(Expr::Permute(Box::new(operand(0)?), axes))
        },
        "aten.mul.Tensor" if args.len() == 2 => {
            let scale = args[1].parse::<f64>().ok()?;
            if scale.is_finite() && (scale as f32).is_finite() { Some(Expr::Scale(Box::new(operand(0)?), scale)) } else { None }
        },
        _ => None,
    }
}

fn inline(access: &AccessGraph, inputs: &[String], outputs: &[super::region_program::Value],
          values: &mut BTreeMap<String, Expr>, dtype: &str) -> Option<()> {
    // Walk every operation, including output-only branches: unknown effects,
    // RNG, detach and dtype changes cannot hide outside the carry slice.
    if access.inputs.len() != inputs.len() || access.outputs.len() != outputs.len()
        || access.values.iter().any(|v| v.dtype != dtype) { return None; }
    let mut local = BTreeMap::new();
    for (port, name) in access.inputs.iter().zip(inputs) {
        if values.get(name)? == &Expr::Input(0) {
            let view = access.values.iter().find(|v| &v.name == port)?;
            let mut stride = 1;
            for (&size, &actual) in view.shape.iter().zip(&view.stride).rev() {
                if actual != stride { return None; }
                stride = stride.checked_mul(size)?;
            }
        }
        local.insert(port.clone(), values.get(name)?.clone());
    }
    for op in &access.operations { local.insert(op.node.clone(), expression(op, access, &local)?); }
    for (port, output) in access.outputs.iter().zip(outputs) { values.insert(output.name.clone(), local.get(port)?.clone()); }
    Some(())
}

fn recognize(program: &Program, imports: &[TensorImport]) -> Option<(Action, f64, f64)> {
    let block = &program.entry.block;
    let Operation::Scan { inputs, outputs, body, invariants, .. } = &block.operations[0] else { return None; };
    if *invariants != block.arguments.len() - 1 || body.block.arguments.len() != block.arguments.len() + 1
        || inputs != &block.arguments.iter().map(|v| v.name.clone()).collect::<Vec<_>>()
        || outputs.len() != 2 || block.yields != outputs.iter().map(|v| v.name.clone()).collect::<Vec<_>>() { return None; }
    let Type::Tensor { dtype, .. } = &block.arguments[0].type_ else { return None; };
    let mut values = BTreeMap::new();
    for (i, arg) in block.arguments.iter().enumerate() {
        if matches!(arg.type_, Type::Tensor { .. }) { values.insert(body.block.arguments[i + 1].name.clone(), Expr::Input(i)); }
    }
    for op in &body.block.operations {
        match op {
            Operation::Select { inputs, outputs } if inputs.len() == 2 && outputs.len() == 1 && inputs[1] == body.block.arguments[0].name => {
                let port = body.block.arguments.iter().position(|v| v.name == inputs[0])?;
                if port == 0 { return None; }
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
    let next = values.get(&body.block.yields[0])?;
    let output = values.get(&body.block.yields[1])?;
    let input = |i| Box::new(Expr::Input(i));
    let transpose = |x| Box::new(Expr::Permute(x, vec![1, 0]));
    if let Expr::Add(carry, increment) = next {
        let Expr::Scale(product, scale) = increment.as_ref() else { return None; };
        let expected = Expr::Matmul(transpose(input(2)), input(1));
        let read = Expr::Matmul(input(1), transpose(input(0)));
        return (block.arguments.len() == 3 && carry.as_ref() == &Expr::Input(0) && product.as_ref() == &expected && output == &read)
            .then_some((Action::Identity, *scale, 1.0));
    }
    let Expr::Sub(decayed, gradient) = next else { return None; };
    let decay = match decayed.as_ref() {
        Expr::Input(0) => 1.0,
        Expr::Scale(carry, decay) if carry.as_ref() == &Expr::Input(0) => *decay,
        _ => return None,
    };
    let Expr::Scale(product, scale) = gradient.as_ref() else { return None; };
    let heads = |x| Box::new(Expr::Permute(x, vec![1, 0, 2]));
    let transpose = |x| Box::new(Expr::Permute(x, vec![0, 2, 1]));
    let prediction = Box::new(Expr::Matmul(heads(input(2)), transpose(decayed.clone())));
    let expected = Expr::Matmul(transpose(Box::new(Expr::Sub(prediction, heads(input(1))))), heads(input(2)));
    let read = heads(Box::new(Expr::Matmul(heads(input(1)), transpose(input(0)))));
    // Distribute (alpha*S - beta*(K*(alpha*S)^T-V)^T*K):
    // S*A+B, A=alpha*(I-beta*K^T*K), B=beta*V^T*K.
    // Chronological right actions compose as (A1*A2, B1*A2+B2).
    (block.arguments.len() == 3 && product.as_ref() == &expected && output == read.as_ref())
        .then_some((Action::RightMatrix, *scale, decay))
}

fn total(parts: &[u64]) -> Result<u64, String> {
    parts.iter().try_fold(0_u64, |n, x| n.checked_add(*x).ok_or("affine scan allocation overflows".into()))
}
fn times(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b).ok_or("affine scan allocation overflows".into())
}

pub(super) fn derive(program: &Program, imports: &[TensorImport], evidence: &Evidence) -> Result<Option<Plan>, String> {
    let block = &program.entry.block;
    if block.arguments.len() != 3 || block.operations.len() != 1 || !program.resources.is_empty() { return Ok(None); }
    let Some(carry) = block.arguments.first() else { return Ok(None); };
    let Type::Tensor { shape, dtype, device } = &carry.type_ else { return Ok(None); };
    if !matches!(shape.len(), 2 | 3) || !device.starts_with("cuda") || !matches!(dtype.as_str(), "torch.float32" | "torch.bfloat16") || shape.contains(&0) { return Ok(None); }
    region_program::verify(program)?;
    if evidence.window == 0 || evidence.window > 64 || !evidence.window.is_power_of_two() {
        return Err("affine scan window must be a power of two between 1 and 64".into());
    }
    for function in &program.functions { imports.iter().find(|i| i.name() == function.name).ok_or("missing affine source import")?.validate(function)?; }
    let Some((action, scale, decay)) = recognize(program, imports) else { return Ok(None); };
    let Operation::Scan { inputs, outputs, count, .. } = &block.operations[0] else { unreachable!() };
    let first = 1;
    let Type::Bank { element: x, size: xs } = &block.arguments[first].type_ else { return Ok(None); };
    let Type::Bank { element: y, size: ys } = &block.arguments[first + 1].type_ else { return Ok(None); };
    let (Type::Tensor { shape: sx, dtype: dx, device: vx }, Type::Tensor { shape: sy, dtype: dy, device: vy }) = (x.as_ref(), y.as_ref()) else { return Ok(None); };
    if xs != count || ys != count || dx != dtype || dy != dtype || vx != device || vy != device || sx.is_empty() || sx[0] == 0 || count.maximum(program) <= evidence.window / 2 { return Ok(None); }
    match action {
        Action::Identity => if shape.len() != 2 || sx != &vec![sx[0], shape[1]] || sy != &vec![sx[0], shape[0]] { return Ok(None); },
        Action::RightMatrix => if dtype != "torch.float32" || shape.len() != 3 || shape[1] != shape[2] || sx != &vec![sx[0], shape[0], shape[1]] || sy != sx { return Ok(None); },
    }
    if outputs[0].type_ != carry.type_ || outputs[1].type_ != block.arguments[first + 1].type_ { return Ok(None); }
    let window = evidence.window;
    let matrices = if action == Action::Identity { 2 * window + 2 } else { 6 * window + 3 };
    // Both forms use the same conservative bound: inputs, pair ping-pong,
    // exclusive states, broadcast initial state, owned output and BLAS scratch.
    let scratch = total(&[times(window, total(&[tensor_bytes(x)?, tensor_bytes(y)?, tensor_bytes(y)?])?)?,
        times(matrices, tensor_bytes(&carry.type_)?)?, evidence.workspace, 512])?;
    let function = Function {
        name: digest(&canonical_ascii_json(&("affine-prefix@4", program, imports, evidence))?),
        inputs: block.arguments.iter().map(|v| v.type_.clone()).collect(), outputs: outputs.iter().map(|v| v.type_.clone()).collect(), resources: vec![], accesses: vec![],
    };
    let implementation = TensorImport::AffineScan { declaration: function.clone(), device: device.clone(),
        allocations: vec![KernelAllocation { name: "prefix-window".into(), device: device.clone(), bytes: scratch }], form: evidence.form.clone(), action };
    let mut result = program.clone();
    result.entry.block.operations = vec![Operation::Call { inputs: inputs.clone(), outputs: outputs.clone(), function: function.name.clone() }];
    result.functions.push(function);
    region_program::verify(&result)?;
    Ok(Some(Plan { program: result, implementation, scale, decay, window, chunk: sx[0] }))
}

#[pyfunction]
fn derive_affine_scan(program_json: &str, imports_json: &str, evidence_json: &str) -> PyResult<String> {
    let run = || -> Result<String, String> {
        let program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let imports: Vec<TensorImport> = serde_json::from_str(imports_json).map_err(|e| e.to_string())?;
        let evidence = serde_json::from_str(evidence_json).map_err(|e| e.to_string())?;
        canonical_ascii_json(&derive(&program, &imports, &evidence)?)
    };
    run().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(derive_affine_scan, module)?)?;
    Ok(())
}

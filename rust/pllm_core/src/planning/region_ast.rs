//! Lower a fixed, verified affine schedule to closed loop instructions.
//! No generated C/Python strings are evaluated and no loop instances are expanded.

use std::ffi::CStr;
use std::cell::RefCell;
use std::collections::VecDeque;

use isl_rs::{Context, UnionMap, UnionSet, Val};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_control::{self, Action};
use super::region_dependence::{
    analyze, canonical_schedule, context, isl, legal, map_text, read_schedule, Result, Statement,
};
use super::region_program::Program;
use super::semantic_graph::{canonical_ascii_json, digest};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Operator {
    And,
    AndThen,
    Or,
    OrElse,
    Max,
    Min,
    Neg,
    Add,
    Sub,
    Mul,
    ExactDiv,
    FloorDiv,
    PositiveDiv,
    PositiveMod,
    ZeroMod,
    Conditional,
    Select,
    Eq,
    Le,
    Lt,
    Ge,
    Gt,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Expression {
    Integer {
        value: i64,
    },
    Iterator {
        name: String,
    },
    Apply {
        operator: Operator,
        arguments: Vec<Expression>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Instruction {
    Invoke {
        statement_id: usize,
        coordinates: Vec<Expression>,
    },
    Loop {
        iterator: String,
        initial: Expression,
        condition: Expression,
        step: Expression,
        body: Box<Instruction>,
    },
    Sequence {
        children: Vec<Instruction>,
    },
    Guard {
        condition: Expression,
        if_true: Box<Instruction>,
        if_false: Box<Instruction>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LoweredSchedule {
    version: u32,
    program_digest: String,
    schedule_map: String,
    statements: Vec<Statement>,
    body: Instruction,
    actions: Vec<Action>,
    signature: String,
}

// Missing isl-rs AST bindings, pinned to its bundled isl/ast_type.h and ast.h.
// All __isl_give results have one RAII owner; borrowed arguments stay alive.
unsafe extern "C" {
    fn isl_ast_build_alloc(ctx: usize) -> usize;
    fn isl_ast_build_free(build: usize) -> usize;
    fn isl_ast_build_node_from_schedule_map(build: usize, map: usize) -> usize;
    fn isl_ast_node_free(node: usize) -> usize;
    fn isl_ast_node_get_type(node: usize) -> i32;
    fn isl_ast_node_for_get_iterator(node: usize) -> usize;
    fn isl_ast_node_for_get_init(node: usize) -> usize;
    fn isl_ast_node_for_get_cond(node: usize) -> usize;
    fn isl_ast_node_for_get_inc(node: usize) -> usize;
    fn isl_ast_node_for_get_body(node: usize) -> usize;
    fn isl_ast_node_if_get_cond(node: usize) -> usize;
    fn isl_ast_node_if_get_then_node(node: usize) -> usize;
    fn isl_ast_node_if_has_else_node(node: usize) -> i32;
    fn isl_ast_node_if_get_else_node(node: usize) -> usize;
    fn isl_ast_node_block_get_children(node: usize) -> usize;
    fn isl_ast_node_list_free(list: usize) -> usize;
    fn isl_ast_node_list_n_ast_node(list: usize) -> i32;
    fn isl_ast_node_list_get_at(list: usize, index: i32) -> usize;
    fn isl_ast_node_user_get_expr(node: usize) -> usize;
    fn isl_ast_expr_free(expr: usize) -> usize;
    fn isl_ast_expr_get_type(expr: usize) -> i32;
    fn isl_ast_expr_get_val(expr: usize) -> usize;
    fn isl_ast_expr_get_id(expr: usize) -> usize;
    fn isl_ast_expr_get_op_type(expr: usize) -> i32;
    fn isl_ast_expr_get_op_n_arg(expr: usize) -> i32;
    fn isl_ast_expr_get_op_arg(expr: usize, index: i32) -> usize;
    fn isl_id_get_name(id: usize) -> *const std::ffi::c_char;
    fn isl_id_free(id: usize) -> usize;
}

struct Owned {
    pointer: usize,
    free: unsafe extern "C" fn(usize) -> usize,
}
impl Owned {
    fn new(pointer: usize, free: unsafe extern "C" fn(usize) -> usize) -> Result<Self> {
        if pointer == 0 {
            Err("ISL AST allocation or accessor failed".into())
        } else {
            Ok(Self { pointer, free })
        }
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe {
            (self.free)(self.pointer);
        }
    }
}

fn identifier(expression: usize) -> Result<String> {
    if unsafe { isl_ast_expr_get_type(expression) } != 1 {
        return Err("AST identifier expected".into());
    }
    let id = Owned::new(unsafe { isl_ast_expr_get_id(expression) }, isl_id_free)?;
    let name = unsafe { isl_id_get_name(id.pointer) };
    if name.is_null() {
        return Err("ISL AST identifier has no name".into());
    }
    unsafe { CStr::from_ptr(name) }
        .to_str()
        .map(str::to_owned)
        .map_err(|e| e.to_string())
}

pub(super) fn expression(pointer: usize, ctx: &Context) -> Result<Expression> {
    let owner = Owned::new(pointer, isl_ast_expr_free)?;
    match unsafe { isl_ast_expr_get_type(owner.pointer) } {
        2 => {
            let pointer = unsafe { isl_ast_expr_get_val(owner.pointer) };
            if pointer == 0 {
                return Err("ISL AST integer accessor failed".into());
            }
            let value = Val {
                ptr: pointer,
                should_free_on_drop: true,
            };
            if !isl(value.is_int())?
                || !isl(value.ge(&isl(Val::int_from_si(ctx, i64::MIN))?))?
                || !isl(value.le(&isl(Val::int_from_si(ctx, i64::MAX))?))?
            {
                return Err("schedule instruction integer exceeds signed 64-bit range".into());
            }
            Ok(Expression::Integer {
                value: isl(value.get_num_si())?,
            })
        }
        1 => Ok(Expression::Iterator {
            name: identifier(owner.pointer)?,
        }),
        0 => {
            let kind = unsafe { isl_ast_expr_get_op_type(owner.pointer) };
            let operator = match kind {
                0 => Operator::And,
                1 => Operator::AndThen,
                2 => Operator::Or,
                3 => Operator::OrElse,
                4 => Operator::Max,
                5 => Operator::Min,
                6 => Operator::Neg,
                7 => Operator::Add,
                8 => Operator::Sub,
                9 => Operator::Mul,
                10 => Operator::ExactDiv,
                11 => Operator::FloorDiv,
                12 => Operator::PositiveDiv,
                13 => Operator::PositiveMod,
                14 => Operator::ZeroMod,
                15 => Operator::Conditional,
                16 => Operator::Select,
                17 => Operator::Eq,
                18 => Operator::Le,
                19 => Operator::Lt,
                20 => Operator::Ge,
                21 => Operator::Gt,
                _ => return Err(format!("unsupported schedule expression opcode {kind}")),
            };
            let count = unsafe { isl_ast_expr_get_op_n_arg(owner.pointer) };
            let valid = match operator {
                Operator::Neg => count == 1,
                Operator::Min | Operator::Max => count >= 1,
                Operator::Conditional | Operator::Select => count == 3,
                _ => count == 2,
            };
            if !valid {
                return Err("invalid schedule expression arity".into());
            }
            let arguments = (0..count)
                .map(|i| expression(unsafe { isl_ast_expr_get_op_arg(owner.pointer, i) }, ctx))
                .collect::<Result<Vec<_>>>()?;
            Ok(Expression::Apply {
                operator,
                arguments,
            })
        }
        _ => Err("unknown ISL AST expression".into()),
    }
}

fn instruction(pointer: usize, ctx: &Context, statements: &[Statement]) -> Result<Instruction> {
    let owner = Owned::new(pointer, isl_ast_node_free)?;
    match unsafe { isl_ast_node_get_type(owner.pointer) } {
        1 => {
            let index = Owned::new(
                unsafe { isl_ast_node_for_get_iterator(owner.pointer) },
                isl_ast_expr_free,
            )?;
            Ok(Instruction::Loop {
                iterator: identifier(index.pointer)?,
                initial: expression(unsafe { isl_ast_node_for_get_init(owner.pointer) }, ctx)?,
                condition: expression(unsafe { isl_ast_node_for_get_cond(owner.pointer) }, ctx)?,
                step: expression(unsafe { isl_ast_node_for_get_inc(owner.pointer) }, ctx)?,
                body: Box::new(instruction(
                    unsafe { isl_ast_node_for_get_body(owner.pointer) },
                    ctx,
                    statements,
                )?),
            })
        }
        2 => {
            let has_else = unsafe { isl_ast_node_if_has_else_node(owner.pointer) };
            if has_else < 0 {
                return Err("ISL AST else query failed".into());
            }
            Ok(Instruction::Guard {
                condition: expression(unsafe { isl_ast_node_if_get_cond(owner.pointer) }, ctx)?,
                if_true: Box::new(instruction(
                    unsafe { isl_ast_node_if_get_then_node(owner.pointer) },
                    ctx,
                    statements,
                )?),
                if_false: Box::new(if has_else == 1 {
                    instruction(
                        unsafe { isl_ast_node_if_get_else_node(owner.pointer) },
                        ctx,
                        statements,
                    )?
                } else {
                    Instruction::Sequence { children: vec![] }
                }),
            })
        }
        3 => {
            let list = Owned::new(
                unsafe { isl_ast_node_block_get_children(owner.pointer) },
                isl_ast_node_list_free,
            )?;
            let count = unsafe { isl_ast_node_list_n_ast_node(list.pointer) };
            if count < 0 {
                return Err("ISL AST child count failed".into());
            }
            Ok(Instruction::Sequence {
                children: (0..count)
                    .map(|i| {
                        instruction(
                            unsafe { isl_ast_node_list_get_at(list.pointer, i) },
                            ctx,
                            statements,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            })
        }
        5 => {
            let call = Owned::new(
                unsafe { isl_ast_node_user_get_expr(owner.pointer) },
                isl_ast_expr_free,
            )?;
            if unsafe { isl_ast_expr_get_type(call.pointer) } != 0
                || unsafe { isl_ast_expr_get_op_type(call.pointer) } != 22
            {
                return Err("schedule leaf must be a statement invocation".into());
            }
            let count = unsafe { isl_ast_expr_get_op_n_arg(call.pointer) };
            if count < 1 {
                return Err("schedule invocation has no target".into());
            }
            let target = Owned::new(
                unsafe { isl_ast_expr_get_op_arg(call.pointer, 0) },
                isl_ast_expr_free,
            )?;
            let name = identifier(target.pointer)?;
            let id = name
                .strip_prefix('S')
                .ok_or("unknown schedule target")?
                .parse::<usize>()
                .map_err(|e| e.to_string())?;
            if statements
                .get(id)
                .is_none_or(|s| s.id != id || s.extents.len() + 1 != count as usize)
            {
                return Err("schedule invocation differs from source statement arity".into());
            }
            Ok(Instruction::Invoke {
                statement_id: id,
                coordinates: (1..count)
                    .map(|i| expression(unsafe { isl_ast_expr_get_op_arg(call.pointer, i) }, ctx))
                    .collect::<Result<Vec<_>>>()?,
            })
        }
        other => Err(format!("unsupported schedule AST node {other}")),
    }
}

fn lower(program: &Program, schedule_map: &str) -> Result<LoweredSchedule> {
    thread_local! {
        static LOWERED: RefCell<VecDeque<(String, LoweredSchedule)>> = const { RefCell::new(VecDeque::new()) };
    }
    let schedule = canonical_schedule(schedule_map)?;
    let key = digest(&canonical_ascii_json(&(program, &schedule))?);
    super::region_cache::memoized(&LOWERED, key, || derive_lowering(program, &schedule))
}

fn derive_lowering(program: &Program, schedule_map: &str) -> Result<LoweredSchedule> {
    let analysis = analyze(program, false)?;
    let ctx = context()?;
    let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain))?;
    let dependencies = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies))?;
    let mut schedule = read_schedule(&ctx, schedule_map)?;
    legal(&domain, &dependencies, &schedule)?;
    let schedule_map = map_text(&schedule)?;
    let body = if isl(domain.is_empty())? {
        Instruction::Sequence { children: vec![] }
    } else {
        let build = Owned::new(unsafe { isl_ast_build_alloc(ctx.ptr) }, isl_ast_build_free)?;
        schedule.should_free_on_drop = false;
        let node = unsafe { isl_ast_build_node_from_schedule_map(build.pointer, schedule.ptr) };
        instruction(node, &ctx, &analysis.statements)?
    };
    let mut lowered = LoweredSchedule {
        version: 2,
        actions: region_control::statements(program, &analysis.statements)?,
        program_digest: analysis.program_digest,
        schedule_map,
        statements: analysis.statements,
        body,
        signature: String::new(),
    };
    lowered.signature = digest(&canonical_ascii_json(&lowered)?);
    Ok(lowered)
}

#[pyfunction]
fn lower_region_schedule(program_json: &str, schedule_map: &str) -> PyResult<String> {
    let program: Program = serde_json::from_str(program_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    canonical_ascii_json(
        &lower(&program, schedule_map).map_err(pyo3::exceptions::PyValueError::new_err)?,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_lowered_region_schedule(program_json: &str, lowered_json: &str) -> PyResult<()> {
    let validate = || -> Result<()> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        let supplied: LoweredSchedule =
            serde_json::from_str(lowered_json).map_err(|e| e.to_string())?;
        let expected = lower(&program, &supplied.schedule_map)?;
        if supplied != expected {
            return Err(
                "lowered instructions differ from the original program and fixed schedule".into(),
            );
        }
        Ok(())
    };
    validate().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lower_region_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(validate_lowered_region_schedule, module)?)?;
    Ok(())
}

/// Compile a single-valued map to a guarded coordinate accessor. Input axes
/// become named parameters, so execution needs no ISL context or string parser.
pub(super) fn accessor(
    mut map: isl_rs::Map,
    ctx: &Context,
) -> Result<(Expression, Vec<Expression>)> {
    use isl_rs::{DimType, Set};
    let rank = u32::try_from(isl(map.dim(DimType::In))?).map_err(|e| e.to_string())?;
    for i in 0..rank {
        map = isl(map.set_dim_name(DimType::In, i, &format!("p{i}")))?;
    }
    map = isl(map.move_dims(DimType::Param, 0, DimType::In, 0, rank))?;
    map = isl(map.reset_tuple_id(DimType::In))?;
    map = isl(map.set_tuple_name(DimType::Out, "target"))?;
    let range = isl(map.range())?;
    let mut domain = isl(isl(range.copy())?.params())?;
    // Never build under the case domain: doing so would erase its guard.
    let mut universe = isl(Set::universe(isl(domain.get_space())?))?;
    universe.should_free_on_drop = false;
    let build = Owned::new(
        unsafe { isl_ast_build_from_context(universe.ptr) },
        isl_ast_build_free,
    )?;
    domain.should_free_on_drop = false;
    let condition = expression(
        unsafe { isl_ast_build_expr_from_set(build.pointer, domain.ptr) },
        ctx,
    )?;
    let mut pma = isl(range.as_pw_multi_aff())?;
    pma.should_free_on_drop = false;
    let call = Owned::new(
        unsafe { isl_ast_build_call_from_pw_multi_aff(build.pointer, pma.ptr) },
        isl_ast_expr_free,
    )?;
    if unsafe { isl_ast_expr_get_type(call.pointer) } != 0
        || unsafe { isl_ast_expr_get_op_type(call.pointer) } != 22
    {
        return Err("value accessor must lower to a coordinate call".into());
    }
    let count = unsafe { isl_ast_expr_get_op_n_arg(call.pointer) };
    if count < 1 {
        return Err("value accessor has no target".into());
    }
    let coordinates = (1..count)
        .map(|i| expression(unsafe { isl_ast_expr_get_op_arg(call.pointer, i) }, ctx))
        .collect::<Result<Vec<_>>>()?;
    Ok((condition, coordinates))
}

unsafe extern "C" {
    fn isl_ast_build_from_context(set: usize) -> usize;
    fn isl_ast_build_expr_from_set(build: usize, set: usize) -> usize;
    fn isl_ast_build_call_from_pw_multi_aff(build: usize, pma: usize) -> usize;
}

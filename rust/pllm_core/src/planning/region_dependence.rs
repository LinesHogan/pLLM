//! Symbolic value provenance and resource conflicts derived from structured SSA.
//! Loop instances and bank elements remain integer sets, never enumerated nodes.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::CStr;

use isl_rs::{Context, DimType, Map, Options, Schedule, ScheduleConstraints, Set, UnionMap, UnionSet, Val};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::region_program::{
    self, AccessMode, AffineIndex, Extent, Operation, Program, Region,
    ScalarKind, ScalarType, Type, Value,
};
use super::semantic_graph::{canonical_ascii_json, digest};

pub(super) type Result<T> = std::result::Result<T, String>;
pub(super) fn isl<T>(value: std::result::Result<T, isl_rs::LibISLError>) -> Result<T> {
    value.map_err(|error| format!("ISL region analysis: {error:?}"))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ValueSite {
    pub(super) id: usize,
    pub(super) path: Vec<u32>,
    pub(super) name: String,
    #[serde(rename = "type")]
    pub(super) type_: Type,
    pub(super) extents: Vec<Extent>,
    pub(super) bank_shape: Vec<Extent>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Statement {
    pub(super) id: usize,
    pub(super) path: Vec<u32>,
    pub(super) extents: Vec<Extent>,
    pub(super) inputs: Vec<usize>,
    pub(super) outputs: Vec<usize>,
    pub(super) opaque: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ScheduleCandidate {
    pub source: String,
    pub schedule_map: String,
    pub schedule_tree: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ResourceDependencies {
    pub reads: String,
    pub writes: String,
    raw: String,
    war: String,
    waw: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Analysis {
    pub(super) program_digest: String,
    pub(super) values: Vec<ValueSite>,
    pub(super) statements: Vec<Statement>,
    pub(super) domain: String,
    aliases: String,
    pub(super) value_sources: String,
    pub(super) arguments: Vec<usize>,
    pub(super) results: Vec<usize>,
    pub(super) scalar_values: String,
    pub(super) value_consumers: String,
    data_dependencies: String,
    effect_dependencies: String,
    pub resource_dependencies: ResourceDependencies,
    pub(super) dependencies: String,
    pub reference_schedule: String,
    pub candidates: Vec<ScheduleCandidate>,
    pub(super) exact_provenance: bool,
    pub diagnostics: Vec<String>,
}

#[derive(Clone)]
struct Affine {
    constant: i64,
    coefficients: Vec<i64>,
    parameters: BTreeMap<String, i64>,
}

impl Affine {
    fn constant(value: i64) -> Self {
        Self { constant: value, coefficients: vec![], parameters: BTreeMap::new() }
    }

    fn is_constant(&self) -> bool {
        self.coefficients.iter().chain(self.parameters.values()).all(|&a| a == 0)
    }

    fn from_access(index: &AffineIndex, inputs: &[Binding], extents: &[Extent], program: &Program) -> Option<Self> {
        let mut result = Self::constant(index.constant);
        for term in &index.terms {
            let value = inputs.get(term.argument)?.affine.as_ref()?;
            let scale = Self::constant(term.coefficient);
            let scaled = scale.combine(value, &ScalarKind::Mul, extents, program)?;
            result = result.combine(&scaled, &ScalarKind::Add, extents, program)?;
        }
        Some(result)
    }

    fn text(&self) -> String {
        let mut result = self.constant.to_string();
        for (i, &a) in self.coefficients.iter().enumerate() {
            if a != 0 {
                result.push_str(&format!("{a:+}*i{i}"));
            }
        }
        for (name, &a) in &self.parameters {
            if a != 0 {
                result.push_str(&format!("{a:+}*__size_{name}"));
            }
        }
        result
    }

    fn bounds(&self, extents: &[Extent], program: &Program) -> Option<(i128, i128)> {
        let (mut low, mut high) = (i128::from(self.constant), i128::from(self.constant));
        for (i, &a) in self.coefficients.iter().enumerate() {
            let d = i128::from(extents.get(i)?.maximum(program))
                .saturating_sub(1)
                .max(0);
            let delta = i128::from(a).checked_mul(d)?;
            low = low.checked_add(delta.min(0))?;
            high = high.checked_add(delta.max(0))?;
        }
        for (name, &a) in &self.parameters {
            let parameter = program.parameters.iter().find(|p| &p.name == name)?;
            let first = i128::from(a).checked_mul(i128::from(parameter.minimum))?;
            let last = i128::from(a).checked_mul(i128::from(parameter.maximum))?;
            low = low.checked_add(first.min(last))?;
            high = high.checked_add(first.max(last))?;
        }
        Some((low, high))
    }

    fn combine(
        &self,
        other: &Self,
        kind: &ScalarKind,
        extents: &[Extent],
        program: &Program,
    ) -> Option<Self> {
        let width = self.coefficients.len().max(other.coefficients.len());
        let mut left = self.coefficients.clone();
        left.resize(width, 0);
        let mut right = other.coefficients.clone();
        right.resize(width, 0);
        let constant = match kind {
            ScalarKind::Add => self.constant.checked_add(other.constant)?,
            ScalarKind::Sub => self.constant.checked_sub(other.constant)?,
            ScalarKind::Mul if self.is_constant() || other.is_constant() =>
                self.constant.checked_mul(other.constant)?,
            _ => return None,
        };
        // A parameter is invariant during execution, but is not a constant
        // coefficient in a parametric affine map. Products need a literal side.
        let coefficient = |a: i64, b: i64| match kind {
            ScalarKind::Add => a.checked_add(b),
            ScalarKind::Sub => a.checked_sub(b),
            ScalarKind::Mul if self.is_constant() => b.checked_mul(self.constant),
            ScalarKind::Mul => a.checked_mul(other.constant),
            _ => None,
        };
        let coefficients = left.iter().zip(&right)
            .map(|(&a, &b)| coefficient(a, b)).collect::<Option<Vec<_>>>()?;
        let names = self.parameters.keys().chain(other.parameters.keys()).collect::<BTreeSet<_>>();
        let mut parameters = BTreeMap::new();
        for name in names {
            let value = coefficient(
                self.parameters.get(name).copied().unwrap_or(0),
                other.parameters.get(name).copied().unwrap_or(0),
            )?;
            if value != 0 {
                parameters.insert(name.clone(), value);
            }
        }
        let result = Self {
            constant,
            coefficients,
            parameters,
        };
        let (low, high) = result.bounds(extents, program)?;
        (low >= i128::from(i64::MIN) && high <= i128::from(i64::MAX)).then_some(result)
    }
}

#[derive(Clone)]
struct Binding {
    id: usize,
    affine: Option<Affine>,
}

#[derive(Clone, Debug, Serialize)]
struct ScheduleInstance {
    statement_id: usize,
    coordinates: Vec<u64>,
    time: Vec<i64>,
}

#[derive(Default)]
struct Builder {
    // During branch analysis, project leaf effects onto the original atomic
    // control statement. Internal loop coordinates stay existentially bound.
    effect_target: Option<String>,
    values: Vec<ValueSite>,
    statements: Vec<Statement>,
    aliases: Vec<String>,
    seeds: Vec<String>,
    scalar_values: Vec<String>,
    uses: Vec<String>,
    reads: Vec<String>,
    writes: Vec<String>,
    times: Vec<Vec<String>>,
    diagnostics: Vec<String>,
}

fn coordinates(prefix: &str, count: usize) -> Vec<String> {
    (0..count).map(|i| format!("{prefix}{i}")).collect()
}
fn tuple(name: &str, coordinates: &[String]) -> String {
    format!("{name}[{}]", coordinates.join(","))
}
fn relation(from: String, to: String, guard: &str) -> String {
    // ISL tuple syntax cannot introduce i in a first occurrence such as i-1.
    // Give both sides fresh dimensions and bind the shared logical coordinates
    // existentially. This also handles loop-entry/final-iteration projections.
    fn side(text: &str, prefix: &str) -> (String, Vec<String>) {
        let (name, dimensions) = text.split_once('[').expect("generated tuple");
        let dimensions = dimensions.trim_end_matches(']');
        let terms = if dimensions.is_empty() {
            vec![]
        } else {
            dimensions.split(',').collect::<Vec<_>>()
        };
        let variables = coordinates(prefix, terms.len());
        let equations = variables
            .iter()
            .zip(terms)
            .map(|(v, t)| format!("{v}=({t})"))
            .collect();
        (tuple(name, &variables), equations)
    }
    let original = format!("{from} {to} {guard}");
    let variables = original
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|s| {
            s.len() > 1
                && matches!(s.as_bytes()[0], b'i' | b'b' | b'r')
                && s.as_bytes()[1..].iter().all(u8::is_ascii_digit)
        })
        .collect::<BTreeSet<_>>();
    let (source, mut equations) = side(&from, "s");
    let (target, right) = side(&to, "t");
    equations.extend(right);
    equations.push(guard.into());
    let condition = equations.join(" and ");
    if variables.is_empty() {
        format!("{source} -> {target} : {condition}")
    } else {
        format!(
            "{source} -> {target} : exists ({} : {condition})",
            variables.into_iter().collect::<Vec<_>>().join(",")
        )
    }
}
pub(super) fn parameter_prefix(program: &Program) -> String {
    if program.parameters.is_empty() {
        String::new()
    } else {
        format!(
            "[{}] -> ",
            program
                .parameters
                .iter()
                .map(|p| format!("__size_{}", p.name))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}
fn parameter_context(program: &Program, ctx: &Context) -> Result<Set> {
    let mut constraints = program
        .parameters
        .iter()
        .map(|p| format!("{} <= __size_{} <= {}", p.minimum, p.name, p.maximum))
        .collect::<Vec<_>>();
    for constraint in &program.constraints {
        let expression = constraint.terms.iter().fold("0".to_owned(), |text, term|
            format!("{text}{:+}*__size_{}", term.coefficient, term.parameter));
        constraints.push(format!("{expression} <= {}", constraint.upper));
    }
    isl(Set::read_from_str(
        ctx,
        &format!(
            "{}{{ : {} }}",
            parameter_prefix(program),
            if constraints.is_empty() {
                "1=1".into()
            } else {
                constraints.join(" and ")
            }
        ),
    ))
}
fn union(parts: &[String], program: &Program) -> String {
    format!("{}{{ {} }}", parameter_prefix(program), parts.join("; "))
}
fn read_map(ctx: &Context, parts: &[String], label: &str, program: &Program) -> Result<UnionMap> {
    let text = union(parts, program);
    UnionMap::read_from_str(ctx, &text)
        .map_err(|error| {
            let bad = parts.iter().find(|part| {
                UnionMap::read_from_str(ctx, &union(&[part.to_string()], program)).is_err()
            });
            format!("{label}: {error:?}; invalid relation {bad:?}")
        })
        .and_then(|map| isl(map.intersect_params(parameter_context(program, ctx)?)))
}
fn bounds(coords: &[String], extents: &[Extent]) -> String {
    let parts = coords
        .iter()
        .zip(extents)
        .map(|(c, e)| format!("0 <= {c} < {}", e.text()))
        .collect::<Vec<_>>();
    if parts.is_empty() {
        "1=1".into()
    } else {
        parts.join(" and ")
    }
}
fn prove_index(
    affine: &Affine,
    extents: &[Extent],
    size: &Extent,
    program: &Program,
) -> Result<bool> {
    prove_no_instance(&format!("{} < 0 or {} >= {}", affine.text(), affine.text(), size.text()), extents, program)
}

fn prove_no_instance(condition: &str, extents: &[Extent], program: &Program) -> Result<bool> {
    let ctx = context()?;
    let coords = coordinates("i", extents.len());
    let text = format!(
        "{} : {} and ({condition})",
        tuple("", &coords),
        bounds(&coords, extents),
    );
    let invalid = isl(UnionSet::read_from_str(&ctx, &union(&[text], program)))?;
    let invalid = isl(invalid.intersect_params(parameter_context(program, &ctx)?))?;
    isl(invalid.is_empty())
}

fn shape(type_: &Type) -> Vec<Extent> {
    match type_ {
        Type::Bank { element, size } => {
            let mut result = vec![size.clone()];
            result.extend(shape(element));
            result
        }
        _ => vec![],
    }
}

impl Builder {
    fn branch_summary(
        program: &Program,
        region: &Region,
        extents: &[Extent],
        inputs: &[Binding],
        target: String,
    ) -> Result<Self> {
        // Analyze the original branch with the caller's affine facts. This
        // temporary builder derives properties only: its statements never
        // become an additional execution path or escape the conditional.
        let mut branch = Self { effect_target: Some(target), ..Self::default() };
        let arguments = region.block.arguments.iter().zip(inputs)
            .map(|(value, input)| branch.define(value, &[], extents, input.affine.clone()))
            .collect();
        branch.region(program, region, &[], extents, &[], arguments)?;
        Ok(branch)
    }

    fn resource_relation(&self, point: &str, resource: &str, indices: &[String], guard: &str, extents: &[Extent]) -> String {
        if let Some(target) = &self.effect_target {
            let coords = coordinates("i", extents.len());
            let guard = format!("{} and ({guard})", bounds(&coords, extents));
            // relation() binds coordinates appearing only in this guard too;
            // their loop bounds must survive projection onto the parent point.
            relation(target.clone(), tuple(resource, indices), &guard)
        } else {
            relation(point.to_owned(), tuple(resource, indices), guard)
        }
    }

    fn define(
        &mut self,
        value: &Value,
        path: &[u32],
        extents: &[Extent],
        affine: Option<Affine>,
    ) -> Binding {
        let id = self.values.len();
        self.values.push(ValueSite {
            id,
            path: path.to_vec(),
            name: value.name.clone(),
            type_: value.type_.clone(),
            extents: extents.to_vec(),
            bank_shape: shape(&value.type_),
        });
        if let Some(a) = &affine {
            self.scalar_values.push(relation(
                self.ordinary_point(id),
                tuple("", &[a.text()]),
                "1=1",
            ));
        }
        Binding { id, affine }
    }

    fn point(&self, id: usize, iterations: &[String], elements: &[String]) -> String {
        let scope = &iterations[..self.values[id].extents.len()];
        tuple(&format!("V{id}"), &[scope, elements].concat())
    }

    fn ordinary_point(&self, id: usize) -> String {
        let value = &self.values[id];
        self.point(
            id,
            &coordinates("i", value.extents.len()),
            &coordinates("b", value.bank_shape.len()),
        )
    }

    fn copy(
        &mut self,
        source: usize,
        source_iterations: &[String],
        target: usize,
        target_iterations: &[String],
        guard: &str,
    ) {
        let elements = coordinates("b", self.values[source].bank_shape.len());
        self.aliases.push(relation(
            self.point(source, source_iterations, &elements),
            self.point(target, target_iterations, &elements),
            guard,
        ));
    }

    fn statement(
        &mut self,
        program: &Program,
        op: &Operation,
        path: &[u32],
        extents: &[Extent],
        time: &[String],
        inputs: &[Binding],
        outputs: &[Binding],
        mut opaque: bool,
    ) -> Result<()> {
        let id = self.statements.len();
        let point = tuple(&format!("S{id}"), &coordinates("i", extents.len()));
        for input in inputs {
            self.uses.push(relation(self.ordinary_point(input.id), point.clone(), "1=1"));
        }
        for output in outputs {
            self.seeds.push(relation(point.clone(), self.ordinary_point(output.id), "1=1"));
        }
        if let Operation::If { then_region, else_region, .. } = op {
            let target = self.effect_target.clone().unwrap_or(point.clone());
            for region in [then_region, else_region] {
                let branch = Self::branch_summary(program, region, extents, &inputs[1..], target.clone())?;
                opaque |= branch.statements.iter().any(|statement| statement.opaque);
                self.reads.extend(branch.reads);
                self.writes.extend(branch.writes);
            }
            if opaque {
                self.diagnostics.push(format!("operation {path:?}: conditional branch contains unresolved effects, checked operations or a dynamic loop"));
            }
        }
        let declared_accesses = match op {
            Operation::Call { function, .. } => program.functions.iter().find(|f| &f.name == function)
                .ok_or("unknown function")?.accesses.as_slice(),
            _ => &[],
        };
        let accesses = match op {
            Operation::If { .. } => vec![], // The branch summaries above retain indexed effects.
            Operation::Snapshot { resource, .. } => vec![(
                resource.as_str(),
                true,
                false,
                Some(&inputs[..program.resources.iter().find(|r| r.name() == resource)
                    .ok_or("unknown effect resource")?.domain().len()]),
            )],
            Operation::Publish { resource, .. } => {
                let original = program
                    .resources
                    .iter()
                    .find(|r| r.name() == resource)
                    .ok_or("unknown effect resource")?;
                let payload = original
                    .payload()
                    .ok_or("external resource cannot be published")?
                    .len();
                vec![(
                    resource.as_str(),
                    false,
                    true,
                    Some(&inputs[payload..payload + original.domain().len()]),
                )]
            }
            _ => resource_effects(program, op)
                .into_iter()
                .filter(|name| !declared_accesses.iter().any(|a| a.resource == *name))
                .map(|r| (r, true, true, None))
                .collect(),
        };
        for (name, read, write, indices) in accesses {
            let ordinal = program
                .resources
                .iter()
                .position(|r| r.name() == name)
                .ok_or("unknown effect resource")?;
            let resource = &program.resources[ordinal];
            if matches!(resource, super::region_program::Resource::External { .. }) {
                // No backend alias proof is imported yet. Do not infer
                // independence from two different external resource names.
                opaque = true;
                self.diagnostics.push(format!(
                    "operation {path:?}: external owner remains an opaque boundary"
                ));
            }
            let mut precise = None;
            if let Some(indices) = indices {
                let mut coordinates = Some(Vec::new());
                for (value, &size) in indices.iter().zip(resource.domain()) {
                    let proven = if let Some(affine) = &value.affine {
                        let cheap = affine.bounds(extents, program)
                            .is_some_and(|(lo, hi)| lo >= 0 && hi < i128::from(size));
                        if cheap || prove_index(affine, extents, &Extent::Fixed(size), program)? {
                            Some(affine.text())
                        } else { None }
                    } else { None };
                    match (&mut coordinates, proven) {
                        (Some(result), Some(coordinate)) => result.push(coordinate),
                        _ => { coordinates = None; break; }
                    }
                }
                precise = coordinates;
            }
            if indices.is_some() && precise.is_none() {
                opaque = true;
                self.diagnostics.push(format!(
                    "operation {path:?}: unproven resource index remains an opaque boundary"
                ));
            }
            let indices = coordinates("r", resource.domain().len());
            let guard = if let Some(expressions) = precise {
                if expressions.is_empty() { "1=1".into() } else {
                    indices.iter().zip(expressions).map(|(index, expression)| format!("{index} = {expression}"))
                        .collect::<Vec<_>>().join(" and ")
                }
            } else {
                bounds(&indices, &resource.domain().iter().map(|&n| Extent::Fixed(n)).collect::<Vec<_>>())
            };
            let access = self.resource_relation(&point, &format!("R{ordinal}"), &indices, &guard, extents);
            if read {
                self.reads.push(access.clone());
            }
            if write {
                self.writes.push(access);
            }
        }
        for access in declared_accesses {
            let ordinal = program.resources.iter().position(|r| r.name() == access.resource)
                .ok_or("unknown indexed resource")?;
            let resource = &program.resources[ordinal];
            let indices = coordinates("r", resource.domain().len());
            let mut precise = Vec::new();
            for (axis, (range, size)) in access.ranges.iter().zip(resource.domain()).enumerate() {
                let pair = Affine::from_access(&range.lower, inputs, extents, program)
                    .zip(Affine::from_access(&range.upper, inputs, extents, program));
                let condition = if let Some((lower, upper)) = pair {
                    let cheap = lower.bounds(extents, program).is_some_and(|(lo, _)| lo >= 0)
                        && upper.bounds(extents, program).is_some_and(|(_, hi)| hi <= i128::from(*size))
                        && upper.combine(&lower, &ScalarKind::Sub, extents, program)
                            .and_then(|a| a.bounds(extents, program)).is_some_and(|(lo, _)| lo >= 0);
                    let invalid = format!("{} < 0 or {} > {size} or {} > {}",
                        lower.text(), upper.text(), lower.text(), upper.text());
                    if cheap || prove_no_instance(&invalid, extents, program)? {
                        Some(format!("{} <= r{axis} < {}", lower.text(), upper.text()))
                    } else { None }
                } else { None };
                precise.push(condition);
            }
            if precise.iter().any(Option::is_none) || matches!(resource, super::region_program::Resource::External { .. }) {
                opaque = true;
                self.diagnostics.push(format!("operation {path:?}: indexed access needs proven bounds and original alias separation"));
            }
            let conditions = precise.into_iter().zip(resource.domain()).enumerate()
                .map(|(axis, (condition, size))| condition.unwrap_or_else(|| format!("0 <= r{axis} < {size}")))
                .collect::<Vec<_>>();
            let guard = if conditions.is_empty() { "1=1".into() } else { conditions.join(" and ") };
            let relation = self.resource_relation(&point, &format!("R{ordinal}"), &indices, &guard, extents);
            match access.mode {
                AccessMode::Read => self.reads.push(relation),
                AccessMode::Write => self.writes.push(relation),
            }
        }
        self.statements.push(Statement {
            id,
            path: path.to_vec(),
            extents: extents.to_vec(),
            inputs: inputs.iter().map(|v| v.id).collect(),
            outputs: outputs.iter().map(|v| v.id).collect(),
            opaque,
        });
        self.times.push(time.to_vec());
        Ok(())
    }

    fn region(
        &mut self,
        program: &Program,
        region: &Region,
        path: &[u32],
        extents: &[Extent],
        time: &[String],
        arguments: Vec<Binding>,
    ) -> Result<Vec<Binding>> {
        let block = &region.block;
        let mut env: BTreeMap<_, _> = block
            .arguments
            .iter()
            .zip(arguments)
            .map(|(v, b)| (v.name.as_str(), b))
            .collect();
        let iterations = coordinates("i", extents.len());
        for (position, op) in block.operations.iter().enumerate() {
            let mut op_path = path.to_vec();
            op_path.push(position as u32);
            let mut op_time = time.to_vec();
            op_time.push(position.to_string());
            let (names, results) = op.ports();
            let inputs = names
                .iter()
                .map(|name| {
                    env.get(name.as_str())
                        .cloned()
                        .ok_or("undefined value".to_owned())
                })
                .collect::<Result<Vec<_>>>()?;
            let mut outputs = results
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if let Operation::Scan { body, invariants, .. } = op {
                        // An unchanged carry is the incoming definition,
                        // including when the loop executes zero iterations.
                        if i < inputs.len() - invariants && body.block.yields[i] == body.block.arguments[i + 1].name {
                            return inputs[i].clone();
                        }
                    }
                    self.define(v, &op_path, extents, None)
                })
                .collect::<Vec<_>>();
            match op {
                Operation::Constant { .. } => {
                    // Affine index values are declared by Coordinate; literals
                    // are ordinary primitive statements.
                    self.statement(program, op, &op_path, extents, &op_time, &inputs, &outputs, false)?;
                }
                Operation::Parameter { parameter, .. } => {
                    let mut value = Affine::constant(0);
                    value.parameters.insert(parameter.clone(), 1);
                    outputs[0].affine = Some(value);
                }
                Operation::Coordinate { affine, .. } => {
                    // Direct affine declaration: no literal/scalar recovery.
                    // Non-affine or out-of-range operands are a hard error.
                    let value = Affine::from_access(affine, &inputs, extents, program)
                        .ok_or("coordinate operands must stay affine within the checked index range")?;
                    outputs[0].affine = Some(value);
                }
                Operation::Scalar { .. } => {
                    // Checked integer arithmetic stays an opaque boundary;
                    // affine forms must be declared with Coordinate.
                    let may_trap = matches!(
                        self.values[outputs[0].id].type_,
                        Type::Scalar {
                            name: ScalarType::Index | ScalarType::Int
                        }
                    );
                    if may_trap {
                        self.diagnostics.push(format!("operation {op_path:?}: checked integer arithmetic remains an opaque boundary"));
                    }
                    self.statement(
                        program, op, &op_path, extents, &op_time, &inputs, &outputs, may_trap,
                    )?;
                }
                Operation::Scan { count, body, .. } | Operation::Map { count, body, .. } => {
                    let carry = if let Operation::Scan { invariants, .. } = op { inputs.len() - invariants } else { 0 };
                    let mut nested_extents = extents.to_vec();
                    nested_extents.push(count.clone());
                    let mut body_path = op_path.clone();
                    body_path.push(0);
                    let mut body_time = op_time.clone();
                    body_time.push(format!("i{}", extents.len()));
                    let mut coefficient = vec![0; nested_extents.len()];
                    coefficient[extents.len()] = 1;
                    let mut args = vec![self.define(
                        &body.block.arguments[0],
                        &body_path,
                        &nested_extents,
                        Some(Affine {
                            constant: 0,
                            coefficients: coefficient,
                            parameters: BTreeMap::new(),
                        }),
                    )];
                    for (i, input) in inputs.iter().enumerate() {
                        let parameter = &body.block.arguments[i + 1];
                        let invariant = i >= carry || body.block.yields[i] == parameter.name;
                        args.push(if invariant { input.clone() } else {
                            self.define(parameter, &body_path, &nested_extents, None)
                        });
                    }
                    let yielded = self.region(
                        program,
                        body,
                        &body_path,
                        &nested_extents,
                        &body_time,
                        args.clone(),
                    )?;
                    let nested = coordinates("i", nested_extents.len());
                    for (i, input) in inputs.iter().enumerate() {
                        if i < carry && body.block.yields[i] != body.block.arguments[i + 1].name {
                            let mut first = iterations.clone();
                            first.push("0".into());
                            self.copy(input.id, &iterations, args[i + 1].id, &first, "1=1");
                            let mut previous = nested.clone();
                            previous[extents.len()] = format!("i{}-1", extents.len());
                            self.copy(
                                yielded[i].id,
                                &previous,
                                args[i + 1].id,
                                &nested,
                                &format!("i{}>=1", extents.len()),
                            );
                            self.copy(
                                input.id,
                                &iterations,
                                outputs[i].id,
                                &iterations,
                                &format!("{}=0", count.text()),
                            );
                            let mut last = iterations.clone();
                            last.push(format!("{}-1", count.text()));
                            self.copy(
                                yielded[i].id,
                                &last,
                                outputs[i].id,
                                &iterations,
                                &format!("{}>0", count.text()),
                            );
                        }
                    }
                    {
                        for (i, value) in yielded.iter().enumerate().skip(carry) {
                            let elements = coordinates("b", self.values[value.id].bank_shape.len());
                            let mut target_elements = vec![format!("i{}", extents.len())];
                            target_elements.extend(elements.clone());
                            self.aliases.push(relation(
                                self.point(value.id, &nested, &elements),
                                self.point(outputs[i].id, &iterations, &target_elements),
                                "1=1",
                            ));
                        }
                    }
                }
                Operation::Select { .. } | Operation::Update { .. } => {
                    let bank_shape = self.values[inputs[0].id].bank_shape.clone();
                    let index = match &inputs[1].affine {
                        Some(affine) if prove_index(affine, extents, &bank_shape[0], program)? => {
                            Some(affine.text())
                        }
                        _ => None,
                    };
                    if let Some(index) = index {
                        let tail = coordinates("b", bank_shape.len() - 1);
                        let mut selected = vec![index.clone()];
                        selected.extend(tail.clone());
                        if matches!(op, Operation::Select { .. }) {
                            self.aliases.push(relation(
                                self.point(inputs[0].id, &iterations, &selected),
                                self.point(outputs[0].id, &iterations, &tail),
                                "1=1",
                            ));
                        } else {
                            self.copy(
                                inputs[0].id,
                                &iterations,
                                outputs[0].id,
                                &iterations,
                                &format!("b0 != ({index})"),
                            );
                            self.aliases.push(relation(
                                self.point(inputs[2].id, &iterations, &tail),
                                self.point(outputs[0].id, &iterations, &selected),
                                "1=1",
                            ));
                        }
                    } else {
                        self.diagnostics.push(format!("operation {op_path:?}: non-affine or unproven index remains an opaque boundary"));
                        self.statement(
                            program, op, &op_path, extents, &op_time, &inputs, &outputs, true,
                        )?;
                    }
                }
                _ => {
                    let opaque = matches!(op, Operation::While { .. });
                    if opaque {
                        self.diagnostics.push(format!("operation {op_path:?}: dynamic control contains unresolved effects, checked operations or a dynamic loop"));
                    }
                    self.statement(
                        program, op, &op_path, extents, &op_time, &inputs, &outputs, opaque,
                    )?;
                }
            }
            for (name, output) in results.iter().zip(outputs) {
                if let Some(a) = &output.affine {
                    // Aliases of an existing definition already have its
                    // scalar relation; only newly defined values add one.
                    if self.values[output.id].path == op_path {
                        self.scalar_values.push(relation(
                            self.ordinary_point(output.id),
                            tuple("", &[a.text()]),
                            "1=1",
                        ));
                    }
                }
                env.insert(name.name.as_str(), output);
            }
        }
        block
            .yields
            .iter()
            .map(|name| {
                env.get(name.as_str())
                    .cloned()
                    .ok_or("undefined region result".into())
            })
            .collect()
    }
}

fn resource_effects<'a>(program: &'a Program, operation: &'a Operation) -> BTreeSet<&'a str> {
    let mut result = BTreeSet::new();
    match operation {
        Operation::Snapshot { resource, .. } | Operation::Publish { resource, .. } => {
            result.insert(resource.as_str());
        }
        Operation::Call { function, .. } => {
            if let Some(f) = program.functions.iter().find(|f| f.name == *function) {
                result.extend(f.resources.iter().map(String::as_str));
            }
        }
        Operation::Scan { body, .. } | Operation::Map { body, .. } => {
            for op in &body.block.operations {
                result.extend(resource_effects(program, op));
            }
        }
        Operation::If {
            then_region,
            else_region,
            ..
        } => {
            for op in then_region
                .block
                .operations
                .iter()
                .chain(&else_region.block.operations)
            {
                result.extend(resource_effects(program, op));
            }
        }
        Operation::While {
            condition, body, ..
        } => {
            for op in condition
                .block
                .operations
                .iter()
                .chain(&body.block.operations)
            {
                result.extend(resource_effects(program, op));
            }
        }
        _ => {}
    }
    result
}

// isl-rs does not wrap transitive_closure, and its to_str wrappers leak owned
// C allocations. Keep both gaps in this narrow, ownership-audited FFI boundary.
unsafe extern "C" {
    fn isl_union_map_transitive_closure(map: usize, exact: *mut i32) -> usize;
    fn isl_union_map_to_str(map: usize) -> *mut std::ffi::c_char;
    fn isl_union_set_to_str(set: usize) -> *mut std::ffi::c_char;
    fn isl_schedule_to_str(schedule: usize) -> *mut std::ffi::c_char;
    fn free(pointer: *mut std::ffi::c_void);
}
fn owned_string(pointer: *mut std::ffi::c_char) -> Result<String> {
    if pointer.is_null() {
        return Err("ISL serialization failed".into());
    }
    unsafe {
        let result = CStr::from_ptr(pointer).to_string_lossy().into_owned();
        free(pointer.cast());
        Ok(result)
    }
}

pub(super) fn schedule_text(schedule: &isl_rs::Schedule) -> Result<String> {
    owned_string(unsafe { isl_schedule_to_str(schedule.ptr) })
}
pub(super) fn map_text(map: &UnionMap) -> Result<String> {
    let text = owned_string(unsafe { isl_union_map_to_str(map.ptr) })?;
    // Union members are printed in ISL hash-table order, which can change
    // between contexts even when every relation is identical. Keep the ISL
    // spelling of each member, but serialize members in a stable order before
    // signing witnesses. Quoted identifiers may contain punctuation.
    let mut quoted = false;
    let mut escaped = false;
    let mut start = None;
    let mut member = 0;
    let mut members = Vec::new();
    for (index, ch) in text.char_indices() {
        if quoted {
            if escaped { escaped = false; }
            else if ch == '\\' { escaped = true; }
            else if ch == '"' { quoted = false; }
            continue;
        }
        match ch {
            '"' => quoted = true,
            '{' => { start = Some(index); member = index + 1; }
            ';' | '}' if start.is_some() => {
                members.push(text[member..index].trim());
                member = index + 1;
                if ch == '}' {
                    members.sort_unstable();
                    return Ok(format!("{}{{ {} }}{}", &text[..start.unwrap()],
                        members.join("; "), &text[index + 1..]));
                }
            }
            _ => {},
        }
    }
    Err("ISL union-map serialization has no complete body".into())
}

pub(super) fn read_schedule(ctx: &Context, text: &str) -> Result<UnionMap> {
    // Parameter bounds implied by nonempty iteration domains must normalize
    // before deriving instructions. Otherwise a print/parse round trip can
    // remove redundant constraints and change the signed accessor syntax.
    normalize_schedule(isl(UnionMap::read_from_str(ctx, text))?)
}

fn normalize_schedule(map: UnionMap) -> Result<UnionMap> {
    isl(isl(isl(map.detect_equalities())?.coalesce())?.remove_redundancies())
}

pub(super) fn canonical_schedule(text: &str) -> Result<String> {
    let ctx = context()?;
    map_text(&read_schedule(&ctx, text)?)
}
pub(super) fn set_text(set: &UnionSet) -> Result<String> {
    owned_string(unsafe { isl_union_set_to_str(set.ptr) })
}
pub(super) fn closure(mut map: UnionMap) -> Result<(UnionMap, bool)> {
    let mut exact = 0;
    map.should_free_on_drop = false;
    let pointer = unsafe { isl_union_map_transitive_closure(map.ptr, &mut exact) };
    if pointer == 0 {
        return Err("ISL value provenance closure failed".into());
    }
    Ok((
        UnionMap {
            ptr: pointer,
            should_free_on_drop: true,
        },
        exact == 1,
    ))
}
pub(super) fn context() -> Result<Context> {
    let ctx = Context::alloc();
    ctx.set_max_operations(20_000_000);
    isl(Options::set_on_error(&ctx, 1))?;
    Ok(ctx)
}

pub(super) fn legal(domain: &UnionSet, dependencies: &UnionMap, schedule: &UnionMap) -> Result<()> {
    let domain_space = isl(domain.get_space())?;
    let schedule_space = isl(schedule.get_space())?;
    let names = (0..isl(domain_space.dim(DimType::Param))?)
        .map(|i| isl(domain_space.get_dim_name(DimType::Param, i as u32)))
        .collect::<Result<BTreeSet<_>>>()?;
    for i in 0..isl(schedule_space.dim(DimType::Param))? {
        if !names.contains(isl(schedule_space.get_dim_name(DimType::Param, i as u32))?) {
            return Err("schedule uses undeclared size parameters".into());
        }
    }
    if !isl(schedule.is_single_valued())?
        || !isl(isl(isl(schedule.copy())?.domain())?.is_equal(domain))?
    {
        return Err("schedule must cover every statement instance with a single time".into());
    }
    if isl(domain.is_empty())? {
        return if isl(dependencies.is_empty())? { Ok(()) }
            else { Err("schedule violates an original value or resource dependency".into()) };
    }
    // Compare times only for real dependency pairs, rather than materializing
    // the order of every pair of statement instances. Mapping both endpoints
    // preserves an offending edge even when independent instances share time.
    // The common time space is also required for independent statements.
    let times = isl(isl(isl(schedule.copy())?.range())?.as_set())?;
    let before = isl(UnionMap::from_map(isl(Map::lex_lt(isl(times.get_space())?))?))?;
    let scheduled = isl(isl(isl(dependencies.copy())?.apply_domain(isl(schedule.copy())?))?
        .apply_range(isl(schedule.copy())?))?;
    if !isl(scheduled.is_subset(&before))? {
        // Name the offending edge, not just the failure class: map the wrongly
        // ordered time pairs back through the (single-valued) schedule and
        // report the lexicographically first original dependency edge.
        let inverse = isl(isl(schedule.copy())?.reverse())?;
        let edges = isl(isl(isl(scheduled.subtract(before))?
            .apply_domain(isl(inverse.copy())?))?
            .apply_range(inverse))?;
        let minimal = isl(edges.lexmin())?;
        return Err(format!(
            "schedule violates an original value or resource dependency; first violated edge: {}",
            map_text(&minimal)?));
    }
    Ok(())
}

/// Remove edges already implied by two original edges before solving the
/// scheduling ILP. Resource hazards and opaque fences often describe dense
/// transitive orders; feeding every such edge to proximity/coincidence makes
/// the optimization problem grow with irrelevant long-distance constraints.
///
/// The reference order makes this a finite DAG for each parameter binding.
/// Replacing a removed edge by its two-edge path preserves reachability: each
/// replacement spans a strictly shorter reference interval. This is only a
/// solver input simplification. Published dependencies and final legality
/// checks retain the complete original relation.
fn scheduling_edges(dependencies: &UnionMap) -> Result<UnionMap> {
    let indirect = isl(isl(dependencies.copy())?.apply_range(isl(dependencies.copy())?))?;
    isl(isl(isl(dependencies.copy())?.subtract(indirect))?.coalesce())
}

/// Compare execution orders without constructing the quadratic union of all
/// statement pairs. Most candidates differ within one statement's iteration
/// domain, so check those pairs first and stop at the first difference.
/// Distinct-statement pairs in both directions also preserve simultaneous
/// instances. This is exact comparison over the full parameter domains.
pub(super) fn equivalent_order(left: &UnionMap, right: &UnionMap) -> Result<bool> {
    if isl(left.is_equal(right))? { return Ok(true); }
    fn pieces(schedule: &UnionMap) -> Result<BTreeMap<String, Map>> {
        let maps = isl(schedule.get_map_list())?;
        (0..isl(maps.n_map())?).map(|i| {
            let map = isl(maps.get_at(i))?;
            Ok((isl(map.get_tuple_name(DimType::In))?.to_owned(), map))
        }).collect()
    }
    let left = pieces(left)?;
    let right = pieces(right)?;
    if left.keys().ne(right.keys()) { return Err("schedule comparison requires the same statement domains".into()); }
    let pairs = left.values().zip(right.values()).collect::<Vec<_>>();
    let equal = |(a, b): (&Map, &Map), (c, d): (&Map, &Map)| -> Result<bool> {
        let original = isl(isl(a.copy())?.lex_lt_map(isl(c.copy())?))?;
        let candidate = isl(isl(b.copy())?.lex_lt_map(isl(d.copy())?))?;
        isl(original.is_equal(&candidate))
    };
    for &pair in &pairs {
        if !equal(pair, pair)? { return Ok(false); }
    }
    for (i, &first) in pairs.iter().enumerate() {
        for &second in &pairs[i + 1..] {
            if !equal(first, second)? || !equal(second, first)? { return Ok(false); }
        }
    }
    Ok(true)
}

pub(super) fn analyze(program: &Program, optimize: bool) -> Result<Analysis> {
    analyze_schedules(program, optimize, false)
}

fn analyze_schedules(program: &Program, optimize: bool, parallel_only: bool) -> Result<Analysis> {
    thread_local! {
        static ANALYSES: RefCell<VecDeque<(String, Analysis)>> = const { RefCell::new(VecDeque::new()) };
    }
    // Source interpretation has one cache and one derivation path. Schedule
    // search consumes its relations; it never reinterprets the source Program.
    let key = digest(&canonical_ascii_json(program)?);
    let mut analysis = super::region_cache::memoized(&ANALYSES, key, || derive_analysis(program))?;
    if optimize {
        thread_local! {
            static SCHEDULES: RefCell<VecDeque<(String, (Vec<ScheduleCandidate>, Vec<String>))>> = const { RefCell::new(VecDeque::new()) };
        }
        let key = digest(&canonical_ascii_json(&(&analysis.domain, &analysis.dependencies, parallel_only))?);
        let (candidates, diagnostics) = super::region_cache::memoized(&SCHEDULES, key, || {
            let ctx = context()?;
            let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain))?;
            let dependencies = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies))?;
            // Programs replicating one request per top-level group are
            // scheduled on a single replica and lifted order-preservingly.
            // Any structural miss or check failure falls back to the exact
            // full-domain search.
            match lifted_schedule_candidates(&ctx, &analysis.statements, &domain, &dependencies, parallel_only) {
                Ok(Some(lifted)) => Ok(lifted),
                Ok(None) => schedule_candidates(&ctx, &domain, &dependencies, parallel_only),
                Err(error) => {
                    let (candidates, mut diagnostics) =
                        schedule_candidates(&ctx, &domain, &dependencies, parallel_only)?;
                    diagnostics.push(format!(
                        "lifted schedule derivation fell back to full-domain search: {error}"
                    ));
                    Ok((candidates, diagnostics))
                },
            }
        })?;
        analysis.candidates.extend(candidates);
        analysis.diagnostics.extend(diagnostics);
    }
    Ok(analysis)
}

fn derive_analysis(program: &Program) -> Result<Analysis> {
    region_program::verify(program)?;
    let mut builder = Builder::default();
    let args: Vec<Binding> = program
        .entry
        .block
        .arguments
        .iter()
        .map(|v| builder.define(v, &[], &[], None))
        .collect();
    let arguments = args.iter().map(|b| b.id).collect();
    let results = builder
        .region(program, &program.entry, &[], &[], &[], args)?
        .iter()
        .map(|b| b.id)
        .collect();
    let ctx = context()?;
    let value_domains = builder
        .values
        .iter()
        .map(|v| {
            let coords = [
                coordinates("i", v.extents.len()),
                coordinates("b", v.bank_shape.len()),
            ]
            .concat();
            format!(
                "{} : {}",
                tuple(&format!("V{}", v.id), &coords),
                bounds(&coords, &[v.extents.clone(), v.bank_shape.clone()].concat())
            )
        })
        .collect::<Vec<_>>();
    let values = isl(isl(UnionSet::read_from_str(
        &ctx,
        &union(&value_domains, program),
    ))?
    .intersect_params(parameter_context(program, &ctx)?))?;
    let domains = builder
        .statements
        .iter()
        .map(|s| {
            let coords = coordinates("i", s.extents.len());
            format!(
                "{} : {}",
                tuple(&format!("S{}", s.id), &coords),
                bounds(&coords, &s.extents)
            )
        })
        .collect::<Vec<_>>();
    let domain = isl(
        isl(UnionSet::read_from_str(&ctx, &union(&domains, program)))?
            .intersect_params(parameter_context(program, &ctx)?),
    )?;
    let aliases = read_map(&ctx, &builder.aliases, "value aliases", program)?;
    let aliases =
        isl(isl(aliases.intersect_domain(isl(values.copy())?))?
            .intersect_range(isl(values.copy())?))?;
    let (paths, exact_provenance) = closure(isl(aliases.copy())?)?;
    let paths = isl(paths.union(isl(isl(values.copy())?.identity())?))?;
    let seeds = read_map(&ctx, &builder.seeds, "value definitions", program)?;
    let uses = read_map(&ctx, &builder.uses, "value uses", program)?;
    let produced = isl(isl(seeds.copy())?.range())?;
    let incoming = isl(isl(aliases.copy())?.range())?;
    let roots = isl(isl(values.copy())?.subtract(incoming))?;
    let leaves = isl(roots.union(produced))?;
    let sources = isl(isl(isl(paths.copy())?.intersect_domain(leaves))?.reverse())?;
    if exact_provenance
        && (!isl(sources.is_single_valued())?
            || !isl(isl(isl(sources.copy())?.domain())?.is_equal(&values))?)
    {
        return Err(
            "exact value provenance must give every value element one original source".into(),
        );
    }
    // Include only real statement domains; zero-trip loops contribute no instances.
    let consumers = isl(isl(isl(sources.copy())?.reverse())?.apply_range(isl(uses.copy())?))?;
    let consumers = isl(consumers.intersect_range(isl(domain.copy())?))?;
    let scalars = isl(
        read_map(&ctx, &builder.scalar_values, "affine scalars", program)?
            .intersect_domain(isl(values.copy())?),
    )?;
    let raw = isl(isl(seeds.apply_range(paths))?.apply_range(uses))?;
    let raw =
        isl(isl(raw.intersect_domain(isl(domain.copy())?))?.intersect_range(isl(domain.copy())?))?;
    let width = builder.times.iter().map(Vec::len).max().unwrap_or(1);
    let reference = builder
        .statements
        .iter()
        .zip(&builder.times)
        .map(|(s, t)| {
            let mut padded = t.clone();
            padded.resize(width, "0".into());
            relation(
                tuple(&format!("S{}", s.id), &coordinates("i", s.extents.len())),
                tuple("", &padded),
                "1=1",
            )
        })
        .collect::<Vec<_>>();
    let reference = isl(read_map(&ctx, &reference, "reference order", program)?
        .intersect_domain(isl(domain.copy())?))?;
    let before = isl(isl(reference.copy())?.lex_lt_union_map(isl(reference.copy())?))?;
    let reads = read_map(&ctx, &builder.reads, "resource reads", program)?;
    let writes = read_map(&ctx, &builder.writes, "resource writes", program)?;
    let raw_effect = isl(isl(writes.copy())?.apply_range(isl(isl(reads.copy())?.reverse())?))?;
    let war = isl(isl(reads.copy())?.apply_range(isl(isl(writes.copy())?.reverse())?))?;
    let waw = isl(isl(writes.copy())?.apply_range(isl(isl(writes.copy())?.reverse())?))?;
    let raw_effect = isl(raw_effect.intersect(isl(before.copy())?))?;
    let war = isl(war.intersect(isl(before.copy())?))?;
    let waw = isl(waw.intersect(isl(before.copy())?))?;
    let resource_dependencies = ResourceDependencies {
        reads: map_text(&reads)?, writes: map_text(&writes)?, raw: map_text(&raw_effect)?,
        war: map_text(&war)?, waw: map_text(&waw)?,
    };
    let mut effects =
        isl(isl(isl(raw_effect.union(war))?.union(waw))?.intersect(isl(before.copy())?))?;
    // Unresolved control effects and unchecked accesses cannot move across the
    // surrounding reference program. Proven branches remain atomic actions;
    // only the selected branch executes, even when its outer action moves.
    let opaque = builder
        .statements
        .iter()
        .filter(|s| s.opaque)
        .map(|s| tuple(&format!("S{}", s.id), &coordinates("i", s.extents.len())))
        .collect::<Vec<_>>();
    let opaque = isl(UnionSet::read_from_str(&ctx, &union(&opaque, program)))?;
    effects = isl(effects.union(isl(
        isl(before.copy())?.intersect_domain(isl(opaque.copy())?)
    )?))?;
    effects = isl(effects.union(isl(isl(before.copy())?.intersect_range(opaque))?))?;
    let dependencies = isl(isl(isl(raw.copy())?.union(isl(effects.copy())?))?.coalesce())?;
    legal(&domain, &dependencies, &reference)?;
    // Policy checks consume this same canonical reference without lowering
    // accessors just to normalize nonempty-domain parameter constraints.
    let reference_text = map_text(&normalize_schedule(reference)?)?;
    let candidates = vec![ScheduleCandidate {
        source: "reference".into(),
        schedule_map: reference_text.clone(),
        schedule_tree: String::new(),
    }];
    if !exact_provenance {
        builder.diagnostics.push("value provenance closure is conservative; extra dependencies may restrict optimization".into());
    }
    Ok(Analysis {
        program_digest: digest(&canonical_ascii_json(program)?),
        values: builder.values,
        statements: builder.statements,
        domain: set_text(&domain)?,
        aliases: map_text(&aliases)?,
        value_sources: map_text(&sources)?,
        arguments,
        results,
        scalar_values: map_text(&scalars)?,
        value_consumers: map_text(&consumers)?,
        data_dependencies: map_text(&raw)?,
        effect_dependencies: map_text(&effects)?,
        resource_dependencies,
        dependencies: map_text(&dependencies)?,
        reference_schedule: reference_text,
        candidates,
        exact_provenance,
        diagnostics: builder.diagnostics,
    })
}

// The ISL choice pass only needs invocation domains and proven dependencies.
// No model classes, source instructions, tensor math or page state enter here.
fn schedule_candidates(ctx: &Context, domain: &UnionSet, dependencies: &UnionMap, parallel_only: bool)
    -> Result<(Vec<ScheduleCandidate>, Vec<String>)>
{
    let mut candidates = Vec::new();
    let mut diagnostics = Vec::new();
    if !isl(domain.is_empty())? {
        ctx.reset_operations();
        let scheduling = match scheduling_edges(dependencies) {
            Ok(edges) => edges,
            Err(error) => {
                diagnostics.push(format!("schedule edge simplification skipped: {error}"));
                ctx.reset_operations();
                isl(dependencies.copy())?
            },
        };
        for &parallel in if parallel_only { &[1][..] } else { &[0, 1][..] } {
            ctx.reset_operations();
            let candidate = (|| {
                isl(Options::set_schedule_outer_coincidence(ctx, parallel))?;
                let constraints = isl(ScheduleConstraints::on_domain(isl(domain.copy())?))?;
                let constraints = isl(constraints.set_validity(isl(scheduling.copy())?))?;
                let constraints = isl(constraints.set_proximity(isl(scheduling.copy())?))?;
                let schedule =
                    isl(isl(constraints.set_coincidence(isl(scheduling.copy())?))?
                        .compute_schedule())?;
                let map = isl(isl(schedule.get_map())?.intersect_domain(isl(domain.copy())?))?;
                legal(domain, dependencies, &map)?;
                Ok::<_, String>(ScheduleCandidate {
                    source: if parallel == 0 {
                        "isl-locality"
                    } else {
                        "isl-parallel"
                    }
                    .into(),
                    schedule_map: map_text(&map)?,
                    schedule_tree: schedule_text(&schedule)?,
                })
            })();
            match candidate {
                Ok(candidate) => candidates.push(candidate),
                Err(error) => diagnostics
                    .push(format!("optimization candidate discarded: {error}")),
            }
        }
    }
    Ok((candidates, diagnostics))
}

// Split a union map/set text into per-piece strings at "; " boundaries. ISL
// prints disjuncts of one map as separate pieces in union text; constraint
// expressions themselves never contain the separator. Text fragility is not a
// soundness risk here: the lifted result is read back and checked by the exact
// full-domain legal(), and any failure falls back to the full-domain search.
fn text_pieces(text: &str) -> Result<Vec<String>> {
    let open = text.find('{').ok_or("ISL text has no opening brace")?;
    let close = text.rfind('}').ok_or("ISL text has no closing brace")?;
    Ok(text[open + 1..close]
        .split("; ")
        .map(str::trim)
        .filter(|piece| !piece.is_empty())
        .map(str::to_string)
        .collect())
}

// Statement groups that replicate one request's structure: same program-tree
// path tails and extents in each group, differing only in the request's size
// parameter suffix. Returns the groups ordered by request plus, per request,
// the parameter renames to apply to request 0's text.
fn replication_groups(
    statements: &[Statement],
    parameters: &[String],
) -> Option<(Vec<Vec<usize>>, Vec<Vec<(String, String)>>)> {
    let mut by_parent: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for statement in statements {
        let (&parent, _) = statement.path.split_first()?;
        by_parent.entry(parent).or_default().push(statement.id);
    }
    if by_parent.len() < 2 {
        return None;
    }
    let mut groups = by_parent.into_values().collect::<Vec<_>>();
    groups.sort_by_key(|group| group[0]);
    let size = groups[0].len();
    if size == 0 || groups.iter().any(|group| group.len() != size)
        || groups.iter().map(Vec::len).sum::<usize>() != statements.len()
    {
        return None;
    }
    let requests = groups.len();
    let by_id: BTreeMap<usize, &Statement> = statements.iter().map(|s| (s.id, s)).collect();
    let tails = |group: &[usize]| group.iter().map(|id| by_id[id].path[1..].to_vec()).collect::<Vec<_>>();
    if (1..requests).any(|r| tails(&groups[r]) != tails(&groups[0])) {
        return None;
    }
    // Size parameter families: base names carrying a trailing request index.
    // Every parameter must belong to a complete family (one member per
    // request) or be request-independent.
    let mut families: BTreeMap<&str, Vec<Option<&str>>> = BTreeMap::new();
    for name in parameters {
        let digits = name.trim_end_matches(|c: char| c.is_ascii_digit());
        let suffix = &name[digits.len()..];
        if suffix.is_empty() {
            continue;
        }
        let index: usize = suffix.parse().ok()?;
        if index >= requests {
            return None;
        }
        families.entry(digits).or_insert_with(|| vec![None; requests])[index] = Some(name.as_str());
    }
    if families.values().any(|members| members.iter().any(Option::is_none)) {
        return None;
    }
    let mut renames = vec![Vec::new(); requests];
    for members in families.values() {
        let first = members[0]?;
        for (r, member) in members.iter().enumerate().skip(1) {
            renames[r].push((first.to_string(), (*member)?.to_string()));
        }
    }
    // Extent names are the bare parameter names; ISL text adds the __size_
    // prefix. Check extents against request 0's with bare names renamed.
    for r in 1..requests {
        let extent_renames = renames[r].iter()
            .map(|(from, to)| (from.trim_start_matches("__size_"), to.trim_start_matches("__size_")))
            .collect::<Vec<_>>();
        for (k, id) in groups[r].iter().enumerate() {
            let expected = by_id[&groups[0][k]].extents.iter()
                .map(|extent| match extent {
                    Extent::Fixed(value) => Extent::Fixed(*value),
                    Extent::Parameter(name) => Extent::Parameter(extent_renames.iter()
                        .find_map(|(from, to)| (*from == name.as_str()).then(|| to.to_string()))
                        .unwrap_or_else(|| name.clone())),
                })
                .collect::<Vec<_>>();
            if by_id[id].extents != expected {
                return None;
            }
        }
    }
    Some((groups, renames))
}

// Schedule one request's replica and lift order-preservingly:
// Theta(S_r(i), p) = (r, phi(i, p_0->p_r)). All cross-request dependencies
// must point from lower to higher request indices; the outermost time
// dimension then satisfies them strictly. The full-domain legal() check is
// the exact arbiter — nothing here relaxes validity.
fn lifted_schedule_candidates(
    ctx: &Context,
    statements: &[Statement],
    domain: &UnionSet,
    dependencies: &UnionMap,
    parallel_only: bool,
) -> Result<Option<(Vec<ScheduleCandidate>, Vec<String>)>> {
    let space = isl(domain.get_space())?;
    let parameters = (0..isl(space.dim(DimType::Param))?)
        .map(|i| isl(space.get_dim_name(DimType::Param, i as u32)).map(str::to_string))
        .collect::<Result<Vec<_>>>()?;
    let Some((groups, renames)) = replication_groups(statements, &parameters) else {
        return Ok(None);
    };
    let requests = groups.len();
    // Restrict the scheduling problem to request 0's statements and project
    // out the other requests' parameters (an existential relaxation; the
    // lifted result is still checked against the full problem).
    let r0: BTreeSet<String> = groups[0].iter().map(|id| format!("S{id}")).collect();
    let list = isl(domain.get_set_list())?;
    let mut half_domain = isl(UnionSet::empty(isl(domain.get_space())?))?;
    for i in 0..isl(list.n_set())? {
        let set = isl(list.get_at(i))?;
        if r0.contains(isl(set.get_tuple_name())?) {
            half_domain = isl(half_domain.add_set(set))?;
        }
    }
    let mut half_dependencies = isl(isl(isl(dependencies.copy())?
        .intersect_domain(isl(half_domain.copy())?))?
        .intersect_range(isl(half_domain.copy())?))?;
    for (index, name) in parameters.iter().enumerate().rev() {
        let digits = name.trim_end_matches(|c: char| c.is_ascii_digit());
        let suffix = &name[digits.len()..];
        if !suffix.is_empty() && suffix.parse::<usize>().map_or(false, |r| r > 0) {
            half_domain = isl(half_domain.project_out(DimType::Param, index as u32, 1))?;
            half_dependencies = isl(half_dependencies.project_out(DimType::Param, index as u32, 1))?;
        }
    }
    let (half_candidates, mut diagnostics) = schedule_candidates(ctx, &half_domain, &half_dependencies, parallel_only)?;
    if half_candidates.is_empty() {
        return Err("replicated program yielded no half-schedule candidates".into());
    }
    let position: BTreeMap<usize, usize> = groups[0].iter().enumerate()
        .map(|(k, id)| (*id, k)).collect();
    let param_decl = if parameters.is_empty() {
        String::new()
    } else {
        format!("[{}] -> ", parameters.join(","))
    };
    let mut candidates = Vec::new();
    for half in &half_candidates {
        let mut pieces = Vec::new();
        for piece in text_pieces(&half.schedule_map)? {
            // piece: S{id}[dims] -> [times] (: constraints)?
            let statement_end = piece.find('[').ok_or("schedule piece has no tuple")?;
            let statement: usize = piece[1..statement_end].parse()
                .map_err(|_| "schedule piece statement id is not numeric".to_string())?;
            let &k = position.get(&statement)
                .ok_or("half schedule maps a statement outside request 0")?;
            let rest = &piece[statement_end..];
            let (head, constraints) = match rest.split_once(" : ") {
                Some((head, constraints)) => (head, Some(constraints)),
                None => (rest, None),
            };
            let (iter, times) = head.rsplit_once("] -> [")
                .ok_or("schedule piece has no time tuple")?;
            let dims = &iter[1..];
            let times = times.strip_suffix(']')
                .ok_or("schedule piece time tuple is not closed")?;
            for r in 0..requests {
                let mut times = times.to_string();
                let mut constraints = constraints.map(str::to_string);
                if r > 0 {
                    for (from, to) in &renames[r] {
                        times = times.replace(from.as_str(), to);
                        constraints = constraints.map(|text| text.replace(from.as_str(), to));
                    }
                }
                let mut lifted = format!("S{}[{}] -> [{}, {}]", groups[r][k], dims, r, times);
                if let Some(constraints) = constraints {
                    lifted.push_str(&format!(" : {constraints}"));
                }
                pieces.push(lifted);
            }
        }
        ctx.reset_operations();
        let text = format!("{param_decl}{{ {} }}", pieces.join("; "));
        let map = isl(isl(UnionMap::read_from_str(ctx, &text))?
            .intersect_domain(isl(domain.copy())?))?;
        legal(domain, dependencies, &map)?;
        let schedule = isl(isl(Schedule::from_domain(isl(domain.copy())?))?
            .insert_partial_schedule(isl(isl(map.copy())?.as_multi_union_pw_aff())?))?;
        candidates.push(ScheduleCandidate {
            source: half.source.clone(),
            schedule_map: map_text(&map)?,
            schedule_tree: schedule_text(&schedule)?,
        });
    }
    diagnostics.push(format!(
        "schedule search lifted one {}-statement request replica across {requests} requests",
        groups[0].len()));
    Ok(Some((candidates, diagnostics)))
}

#[pyfunction]
#[pyo3(signature = (program_json, optimize = true, parallel_only = false))]
fn analyze_region_program(program_json: &str, optimize: bool, parallel_only: bool) -> PyResult<String> {
    let program: Program = serde_json::from_str(program_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let analysis = analyze_schedules(&program, optimize, parallel_only).map_err(pyo3::exceptions::PyValueError::new_err)?;
    canonical_ascii_json(&analysis).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn validate_region_schedule(program_json: &str, schedule_map: &str) -> PyResult<()> {
    let validate = || -> Result<()> {
        let program: Program =
            serde_json::from_str(program_json).map_err(|error| error.to_string())?;
        // Re-derive from the original program. Never trust candidate-supplied edges.
        let analysis = analyze(&program, false)?;
        let ctx = context()?;
        let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain))?;
        let dependencies = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies))?;
        let schedule = isl(UnionMap::read_from_str(&ctx, schedule_map))?;
        legal(&domain, &dependencies, &schedule)
    };
    validate().map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn preview_region_schedule(
    program_json: &str,
    schedule_map: &str,
    max_instances: u64,
) -> PyResult<String> {
    let preview = || -> Result<String> {
        let program: Program = serde_json::from_str(program_json).map_err(|e| e.to_string())?;
        if !program.parameters.is_empty() {
            return Err("preview requires concrete size bindings".into());
        }
        let analysis = analyze(&program, false)?;
        let ctx = context()?;
        let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain))?;
        let dependencies = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies))?;
        let schedule = isl(UnionMap::read_from_str(&ctx, schedule_map))?;
        legal(&domain, &dependencies, &schedule)?;
        let mut total = 0u64;
        for statement in &analysis.statements {
            let count = if statement.extents.contains(&Extent::Fixed(0)) {
                0
            } else {
                statement
                    .extents
                    .iter()
                    .try_fold(1u64, |n, e| n.checked_mul(e.fixed().ok()?))
                    .ok_or("preview instance budget exceeded")?
            };
            total = total
                .checked_add(count)
                .ok_or("preview instance budget exceeded")?;
        }
        if total > max_instances {
            return Err(
                "preview instance budget exceeded; the symbolic program remains valid".into(),
            );
        }
        let minimum = isl(Val::int_from_si(&ctx, i64::MIN))?;
        let maximum = isl(Val::int_from_si(&ctx, i64::MAX))?;
        let mut instances = Vec::new();
        for statement in &analysis.statements {
            if statement.extents.contains(&Extent::Fixed(0)) {
                continue;
            }
            let mut points = vec![vec![]];
            for extent in &statement.extents {
                let extent = extent.fixed()?;
                points = points
                    .into_iter()
                    .flat_map(|prefix| {
                        (0..extent).map(move |i| {
                            let mut p = prefix.clone();
                            p.push(i);
                            p
                        })
                    })
                    .collect();
            }
            for point in points {
                let instance = tuple(
                    &format!("S{}", statement.id),
                    &point.iter().map(u64::to_string).collect::<Vec<_>>(),
                );
                let set = isl(UnionSet::read_from_str(&ctx, &format!("{{ {instance} }}")))?;
                let mapped = isl(isl(set.apply(isl(schedule.copy())?))?.as_set())?;
                let width = isl(mapped.dim(DimType::Set))?;
                let sample = isl(mapped.sample_point())?;
                let mut time = Vec::new();
                for dim in 0..width {
                    let value = isl(sample.get_coordinate_val(DimType::Set, dim))?;
                    if !isl(value.is_int())?
                        || !isl(value.ge(&minimum))?
                        || !isl(value.le(&maximum))?
                    {
                        return Err("preview time coordinate exceeds signed 64-bit range".into());
                    }
                    time.push(isl(value.get_num_si())?);
                }
                instances.push(ScheduleInstance {
                    statement_id: statement.id,
                    coordinates: point,
                    time,
                });
            }
        }
        instances.sort_by(|a, b| {
            (&a.time, a.statement_id, &a.coordinates).cmp(&(
                &b.time,
                b.statement_id,
                &b.coordinates,
            ))
        });
        canonical_ascii_json(&instances)
    };
    preview().map_err(pyo3::exceptions::PyValueError::new_err)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(analyze_region_program, module)?)?;
    module.add_function(wrap_pyfunction!(validate_region_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(preview_region_schedule, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicated_requests_lift_the_schedule_search() {
        let integer = json!({"kind": "scalar", "name": "int"});
        let index = json!({"kind": "scalar", "name": "index"});
        let replica = |count: &str| json!({"op": "scan", "inputs": [], "outputs": [],
            "count": count, "body": {"block": {
                "arguments": [{"name": "i", "type": index}],
                "yields": [],
                "operations": [
                    {"op": "snapshot", "resource": "w", "inputs": [], "outputs": [{"name": "x", "type": integer}]},
                    {"op": "publish", "resource": "w", "inputs": ["x"], "outputs": []}
                ]
            }}});
        let mut program: Program = serde_json::from_value(json!({
            "functions": [],
            "resources": [{"kind": "state", "name": "w", "payload": [integer], "domain": []}],
            "entry": {"block": {
                "arguments": [],
                "yields": [],
                "operations": [replica("tokens0"), replica("tokens1")]
            }}
        })).unwrap();
        program.parameters = vec![
            region_program::SizeParameter { name: "tokens0".into(), minimum: 1, maximum: 8 },
            region_program::SizeParameter { name: "tokens1".into(), minimum: 1, maximum: 8 },
        ];
        // The shared atomic resource orders every instance across both
        // request replicas; cross-request conflicts point low to high, so the
        // order-preserving lift is exact here.
        let analysis = analyze(&program, true).unwrap();
        assert_eq!(analysis.statements.len(), 4);
        assert!(analysis.diagnostics.iter().any(|d| d.contains(
            "schedule search lifted one 2-statement request replica across 2 requests")));
        assert!(!analysis.diagnostics.iter().any(|d| d.contains("fell back")));
        assert_eq!(analysis.candidates.len(), 3);
        let ctx = context().unwrap();
        let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain)).unwrap();
        let deps = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies)).unwrap();
        for candidate in &analysis.candidates {
            let map = isl(UnionMap::read_from_str(&ctx, &candidate.schedule_map)).unwrap();
            legal(&domain, &deps, &map).unwrap();
        }
    }

    #[test]
    fn illegal_schedule_names_the_first_violated_edge() {
        let ctx = context().unwrap();
        let domain = isl(UnionSet::read_from_str(&ctx, "{ A[i] : 0<=i<4; B[i] : 0<=i<4 }")).unwrap();
        let dependencies = isl(UnionMap::read_from_str(&ctx, "{ A[i] -> B[i] }")).unwrap();
        // B strictly before A violates A -> B on every instance.
        let schedule = isl(UnionMap::read_from_str(&ctx, "{ B[i] -> [0,i] : 0<=i<4; A[i] -> [1,i] : 0<=i<4 }")).unwrap();
        let error = legal(&domain, &dependencies, &schedule).unwrap_err();
        assert!(error.contains("first violated edge: { A[i] -> B[i' = i]"), "{error}");
    }

    #[test]
    fn schedule_edge_reduction_preserves_parameterized_reachability() {
        let ctx = context().unwrap();
        // Dense ordered hazards reduce to adjacent versions, including an
        // empty invocation. Independent resource columns remain independent.
        let original = UnionMap::read_from_str(&ctx,
            "[N] -> { S[i,k] -> S[j,k] : 0<=N<=1000000 and 0<=i<j<N and 0<=k<2 }").unwrap();
        let reduced = scheduling_edges(&original).unwrap();
        let adjacent = UnionMap::read_from_str(&ctx,
            "[N] -> { S[i,k] -> S[i+1,k] : 0<=N<=1000000 and 0<=i<N-1 and 0<=k<2 }").unwrap();
        assert!(reduced.is_equal(&adjacent).unwrap());
        let (reachable, exact) = closure(reduced).unwrap();
        assert!(exact && reachable.is_equal(&original).unwrap());
        // A cross-statement shortcut must also preserve its original path.
        let original = UnionMap::read_from_str(&ctx,
            "{ A[] -> B[]; B[] -> C[]; A[] -> C[]; D[] -> C[] }").unwrap();
        let reduced = scheduling_edges(&original).unwrap();
        let expected = UnionMap::read_from_str(&ctx,
            "{ A[] -> B[]; B[] -> C[]; D[] -> C[] }").unwrap();
        assert!(reduced.is_equal(&expected).unwrap());
        let (before, exact_before) = closure(original).unwrap();
        let (after, exact_after) = closure(reduced).unwrap();
        assert!(exact_before && exact_after && before.is_equal(&after).unwrap());
    }

    #[test]
    fn pairwise_order_comparison_preserves_cross_statement_order_and_ties() {
        let ctx = context().unwrap();
        let source = UnionMap::read_from_str(&ctx,
            "[N] -> { A[i] -> [i,0] : 0<=i<N<=8; B[i] -> [i,1] : 0<=i<N<=8 }").unwrap();
        for (text, expected) in [
            ("[N] -> { A[i] -> [2*i+3,0,0] : 0<=i<N<=8; B[i] -> [2*i+3,0,1] : 0<=i<N<=8 }", true),
            ("[N] -> { A[i] -> [i,1] : 0<=i<N<=8; B[i] -> [i,0] : 0<=i<N<=8 }", false),
            ("[N] -> { A[i] -> [i,0] : 0<=i<N<=8; B[i] -> [i,0] : 0<=i<N<=8 }", false),
            ("[N] -> { A[i] -> [-i,0] : 0<=i<N<=8; B[i] -> [-i,1] : 0<=i<N<=8 }", false),
        ] {
            let candidate = UnionMap::read_from_str(&ctx, text).unwrap();
            assert_eq!(equivalent_order(&source, &candidate).unwrap(), expected);
            // Independently compare the complete pair relation, including
            // cross-statement pairs and zero/one-iteration parameter bindings.
            let before = source.copy().unwrap().lex_lt_union_map(source.copy().unwrap()).unwrap();
            let after = candidate.copy().unwrap().lex_lt_union_map(candidate.copy().unwrap()).unwrap();
            assert_eq!(before.is_equal(&after).unwrap(), expected);
        }
    }

    #[test]
    fn schedule_relations_keep_exact_integers_beyond_machine_width() {
        let ctx = context().unwrap();
        // 2^80 and 2^96 require the arbitrary-precision fallback. Intermediate
        // solver coefficients are not limited by the runtime's i64 index ABI.
        let schedule = UnionMap::read_from_str(&ctx,
            "{ S[i] -> [1208925819614629174706176*i + 79228162514264337593543950336] : 0<=i<8 }").unwrap();
        let order = schedule.copy().unwrap().lex_lt_union_map(schedule).unwrap();
        let expected = UnionMap::read_from_str(&ctx, "{ S[i] -> S[j] : 0<=i<j<8 }").unwrap();
        assert!(order.is_equal(&expected).unwrap());
    }
    use serde_json::json;

    fn resource_loop(indexed: bool, count: u64) -> Program {
        let index = json!({"kind": "scalar", "name": "index"});
        let integer = json!({"kind": "scalar", "name": "int"});
        serde_json::from_value(json!({
            "functions": [],
            "resources": [{"kind": "state","name": "w", "payload": [integer], "domain": if indexed { vec![count] } else { vec![] }}],
            "entry": {"block": {
                "arguments": [],
                "yields": [],
                "operations": [{"op": "scan", "inputs": [], "outputs": [],
                    "count": count, "body": {"block": {
                        "arguments": [{"name": "i", "type": index}],
                        "yields": [],
                        "operations": [
                            {"op": "snapshot", "resource": "w", "inputs": if indexed { vec!["i"] } else { vec![] },
                             "outputs": [{"name": "x", "type": integer}]},
                            {"op": "publish", "resource": "w", "inputs": if indexed { vec!["x", "i"] } else { vec!["x"] },
                             "outputs": []}
                        ]
                    }}}
                ]
            }}
        })).unwrap()
    }

    #[test]
    fn dependence_analysis_preserves_atomic_resource_conflicts() {
        let ctx = context().unwrap();
        let independent = analyze(&resource_loop(true, 4), true).unwrap();
        let atomic = analyze(&resource_loop(false, 4), true).unwrap();
        let expected = isl(UnionMap::read_from_str(&ctx, "{ S0[i] -> S1[i] : 0<=i<4 }")).unwrap();
        let actual = isl(UnionMap::read_from_str(&ctx, &independent.dependencies)).unwrap();
        assert!(isl(actual.is_equal(&expected)).unwrap());
        let schedule = isl(UnionMap::read_from_str(
            &ctx,
            "{ S0[i] -> [0,i] : 0<=i<4; S1[i] -> [1,i] : 0<=i<4 }",
        ))
        .unwrap();
        for (analysis, allowed) in [(&independent, true), (&atomic, false)] {
            assert!(analysis.arguments.is_empty() && analysis.results.is_empty());
            assert!(analysis.statements[1].outputs.is_empty());
            let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain)).unwrap();
            let deps = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies)).unwrap();
            assert_eq!(legal(&domain, &deps, &schedule).is_ok(), allowed);
        }
        // Choosing schedules must not create another interpretation of the
        // source or contaminate its reusable facts with schedule candidates.
        for (program, optimized) in [(resource_loop(true, 4), independent), (resource_loop(false, 4), atomic)] {
            let facts = analyze(&program, false).unwrap();
            let mut projected = optimized;
            projected.candidates = facts.candidates.clone();
            projected.diagnostics.truncate(facts.diagnostics.len());
            assert_eq!(projected, facts);
            assert_eq!(facts.candidates.len(), 1);
        }
    }

    #[test]
    fn dependence_analysis_does_not_expand_iteration_instances() {
        let small = analyze(&resource_loop(true, 4), true).unwrap();
        let large = analyze(&resource_loop(true, 1_000_000), true).unwrap();
        assert_eq!(small.values.len(), large.values.len());
        assert_eq!(large.statements.len(), 2);
        assert!(large.exact_provenance);
        assert!(large.dependencies.len() < 1000);
        assert_eq!(large.candidates.len(), 3);
    }

    #[test]
    fn parameter_domains_preserve_independent_and_atomic_effects_for_every_size() {
        let mut independent = resource_loop(true, 8);
        let mut atomic = resource_loop(false, 8);
        for program in [&mut independent, &mut atomic] {
            program.parameters.push(region_program::SizeParameter {
                name: "n".into(),
                minimum: 0,
                maximum: 8,
            });
            if let Operation::Scan { count, .. } = &mut program.entry.block.operations[0] {
                *count = Extent::Parameter("n".into());
            }
        }
        let ctx = context().unwrap();
        for (program, allowed) in [(&independent, true), (&atomic, false)] {
            let analysis = analyze(program, true).unwrap();
            assert!(analysis.exact_provenance);
            assert_eq!(analysis.statements.len(), 2);
            assert_eq!(
                analysis.statements[0].extents,
                vec![Extent::Parameter("n".into())]
            );
            let domain = isl(UnionSet::read_from_str(&ctx, &analysis.domain)).unwrap();
            let dependencies = isl(UnionMap::read_from_str(&ctx, &analysis.dependencies)).unwrap();
            let reordered = read_schedule(&ctx, "[__size_n] -> { S0[i] -> [0,i] : 0<=__size_n<=8 and 0<=i<__size_n; S1[i] -> [1,i] : 0<=__size_n<=8 and 0<=i<__size_n }").unwrap();
            assert_eq!(legal(&domain, &dependencies, &reordered).is_ok(), allowed);
            assert_eq!(analysis.candidates.len(), 3);
        }
    }
}

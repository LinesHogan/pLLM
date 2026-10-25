//! Structured source semantics. No task placement, page ownership, or ATen evaluator.

use std::collections::{BTreeMap, BTreeSet};

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum ScalarType {
    Index,
    Int,
    Float,
    Bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Type {
    Scalar {
        name: ScalarType,
    },
    Tensor {
        shape: Vec<u64>,
        dtype: String,
        device: String,
    },
    Bank {
        element: Box<Type>,
        size: Extent,
    },
}

fn scalar(name: ScalarType) -> Type {
    Type::Scalar { name }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub(super) enum Extent {
    Fixed(u64),
    Parameter(String),
}

impl Extent {
    pub(super) fn text(&self) -> String {
        match self {
            Self::Fixed(value) => value.to_string(),
            Self::Parameter(name) => format!("__size_{name}"),
        }
    }

    pub(super) fn maximum(&self, program: &Program) -> u64 {
        match self {
            Self::Fixed(value) => *value,
            Self::Parameter(name) => {
                program
                    .parameters
                    .iter()
                    .find(|p| &p.name == name)
                    .expect("verified extent parameter")
                    .maximum
            }
        }
    }

    pub(super) fn fixed(&self) -> Result<u64, String> {
        match self {
            Self::Fixed(value) => Ok(*value),
            Self::Parameter(_) => Err("operation requires a concrete extent".into()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct SizeParameter {
    pub name: String,
    pub minimum: u64,
    pub maximum: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ParameterTerm {
    pub parameter: String,
    pub coefficient: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct SizeConstraint {
    pub terms: Vec<ParameterTerm>,
    pub upper: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Value {
    pub name: String,
    #[serde(rename = "type")]
    pub type_: Type,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(untagged)]
pub(super) enum Literal {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl<'de> Deserialize<'de> for Literal {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LiteralVisitor;
        impl<'de> serde::de::Visitor<'de> for LiteralVisitor {
            type Value = Literal;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a boolean, signed 64-bit integer or finite floating literal")
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Literal, E> {
                Ok(Literal::Bool(value))
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Literal, E> {
                Ok(Literal::Int(value))
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Literal, E> {
                // Do not retry an overflowing unsigned integer as a float.
                i64::try_from(value).map(Literal::Int)
                    .map_err(|_| E::custom("integer constant exceeds signed 64-bit range"))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Literal, E> {
                if !value.is_finite() { return Err(E::custom("constant attributes must be finite")); }
                Ok(Literal::Float(value))
            }
        }
        deserializer.deserialize_any(LiteralVisitor)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum ScalarKind {
    Add,
    Sub,
    Mul,
    Less,
    Equal,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Operation {
    Constant {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        value: Literal,
    },
    Parameter {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        parameter: String,
    },
    Coordinate {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        affine: AffineIndex,
    },
    Scalar {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        kind: ScalarKind,
    },
    Call {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        function: String,
    },
    Select {
        inputs: Vec<String>,
        outputs: Vec<Value>,
    },
    Update {
        inputs: Vec<String>,
        outputs: Vec<Value>,
    },
    Snapshot {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        resource: String,
    },
    Publish {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        resource: String,
    },
    Scan {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        count: Extent,
        body: Region,
        #[serde(default)]
        invariants: usize,
    },
    Map {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        count: Extent,
        body: Region,
    },
    If {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        then_region: Region,
        else_region: Region,
    },
    While {
        inputs: Vec<String>,
        outputs: Vec<Value>,
        limit: u64,
        condition: Region,
        body: Region,
    },
}

impl Operation {
    pub(super) fn ports(&self) -> (&[String], &[Value]) {
        match self {
            Self::Constant {
                inputs, outputs, ..
            }
            | Self::Parameter {
                inputs, outputs, ..
            }
            | Self::Coordinate {
                inputs, outputs, ..
            }
            | Self::Scalar {
                inputs, outputs, ..
            }
            | Self::Call {
                inputs, outputs, ..
            }
            | Self::Select { inputs, outputs }
            | Self::Update { inputs, outputs }
            | Self::Snapshot {
                inputs, outputs, ..
            }
            | Self::Publish {
                inputs, outputs, ..
            }
            | Self::Scan {
                inputs, outputs, ..
            }
            | Self::Map {
                inputs, outputs, ..
            }
            | Self::If {
                inputs, outputs, ..
            }
            | Self::While {
                inputs, outputs, ..
            } => (inputs, outputs),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Block {
    pub arguments: Vec<Value>,
    pub operations: Vec<Operation>,
    pub yields: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Region {
    pub block: Block,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct IndexTerm {
    pub argument: usize,
    pub coefficient: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct AffineIndex {
    pub constant: i64,
    pub terms: Vec<IndexTerm>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct AccessRange {
    pub lower: AffineIndex,
    pub upper: AffineIndex,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum AccessMode { Read, Write }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ResourceAccess {
    pub resource: String,
    pub mode: AccessMode,
    pub ranges: Vec<AccessRange>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Function {
    pub name: String,
    pub inputs: Vec<Type>,
    pub outputs: Vec<Type>,
    pub resources: Vec<String>,
    #[serde(default)]
    pub accesses: Vec<ResourceAccess>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Resource {
    State {
        name: String,
        payload: Vec<Type>,
        domain: Vec<u64>,
    },
    External {
        name: String,
        #[serde(default)]
        domain: Vec<u64>,
    },
    Borrowed {
        name: String,
        domain: Vec<u64>,
    },
}

impl Resource {
    pub(super) fn name(&self) -> &str {
        match self {
            Self::State { name, .. } | Self::External { name, .. } | Self::Borrowed { name, .. } => name,
        }
    }
    pub(super) fn domain(&self) -> &[u64] {
        match self {
            Self::State { domain, .. } | Self::External { domain, .. } | Self::Borrowed { domain, .. } => domain,
        }
    }
    pub(super) fn payload(&self) -> Option<&[Type]> {
        match self {
            Self::State { payload, .. } => Some(payload),
            Self::External { .. } | Self::Borrowed { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Program {
    pub entry: Region,
    pub functions: Vec<Function>,
    pub resources: Vec<Resource>,
    #[serde(default)]
    pub parameters: Vec<SizeParameter>,
    #[serde(default)]
    pub constraints: Vec<SizeConstraint>,
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

struct Verifier<'a> {
    functions: BTreeMap<&'a str, &'a Function>,
    resources: BTreeMap<&'a str, &'a Resource>,
    domains: BTreeMap<&'a str, &'a [u64]>,
    parameters: BTreeMap<&'a str, &'a SizeParameter>,
}

impl Verifier<'_> {
    fn extent(&self, extent: &Extent) -> Result<(), String> {
        require(
            match extent {
                Extent::Fixed(value) => *value <= i64::MAX as u64,
                Extent::Parameter(name) => self.parameters.contains_key(name.as_str()),
            },
            "extent must fit index range or name a declared size parameter",
        )
    }

    fn type_(&self, type_: &Type) -> Result<(), String> {
        match type_ {
            Type::Bank { element, size } => {
                self.extent(size)?;
                self.type_(element)
            }
            Type::Tensor {
                shape,
                dtype,
                device,
            } => require(
                !dtype.is_empty()
                    && !device.is_empty()
                    && shape.iter().all(|&d| d <= i64::MAX as u64),
                "tensor dtype and device must be explicit",
            ),
            Type::Scalar { .. } => Ok(()),
        }
    }

    fn region(
        &self,
        region: &Region,
        expected: Option<&[Type]>,
        pure: bool,
    ) -> Result<Vec<Type>, String> {
        let block = &region.block;
        let arguments: Vec<_> = block.arguments.iter().map(|v| v.type_.clone()).collect();
        if let Some(expected) = expected {
            require(arguments == expected, "region argument signature mismatch")?;
        }
        let mut values = BTreeMap::<String, Type>::new();
        for value in &block.arguments {
            self.define(value, &mut values)?;
        }
        for operation in &block.operations {
            let (inputs, outputs) = operation.ports();
            let ins = inputs
                .iter()
                .map(|name| Self::use_(name, &mut values))
                .collect::<Result<Vec<_>, _>>()?;
            let outs = outputs.iter().map(|v| v.type_.clone()).collect::<Vec<_>>();
            match operation {
                Operation::Constant { value, .. } => {
                    require(
                        ins.is_empty() && outs.len() == 1,
                        "constant signature mismatch",
                    )?;
                    let valid = matches!(
                        (value, &outs[0]),
                        (
                            Literal::Bool(_),
                            Type::Scalar {
                                name: ScalarType::Bool
                            }
                        ) | (
                            Literal::Int(_),
                            Type::Scalar {
                                name: ScalarType::Int | ScalarType::Index
                            }
                        ) | (
                            Literal::Float(_),
                            Type::Scalar {
                                name: ScalarType::Float
                            }
                        )
                    );
                    require(valid, "constant scalar type mismatch")?;
                }
                Operation::Parameter { parameter, .. } => {
                    require(
                        ins.is_empty() && outs == [scalar(ScalarType::Index)],
                        "parameter value requires no inputs and one index result",
                    )?;
                    require(
                        self.parameters.contains_key(parameter.as_str()),
                        "parameter value requires a declared size parameter",
                    )?;
                }
                Operation::Coordinate { affine, .. } => {
                    require(
                        outs == [scalar(ScalarType::Index)],
                        "coordinate requires one index result",
                    )?;
                    require(
                        affine
                            .terms
                            .iter()
                            .all(|term| term.argument < ins.len() && term.coefficient != 0),
                        "coordinate terms must reference inputs with nonzero coefficients",
                    )?;
                    require(
                        affine
                            .terms
                            .iter()
                            .all(|term| ins[term.argument] == scalar(ScalarType::Index)),
                        "coordinate inputs must be index values",
                    )?;
                    let mut arguments: Vec<usize> =
                        affine.terms.iter().map(|term| term.argument).collect();
                    arguments.sort_unstable();
                    arguments.dedup();
                    require(
                        arguments.len() == affine.terms.len(),
                        "coordinate terms must reference distinct inputs",
                    )?;
                }
                Operation::Scalar { kind, .. } => {
                    require(
                        ins.len() == 2 && ins[0] == ins[1],
                        "scalar operands must have matching types",
                    )?;
                    require(
                        matches!(ins[0], Type::Scalar { .. }),
                        "scalar operation requires scalar inputs",
                    )?;
                    if ins[0] == scalar(ScalarType::Bool) {
                        require(
                            *kind == ScalarKind::Equal,
                            "boolean arithmetic is not defined",
                        )?;
                    }
                    let result = if matches!(kind, ScalarKind::Less | ScalarKind::Equal) {
                        scalar(ScalarType::Bool)
                    } else {
                        ins[0].clone()
                    };
                    require(outs == [result], "scalar result signature mismatch")?;
                }
                Operation::Call { function, .. } => {
                    let f = self
                        .functions
                        .get(function.as_str())
                        .ok_or("unknown function")?;
                    require(
                        !pure || f.resources.is_empty(),
                        "effectful call in a pure region",
                    )?;
                    require(
                        ins == f.inputs && outs == f.outputs,
                        "call signature mismatch",
                    )?;
                }
                Operation::Select { .. } | Operation::Update { .. } => {
                    let Some(Type::Bank { element, .. }) = ins.first() else {
                        return Err("indexed access requires a logical bank".into());
                    };
                    let select = matches!(operation, Operation::Select { .. });
                    let mut wanted = vec![ins[0].clone(), scalar(ScalarType::Index)];
                    if !select {
                        wanted.push(*element.clone());
                    }
                    let result = if select {
                        *element.clone()
                    } else {
                        ins[0].clone()
                    };
                    require(
                        ins == wanted && outs == [result],
                        "indexed access signature mismatch",
                    )?;
                }
                Operation::Snapshot { resource, .. } | Operation::Publish { resource, .. } => {
                    require(!pure, "state effect in a pure region")?;
                    let data = self
                        .resources
                        .get(resource.as_str())
                        .ok_or("unknown state resource")?
                        .payload()
                        .ok_or("external resources require imported calls, not snapshot/publish")?;
                    let indices =
                        vec![scalar(ScalarType::Index); self.domains[resource.as_str()].len()];
                    let (wanted_in, wanted_out) = if matches!(operation, Operation::Snapshot { .. })
                    {
                        (indices, data.to_vec())
                    } else {
                        let mut args = data.to_vec();
                        args.extend(indices);
                        (args, vec![])
                    };
                    require(
                        ins == wanted_in && outs == wanted_out,
                        "state effect signature mismatch",
                    )?;
                }
                Operation::Scan { body, count, invariants, .. } => {
                    self.extent(count)?;
                    let carry = ins.len().checked_sub(*invariants)
                        .ok_or("scan invariant count exceeds its inputs")?;
                    let mut args = vec![scalar(ScalarType::Index)];
                    args.extend(ins.iter().cloned());
                    let results = self.region(body, Some(&args), pure)?;
                    require(
                        results.len() >= carry && results[..carry] == ins[..carry],
                        "scan carry signature mismatch",
                    )?;
                    let mut wanted = ins[..carry].to_vec();
                    for result in &results[carry..] {
                        self.type_(result)?;
                        wanted.push(Type::Bank {
                            element: Box::new(result.clone()),
                            size: count.clone(),
                        });
                    }
                    require(outs == wanted, "scan collection signature mismatch")?;
                }
                Operation::Map { body, count, .. } => {
                    self.extent(count)?;
                    let mut args = vec![scalar(ScalarType::Index)];
                    args.extend(ins.iter().cloned());
                    let results = self.region(body, Some(&args), true)?;
                    for result in &results {
                        self.type_(result)?;
                    }
                    let wanted = results
                        .into_iter()
                        .map(|element| Type::Bank {
                            element: Box::new(element),
                            size: count.clone(),
                        })
                        .collect::<Vec<_>>();
                    require(outs == wanted, "map collection signature mismatch")?;
                }
                Operation::If {
                    then_region,
                    else_region,
                    ..
                } => {
                    require(
                        ins.first() == Some(&scalar(ScalarType::Bool)),
                        "if requires a boolean predicate",
                    )?;
                    require(
                        self.region(then_region, Some(&ins[1..]), pure)? == outs
                            && self.region(else_region, Some(&ins[1..]), pure)? == outs,
                        "if branch result signature mismatch",
                    )?;
                }
                Operation::While {
                    condition,
                    body,
                    limit,
                    ..
                } => {
                    require(*limit <= i64::MAX as u64, "while bound exceeds index range")?;
                    require(
                        self.region(condition, Some(&ins), true)? == [scalar(ScalarType::Bool)],
                        "while condition signature mismatch",
                    )?;
                    require(
                        self.region(body, Some(&ins), pure)? == ins && outs == ins,
                        "while carry signature mismatch",
                    )?;
                }
            }
            for value in outputs {
                self.define(value, &mut values)?;
            }
        }
        let results = block
            .yields
            .iter()
            .map(|name| Self::use_(name, &mut values))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    }

    fn define(
        &self,
        value: &Value,
        values: &mut BTreeMap<String, Type>,
    ) -> Result<(), String> {
        self.type_(&value.type_)?;
        require(
            !value.name.is_empty() && !values.contains_key(&value.name),
            "duplicate or empty SSA definition",
        )?;
        values.insert(value.name.clone(), value.type_.clone());
        Ok(())
    }

    fn use_(name: &str, values: &mut BTreeMap<String, Type>) -> Result<Type, String> {
        let type_ = values
            .get(name)
            .ok_or_else(|| format!("undefined value {name}; captures must be explicit"))?;
        Ok(type_.clone())
    }
}

pub(super) fn verify(program: &Program) -> Result<Vec<Type>, String> {
    verify_with_declaration(program, None)
}

pub(super) fn verify_function_body(program: &Program, declaration: &Function) -> Result<(), String> {
    verify_with_declaration(program, Some(declaration)).map(|_| ())
}

fn verify_with_declaration(program: &Program, declaration: Option<&Function>) -> Result<Vec<Type>, String> {
    let verifier = Verifier {
        parameters: program
            .parameters
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect(),
        functions: program
            .functions
            .iter()
            .map(|f| (f.name.as_str(), f))
            .collect(),
        resources: program.resources.iter().map(|r| (r.name(), r)).collect(),
        domains: program
            .resources
            .iter()
            .map(|r| (r.name(), r.domain()))
            .collect(),
    };
    if let Some(declaration) = declaration {
        require(!verifier.functions.contains_key(declaration.name.as_str()),
            "function body imports its own declaration")?;
        require(!declaration.name.is_empty(), "empty function or resource symbol")?;
    }
    require(
        verifier.parameters.len() == program.parameters.len(),
        "duplicate size parameter",
    )?;
    for parameter in &program.parameters {
        let mut chars = parameter.name.chars();
        require(
            chars
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
                && parameter.minimum <= parameter.maximum
                && parameter.maximum <= i64::MAX as u64,
            "size parameter requires a name and bounded nonnegative index range",
        )?;
    }
    for constraint in &program.constraints {
        let mut names = BTreeSet::new();
        let (mut low, mut high) = (0_i128, 0_i128);
        for term in &constraint.terms {
            require(term.coefficient != 0 && names.insert(&term.parameter),
                "size constraint requires distinct parameters and nonzero coefficients")?;
            let parameter = verifier.parameters.get(term.parameter.as_str())
                .ok_or("size constraint requires a declared parameter")?;
            let first = i128::from(term.coefficient) * i128::from(parameter.minimum);
            let last = i128::from(term.coefficient) * i128::from(parameter.maximum);
            low = low.checked_add(first.min(last)).ok_or("size constraint arithmetic exceeds i128")?;
            high = high.checked_add(first.max(last)).ok_or("size constraint arithmetic exceeds i128")?;
        }
    }
    require(
        verifier.functions.len() == program.functions.len()
            && verifier.resources.len() == program.resources.len(),
        "duplicate function or resource symbol",
    )?;
    require(
        verifier
            .functions
            .keys()
            .chain(verifier.resources.keys())
            .all(|name| !name.is_empty()),
        "empty function or resource symbol",
    )?;
    for resource in &program.resources {
        if let Some(payload) = resource.payload() {
            require(
                !payload.is_empty(),
                "resource payload must contain at least one data field",
            )?;
            for type_ in payload {
                verifier.type_(type_)?;
            }
        }
        require(
            resource.domain().iter().all(|&d| d <= i64::MAX as u64),
            "resource extent exceeds index range",
        )?;
    }
    // The outer signature is checked as a declaration, but never added to the
    // callable table. A body cannot recursively call itself through this check.
    for f in program.functions.iter().chain(declaration) {
        for t in f.inputs.iter().chain(&f.outputs) {
            verifier.type_(t)?;
        }
        require(
            f.resources.iter().collect::<BTreeSet<_>>().len() == f.resources.len(),
            "function repeats a resource",
        )?;
        for r in &f.resources {
            require(verifier.resources.contains_key(r.as_str()), "unknown function resource")?;
            if matches!(verifier.resources[r.as_str()], Resource::Borrowed { .. }) {
                require(f.accesses.iter().any(|a| &a.resource == r),
                    "borrowed resource calls require explicit indexed accesses")?;
            }
        }
        for access in &f.accesses {
            require(f.resources.contains(&access.resource), "access names an undeclared function resource")?;
            let resource = verifier.resources[access.resource.as_str()];
            require(access.ranges.len() == resource.domain().len(), "access rank differs from resource domain")?;
            for range in &access.ranges {
                for index in [&range.lower, &range.upper] {
                    let mut arguments = BTreeSet::new();
                    for term in &index.terms {
                        require(term.coefficient != 0 && arguments.insert(term.argument)
                            && f.inputs.get(term.argument) == Some(&scalar(ScalarType::Index)),
                            "access terms require distinct index operands and nonzero signed coefficients")?;
                    }
                }
            }
        }
    }
    let results = verifier.region(&program.entry, None, false)?;
    if let Some(declaration) = declaration {
        let arguments: Vec<_> = program.entry.block.arguments.iter().map(|v| v.type_.clone()).collect();
        require(arguments == declaration.inputs && results == declaration.outputs
            && verifier.resources.keys().copied().collect::<BTreeSet<_>>()
                == declaration.resources.iter().map(String::as_str).collect()
            && program.parameters.is_empty() && program.constraints.is_empty(),
            "function body differs from its closed source interface")?;
    }
    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (program_json, declaration_json=None))]
fn validate_region_program(program_json: &str, declaration_json: Option<&str>) -> PyResult<()> {
    let program: Program = serde_json::from_str(program_json)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let declaration: Option<Function> = declaration_json.map(serde_json::from_str).transpose()
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    verify_with_declaration(&program, declaration.as_ref()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(validate_region_program, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_body_signature_does_not_introduce_a_recursive_import() {
        use serde_json::json;
        let integer = json!({"kind": "scalar", "name": "int"});
        let mut program: Program = serde_json::from_value(json!({
            "entry": {"block": {"arguments": [{"name": "x", "type": integer}],
                "operations": [], "yields": ["x"]}}, "functions": [], "resources": []
        })).unwrap();
        let declaration: Function = serde_json::from_value(json!({
            "name": "outer", "inputs": [integer], "outputs": [integer], "resources": []
        })).unwrap();
        verify_function_body(&program, &declaration).unwrap();
        let mut invalid = declaration.clone();
        invalid.outputs.clear();
        assert!(verify_function_body(&program, &invalid).is_err());
        invalid = declaration.clone();
        invalid.resources.push("weights".into());
        assert!(verify_function_body(&program, &invalid).is_err());
        program.resources.push(Resource::Borrowed { name: "weights".into(), domain: vec![1] });
        assert!(verify_function_body(&program, &invalid).unwrap_err().contains("indexed accesses"));
        program.resources.clear();
        program.entry.block.operations.push(Operation::Call {
            inputs: vec!["x".into()], outputs: vec![Value { name: "y".into(), type_: scalar(ScalarType::Int) }],
            function: declaration.name.clone(),
        });
        assert!(verify_function_body(&program, &declaration).unwrap_err().contains("unknown function"));
        program.functions.push(declaration.clone());
        assert!(verify_function_body(&program, &declaration).unwrap_err().contains("own declaration"));
    }

    #[test]
    fn obsolete_order_types_are_rejected_even_inside_banks() {
        for wire in [
            r#"{"kind":"order","resource":"w"}"#,
            r#"{"kind":"bank","size":1,"element":{"kind":"order","resource":"w"}}"#,
        ] {
            assert!(serde_json::from_str::<Type>(wire).is_err());
        }
    }

    #[test]
    fn loops_remain_structural_and_must_return_the_same_carry() {
        let body = Region {
            block: Block {
                arguments: vec![
                    Value {
                        name: "i".into(),
                        type_: scalar(ScalarType::Index),
                    },
                    Value {
                        name: "x".into(),
                        type_: scalar(ScalarType::Int),
                    },
                ],
                operations: vec![],
                yields: vec!["x".into()],
            },
        };
        let mut program = Program {
            parameters: vec![],
            constraints: vec![],
            functions: vec![],
            resources: vec![],
            entry: Region {
                block: Block {
                    arguments: vec![Value {
                        name: "x".into(),
                        type_: scalar(ScalarType::Int),
                    }],
                    operations: vec![Operation::Scan {
                        invariants: 0,
                        inputs: vec!["x".into()],
                        outputs: vec![Value {
                            name: "y".into(),
                            type_: scalar(ScalarType::Int),
                        }],
                        count: Extent::Fixed(1_000_000),
                        body,
                    }],
                    yields: vec!["y".into()],
                },
            },
        };
        verify(&program).unwrap();
        if let Operation::Scan { body, .. } = &mut program.entry.block.operations[0] {
            body.block.yields[0] = "i".into();
        }
        assert!(verify(&program).is_err());
    }

    #[test]
    fn coordinates_declare_checked_affine_index_values() {
        let index = || Value {
            name: String::new(),
            type_: scalar(ScalarType::Index),
        };
        let named = |name: &str| {
            let mut value = index();
            value.name = name.into();
            value
        };
        let program = |operations: Vec<Operation>| Program {
            parameters: vec![SizeParameter {
                name: "n".into(),
                minimum: 0,
                maximum: 8,
            }],
            constraints: vec![],
            functions: vec![],
            resources: vec![],
            entry: Region {
                block: Block {
                    arguments: vec![],
                    operations,
                    yields: vec!["c".into()],
                },
            },
        };
        let parameter = Operation::Parameter {
            inputs: vec![],
            outputs: vec![named("p")],
            parameter: "n".into(),
        };
        let coordinate = |constant: i64, terms: Vec<IndexTerm>| Operation::Coordinate {
            inputs: vec!["p".into()],
            outputs: vec![named("c")],
            affine: AffineIndex { constant, terms },
        };
        verify(&program(vec![
            parameter.clone(),
            coordinate(3, vec![IndexTerm { argument: 0, coefficient: 2 }]),
        ]))
        .unwrap();
        // A constant-only coordinate takes no inputs.
        verify(&program(vec![Operation::Coordinate {
            inputs: vec![],
            outputs: vec![named("c")],
            affine: AffineIndex { constant: 3, terms: vec![] },
        }]))
        .unwrap();
        for bad in [
            // Out-of-range term reference.
            coordinate(0, vec![IndexTerm { argument: 1, coefficient: 1 }]),
            // Zero coefficient is a malformed declaration, not a dropped term.
            coordinate(0, vec![IndexTerm { argument: 0, coefficient: 0 }]),
            // Duplicate references must be merged by the constructor.
            coordinate(0, vec![
                IndexTerm { argument: 0, coefficient: 1 },
                IndexTerm { argument: 0, coefficient: 2 },
            ]),
        ] {
            assert!(verify(&program(vec![parameter.clone(), bad])).is_err());
        }
        // Non-index operands and results are rejected.
        let mut wrong_result = coordinate(0, vec![IndexTerm { argument: 0, coefficient: 1 }]);
        if let Operation::Coordinate { outputs, .. } = &mut wrong_result {
            outputs[0].type_ = scalar(ScalarType::Int);
        }
        assert!(verify(&program(vec![parameter.clone(), wrong_result])).is_err());
        let mut wrong_input = program(vec![
            Operation::Constant {
                inputs: vec![],
                outputs: vec![{
                    let mut value = named("p");
                    value.type_ = scalar(ScalarType::Int);
                    value
                }],
                value: Literal::Int(1),
            },
            coordinate(0, vec![IndexTerm { argument: 0, coefficient: 1 }]),
        ]);
        assert!(verify(&wrong_input).is_err());
        wrong_input.entry.block.yields.clear();
        assert!(verify(&wrong_input).is_err());
    }
}

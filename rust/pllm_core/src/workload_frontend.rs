//! Rust-owned normalization for the thin Python workload declaration layer.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ConfigKind {
    Bool,
    Int,
    Float,
    Str,
    IntTuple,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NumericConstraint {
    Any,
    Positive,
    Nonnegative,
    UnitInterval,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFieldSchema {
    name: String,
    kind: ConfigKind,
    constraint: NumericConstraint,
    #[serde(default)]
    choices: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigSchema {
    fields: Vec<ConfigFieldSchema>,
}

fn error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn normalize_value(
    name: &str,
    kind: ConfigKind,
    constraint: NumericConstraint,
    choices: &[String],
    value: Value,
) -> PyResult<Value> {
    let normalized = match kind {
        ConfigKind::Bool if value.is_boolean() => Ok(value),
        ConfigKind::Int if value.as_i64().is_some() => Ok(value),
        ConfigKind::Float => {
            let number = value
                .as_f64()
                .filter(|value| value.is_finite())
                .and_then(Number::from_f64)
                .ok_or_else(|| error(format!("config field {name:?} must be a finite float")))?;
            Ok(Value::Number(number))
        }
        ConfigKind::Str if value.is_string() => Ok(value),
        ConfigKind::IntTuple => {
            let values = value
                .as_array()
                .filter(|values| values.iter().all(|item| item.as_i64().is_some()))
                .ok_or_else(|| error(format!("config field {name:?} must be an integer tuple")))?;
            Ok(Value::Array(values.clone()))
        }
        _ => Err(error(format!(
            "config field {name:?} does not match its declared type"
        ))),
    }?;
    if !choices.is_empty() {
        let text = normalized.as_str().ok_or_else(|| {
            error(format!(
                "config field {name:?} uses choices on a non-string type"
            ))
        })?;
        if !choices.iter().any(|choice| choice == text) {
            return Err(error(format!(
                "config field {name:?} must be one of: {}",
                choices.join(", ")
            )));
        }
    }
    if !matches!(constraint, NumericConstraint::Any) {
        let number = normalized.as_f64().ok_or_else(|| {
            error(format!(
                "config field {name:?} uses a numeric constraint on a non-numeric type"
            ))
        })?;
        let valid = match constraint {
            NumericConstraint::Any => true,
            NumericConstraint::Positive => number > 0.0,
            NumericConstraint::Nonnegative => number >= 0.0,
            NumericConstraint::UnitInterval => (0.0..=1.0).contains(&number),
        };
        if !valid {
            return Err(error(format!(
                "config field {name:?} violates its {:?} constraint",
                constraint
            )));
        }
    }
    Ok(normalized)
}

#[pyfunction]
fn normalize_workload_config(schema_json: &str, payload_json: &str) -> PyResult<String> {
    let schema: ConfigSchema = serde_json::from_str(schema_json)
        .map_err(|source| error(format!("invalid workload config schema: {source}")))?;
    let payload: Map<String, Value> = serde_json::from_str(payload_json)
        .map_err(|source| error(format!("invalid workload config payload: {source}")))?;

    let mut names = BTreeSet::new();
    let mut fields = BTreeMap::new();
    for field in schema.fields {
        if field.name.is_empty() || !names.insert(field.name.clone()) {
            return Err(error(
                "workload config field names must be non-empty and unique",
            ));
        }
        fields.insert(field.name, (field.kind, field.constraint, field.choices));
    }
    if payload.keys().collect::<BTreeSet<_>>() != fields.keys().collect::<BTreeSet<_>>() {
        return Err(error(
            "workload config payload must contain exactly the declared fields",
        ));
    }

    let normalized = fields
        .into_iter()
        .map(|(name, (kind, constraint, choices))| {
            let value = payload
                .get(&name)
                .cloned()
                .expect("exact config field set validated");
            normalize_value(&name, kind, constraint, &choices, value).map(|value| (name, value))
        })
        .collect::<PyResult<Map<String, Value>>>()?;
    serde_json::to_string(&normalized)
        .map_err(|source| error(format!("failed to encode workload config: {source}")))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_workload_config, module)?)?;
    Ok(())
}

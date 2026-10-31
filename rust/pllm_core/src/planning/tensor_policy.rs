//! Numerical permission and preparation choices shared by tensor planning.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuningPolicy {
    Heuristic,
    Measured,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Policy {
    pub allow_numerical_transforms: bool,
    // Heuristic ranks funded tensor paths; measured uses completed execution.
    pub tuning: TuningPolicy,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            allow_numerical_transforms: true,
            tuning: TuningPolicy::Heuristic,
        }
    }
}

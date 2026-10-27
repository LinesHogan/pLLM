//! Numerical permission and preparation choices shared by tensor planning.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SchedulePolicy { Auto, Serial }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuningPolicy { Heuristic, Measured }

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Policy {
    pub allow_numerical_transforms: bool,
    pub schedule_policy: SchedulePolicy,
    // Heuristic ranks funded tensor paths; measured uses completed execution.
    pub tuning: TuningPolicy,
}

impl Default for Policy {
    fn default() -> Self {
        Self { allow_numerical_transforms: true, schedule_policy: SchedulePolicy::Auto,
               tuning: TuningPolicy::Heuristic }
    }
}


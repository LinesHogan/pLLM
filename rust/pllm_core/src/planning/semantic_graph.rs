//! Semantic graph and timeline as the Rust planner reads them, and the checks of the declaration facts
//! a user can get wrong around a fed InstanceState.
//!
//! Python's declaration layer produces these structs; Rust decodes only the fields it reads.

use serde::{Deserialize, Serialize};
use std::fmt::Write;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum InputSource {
    Hidden,
    InputTokens,
    StepId,
    HostService {
        service: String,
    },
    StateSelection {
        #[serde(default)]
        metadata: bool,
    },
    Result {
        task: String,
        output: String,
    },
    PackedRows,
    PackedSamplerRows,
    SamplerFeedback,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutputKind {
    Host,
    Temporary,
    Sampler,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateScope {
    Request,
    Instance,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestKeyField {
    RequestId,
    PromptIndex,
    SampleIndex,
    Route,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateEffectKind {
    Read,
    Transition,
    TransitionMaybe,
    Update,
    UpdateMaybe,
    Publish,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct TimingWindow {
    pub(crate) after: String,
    pub(crate) before: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct TypedInput {
    pub(crate) name: String,
    pub(crate) source: InputSource,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct TypedOutput {
    pub(crate) name: String,
    pub(crate) kind: OutputKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct SemanticState {
    pub(crate) resource: String,
    pub(crate) scope: StateScope,
    pub(crate) key_fields: Vec<RequestKeyField>,
    #[serde(default)]
    pub(crate) immutable: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct SemanticEffect {
    pub(crate) resource: String,
    pub(crate) kind: StateEffectKind,
    #[serde(default)]
    pub(crate) version: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct ImplementationCandidate {
    pub(crate) implementation_id: String,
    pub(crate) required_capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct TokenProgramEffect {
    #[serde(default)]
    pub(crate) span: bool,
    #[serde(default)]
    pub(crate) adaptive: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct MathContract {
    pub(crate) input_gradients: Vec<String>,
    pub(crate) parameter_gradients: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct SemanticTask {
    pub(crate) symbol: String,
    pub(crate) timing: TimingWindow,
    pub(crate) inputs: Vec<TypedInput>,
    pub(crate) outputs: Vec<TypedOutput>,
    pub(crate) effects: Vec<SemanticEffect>,
    pub(crate) implementations: Vec<ImplementationCandidate>,
    #[serde(default)]
    pub(crate) custom_forward: Option<TokenProgramEffect>,
    #[serde(default)]
    pub(crate) math: Option<MathContract>,
    #[serde(default)]
    pub(super) visibility: Option<VisibilitySource>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct SemanticEdge {
    pub(crate) source: String,
    pub(crate) target: String,
    #[serde(default)]
    pub(crate) step_distance: i64,
    #[serde(default)]
    pub(crate) separation: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct ConditionalInputGroup {
    pub(crate) resource: String,
    pub(crate) borrowed_resources: Vec<String>,
    pub(crate) tasks: Vec<String>,
    pub(crate) readers: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct SemanticGraph {
    pub(crate) tasks: Vec<SemanticTask>,
    pub(crate) edges: Vec<SemanticEdge>,
    pub(crate) states: Vec<SemanticState>,
    pub(crate) input_groups: Vec<ConditionalInputGroup>,
    #[serde(default)]
    pub(crate) model_reads: Vec<String>,
    #[serde(default)]
    pub(crate) model_results: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct TimelineAnchor {
    pub(crate) symbol: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct ValueResourceFact {
    pub(crate) task: String,
    pub(crate) output: String,
    pub(crate) bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct TaskResourceFact {
    pub(crate) task: String,
    pub(crate) workspace_bytes: u64,
    #[serde(default)]
    pub(crate) resident_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct Timeline {
    pub(crate) anchors: Vec<TimelineAnchor>,
    pub(crate) capabilities: Vec<String>,
    #[serde(default)]
    pub(crate) value_resources: Vec<ValueResourceFact>,
    #[serde(default)]
    pub(crate) task_resources: Vec<TaskResourceFact>,
    #[serde(default)]
    pub(crate) execution_peak_budget_bytes: Option<u64>,
    #[serde(default)]
    pub(crate) reserved_bytes: Vec<u64>,
    #[serde(default)]
    pub(super) visibility: Option<VisibilitySource>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum VisibilityKind {
    ForwardPinned,
    Exact,
    StepLag,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(super) struct VisibilitySource {
    pub policy: VisibilityKind,
    #[serde(default)]
    pub lag: u64,
}

pub(crate) fn decode_graph(payload: &str) -> Result<SemanticGraph, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid SemanticGraph: {source}"))
}

pub(crate) fn decode_timeline(payload: &str) -> Result<Timeline, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid Timeline: {source}"))
}

/// Check the declaration facts a user can get wrong; everything else is pLLM-produced and trusted.
pub(crate) fn validate_graph(graph: &SemanticGraph) -> Result<(), String> {
    debug_assert!(
        graph
            .tasks
            .windows(2)
            .all(|pair| pair[0].symbol < pair[1].symbol),
        "tasks are sorted by symbol for binary search"
    );
    for group in &graph.input_groups {
        validate_feed_chain(graph, group)?;
    }
    Ok(())
}

/// Check what a user put around a fed InstanceState. pllm builds the group itself (the publication closure
/// of the @pllm.write that fills the patched state, and the custom forwards that read it), so its shape is
/// trusted; a step without a new fed value skips the whole closure, which is why nothing else may run.
fn validate_feed_chain(graph: &SemanticGraph, group: &ConditionalInputGroup) -> Result<(), String> {
    if graph.states.len() != group.borrowed_resources.len() + 1 {
        return Err(
            "a workload with a fed InstanceState may declare only the fed state and the state its \
             @pllm.write fills; remove the other states"
                .into(),
        );
    }
    for task in &graph.tasks {
        if group.tasks.contains(&task.symbol) {
            continue;
        }
        if !group.readers.contains(&task.symbol) {
            return Err(format!(
                "{}: a workload with a fed InstanceState runs only its @pllm.write and the model \
                 forward, because a step without a new fed value skips the write; remove {} or feed=",
                task.symbol, task.symbol
            ));
        }
        if let Some(effect) =
            (task.effects.iter()).find(|effect| group.borrowed_resources.contains(&effect.resource))
        {
            return Err(format!(
                "{}: the model sees the fed state {:?} only through the state its @pllm.write fills; \
                 read {:?} instead",
                task.symbol, effect.resource, group.resource
            ));
        }
    }
    let writer_reads = |resource: &String| {
        (graph.tasks.iter())
            .filter(|task| group.tasks.contains(&task.symbol))
            .any(|task| {
                task.effects
                    .iter()
                    .any(|effect| &effect.resource == resource)
            })
    };
    if !group.borrowed_resources.iter().all(writer_reads) {
        return Err(
            "the writer that consumes a fed InstanceState must read it; read the fed \
             state or remove feed="
                .into(),
        );
    }
    Ok(())
}

pub(crate) fn canonical_ascii_json<T: Serialize>(value: &T) -> Result<String, String> {
    let value = serde_json::to_value(value)
        .map_err(|source| format!("failed to encode canonical JSON: {source}"))?;
    let encoded = serde_json::to_string(&value)
        .map_err(|source| format!("failed to serialize canonical JSON: {source}"))?;
    let mut canonical = String::with_capacity(encoded.len());
    for character in encoded.chars() {
        if character.is_ascii() {
            canonical.push(character);
            continue;
        }
        let scalar = character as u32;
        if scalar <= 0xffff {
            write!(&mut canonical, "\\u{scalar:04x}").expect("writing to String cannot fail");
            continue;
        }
        let surrogate = scalar - 0x1_0000;
        let high = 0xd800 + (surrogate >> 10);
        let low = 0xdc00 + (surrogate & 0x3ff);
        write!(&mut canonical, "\\u{high:04x}\\u{low:04x}").expect("writing to String cannot fail");
    }
    Ok(canonical)
}

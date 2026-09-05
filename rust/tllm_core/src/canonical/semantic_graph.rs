//! Closed semantic graph and timeline contracts for clean-room scheduling.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write;

pub(crate) const SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HiddenSite {
    Post,
    BlockOutput,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HiddenPhase {
    Prefill,
    Decode,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum InputSource {
    Hidden {
        layer: i64,
        site: HiddenSite,
        phase: HiddenPhase,
    },
    StepId,
    HostService {
        service: String,
        port: String,
        schema: String,
    },
    StateSelection {
        resource: String,
        phase: HiddenPhase,
    },
    Result {
        task: String,
        output: String,
    },
    PackedRows,
    PackedSamplerRows,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutputKind {
    Host,
    Temporary,
    Sampler,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateScope {
    Request,
    Instance,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestKeyField {
    RequestId,
    PromptIdx,
    SampleIdx,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateEffectKind {
    Read,
    Transition,
    TransitionMaybe,
    Publish,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionKind {
    HostService,
    NativeTask,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimingWindow {
    pub(crate) after: String,
    pub(crate) before: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedInput {
    pub(crate) name: String,
    pub(crate) source: InputSource,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedOutput {
    pub(crate) name: String,
    pub(crate) kind: OutputKind,
    pub(crate) schema: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SemanticState {
    pub(crate) resource: String,
    pub(crate) scope: StateScope,
    pub(crate) key_fields: Vec<RequestKeyField>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SemanticEffect {
    pub(crate) resource: String,
    pub(crate) kind: StateEffectKind,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImplementationCandidate {
    pub(crate) implementation_id: String,
    pub(crate) execution_kind: ExecutionKind,
    pub(crate) required_capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SemanticTask {
    pub(crate) symbol: String,
    pub(crate) timing: TimingWindow,
    pub(crate) inputs: Vec<TypedInput>,
    pub(crate) outputs: Vec<TypedOutput>,
    pub(crate) effects: Vec<SemanticEffect>,
    pub(crate) implementations: Vec<ImplementationCandidate>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SemanticEdge {
    pub(crate) source: String,
    pub(crate) target: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SemanticGraph {
    pub(crate) workload_id: String,
    pub(crate) tasks: Vec<SemanticTask>,
    pub(crate) execution_parameters_digest: String,
    pub(crate) edges: Vec<SemanticEdge>,
    pub(crate) states: Vec<SemanticState>,
    pub(crate) schema_version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimelineAnchor {
    pub(crate) symbol: String,
    pub(crate) ordinal: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Timeline {
    pub(crate) anchors: Vec<TimelineAnchor>,
    pub(crate) capabilities: Vec<String>,
    pub(crate) schema_version: u64,
}

pub(crate) fn decode_graph(payload: &str) -> Result<SemanticGraph, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid SemanticGraph: {source}"))
}

pub(crate) fn decode_timeline(payload: &str) -> Result<Timeline, String> {
    serde_json::from_str(payload).map_err(|source| format!("invalid Timeline: {source}"))
}

pub(crate) fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{label} must be a non-empty string"));
    }
    Ok(())
}

pub(crate) fn require_sorted_unique_text(values: &[String], label: &str) -> Result<(), String> {
    for value in values {
        require_text(value, label)?;
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(format!("{label} must be sorted unique names"));
    }
    Ok(())
}

fn validate_task(task: &SemanticTask) -> Result<(), String> {
    require_text(&task.symbol, "tasks.symbol")?;
    require_text(&task.timing.after, "tasks.timing.after")?;
    require_text(&task.timing.before, "tasks.timing.before")?;
    for input in &task.inputs {
        require_text(&input.name, "tasks.inputs.name")?;
        match &input.source {
            InputSource::StateSelection { resource, .. } => {
                require_text(resource, "tasks.inputs.source.resource")?;
            }
            InputSource::Result { task, output } => {
                require_text(task, "tasks.inputs.source.task")?;
                require_text(output, "tasks.inputs.source.output")?;
            }
            InputSource::HostService {
                service,
                port,
                schema,
            } => {
                require_text(service, "tasks.inputs.source.service")?;
                require_text(port, "tasks.inputs.source.port")?;
                require_text(schema, "tasks.inputs.source.schema")?;
            }
            InputSource::Hidden { .. }
            | InputSource::StepId
            | InputSource::PackedRows
            | InputSource::PackedSamplerRows => {}
        }
    }
    if task
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect::<BTreeSet<_>>()
        .len()
        != task.inputs.len()
    {
        return Err("task inputs must have unique names".to_owned());
    }
    for output in &task.outputs {
        require_text(&output.name, "tasks.outputs.name")?;
        match output.kind {
            OutputKind::Host => require_text(&output.schema, "tasks.outputs.schema")?,
            OutputKind::Temporary | OutputKind::Sampler => {
                if !output.schema.is_empty() {
                    return Err("non-host task output cannot declare a host schema".to_owned());
                }
            }
        }
    }
    if task
        .outputs
        .iter()
        .map(|output| output.name.as_str())
        .collect::<BTreeSet<_>>()
        .len()
        != task.outputs.len()
    {
        return Err("task outputs must have unique names".to_owned());
    }
    for effect in &task.effects {
        require_text(&effect.resource, "tasks.effects.resource")?;
    }
    if task
        .effects
        .iter()
        .map(|effect| effect.resource.as_str())
        .collect::<BTreeSet<_>>()
        .len()
        != task.effects.len()
    {
        return Err("task may have at most one effect per state resource".to_owned());
    }
    if task.implementations.is_empty() {
        return Err("task requires at least one implementation candidate".to_owned());
    }
    for implementation in &task.implementations {
        require_text(
            &implementation.implementation_id,
            "tasks.implementations.implementation_id",
        )?;
        require_sorted_unique_text(
            &implementation.required_capabilities,
            "tasks.implementations.required_capabilities",
        )?;
    }
    if task
        .implementations
        .windows(2)
        .any(|pair| pair[0].implementation_id >= pair[1].implementation_id)
    {
        return Err("task implementations must be sorted by unique ID".to_owned());
    }
    Ok(())
}

pub(crate) fn validate_graph(graph: &SemanticGraph) -> Result<(), String> {
    if graph.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported SemanticGraph schema_version: {}",
            graph.schema_version
        ));
    }
    require_text(&graph.workload_id, "workload_id")?;
    if graph.execution_parameters_digest.len() != 64
        || !graph
            .execution_parameters_digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("execution_parameters_digest must be lowercase SHA-256".to_owned());
    }
    if graph.tasks.is_empty() {
        return Err("semantic graph requires at least one task".to_owned());
    }
    for task in &graph.tasks {
        validate_task(task)?;
    }
    if graph
        .tasks
        .windows(2)
        .any(|pair| pair[0].symbol >= pair[1].symbol)
    {
        return Err("semantic graph tasks must be sorted by unique symbol".to_owned());
    }
    for state in &graph.states {
        require_text(&state.resource, "states.resource")?;
        match state.scope {
            StateScope::Request => {
                if state.key_fields.is_empty()
                    || state.key_fields.windows(2).any(|pair| pair[0] >= pair[1])
                {
                    return Err("request state key fields must be sorted unique".to_owned());
                }
            }
            StateScope::Instance => {
                if !state.key_fields.is_empty() {
                    return Err("instance state cannot declare request keys".to_owned());
                }
            }
        }
    }
    if graph
        .states
        .windows(2)
        .any(|pair| pair[0].resource >= pair[1].resource)
    {
        return Err("semantic graph states must be sorted by unique resource".to_owned());
    }
    for edge in &graph.edges {
        require_text(&edge.source, "edges.source")?;
        require_text(&edge.target, "edges.target")?;
        if edge.source == edge.target {
            return Err("semantic graph edge cannot be self-referential".to_owned());
        }
    }
    if graph.edges.windows(2).any(|pair| {
        (pair[0].source.as_str(), pair[0].target.as_str())
            >= (pair[1].source.as_str(), pair[1].target.as_str())
    }) {
        return Err("semantic graph edges must be sorted unique".to_owned());
    }
    for edge in &graph.edges {
        if graph
            .tasks
            .binary_search_by(|task| task.symbol.cmp(&edge.source))
            .is_err()
            || graph
                .tasks
                .binary_search_by(|task| task.symbol.cmp(&edge.target))
                .is_err()
        {
            return Err("semantic graph edge references an unknown task".to_owned());
        }
    }
    for task in &graph.tasks {
        for input in &task.inputs {
            match &input.source {
                InputSource::Hidden { .. }
                | InputSource::StepId
                | InputSource::HostService { .. } => {}
                InputSource::StateSelection { resource, .. } => {
                    let state = graph
                        .states
                        .binary_search_by(|state| state.resource.cmp(resource))
                        .map(|index| &graph.states[index])
                        .map_err(|_| {
                            "state_selection input references an unknown state resource".to_owned()
                        })?;
                    if state.scope != StateScope::Request {
                        return Err("state_selection input must reference request state".to_owned());
                    }
                }
                InputSource::Result {
                    task: producer,
                    output,
                } => {
                    let producer_index = graph
                        .tasks
                        .binary_search_by(|candidate| candidate.symbol.as_str().cmp(producer))
                        .map_err(|_| "result input producer is unknown".to_owned())?;
                    if !graph.tasks[producer_index]
                        .outputs
                        .iter()
                        .any(|candidate| candidate.name == *output)
                    {
                        return Err("result input producer output is unknown".to_owned());
                    }
                }
                InputSource::PackedRows | InputSource::PackedSamplerRows => {}
            }
        }
        for effect in &task.effects {
            if graph
                .states
                .binary_search_by(|state| state.resource.cmp(&effect.resource))
                .is_err()
            {
                return Err("state effect references an unknown state resource".to_owned());
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_timeline(timeline: &Timeline) -> Result<(), String> {
    if timeline.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported Timeline schema_version: {}",
            timeline.schema_version
        ));
    }
    if timeline.anchors.is_empty() {
        return Err("timeline requires at least one anchor".to_owned());
    }
    for anchor in &timeline.anchors {
        require_text(&anchor.symbol, "anchors.symbol")?;
    }
    if timeline
        .anchors
        .iter()
        .enumerate()
        .any(|(index, anchor)| anchor.ordinal != index as u64)
    {
        return Err("timeline anchors must be uniquely ordered by ordinal".to_owned());
    }
    if timeline
        .anchors
        .iter()
        .map(|anchor| anchor.symbol.as_str())
        .collect::<BTreeSet<_>>()
        .len()
        != timeline.anchors.len()
    {
        return Err("timeline anchors must have unique IDs".to_owned());
    }
    require_sorted_unique_text(&timeline.capabilities, "timeline.capabilities")
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

pub(crate) fn digest(payload: &str) -> String {
    format!("{:x}", Sha256::digest(payload.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::{canonical_ascii_json, decode_graph, digest, validate_graph};
    use serde_json::json;

    #[test]
    fn semantic_graph_closed_codec_rejects_schema_drift() {
        let graph = json!({
            "schema_version": 1, "workload_id": "closed",
            "execution_parameters_digest": "a".repeat(64), "edges": [],
            "states": [{"resource": "weights", "scope": "request", "key_fields": ["request_id"]}],
            "tasks": [{
                "symbol": "predict", "timing": {"after": "begin", "before": "end"},
                "inputs": [{"name": "hidden", "source": {"kind": "hidden", "layer": 0,
                    "site": "block_output", "phase": "decode"}}],
                "outputs": [{"name": "prediction", "kind": "temporary", "schema": ""}],
                "effects": [{"resource": "weights", "kind": "read"}],
                "implementations": [{"implementation_id": "cpu", "execution_kind": "native_task",
                    "required_capabilities": ["cpu"]}]
            }]
        });
        let encoded = graph.to_string();
        let decoded = decode_graph(&encoded).unwrap();
        validate_graph(&decoded).unwrap();
        let canonical = canonical_ascii_json(&decoded).unwrap();
        assert_eq!(decode_graph(&canonical).unwrap(), decoded);
        let duplicate = format!("{{\"schema_version\":1,{}", &encoded[1..]);
        assert!(decode_graph(&duplicate).is_err());
        let mut missing = graph.clone();
        missing.as_object_mut().unwrap().remove("states");
        assert!(decode_graph(&missing.to_string()).is_err());
        let mut unknown = graph.clone();
        unknown["tasks"][0]["inputs"][0]["source"]["callback"] = json!("unregistered");
        assert!(decode_graph(&unknown.to_string()).is_err());
        let mut wrong_kind = graph.clone();
        wrong_kind["tasks"][0]["inputs"][0]["source"]["kind"] = json!("python_hook");
        assert!(decode_graph(&wrong_kind.to_string()).is_err());
        let mut wrong_schema = graph.clone();
        wrong_schema["schema_version"] = json!(2);
        assert!(validate_graph(&decode_graph(&wrong_schema.to_string()).unwrap()).is_err());
        let mut unknown_state = graph.clone();
        unknown_state["tasks"][0]["effects"][0]["resource"] = json!("missing");
        assert!(validate_graph(&decode_graph(&unknown_state.to_string()).unwrap()).is_err());
        let mut changed = graph;
        changed["tasks"][0]["symbol"] = json!("changed");
        let changed = decode_graph(&changed.to_string()).unwrap();
        validate_graph(&changed).unwrap();
        assert_ne!(digest(&canonical), digest(&canonical_ascii_json(&changed).unwrap()));
    }
}

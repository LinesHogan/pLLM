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
    ModelOutput,
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
    SamplerFeedback,
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
    Route,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateEffectKind {
    Read,
    Transition,
    TransitionMaybe,
    Update,
    UpdateMaybe,
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
    #[serde(default)]
    pub(crate) version: Option<String>,
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
pub(crate) struct ConditionalLayerReplay {
    pub(crate) layers: Vec<u32>,
    pub(crate) max_iterations: u32,
    pub(crate) relative_l2_ppm: u32,
    pub(crate) predicate: String,
    pub(crate) rows: String,
    pub(crate) carry: String,
    pub(crate) kv: String,
    pub(crate) logits: String,
    pub(crate) rng: String,
    pub(crate) gradient: String,
    pub(crate) cancellation: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TokenwiseLayerReplay {
    pub(crate) source_layer: u32,
    pub(crate) destination_layer: u32,
    pub(crate) layer_count: u32,
    pub(crate) alpha_bits: u64,
    pub(crate) beta_bits: Option<u64>,
    pub(crate) ramp_tokens: u32,
    pub(crate) enabled: bool,
    pub(crate) rows: String,
    pub(crate) kv: String,
    pub(crate) logits: String,
    pub(crate) rng: String,
    pub(crate) gradient: String,
    pub(crate) cancellation: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub(crate) enum LayerReplay {
    Conditional(ConditionalLayerReplay),
    Tokenwise(TokenwiseLayerReplay),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct MathContract {
    pub(crate) input_gradients: Vec<String>,
    pub(crate) parameter_gradients: Vec<String>,
    pub(crate) detached_inputs: Vec<String>,
    pub(crate) forward_rows: String,
    pub(crate) input_gradient_rows: String,
    pub(crate) parameter_gradient_rows: String,
    pub(crate) update_states: String,
    pub(crate) reduction: String,
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
    #[serde(default)]
    pub(crate) replay: Option<LayerReplay>,
    #[serde(default)]
    pub(crate) math: Option<MathContract>,
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
pub(crate) struct ValueResourceFact {
    pub(crate) task: String,
    pub(crate) output: String,
    pub(crate) bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskResourceFact {
    pub(crate) task: String,
    pub(crate) workspace_bytes: u64,
    #[serde(default)]
    pub(crate) resident_bytes: u64,
    #[serde(default)]
    pub(crate) input_shapes: Vec<Vec<u64>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Timeline {
    pub(crate) anchors: Vec<TimelineAnchor>,
    pub(crate) capabilities: Vec<String>,
    pub(crate) schema_version: u64,
    #[serde(default)]
    pub(crate) value_resources: Vec<ValueResourceFact>,
    #[serde(default)]
    pub(crate) task_resources: Vec<TaskResourceFact>,
    #[serde(default)]
    pub(crate) memory_budget_bytes: Option<u64>,
    #[serde(default)]
    pub(crate) reserved_bytes: Vec<u64>,
    #[serde(default)]
    pub(crate) region_policy: super::region_selection::Policy,
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
    if let Some(math) = &task.math {
        let inputs: BTreeSet<_> = task.inputs.iter().map(|input| input.name.as_str()).collect();
        for names in [&math.input_gradients, &math.detached_inputs, &math.parameter_gradients] {
            if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
                return Err(format!("{}: duplicate mathematical selection", task.symbol));
            }
        }
        if math.input_gradients.iter().any(|name| math.detached_inputs.contains(name))
            || math.input_gradients.iter().chain(&math.detached_inputs).any(|name| !inputs.contains(name.as_str()))
            || [&math.forward_rows, &math.input_gradient_rows, &math.parameter_gradient_rows, &math.update_states]
                .iter().any(|name| !name.is_empty() && !inputs.contains(name.as_str()))
            || math.parameter_gradients.iter().any(|resource| !task.effects.iter().any(|effect| effect.resource == *resource))
            || !matches!(math.reduction.as_str(), "none" | "sum" | "mean" | "state_mean" | "weighted_state_mean" | "weighted_state_mean_then_mean" | "custom")
        {
            return Err(format!("{}: mathematical contract differs from original inputs/state", task.symbol));
        }
        if !math.update_states.is_empty() && !task.effects.iter().any(|effect|
            matches!(effect.kind, StateEffectKind::Transition | StateEffectKind::TransitionMaybe | StateEffectKind::Update | StateEffectKind::UpdateMaybe)) {
            return Err(format!("{}: update eligibility requires a declared mutation", task.symbol));
        }
    }
    if let Some(LayerReplay::Conditional(replay)) = &task.replay {
        if replay.layers.is_empty()
            || replay.layers.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(format!("{}: replay layers must be sorted and unique", task.symbol));
        }
        if replay.max_iterations == 0
            || replay.max_iterations > 2
            || replay.relative_l2_ppm > 1_000_000
            || replay.predicate != "weight_updated"
            || replay.rows != "last_scheduled_token"
            || replay.carry != "block_output"
            || replay.kv != "overwrite_current_slot"
            || replay.logits != "preserve"
            || replay.rng != "preserve"
            || replay.gradient != "detach"
            || replay.cancellation != "between_iterations_restore_kv"
        {
            return Err("unsupported conditional replay effect".to_owned());
        }
    }
    if let Some(LayerReplay::Tokenwise(replay)) = &task.replay {
        let alpha = f64::from_bits(replay.alpha_bits);
        let beta = replay.beta_bits.map(f64::from_bits);
        if replay.destination_layer >= replay.source_layer
            || replay.source_layer >= replay.layer_count
            || !(0.0..=1.0).contains(&alpha)
            || beta.is_some_and(|value| !(0.0..=1.0).contains(&value))
            || replay.rows != "prefill" || replay.kv != "overwrite_current_slot"
            || replay.logits != "normal" || replay.rng != "preserve"
            || replay.gradient != "detach" || replay.cancellation != "between_tokens"
            || !task.effects.is_empty()
        {
            return Err("unsupported tokenwise layer replay effect".to_owned());
        }
    }
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
            | InputSource::PackedSamplerRows
            | InputSource::SamplerFeedback => {}
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
        if matches!(task.replay, Some(LayerReplay::Conditional(_))) {
            let reads = task
                .effects
                .iter()
                .filter(|effect| effect.kind == StateEffectKind::Read)
                .collect::<Vec<_>>();
            if reads.len() != 1
                || task.effects.len() != 1
                || !graph.states.iter().any(|state| {
                    state.resource == reads[0].resource && state.scope == StateScope::Request
                })
                || !task.inputs.iter().any(|input| match &input.source {
                    InputSource::Result { task: source, .. } => {
                        graph.tasks.iter().any(|producer| {
                            producer.symbol == *source
                                && reads[0].version.as_ref() == Some(source)
                                && producer.effects.iter().any(|effect| {
                                    effect.resource == reads[0].resource
                                        && effect.kind == StateEffectKind::Publish
                                })
                        })
                    }
                    _ => false,
                })
            {
                return Err(
                    "replay requires an exact request-state publication dependency".to_owned(),
                );
            }
        }
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
                    graph
                        .states
                        .binary_search_by(|state| state.resource.cmp(resource))
                        .map_err(|_| {
                            "state_selection input references an unknown state resource".to_owned()
                        })?;
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
                InputSource::PackedRows
                | InputSource::PackedSamplerRows
                | InputSource::SamplerFeedback => {}
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
    require_sorted_unique_text(&timeline.capabilities, "timeline.capabilities")?;
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
        assert_ne!(
            digest(&canonical),
            digest(&canonical_ascii_json(&changed).unwrap())
        );
    }
}

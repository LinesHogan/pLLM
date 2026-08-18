//! Dependency-aware placement compiler for consumer flows.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};

const HOOK_POINTS: [&str; 13] = [
    "load_model.pre",
    "load_model.post",
    "prepare_inputs.pre",
    "prepare_inputs.post",
    "layer.pre",
    "layer.post",
    "block.end",
    "stack.end",
    "execute_model.pre",
    "execute_model.post",
    "sample_tokens.pre",
    "sampler.pre_draw",
    "sample_tokens.post",
];

const IN_CAPTURE_HOOKS: [&str; 4] = ["layer.pre", "layer.post", "block.end", "stack.end"];
const MOBILITIES: [&str; 4] = ["pinned", "deferrable", "advanceable", "free"];
const DEADLINES: [&str; 3] = ["step_end", "next_prepare", "generate_end"];
const OBSERVATIONS: [&str; 3] = ["at_declared_window", "latest_at_execution", "stale_ok"];
const GRAPH_MODES: [&str; 3] = ["eager", "piecewise", "full"];
const LEASE_LIFETIMES: [&str; 2] = ["consume_call", "step"];
const FUSION_KINDS: [&str; 2] = ["none", "side_stream_cudagraph"];

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct LocatorFieldDesc {
    #[pyo3(attribute)]
    path: Vec<String>,
    #[pyo3(attribute)]
    value_type: String,
    #[pyo3(attribute)]
    value: String,
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct PortAddressDesc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    locator_type: String,
    #[pyo3(attribute)]
    locator_fields: Vec<LocatorFieldDesc>,
}

impl PortAddressDesc {
    fn label(&self) -> String {
        if self.locator_type.is_empty() {
            return self.kind.clone();
        }
        let fields: Vec<String> = self
            .locator_fields
            .iter()
            .map(|field| {
                format!(
                    "{}:{}={}",
                    field.path.join("."),
                    field.value_type,
                    field.value
                )
            })
            .collect();
        format!("{}|{}|{}", self.kind, self.locator_type, fields.join(","))
    }
}

#[derive(Clone, Debug, Eq, FromPyObject, Hash, PartialEq)]
struct StateResourceDesc {
    #[pyo3(attribute)]
    kind: String,
    #[pyo3(attribute)]
    namespace: String,
    #[pyo3(attribute)]
    name: String,
    #[pyo3(attribute)]
    version: i64,
}

impl StateResourceDesc {
    fn same_resource(&self, other: &Self) -> bool {
        self.kind == other.kind && self.namespace == other.namespace && self.name == other.name
    }

    fn label(&self) -> String {
        format!(
            "{}|{}|{}@{}",
            self.kind, self.namespace, self.name, self.version
        )
    }
}

#[derive(Clone, FromPyObject)]
struct FlowWriteDesc {
    #[pyo3(attribute)]
    port: PortAddressDesc,
    #[pyo3(attribute)]
    #[allow(dead_code)]
    order: i64,
    #[pyo3(attribute)]
    mode: String,
    #[pyo3(attribute)]
    fusion_candidate: bool,
}

#[derive(Clone, FromPyObject)]
struct FlowResourceWriteDesc {
    #[pyo3(attribute)]
    resource: StateResourceDesc,
}

#[derive(Clone, FromPyObject)]
struct FlowTimingDesc {
    #[pyo3(attribute)]
    flow_id: String,
    #[pyo3(attribute)]
    consumer_id: String,
    #[pyo3(attribute)]
    declared_hook: String,
    #[pyo3(attribute)]
    mobility: Option<String>,
    #[pyo3(attribute)]
    deadline: Option<String>,
    #[pyo3(attribute)]
    max_defer_steps: Option<i64>,
    #[pyo3(attribute)]
    observes_version: Option<String>,
    #[pyo3(attribute)]
    reads: Vec<PortAddressDesc>,
    #[pyo3(attribute)]
    writes: Vec<FlowWriteDesc>,
    #[pyo3(attribute)]
    delivery: String,
    #[pyo3(attribute)]
    ownership: String,
    #[pyo3(attribute)]
    lease_lifetime: String,
    #[pyo3(attribute)]
    max_bundle_rows: i64,
    #[pyo3(attribute)]
    effect_hook: Option<String>,
    #[pyo3(attribute)]
    earliest_hook: Option<String>,
    #[pyo3(attribute)]
    allowed_hooks: Vec<String>,
    #[pyo3(attribute)]
    fusion_kind: String,
    #[pyo3(attribute)]
    resource_reads: Vec<StateResourceDesc>,
    #[pyo3(attribute)]
    resource_writes: Vec<FlowResourceWriteDesc>,
}

#[derive(Clone, FromPyObject)]
struct HookCostDesc {
    #[pyo3(attribute)]
    hook: String,
    #[pyo3(attribute)]
    bubble_score: i64,
}

impl FlowTimingDesc {
    fn mobility(&self) -> &str {
        self.mobility.as_deref().unwrap_or("pinned")
    }

    fn deadline(&self) -> &str {
        self.deadline.as_deref().unwrap_or("step_end")
    }

    fn max_defer_steps(&self) -> i64 {
        self.max_defer_steps.unwrap_or(0)
    }

    fn observes_version(&self) -> &str {
        self.observes_version
            .as_deref()
            .unwrap_or("at_declared_window")
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct PlacementSchedule {
    /// (flow_id, hook_point) in declaration order.
    #[pyo3(get)]
    placements: Vec<(String, String)>,
    /// Flow ids in dependency-safe topological execution order.
    #[pyo3(get)]
    execution_order: Vec<String>,
    #[pyo3(get)]
    fusion_groups: Vec<Vec<String>>,
    #[pyo3(get)]
    fusion_rejections: Vec<(String, String)>,
}

fn qr(s: &str) -> String {
    format!("'{}'", s)
}

fn qr_list(v: &[&str]) -> String {
    let inner: Vec<String> = v.iter().map(|s| qr(s)).collect();
    format!("[{}]", inner.join(", "))
}

fn plan_err(msg: String) -> PyErr {
    PyRuntimeError::new_err(msg)
}

fn hook_index(hook: &str) -> Option<usize> {
    HOOK_POINTS.iter().position(|candidate| *candidate == hook)
}

fn default_bubble_score(hook: &str) -> i64 {
    match hook {
        "sampler.pre_draw" => 120,
        "sample_tokens.pre" => 100,
        "prepare_inputs.post" => 60,
        "sample_tokens.post" => 40,
        "execute_model.post" => 20,
        "execute_model.pre" => 10,
        _ => 0,
    }
}

fn add_edge(
    from: usize,
    to: usize,
    successors: &mut [BTreeSet<usize>],
    predecessors: &mut [BTreeSet<usize>],
) {
    if from != to && successors[from].insert(to) {
        predecessors[to].insert(from);
    }
}

fn add_state_read_write_edge(
    read_flow: usize,
    read: &StateResourceDesc,
    write_flow: usize,
    write: &StateResourceDesc,
    successors: &mut [BTreeSet<usize>],
    predecessors: &mut [BTreeSet<usize>],
) {
    if !read.same_resource(write) {
        return;
    }
    // A write publishes its own generation. Reads of that generation or a
    // later one follow it (RAW); a read of an older generation precedes the
    // update (WAR).
    let (from, to) = if write.version <= read.version {
        (write_flow, read_flow)
    } else {
        (read_flow, write_flow)
    };
    add_edge(from, to, successors, predecessors);
}

/// Validate timing contracts and emit a dependency-safe placement.
#[pyfunction]
#[pyo3(signature = (flows, graph_mode, hook_costs=None))]
fn compile_placement_schedule(
    flows: Vec<FlowTimingDesc>,
    graph_mode: &str,
    hook_costs: Option<Vec<HookCostDesc>>,
) -> PyResult<PlacementSchedule> {
    if !GRAPH_MODES.contains(&graph_mode) {
        return Err(plan_err(format!(
            "unsupported graph mode {}; expected one of {}",
            qr(graph_mode),
            qr_list(&GRAPH_MODES)
        )));
    }
    let mut placements: Vec<(String, String)> = Vec::with_capacity(flows.len());
    for flow in &flows {
        if !HOOK_POINTS.contains(&flow.declared_hook.as_str()) {
            return Err(plan_err(format!(
                "flow {} declares unknown hook point {}; expected one of {}",
                qr(&flow.flow_id),
                qr(&flow.declared_hook),
                qr_list(&HOOK_POINTS)
            )));
        }
        if let Some(earliest_hook) = flow.earliest_hook.as_deref() {
            if !HOOK_POINTS.contains(&earliest_hook) {
                return Err(plan_err(format!(
                    "flow {} declares unknown earliest hook {}; expected one of {}",
                    qr(&flow.flow_id),
                    qr(earliest_hook),
                    qr_list(&HOOK_POINTS)
                )));
            }
        }
        for allowed_hook in &flow.allowed_hooks {
            if !HOOK_POINTS.contains(&allowed_hook.as_str()) {
                return Err(plan_err(format!(
                    "flow {} allows unknown hook point {}; expected one of {}",
                    qr(&flow.flow_id),
                    qr(allowed_hook),
                    qr_list(&HOOK_POINTS)
                )));
            }
        }
        if !FUSION_KINDS.contains(&flow.fusion_kind.as_str()) {
            return Err(plan_err(format!(
                "flow {} declares unsupported fusion kind {}; expected one of {}",
                qr(&flow.flow_id),
                qr(&flow.fusion_kind),
                qr_list(&FUSION_KINDS)
            )));
        }
        let mobility = flow.mobility();
        if !MOBILITIES.contains(&mobility) {
            return Err(plan_err(format!(
                "flow {} declares unsupported timing mobility {}; expected one of {}",
                qr(&flow.flow_id),
                qr(mobility),
                qr_list(&MOBILITIES)
            )));
        }
        let deadline = flow.deadline();
        if !DEADLINES.contains(&deadline) {
            return Err(plan_err(format!(
                "flow {} declares unsupported timing deadline {}; expected one of {}",
                qr(&flow.flow_id),
                qr(deadline),
                qr_list(&DEADLINES)
            )));
        }
        let observes = flow.observes_version();
        if !OBSERVATIONS.contains(&observes) {
            return Err(plan_err(format!(
                "flow {} declares unsupported version observation {}; expected one of {}",
                qr(&flow.flow_id),
                qr(observes),
                qr_list(&OBSERVATIONS)
            )));
        }
        let max_defer = flow.max_defer_steps();
        if max_defer < 0 {
            return Err(plan_err(format!(
                "flow {} declares negative max_defer_steps {}",
                qr(&flow.flow_id),
                max_defer
            )));
        }
        if mobility == "pinned" {
            if max_defer != 0 {
                return Err(plan_err(format!(
                    "flow {} is pinned but declares max_defer_steps={}",
                    qr(&flow.flow_id),
                    max_defer
                )));
            }
            if observes != "at_declared_window" {
                return Err(plan_err(format!(
                    "flow {} is pinned but declares observes_version={}",
                    qr(&flow.flow_id),
                    qr(observes)
                )));
            }
        }
        if matches!(mobility, "deferrable" | "free") && max_defer < 1 {
            return Err(plan_err(format!(
                "flow {} is {} but declares max_defer_steps={} (must be >=1)",
                qr(&flow.flow_id),
                mobility,
                max_defer
            )));
        }
        if matches!(mobility, "deferrable" | "free") && deadline == "step_end" {
            return Err(plan_err(format!(
                "flow {} is {} with deadline 'step_end'; deferral is across steps, so this admits no movement",
                qr(&flow.flow_id),
                mobility
            )));
        }
        if mobility == "advanceable" && observes == "at_declared_window" {
            return Err(plan_err(format!(
                "flow {} is advanceable but observes 'at_declared_window'; running earlier can never observe that version",
                qr(&flow.flow_id)
            )));
        }
        if !LEASE_LIFETIMES.contains(&flow.lease_lifetime.as_str()) {
            return Err(plan_err(format!(
                "flow {} declares unsupported lease_lifetime {}; expected one of {}",
                qr(&flow.flow_id),
                qr(&flow.lease_lifetime),
                qr_list(&LEASE_LIFETIMES)
            )));
        }
        if mobility != "pinned" && flow.delivery == "device_lease" {
            if flow.ownership != "runtime_lease" || flow.lease_lifetime != "step" {
                return Err(plan_err(format!(
                    "flow {} uses timing-movable device_lease delivery; this requires ownership='runtime_lease' and lease_lifetime='step'",
                    qr(&flow.flow_id)
                )));
            }
            if deadline != "next_prepare" {
                return Err(plan_err(format!(
                    "flow {} uses a timing-movable step-lifetime device lease but declares deadline {}; deadline 'next_prepare' is required",
                    qr(&flow.flow_id),
                    qr(deadline)
                )));
            }
        }
        if graph_mode != "eager"
            && mobility != "pinned"
            && IN_CAPTURE_HOOKS.contains(&flow.declared_hook.as_str())
        {
            return Err(plan_err(format!(
                "flow {} declares mobility {} at in-capture hook {} under graph mode {}; \
                 timing movement across the capture boundary requires the full placement compiler — \
                 pin the flow or use graph mode 'eager'",
                qr(&flow.flow_id),
                qr(mobility),
                qr(&flow.declared_hook),
                qr(graph_mode)
            )));
        }
        placements.push((flow.flow_id.clone(), flow.declared_hook.clone()));
    }

    let n = flows.len();
    let mut flow_ids: HashSet<&str> = HashSet::with_capacity(n);
    let mut read_sets: Vec<HashSet<&PortAddressDesc>> = Vec::with_capacity(n);
    let mut write_orders: Vec<HashMap<&PortAddressDesc, i64>> = Vec::with_capacity(n);
    for flow in &flows {
        if !flow_ids.insert(flow.flow_id.as_str()) {
            return Err(plan_err(format!(
                "duplicate flow_id {} in placement input",
                qr(&flow.flow_id)
            )));
        }
        read_sets.push(flow.reads.iter().collect());
        let mut orders = HashMap::with_capacity(flow.writes.len());
        for write in &flow.writes {
            if orders.insert(&write.port, write.order).is_some() {
                return Err(plan_err(format!(
                    "flow {} writes port {} more than once",
                    qr(&flow.flow_id),
                    qr(&write.port.label())
                )));
            }
        }
        write_orders.push(orders);
        let mut resources = HashSet::with_capacity(flow.resource_writes.len());
        for write in &flow.resource_writes {
            if !resources.insert(&write.resource) {
                return Err(plan_err(format!(
                    "flow {} writes state resource {} more than once",
                    qr(&flow.flow_id),
                    qr(&write.resource.label())
                )));
            }
        }
    }
    let mut successors = vec![BTreeSet::new(); n];
    let mut predecessors = vec![BTreeSet::new(); n];
    for left in 0..n {
        for right in (left + 1)..n {
            let same_consumer = !flows[left].consumer_id.is_empty()
                && flows[left].consumer_id == flows[right].consumer_id;
            let raw_or_war = write_orders[left]
                .keys()
                .any(|port| read_sets[right].contains(port))
                || write_orders[right]
                    .keys()
                    .any(|port| read_sets[left].contains(port));
            if same_consumer || raw_or_war {
                add_edge(left, right, &mut successors, &mut predecessors);
            }
            for left_write in &flows[left].writes {
                let port = &left_write.port;
                let left_order = left_write.order;
                let Some(right_order) = write_orders[right].get(port) else {
                    continue;
                };
                if left_order == *right_order {
                    return Err(plan_err(format!(
                        "flows {} and {} write port {} with duplicate order {}",
                        qr(&flows[left].flow_id),
                        qr(&flows[right].flow_id),
                        qr(&port.label()),
                        left_order
                    )));
                }
                let (from, to) = if left_order < *right_order {
                    (left, right)
                } else {
                    (right, left)
                };
                add_edge(from, to, &mut successors, &mut predecessors);
            }
            for left_read in &flows[left].resource_reads {
                for right_write in &flows[right].resource_writes {
                    add_state_read_write_edge(
                        left,
                        left_read,
                        right,
                        &right_write.resource,
                        &mut successors,
                        &mut predecessors,
                    );
                }
            }
            for right_read in &flows[right].resource_reads {
                for left_write in &flows[left].resource_writes {
                    add_state_read_write_edge(
                        right,
                        right_read,
                        left,
                        &left_write.resource,
                        &mut successors,
                        &mut predecessors,
                    );
                }
            }
            for left_write in &flows[left].resource_writes {
                let resource = &left_write.resource;
                for right_write in &flows[right].resource_writes {
                    let right_resource = &right_write.resource;
                    if !resource.same_resource(right_resource) {
                        continue;
                    }
                    if resource.version == right_resource.version {
                        return Err(plan_err(format!(
                            "flows {} and {} write state resource {} with duplicate version {}",
                            qr(&flows[left].flow_id),
                            qr(&flows[right].flow_id),
                            qr(&right_resource.label()),
                            resource.version
                        )));
                    }
                    let (from, to) = if resource.version < right_resource.version {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    add_edge(from, to, &mut successors, &mut predecessors);
                }
            }
        }
    }

    let mut ready: BTreeSet<usize> = predecessors
        .iter()
        .enumerate()
        .filter_map(|(index, incoming)| incoming.is_empty().then_some(index))
        .collect();
    let mut remaining_incoming: Vec<usize> = predecessors.iter().map(BTreeSet::len).collect();
    let mut topo = Vec::with_capacity(n);
    while let Some(index) = ready.pop_first() {
        topo.push(index);
        for successor in successors[index].iter().copied() {
            remaining_incoming[successor] -= 1;
            if remaining_incoming[successor] == 0 {
                ready.insert(successor);
            }
        }
    }
    if topo.len() != n {
        let cycle_ids: Vec<&str> = (0..n)
            .filter(|index| remaining_incoming[*index] > 0)
            .map(|index| flows[index].flow_id.as_str())
            .collect();
        return Err(plan_err(format!(
            "placement dependency cycle among flows {}",
            qr_list(&cycle_ids)
        )));
    }
    let execution_order = topo
        .iter()
        .map(|index| flows[*index].flow_id.clone())
        .collect();

    let mut lower = vec![0usize; n];
    let mut upper = vec![HOOK_POINTS.len() - 1; n];
    for (index, flow) in flows.iter().enumerate() {
        let declared = hook_index(&flow.declared_hook).expect("validated declared hook");
        let earliest = flow
            .earliest_hook
            .as_deref()
            .and_then(hook_index)
            .unwrap_or(declared);
        match flow.mobility() {
            "pinned" => {
                lower[index] = declared;
                upper[index] = declared;
            }
            "deferrable" => {
                lower[index] = declared;
            }
            "advanceable" => {
                lower[index] = earliest.min(declared);
                upper[index] = declared;
            }
            "free" => {
                lower[index] = earliest.min(declared);
            }
            _ => unreachable!("validated mobility"),
        }
    }
    for index in topo.iter().rev().copied() {
        for successor in successors[index].iter().copied() {
            upper[index] = upper[index].min(upper[successor]);
        }
    }

    let score_overrides: HashMap<&str, i64> = hook_costs
        .as_ref()
        .map(|costs| {
            costs
                .iter()
                .map(|cost| (cost.hook.as_str(), cost.bubble_score))
                .collect()
        })
        .unwrap_or_default();
    let mut selected = vec![0usize; n];
    for index in topo.iter().copied() {
        for predecessor in predecessors[index].iter().copied() {
            lower[index] = lower[index].max(selected[predecessor]);
        }
        let flow = &flows[index];
        let declared = hook_index(&flow.declared_hook).expect("validated declared hook");
        let allowed: HashSet<&str> = if flow.allowed_hooks.is_empty() {
            HOOK_POINTS.iter().copied().collect()
        } else {
            flow.allowed_hooks.iter().map(String::as_str).collect()
        };
        let movable_step_lease = flow.mobility() != "pinned"
            && flow.delivery == "device_lease"
            && flow.ownership == "runtime_lease"
            && flow.lease_lifetime == "step";
        let mut best: Option<(i64, i64, usize)> = None;
        if lower[index] <= upper[index] {
            for hook_i in lower[index]..=upper[index] {
                let hook = HOOK_POINTS[hook_i];
                if !allowed.contains(hook) {
                    continue;
                }
                if movable_step_lease
                    && hook_i != declared
                    && (flow.declared_hook != "execute_model.post"
                        || !matches!(
                            hook,
                            "sample_tokens.pre" | "sampler.pre_draw" | "sample_tokens.post"
                        ))
                {
                    continue;
                }
                let is_capture = IN_CAPTURE_HOOKS.contains(&hook);
                if graph_mode != "eager" && is_capture && hook_i != declared {
                    continue;
                }
                let bubble = score_overrides
                    .get(hook)
                    .copied()
                    .unwrap_or_else(|| default_bubble_score(hook));
                let distance = declared.abs_diff(hook_i) as i64;
                let score = (bubble, -distance, hook_i);
                if best.is_none_or(|current| score > current) {
                    best = Some(score);
                }
            }
        }
        let Some((_benefit, _distance, hook_i)) = best else {
            return Err(plan_err(format!(
                "flow {} has no legal placement after dependency, mobility, graph, and backend-hook constraints",
                qr(&flow.flow_id)
            )));
        };
        selected[index] = hook_i;
        placements[index].1 = HOOK_POINTS[hook_i].to_string();
    }
    let mut fusion_groups: Vec<Vec<String>> = Vec::new();
    let mut current_group: Vec<String> = Vec::new();
    let mut current_hook: Option<&str> = None;
    let mut fusion_rejections: Vec<(String, String)> = Vec::new();
    for index in topo.iter().copied() {
        let flow = &flows[index];
        let side_stream_candidate = flow.fusion_kind == "side_stream_cudagraph";
        if !side_stream_candidate && !flow.writes.iter().any(|write| write.fusion_candidate) {
            if !current_group.is_empty() {
                fusion_groups.push(std::mem::take(&mut current_group));
                current_hook = None;
            }
            continue;
        }
        let invalid_mode = !side_stream_candidate
            && flow.writes.iter().any(|write| {
                write.fusion_candidate && !matches!(write.mode.as_str(), "replace" | "add")
            });
        let supported_effect_hook = flow
            .effect_hook
            .as_deref()
            .map(|hook| {
                if side_stream_candidate {
                    hook == "sampler.pre_draw"
                } else {
                    IN_CAPTURE_HOOKS.contains(&hook)
                }
            })
            .unwrap_or(false);
        let reason = if invalid_mode {
            Some("fusion candidates require replace/add writes")
        } else if flow.delivery != "device_lease" || flow.ownership != "runtime_lease" {
            Some("fusion candidates require stable runtime-owned device leases")
        } else if flow.max_bundle_rows <= 0 {
            Some("fusion candidates require a static max_bundle_rows shape bucket")
        } else if graph_mode == "eager" {
            Some("graph mode eager forces the compiler fallback path")
        } else if !supported_effect_hook {
            Some("fusion candidate has no supported execution hook")
        } else {
            None
        };
        if let Some(reason) = reason {
            if !current_group.is_empty() {
                fusion_groups.push(std::mem::take(&mut current_group));
                current_hook = None;
            }
            fusion_rejections.push((flow.flow_id.clone(), reason.to_string()));
            continue;
        }
        let hook = flow.effect_hook.as_deref().expect("validated effect hook");
        if current_hook != Some(hook) && !current_group.is_empty() {
            fusion_groups.push(std::mem::take(&mut current_group));
        }
        current_hook = Some(hook);
        current_group.push(flow.flow_id.clone());
    }
    if !current_group.is_empty() {
        fusion_groups.push(current_group);
    }
    Ok(PlacementSchedule {
        placements,
        execution_order,
        fusion_groups,
        fusion_rejections,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_placement_schedule, m)?)?;
    m.add_class::<PlacementSchedule>()?;
    Ok(())
}

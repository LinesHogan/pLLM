//! Exact small-graph partition + placement over the original semantic graph.
//!
//! The memory witness is an anchor envelope: boundary values remain live for
//! the whole producer-through-last-consumer anchor interval. Regions execute
//! serially at an anchor, so their private scratch combines by maximum.
//! Cold task allocations (including captured graph pools) remain resident at
//! every anchor and sum across tasks, independently of region fusion.

use super::scheduler::{GraphIndex, PhysicalTask, ValueTransfer};
use super::semantic_graph::{
    ImplementationCandidate, InputSource, OutputKind, SemanticGraph, Timeline,
};
use super::state_representation::StatePhysicalPlan;
use std::collections::BTreeSet;

struct ValueUse {
    producer: usize,
    users: BTreeSet<usize>,
    bytes: u64,
    exported: bool,
}

pub(super) struct Solution {
    pub(super) regions: Vec<(Vec<usize>, u32)>,
    pub(super) materialized_values: Vec<u32>,
    pub(super) score: [u64; 5],
    pub(super) domains: Vec<u32>,
    pub(super) transfers: Vec<ValueTransfer>,
    pub(super) release_values: Vec<Vec<u32>>,
    slots: Vec<u32>,
}

struct Search<'a> {
    graph: &'a SemanticGraph,
    timeline: &'a Timeline,
    index: &'a GraphIndex,
    implementations: &'a [&'a ImplementationCandidate],
    state: Option<&'a StatePhysicalPlan>,
    values: Vec<ValueUse>,
    workspace: Vec<u64>,
    domains: Vec<Vec<u32>>,
    best: Option<Solution>,
}

fn prepare<'a>(
    graph: &'a SemanticGraph,
    timeline: &'a Timeline,
    index: &'a GraphIndex,
    implementations: &'a [&'a ImplementationCandidate],
    state: Option<&'a StatePhysicalPlan>,
) -> Result<Search<'a>, String> {
    if (!timeline.reserved_bytes.is_empty()
        && timeline.reserved_bytes.len() != timeline.anchors.len())
        || timeline
            .task_resources
            .windows(2)
            .any(|pair| pair[0].task >= pair[1].task)
        || timeline
            .value_resources
            .windows(2)
            .any(|pair| (&pair[0].task, &pair[0].output) >= (&pair[1].task, &pair[1].output))
    {
        return Err(
            "AER resource facts require exact anchor coverage and sorted unique identities".into(),
        );
    }
    let mut workspace = vec![0; graph.tasks.len()];
    let mut domains = vec![vec![0]; graph.tasks.len()];
    for fact in &timeline.task_resources {
        let task = graph
            .tasks
            .binary_search_by(|task| task.symbol.cmp(&fact.task))
            .map_err(|_| "AER workspace references an unknown task")?;
        workspace[task] = fact.workspace_bytes;
        domains[task] = fact.domains.clone();
    }
    let mut values = Vec::new();
    for (producer, task) in graph.tasks.iter().enumerate() {
        for output in &task.outputs {
            let fact = timeline
                .value_resources
                .iter()
                .find(|fact| fact.task == task.symbol && fact.output == output.name);
            if (timeline.memory_budget_bytes.is_some() || !timeline.domains.is_empty()) && fact.is_none() {
                return Err("memory-constrained AER requires every output size, including explicit zero-byte values".into());
            }
            let mut users: BTreeSet<_> = graph.tasks.iter().enumerate().filter_map(|(user, task)|
                task.inputs.iter().any(|input| matches!(&input.source,
                    InputSource::Result { task, output: name } if task == &graph.tasks[producer].symbol && name == &output.name))
                    .then_some(user)).collect();
            // Autograd may save a consumer's inputs until its downstream gradient sink.
            for &user in &index.order {
                if graph.tasks[user].math.as_ref().is_some_and(|math|
                    graph.tasks[user].inputs.iter().any(|input| math.input_gradients.contains(&input.name)
                        && matches!(&input.source, InputSource::Result { task, .. }
                            if users.iter().any(|&parent| &graph.tasks[parent].symbol == task)))) {
                    users.insert(user);
                }
            }
            values.push(ValueUse {
                producer,
                users,
                bytes: fact.map_or(0, |fact| fact.bytes),
                exported: output.kind != OutputKind::Temporary,
            });
        }
    }
    if (timeline.memory_budget_bytes.is_some() || !timeline.domains.is_empty()) && timeline.task_resources.len() != graph.tasks.len()
    {
        return Err("memory-constrained AER requires every task workspace size".into());
    }
    for fact in &timeline.value_resources {
        if !graph.tasks.iter().any(|task| {
            task.symbol == fact.task && task.outputs.iter().any(|output| output.name == fact.output)
        }) {
            return Err("AER value size references an unknown output".into());
        }
    }
    if let Some(state) = state {
        for provider in &state.representations {
            if provider.execution_domain as usize >= timeline.domains.len().max(1)
                || timeline.domains.get(provider.execution_domain as usize).is_some_and(|domain|
                    domain.device_kind != provider.device_kind || domain.device_ordinal != provider.device_ordinal)
            {
                return Err("state provider differs from its signed execution domain".into());
            }
        }
    } else if !timeline.domains.is_empty() && !graph.states.is_empty() {
        return Err("distributed state tasks require their original selected provider plans".into());
    }
    for (id, task) in graph.tasks.iter().enumerate() {
        domains[id].retain(|&domain| {
            timeline.domains.get(domain as usize).is_none_or(|facts|
                implementations[id].required_capabilities.iter().all(|cap| facts.capabilities.binary_search(cap).is_ok()))
                && state.is_none_or(|state| state.representations.iter().all(|provider| {
                    let uses = task.effects.iter().any(|effect| effect.resource == provider.resource)
                        || task.inputs.iter().any(|input| matches!(&input.source,
                            InputSource::StateSelection { resource, .. } if resource == &provider.resource));
                    !uses || provider.execution_domain == domain
                }))
        });
        if domains[id].is_empty() {
            return Err("task has no execution domain covering its implementation and exact state".into());
        }
    }
    values
        .iter()
        .map(|value| value.bytes)
        .chain(workspace.iter().copied())
        .chain(timeline.task_resources.iter().map(|fact| fact.resident_bytes))
        .chain(timeline.reserved_bytes.iter().copied())
        .chain(timeline.domains.iter().flat_map(|domain| domain.reserved_bytes.iter().copied()))
        .chain(state.map(|plan| plan.reserved_bytes))
        .try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or("AER resource envelope overflows u64")?;
    Ok(Search {
        graph,
        timeline,
        index,
        implementations,
        state,
        values,
        workspace,
        domains,
        best: None,
    })
}

pub(super) fn solve(
    graph: &SemanticGraph,
    timeline: &Timeline,
    index: &GraphIndex,
    implementations: &[&ImplementationCandidate],
    state: Option<&StatePhysicalPlan>,
) -> Result<Solution, String> {
    let mut search = prepare(graph, timeline, index, implementations, state)?;
    search.partition(0, &mut Vec::new(), &mut vec![usize::MAX; graph.tasks.len()]);
    search.best.ok_or_else(|| {
        "no AER partition and placement fits dependencies, windows and memory envelope".into()
    })
}

pub(super) fn check_placement(
    graph: &SemanticGraph,
    timeline: &Timeline,
    index: &GraphIndex,
    implementations: &[&ImplementationCandidate],
    state: Option<&StatePhysicalPlan>,
    tasks: &[PhysicalTask],
    region_ends: &[u32],
) -> Result<Solution, String> {
    if tasks.len() != graph.tasks.len() || region_ends.last().copied() != Some(tasks.len() as u32) {
        return Err("AER placement does not cover every task".into());
    }
    let mut search = prepare(graph, timeline, index, implementations, state)?;
    let mut groups = Vec::new();
    let mut membership = vec![usize::MAX; tasks.len()];
    let mut positions = vec![usize::MAX; tasks.len()];
    let mut task_slots = vec![0; tasks.len()];
    let mut slots = Vec::new();
    let mut domains = Vec::new();
    let mut start = 0;
    for &end in region_ends {
        let end = end as usize;
        if end <= start || end > tasks.len() {
            return Err("AER regions must be nonempty contiguous ranges".into());
        }
        let slot = tasks[start].anchor_id;
        let domain = tasks[start].domain_id;
        let mut members = Vec::new();
        for (position, task) in tasks.iter().enumerate().take(end).skip(start) {
            let id = task.task_id as usize;
            if id >= tasks.len() || membership[id] != usize::MAX || task.anchor_id != slot
                || task.domain_id != domain || !search.domains[id].contains(&domain)
                || slot < index.windows[id].0 || slot >= index.windows[id].1
            {
                return Err("AER task identity or timing differs from its declared region".into());
            }
            if end - start > 1 && (graph.tasks[id].replay.is_some()
                || implementations[id].execution_kind != implementations[tasks[start].task_id as usize].execution_kind)
            {
                return Err("AER region crosses an execution or replay boundary".into());
            }
            membership[id] = groups.len();
            positions[id] = position;
            task_slots[id] = slot;
            members.push(id);
        }
        slots.push(slot);
        domains.push(domain);
        groups.push(members);
        start = end;
    }
    let mut predecessors = vec![BTreeSet::new(); groups.len()];
    for (task, parents) in index.predecessors.iter().enumerate() {
        for &parent in parents {
            if positions[parent] >= positions[task] || task_slots[parent] > task_slots[task] {
                return Err("AER placement violates an exact-version or result dependency".into());
            }
            if membership[parent] != membership[task] {
                predecessors[membership[task]].insert(membership[parent]);
            }
        }
    }
    if slots.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("AER regions are not in execution order".into());
    }
    search.evaluate(&groups, &membership, &(0..groups.len()).collect::<Vec<_>>(), &predecessors, &slots, &domains);
    search.best.ok_or_else(|| "selected AER placement exceeds its memory envelope".into())
}

impl Search<'_> {
    fn partition(
        &mut self,
        next: usize,
        groups: &mut Vec<Vec<usize>>,
        membership: &mut Vec<usize>,
    ) {
        let cut_lower_bound: u64 = self
            .values
            .iter()
            .filter(|value| {
                value.exported
                    || (value.producer < next
                        && value.users.iter().any(|&user| {
                            user < next && membership[user] != membership[value.producer]
                        }))
            })
            .map(|value| value.bytes)
            .sum();
        if self.best.as_ref().is_some_and(|best| {
            cut_lower_bound > best.score[0]
        }) {
            return;
        }
        if next == self.graph.tasks.len() {
            self.place_partition(groups, membership);
            return;
        }
        // Restricted-growth membership enumerates each partition once. Conditional
        // tasks are closed control regions and cannot absorb unconditional work.
        for group in 0..groups.len() {
            let first = groups[group][0];
            if self.graph.tasks[next].replay.is_some()
                || self.graph.tasks[first].replay.is_some()
                || self.implementations[next].execution_kind
                    != self.implementations[first].execution_kind
            {
                continue;
            }
            let lo = groups[group]
                .iter()
                .map(|&task| self.index.windows[task].0)
                .max()
                .unwrap()
                .max(self.index.windows[next].0);
            let hi = groups[group]
                .iter()
                .map(|&task| self.index.windows[task].1)
                .min()
                .unwrap()
                .min(self.index.windows[next].1);
            if lo >= hi {
                continue;
            }
            if !self.domains[next].iter().any(|domain|
                groups[group].iter().all(|&task| self.domains[task].contains(domain))) {
                continue;
            }
            groups[group].push(next);
            membership[next] = group;
            self.partition(next + 1, groups, membership);
            groups[group].pop();
        }
        membership[next] = groups.len();
        groups.push(vec![next]);
        self.partition(next + 1, groups, membership);
        groups.pop();
        membership[next] = usize::MAX;
    }

    fn place_partition(&mut self, groups: &[Vec<usize>], membership: &[usize]) {
        let mut predecessors = vec![BTreeSet::new(); groups.len()];
        for (task, parents) in self.index.predecessors.iter().enumerate() {
            for &parent in parents {
                if membership[parent] != membership[task] {
                    predecessors[membership[task]].insert(membership[parent]);
                }
            }
        }
        let mut order = Vec::new();
        let mut pending: BTreeSet<_> = (0..groups.len()).collect();
        while let Some(group) = pending
            .iter()
            .find(|&&group| {
                predecessors[group]
                    .iter()
                    .all(|parent| !pending.contains(parent))
            })
            .copied()
        {
            pending.remove(&group);
            order.push(group);
        }
        if !pending.is_empty() {
            return;
        }
        let bounds: Vec<_> = groups
            .iter()
            .map(|members| {
                (
                    members
                        .iter()
                        .map(|&task| self.index.windows[task].0)
                        .max()
                        .unwrap(),
                    members
                        .iter()
                        .map(|&task| self.index.windows[task].1)
                        .min()
                        .unwrap(),
                )
            })
            .collect();
        self.placements(
            groups,
            membership,
            &order,
            &predecessors,
            &bounds,
            &mut vec![0; groups.len()],
            &mut vec![0; groups.len()],
            0,
        );
    }

    fn placements(
        &mut self,
        groups: &[Vec<usize>],
        membership: &[usize],
        order: &[usize],
        predecessors: &[BTreeSet<usize>],
        bounds: &[(u32, u32)],
        slots: &mut [u32],
        domains: &mut [u32],
        next: usize,
    ) {
        if next != order.len() {
            let group = order[next];
            let after = bounds[group].0.max(
                predecessors[group]
                    .iter()
                    .map(|&parent| slots[parent])
                    .max()
                    .unwrap_or(0),
            );
            for domain in self.domains[groups[group][0]].clone() {
                if !groups[group].iter().all(|&task| self.domains[task].contains(&domain)) {
                    continue;
                }
                domains[group] = domain;
                for anchor in after..bounds[group].1 {
                    slots[group] = anchor;
                    self.placements(groups, membership, order, predecessors, bounds, slots, domains, next + 1);
                }
            }
            return;
        }
        self.evaluate(groups, membership, order, predecessors, slots, domains);
    }

    fn evaluate(
        &mut self,
        groups: &[Vec<usize>],
        membership: &[usize],
        order: &[usize],
        predecessors: &[BTreeSet<usize>],
        slots: &[u32],
        domains: &[u32],
    ) {
        let count = self.timeline.domains.len().max(1);
        let anchors = self.timeline.anchors.len();
        let mut internal = vec![0_u128; groups.len()];
        let mut live = vec![vec![0_u128; anchors]; count];
        let mut materialized_values = Vec::new();
        let mut transfers = Vec::new();
        let mut materialization = 0_u128;
        let mut transfer_cost = 0_u128;
        let mut placed_order = order.to_vec();
        placed_order.sort_by_key(|&group| slots[group]);
        let execution_order: Vec<_> = placed_order.iter().flat_map(|&group|
            self.index.order.iter().copied().filter(move |&task| membership[task] == group)).collect();
        let mut positions = vec![0; membership.len()];
        for (position, &task) in execution_order.iter().enumerate() { positions[task] = position; }
        let mut release_values = vec![Vec::new(); membership.len()];
        for (value_id, value) in self.values.iter().enumerate() {
            let source = membership[value.producer];
            let source_domain = domains[source] as usize;
            if !value.exported {
                let last = value.users.iter().copied().chain(std::iter::once(value.producer))
                    .max_by_key(|&task| positions[task]).unwrap();
                release_values[last].push(value_id as u32);
            }
            if value.exported || value.users.iter().any(|&user| membership[user] != source) {
                materialized_values.push(value_id as u32);
                materialization += value.bytes as u128;
                let last = if value.exported {
                    anchors as u32 - 1
                } else {
                    value
                        .users
                        .iter()
                        .map(|&user| slots[membership[user]])
                        .max()
                        .unwrap_or(slots[source])
                };
                for bytes in &mut live[source_domain][slots[source] as usize..=last as usize] {
                    *bytes += value.bytes as u128;
                }
                let targets: BTreeSet<_> = value.users.iter().map(|&user| domains[membership[user]])
                    .filter(|&domain| domain as usize != source_domain).collect();
                for target in targets {
                    let Some(link) = self.timeline.transfer_links.iter().find(|link|
                        link.source as usize == source_domain && link.target == target) else { return; };
                    let users: Vec<_> = value.users.iter().copied()
                        .filter(|&user| domains[membership[user]] == target).collect();
                    let first = *users.iter().min_by_key(|&&user| positions[user]).unwrap();
                    if (slots[membership[first]] as u64) < slots[source] as u64 + link.anchor_delay as u64 {
                        return;
                    }
                    let last = users.iter().map(|&user| slots[membership[user]]).max().unwrap();
                    // A destination owns one copy shared by all of its consumers.
                    // Reserve it throughout the producer-to-last-reader envelope.
                    for bytes in &mut live[target as usize][slots[source] as usize..=last as usize] {
                        *bytes += value.bytes as u128;
                    }
                    materialization += value.bytes as u128;
                    transfer_cost = transfer_cost.saturating_add(value.bytes as u128 * link.cost_per_byte as u128);
                    transfers.push(ValueTransfer { value_id: value_id as u32, source: source_domain as u32,
                        target, after_task: value.producer as u32, before_task: first as u32 });
                }
            } else {
                internal[source] += value.bytes as u128;
            }
        }
        let mut scratch = vec![vec![0_u128; anchors]; count];
        for (group, members) in groups.iter().enumerate() {
            let bytes = internal[group]
                + members
                    .iter()
                    .map(|&task| self.workspace[task])
                    .max()
                    .unwrap() as u128;
            let slot = &mut scratch[domains[group] as usize][slots[group] as usize];
            *slot = (*slot).max(bytes);
        }
        let mut resident = vec![0_u128; count];
        for fact in &self.timeline.task_resources {
            let task = self.graph.tasks.binary_search_by(|task| task.symbol.cmp(&fact.task)).unwrap();
            resident[domains[membership[task]] as usize] += fact.resident_bytes as u128;
        }
        if let Some(state) = self.state {
            for provider in &state.representations {
                resident[provider.execution_domain as usize] += provider.reserved_bytes as u128;
            }
        }
        let mut peak = 0_u128;
        for anchor in 0..anchors {
            let mut total = self.timeline.reserved_bytes.get(anchor).copied().unwrap_or(0) as u128;
            for domain in 0..count {
                let facts = self.timeline.domains.get(domain);
                let used = live[domain][anchor] + scratch[domain][anchor] + resident[domain]
                    + facts.and_then(|facts| facts.reserved_bytes.get(anchor)).copied().unwrap_or(0) as u128;
                if facts.and_then(|facts| facts.memory_budget_bytes).is_some_and(|budget| used > budget as u128) {
                    return;
                }
                total += used;
            }
            peak = peak.max(total);
        }
        if self.timeline.memory_budget_bytes.is_some_and(|budget| peak > budget as u128) { return; }
        let (Ok(materialization), Ok(transfer_cost), Ok(peak)) =
            (u64::try_from(materialization), u64::try_from(transfer_cost), u64::try_from(peak)) else { return; };
        let score = [
            materialization,
            transfer_cost,
            groups.len() as u64,
            peak,
            predecessors.iter().map(|p| p.len() as u64).sum::<u64>() + transfers.len() as u64,
        ];
        let task_slots: Vec<_> = membership.iter().map(|&group| slots[group]).collect();
        let task_domains: Vec<_> = membership.iter().map(|&group| domains[group]).collect();
        if self
            .best
            .as_ref()
            .is_some_and(|best| (best.score, &best.slots, &best.domains) <= (score, &task_slots, &task_domains))
        {
            return;
        }
        let regions = placed_order
            .iter()
            .map(|&group| {
                (
                    self.index
                        .order
                        .iter()
                        .copied()
                        .filter(|&task| membership[task] == group)
                        .collect(),
                    slots[group],
                )
            })
            .collect();
        self.best = Some(Solution {
            regions,
            materialized_values,
            score,
            slots: task_slots,
            domains: task_domains,
            transfers,
            release_values,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::scheduler::compile;
    use super::super::semantic_graph::{decode_graph, decode_timeline};
    use serde_json::json;
    use std::collections::BTreeSet;

    #[test]
    fn aer_matches_bruteforce_partition_oracle() {
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let windows = [(0, 1), (0, 3), (1, 3)];
        let bytes = [3_u64, 5, 7];
        let workspace = [2_u64, 4, 1];
        for mask in 0..8 {
            let arcs: Vec<_> = [(0, 1), (0, 2), (1, 2)]
                .into_iter()
                .enumerate()
                .filter_map(|(bit, arc)| (mask & (1 << bit) != 0).then_some(arc))
                .collect();
            for window_code in 0..27 {
                let bounds = [
                    windows[window_code % 3],
                    windows[window_code / 3 % 3],
                    windows[window_code / 9],
                ];
                for mixed_kind in [false, true] {
                    for budget in [8_u64, 16, 40] {
                        let reserved = [0_u64, 9, 1, 0];
                        let resident = if mixed_kind { [2_u64, 5, 1] } else { [0; 3] };
                        let graph = decode_graph(&json!({
                "workload_id": "aer_oracle", "execution_parameters_digest": "a".repeat(64),
                "schema_version": 1, "edges": [], "states": [],
                "tasks": (0..3).map(|task| json!({
                    "symbol": format!("t{task}"),
                    "timing": {"after": format!("a{}", bounds[task].0), "before": format!("a{}", bounds[task].1)},
                    "inputs": arcs.iter().filter(|(_, target)| *target == task).map(|(source, _)| json!({
                        "name": format!("from{source}"), "source": {"kind": "result", "task": format!("t{source}"), "output": "value"}
                    })).collect::<Vec<_>>(),
                    "outputs": [{"name": "value", "kind": "temporary", "schema": ""}], "effects": [],
                    "implementations": [{"implementation_id": "kernel", "execution_kind":
                        if mixed_kind && task == 1 {"host_service"} else {"native_task"}, "required_capabilities": []}]
                })).collect::<Vec<_>>()
              }).to_string()).unwrap();
                        let timeline = decode_timeline(&json!({
                "schema_version": 1, "capabilities": [],
                "anchors": (0..4).map(|anchor| json!({"symbol": format!("a{anchor}"), "ordinal": anchor})).collect::<Vec<_>>(),
                "reserved_bytes": reserved, "memory_budget_bytes": budget,
                "value_resources": (0..3).map(|task| json!({"task": format!("t{task}"), "output": "value", "bytes": bytes[task]})).collect::<Vec<_>>(),
                "task_resources": (0..3).map(|task| json!({"task": format!("t{task}"), "workspace_bytes": workspace[task],
                    "resident_bytes": resident[task]})).collect::<Vec<_>>()
              }).to_string()).unwrap();
                        let mut expected = None;
                        // Deliberately enumerate labelled partitions and per-task placements,
                        // not the production restricted-growth or region-placement recursion.
                        for labels in 0..27 {
                            let group = [labels % 3, labels / 3 % 3, labels / 9];
                            let count = group.iter().copied().collect::<BTreeSet<_>>().len();
                            if (0..3).any(|a| {
                                (0..3).any(|b| {
                                    group[a] == group[b] && mixed_kind && (a == 1) != (b == 1)
                                })
                            }) {
                                continue;
                            }
                            let legal_order = permutations.iter().any(|order| {
                                let rank =
                                    |task| order.iter().position(|&known| known == task).unwrap();
                                arcs.iter().all(|&(a, b)| rank(a) < rank(b))
                                    && !(group[order[0]] == group[order[2]]
                                        && group[order[0]] != group[order[1]])
                            });
                            if !legal_order {
                                continue;
                            }
                            for assignment in 0..27 {
                                let at = [assignment % 3, assignment / 3 % 3, assignment / 9];
                                if (0..3).any(|task| {
                                    at[task] < bounds[task].0 || at[task] >= bounds[task].1
                                }) || arcs.iter().any(|&(a, b)| at[a] > at[b])
                                    || (0..3).any(|a| {
                                        (0..3).any(|b| group[a] == group[b] && at[a] != at[b])
                                    })
                                {
                                    continue;
                                }
                                let cut: Vec<_> = (0..3)
                                    .filter(|&producer| {
                                        arcs.iter()
                                            .any(|&(a, b)| a == producer && group[a] != group[b])
                                    })
                                    .collect();
                                let mut peak = 0;
                                for anchor in 0..4 {
                                    let live: u64 = cut
                                        .iter()
                                        .filter(|&&producer| {
                                            at[producer] <= anchor
                                                && anchor
                                                    <= arcs
                                                        .iter()
                                                        .filter(|&&(a, _)| a == producer)
                                                        .map(|&(_, b)| at[b])
                                                        .max()
                                                        .unwrap()
                                        })
                                        .map(|&producer| bytes[producer])
                                        .sum();
                                    let scratch = (0..3)
                                        .filter(|&member| at[member] == anchor)
                                        .map(|member| {
                                            let internal: u64 = (0..3)
                                                .filter(|task| {
                                                    group[*task] == group[member]
                                                        && !cut.contains(task)
                                                })
                                                .map(|task| bytes[task])
                                                .sum();
                                            internal
                                                + (0..3)
                                                    .filter(|&task| group[task] == group[member])
                                                    .map(|task| workspace[task])
                                                    .max()
                                                    .unwrap()
                                        })
                                        .max()
                                        .unwrap_or(0);
                                    peak = peak.max(reserved[anchor as usize] + live + scratch + resident.iter().sum::<u64>());
                                }
                                if peak > budget {
                                    continue;
                                }
                                let fences = arcs
                                    .iter()
                                    .filter_map(|&(a, b)| {
                                        (group[a] != group[b]).then_some((group[a], group[b]))
                                    })
                                    .collect::<BTreeSet<_>>()
                                    .len() as u64;
                                let score = [
                                    cut.iter().map(|&producer| bytes[producer]).sum(),
                                    count as u64,
                                    peak,
                                    fences,
                                ];
                                let candidate = (score, at);
                                if expected.as_ref().is_none_or(|best| candidate < *best) {
                                    expected = Some(candidate);
                                }
                            }
                        }
                        let actual = compile(&graph, &timeline, None);
                        assert_eq!(
                            actual.is_ok(),
                            expected.is_some(),
                            "mask={mask}, bounds={bounds:?}, budget={budget}"
                        );
                        if let Some((score, slots)) = expected {
                            let wire = serde_json::to_value(actual.unwrap()).unwrap();
                            let mut actual_slots = [0; 3];
                            for task in wire["tasks"].as_array().unwrap() {
                                actual_slots[task["task_id"].as_u64().unwrap() as usize] =
                                    task["anchor_id"].as_u64().unwrap() as u32;
                            }
                            assert_eq!(actual_slots, slots);
                            assert_eq!(
                                [
                                    wire["materialization_bytes"].as_u64().unwrap(),
                                    wire["region_ends"].as_array().unwrap().len() as u64,
                                    wire["peak_memory_bytes"].as_u64().unwrap(),
                                    wire["fence_count"].as_u64().unwrap()
                                ],
                                score
                            );
                        }
                    }
                }
            }
        }
    }
}

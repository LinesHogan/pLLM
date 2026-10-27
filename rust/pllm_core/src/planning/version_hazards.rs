//! SSA state-version dependencies over original graph and representation facts.

use super::semantic_graph::{SemanticEffect, SemanticGraph, StateEffectKind};
use super::state_representation::StatePhysicalPlan;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn dependencies(
    graph: &SemanticGraph,
    state: Option<&StatePhysicalPlan>,
    predecessors: &mut [BTreeSet<usize>],
) -> Result<(), String> {
    for resource in &graph.states {
        let effects: BTreeMap<_, _> = graph
            .tasks
            .iter()
            .enumerate()
            .filter_map(|(task, node)| {
                node.effects
                    .iter()
                    .find(|effect| effect.resource == resource.resource)
                    .map(|effect| (task, effect))
            })
            .collect();
        let source = |effect: &SemanticEffect| -> Result<Option<usize>, String> {
            effect
                .version
                .as_ref()
                .map(|symbol| {
                    let task = graph
                        .tasks
                        .binary_search_by(|node| node.symbol.cmp(symbol))
                        .map_err(|_| "state version references an unknown task")?;
                    if effects
                        .get(&task)
                        .is_none_or(|effect| effect.kind == StateEffectKind::Read)
                    {
                        return Err("state version source does not produce this resource".into());
                    }
                    Ok(task)
                })
                .transpose()
        };
        let mut publications = BTreeMap::new();
        for (&task, effect) in &effects {
            if let Some(producer) = source(effect)? {
                if producer == task {
                    return Err("state version cannot consume itself".into());
                }
                predecessors[task].insert(producer);
            }
            if effect.kind == StateEffectKind::Publish {
                let producer =
                    source(effect)?.ok_or("publication requires its private transition version")?;
                if !matches!(
                    effects[&producer].kind,
                    StateEffectKind::Transition | StateEffectKind::TransitionMaybe
                ) || publications.insert(producer, task).is_some()
                {
                    return Err("private state version requires exactly one publication".into());
                }
            }
        }
        // A publication aliases its writer's value; it is not another version.
        let origin = |effect: &SemanticEffect| -> Result<Option<usize>, String> {
            match source(effect)? {
                Some(producer) if effects[&producer].kind == StateEffectKind::Publish => {
                    source(effects[&producer])
                }
                value => Ok(value),
            }
        };
        let mut commits = BTreeMap::new();
        for (&writer, effect) in &effects {
            let commit = match effect.kind {
                StateEffectKind::Transition | StateEffectKind::TransitionMaybe => *publications
                    .get(&writer)
                    .ok_or("private transition has no publication")?,
                StateEffectKind::Update | StateEffectKind::UpdateMaybe => writer,
                StateEffectKind::Read | StateEffectKind::Publish => continue,
            };
            if commits.insert(origin(effect)?, (writer, commit)).is_some() {
                return Err(
                    "WAW: two commits consume the same state version; declare a version chain"
                        .into(),
                );
            }
        }
        for (&before, &(writer, _)) in &commits {
            if let Some(previous) = before {
                predecessors[writer]
                    .insert(publications.get(&previous).copied().unwrap_or(previous));
            }
        }
        let provider = state.and_then(|state| {
            state
                .representations
                .iter()
                .find(|provider| provider.resource == resource.resource)
        });
        let retains_versions = provider.is_some_and(|provider| {
            provider
                .capabilities
                .iter()
                .any(|capability| capability == "state/exact_versions@1")
        });
        if retains_versions && !commits.is_empty() {
            let provider = provider.unwrap();
            // Minimum simultaneous current/private roots, not a ceiling on retained versions.
            if provider
                .max_admitted_lineages
                .checked_mul(2)
                .is_none_or(|minimum| provider.page_capacity < minimum)
            {
                return Err(
                    "state pool cannot hold current and private roots for admitted writers".into(),
                );
            }
        }
        if !retains_versions {
            // Mutable host state must finish old-version reads before overwrite.
            for (&reader, effect) in &effects {
                if let Some(&(_, commit)) = commits.get(&origin(effect)?) {
                    if reader != commit && effect.kind == StateEffectKind::Read {
                        predecessors[commit].insert(reader);
                    }
                }
            }
        }
    }
    Ok(())
}

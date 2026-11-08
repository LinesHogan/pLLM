//! SSA state-version dependencies over original graph and representation facts.

use super::semantic_graph::{SemanticEffect, SemanticGraph, StateEffectKind};
use super::state_representation::StateRepresentationDecision;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn dependencies(
    graph: &SemanticGraph,
    state: Option<&StateRepresentationDecision>,
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
        // pllm sets every version: a publication names its transition, a model read the publication.
        let source = |effect: &SemanticEffect| -> Option<usize> {
            effect.version.as_ref().map(|symbol| {
                graph
                    .tasks
                    .binary_search_by(|node| node.symbol.cmp(symbol))
                    .expect("state versions name declared tasks")
            })
        };
        let mut publications = BTreeMap::new();
        for (&task, effect) in &effects {
            if let Some(producer) = source(effect) {
                predecessors[task].insert(producer);
                if effect.kind == StateEffectKind::Publish {
                    publications.insert(producer, task);
                }
            }
        }
        // A publication aliases its writer's value; it is not another version.
        let origin = |effect: &SemanticEffect| match source(effect) {
            Some(producer) if effects[&producer].kind == StateEffectKind::Publish => {
                source(effects[&producer])
            }
            value => value,
        };
        let mut commits = BTreeMap::new();
        for (&writer, effect) in &effects {
            let commit = match effect.kind {
                StateEffectKind::Transition | StateEffectKind::TransitionMaybe => {
                    publications[&writer]
                }
                StateEffectKind::Update | StateEffectKind::UpdateMaybe => writer,
                StateEffectKind::Read | StateEffectKind::Publish => continue,
            };
            if let Some((other, _)) = commits.insert(origin(effect), (writer, commit)) {
                return Err(format!(
                    "{} and {} both write state {:?} from the same version; let one method write it",
                    graph.tasks[other].symbol, graph.tasks[writer].symbol, resource.resource
                ));
            }
        }
        for (&before, &(writer, _)) in &commits {
            if let Some(previous) = before {
                predecessors[writer]
                    .insert(publications.get(&previous).copied().unwrap_or(previous));
            }
        }
        let retains_versions = state
            .and_then(|state| (state.plans.iter()).find(|plan| plan.resource == resource.resource))
            .is_some_and(|plan| {
                (plan.capabilities.iter()).any(|capability| capability == "state/exact_versions@1")
            });
        if !retains_versions {
            // Mutable host state must finish old-version reads before overwrite.
            for (&reader, effect) in &effects {
                if let Some(&(_, commit)) = commits.get(&origin(effect)) {
                    if reader != commit && effect.kind == StateEffectKind::Read {
                        predecessors[commit].insert(reader);
                    }
                }
            }
        }
    }
    Ok(())
}

"""Metadata-only retained-plan reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tllm.workgraph.ir import CompiledWorkGraph, NodeKey
from tllm.workgraph.signature import structural_signature


@dataclass(frozen=True, slots=True)
class KeepRegion:
    node_keys: tuple[NodeKey, ...]
    reason: str = "structural_signature_unchanged"


@dataclass(frozen=True, slots=True)
class RebuildRegion:
    node_keys: tuple[NodeKey, ...]
    reason: str = "structural_signature_changed"


@dataclass(frozen=True, slots=True)
class PlanDiff:
    previous_signature: str | None
    current_signature: str
    previous_epoch: int | None
    epoch: int
    keep_regions: tuple[KeepRegion, ...] = ()
    rebuild_regions: tuple[RebuildRegion, ...] = ()

    @property
    def changed(self) -> bool:
        return self.previous_signature != self.current_signature

    @property
    def keep(self) -> tuple[KeepRegion, ...]:
        return self.keep_regions

    @property
    def rebuild(self) -> tuple[RebuildRegion, ...]:
        return self.rebuild_regions

    def to_dict(self) -> dict[str, Any]:
        # PlanDiff is report metadata, rather than part of the closed-world
        # structural IR serializer.  Render its stable fields explicitly so
        # explain() remains JSON-safe without admitting arbitrary dataclasses
        # into structural_signature().
        def region_dict(region: KeepRegion | RebuildRegion) -> dict[str, Any]:
            return {
                "node_keys": [str(node_key) for node_key in region.node_keys],
                "reason": region.reason,
            }

        return {
            "previous_signature": self.previous_signature,
            "current_signature": self.current_signature,
            "previous_epoch": self.previous_epoch,
            "epoch": self.epoch,
            "changed": self.changed,
            "keep_regions": [region_dict(region) for region in self.keep_regions],
            "rebuild_regions": [region_dict(region) for region in self.rebuild_regions],
        }


def _node_map(graph: CompiledWorkGraph) -> dict[NodeKey, str]:
    return {
        node.node_key: structural_signature(node)
        for node in graph.semantic.nodes
    }


def reconcile(
    previous: CompiledWorkGraph | None,
    current: CompiledWorkGraph,
    *,
    previous_epoch: int | None = None,
) -> PlanDiff:
    """Compute a metadata-only plan diff.

    A matching whole-graph signature retains the prior epoch.  Any structural
    change increments it once, while unchanged node keys remain explainable as
    retained regions.  No runtime objects are touched or released here.
    """

    current_signature = current.structural_signature or structural_signature(current)
    if previous is None:
        base_epoch = 0 if previous_epoch is None else int(previous_epoch)
        node_keys = tuple(node.node_key for node in current.semantic.nodes)
        rebuild = (RebuildRegion(node_keys=node_keys, reason="initial_plan"),) if node_keys else ()
        return PlanDiff(None, current_signature, previous_epoch, base_epoch + 1, (), rebuild)
    previous_signature = previous.structural_signature or structural_signature(previous)
    old_epoch = previous.epoch if previous_epoch is None else int(previous_epoch)
    if previous_signature == current_signature:
        keys = tuple(node.node_key for node in current.semantic.nodes)
        keep = (KeepRegion(node_keys=keys),) if keys else ()
        return PlanDiff(previous_signature, current_signature, old_epoch, old_epoch, keep, ())

    old_nodes = _node_map(previous)
    new_nodes = _node_map(current)
    retained = tuple(
        key for key in (node.node_key for node in current.semantic.nodes)
        if key in old_nodes and old_nodes[key] == new_nodes[key]
    )
    rebuilt = tuple(
        key
        for key in tuple(node.node_key for node in current.semantic.nodes)
        if key not in old_nodes or old_nodes[key] != new_nodes[key]
    )
    removed = tuple(key for key in old_nodes if key not in new_nodes)
    all_rebuilt = rebuilt + removed
    if not all_rebuilt and current.semantic.nodes:
        # A backend/topology/graph-mode change can invalidate the physical
        # plan even when individual semantic node records are unchanged.
        all_rebuilt = tuple(node.node_key for node in current.semantic.nodes)
        rebuild_reason = "graph_metadata_changed"
    else:
        rebuild_reason = "structural_signature_changed"
    keep_regions = (KeepRegion(retained),) if retained else ()
    rebuild_regions = (RebuildRegion(all_rebuilt, reason=rebuild_reason),) if all_rebuilt else ()
    return PlanDiff(
        previous_signature,
        current_signature,
        old_epoch,
        old_epoch + 1,
        keep_regions,
        rebuild_regions,
    )



"""Deterministic, JSON-safe WorkGraph compile explanations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tllm.workgraph.ir import CompiledWorkGraph
from tllm.workgraph.reconcile import PlanDiff
from tllm.workgraph.signature import canonical_value, structural_signature


@dataclass(frozen=True, slots=True)
class WorkGraphExplain:
    graph_signature: str
    epoch: int
    source: tuple[str, ...] = ()
    nodes: tuple[dict[str, Any], ...] = ()
    semantic_edges: tuple[dict[str, str], ...] = ()
    values: tuple[dict[str, Any], ...] = ()
    residences: tuple[dict[str, Any], ...] = ()
    moves: tuple[dict[str, Any], ...] = ()
    leases: tuple[dict[str, Any], ...] = ()
    bindings: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[str, ...] = ()
    reconciliation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "graph_signature": self.graph_signature,
            "epoch": self.epoch,
            "source": list(self.source),
            "nodes": list(self.nodes),
            "semantic_edges": list(self.semantic_edges),
            "values": list(self.values),
            "residences": list(self.residences),
            "moves": list(self.moves),
            "leases": list(self.leases),
            "bindings": list(self.bindings),
            "diagnostics": list(self.diagnostics),
        }
        if self.reconciliation is not None:
            result["reconciliation"] = self.reconciliation
        # All nested descriptor fields were converted explicitly above.  The
        # report itself is an ordinary JSON object, intentionally separate
        # from the strict structural serializer (which rejects mappings).
        return result

    def as_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def json(self) -> str:
        return self.to_json()


def explain_workgraph(graph: CompiledWorkGraph, diff: PlanDiff | None = None) -> WorkGraphExplain:
    """Render only stable fields from a compiled WorkGraph."""

    signature = graph.structural_signature or structural_signature(graph)
    backend = graph.semantic.backend
    source = tuple(
        node.source
        for node in graph.semantic.nodes
        if node.source
    )
    nodes = tuple(
        {
            "node_key": str(node.node_key),
            "kind": node.kind,
            "executor": None if node.executor is None else str(node.executor),
            "reads": [access.value.key for access in node.reads],
            "writes": [access.value.key for access in node.writes],
            "effects": list(node.effects),
            "timing": canonical_value(node.timing),
            "anchor": None if node.anchor is None else str(node.anchor.node_key),
            "graph_safety": node.graph_safety,
            "legacy_boundary": node.legacy_boundary,
        }
        for node in graph.semantic.nodes
    )
    semantic_edges = tuple(
        {
            "source": str(source),
            "target": str(target),
            "kind": kind.value,
        }
        for source, target, kind in graph.semantic.edges
    )
    values = tuple(
        {
            "value": value.key,
            "port": canonical_value(value.port),
            "shape": list(value.shape),
            "dtype": value.dtype,
            "layout": value.layout,
            "alias_set": value.alias_set,
        }
        for value in graph.data_centric.values
    )
    residences = tuple(
        {
            "value": residence.value.key,
            "device": residence.device,
            "rank": residence.rank,
            "memory_kind": residence.memory_kind,
            "valid_version": residence.valid_version,
        }
        for residence in graph.data_centric.residences
    )
    moves = tuple(canonical_value(move) for move in graph.data_centric.moves)
    leases = tuple(
        {
            "value": lease.value.key,
            "storage": lease.storage,
            "live_start": lease.live_start,
            "live_end": lease.live_end,
            "stable_address": lease.stable_address,
            "materialized": lease.materialized,
        }
        for lease in graph.data_centric.leases
    )
    bindings = tuple(
        {
            "node_key": str(node.node_key),
            "device": node.device,
            "rank": node.rank,
            "anchor": None if node.anchor is None else str(node.anchor.node_key),
            "residences": list(node.residence_keys),
            "leases": list(node.lease_keys),
        }
        for node in graph.data_centric.nodes
    )
    diagnostics = list(graph.diagnostics)
    if backend is not None:
        diagnostics.append(
            f"backend={backend.backend_id};runner={backend.runner_family or 'unknown'};graph_mode={backend.graph_mode}"
        )
    return WorkGraphExplain(
        graph_signature=signature,
        epoch=graph.epoch,
        source=source,
        nodes=nodes,
        semantic_edges=semantic_edges,
        values=values,
        residences=residences,
        moves=moves,
        leases=leases,
        bindings=bindings,
        diagnostics=tuple(diagnostics),
        reconciliation=None if diff is None else diff.to_dict(),
    )


explain = explain_workgraph

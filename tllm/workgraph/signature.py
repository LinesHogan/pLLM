"""Deterministic structural serialization for WorkGraph descriptors.

The serializer is deliberately closed-world.  WorkGraph metadata has an
explicit schema, so accepting arbitrary mappings or objects here would make a
runtime Tensor/private backend value look like a valid compiler key.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, cast

from tllm.ports.address import LocatorFieldDesc, PortAddressDesc
from tllm.ports.port_id import PortId
from tllm.workgraph.ir import (
    AccessDesc,
    AnchorDesc,
    BackendSnapshot,
    CompiledWorkGraph,
    DataCentricIR,
    ExecutionNodeDesc,
    ExecutorKey,
    LeaseDesc,
    MoveDesc,
    NodeKey,
    ResidenceDesc,
    RuntimePlanBinding,
    SemanticGraph,
    SemanticNodeDesc,
    TimingDesc,
    ValueDesc,
)


def _unsupported(value: Any) -> TypeError:
    value_type = cast(type[Any], type(value))
    return TypeError(
        "structural metadata must contain only the declared WorkGraph "
        "descriptors, primitives, Enum, PortId, PortAddressDesc, or tuples; "
        f"got {value_type.__module__}.{value_type.__qualname__}"
    )


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, PortId):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "version": value.version,
        }
    if isinstance(value, LocatorFieldDesc):
        return {
            "path": _canonical(value.path),
            "value_type": value.value_type,
            "value": value.value,
        }
    if isinstance(value, PortAddressDesc):
        return {
            "kind": value.kind,
            "locator_type": value.locator_type,
            "locator_fields": _canonical(value.locator_fields),
        }
    if isinstance(value, NodeKey):
        return {
            "namespace": value.namespace,
            "owner": value.owner,
            "ordinal": value.ordinal,
        }
    if isinstance(value, ExecutorKey):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "version": value.version,
        }
    if isinstance(value, BackendSnapshot):
        return {
            "backend_id": value.backend_id,
            "runner_family": value.runner_family,
            "graph_mode": value.graph_mode,
            "backend_version": value.backend_version,
            "capabilities": _canonical(value.capabilities),
            "device": value.device,
            "rank": value.rank,
            "tp_size": value.tp_size,
            "pp_size": value.pp_size,
            "dp_size": value.dp_size,
            "topology": _canonical(value.topology),
        }
    if isinstance(value, AnchorDesc):
        return {
            "backend": value.backend,
            "runner": value.runner,
            "graph_mode": value.graph_mode,
            "hook": value.hook,
            "externally_owned": value.externally_owned,
            "mobility": value.mobility,
            "service_class": value.service_class,
        }
    if isinstance(value, TimingDesc):
        return {
            "window": value.window,
            "mobility": value.mobility,
            "deadline": value.deadline,
            "max_defer_steps": value.max_defer_steps,
            "observes_version": value.observes_version,
        }
    if isinstance(value, ValueDesc):
        return {
            "value_id": value.value_id,
            "version": value.version,
            "port": _canonical(value.port),
            "shape": _canonical(value.shape),
            "dtype": value.dtype,
            "layout": value.layout,
            "alias_set": value.alias_set,
        }
    if isinstance(value, AccessDesc):
        return {
            "node_key": _canonical(value.node_key),
            "value": _canonical(value.value),
            "mode": _canonical(value.mode),
            "port": _canonical(value.port),
            "subset": _canonical(value.subset),
            "consistency": value.consistency,
        }
    if isinstance(value, ResidenceDesc):
        return {
            "value": _canonical(value.value),
            "device": value.device,
            "rank": value.rank,
            "memory_kind": value.memory_kind,
            "valid_version": value.valid_version,
        }
    if isinstance(value, MoveDesc):
        return {
            "value": _canonical(value.value),
            "source": _canonical(value.source),
            "destination": _canonical(value.destination),
            "subset": _canonical(value.subset),
            "layout": value.layout,
            "engine": value.engine,
            "asynchronous": value.asynchronous,
        }
    if isinstance(value, LeaseDesc):
        return {
            "value": _canonical(value.value),
            "storage": value.storage,
            "live_start": value.live_start,
            "live_end": value.live_end,
            "ready_fence": value.ready_fence,
            "retire_fence": value.retire_fence,
            "reader_count": value.reader_count,
            "stable_address": value.stable_address,
            "materialized": value.materialized,
        }
    if isinstance(value, SemanticNodeDesc):
        return {
            "node_key": _canonical(value.node_key),
            "kind": _canonical(value.kind),
            "executor": _canonical(value.executor),
            "reads": _canonical(value.reads),
            "writes": _canonical(value.writes),
            "effects": _canonical(value.effects),
            "timing": _canonical(value.timing),
            "anchor": _canonical(value.anchor),
            "graph_safety": _canonical(value.graph_safety),
            "source": value.source,
            "legacy_boundary": _canonical(value.legacy_boundary),
            "shape_guard": _canonical(value.shape_guard),
        }
    if isinstance(value, SemanticGraph):
        return {
            "schema_version": value.schema_version,
            "nodes": _canonical(value.nodes),
            "edges": _canonical(value.edges),
            "values": _canonical(value.values),
            "backend": _canonical(value.backend),
            "graph_mode": value.graph_mode,
        }
    if isinstance(value, ExecutionNodeDesc):
        return {
            "node_key": _canonical(value.node_key),
            "semantic_key": _canonical(value.semantic_key),
            "kind": _canonical(value.kind),
            "executor": _canonical(value.executor),
            "accesses": _canonical(value.accesses),
            "residence_keys": _canonical(value.residence_keys),
            "move_keys": _canonical(value.move_keys),
            "lease_keys": _canonical(value.lease_keys),
            "device": value.device,
            "rank": value.rank,
            "anchor": _canonical(value.anchor),
            "graph_safety": _canonical(value.graph_safety),
            "fusion_group": value.fusion_group,
        }
    if isinstance(value, DataCentricIR):
        return {
            "schema_version": value.schema_version,
            "nodes": _canonical(value.nodes),
            "values": _canonical(value.values),
            "accesses": _canonical(value.accesses),
            "residences": _canonical(value.residences),
            "moves": _canonical(value.moves),
            "leases": _canonical(value.leases),
            "edges": _canonical(value.edges),
            "backend": _canonical(value.backend),
            "graph_mode": value.graph_mode,
        }
    if isinstance(value, RuntimePlanBinding):
        return {
            "dispatch_flow_ids": _canonical(value.dispatch_flow_ids),
            "dispatch_hooks": _canonical(value.dispatch_hooks),
            "dispatch_fusion_groups": _canonical(value.dispatch_fusion_groups),
            "execution_producer_ids": _canonical(value.execution_producer_ids),
            "producer_hooks": _canonical(value.producer_hooks),
            "execution_port_sources": _canonical(value.execution_port_sources),
            "capture_resident_producer_ids": _canonical(value.capture_resident_producer_ids),
        }
    if isinstance(value, CompiledWorkGraph):
        # Plan epoch and a cached digest are revision metadata, not graph
        # structure.  The two IR layers are the complete signature input.
        return _canonical((value.semantic, value.data_centric))
    if isinstance(value, tuple):
        tuple_value = cast(tuple[Any, ...], value)
        return [_canonical(item) for item in tuple_value]
    raise _unsupported(value)


def canonical_value(value: Any) -> Any:
    """Return the JSON-compatible canonical structural representation."""

    return _canonical(value)


def canonical_json(value: Any) -> str:
    """Serialize metadata with stable key ordering and no runtime payload."""

    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def structural_signature(value: Any) -> str:
    """Return the full SHA-256 digest of canonical structural metadata."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signature_payload(value: Any) -> Any:
    """Return the canonical payload used by the signature oracle."""

    return canonical_value(value)


sha256_signature = structural_signature

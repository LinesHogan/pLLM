#!/usr/bin/env python3
"""Producer registry + execution-plan compiler.

The registry keeps *all* candidate producers per port (unlike the legacy
``ProviderRegistry`` where a later registration silently overwrote the
earlier one). ``ExecutionPlanCompiler`` resolves producer/consumer
declarations into an ordered DAG at startup and fails fast on:

* a consumer port with no candidate producer;
* two producers for the same port with equal top priority;
* a producer requiring a port nobody offers;
* a dependency cycle;
* a producer whose hook points or graph safety the backend cannot serve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tllm.backends.protocol import BackendCapabilities
from tllm.ports.base import PortKind
from tllm.ports.port_id import PortId
from tllm.producers.base import (
    HookPoint,
    producer_graph_safety,
    producer_hook_points,
    producer_output_specs,
    producer_requires,
    validate_producer,
)


class ExecutionPlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProducerEntry:
    producer: Any
    priority: int = 0

    @property
    def producer_id(self) -> str:
        return str(getattr(self.producer, "producer_id", ""))


class ProducerRegistry:
    def __init__(self) -> None:
        self._entries: list[ProducerEntry] = []

    def register(self, producer: Any, *, priority: int = 0) -> None:
        validate_producer(producer)
        entry = ProducerEntry(producer=producer, priority=int(priority))
        for existing in self._entries:
            if existing.producer_id == entry.producer_id:
                raise ExecutionPlanError(
                    f"duplicate producer_id {entry.producer_id!r} registered"
                )
        self._entries.append(entry)

    def entries(self) -> tuple[ProducerEntry, ...]:
        return tuple(self._entries)

    def candidates_for_kind(self, kind: PortKind) -> list[ProducerEntry]:
        return [
            entry
            for entry in self._entries
            if any(spec.kind == kind for spec in producer_output_specs(entry.producer))
        ]

    def provider_of(self, port: PortId) -> ProducerEntry | None:
        for entry in self._entries:
            if any(spec.id == port for spec in producer_output_specs(entry.producer)):
                return entry
        return None

    def clear(self) -> None:
        self._entries.clear()


@dataclass(frozen=True)
class PlannedProducer:
    """One node of the compiled plan."""

    entry: ProducerEntry
    emits_at: frozenset[HookPoint]
    output_ports: tuple[PortId, ...]
    depends_on: tuple[PortId, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    """Compiled producer DAG in topological order."""

    producers: tuple[PlannedProducer, ...]
    port_sources: dict[PortId, str]  # port -> producer_id

    def producers_for_hook(self, hook: HookPoint) -> list[PlannedProducer]:
        return [p for p in self.producers if hook in p.emits_at]


def compile_execution_plan(
    *,
    registry: ProducerRegistry,
    consumers: Sequence[Any] = (),
    backend_capabilities: BackendCapabilities | None = None,
    supported_hooks: frozenset[HookPoint] | None = None,
    enforced_kinds: frozenset[PortKind] | None = None,
) -> ExecutionPlan:
    """Compile the producer DAG via the Rust core (hard dependency).

    ``enforced_kinds`` limits the consumer-port fail-fast check to the given
    kinds. ``None`` enforces every kind any consumer reads (strict mode).
    The legacy Python implementation survives only as the differential
    oracle under ``test/core/oracles/`` (see its sunset header).
    """
    from tllm import core as _core

    producers: list[dict[str, object]] = [
        {
            "id": entry.producer_id,
            "priority": entry.priority,
            "outputs": [
                {
                    "port": str(spec.id),
                    "kind": spec.kind.value,
                    "writable": bool(spec.writable),
                    "write_modes": [str(m) for m in spec.write_modes],
                }
                for spec in producer_output_specs(entry.producer)
            ],
            "requires": [str(port) for port in producer_requires(entry.producer)],
            "hook_points": [hook.value for hook in producer_hook_points(entry.producer)],
            "graph_safety": producer_graph_safety(entry.producer).name,
        }
        for entry in registry.entries()
    ]
    consumer_descs: list[dict[str, object]] = []
    for consumer in consumers:
        consumer_id = str(getattr(consumer, "consumer_id", repr(consumer)))
        flows = consumer.flows() if callable(getattr(consumer, "flows", None)) else ()
        reads: list[str] = []
        writes: list[dict[str, str]] = []
        for flow in flows or ():
            for read in getattr(flow, "reads", ()) or ():
                kind = getattr(read, "kind", None)
                if isinstance(kind, PortKind):
                    reads.append(kind.value)
            for write in getattr(flow, "writes", ()) or ():
                kind = getattr(write, "kind", None)
                if isinstance(kind, PortKind):
                    writes.append(
                        {"kind": kind.value, "mode": str(getattr(write, "mode", "") or "")}
                    )
        consumer_descs.append({"id": consumer_id, "reads": reads, "writes": writes})

    backend = None
    if backend_capabilities is not None:
        backend = (
            str(backend_capabilities.backend_id),
            backend_capabilities.graph_safety.name,
        )
    try:
        compiled = _core.compile_execution_plan(
            producers,
            consumer_descs,
            enforced_kinds=None if enforced_kinds is None else [k.value for k in enforced_kinds],
            supported_hooks=None if supported_hooks is None else [h.value for h in supported_hooks],
            backend=backend,
        )
    except RuntimeError as exc:
        raise ExecutionPlanError(str(exc)) from exc

    by_id = {entry.producer_id: entry for entry in registry.entries()}
    planned = tuple(
        PlannedProducer(
            entry=by_id[pid],
            emits_at=frozenset(producer_hook_points(by_id[pid].producer)),
            output_ports=tuple(spec.id for spec in producer_output_specs(by_id[pid].producer)),
            depends_on=tuple(producer_requires(by_id[pid].producer)),
        )
        for pid in compiled.ordered_producer_ids
    )
    port_sources = {PortId.parse(port): pid for port, pid in compiled.port_sources}
    return ExecutionPlan(producers=planned, port_sources=port_sources)

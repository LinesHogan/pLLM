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

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.ports.base import PortKind
from tllm.ports.port_id import PortId
from tllm.producers.base import (
    HookPoint,
    Producer,
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


def _consumer_required_kinds(consumers: Iterable[Any]) -> list[tuple[str, PortKind]]:
    required: list[tuple[str, PortKind]] = []
    for consumer in consumers:
        consumer_id = str(getattr(consumer, "consumer_id", repr(consumer)))
        flows = consumer.flows() if callable(getattr(consumer, "flows", None)) else ()
        for flow in flows or ():
            for read in getattr(flow, "reads", ()) or ():
                kind = getattr(read, "kind", None)
                if isinstance(kind, PortKind):
                    required.append((consumer_id, kind))
    return required


def _consumer_written_ports(consumers: Iterable[Any]) -> list[tuple[str, PortKind, str]]:
    written: list[tuple[str, PortKind, str]] = []
    for consumer in consumers:
        consumer_id = str(getattr(consumer, "consumer_id", repr(consumer)))
        flows = consumer.flows() if callable(getattr(consumer, "flows", None)) else ()
        for flow in flows or ():
            for write in getattr(flow, "writes", ()) or ():
                kind = getattr(write, "kind", None)
                if isinstance(kind, PortKind):
                    written.append((consumer_id, kind, str(getattr(write, "mode", "") or "")))
    return written


def _validate_consumer_writes(
    *,
    registry: ProducerRegistry,
    consumers: Sequence[Any],
    enforced_kinds: frozenset[PortKind] | None,
) -> None:
    """Fail fast on writes no registered producer declared writable."""
    for consumer_id, kind, mode in _consumer_written_ports(consumers):
        if enforced_kinds is not None and kind not in enforced_kinds:
            continue
        candidates = registry.candidates_for_kind(kind)
        if not candidates:
            raise ExecutionPlanError(
                f"consumer {consumer_id!r} writes port kind {kind.value!r} "
                "but no registered producer offers it"
            )
        specs = [
            spec
            for entry in candidates
            for spec in producer_output_specs(entry.producer)
            if spec.kind == kind
        ]
        writable = [spec for spec in specs if bool(getattr(spec, "writable", False))]
        if not writable:
            raise ExecutionPlanError(
                f"consumer {consumer_id!r} writes port kind {kind.value!r} "
                "but no producer for that kind declares it writable"
            )
        if mode:
            supported = sorted({m for spec in writable for m in getattr(spec, "write_modes", ())})
            if mode not in supported:
                raise ExecutionPlanError(
                    f"consumer {consumer_id!r} writes port kind {kind.value!r} "
                    f"with mode {mode!r} but producers only support: {supported}"
                )


def compile_execution_plan(
    *,
    registry: ProducerRegistry,
    consumers: Sequence[Any] = (),
    backend_capabilities: BackendCapabilities | None = None,
    supported_hooks: frozenset[HookPoint] | None = None,
    enforced_kinds: frozenset[PortKind] | None = None,
) -> ExecutionPlan:
    """Compile the producer DAG.

    ``enforced_kinds`` limits the consumer-port fail-fast check to the given
    kinds. ``None`` enforces every kind any consumer reads (strict mode).
    During the incremental migration the runtime passes the kinds the new
    registry actually covers so legacy hardcoded producers keep working.
    """
    entries = list(registry.entries())
    supported = HookPoint.all() if supported_hooks is None else supported_hooks
    _validate_consumer_writes(
        registry=registry,
        consumers=consumers,
        enforced_kinds=enforced_kinds,
    )

    # 1. Every consumer-read port kind must have at least one producer; each
    # PortId must resolve to exactly one producer (priority breaks ties; a
    # same-priority tie on one port is a startup error). Producers offering
    # different ports of the same kind are complementary (e.g. decode vs
    # prefill residual), not competitors.
    selected: dict[str, ProducerEntry] = {}
    port_winners: dict[PortId, ProducerEntry] = {}
    for consumer_id, kind in _consumer_required_kinds(consumers):
        if enforced_kinds is not None and kind not in enforced_kinds:
            continue
        candidates = registry.candidates_for_kind(kind)
        if not candidates:
            raise ExecutionPlanError(
                f"consumer {consumer_id!r} reads port kind {kind.value!r} "
                "but no registered producer offers it"
            )
        if kind in selected:
            continue
        by_port: dict[PortId, list[ProducerEntry]] = {}
        for entry in candidates:
            for spec in producer_output_specs(entry.producer):
                if spec.kind == kind:
                    by_port.setdefault(spec.id, []).append(entry)
        for port, port_entries in by_port.items():
            if port in port_winners:
                continue
            best_priority = max(e.priority for e in port_entries)
            top = [e for e in port_entries if e.priority == best_priority]
            if len(top) > 1:
                raise ExecutionPlanError(
                    f"port {port} required by consumer {consumer_id!r} has "
                    f"{len(top)} producers at priority {best_priority}: "
                    f"{sorted(e.producer_id for e in top)}; disambiguate via priority"
                )
            port_winners[port] = top[0]
        for winner in port_winners.values():
            selected.setdefault(winner.producer_id, winner)

    # Transitively include producers required by selected producers.
    included: dict[str, ProducerEntry] = {}
    stack = list(selected.values())
    by_id = {entry.producer_id: entry for entry in entries}
    offered: dict[PortId, ProducerEntry] = {}
    for entry in entries:
        for spec in producer_output_specs(entry.producer):
            offered[spec.id] = entry
    # Dependency resolution follows the same per-port winners when available.
    for port, winner in port_winners.items():
        offered[port] = winner
    while stack:
        entry = stack.pop()
        if entry.producer_id in included:
            continue
        included[entry.producer_id] = entry
        for req in producer_requires(entry.producer):
            source = offered.get(req)
            if source is None:
                raise ExecutionPlanError(
                    f"producer {entry.producer_id!r} requires port {req} "
                    "but no registered producer offers it"
                )
            if source.producer_id not in included:
                stack.append(source)

    # 2. Topological order over the included producers (Kahn's algorithm).
    indegree: dict[str, int] = {pid: 0 for pid in included}
    edges: dict[str, list[str]] = {pid: [] for pid in included}
    for pid, entry in included.items():
        for req in producer_requires(entry.producer):
            source = offered[req]
            if source.producer_id in included:
                edges[source.producer_id].append(pid)
                indegree[pid] += 1
    queue = sorted(pid for pid, deg in indegree.items() if deg == 0)
    ordered: list[str] = []
    while queue:
        pid = queue.pop(0)
        ordered.append(pid)
        for nxt in sorted(edges[pid]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(ordered) != len(included):
        remaining = sorted(pid for pid, deg in indegree.items() if deg > 0)
        raise ExecutionPlanError(
            f"producer dependency cycle involving: {remaining}"
        )

    # 3. Backend feasibility: hook points and graph safety.
    planned: list[PlannedProducer] = []
    for pid in ordered:
        entry = included[pid]
        hooks = producer_hook_points(entry.producer)
        unsupported = hooks - supported
        if unsupported:
            raise ExecutionPlanError(
                f"producer {pid!r} needs hook points the backend does not "
                f"support: {sorted(h.value for h in unsupported)}"
            )
        if backend_capabilities is not None:
            safety = producer_graph_safety(entry.producer)
            if (
                backend_capabilities.graph_safety == GraphSafety.EAGER_ONLY
                and safety != GraphSafety.EAGER_ONLY
            ):
                raise ExecutionPlanError(
                    f"producer {pid!r} declares graph safety {safety.name} but "
                    f"backend {backend_capabilities.backend_id!r} is EAGER_ONLY"
                )
        planned.append(
            PlannedProducer(
                entry=entry,
                emits_at=frozenset(hooks),
                output_ports=tuple(spec.id for spec in producer_output_specs(entry.producer)),
                depends_on=tuple(producer_requires(entry.producer)),
            )
        )

    port_sources = {
        port: pid
        for pid, entry in included.items()
        for port in (spec.id for spec in producer_output_specs(entry.producer))
    }
    return ExecutionPlan(producers=tuple(planned), port_sources=port_sources)

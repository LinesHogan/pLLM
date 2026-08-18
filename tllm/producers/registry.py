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
from typing import Protocol, Sequence

from tllm.backends.protocol import BackendCapabilities
from tllm.ports.base import ConsumerFlow, PortKind
from tllm.ports.address import port_phase
from tllm.ports.port_id import PortId
from tllm.producers.base import (
    HookPoint,
    Producer,
    validate_producer,
)
from tllm.producers.ir import (
    PlanConsumerDesc,
    PlanConsumerFlowDesc,
    PlanConsumerReadDesc,
    PlanConsumerWriteDesc,
    PlanPortSpecDesc,
    PlanProducerDesc,
)


class ExecutionPlanError(RuntimeError):
    pass


class FlowConsumer(Protocol):
    @property
    def consumer_id(self) -> str: ...

    def flows(self) -> Sequence[ConsumerFlow]: ...


@dataclass(frozen=True)
class ProducerEntry:
    producer: Producer
    priority: int = 0

    @property
    def producer_id(self) -> str:
        return self.producer.producer_id


class ProducerRegistry:
    def __init__(self) -> None:
        self._entries: list[ProducerEntry] = []

    def register(self, producer: object, *, priority: int = 0) -> None:
        typed_producer = validate_producer(producer)
        entry = ProducerEntry(producer=typed_producer, priority=int(priority))
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
            if any(spec.kind == kind for spec in entry.producer.outputs)
        ]

    def provider_of(self, port: PortId) -> ProducerEntry | None:
        for entry in self._entries:
            if any(spec.id == port for spec in entry.producer.outputs):
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
    capture_resident_producer_ids: tuple[str, ...] = ()

    def producers_for_hook(self, hook: HookPoint) -> list[PlannedProducer]:
        return [p for p in self.producers if hook in p.emits_at]


def compile_execution_plan(
    *,
    registry: ProducerRegistry,
    consumers: Sequence[FlowConsumer] = (),
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

    producers = [
        PlanProducerDesc(
            id=entry.producer_id,
            priority=entry.priority,
            outputs=tuple(
                PlanPortSpecDesc(
                    port=str(spec.id),
                    kind=spec.kind.value,
                    writable=spec.writable,
                    write_modes=tuple(spec.write_modes),
                    phases=tuple(spec.phases),
                )
                for spec in entry.producer.outputs
            ),
            requires=tuple(str(port) for port in entry.producer.requires),
            hook_points=tuple(hook.value for hook in entry.producer.hook_points),
            graph_safety=entry.producer.graph_safety.name,
        )
        for entry in registry.entries()
    ]
    consumer_descs: list[PlanConsumerDesc] = []
    for consumer in consumers:
        flows = tuple(consumer.flows())
        consumer_descs.append(
            PlanConsumerDesc(
                id=consumer.consumer_id,
                flows=tuple(
                    PlanConsumerFlowDesc(
                        reads=tuple(
                            PlanConsumerReadDesc(
                                kind=read.kind.value,
                                role=read.role,
                                phase=port_phase(read),
                            )
                            for read in flow.reads
                        ),
                        writes=tuple(
                            PlanConsumerWriteDesc(
                                kind=write.kind.value,
                                mode=write.mode,
                                phase=port_phase(write),
                            )
                            for write in flow.writes
                        ),
                        bundle_key=tuple(flow.bundle_key),
                    )
                    for flow in flows
                ),
            )
        )

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
            emits_at=by_id[pid].producer.hook_points,
            output_ports=tuple(spec.id for spec in by_id[pid].producer.outputs),
            depends_on=by_id[pid].producer.requires,
        )
        for pid in compiled.ordered_producer_ids
    )
    port_sources = {PortId.parse(port): pid for port, pid in compiled.port_sources}
    return ExecutionPlan(
        producers=planned,
        port_sources=port_sources,
        capture_resident_producer_ids=tuple(
            str(producer_id) for producer_id in compiled.capture_resident_producer_ids
        ),
    )

#!/usr/bin/env python3
"""LEGACY oracle — 仅作 Rust core 对照基准,禁止运行时引用。

This is the original Python implementation of the execution-plan compiler,
preserved verbatim as the differential oracle for ``tllm._core``'s
``compile_execution_plan``. Sunset plan: once the Rust core shows oracle
matrix equivalence + dual-runner parity + no ratio regression for two
consecutive development cycles, delete this file and evolve only in Rust.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence

from tllm.backends.protocol import GraphSafety
from tllm.ports.base import PortKind
from tllm.ports.port_id import PortId
from tllm.ports.port_spec import PortSpec
from tllm.producers.base import HookPoint


class OracleExecutionPlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class OracleProducerDesc:
    id: str
    priority: int
    outputs: tuple  # tuple[PortSpec]
    requires: tuple  # tuple[PortId]
    hook_points: frozenset  # frozenset[HookPoint]
    graph_safety: GraphSafety


def producer_desc(
    *,
    id: str,
    priority: int = 0,
    outputs: Sequence[dict] = (),
    requires: Sequence[str] = (),
    hook_points: Sequence[str] = (),
    graph_safety: str = "CAPTURE_SAFE",
) -> OracleProducerDesc:
    """Build a real-typed producer description from plain data."""
    return OracleProducerDesc(
        id=str(id),
        priority=int(priority),
        outputs=tuple(
            PortSpec(
                id=PortId.parse(str(spec["port"])),
                kind=PortKind(str(spec["kind"])),
                writable=bool(spec.get("writable", False)),
                write_modes=tuple(str(m) for m in spec.get("write_modes", ())),
            )
            for spec in outputs
        ),
        requires=tuple(PortId.parse(str(p)) for p in requires),
        hook_points=frozenset(HookPoint(str(h)) for h in hook_points),
        graph_safety=GraphSafety[str(graph_safety)],
    )


def consumer_desc(
    *,
    id: str,
    reads: Sequence[str] = (),
    writes: Sequence[dict] = (),
) -> Any:
    flows = [
        SimpleNamespace(
            reads=[SimpleNamespace(kind=PortKind(str(k))) for k in reads],
            writes=[
                SimpleNamespace(kind=PortKind(str(w["kind"])), mode=str(w.get("mode", "")))
                for w in writes
            ],
        )
    ]
    return SimpleNamespace(consumer_id=str(id), flows=lambda: flows)


def _required_kinds(consumers: Iterable[Any]) -> list[tuple[str, PortKind]]:
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


def _validate_writes(
    *,
    producers: Sequence[OracleProducerDesc],
    consumers: Sequence[Any],
    enforced_kinds: Optional[frozenset],
) -> None:
    for consumer in consumers:
        consumer_id = str(getattr(consumer, "consumer_id", repr(consumer)))
        flows = consumer.flows() if callable(getattr(consumer, "flows", None)) else ()
        for flow in flows or ():
            for write in getattr(flow, "writes", ()) or ():
                kind = getattr(write, "kind", None)
                if not isinstance(kind, PortKind):
                    continue
                if enforced_kinds is not None and kind not in enforced_kinds:
                    continue
                specs = [spec for p in producers for spec in p.outputs if spec.kind == kind]
                if not specs:
                    raise OracleExecutionPlanError(
                        f"consumer {consumer_id!r} writes port kind {kind.value!r} "
                        "but no registered producer offers it"
                    )
                writable = [spec for spec in specs if spec.writable]
                if not writable:
                    raise OracleExecutionPlanError(
                        f"consumer {consumer_id!r} writes port kind {kind.value!r} "
                        "but no producer for that kind declares it writable"
                    )
                mode = str(getattr(write, "mode", "") or "")
                if mode:
                    supported = sorted({m for spec in writable for m in spec.write_modes})
                    if mode not in supported:
                        raise OracleExecutionPlanError(
                            f"consumer {consumer_id!r} writes port kind {kind.value!r} "
                            f"with mode {mode!r} but producers only support: {supported}"
                        )


def oracle_compile_execution_plan(
    *,
    producers: Sequence[OracleProducerDesc],
    consumers: Sequence[Any] = (),
    enforced_kinds: Optional[frozenset] = None,
    supported_hooks: Optional[frozenset] = None,
    backend: Optional[tuple[str, str]] = None,
) -> tuple[list[str], dict[str, str]]:
    """The legacy compile, verbatim semantics, plain-data inputs.

    Returns ``(ordered_producer_ids, {port_str: producer_id})``.
    """
    supported = HookPoint.all() if supported_hooks is None else supported_hooks
    _validate_writes(producers=producers, consumers=consumers, enforced_kinds=enforced_kinds)

    selected: dict[str, OracleProducerDesc] = {}
    port_winners: dict[PortId, OracleProducerDesc] = {}
    for consumer_id, kind in _required_kinds(consumers):
        if enforced_kinds is not None and kind not in enforced_kinds:
            continue
        candidates = [p for p in producers if any(spec.kind == kind for spec in p.outputs)]
        if not candidates:
            raise OracleExecutionPlanError(
                f"consumer {consumer_id!r} reads port kind {kind.value!r} "
                "but no registered producer offers it"
            )
        if kind in selected:
            continue
        by_port: dict[PortId, list[OracleProducerDesc]] = {}
        for entry in candidates:
            for spec in entry.outputs:
                if spec.kind == kind:
                    by_port.setdefault(spec.id, []).append(entry)
        for port, port_entries in by_port.items():
            if port in port_winners:
                continue
            best_priority = max(e.priority for e in port_entries)
            top = [e for e in port_entries if e.priority == best_priority]
            if len(top) > 1:
                raise OracleExecutionPlanError(
                    f"port {port} required by consumer {consumer_id!r} has "
                    f"{len(top)} producers at priority {best_priority}: "
                    f"{sorted(e.id for e in top)}; disambiguate via priority"
                )
            port_winners[port] = top[0]
        for winner in port_winners.values():
            selected.setdefault(winner.id, winner)

    included: dict[str, OracleProducerDesc] = {}
    stack = [p for p in selected.values()]
    offered: dict[PortId, OracleProducerDesc] = {}
    for entry in producers:
        for spec in entry.outputs:
            offered[spec.id] = entry
    for port, winner in port_winners.items():
        offered[port] = winner
    while stack:
        entry = stack.pop()
        if entry.id in included:
            continue
        included[entry.id] = entry
        for req in entry.requires:
            source = offered.get(req)
            if source is None:
                raise OracleExecutionPlanError(
                    f"producer {entry.id!r} requires port {req} "
                    "but no registered producer offers it"
                )
            if source.id not in included:
                stack.append(source)

    indegree: dict[str, int] = {pid: 0 for pid in included}
    edges: dict[str, list[str]] = {pid: [] for pid in included}
    for pid, entry in included.items():
        for req in entry.requires:
            source = offered[req]
            if source.id in included:
                edges[source.id].append(pid)
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
        raise OracleExecutionPlanError(
            f"producer dependency cycle involving: {remaining}"
        )

    for pid in ordered:
        entry = included[pid]
        unsupported = entry.hook_points - supported
        if unsupported:
            raise OracleExecutionPlanError(
                f"producer {pid!r} needs hook points the backend does not "
                f"support: {sorted(h.value for h in unsupported)}"
            )
        if backend is not None:
            backend_id, backend_safety = backend
            if backend_safety == "EAGER_ONLY" and entry.graph_safety != GraphSafety.EAGER_ONLY:
                raise OracleExecutionPlanError(
                    f"producer {pid!r} declares graph safety {entry.graph_safety.name} but "
                    f"backend {backend_id!r} is EAGER_ONLY"
                )

    port_sources = {
        str(spec.id): pid
        for pid, entry in included.items()
        for spec in entry.outputs
    }
    return ordered, port_sources

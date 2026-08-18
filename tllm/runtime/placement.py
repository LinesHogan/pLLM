#!/usr/bin/env python3
"""Placement compilation: lower typed flow contracts through the Rust core.

The compiler validates timing and dependency contracts, derives a stable
topological order, and chooses a legal materialized hook using the current
bubble-score cost model. The runtime consumes the resulting table; it does not
re-run placement decisions on the per-step Tensor/CUDA path.

Today's effective dispatch point for every per-step flow is the decode
dispatch inside the execute_model wrapper, so the declared hook is
``execute_model.post``; post-generate flows (generation-trace reads, reward /
retry / barrier writes) live outside per-step placement and are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tllm.core import compile_placement_schedule
from tllm.ports.base import (
    ConsumerFlow,
    PortKind,
    PortRead,
    PortWrite,
    TimingContract,
    TimingMobility,
    VersionObservation,
)
from tllm.ports.address import PortAddressDesc, port_address_desc
from tllm.runtime.graph_policy import GraphMode

#: Where per-step flows actually dispatch today (decode bundle dispatch runs
#: inside the execute_model wrapper, after the model forward).
DECLARED_DISPATCH_HOOK = "execute_model.post"
MATERIALIZED_FLOW_HOOKS = (
    "execute_model.post",
    "sample_tokens.pre",
    "sampler.pre_draw",
    "sample_tokens.post",
)


@dataclass(frozen=True, slots=True)
class PlacementWriteDesc:
    port: PortAddressDesc
    order: int
    mode: str
    fusion_candidate: bool

    def __post_init__(self) -> None:
        # Compatibility for direct compiler/oracle tests written before the
        # structural address IR. Production lowering always passes a descriptor.
        if isinstance(self.port, str):
            object.__setattr__(self, "port", PortAddressDesc.opaque(self.port))


@dataclass(frozen=True, slots=True)
class VersionedStateResourceDesc:
    """Compiler-visible identity for one plan-local state generation.

    ``version`` is an SSA-style relative generation, not a runtime epoch
    counter. Repeated decode steps instantiate the same version relationship.
    """

    kind: str
    namespace: str
    name: str
    version: int

    @property
    def label(self) -> str:
        return f"{self.kind}|{self.namespace}|{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class PlacementResourceWriteDesc:
    resource: VersionedStateResourceDesc


@dataclass(frozen=True, slots=True)
class PlacementFlowDesc:
    flow_id: str
    consumer_id: str
    declared_hook: str
    mobility: str | None
    deadline: str | None
    max_defer_steps: int | None
    observes_version: str | None
    reads: tuple[PortAddressDesc, ...]
    writes: tuple[PlacementWriteDesc, ...]
    delivery: str = "bundle"
    ownership: str = "borrowed"
    lease_lifetime: str = "consume_call"
    max_bundle_rows: int = 0
    effect_hook: str | None = None
    earliest_hook: str | None = None
    allowed_hooks: tuple[str, ...] = ()
    fusion_kind: str = "none"
    resource_reads: tuple[VersionedStateResourceDesc, ...] = ()
    resource_writes: tuple[PlacementResourceWriteDesc, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reads",
            tuple(
                PortAddressDesc.opaque(address) if isinstance(address, str) else address
                for address in self.reads
            ),
        )


@dataclass(frozen=True, slots=True)
class PlacementHookCost:
    hook: str
    bubble_score: int


@dataclass(frozen=True, slots=True)
class CompiledPlacement:
    by_flow_id: dict[str, str]
    execution_order: tuple[str, ...]
    fusion_groups: tuple[tuple[str, ...], ...]
    fusion_rejections: tuple[tuple[str, str], ...]


def flow_timing_desc(flow_id: str, flow: ConsumerFlow, *, consumer_id: str = "") -> PlacementFlowDesc:
    """Build the plain-data descriptor the Rust compiler consumes."""
    raw = flow.timing
    # ConsumerFlow.__post_init__ always normalizes to TimingContract; the
    # isinstance keeps the type checker honest about the declared union.
    timing = raw if isinstance(raw, TimingContract) else TimingContract(mobility=TimingMobility(str(raw)))
    if (
        timing.mobility != TimingMobility.PINNED
        and timing.observes_version != VersionObservation.AT_DECLARED_WINDOW
    ):
        raise RuntimeError(
            f"flow {flow_id!r} requests observes_version={str(timing.observes_version)!r}, "
            "but the runtime has no execution-time port-version resolver; only "
            "'at_declared_window' is executable for movable flows"
        )
    fusion_writes = tuple(write for write in flow.writes if write.fusion_candidate)
    fusion_kind = str(flow.fusion)
    if fusion_kind == "side_stream_cudagraph":
        effect_hook = "sampler.pre_draw"
    else:
        effect_hook = (
            "layer.post"
            if fusion_writes and all(write.kind == PortKind.RESIDUAL_STREAM for write in fusion_writes)
            else None
        )
    return PlacementFlowDesc(
        flow_id=str(flow_id),
        consumer_id=str(consumer_id),
        declared_hook=DECLARED_DISPATCH_HOOK,
        mobility=str(timing.mobility),
        deadline=str(timing.deadline),
        max_defer_steps=int(timing.max_defer_steps),
        observes_version=str(timing.observes_version),
        reads=tuple(port_address_desc(read) for read in flow.reads),
        writes=tuple(
            PlacementWriteDesc(
                port=port_address_desc(write),
                order=int(write.order),
                mode=str(write.mode),
                fusion_candidate=bool(write.fusion_candidate),
            )
            for write in flow.writes
        ),
        delivery=str(flow.delivery),
        ownership=str(flow.ownership),
        lease_lifetime=str(flow.lease_lifetime),
        max_bundle_rows=int(flow.max_bundle_rows),
        effect_hook=effect_hook,
        earliest_hook=DECLARED_DISPATCH_HOOK,
        # The current bundle assembler snapshots all flow inputs after model
        # execution. The 12-hook compiler therefore sees the full domain, but
        # may only choose hooks at which this materialized bundle exists.
        allowed_hooks=MATERIALIZED_FLOW_HOOKS,
        fusion_kind=fusion_kind,
        resource_reads=(),
        resource_writes=(),
    )


def compile_flow_placement(
    flows: Sequence[tuple[str, ConsumerFlow]],
    *,
    graph_mode: GraphMode = GraphMode.EAGER,
) -> CompiledPlacement:
    """Fail fast on illegal timing contracts / graph-mode combinations."""
    if not flows:
        return CompiledPlacement(by_flow_id={}, execution_order=(), fusion_groups=(), fusion_rejections=())
    descriptors = [
        flow_timing_desc(flow_id, flow, consumer_id=flow_id.rsplit("#", 1)[0])
        for flow_id, flow in flows
    ]
    schedule = compile_placement_schedule(
        descriptors,
        graph_mode.name.lower(),
    )
    return CompiledPlacement(
        by_flow_id={str(flow_id): str(hook) for flow_id, hook in schedule.placements},
        execution_order=tuple(str(flow_id) for flow_id in schedule.execution_order),
        fusion_groups=tuple(tuple(str(flow_id) for flow_id in group) for group in schedule.fusion_groups),
        fusion_rejections=tuple(
            (str(flow_id), str(reason)) for flow_id, reason in schedule.fusion_rejections
        ),
    )


def validate_flow_timing(
    flows: Sequence[tuple[str, ConsumerFlow]],
    *,
    graph_mode: GraphMode = GraphMode.EAGER,
) -> dict[str, str]:
    """Compatibility facade returning only the compiled hook table."""
    return compile_flow_placement(flows, graph_mode=graph_mode).by_flow_id

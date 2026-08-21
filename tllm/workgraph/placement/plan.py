#!/usr/bin/env python3
"""Placement plan compilation: assign values to devices and lower moves."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import cast
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.physical import PhysicalFlowPlan
from tllm.workgraph.placement.implementation import (
    CopyEngineCapacityDesc,
    DurationProfile,
    ImplementationDesc,
    ImplementationRegistry,
    ImplementationSelectionDesc,
    MoveKey,
    ScheduleCandidate,
    ScheduledMove,
    SemanticOracleKey,
    TransferProfile,
    TransferProfileDesc,
    ValueFlowDesc,
    schedule_candidates,
)
from tllm.workgraph.placement.transfer import (
    DeviceTopology,
    Move,
    TransferRequest,
    compile_moves,
)
from tllm.workgraph.placement.vocab import (
    CopyEngine,
    DeviceKind,
    MemoryKind,
    MoveKind,
    Placement,
    PlacementCompilationError,
    Residency,
    ValueKey,
    require_executor_key,
    require_instance,
    require_int,
    require_node_key,
    require_tuple_of,
)


@dataclass(frozen=True, slots=True)
class PlacedNodeDesc:
    node_key: NodeKey
    implementation: ExecutorKey
    placement: Placement
    start_at: int
    end_at: int
    depends_on: tuple[NodeKey, ...] = ()

    def __post_init__(self) -> None:
        node_key = require_node_key(self.node_key, "PlacedNodeDesc.node_key")
        implementation = require_executor_key(
            self.implementation, "PlacedNodeDesc.implementation"
        )
        placement = require_instance(self.placement, Placement, "PlacedNodeDesc.placement")
        start = require_int(self.start_at, "PlacedNodeDesc.start_at")
        end = require_int(self.end_at, "PlacedNodeDesc.end_at", minimum=1)
        dependencies = require_tuple_of(
            self.depends_on, NodeKey, "PlacedNodeDesc.depends_on"
        )
        if end <= start:
            raise ValueError("PlacedNodeDesc.end_at must be greater than start_at")
        if node_key in dependencies or len(set(dependencies)) != len(dependencies):
            raise ValueError("PlacedNodeDesc dependencies must be unique and acyclic")
        object.__setattr__(self, "node_key", node_key)
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "start_at", start)
        object.__setattr__(self, "end_at", end)
        object.__setattr__(self, "depends_on", dependencies)
@dataclass(frozen=True, slots=True)
class PlacementLeaseDesc:
    residency: Residency
    live_start: int
    live_end: int

    def __post_init__(self) -> None:
        residency = require_instance(
            self.residency, Residency, "PlacementLeaseDesc.residency"
        )
        start = require_int(self.live_start, "PlacementLeaseDesc.live_start")
        end = require_int(self.live_end, "PlacementLeaseDesc.live_end")
        if end < start:
            raise ValueError("PlacementLeaseDesc.live_end must be >= live_start")
        object.__setattr__(self, "residency", residency)
        object.__setattr__(self, "live_start", start)
        object.__setattr__(self, "live_end", end)
@dataclass(frozen=True, slots=True)
class PlacementPlan:
    plan_epoch: int
    selections: tuple[ImplementationSelectionDesc, ...]
    schedule: tuple[PlacedNodeDesc, ...]
    topology: DeviceTopology
    value_flows: tuple[ValueFlowDesc, ...] = ()
    residencies: tuple[Residency, ...] = ()
    moves: tuple[ScheduledMove, ...] = ()
    leases: tuple[PlacementLeaseDesc, ...] = ()
    copy_engines: tuple[CopyEngineCapacityDesc, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        epoch = require_int(self.plan_epoch, "PlacementPlan.plan_epoch", minimum=1)
        selections = require_tuple_of(
            self.selections,
            ImplementationSelectionDesc,
            "PlacementPlan.selections",
        )
        schedule = require_tuple_of(self.schedule, PlacedNodeDesc, "PlacementPlan.schedule")
        residencies = require_tuple_of(
            self.residencies, Residency, "PlacementPlan.residencies"
        )
        moves = require_tuple_of(self.moves, ScheduledMove, "PlacementPlan.moves")
        leases = require_tuple_of(self.leases, PlacementLeaseDesc, "PlacementPlan.leases")
        copy_engines = require_tuple_of(
            self.copy_engines,
            CopyEngineCapacityDesc,
            "PlacementPlan.copy_engines",
        )
        raw_topology = require_instance(
            self.topology, DeviceTopology, "PlacementPlan.topology"
        )
        value_flows = require_tuple_of(
            self.value_flows, ValueFlowDesc, "PlacementPlan.value_flows"
        )
        topology = DeviceTopology(
            peer_access=raw_topology.peer_access,
            cuda_devices=raw_topology.cuda_devices,
        )

        def canonical_placement(value: Placement) -> Placement:
            return Placement(value.kind, value.index)

        def canonical_value(value: ValueKey) -> ValueKey:
            return ValueKey(value.name, value.version)

        def canonical_residency(value: Residency) -> Residency:
            return Residency(
                canonical_value(value.value),
                canonical_placement(value.placement),
                value.memory,
            )

        def canonical_oracle(
            value: SemanticOracleKey | None,
        ) -> SemanticOracleKey | None:
            if value is None:
                return None
            return SemanticOracleKey(value.namespace, value.name, value.version)

        selections = tuple(
            ImplementationSelectionDesc(
                node_key=item.node_key,
                logical_executor=item.logical_executor,
                key=item.key,
                placement=canonical_placement(item.placement),
                duration=DurationProfile(
                    item.duration.microseconds, item.duration.samples
                ),
                oracle_key=canonical_oracle(item.oracle_key),
            )
            for item in selections
        )
        schedule = tuple(
            PlacedNodeDesc(
                node_key=item.node_key,
                implementation=item.implementation,
                placement=canonical_placement(item.placement),
                start_at=item.start_at,
                end_at=item.end_at,
                depends_on=tuple(
                    require_node_key(dependency, "PlacedNodeDesc.depends_on")
                    for dependency in item.depends_on
                ),
            )
            for item in schedule
        )
        residencies = tuple(canonical_residency(item) for item in residencies)
        moves = tuple(
            ScheduledMove(
                key=MoveKey(canonical_value(item.key.value), item.key.ordinal),
                move=Move(
                    value=canonical_value(item.move.value),
                    source=canonical_residency(item.move.source),
                    destination=canonical_residency(item.move.destination),
                    kind=item.move.kind,
                    byte_count=item.move.byte_count,
                    asynchronous=item.move.asynchronous,
                ),
                engine=item.engine,
                start_at=item.start_at,
                end_at=item.end_at,
                producer=item.producer,
                consumer=item.consumer,
                depends_on=tuple(
                    MoveKey(
                        canonical_value(dependency.value), dependency.ordinal
                    )
                    for dependency in item.depends_on
                ),
                used_fallback=item.used_fallback,
            )
            for item in moves
        )
        leases = tuple(
            PlacementLeaseDesc(
                canonical_residency(item.residency),
                item.live_start,
                item.live_end,
            )
            for item in leases
        )
        copy_engines = tuple(
            CopyEngineCapacityDesc(item.engine, item.capacity)
            for item in copy_engines
        )
        value_flows = tuple(
            ValueFlowDesc(
                canonical_value(item.value),
                item.producer,
                item.consumer,
                item.byte_count,
            )
            for item in value_flows
        )
        if self.schema_version != 1:
            raise ValueError("PlacementPlan.schema_version must be 1")
        selection_nodes = tuple(item.node_key for item in selections)
        schedule_nodes = tuple(item.node_key for item in schedule)
        if len(set(selection_nodes)) != len(selection_nodes):
            raise ValueError("PlacementPlan selections must have unique nodes")
        selections_by_logical: dict[ExecutorKey, list[ImplementationSelectionDesc]] = {}
        for selection in selections:
            selections_by_logical.setdefault(selection.logical_executor, []).append(selection)
        for logical_selections in selections_by_logical.values():
            if len({item.placement for item in logical_selections}) > 1:
                oracle_keys = {item.oracle_key for item in logical_selections}
                if None in oracle_keys or len(oracle_keys) != 1:
                    raise ValueError(
                        "cross-placement selections require one semantic oracle key"
                    )
        if any(
            not topology.contains(item.placement) for item in selections
        ):
            raise ValueError("PlacementPlan selection is outside its topology")
        if len(set(schedule_nodes)) != len(schedule_nodes):
            raise ValueError("PlacementPlan schedule must have unique nodes")
        if set(selection_nodes) != set(schedule_nodes):
            raise ValueError("PlacementPlan selections and schedule must cover the same nodes")
        selection_by_node = {item.node_key: item for item in selections}
        positions = {node: index for index, node in enumerate(schedule_nodes)}
        if len(
            {(item.value, item.producer, item.consumer) for item in value_flows}
        ) != len(value_flows):
            raise ValueError("PlacementPlan value flows must be unique")
        flow_producers: dict[ValueKey, NodeKey] = {}
        for flow in value_flows:
            if (
                flow.producer not in positions
                or flow.consumer not in positions
                or positions[flow.producer] >= positions[flow.consumer]
            ):
                raise ValueError("PlacementPlan value flow must follow its schedule")
            previous = flow_producers.setdefault(flow.value, flow.producer)
            if previous != flow.producer:
                raise ValueError("PlacementPlan ValueKey must have one producer")
        for item in schedule:
            selected = selection_by_node[item.node_key]
            if (
                selected.key != item.implementation
                or selected.placement != item.placement
            ):
                raise ValueError("PlacementPlan schedule differs from its selection")
            if item.end_at - item.start_at != selected.duration.microseconds:
                raise ValueError(
                    "PlacementPlan schedule duration differs from its selection"
                )
            if any(
                dependency not in positions
                or positions[dependency] >= positions[item.node_key]
                for dependency in item.depends_on
            ):
                raise ValueError("PlacementPlan schedule dependencies must be ordered")
            if any(
                schedule[positions[dependency]].end_at > item.start_at
                for dependency in item.depends_on
            ):
                raise ValueError(
                    "PlacementPlan schedule dependencies must finish before launch"
                )
        for position, item in enumerate(schedule):
            if any(
                other.placement == item.placement
                and other.start_at < item.end_at
                and other.end_at > item.start_at
                for other in schedule[:position]
            ):
                raise ValueError(
                    "PlacementPlan compute intervals must not overlap on one placement"
                )
        if len(set(residencies)) != len(residencies):
            raise ValueError("PlacementPlan residencies must be unique")
        if any(
            not topology.contains(item.placement) for item in residencies
        ):
            raise ValueError("PlacementPlan residency is outside its topology")
        residency_set = set(residencies)
        expected_residencies: set[Residency] = set()
        for flow in value_flows:
            required = (
                Residency(flow.value, selection_by_node[flow.producer].placement),
                Residency(flow.value, selection_by_node[flow.consumer].placement),
            )
            expected_residencies.update(required)
            if any(item not in residency_set for item in required):
                raise ValueError(
                    "PlacementPlan value flow is missing a required residency"
                )
        if any(
            item.move.source not in residency_set
            or item.move.destination not in residency_set
            for item in moves
        ):
            raise ValueError("PlacementPlan move references an unknown residency")
        expected_residencies.update(
            residency
            for item in moves
            for residency in (item.move.source, item.move.destination)
        )
        if residency_set != expected_residencies:
            raise ValueError(
                "PlacementPlan residencies must exactly cover value flows and routes"
            )
        schedule_by_node = {item.node_key: item for item in schedule}
        if any(item.producer is None or item.consumer is None for item in moves):
            raise ValueError("PlacementPlan moves require explicit producer and consumer nodes")
        for item in moves:
            assert item.producer is not None and item.consumer is not None
            if (
                item.move.kind == MoveKind.P2P
                and not topology.supports_peer(
                    item.move.source.placement,
                    item.move.destination.placement,
                )
            ):
                raise ValueError("PlacementPlan P2P move is outside its topology")
            producer = schedule_by_node.get(item.producer)
            consumer = schedule_by_node.get(item.consumer)
            if producer is None or consumer is None:
                raise ValueError("PlacementPlan move references an unknown scheduled node")
            if (
                positions[item.producer] >= positions[item.consumer]
                or item.start_at < producer.end_at
                or item.end_at > consumer.start_at
            ):
                raise ValueError("PlacementPlan move must complete between producer and consumer")
        routes: dict[
            tuple[NodeKey, NodeKey, ValueKey], list[ScheduledMove]
        ] = {}
        for item in moves:
            assert item.producer is not None and item.consumer is not None
            routes.setdefault(
                (item.producer, item.consumer, item.move.value), []
            ).append(item)
        for (producer_key, consumer_key, _value), route in routes.items():
            producer_placement = selection_by_node[producer_key].placement
            consumer_placement = selection_by_node[consumer_key].placement
            if route[0].move.source.placement != producer_placement:
                raise ValueError("PlacementPlan route source differs from its producer")
            if route[-1].move.destination.placement != consumer_placement:
                raise ValueError("PlacementPlan route destination differs from its consumer")
            if route[0].depends_on:
                raise ValueError("PlacementPlan route head must not depend on another move")
            if producer_placement == consumer_placement:
                raise ValueError("same-placement values must not have a Move route")
            if producer_placement.kind == DeviceKind.CPU:
                expected_kinds = (MoveKind.H2D,)
            elif consumer_placement.kind == DeviceKind.CPU:
                expected_kinds = (MoveKind.D2H,)
            elif topology.supports_peer(producer_placement, consumer_placement):
                expected_kinds = (MoveKind.P2P,)
            else:
                expected_kinds = (MoveKind.D2H, MoveKind.H2D)
            if tuple(item.move.kind for item in route) != expected_kinds:
                raise ValueError("PlacementPlan route differs from its topology matrix")
            if len(route) == 2:
                if (
                    route[0].move.destination != route[1].move.source
                    or route[0].move.destination.memory != MemoryKind.PINNED_HOST
                    or route[1].depends_on != (route[0].key,)
                ):
                    raise ValueError(
                        "PlacementPlan staged route must use one continuous pinned chain"
                    )
        expected_routes = {
            (flow.producer, flow.consumer, flow.value)
            for flow in value_flows
            if selection_by_node[flow.producer].placement
            != selection_by_node[flow.consumer].placement
        }
        if set(routes) != expected_routes:
            raise ValueError(
                "PlacementPlan routes must exactly cover cross-placement value flows"
            )
        move_positions = {item.key: index for index, item in enumerate(moves)}
        if len(move_positions) != len(moves):
            raise ValueError("PlacementPlan moves must have unique keys")
        if any(
            dependency not in move_positions
            or move_positions[dependency] >= move_positions[item.key]
            for item in moves
            for dependency in item.depends_on
        ):
            raise ValueError("PlacementPlan move dependencies must be ordered")
        if any(
            moves[move_positions[dependency]].end_at > item.start_at
            for item in moves
            for dependency in item.depends_on
        ):
            raise ValueError("PlacementPlan move dependencies must finish before launch")
        for item in moves:
            if item.move.source.memory == MemoryKind.PINNED_HOST and not any(
                moves[move_positions[dependency]].move.destination == item.move.source
                for dependency in item.depends_on
            ):
                raise ValueError("pinned-host H2D requires its D2H dependency")
        if any(lease.residency not in residency_set for lease in leases):
            raise ValueError("PlacementPlan lease references an unknown residency")
        lease_residencies = tuple(lease.residency for lease in leases)
        if len(set(lease_residencies)) != len(lease_residencies):
            raise ValueError("PlacementPlan leases must have unique residencies")
        if set(lease_residencies) != residency_set:
            raise ValueError("every PlacementPlan residency requires one lease")
        lease_by_residency = {lease.residency: lease for lease in leases}
        for flow in value_flows:
            producer = schedule_by_node[flow.producer]
            consumer = schedule_by_node[flow.consumer]
            for residency in (
                Residency(flow.value, selection_by_node[flow.producer].placement),
                Residency(flow.value, selection_by_node[flow.consumer].placement),
            ):
                lease = lease_by_residency[residency]
                if (
                    lease.live_start > producer.end_at
                    or lease.live_end < consumer.end_at
                ):
                    raise ValueError(
                        "PlacementPlan value-flow lease must cover producer to consumer"
                    )
        if any(
            lease_by_residency[residency].live_start > item.start_at
            or lease_by_residency[residency].live_end < item.end_at
            for item in moves
            for residency in (item.move.source, item.move.destination)
        ):
            raise ValueError("PlacementPlan lease does not cover its scheduled move")
        engines = tuple(item.engine for item in copy_engines)
        if len(set(engines)) != len(engines):
            raise ValueError("PlacementPlan copy engines must be unique")
        capacity_by_engine = {
            item.engine: item.capacity for item in copy_engines
        }
        if any(item.engine not in capacity_by_engine for item in moves):
            raise ValueError("PlacementPlan move references an undeclared copy engine")
        for item in moves:
            overlapping = sum(
                other.engine == item.engine
                and other.start_at < item.end_at
                and other.end_at > item.start_at
                for other in moves
            )
            if overlapping > capacity_by_engine[item.engine]:
                raise ValueError("PlacementPlan moves exceed copy-engine capacity")
        object.__setattr__(self, "plan_epoch", epoch)
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "value_flows", value_flows)
        object.__setattr__(self, "residencies", residencies)
        object.__setattr__(self, "moves", moves)
        object.__setattr__(self, "leases", leases)
        object.__setattr__(self, "copy_engines", copy_engines)
_MAX_EXACT_PLACEMENT_ASSIGNMENTS = 100_000
def _residency_sort_key(residency: Residency) -> tuple[str, int, str, int]:
    placement = residency.placement
    memory = cast(MemoryKind, residency.memory)
    return (
        residency.value.name,
        residency.value.version,
        f"{placement.kind.value}:{memory.value}",
        -1 if placement.index is None else placement.index,
    )
def _profiled_move_duration(
    move: Move,
    *,
    aggregate: TransferProfile,
    profiles: tuple[TransferProfileDesc, ...],
) -> tuple[DurationProfile, bool]:
    for profile in profiles:
        if (
            profile.kind == move.kind
            and profile.source == move.source.placement
            and profile.destination == move.destination.placement
        ):
            return DurationProfile(
                profile.duration(move.byte_count),
                samples=profile.samples,
            ), False
    return aggregate.for_engine(move.engine)
def compile_placement_plan(
    physical: PhysicalFlowPlan,
    *,
    registry: ImplementationRegistry,
    topology: DeviceTopology,
    value_flows: tuple[ValueFlowDesc, ...] = (),
    transfer_profile: TransferProfile = TransferProfile(),
    transfer_profiles: tuple[TransferProfileDesc, ...] = (),
    copy_engines: tuple[CopyEngineCapacityDesc, ...] = (),
    placement_overrides: Mapping[ExecutorKey, ExecutorKey] | None = None,
) -> PlacementPlan:
    """Compile one deterministic local CPU/GPU placement and transfer plan."""

    physical = require_instance(physical, PhysicalFlowPlan, "compile_placement_plan.physical")
    registry = require_instance(
        registry, ImplementationRegistry, "compile_placement_plan.registry"
    )
    topology = require_instance(topology, DeviceTopology, "compile_placement_plan.topology")
    flows = require_tuple_of(value_flows, ValueFlowDesc, "compile_placement_plan.value_flows")
    transfer_profile = require_instance(
        transfer_profile, TransferProfile, "compile_placement_plan.transfer_profile"
    )
    route_profiles = require_tuple_of(
        transfer_profiles,
        TransferProfileDesc,
        "compile_placement_plan.transfer_profiles",
    )
    profile_keys = tuple(
        (profile.kind, profile.source, profile.destination)
        for profile in route_profiles
    )
    if len(set(profile_keys)) != len(profile_keys):
        raise ValueError("compile_placement_plan transfer profiles must be unique")
    capacities = require_tuple_of(
        copy_engines,
        CopyEngineCapacityDesc,
        "compile_placement_plan.copy_engines",
    )
    if not capacities:
        capacities = tuple(CopyEngineCapacityDesc(engine) for engine in CopyEngine)
    if len({item.engine for item in capacities}) != len(capacities):
        raise ValueError("compile_placement_plan copy engines must be unique")

    overrides: dict[ExecutorKey, ExecutorKey] = {}
    if placement_overrides is not None:
        for logical, implementation in placement_overrides.items():
            overrides[require_instance(logical, ExecutorKey, "placement override logical key")] = require_instance(
                implementation,
                ExecutorKey,
                "placement override implementation key",
            )

    nodes = {node.node_key: node for node in physical.nodes}
    positions = {node.node_key: index for index, node in enumerate(physical.nodes)}
    if any(
        flow.producer not in nodes
        or flow.consumer not in nodes
        or positions[flow.producer] >= positions[flow.consumer]
        for flow in flows
    ):
        raise ValueError("value flow must follow the physical DAG")
    if len({(flow.value, flow.producer, flow.consumer) for flow in flows}) != len(flows):
        raise ValueError("value flows must be unique")
    producer_by_value: dict[ValueKey, NodeKey] = {}
    for flow in flows:
        previous_producer = producer_by_value.setdefault(flow.value, flow.producer)
        if previous_producer != flow.producer:
            raise ValueError(
                "each ValueKey must have exactly one producer; use distinct value versions"
            )
    logical_executors = {node.executor for node in physical.nodes}
    if any(logical not in logical_executors for logical in overrides):
        raise ValueError("placement override references an unknown logical executor")

    candidate_cache: dict[ExecutorKey, tuple[ImplementationDesc, ...]] = {}

    def available_candidates(logical: ExecutorKey) -> tuple[ImplementationDesc, ...]:
        cached = candidate_cache.get(logical)
        if cached is not None:
            return cached
        available = tuple(
            sorted(
                (
                    candidate
                    for candidate in registry.candidates(logical)
                    if topology.contains(candidate.placement)
                ),
                key=lambda candidate: (
                    candidate.key.value,
                    candidate.placement.kind.value,
                    -1
                    if candidate.placement.index is None
                    else candidate.placement.index,
                    candidate.duration.microseconds,
                ),
            )
        )
        candidate_cache[logical] = available
        return available

    dependencies: dict[NodeKey, set[NodeKey]] = {
        node.node_key: {fence.producer for fence in node.fences}
        for node in physical.nodes
    }
    for flow in flows:
        dependencies[flow.consumer].add(flow.producer)

    choices_by_node: dict[NodeKey, tuple[ImplementationDesc, ...]] = {}
    for node in physical.nodes:
        node_choices = available_candidates(node.executor)
        if not node_choices:
            raise ValueError(f"no implementation registered for {node.executor.value}")
        override = overrides.get(node.executor)
        if override is not None:
            node_choices = tuple(
                item for item in node_choices if item.key == override
            )
        if not node_choices:
            raise ValueError("placement override is not registered for its logical executor")
        choices_by_node[node.node_key] = node_choices

    schedule_inputs = tuple(
        ScheduleCandidate(
            key=node.node_key.value,
            priority=0,
            ready_at=0,
            depends_on=tuple(
                dependency.value for dependency in sorted(dependencies[node.node_key])
            ),
            duration=min(
                item.duration.microseconds
                for item in choices_by_node[node.node_key]
            ),
        )
        for node in physical.nodes
    )
    node_by_text = {node.node_key.value: node for node in physical.nodes}
    node_order = tuple(
        node_by_text[item.key] for item in schedule_candidates(schedule_inputs)
    )

    assignment_count = 1
    for node in node_order:
        assignment_count *= len(choices_by_node[node.node_key])
        if assignment_count > _MAX_EXACT_PLACEMENT_ASSIGNMENTS:
            raise PlacementCompilationError(
                "exact placement assignment search exceeds the placement bound; "
                "reduce implementation candidates or provide placement overrides"
            )

    capacity_by_engine = {item.engine: item.capacity for item in capacities}

    def assignment_score(
        assignment: tuple[ImplementationDesc, ...],
    ) -> tuple[int, int, int, tuple[str, ...]]:
        selected = {
            node.node_key: choice
            for node, choice in zip(node_order, assignment, strict=True)
        }
        engine_lanes = {
            engine: [0] * capacity
            for engine, capacity in capacity_by_engine.items()
        }
        placement_available: dict[Placement, int] = {}
        node_end: dict[NodeKey, int] = {}
        transfer_work = 0
        compute_work = 0
        makespan = 0
        for node in node_order:
            choice = selected[node.node_key]
            dependency_end = max(
                (node_end[item] for item in dependencies[node.node_key]),
                default=0,
            )
            inbound_end = dependency_end
            for flow in sorted(
                (item for item in flows if item.consumer == node.node_key),
                key=lambda item: (item.value.value, item.producer.value),
            ):
                producer = selected[flow.producer]
                route = compile_moves(
                    (
                        TransferRequest(
                            flow.value,
                            Residency(flow.value, producer.placement),
                            Residency(flow.value, choice.placement),
                            flow.byte_count,
                        ),
                    ),
                    topology=topology,
                )
                ready = node_end[flow.producer]
                for move in route:
                    lanes = engine_lanes[move.engine]
                    lane = min(
                        range(len(lanes)), key=lambda index: (lanes[index], index)
                    )
                    begin = max(ready, lanes[lane])
                    duration, _used_fallback = _profiled_move_duration(
                        move,
                        aggregate=transfer_profile,
                        profiles=route_profiles,
                    )
                    ready = begin + duration.microseconds
                    lanes[lane] = ready
                    transfer_work += duration.microseconds
                inbound_end = max(inbound_end, ready)
            start = max(
                dependency_end,
                inbound_end,
                placement_available.get(choice.placement, 0),
            )
            end = start + choice.duration.microseconds
            node_end[node.node_key] = end
            placement_available[choice.placement] = end
            compute_work += choice.duration.microseconds
            makespan = max(makespan, end)
        signature = tuple(choice.key.value for choice in assignment)
        return makespan, transfer_work, compute_work, signature

    assignments = product(
        *(choices_by_node[node.node_key] for node in node_order)
    )
    chosen_assignment = min(assignments, key=assignment_score)
    selections_by_node: dict[NodeKey, ImplementationSelectionDesc] = {}
    for node, chosen in zip(node_order, chosen_assignment, strict=True):
        selections_by_node[node.node_key] = ImplementationSelectionDesc(
            node_key=node.node_key,
            logical_executor=node.executor,
            key=chosen.key,
            placement=chosen.placement,
            duration=chosen.duration,
            oracle_key=chosen.oracle_key,
        )

    engine_lanes: dict[CopyEngine, list[int]] = {
        engine: [0] * capacity for engine, capacity in capacity_by_engine.items()
    }
    placement_available: dict[Placement, int] = {}
    scheduled_nodes: list[PlacedNodeDesc] = []
    scheduled_by_node: dict[NodeKey, PlacedNodeDesc] = {}
    scheduled_moves: list[ScheduledMove] = []
    residencies: set[Residency] = set()

    for node in node_order:
        selected = selections_by_node[node.node_key]
        dependency_end = max(
            (scheduled_by_node[item].end_at for item in dependencies[node.node_key]),
            default=0,
        )
        inbound_end = dependency_end
        for flow in sorted(
            (item for item in flows if item.consumer == node.node_key),
            key=lambda item: (item.value.value, item.producer.value),
        ):
            producer = selections_by_node[flow.producer]
            source = Residency(flow.value, producer.placement)
            destination = Residency(flow.value, selected.placement)
            residencies.update((source, destination))
            route = compile_moves(
                (TransferRequest(flow.value, source, destination, flow.byte_count),),
                topology=topology,
            )
            ready = scheduled_by_node[flow.producer].end_at
            previous_key: MoveKey | None = None
            for move in route:
                lanes = engine_lanes[move.engine]
                lane = min(range(len(lanes)), key=lambda index: (lanes[index], index))
                begin = max(ready, lanes[lane])
                duration, used_fallback = _profiled_move_duration(
                    move,
                    aggregate=transfer_profile,
                    profiles=route_profiles,
                )
                end = begin + duration.microseconds
                key = MoveKey(move.value, len(scheduled_moves))
                scheduled = ScheduledMove(
                    key=key,
                    move=move,
                    engine=move.engine,
                    start_at=begin,
                    end_at=end,
                    producer=flow.producer,
                    consumer=flow.consumer,
                    depends_on=() if previous_key is None else (previous_key,),
                    used_fallback=used_fallback,
                )
                scheduled_moves.append(scheduled)
                residencies.update((move.source, move.destination))
                lanes[lane] = end
                ready = end
                previous_key = key
            inbound_end = max(inbound_end, ready)
        start = max(
            dependency_end,
            inbound_end,
            placement_available.get(selected.placement, 0),
        )
        scheduled = PlacedNodeDesc(
            node_key=node.node_key,
            implementation=selected.key,
            placement=selected.placement,
            start_at=start,
            end_at=start + selected.duration.microseconds,
            depends_on=tuple(sorted(dependencies[node.node_key])),
        )
        scheduled_nodes.append(scheduled)
        scheduled_by_node[node.node_key] = scheduled
        placement_available[selected.placement] = scheduled.end_at

    lease_ranges: dict[Residency, tuple[int, int]] = {}
    for flow in flows:
        producer_node = scheduled_by_node[flow.producer]
        consumer_node = scheduled_by_node[flow.consumer]
        producer = selections_by_node[flow.producer]
        consumer = selections_by_node[flow.consumer]
        source = Residency(flow.value, producer.placement)
        destination = Residency(flow.value, consumer.placement)
        for residency, start, end in (
            (source, producer_node.end_at, consumer_node.end_at),
            (destination, producer_node.end_at, consumer_node.end_at),
        ):
            current = lease_ranges.get(residency)
            lease_ranges[residency] = (
                start if current is None else min(start, current[0]),
                end if current is None else max(end, current[1]),
            )
    for move in scheduled_moves:
        if move.move.destination.memory == MemoryKind.PINNED_HOST:
            lease_ranges[move.move.destination] = (move.start_at, move.end_at)
        elif move.move.source.memory == MemoryKind.PINNED_HOST:
            current = lease_ranges.get(move.move.source, (move.start_at, move.start_at))
            lease_ranges[move.move.source] = (current[0], max(current[1], move.end_at))

    ordered_residencies = tuple(sorted(residencies, key=_residency_sort_key))
    leases = tuple(
        PlacementLeaseDesc(residency, *lease_ranges[residency])
        for residency in ordered_residencies
        if residency in lease_ranges
    )
    result = PlacementPlan(
        plan_epoch=physical.plan_epoch,
        selections=tuple(
            selections_by_node[node.node_key] for node in node_order
        ),
        schedule=tuple(scheduled_nodes),
        topology=topology,
        value_flows=flows,
        residencies=ordered_residencies,
        moves=tuple(scheduled_moves),
        leases=leases,
        copy_engines=capacities,
    )
    try:
        from tllm import core as _core

        _core.validate_placement_plan(result)
    except (TypeError, ValueError) as exc:
        raise PlacementCompilationError(
            "Rust placement-plan descriptor validation failed"
        ) from exc
    return result

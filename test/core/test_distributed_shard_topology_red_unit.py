"""RED contracts for coordinate-based Phase 4 state sharding.

State affinity is a set of topology ranks, not an ordered list of anonymous
shard candidates.  A tensor, pipeline, or data shard must therefore contain
every rank on its corresponding topology axis for each replica.  These tests
also pass frozen attribute-only proxies to the Rust validator so malformed
descriptors cannot be hidden by Python dataclass constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable

import pytest

from tllm.workgraph import distributed
from tllm.workgraph.frontend import StateScope
from tllm.workgraph.placement import Placement


@dataclass(frozen=True, slots=True)
class _FrozenAttrs:
    """Small immutable object exposing only the attributes Rust consumes."""

    entries: tuple[tuple[str, object], ...]

    def __getattr__(self, name: str) -> object:
        for key, value in self.entries:
            if key == name:
                return value
        raise AttributeError(name)


def _attrs(value: object) -> dict[str, object]:
    if isinstance(value, _FrozenAttrs):
        return dict(value.entries)
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _proxy(value: object, **changes: object) -> _FrozenAttrs:
    raw = _attrs(value)
    raw.update(changes)
    return _FrozenAttrs(tuple(raw.items()))


def _topology(*, permute_tensor_order: bool = False) -> distributed.DistributedTopology:
    coordinates = [
        (tp, pp, dp)
        for dp in range(2)
        for pp in range(2)
        for tp in range(2)
    ]
    if permute_tensor_order:
        # Keep rank ids contiguous while making rank order independent of the
        # tensor coordinate.  Ordinal assignment then disagrees with TP.
        coordinates = [
            (1, 0, 0),
            (0, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (1, 0, 1),
            (0, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
    ranks = tuple(
        distributed.TopologyRankDesc(
            rank=rank,
            node_id=f"node-{rank}",
            local_rank=0,
            placement=Placement("cpu"),
            tp_rank=tp,
            pp_rank=pp,
            dp_rank=dp,
        )
        for rank, (tp, pp, dp) in enumerate(coordinates)
    )
    return distributed.DistributedTopology(
        ranks=ranks,
        tp_size=2,
        pp_size=2,
        dp_size=2,
    )


def _requirement(
    kind: distributed.ShardKind,
    affinity: distributed.PlacementAffinity | None = None,
) -> distributed.StateRequirementDesc:
    return distributed.StateRequirementDesc(
        key=distributed.StateKey("phase4", f"{kind.value}-weights"),
        scope=StateScope.SHARD,
        shard=distributed.ShardSpec(
            kind=kind,
            partitions=2,
            axis=None if kind == distributed.ShardKind.PIPELINE else 0,
        ),
        affinity=affinity or distributed.PlacementAffinity(),
    )


def _state_plan(
    kind: distributed.ShardKind,
    *,
    topology: distributed.DistributedTopology | None = None,
) -> distributed.DistributedFlowPlan:
    topology = topology or _topology()
    requirement = _requirement(kind)
    residences = distributed._compile_state_residences(topology, (requirement,))
    return distributed.DistributedFlowPlan(
        plan_epoch=1,
        topology=topology,
        state_requirements=(requirement,),
        state_residences=residences,
        local_schedules=tuple(
            distributed.LocalCollectiveSchedule(rank.rank, ())
            for rank in topology.ranks
        ),
    )


def _ordinal_residences(
    requirement: object,
    topology: distributed.DistributedTopology,
    ranks: Iterable[int],
) -> tuple[_FrozenAttrs, ...]:
    key = getattr(requirement, "key")
    kind = getattr(getattr(requirement, "shard"), "kind")
    kind = getattr(kind, "value", kind)
    result = []
    for ordinal, rank_id in enumerate(ranks):
        rank = topology.rank(rank_id)
        result.append(
            _FrozenAttrs(
                (
                    ("key", key),
                    ("rank", rank_id),
                    ("placement", rank.placement),
                    ("shard_index", 0 if kind == "replicated" else ordinal % 2),
                    ("replica_index", ordinal if kind == "replicated" else ordinal // 2),
                )
            )
        )
    return tuple(result)


def _ordinal_dataclass_residences(
    requirement: distributed.StateRequirementDesc,
    topology: distributed.DistributedTopology,
    ranks: Iterable[int],
) -> tuple[distributed.StateResidenceDesc, ...]:
    kind = requirement.shard.kind
    return tuple(
        distributed.StateResidenceDesc(
            key=requirement.key,
            rank=rank_id,
            placement=topology.rank(rank_id).placement,
            shard_index=0 if kind == distributed.ShardKind.REPLICATED else ordinal % 2,
            replica_index=ordinal if kind == distributed.ShardKind.REPLICATED else ordinal // 2,
        )
        for ordinal, rank_id in enumerate(ranks)
    )


def _rust_validate(candidate: object) -> object:
    from tllm import core

    return core.validate_distributed_flow_plan(candidate)


@pytest.mark.parametrize(
    ("kind", "candidate_ranks"),
    (
        (distributed.ShardKind.TENSOR, (0, 2)),
        (distributed.ShardKind.PIPELINE, (0, 1)),
        (distributed.ShardKind.DATA, (0, 1)),
    ),
)
def test_python_compiler_rejects_affinity_that_is_not_coordinate_complete(
    kind: distributed.ShardKind,
    candidate_ranks: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tllm import core

    # Keep this assertion at the Python compiler boundary.  Rust coverage is
    # below and uses attribute-only descriptors explicitly.
    monkeypatch.setattr(core, "validate_distributed_flow_plan", lambda _: None)
    topology = _topology()
    requirement = _requirement(
        kind,
        distributed.PlacementAffinity(ranks=candidate_ranks),
    )

    with pytest.raises(ValueError):
        distributed.compile_distributed_flow_plan(
            plan_epoch=1,
            topology=topology,
            state_requirements=(requirement,),
        )


@pytest.mark.parametrize(
    ("kind", "topology"),
    (
        (distributed.ShardKind.TENSOR, _topology(permute_tensor_order=True)),
        (distributed.ShardKind.PIPELINE, _topology()),
        (distributed.ShardKind.DATA, _topology()),
    ),
)
def test_python_plan_rejects_ordinal_residences_for_coordinate_shards(
    kind: distributed.ShardKind,
    topology: distributed.DistributedTopology,
) -> None:
    plan = _state_plan(kind, topology=topology)
    requirement = plan.state_requirements[0]
    malformed_residences = _ordinal_dataclass_residences(
        requirement,
        topology,
        (residence.rank for residence in plan.state_residences),
    )

    with pytest.raises(ValueError):
        distributed.DistributedFlowPlan(
            plan_epoch=plan.plan_epoch,
            topology=topology,
            state_requirements=plan.state_requirements,
            state_residences=malformed_residences,
        )


def test_python_and_rust_reject_noncanonical_state_requirement_order() -> None:
    topology = _topology()
    requirements = (
        _requirement(distributed.ShardKind.TENSOR),
        _requirement(distributed.ShardKind.PIPELINE),
    )
    canonical_requirements = tuple(
        sorted(requirements, key=lambda item: item.key.value)
    )
    canonical_residences = distributed._compile_state_residences(
        topology,
        canonical_requirements,
    )
    canonical_plan = distributed.DistributedFlowPlan(
        plan_epoch=1,
        topology=topology,
        state_requirements=canonical_requirements,
        state_residences=canonical_residences,
        local_schedules=tuple(
            distributed.LocalCollectiveSchedule(rank.rank, ())
            for rank in topology.ranks
        ),
    )
    shuffled_requirements = tuple(reversed(canonical_requirements))

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        distributed.DistributedFlowPlan(
            plan_epoch=canonical_plan.plan_epoch,
            topology=topology,
            state_requirements=shuffled_requirements,
            state_residences=canonical_residences,
            local_schedules=canonical_plan.local_schedules,
        )

    candidate = _proxy(canonical_plan, state_requirements=shuffled_requirements)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _rust_validate(candidate)


@pytest.mark.parametrize(
    ("kind", "candidate_ranks"),
    (
        (distributed.ShardKind.TENSOR, (0, 2)),
        (distributed.ShardKind.PIPELINE, (0, 1)),
        (distributed.ShardKind.DATA, (0, 1)),
    ),
)
def test_rust_attribute_only_validator_rejects_malformed_affinity_and_residences(
    kind: distributed.ShardKind,
    candidate_ranks: tuple[int, ...],
) -> None:
    topology = _topology()
    plan = _state_plan(kind, topology=topology)
    requirement = plan.state_requirements[0]
    malformed_requirement = _proxy(
        requirement,
        affinity=_FrozenAttrs(
            (
                ("ranks", candidate_ranks),
                ("node_ids", ()),
                ("role", "model"),
            )
        ),
    )
    malformed_residences = _ordinal_residences(
        malformed_requirement,
        topology,
        candidate_ranks,
    )
    candidate = _proxy(
        plan,
        state_requirements=(malformed_requirement,),
        state_residences=malformed_residences,
    )

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _rust_validate(candidate)


@pytest.mark.parametrize(
    ("kind", "topology"),
    (
        (distributed.ShardKind.TENSOR, _topology(permute_tensor_order=True)),
        (distributed.ShardKind.PIPELINE, _topology()),
        (distributed.ShardKind.DATA, _topology()),
    ),
)
def test_rust_attribute_only_validator_rejects_coordinate_mismatch_in_residences(
    kind: distributed.ShardKind,
    topology: distributed.DistributedTopology,
) -> None:
    plan = _state_plan(kind, topology=topology)
    requirement = plan.state_requirements[0]
    malformed_residences = _ordinal_residences(
        requirement,
        topology,
        (residence.rank for residence in plan.state_residences),
    )
    candidate = _proxy(plan, state_residences=malformed_residences)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _rust_validate(candidate)


def test_rust_rejects_noncanonical_extra_affinity_node_id() -> None:
    topology = _topology()
    requirement = _requirement(
        distributed.ShardKind.TENSOR,
        distributed.PlacementAffinity(node_ids=("node-0", "node-1")),
    )
    plan = distributed.compile_distributed_flow_plan(
        plan_epoch=1,
        topology=topology,
        state_requirements=(requirement,),
    )
    malformed_requirement = _proxy(
        requirement,
        affinity=_proxy(
            requirement.affinity,
            node_ids=("node-0", "node-1", "bad/path"),
        ),
    )
    candidate = _proxy(plan, state_requirements=(malformed_requirement,))

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _rust_validate(candidate)

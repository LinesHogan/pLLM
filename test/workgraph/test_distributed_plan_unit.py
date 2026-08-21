"""Red contracts for global distributed WorkGraph planning.

The phase-4 module is intentionally imported from inside each test.  This
keeps collection useful on the pre-Phase-4 baseline while making each missing
contract an explicit, actionable failure.
"""

from __future__ import annotations

import dataclasses
import importlib
import unittest

from tllm.workgraph.frontend import StateScope
from tllm.workgraph.placement import Placement, ValueKey
from tllm.workgraph.signature import structural_signature


def _phase4(test_case: unittest.TestCase):
    try:
        return importlib.import_module("tllm.workgraph.distributed")
    except (ImportError, ModuleNotFoundError) as exc:
        test_case.fail(
            "distributed WorkGraph contracts are not implemented: "
            "tllm.workgraph.distributed is required"
        )
        raise AssertionError from exc


def _topology(distributed: object, *, helper: bool = False) -> object:
    """Build a two-node 2x2x2 model mesh, optionally with one helper GPU."""

    ranks = []
    for dp in range(2):
        for pp in range(2):
            for tp in range(2):
                rank = ((dp * 2) + pp) * 2 + tp
                node_id = "node-a" if rank < 4 else "node-b"
                local_rank = rank if rank < 4 else rank - 4
                ranks.append(
                    distributed.TopologyRankDesc(
                        rank=rank,
                        node_id=node_id,
                        local_rank=local_rank,
                        placement=Placement("cuda", local_rank),
                        tp_rank=tp,
                        pp_rank=pp,
                        dp_rank=dp,
                    )
                )
    if helper:
        ranks.append(
            distributed.TopologyRankDesc(
                rank=8,
                node_id="node-b",
                local_rank=4,
                placement=Placement("cuda", 4),
                role=distributed.RankRole.HELPER,
                tp_rank=None,
                pp_rank=None,
                dp_rank=None,
            )
        )
    return distributed.DistributedTopology(
        ranks=tuple(ranks),
        tp_size=2,
        pp_size=2,
        dp_size=2,
    )


def _request_set(distributed: object) -> tuple[object, ...]:
    """A dependency chain touching TP, PP, DP, and helper communicators."""

    tp = distributed.CollectiveKey("train", "tp", 0)
    pp = distributed.CollectiveKey("train", "pp", 0)
    dp = distributed.CollectiveKey("train", "dp", 0)
    helper = distributed.CollectiveKey("train", "helper", 0)
    return (
        distributed.CollectiveRequestDesc(
            key=tp,
            kind=distributed.CollectiveKind.ALL_REDUCE,
            group=distributed.ParallelGroupKind.TENSOR,
            group_index=0,
            backend=distributed.CollectiveBackend.NCCL,
            window=distributed.CollectiveWindow.BEFORE_COMPUTE,
            value=ValueKey("hidden", 1),
        ),
        distributed.CollectiveRequestDesc(
            key=pp,
            kind=distributed.CollectiveKind.ALL_GATHER,
            group=distributed.ParallelGroupKind.PIPELINE,
            group_index=0,
            backend=distributed.CollectiveBackend.NCCL,
            window=distributed.CollectiveWindow.AFTER_COMPUTE,
            depends_on=(tp,),
        ),
        distributed.CollectiveRequestDesc(
            key=dp,
            kind=distributed.CollectiveKind.REDUCE_SCATTER,
            group=distributed.ParallelGroupKind.DATA,
            group_index=0,
            backend=distributed.CollectiveBackend.NCCL,
            window=distributed.CollectiveWindow.STEP_END,
            depends_on=(pp,),
        ),
        distributed.CollectiveRequestDesc(
            key=helper,
            kind=distributed.CollectiveKind.BARRIER,
            group=distributed.ParallelGroupKind.HELPER,
            group_index=0,
            backend=distributed.CollectiveBackend.NCCL,
            window=distributed.CollectiveWindow.STEP_END,
            depends_on=(dp,),
        ),
    )


class Phase4TopologyTest(unittest.TestCase):
    def test_tp_pp_dp_groups_are_complete_and_canonically_ordered(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed)
        reordered = distributed.DistributedTopology(
            ranks=tuple(reversed(topology.ranks)),
            tp_size=topology.tp_size,
            pp_size=topology.pp_size,
            dp_size=topology.dp_size,
        )

        self.assertEqual(
            topology.groups(distributed.ParallelGroupKind.TENSOR),
            ((0, 1), (2, 3), (4, 5), (6, 7)),
        )
        self.assertEqual(
            reordered.groups(distributed.ParallelGroupKind.TENSOR),
            topology.groups(distributed.ParallelGroupKind.TENSOR),
        )
        self.assertEqual(
            reordered.groups(distributed.ParallelGroupKind.PIPELINE),
            topology.groups(distributed.ParallelGroupKind.PIPELINE),
        )
        self.assertEqual(
            reordered.groups(distributed.ParallelGroupKind.DATA),
            topology.groups(distributed.ParallelGroupKind.DATA),
        )
        self.assertEqual(
            topology.groups(distributed.ParallelGroupKind.PIPELINE),
            ((0, 2), (1, 3), (4, 6), (5, 7)),
        )
        self.assertEqual(
            topology.groups(distributed.ParallelGroupKind.DATA),
            ((0, 4), (1, 5), (2, 6), (3, 7)),
        )
        self.assertEqual(topology.groups(distributed.ParallelGroupKind.GLOBAL), (tuple(range(8)),))

    def test_helper_gpu_has_a_separate_group_and_cross_node_model_mesh_is_preserved(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed, helper=True)

        self.assertEqual(topology.groups(distributed.ParallelGroupKind.HELPER), ((8,),))
        self.assertEqual(topology.rank(0).node_id, "node-a")
        self.assertEqual(topology.rank(4).node_id, "node-b")
        self.assertNotEqual(topology.rank(0).node_id, topology.rank(4).node_id)
        self.assertEqual(topology.rank(0).local_rank, topology.rank(4).local_rank)


class Phase4StatePlanningTest(unittest.TestCase):
    def test_state_key_rejects_value_delimiters_in_namespace_and_name(self) -> None:
        distributed = _phase4(self)

        for field in ("namespace", "name"):
            for delimiter in (":", "@"):
                with self.subTest(field=field, delimiter=delimiter):
                    values = {"namespace": "model", "name": "weights"}
                    values[field] = f"{values[field]}{delimiter}part"
                    with self.assertRaises(ValueError):
                        distributed.StateKey(**values)

    def test_frontend_state_scope_and_all_shard_residences_are_typed(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed)
        requirements = (
            distributed.StateRequirementDesc(
                key=distributed.StateKey("model", "tensor"),
                scope=StateScope.SHARD,
                shard=distributed.ShardSpec(distributed.ShardKind.TENSOR, partitions=2, axis=0),
                affinity=distributed.PlacementAffinity(ranks=(0, 1, 2, 3)),
            ),
            distributed.StateRequirementDesc(
                key=distributed.StateKey("model", "pipeline"),
                scope=StateScope.SHARD,
                shard=distributed.ShardSpec(distributed.ShardKind.PIPELINE, partitions=2),
                affinity=distributed.PlacementAffinity(node_ids=("node-b",)),
            ),
            distributed.StateRequirementDesc(
                key=distributed.StateKey("model", "data"),
                scope=StateScope.SHARD,
                shard=distributed.ShardSpec(distributed.ShardKind.DATA, partitions=2, axis=0),
            ),
            distributed.StateRequirementDesc(
                key=distributed.StateKey("model", "replica"),
                scope=StateScope.REPLICA,
                shard=distributed.ShardSpec(distributed.ShardKind.REPLICATED),
                affinity=distributed.PlacementAffinity(node_ids=("node-a",)),
            ),
        )
        plan = distributed.compile_distributed_flow_plan(
            plan_epoch=1,
            topology=topology,
            state_requirements=requirements,
        )

        self.assertTrue(all(isinstance(item.scope, StateScope) for item in plan.state_requirements))
        by_name = {
            item.key.name: tuple(
                residence.shard_index for residence in plan.state_residences if residence.key == item.key
            )
            for item in plan.state_requirements
        }
        ranks_by_name = {
            item.key.name: tuple(
                residence.rank for residence in plan.state_residences if residence.key == item.key
            )
            for item in plan.state_requirements
        }
        self.assertEqual(by_name["tensor"], (0, 1, 0, 1))
        self.assertEqual(by_name["pipeline"], (0, 0, 1, 1))
        self.assertEqual(by_name["data"], (0, 0, 0, 0, 1, 1, 1, 1))
        self.assertEqual(by_name["replica"], (0, 0, 0, 0))
        self.assertEqual(ranks_by_name["tensor"], (0, 1, 2, 3))
        self.assertEqual(ranks_by_name["pipeline"], (4, 5, 6, 7))
        self.assertEqual(ranks_by_name["data"], tuple(range(8)))
        self.assertEqual(ranks_by_name["replica"], (0, 1, 2, 3))
        self.assertTrue(all(isinstance(item.placement, Placement) for item in plan.state_residences))

    def test_state_scope_and_shard_contracts_reject_untyped_or_incompatible_declarations(self) -> None:
        distributed = _phase4(self)
        typed_requirement = distributed.StateRequirementDesc(
            distributed.StateKey("model", "typed-scope"),
            StateScope.SHARD,
            distributed.ShardSpec(distributed.ShardKind.TENSOR, partitions=2, axis=0),
        )
        self.assertIsInstance(typed_requirement.scope, StateScope)
        topology = _topology(distributed)
        with self.assertRaises((TypeError, ValueError)):
            distributed.compile_distributed_flow_plan(
                plan_epoch=1,
                topology=topology,
                state_requirements=(
                    distributed.StateRequirementDesc(
                        distributed.StateKey("model", "request"),
                        StateScope.REQUEST,
                        distributed.ShardSpec(distributed.ShardKind.REPLICATED),
                    ),
                ),
            )
        with self.assertRaises((TypeError, ValueError)):
            distributed.StateRequirementDesc(
                distributed.StateKey("model", "replicated-shard"),
                StateScope.SHARD,
                distributed.ShardSpec(distributed.ShardKind.REPLICATED),
            )


class Phase4CollectivePlanningTest(unittest.TestCase):
    def test_collective_key_rejects_value_delimiters_in_namespace_and_name(self) -> None:
        distributed = _phase4(self)

        for field in ("namespace", "name"):
            for delimiter in (":", "/"):
                with self.subTest(field=field, delimiter=delimiter):
                    values = {"namespace": "train", "name": "hidden"}
                    values[field] = f"{values[field]}{delimiter}part"
                    with self.assertRaises(ValueError):
                        distributed.CollectiveKey(**values)

    def test_collective_global_order_is_window_monotonic_and_rejects_time_travel(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed)
        after = distributed.CollectiveRequestDesc(
            key=distributed.CollectiveKey("window", "after"),
            kind=distributed.CollectiveKind.ALL_REDUCE,
            group=distributed.ParallelGroupKind.GLOBAL,
            backend=distributed.CollectiveBackend.NCCL,
            window=distributed.CollectiveWindow.AFTER_COMPUTE,
        )
        before = distributed.CollectiveRequestDesc(
            key=distributed.CollectiveKey("window", "before"),
            kind=distributed.CollectiveKind.ALL_REDUCE,
            group=distributed.ParallelGroupKind.GLOBAL,
            backend=distributed.CollectiveBackend.NCCL,
            window=distributed.CollectiveWindow.BEFORE_COMPUTE,
        )
        plan = distributed.compile_distributed_flow_plan(
            plan_epoch=1,
            topology=topology,
            collective_requests=(after, before),
        )
        self.assertEqual(
            tuple(item.key for item in plan.collectives),
            (before.key, after.key),
        )

        with self.assertRaises(ValueError):
            distributed.compile_distributed_flow_plan(
                plan_epoch=2,
                topology=topology,
                collective_requests=(
                    after,
                    dataclasses.replace(before, depends_on=(after.key,)),
                ),
            )

    def test_collective_plan_builds_typed_communicators_and_rank_local_schedules(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed, helper=True)
        plan = distributed.compile_distributed_flow_plan(
            plan_epoch=2,
            topology=topology,
            collective_requests=_request_set(distributed),
        )

        self.assertEqual(
            tuple(item.key for item in plan.collectives),
            tuple(item.key for item in _request_set(distributed)),
        )
        self.assertEqual(
            {
                item.key.kind: (item.ranks, item.backend)
                for item in plan.communicators
            },
            {
                distributed.ParallelGroupKind.TENSOR: ((0, 1), distributed.CollectiveBackend.NCCL),
                distributed.ParallelGroupKind.PIPELINE: ((0, 2), distributed.CollectiveBackend.NCCL),
                distributed.ParallelGroupKind.DATA: ((0, 4), distributed.CollectiveBackend.NCCL),
                distributed.ParallelGroupKind.HELPER: ((8,), distributed.CollectiveBackend.NCCL),
            },
        )
        schedules = {item.rank: item.collectives for item in plan.local_schedules}
        self.assertEqual(schedules[0], tuple(item.key for item in plan.collectives[:3]))
        self.assertEqual(schedules[1], (plan.collectives[0].key,))
        self.assertEqual(schedules[8], (plan.collectives[3].key,))
        self.assertEqual(schedules[7], ())

    def test_different_declaration_order_has_one_structural_plan_and_signature(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed, helper=True)
        requests = _request_set(distributed)
        first = distributed.compile_distributed_flow_plan(
            plan_epoch=3,
            topology=topology,
            collective_requests=requests,
        )
        second = distributed.compile_distributed_flow_plan(
            plan_epoch=3,
            topology=topology,
            collective_requests=tuple(reversed(requests)),
        )
        self.assertEqual(first, second)
        self.assertEqual(structural_signature(first), structural_signature(second))

    def test_compile_plan_is_independent_of_state_and_collective_declaration_order(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed, helper=True)
        requirements = (
            distributed.StateRequirementDesc(
                distributed.StateKey("model", "a"),
                StateScope.REPLICA,
                distributed.ShardSpec(distributed.ShardKind.REPLICATED),
            ),
            distributed.StateRequirementDesc(
                distributed.StateKey("model-x", "a"),
                StateScope.REPLICA,
                distributed.ShardSpec(distributed.ShardKind.REPLICATED),
            ),
        )
        requests = (
            *_request_set(distributed),
            distributed.CollectiveRequestDesc(
                key=distributed.CollectiveKey("train", "a"),
                kind=distributed.CollectiveKind.ALL_REDUCE,
                group=distributed.ParallelGroupKind.GLOBAL,
                window=distributed.CollectiveWindow.BEFORE_COMPUTE,
            ),
            distributed.CollectiveRequestDesc(
                key=distributed.CollectiveKey("train-x", "a"),
                kind=distributed.CollectiveKind.ALL_REDUCE,
                group=distributed.ParallelGroupKind.GLOBAL,
                window=distributed.CollectiveWindow.BEFORE_COMPUTE,
            ),
        )
        first = distributed.compile_distributed_flow_plan(
            plan_epoch=5,
            topology=topology,
            state_requirements=requirements,
            collective_requests=requests,
        )
        second = distributed.compile_distributed_flow_plan(
            plan_epoch=5,
            topology=topology,
            state_requirements=tuple(reversed(requirements)),
            collective_requests=tuple(reversed(requests)),
        )

        self.assertEqual(first, second)
        self.assertEqual(structural_signature(first), structural_signature(second))
        self.assertEqual(
            tuple(item.key for item in first.state_requirements),
            tuple(sorted((item.key for item in requirements))),
        )
        self.assertEqual(
            tuple(item.key for item in first.collectives[:3]),
            (
                distributed.CollectiveKey("train", "a"),
                distributed.CollectiveKey("train", "tp"),
                distributed.CollectiveKey("train-x", "a"),
            ),
        )


class Phase4FailFastTest(unittest.TestCase):
    def test_bad_or_missing_topology_rank_fails_fast(self) -> None:
        distributed = _phase4(self)
        with self.assertRaises((TypeError, ValueError)):
            distributed.DistributedTopology(
                ranks=(
                    distributed.TopologyRankDesc(
                        rank=1,
                        node_id="node-a",
                        local_rank=0,
                        placement=Placement("cuda", 0),
                    ),
                ),
            )
        with self.assertRaises((TypeError, ValueError)):
            distributed.DistributedTopology(
                ranks=(
                    distributed.TopologyRankDesc(
                        rank=0,
                        node_id="node-a",
                        local_rank=0,
                        placement=Placement("cuda", 0),
                    ),
                    distributed.TopologyRankDesc(
                        rank=1,
                        node_id="node-a",
                        local_rank=0,
                        placement=Placement("cuda", 0),
                    ),
                ),
            )

    def test_bad_group_dependency_and_shard_fail_fast(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed)
        unknown_dependency = distributed.CollectiveRequestDesc(
            key=distributed.CollectiveKey("bad", "dependent"),
            kind=distributed.CollectiveKind.ALL_REDUCE,
            group=distributed.ParallelGroupKind.TENSOR,
            depends_on=(distributed.CollectiveKey("bad", "missing"),),
        )
        bad_group = distributed.CollectiveRequestDesc(
            key=distributed.CollectiveKey("bad", "group"),
            kind=distributed.CollectiveKind.ALL_REDUCE,
            group=distributed.ParallelGroupKind.TENSOR,
            group_index=99,
        )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            distributed.compile_distributed_flow_plan(
                plan_epoch=1,
                topology=topology,
                collective_requests=(unknown_dependency,),
            )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            distributed.compile_distributed_flow_plan(
                plan_epoch=1,
                topology=topology,
                collective_requests=(bad_group,),
            )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            distributed.compile_distributed_flow_plan(
                plan_epoch=1,
                topology=topology,
                state_requirements=(
                    distributed.StateRequirementDesc(
                        distributed.StateKey("bad", "shard"),
                        StateScope.SHARD,
                        distributed.ShardSpec(distributed.ShardKind.TENSOR, partitions=3, axis=0),
                    ),
                ),
            )

    def test_direct_plan_rejects_phantom_or_omitted_residences_communicators_collectives_and_schedules(self) -> None:
        distributed = _phase4(self)
        topology = _topology(distributed, helper=True)
        plan = distributed.compile_distributed_flow_plan(
            plan_epoch=4,
            topology=topology,
            state_requirements=(
                distributed.StateRequirementDesc(
                    distributed.StateKey("model", "weights"),
                    StateScope.REPLICA,
                    distributed.ShardSpec(distributed.ShardKind.REPLICATED),
                ),
            ),
            collective_requests=_request_set(distributed),
        )
        phantom_residence = distributed.StateResidenceDesc(
            distributed.StateKey("phantom", "state"),
            rank=0,
            placement=Placement("cuda", 0),
            shard_index=0,
            replica_index=0,
        )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            dataclasses.replace(
                plan,
                state_residences=(*plan.state_residences, phantom_residence),
            )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            dataclasses.replace(plan, state_residences=plan.state_residences[:-1])
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            dataclasses.replace(plan, communicators=plan.communicators[:-1])
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            dataclasses.replace(plan, collectives=plan.collectives[:-1])
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            dataclasses.replace(plan, local_schedules=plan.local_schedules[:-1])


if __name__ == "__main__":
    unittest.main()

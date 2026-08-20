"""Red contracts for Phase 4 global distributed WorkGraph planning.

The phase-4 module is intentionally imported from inside each test.  This
keeps collection useful on the pre-Phase-4 baseline while making each missing
contract an explicit, actionable failure.
"""

from __future__ import annotations

import dataclasses
import importlib
import unittest

from tllm.workgraph.frontend import StateScope
from tllm.workgraph.phase3 import Placement, ValueKey
from tllm.workgraph.signature import structural_signature


def _phase4(test_case: unittest.TestCase):
    try:
        return importlib.import_module("tllm.workgraph.phase4")
    except (ImportError, ModuleNotFoundError) as exc:
        test_case.fail(
            "Phase 4 WorkGraph contracts are not implemented: "
            "tllm.workgraph.phase4 is required"
        )
        raise AssertionError from exc


def _topology(phase4: object, *, helper: bool = False) -> object:
    """Build a two-node 2x2x2 model mesh, optionally with one helper GPU."""

    ranks = []
    for dp in range(2):
        for pp in range(2):
            for tp in range(2):
                rank = ((dp * 2) + pp) * 2 + tp
                node_id = "node-a" if rank < 4 else "node-b"
                local_rank = rank if rank < 4 else rank - 4
                ranks.append(
                    phase4.TopologyRankDesc(
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
            phase4.TopologyRankDesc(
                rank=8,
                node_id="node-b",
                local_rank=4,
                placement=Placement("cuda", 4),
                role=phase4.RankRole.HELPER,
                tp_rank=None,
                pp_rank=None,
                dp_rank=None,
            )
        )
    return phase4.DistributedTopology(
        ranks=tuple(ranks),
        tp_size=2,
        pp_size=2,
        dp_size=2,
    )


def _request_set(phase4: object) -> tuple[object, ...]:
    """A dependency chain touching TP, PP, DP, and helper communicators."""

    tp = phase4.CollectiveKey("train", "tp", 0)
    pp = phase4.CollectiveKey("train", "pp", 0)
    dp = phase4.CollectiveKey("train", "dp", 0)
    helper = phase4.CollectiveKey("train", "helper", 0)
    return (
        phase4.CollectiveRequestDesc(
            key=tp,
            kind=phase4.CollectiveKind.ALL_REDUCE,
            group=phase4.ParallelGroupKind.TENSOR,
            group_index=0,
            backend=phase4.CollectiveBackend.NCCL,
            window=phase4.CollectiveWindow.BEFORE_COMPUTE,
            value=ValueKey("hidden", 1),
        ),
        phase4.CollectiveRequestDesc(
            key=pp,
            kind=phase4.CollectiveKind.ALL_GATHER,
            group=phase4.ParallelGroupKind.PIPELINE,
            group_index=0,
            backend=phase4.CollectiveBackend.NCCL,
            window=phase4.CollectiveWindow.AFTER_COMPUTE,
            depends_on=(tp,),
        ),
        phase4.CollectiveRequestDesc(
            key=dp,
            kind=phase4.CollectiveKind.REDUCE_SCATTER,
            group=phase4.ParallelGroupKind.DATA,
            group_index=0,
            backend=phase4.CollectiveBackend.NCCL,
            window=phase4.CollectiveWindow.STEP_END,
            depends_on=(pp,),
        ),
        phase4.CollectiveRequestDesc(
            key=helper,
            kind=phase4.CollectiveKind.BARRIER,
            group=phase4.ParallelGroupKind.HELPER,
            group_index=0,
            backend=phase4.CollectiveBackend.NCCL,
            window=phase4.CollectiveWindow.STEP_END,
            depends_on=(dp,),
        ),
    )


class Phase4TopologyTest(unittest.TestCase):
    def test_tp_pp_dp_groups_are_complete_and_canonically_ordered(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4)
        reordered = phase4.DistributedTopology(
            ranks=tuple(reversed(topology.ranks)),
            tp_size=topology.tp_size,
            pp_size=topology.pp_size,
            dp_size=topology.dp_size,
        )

        self.assertEqual(
            topology.groups(phase4.ParallelGroupKind.TENSOR),
            ((0, 1), (2, 3), (4, 5), (6, 7)),
        )
        self.assertEqual(
            reordered.groups(phase4.ParallelGroupKind.TENSOR),
            topology.groups(phase4.ParallelGroupKind.TENSOR),
        )
        self.assertEqual(
            reordered.groups(phase4.ParallelGroupKind.PIPELINE),
            topology.groups(phase4.ParallelGroupKind.PIPELINE),
        )
        self.assertEqual(
            reordered.groups(phase4.ParallelGroupKind.DATA),
            topology.groups(phase4.ParallelGroupKind.DATA),
        )
        self.assertEqual(
            topology.groups(phase4.ParallelGroupKind.PIPELINE),
            ((0, 2), (1, 3), (4, 6), (5, 7)),
        )
        self.assertEqual(
            topology.groups(phase4.ParallelGroupKind.DATA),
            ((0, 4), (1, 5), (2, 6), (3, 7)),
        )
        self.assertEqual(topology.groups(phase4.ParallelGroupKind.GLOBAL), (tuple(range(8)),))

    def test_helper_gpu_has_a_separate_group_and_cross_node_model_mesh_is_preserved(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4, helper=True)

        self.assertEqual(topology.groups(phase4.ParallelGroupKind.HELPER), ((8,),))
        self.assertEqual(topology.rank(0).node_id, "node-a")
        self.assertEqual(topology.rank(4).node_id, "node-b")
        self.assertNotEqual(topology.rank(0).node_id, topology.rank(4).node_id)
        self.assertEqual(topology.rank(0).local_rank, topology.rank(4).local_rank)


class Phase4StatePlanningTest(unittest.TestCase):
    def test_state_key_rejects_value_delimiters_in_namespace_and_name(self) -> None:
        phase4 = _phase4(self)

        for field in ("namespace", "name"):
            for delimiter in (":", "@"):
                with self.subTest(field=field, delimiter=delimiter):
                    values = {"namespace": "model", "name": "weights"}
                    values[field] = f"{values[field]}{delimiter}part"
                    with self.assertRaises(ValueError):
                        phase4.StateKey(**values)

    def test_frontend_state_scope_and_all_shard_residences_are_typed(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4)
        requirements = (
            phase4.StateRequirementDesc(
                key=phase4.StateKey("model", "tensor"),
                scope=StateScope.SHARD,
                shard=phase4.ShardSpec(phase4.ShardKind.TENSOR, partitions=2, axis=0),
                affinity=phase4.PlacementAffinity(ranks=(0, 1, 2, 3)),
            ),
            phase4.StateRequirementDesc(
                key=phase4.StateKey("model", "pipeline"),
                scope=StateScope.SHARD,
                shard=phase4.ShardSpec(phase4.ShardKind.PIPELINE, partitions=2),
                affinity=phase4.PlacementAffinity(node_ids=("node-b",)),
            ),
            phase4.StateRequirementDesc(
                key=phase4.StateKey("model", "data"),
                scope=StateScope.SHARD,
                shard=phase4.ShardSpec(phase4.ShardKind.DATA, partitions=2, axis=0),
            ),
            phase4.StateRequirementDesc(
                key=phase4.StateKey("model", "replica"),
                scope=StateScope.REPLICA,
                shard=phase4.ShardSpec(phase4.ShardKind.REPLICATED),
                affinity=phase4.PlacementAffinity(node_ids=("node-a",)),
            ),
        )
        plan = phase4.compile_distributed_flow_plan(
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
        phase4 = _phase4(self)
        typed_requirement = phase4.StateRequirementDesc(
            phase4.StateKey("model", "typed-scope"),
            StateScope.SHARD,
            phase4.ShardSpec(phase4.ShardKind.TENSOR, partitions=2, axis=0),
        )
        self.assertIsInstance(typed_requirement.scope, StateScope)
        topology = _topology(phase4)
        with self.assertRaises((TypeError, ValueError)):
            phase4.compile_distributed_flow_plan(
                plan_epoch=1,
                topology=topology,
                state_requirements=(
                    phase4.StateRequirementDesc(
                        phase4.StateKey("model", "request"),
                        StateScope.REQUEST,
                        phase4.ShardSpec(phase4.ShardKind.REPLICATED),
                    ),
                ),
            )
        with self.assertRaises((TypeError, ValueError)):
            phase4.StateRequirementDesc(
                phase4.StateKey("model", "replicated-shard"),
                StateScope.SHARD,
                phase4.ShardSpec(phase4.ShardKind.REPLICATED),
            )


class Phase4CollectivePlanningTest(unittest.TestCase):
    def test_collective_key_rejects_value_delimiters_in_namespace_and_name(self) -> None:
        phase4 = _phase4(self)

        for field in ("namespace", "name"):
            for delimiter in (":", "/"):
                with self.subTest(field=field, delimiter=delimiter):
                    values = {"namespace": "train", "name": "hidden"}
                    values[field] = f"{values[field]}{delimiter}part"
                    with self.assertRaises(ValueError):
                        phase4.CollectiveKey(**values)

    def test_collective_global_order_is_window_monotonic_and_rejects_time_travel(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4)
        after = phase4.CollectiveRequestDesc(
            key=phase4.CollectiveKey("window", "after"),
            kind=phase4.CollectiveKind.ALL_REDUCE,
            group=phase4.ParallelGroupKind.GLOBAL,
            backend=phase4.CollectiveBackend.NCCL,
            window=phase4.CollectiveWindow.AFTER_COMPUTE,
        )
        before = phase4.CollectiveRequestDesc(
            key=phase4.CollectiveKey("window", "before"),
            kind=phase4.CollectiveKind.ALL_REDUCE,
            group=phase4.ParallelGroupKind.GLOBAL,
            backend=phase4.CollectiveBackend.NCCL,
            window=phase4.CollectiveWindow.BEFORE_COMPUTE,
        )
        plan = phase4.compile_distributed_flow_plan(
            plan_epoch=1,
            topology=topology,
            collective_requests=(after, before),
        )
        self.assertEqual(
            tuple(item.key for item in plan.collectives),
            (before.key, after.key),
        )

        with self.assertRaises(ValueError):
            phase4.compile_distributed_flow_plan(
                plan_epoch=2,
                topology=topology,
                collective_requests=(
                    after,
                    dataclasses.replace(before, depends_on=(after.key,)),
                ),
            )

    def test_collective_plan_builds_typed_communicators_and_rank_local_schedules(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4, helper=True)
        plan = phase4.compile_distributed_flow_plan(
            plan_epoch=2,
            topology=topology,
            collective_requests=_request_set(phase4),
        )

        self.assertEqual(
            tuple(item.key for item in plan.collectives),
            tuple(item.key for item in _request_set(phase4)),
        )
        self.assertEqual(
            {
                item.key.kind: (item.ranks, item.backend)
                for item in plan.communicators
            },
            {
                phase4.ParallelGroupKind.TENSOR: ((0, 1), phase4.CollectiveBackend.NCCL),
                phase4.ParallelGroupKind.PIPELINE: ((0, 2), phase4.CollectiveBackend.NCCL),
                phase4.ParallelGroupKind.DATA: ((0, 4), phase4.CollectiveBackend.NCCL),
                phase4.ParallelGroupKind.HELPER: ((8,), phase4.CollectiveBackend.NCCL),
            },
        )
        schedules = {item.rank: item.collectives for item in plan.local_schedules}
        self.assertEqual(schedules[0], tuple(item.key for item in plan.collectives[:3]))
        self.assertEqual(schedules[1], (plan.collectives[0].key,))
        self.assertEqual(schedules[8], (plan.collectives[3].key,))
        self.assertEqual(schedules[7], ())

    def test_different_declaration_order_has_one_structural_plan_and_signature(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4, helper=True)
        requests = _request_set(phase4)
        first = phase4.compile_distributed_flow_plan(
            plan_epoch=3,
            topology=topology,
            collective_requests=requests,
        )
        second = phase4.compile_distributed_flow_plan(
            plan_epoch=3,
            topology=topology,
            collective_requests=tuple(reversed(requests)),
        )
        self.assertEqual(first, second)
        self.assertEqual(structural_signature(first), structural_signature(second))

    def test_compile_plan_is_independent_of_state_and_collective_declaration_order(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4, helper=True)
        requirements = (
            phase4.StateRequirementDesc(
                phase4.StateKey("model", "a"),
                StateScope.REPLICA,
                phase4.ShardSpec(phase4.ShardKind.REPLICATED),
            ),
            phase4.StateRequirementDesc(
                phase4.StateKey("model-x", "a"),
                StateScope.REPLICA,
                phase4.ShardSpec(phase4.ShardKind.REPLICATED),
            ),
        )
        requests = (
            *_request_set(phase4),
            phase4.CollectiveRequestDesc(
                key=phase4.CollectiveKey("train", "a"),
                kind=phase4.CollectiveKind.ALL_REDUCE,
                group=phase4.ParallelGroupKind.GLOBAL,
                window=phase4.CollectiveWindow.BEFORE_COMPUTE,
            ),
            phase4.CollectiveRequestDesc(
                key=phase4.CollectiveKey("train-x", "a"),
                kind=phase4.CollectiveKind.ALL_REDUCE,
                group=phase4.ParallelGroupKind.GLOBAL,
                window=phase4.CollectiveWindow.BEFORE_COMPUTE,
            ),
        )
        first = phase4.compile_distributed_flow_plan(
            plan_epoch=5,
            topology=topology,
            state_requirements=requirements,
            collective_requests=requests,
        )
        second = phase4.compile_distributed_flow_plan(
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
                phase4.CollectiveKey("train", "a"),
                phase4.CollectiveKey("train", "tp"),
                phase4.CollectiveKey("train-x", "a"),
            ),
        )


class Phase4FailFastTest(unittest.TestCase):
    def test_bad_or_missing_topology_rank_fails_fast(self) -> None:
        phase4 = _phase4(self)
        with self.assertRaises((TypeError, ValueError)):
            phase4.DistributedTopology(
                ranks=(
                    phase4.TopologyRankDesc(
                        rank=1,
                        node_id="node-a",
                        local_rank=0,
                        placement=Placement("cuda", 0),
                    ),
                ),
            )
        with self.assertRaises((TypeError, ValueError)):
            phase4.DistributedTopology(
                ranks=(
                    phase4.TopologyRankDesc(
                        rank=0,
                        node_id="node-a",
                        local_rank=0,
                        placement=Placement("cuda", 0),
                    ),
                    phase4.TopologyRankDesc(
                        rank=1,
                        node_id="node-a",
                        local_rank=0,
                        placement=Placement("cuda", 0),
                    ),
                ),
            )

    def test_bad_group_dependency_and_shard_fail_fast(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4)
        unknown_dependency = phase4.CollectiveRequestDesc(
            key=phase4.CollectiveKey("bad", "dependent"),
            kind=phase4.CollectiveKind.ALL_REDUCE,
            group=phase4.ParallelGroupKind.TENSOR,
            depends_on=(phase4.CollectiveKey("bad", "missing"),),
        )
        bad_group = phase4.CollectiveRequestDesc(
            key=phase4.CollectiveKey("bad", "group"),
            kind=phase4.CollectiveKind.ALL_REDUCE,
            group=phase4.ParallelGroupKind.TENSOR,
            group_index=99,
        )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            phase4.compile_distributed_flow_plan(
                plan_epoch=1,
                topology=topology,
                collective_requests=(unknown_dependency,),
            )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            phase4.compile_distributed_flow_plan(
                plan_epoch=1,
                topology=topology,
                collective_requests=(bad_group,),
            )
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            phase4.compile_distributed_flow_plan(
                plan_epoch=1,
                topology=topology,
                state_requirements=(
                    phase4.StateRequirementDesc(
                        phase4.StateKey("bad", "shard"),
                        StateScope.SHARD,
                        phase4.ShardSpec(phase4.ShardKind.TENSOR, partitions=3, axis=0),
                    ),
                ),
            )

    def test_direct_plan_rejects_phantom_or_omitted_residences_communicators_collectives_and_schedules(self) -> None:
        phase4 = _phase4(self)
        topology = _topology(phase4, helper=True)
        plan = phase4.compile_distributed_flow_plan(
            plan_epoch=4,
            topology=topology,
            state_requirements=(
                phase4.StateRequirementDesc(
                    phase4.StateKey("model", "weights"),
                    StateScope.REPLICA,
                    phase4.ShardSpec(phase4.ShardKind.REPLICATED),
                ),
            ),
            collective_requests=_request_set(phase4),
        )
        phantom_residence = phase4.StateResidenceDesc(
            phase4.StateKey("phantom", "state"),
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

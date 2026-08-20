"""Red CPU-only contracts for Phase 4 distributed collective execution.

The distributed plan is frozen control-plane metadata.  Communicator handles,
payload tensors, and asynchronous fences are deliberately fake Python objects
in this module so the tests never initialize CUDA or ``torch.distributed``.
"""

from __future__ import annotations

from dataclasses import fields, replace
import importlib
from types import SimpleNamespace
import unittest
from unittest import mock

from tllm.runtime import residual_runtime
from tllm.runtime.state import ResidualRuntimeState
from tllm.workgraph import phase3
from tllm.workgraph.phase4 import (
    CollectiveBackend,
    CollectiveKey,
    CollectiveKind,
    CollectiveRequestDesc,
    CollectiveWindow,
    DistributedTopology,
    ParallelGroupKind,
    RankRole,
    TopologyRankDesc,
)


def _runtime(test_case: unittest.TestCase):
    try:
        return importlib.import_module("tllm.runtime.scheduling.phase4_collective")
    except (ImportError, ModuleNotFoundError) as exc:
        test_case.fail(
            "Phase 4 collective runtime is required: "
            "tllm.runtime.scheduling.phase4_collective"
        )
        raise AssertionError from exc


class _FakeTensor:
    """Tensor-shaped payload with no torch or device state."""

    def __init__(self, label: str) -> None:
        self.label = label


class _FakeFence:
    def __init__(self, *, complete: bool = False, error: BaseException | None = None) -> None:
        self.complete = complete
        self.error = error

    def query(self) -> bool:
        if self.error is not None:
            raise self.error
        return self.complete


class _FakeCollectiveAdapter:
    def __init__(self, *, fence: _FakeFence | None = None, error: BaseException | None = None) -> None:
        self.calls: list[tuple[object, object, object, object]] = []
        self.fence = fence or _FakeFence()
        self.error = error

    def launch(self, collective: object, communicator_desc: object, payload: object, communicator: object) -> _FakeFence:
        if self.error is not None:
            raise self.error
        self.calls.append((collective, communicator_desc, payload, communicator))
        return self.fence


class _FakeAgreement:
    def exchange(self, local):
        return tuple(
            replace(local, rank=rank) for rank in range(local.topology_size)
        )


def _topology(*, helper: bool = False) -> DistributedTopology:
    """A CPU/Gloo topology with a TP x PP x DP mesh and optional helper."""

    ranks: list[TopologyRankDesc] = []
    rank = 0
    for dp in range(2):
        for pp in range(2):
            for tp in range(2):
                ranks.append(
                    TopologyRankDesc(
                        rank=rank,
                        node_id=f"node-{rank // 4}",
                        local_rank=rank % 4,
                        placement=phase3.Placement("cpu"),
                        tp_rank=tp,
                        pp_rank=pp,
                        dp_rank=dp,
                    )
                )
                rank += 1
    if helper:
        ranks.append(
            TopologyRankDesc(
                rank=rank,
                node_id="helper-node",
                local_rank=0,
                placement=phase3.Placement("cpu"),
                role=RankRole.HELPER,
                tp_rank=None,
                pp_rank=None,
                dp_rank=None,
            )
        )
    return DistributedTopology(tuple(ranks), tp_size=2, pp_size=2, dp_size=2)


def _request(
    name: str,
    group: ParallelGroupKind,
    *,
    ordinal: int = 0,
    window: CollectiveWindow = CollectiveWindow.AFTER_COMPUTE,
    depends_on: tuple[CollectiveKey, ...] = (),
) -> CollectiveRequestDesc:
    return CollectiveRequestDesc(
        key=CollectiveKey("phase4.runtime", name, ordinal),
        kind=CollectiveKind.ALL_REDUCE,
        group=group,
        backend=CollectiveBackend.GLOO,
        window=window,
        depends_on=depends_on,
    )


def _plan(
    *,
    epoch: int = 1,
    topology: DistributedTopology | None = None,
    requests: tuple[CollectiveRequestDesc, ...] | None = None,
):
    runtime = importlib.import_module("tllm.workgraph.phase4")
    topology = _topology() if topology is None else topology
    requests = (
        (_request("first", ParallelGroupKind.GLOBAL),)
        if requests is None
        else requests
    )
    return runtime.compile_distributed_flow_plan(
        plan_epoch=epoch,
        topology=topology,
        collective_requests=requests,
    )


def _executor(
    test_case: unittest.TestCase,
    *,
    rank: int = 0,
    plan: object | None = None,
    adapter: _FakeCollectiveAdapter | None = None,
    communicators: tuple[tuple[object, object], ...] | None = None,
):
    runtime = _runtime(test_case)
    plan = _plan() if plan is None else plan
    adapter = _FakeCollectiveAdapter() if adapter is None else adapter
    required = set()
    if 0 <= rank < len(plan.local_schedules):
        required = {
            item.communicator
            for item in plan.collectives
            if item.key in plan.local_schedules[rank].collectives
        }
    if communicators is None:
        communicators = tuple((key, SimpleNamespace(owner_rank=rank)) for key in required)
    return runtime.DistributedPlanExecutor(
        plan=plan,
        rank=rank,
        communicators=communicators,
        adapter=adapter,
    ), adapter


class Phase4CollectiveRuntimeUnitTest(unittest.TestCase):
    def test_runtime_handles_keep_communicators_payloads_and_fences_python_owned(self) -> None:
        runtime = _runtime(self)
        adapter = _FakeCollectiveAdapter(fence=_FakeFence(complete=False))
        executor, _ = _executor(self, adapter=adapter)
        payload = _FakeTensor("hidden")
        executor.prepare_step(7)
        executor.advance_window(7, CollectiveWindow.AFTER_COMPUTE)
        key = executor.plan.local_schedules[0].collectives[0]
        handle = executor.launch(
            7,
            collective=key,
            payload=payload,
        )

        self.assertIs(handle.fence, adapter.fence)
        self.assertIs(adapter.calls[0][2], payload)
        self.assertIs(adapter.calls[0][3], next(iter(executor._communicators.values())))
        structural_fields = {field.name for field in fields(executor.plan)}
        self.assertNotIn("communicator", structural_fields)
        self.assertNotIn("payload", structural_fields)
        self.assertNotIn("fence", structural_fields)
        self.assertEqual(handle.step_id, 7)
        self.assertIsInstance(handle, runtime.CollectiveHandle)

    def test_prepare_launch_and_finish_follow_frozen_local_order_and_window(self) -> None:
        first = _request("first", ParallelGroupKind.GLOBAL, window=CollectiveWindow.PREPARE)
        second = _request(
            "second",
            ParallelGroupKind.GLOBAL,
            window=CollectiveWindow.AFTER_COMPUTE,
            depends_on=(first.key,),
        )
        plan = _plan(requests=(second, first))
        executor, adapter = _executor(self, plan=plan)
        executor.prepare_step(11)
        expected = plan.local_schedules[0].collectives
        self.assertEqual(expected, (first.key, second.key))

        with self.assertRaises(RuntimeError):
            executor.launch(
                11,
                collective=second.key,
                payload=_FakeTensor("out-of-order"),
            )
        executor.launch(
            11,
            collective=first.key,
            payload=_FakeTensor("first"),
        )
        with self.assertRaises(RuntimeError):
            executor.launch(
                11,
                collective=second.key,
                payload=_FakeTensor("wrong-window"),
            )
        executor.advance_window(11, CollectiveWindow.AFTER_COMPUTE)
        executor.launch(
            11,
            collective=second.key,
            payload=_FakeTensor("second"),
        )
        executor.finish_step(11)
        self.assertEqual([call[0].key for call in adapter.calls], [first.key, second.key])
        with self.assertRaises(RuntimeError):
            executor.launch(
                11,
                collective=first.key,
                payload=_FakeTensor("closed"),
            )

    def test_out_of_order_wrong_rank_and_missing_communicator_fail_fast(self) -> None:
        plan = _plan()
        with self.assertRaises(ValueError):
            _executor(self, plan=plan, rank=len(plan.topology.ranks))

        runtime = _runtime(self)
        with self.assertRaises(ValueError):
            runtime.DistributedPlanExecutor(
                plan=plan,
                rank=0,
                communicators=(),
                adapter=_FakeCollectiveAdapter(),
            )

        executor, _ = _executor(self, plan=plan)
        with self.assertRaises(RuntimeError):
            executor.launch(
                1,
                collective=plan.local_schedules[0].collectives[0],
                payload=_FakeTensor("unprepared"),
            )

    def test_failed_adapter_or_fence_poison_executor_and_fails_fast(self) -> None:
        plan = _plan()
        adapter = _FakeCollectiveAdapter(error=RuntimeError("launch failed"))
        executor, _ = _executor(self, plan=plan, adapter=adapter)
        executor.prepare_step(3)
        executor.advance_window(3, CollectiveWindow.AFTER_COMPUTE)
        key = plan.local_schedules[0].collectives[0]
        with self.assertRaises(RuntimeError):
            executor.launch(3, collective=key, payload=_FakeTensor("x"))
        with self.assertRaises(RuntimeError):
            executor.prepare_step(4)

        failing_fence = _FakeFence(error=RuntimeError("fence failed"))
        executor, _ = _executor(self, plan=plan, adapter=_FakeCollectiveAdapter(fence=failing_fence))
        executor.prepare_step(5)
        executor.advance_window(5, CollectiveWindow.AFTER_COMPUTE)
        executor.launch(5, collective=key, payload=_FakeTensor("x"))
        with self.assertRaises(RuntimeError):
            executor.poll()
        with self.assertRaises(RuntimeError):
            executor.finish_step(5)

    def test_old_epoch_is_not_reusable_while_collective_is_in_flight(self) -> None:
        runtime = _runtime(self)
        fence = _FakeFence(complete=False)
        executor, _ = _executor(self, adapter=_FakeCollectiveAdapter(fence=fence))
        executor.prepare_step(9)
        executor.advance_window(9, CollectiveWindow.AFTER_COMPUTE)
        key = executor.plan.local_schedules[0].collectives[0]
        executor.launch(9, collective=key, payload=_FakeTensor("x"))
        self.assertFalse(executor.reusable)
        self.assertEqual(executor.in_flight, 1)
        fence.complete = True
        self.assertEqual(executor.poll(), 1)
        executor.finish_step(9)
        self.assertTrue(executor.reusable)

    def test_all_ranks_project_one_global_collective_sequence(self) -> None:
        topology = _topology(helper=True)
        requests = (
            _request("global", ParallelGroupKind.GLOBAL),
            _request("tensor", ParallelGroupKind.TENSOR),
            _request("pipeline", ParallelGroupKind.PIPELINE),
            _request("data", ParallelGroupKind.DATA),
            _request("helper", ParallelGroupKind.HELPER),
        )
        plan = _plan(topology=topology, requests=requests)
        global_keys = tuple(item.key for item in plan.collectives)
        self.assertEqual(global_keys, tuple(sorted(global_keys, key=lambda key: key.value)))
        by_key = {item.key: item for item in plan.collectives}
        by_comm = {item.key: item for item in plan.communicators}
        for schedule in plan.local_schedules:
            self.assertEqual(
                schedule.collectives,
                tuple(
                    key
                    for key in global_keys
                    if schedule.rank in by_comm[by_key[key].communicator].ranks
                ),
            )

        # The helper has its own local communicator, while model ranks retain
        # the same global order projection.  No rank invents a local order.
        helper_schedule = plan.local_schedules[-1]
        self.assertEqual(helper_schedule.rank, 8)
        self.assertEqual(
            helper_schedule.collectives,
            (CollectiveKey("phase4.runtime", "helper", 0),),
        )

        # Every rank can execute only its projection, while the projection is
        # still derived from one global order.  The adapter trace is the
        # runtime witness, including the helper-only rank.
        for rank, schedule in enumerate(plan.local_schedules):
            adapter = _FakeCollectiveAdapter(fence=_FakeFence(complete=True))
            executor, _ = _executor(self, rank=rank, plan=plan, adapter=adapter)
            executor.prepare_step(20 + rank)
            executor.advance_window(20 + rank, CollectiveWindow.AFTER_COMPUTE)
            for key in schedule.collectives:
                executor.launch(
                    20 + rank,
                    collective=key,
                    payload=_FakeTensor(f"rank-{rank}-{key.name}"),
                )
            executor.finish_step(20 + rank)
            self.assertEqual(
                tuple(call[0].key for call in adapter.calls),
                schedule.collectives,
            )

    def test_residual_runtime_cold_publication_and_launch_use_prepared_epoch(self) -> None:
        """The legacy path must remain unchanged when no distributed binding exists."""

        runtime = _runtime(self)
        adapter = _FakeCollectiveAdapter(fence=_FakeFence(complete=True))
        binding = runtime.DistributedRuntimeBinding(
            topology=_topology(),
            rank=0,
            communicators=((
                next(iter({item.communicator for item in _plan().collectives})),
                SimpleNamespace(owner_rank=0),
            ),),
            adapter=adapter,
            agreement=_FakeAgreement(),
            collective_requests=(_request("first", ParallelGroupKind.GLOBAL),),
        )
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.configure_workgraph_distributed(binding)
            residual_runtime.publish_workgraph_distributed(plan_epoch=17)
            residual_runtime.prepare_workgraph_step(31)
            plan = state.distributed_plan_store.active(31)
            key = plan.local_schedules[0].collectives[0]
            state.distributed_plan_store.executor(31).advance_window(
                31, CollectiveWindow.AFTER_COMPUTE
            )
            residual_runtime.launch_workgraph_collective(
                31,
                collective=key,
                payload=_FakeTensor("prepared"),
            )
            self.assertEqual(adapter.calls[0][0].key, key)
            self.assertEqual(adapter.calls[0][0].sequence, 0)
            self.assertEqual(state.distributed_plan_store.active(31).plan_epoch, 17)

    def test_no_distributed_binding_preserves_existing_prepare_behavior(self) -> None:
        state = ResidualRuntimeState()
        marker: list[int] = []
        store = mock.Mock()
        store.prepare_step.side_effect = lambda step: marker.append(int(step))
        state.physical_plan_store = store
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.prepare_workgraph_step(23)
        self.assertEqual(marker, [23])
        self.assertIsNone(getattr(state, "distributed_runtime_binding", None))


if __name__ == "__main__":
    unittest.main()

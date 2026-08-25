from __future__ import annotations

import unittest
from unittest import mock
from types import SimpleNamespace
from typing import Any, cast

from tllm.runtime.workload_execution import (
    NativeWorkloadExecution,
    NativeWorkloadExecutionStore,
    _NativeShapeSpecializationQueue,
)
from tllm.runtime import residual_runtime
from tllm.runtime.state import ResidualRuntimeState
from tllm.contracts.shape_specialization import (
    RegionShapeSpecializationInput,
    RegionShapeSpecializationRequest,
)


class _Executor:
    def __init__(self) -> None:
        self.prepared: list[int] = []
        self.finished: list[int] = []

    def prepare_step(self, step_id: int) -> None:
        self.prepared.append(step_id)

    def finish_step(self, step_id: int) -> None:
        self.finished.append(step_id)


def _execution(epoch: int) -> tuple[NativeWorkloadExecution, _Executor]:
    executor = _Executor()
    compilation = SimpleNamespace(
        physical=SimpleNamespace(plan_epoch=epoch),
        graph=SimpleNamespace(workload_key=f"epoch-{epoch}"),
    )
    execution = NativeWorkloadExecution(
        compilation=cast(Any, compilation),
        executor=cast(Any, executor),
        completion_anchor="done",
        _owned_consumers=(),
    )
    return execution, executor


class NativeWorkloadExecutionStoreUnitTest(unittest.TestCase):
    def test_staged_epoch_switches_only_after_prior_step_retires(self) -> None:
        first, first_executor = _execution(4)
        second, second_executor = _execution(5)
        store = NativeWorkloadExecutionStore(first)

        self.assertIs(store.prepare_step(10), first)
        store.stage(second)
        self.assertIs(store.prepare_step(10), first)
        self.assertIs(store.current, first)

        self.assertTrue(store.finish_step(10))
        self.assertIs(store.prepare_step(11), second)
        self.assertIs(store.current, second)
        self.assertEqual(store.retired_count, 1)
        self.assertEqual(first_executor.finished, [10])
        self.assertEqual(second_executor.prepared, [11])

        self.assertEqual(store.collect_retired(), 1)
        self.assertEqual(store.retired_count, 0)

    def test_stage_requires_a_strictly_newer_epoch(self) -> None:
        first, _ = _execution(4)
        store = NativeWorkloadExecutionStore(first)

        for epoch in (3, 4):
            candidate, _ = _execution(epoch)
            with self.assertRaisesRegex(ValueError, "strictly newer"):
                store.stage(candidate)

    def test_abort_closes_only_the_staged_epoch(self) -> None:
        first, first_executor = _execution(1)
        staged, staged_executor = _execution(2)
        store = NativeWorkloadExecutionStore(first)
        store.stage(staged)

        self.assertTrue(store.abort_staged())
        self.assertFalse(store.abort_staged())
        self.assertIs(store.current, first)
        self.assertEqual(first_executor.finished, [])
        self.assertEqual(staged_executor.finished, [])

    def test_next_step_cannot_prepare_before_previous_step_retires(self) -> None:
        first, _ = _execution(1)
        store = NativeWorkloadExecutionStore(first)
        store.prepare_step(7)

        with self.assertRaisesRegex(RuntimeError, "must retire"):
            store.prepare_step(8)

    def test_activation_gate_blocks_mp_promotion_until_global_vote(self) -> None:
        first, _ = _execution(1)
        replacement, _ = _execution(2)
        store = NativeWorkloadExecutionStore(first)
        store.stage(replacement, activation_required=True)

        self.assertIs(store.prepare_step(10), first)
        store.finish_step(10)
        self.assertIs(store.prepare_step(11), first)
        self.assertFalse(store.staged_activated)
        store.finish_step(11)

        self.assertTrue(store.activate_staged(2))
        self.assertTrue(store.staged_activated)
        self.assertIs(store.prepare_step(12), replacement)

    def test_runtime_safe_point_promotes_only_on_the_next_step(self) -> None:
        first, first_executor = _execution(4)
        second, second_executor = _execution(5)
        store = NativeWorkloadExecutionStore(first)
        state = ResidualRuntimeState()
        state.workload_execution_store = store
        state.workload_execution = first
        state.physical_executor = first

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.prepare_physical_workgraph_step(10)
            store.stage(second)
            residual_runtime.prepare_physical_workgraph_step(10)

            self.assertIs(state.workload_execution, first)
            self.assertEqual(second_executor.prepared, [])

            residual_runtime.prepare_physical_workgraph_step(11)

        self.assertIs(state.workload_execution, second)
        self.assertIs(state.workload_compilation, second.compilation)
        self.assertIs(state.physical_executor, second)
        self.assertEqual(first_executor.finished, [10])
        self.assertEqual(second_executor.prepared, [11])
        self.assertEqual(store.retired_count, 0)

    def test_runtime_safe_point_acknowledges_shape_request_after_staging(self) -> None:
        first, _ = _execution(4)
        queue = _NativeShapeSpecializationQueue()
        request = RegionShapeSpecializationRequest(
            base_cache_key="a" * 64,
            base_signature="b" * 64,
            observed_plan_epoch=4,
            region_id="region-a",
            artifact_key="c" * 64,
            inputs=(
                RegionShapeSpecializationInput(
                    "input-a", (2, 4), "torch.float32", "cuda:0"
                ),
            ),
        )
        queue.enqueue(request)
        first._shape_specializations = queue
        store = NativeWorkloadExecutionStore(first)
        state = ResidualRuntimeState(
            workload_execution_store=store,
            workload_execution=first,
            physical_executor=first,
        )
        owner = mock.Mock()
        owner.apply_region_shape_specializations.return_value = True

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch("tllm.runtime.runtime.active_runtime", return_value=owner),
        ):
            residual_runtime.prepare_physical_workgraph_step(10)

        owner.apply_region_shape_specializations.assert_called_once_with((request,))
        self.assertEqual(first.peek_shape_specialization_requests(), ())


if __name__ == "__main__":
    unittest.main()

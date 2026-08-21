from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest import mock

from tllm.consumers.base import BaseConsumer
from tllm.runtime import residual_runtime
from tllm.runtime.scheduling.workgraph_executor import WorkGraphRuntimeStats
from tllm.runtime.state import ResidualRuntimeState


class _StatsConsumer(BaseConsumer[object]):
    def __init__(self, stats: WorkGraphRuntimeStats) -> None:
        self._stats = stats
        self.read_count = 0

    @property
    def consumer_id(self) -> str:
        return "phase2.stats"

    def read_workgraph_runtime_stats(self) -> WorkGraphRuntimeStats:
        self.read_count += 1
        return self._stats


class _PhysicalStatsOwner:
    def __init__(self, stats: WorkGraphRuntimeStats) -> None:
        self._stats = stats
        self.read_count = 0

    def read_workgraph_runtime_stats(self) -> WorkGraphRuntimeStats:
        self.read_count += 1
        return self._stats


def _stats(**overrides: int) -> WorkGraphRuntimeStats:
    values = {
        "active_plan_epoch": 0,
        "prepare_count": 0,
        "execute_count": 0,
        "finish_count": 0,
        "replay_count": 0,
        "eager_count": 0,
        "fallback_count": 0,
        "fence_wait_count": 0,
        "fence_record_count": 0,
        "bucket_miss_count": 0,
        "specialization_request_count": 0,
        "active_leases": 0,
        "retired_plans": 0,
        "deadline_flush_count": 0,
    }
    values.update(overrides)
    return WorkGraphRuntimeStats(**values)


class GpuGateGpuGateStatsUnitTest(unittest.TestCase):
    def test_workgraph_runtime_stats_is_public_frozen_and_covers_the_gpu_gate_fields(self) -> None:
        stats = _stats(
            active_plan_epoch=11,
            prepare_count=2,
            execute_count=3,
            finish_count=4,
            replay_count=5,
            eager_count=6,
            fallback_count=7,
            fence_wait_count=8,
            fence_record_count=9,
            bucket_miss_count=10,
            specialization_request_count=11,
            active_leases=12,
            retired_plans=13,
            deadline_flush_count=14,
        )

        self.assertIsInstance(stats, WorkGraphRuntimeStats)
        self.assertEqual(stats.active_plan_epoch, 11)
        self.assertEqual(stats.prepare_count, 2)
        self.assertEqual(stats.execute_count, 3)
        self.assertEqual(stats.finish_count, 4)
        self.assertEqual(stats.replay_count, 5)
        self.assertEqual(stats.eager_count, 6)
        self.assertEqual(stats.fallback_count, 7)
        self.assertEqual(stats.fence_wait_count, 8)
        self.assertEqual(stats.fence_record_count, 9)
        self.assertEqual(stats.bucket_miss_count, 10)
        self.assertEqual(stats.specialization_request_count, 11)
        self.assertEqual(stats.active_leases, 12)
        self.assertEqual(stats.retired_plans, 13)
        self.assertEqual(stats.deadline_flush_count, 14)
        with self.assertRaises(FrozenInstanceError):
            stats.execute_count = 99  # type: ignore[misc]

    def test_runtime_read_aggregates_physical_consumer_and_placement_stats_without_reset(self) -> None:
        physical = _PhysicalStatsOwner(
            _stats(
                active_plan_epoch=11,
                prepare_count=2,
                execute_count=3,
                finish_count=4,
                replay_count=5,
                eager_count=6,
                fallback_count=7,
                fence_wait_count=8,
                fence_record_count=9,
                bucket_miss_count=10,
                specialization_request_count=11,
                active_leases=12,
                retired_plans=13,
            )
        )
        consumer = _StatsConsumer(
            _stats(
                prepare_count=20,
                execute_count=30,
                finish_count=40,
                replay_count=50,
                eager_count=60,
                fallback_count=70,
                fence_wait_count=80,
                fence_record_count=90,
                bucket_miss_count=100,
                specialization_request_count=110,
            )
        )
        state = ResidualRuntimeState()
        state.consumer = consumer
        state.physical_executor = physical  # type: ignore[attr-defined]
        state.placement_stats.deadline_flush_count = 17

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            first = residual_runtime.read_workgraph_runtime_stats()
            second = residual_runtime.read_workgraph_runtime_stats()

        self.assertIsInstance(first, WorkGraphRuntimeStats)
        self.assertEqual(first, second)
        self.assertEqual(first.active_plan_epoch, 11)
        self.assertEqual(first.prepare_count, 22)
        self.assertEqual(first.execute_count, 33)
        self.assertEqual(first.finish_count, 44)
        self.assertEqual(first.replay_count, 55)
        self.assertEqual(first.eager_count, 66)
        self.assertEqual(first.fallback_count, 77)
        self.assertEqual(first.fence_wait_count, 88)
        self.assertEqual(first.fence_record_count, 99)
        self.assertEqual(first.bucket_miss_count, 110)
        self.assertEqual(first.specialization_request_count, 121)
        self.assertEqual(first.active_leases, 12)
        self.assertEqual(first.retired_plans, 13)
        self.assertEqual(first.deadline_flush_count, 17)
        self.assertEqual(physical.read_count, 2)
        self.assertEqual(consumer.read_count, 2)

        self.assertTrue(callable(BaseConsumer.read_workgraph_runtime_stats))


if __name__ == "__main__":
    unittest.main()

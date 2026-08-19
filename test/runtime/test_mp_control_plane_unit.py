#!/usr/bin/env python3
"""Unit tests for the worker-extension control plane (A2)."""

from __future__ import annotations

import os
import unittest
from dataclasses import asdict
from unittest import mock

from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.contracts.consumer_stats import TrainingStats, merge_training_stats_maps
from tllm.runtime import residual_runtime
from tllm.runtime.mp.control import (
    ControlPlane,
    TLLM_WORKER_EXTENSION_QUALNAME,
    WorkerStatsReply,
    set_active_control_plane,
)
from tllm.runtime.stats import DistillerTimingStats, PathHotspotStats, PlacementStats
from tllm.runtime.mp.worker_extension import TllmWorkerExtension, _to_plain


class _FakeEngineCore:
    """Stands in for the EngineCore client; records collective_rpc calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, float | None]] = []
        self.handler = TllmWorkerExtension()

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((str(method), tuple(args), timeout))
        return [getattr(self.handler, str(method))(*args)]


class _FakeLLM:
    def __init__(self) -> None:
        self.llm_engine = type("Engine", (), {"engine_core": _FakeEngineCore()})()


class ControlPlaneUnitTest(unittest.TestCase):
    def test_extension_qualname_resolves(self) -> None:
        from vllm.utils.import_utils import resolve_obj_by_qualname

        self.assertIs(resolve_obj_by_qualname(TLLM_WORKER_EXTENSION_QUALNAME), TllmWorkerExtension)

    def test_call_uses_string_method_names(self) -> None:
        llm = _FakeLLM()
        plane = ControlPlane(llm)
        replies = plane.ping()
        self.assertEqual(len(replies), 1)
        method, _args, _timeout = llm.llm_engine.engine_core.calls[0]
        self.assertEqual(method, "tllm_ping")  # str, never a closure
        self.assertEqual(replies[0]["pid"], os.getpid())

    def test_push_request_mapping_serializes_rows(self) -> None:
        llm = _FakeLLM()
        plane = ControlPlane(llm)
        saved_prompt = dict(residual_runtime.RUNTIME.reqid_to_promptidx)
        saved_sample = dict(residual_runtime.RUNTIME.reqid_to_sampleidx)
        try:
            plane.push_request_mapping([("req-a", 3, 0), ("0_req-b", 4, 1)])
            self.assertEqual(residual_runtime.RUNTIME.reqid_to_promptidx["req-a"], 3)
            self.assertEqual(residual_runtime.RUNTIME.reqid_to_sampleidx["0_req-b"], 1)
            plane.push_request_mapping([])  # empty: no RPC at all
            methods = [c[0] for c in llm.llm_engine.engine_core.calls]
            self.assertEqual(methods, ["tllm_push_request_mapping"])
        finally:
            residual_runtime.RUNTIME.reqid_to_promptidx = saved_prompt
            residual_runtime.RUNTIME.reqid_to_sampleidx = saved_sample

    def test_apply_enable_flags_round_trips(self) -> None:
        llm = _FakeLLM()
        plane = ControlPlane(llm)
        saved = residual_runtime.RUNTIME.test_time_lora_enabled
        try:
            plane.apply_enable_flags({"test_time_lora_enabled": True})
            self.assertTrue(residual_runtime.RUNTIME.test_time_lora_enabled)
            with self.assertRaises(ValueError):
                plane.apply_enable_flags({"nonsense_flag": True})
        finally:
            residual_runtime.RUNTIME.test_time_lora_enabled = saved

    def test_register_consumer_builds_via_registry(self) -> None:
        llm = _FakeLLM()
        plane = ControlPlane(llm)
        saved = residual_runtime.RUNTIME.consumer
        try:
            from tllm.common.dataclass_json import dataclass_to_jsonable

            config = dataclass_to_jsonable(DummyConsumerConfig(consumer_id="plane_dummy"))
            plane.register_consumer("dummy", config, primary=True)
            consumer = residual_runtime.RUNTIME.consumer
            self.assertIsNotNone(consumer)
            self.assertEqual(consumer.config.consumer_id, "plane_dummy")
        finally:
            residual_runtime.set_runtime_consumer(saved)

    def test_pull_stats_returns_plain_data(self) -> None:
        llm = _FakeLLM()
        plane = ControlPlane(llm)
        saved = residual_runtime.RUNTIME.consumer
        try:
            residual_runtime.set_runtime_consumer(None)
            (reply,) = plane.pull_stats()
            self.assertEqual(reply.esamp.loss_count, 0)
            self.assertEqual(reply.esamp_per_request, {})
            self.assertIsInstance(reply.distiller_timing, DistillerTimingStats)
            self.assertIsInstance(reply.path_hotspot, PathHotspotStats)
            self.assertIsInstance(reply.placement, PlacementStats)
            self.assertEqual(reply.workgraph.execute_count, 0)
            import json

            json.dumps(asdict(reply))
        finally:
            residual_runtime.set_runtime_consumer(saved)

    def test_category_pull_does_not_reset_unrequested_stats(self) -> None:
        llm = _FakeLLM()
        plane = ControlPlane(llm)
        saved = residual_runtime.RUNTIME.placement_stats
        try:
            residual_runtime.RUNTIME.placement_stats = PlacementStats(moved_dispatch_count=3)
            plane.pull_stats("esamp")
            self.assertEqual(residual_runtime.RUNTIME.placement_stats.moved_dispatch_count, 3)
            (reply,) = plane.pull_stats("placement")
            self.assertEqual(reply.placement.moved_dispatch_count, 3)
            self.assertEqual(residual_runtime.RUNTIME.placement_stats.moved_dispatch_count, 0)
        finally:
            residual_runtime.RUNTIME.placement_stats = saved

    def test_fast_weight_options_round_trip_without_magic_mapping(self) -> None:
        plane = ControlPlane(_FakeLLM())
        with mock.patch("tllm.runtime.residual_runtime.set_fast_weight_enabled") as setter:
            plane.set_fast_weight_enabled(
                True,
                layers=(2, 4),
                chunk_size=128,
                learning_rate=0.25,
                mode="exact",
                decode_update_enabled=True,
                key_mode="request_sample",
                max_slots=7,
                max_bytes=1024,
                fail_on_unsupported=True,
            )
        setter.assert_called_once_with(
            True,
            layers=[2, 4],
            chunk_size=128,
            learning_rate=0.25,
            mode="exact",
            decode_update_enabled=True,
            key_mode="request_sample",
            max_slots=7,
            max_bytes=1024,
            fail_on_unsupported=True,
        )

    def test_esamp_live_toggle_rebuilds_dispatch_plan(self) -> None:
        runtime = residual_runtime.RUNTIME
        saved = (
            runtime.consumer,
            runtime.consumer_registry,
            runtime.dispatch_plan,
            runtime.execution_plan,
        )
        try:
            consumer = ESampConsumer(ESampConsumerConfig(enable_esamp_training=False))
            residual_runtime.set_runtime_consumer(consumer)
            self.assertIsNone(runtime.dispatch_plan)

            residual_runtime.set_esamp_training_enabled(True)
            self.assertIsNotNone(runtime.dispatch_plan)

            residual_runtime.set_esamp_training_enabled(False)
            self.assertIsNone(runtime.dispatch_plan)
        finally:
            (
                runtime.consumer,
                runtime.consumer_registry,
                runtime.dispatch_plan,
                runtime.execution_plan,
            ) = saved

    def test_call_requires_llm_engine(self) -> None:
        with self.assertRaises(ValueError):
            ControlPlane(object()).ping()


class ToPlainUnitTest(unittest.TestCase):
    def test_rejects_tensors(self) -> None:
        import torch

        with self.assertRaises(TypeError):
            _to_plain({"loss": torch.zeros(1)})

    def test_accepts_nested_plain_data(self) -> None:
        self.assertEqual(_to_plain({"a": [1, 2.5, "x", None], "b": (True, 3)}), {"a": [1, 2.5, "x", None], "b": [True, 3]})


class WorkerStatsMergeUnitTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_active_control_plane(None)

    @staticmethod
    def _reply(*, loss_avg: float, loss_count: int, request_key: int, queue_depth: int) -> WorkerStatsReply:
        return WorkerStatsReply.from_wire(
            {
                "esamp": {"loss_avg": loss_avg, "loss_count": loss_count},
                "esamp_per_request": {
                    str(request_key): {"loss_avg": loss_avg, "loss_count": loss_count}
                },
                "distiller_timing": {"precompute_ms_total": loss_avg, "precompute_count": loss_count},
                "path_hotspot": {"cpu_ms_total": {"dispatch": loss_avg}, "counts": {"dispatch": loss_count}},
                "placement": {"moved_dispatch_count": loss_count, "max_queue_depth": queue_depth},
                "workgraph": {"execute_count": loss_count, "active_plan_epoch": queue_depth},
            }
        )

    def test_typed_stats_merge_uses_field_semantics(self) -> None:
        first = self._reply(loss_avg=2.0, loss_count=1, request_key=4, queue_depth=2)
        second = self._reply(loss_avg=5.0, loss_count=3, request_key=4, queue_depth=7)

        training = TrainingStats.merge((first.esamp, second.esamp))
        self.assertEqual(training.loss_count, 4)
        self.assertAlmostEqual(training.loss_avg, 4.25)
        per_request = merge_training_stats_maps((first.esamp_per_request, second.esamp_per_request))
        self.assertAlmostEqual(per_request[4].loss_avg, 4.25)
        timing = DistillerTimingStats.merge((first.distiller_timing, second.distiller_timing))
        self.assertEqual(timing.precompute_count, 4)
        self.assertAlmostEqual(timing.precompute_ms_total, 7.0)
        hotspots = PathHotspotStats.merge((first.path_hotspot, second.path_hotspot))
        self.assertEqual(hotspots.counts["dispatch"], 4)
        self.assertAlmostEqual(hotspots.cpu_ms_total["dispatch"], 7.0)
        placement = PlacementStats.merge((first.placement, second.placement))
        self.assertEqual(placement.moved_dispatch_count, 4)
        self.assertEqual(placement.max_queue_depth, 7)
        self.assertEqual(first.workgraph.execute_count, 1)
        self.assertEqual(second.workgraph.active_plan_epoch, 7)

    def test_public_stats_api_merges_active_worker_replies(self) -> None:
        reply = self._reply(loss_avg=3.5, loss_count=2, request_key=9, queue_depth=6)
        plane = mock.Mock()
        plane.pull_stats.return_value = [reply]
        saved_consumer = residual_runtime.RUNTIME.consumer
        saved_placement = residual_runtime.RUNTIME.placement_stats
        try:
            residual_runtime.RUNTIME.consumer = None
            residual_runtime.RUNTIME.placement_stats = PlacementStats(
                moved_dispatch_count=1,
                max_queue_depth=2,
            )
            with mock.patch("tllm.runtime.mp.control.active_control_plane", return_value=plane):
                training = residual_runtime.read_and_reset_esamp_stats(sync=False)
                placement = residual_runtime.read_and_reset_placement_stats()
            self.assertEqual(training, TrainingStats(loss_avg=3.5, loss_count=2))
            self.assertEqual(placement.moved_dispatch_count, 3)
            self.assertEqual(placement.max_queue_depth, 6)
            self.assertEqual(
                [call.args[0] for call in plane.pull_stats.call_args_list],
                ["esamp", "placement"],
            )
        finally:
            residual_runtime.RUNTIME.consumer = saved_consumer
            residual_runtime.RUNTIME.placement_stats = saved_placement


if __name__ == "__main__":
    unittest.main()

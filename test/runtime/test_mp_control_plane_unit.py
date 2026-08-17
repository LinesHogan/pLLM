#!/usr/bin/env python3
"""Unit tests for the worker-extension control plane (A2)."""

from __future__ import annotations

import os
import unittest

from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.runtime import residual_runtime
from tllm.runtime.mp.control import ControlPlane, TLLM_WORKER_EXTENSION_QUALNAME
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
            self.assertEqual(set(reply), {"esamp", "esamp_per_request", "distiller_timing", "path_hotspot"})
            import json

            json.dumps(reply)  # must be wire-safe plain data
        finally:
            residual_runtime.set_runtime_consumer(saved)

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


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for the instance-scoped TllmRuntime and the legacy façade."""

from __future__ import annotations

import unittest
from unittest import mock

from tllm.runtime.config import (
    BufferPolicy,
    ConsumerErrorPolicy,
    FailurePolicy,
    TllmRuntimeConfig,
)
from tllm.runtime.runtime import (
    TllmRuntime,
    active_runtime,
    bind_runner_runtime,
    default_runtime,
    resolve_runner_runtime,
)


class _Consumer:
    def __init__(self, cid: str) -> None:
        self.consumer_id = cid

    def flows(self):
        return ()


class _Runner:
    pass


class TllmRuntimeUnitTest(unittest.TestCase):
    def test_default_runtime_is_singleton(self) -> None:
        self.assertIs(default_runtime(), default_runtime())

    def test_non_auto_backend_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TllmRuntime(backend="vllm-0.24")

    def test_runner_binding_roundtrip_and_weak_cleanup(self) -> None:
        runtime = TllmRuntime()
        runner = _Runner()
        bind_runner_runtime(runner, runtime)
        self.assertIs(resolve_runner_runtime(runner), runtime)
        del runner
        import gc

        gc.collect()
        # entry vanished with the runner
        self.assertEqual(resolve_runner_runtime(_Runner()), None)

    def test_activate_routes_and_applies_consumers(self) -> None:
        runtime = TllmRuntime(consumers=[_Consumer("c1")])
        with mock.patch("tllm.runtime.residual_runtime.clear_dispatch_consumers") as clear, mock.patch(
            "tllm.runtime.residual_runtime.register_dispatch_consumer"
        ) as register:
            runtime.activate()
        self.assertIs(active_runtime(), runtime)
        clear.assert_called_once()
        register.assert_called_once()

    def test_empty_runtime_activation_preserves_legacy_consumers(self) -> None:
        runtime = TllmRuntime()
        with mock.patch("tllm.runtime.residual_runtime.clear_dispatch_consumers") as clear:
            runtime.activate()
        clear.assert_not_called()

    def test_active_runtime_register_consumer_mirrors_to_dispatch(self) -> None:
        runtime = TllmRuntime()
        with mock.patch("tllm.runtime.residual_runtime.clear_dispatch_consumers"), mock.patch(
            "tllm.runtime.residual_runtime.register_dispatch_consumer"
        ) as register:
            runtime.activate()
            consumer = _Consumer("c2")
            runtime.register_consumer(consumer)
        register.assert_called_once_with(consumer)

    def test_config_defaults(self) -> None:
        cfg = TllmRuntimeConfig()
        self.assertEqual(cfg.failure.on_consumer_error, ConsumerErrorPolicy.FAIL_STEP)
        self.assertFalse(cfg.graph.allow_consumer_launch_in_graph)
        custom = TllmRuntimeConfig(buffer=BufferPolicy(graph_scratch_rows=128))
        self.assertEqual(custom.buffer.graph_scratch_rows, 128)


class FacadeUnitTest(unittest.TestCase):
    def test_public_facade_delegates_to_default_runtime(self) -> None:
        import tllm

        with mock.patch.object(TllmRuntime, "make_llm", autospec=True) as make:
            tllm.make_llm(model_name="m", dtype="bfloat16")
        make.assert_called_once()
        with mock.patch.object(TllmRuntime, "register_consumer", autospec=True) as reg:
            tllm.register_consumer(_Consumer("c"))
        reg.assert_called_once()


if __name__ == "__main__":
    unittest.main()

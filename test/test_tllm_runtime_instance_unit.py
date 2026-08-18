#!/usr/bin/env python3
"""Unit tests for the instance-scoped TllmRuntime and the legacy façade."""

from __future__ import annotations

import unittest
from unittest import mock

from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.consumer import DummyConsumer
from tllm.runtime.runtime import TllmRuntime, active_runtime, default_runtime


class _Consumer:
    def __init__(self, cid: str) -> None:
        self.consumer_id = cid

    def flows(self):
        return ()


class TllmRuntimeUnitTest(unittest.TestCase):
    def test_default_runtime_is_singleton(self) -> None:
        self.assertIs(default_runtime(), default_runtime())

    def test_activate_routes_and_applies_consumers(self) -> None:
        runtime = TllmRuntime(consumers=[_Consumer("c1")])
        with mock.patch("tllm.runtime.residual_runtime.clear_dispatch_consumers") as clear, mock.patch(
            "tllm.runtime.residual_runtime.register_dispatch_consumer"
        ) as register:
            runtime.activate()
        self.assertIs(active_runtime(), runtime)
        clear.assert_called_once()
        register.assert_called_once()

    def test_runtime_registration_and_clear_forward_to_active_plane(self) -> None:
        runtime = TllmRuntime()
        plane = mock.Mock()
        consumer = DummyConsumer(DummyConsumerConfig(consumer_id="runtime_plane"))
        with mock.patch("tllm.runtime.runtime._ACTIVE_RUNTIME", runtime), mock.patch(
            "tllm.runtime.mp.control.active_control_plane", return_value=plane
        ), mock.patch("tllm.runtime.residual_runtime.register_dispatch_consumer") as register, mock.patch(
            "tllm.runtime.residual_runtime.clear_dispatch_consumers"
        ) as clear, mock.patch("tllm.runtime.mp.bootstrap.refresh_exported_bootstrap_spec") as refresh:
            runtime.register_consumer(consumer)
            runtime.clear_consumers()

        register.assert_called_once_with(consumer)
        clear.assert_called_once_with()
        plane.register_consumer.assert_called_once()
        self.assertEqual(plane.register_consumer.call_args.args[0], "dummy")
        plane.clear_consumers.assert_called_once_with()
        self.assertEqual(refresh.call_count, 2)

    def test_bind_engine_runners_installs_active_control_plane(self) -> None:
        runtime = TllmRuntime()
        plane = mock.Mock()
        with mock.patch("tllm.runtime.mp.control.ControlPlane.try_bind", return_value=plane), mock.patch(
            "tllm.runtime.mp.control.set_active_control_plane"
        ) as set_active:
            runtime._bind_engine_runners(object())
        plane.call.assert_called_once_with("tllm_bind_runner_runtime")
        set_active.assert_called_once_with(plane)

    def test_legacy_make_llm_installs_active_control_plane(self) -> None:
        from tllm.runtime import lifecycle

        llm = object()
        plane = mock.Mock()
        core = mock.Mock()
        with mock.patch.object(lifecycle, "install_runner_patch"), mock.patch(
            "tllm.runtime.mp.bootstrap.export_bootstrap_spec_env"
        ), mock.patch(
            "tllm.util.tools.make_plain_llm", return_value=llm
        ) as make_plain, mock.patch(
            "tllm.runtime.mp.control.ControlPlane.try_bind", return_value=plane
        ) as try_bind, mock.patch(
            "tllm.runtime.mp.control.set_active_control_plane"
        ) as set_active:
            result = lifecycle.make_llm(core, model_name="model", dtype="bfloat16")

        self.assertIs(result, llm)
        make_plain.assert_called_once()
        self.assertEqual(
            make_plain.call_args.kwargs["worker_extension_cls"],
            "tllm.runtime.mp.worker_extension.TllmWorkerExtension",
        )
        try_bind.assert_called_once_with(llm)
        set_active.assert_called_once_with(plane)

    def test_plain_llm_forwards_explicit_scheduler_capacity(self) -> None:
        from tllm.util.tools import make_plain_llm

        llm = object()
        with mock.patch("tllm.util.tools.LLM", return_value=llm) as llm_cls:
            result = make_plain_llm(
                model_name="model",
                dtype="bfloat16",
                gpu_memory_utilization=0.5,
                max_model_len=1024,
                enable_prefix_caching=False,
                max_num_seqs=128,
                max_num_batched_tokens=128,
            )

        self.assertIs(result, llm)
        self.assertEqual(llm_cls.call_args.kwargs["max_num_seqs"], 128)
        self.assertEqual(llm_cls.call_args.kwargs["max_num_batched_tokens"], 128)

    def test_plain_llm_preserves_vllm_scheduler_defaults(self) -> None:
        from tllm.util.tools import make_plain_llm

        with mock.patch("tllm.util.tools.LLM") as llm_cls:
            make_plain_llm(
                model_name="model",
                dtype="bfloat16",
                gpu_memory_utilization=0.5,
                max_model_len=1024,
                enable_prefix_caching=False,
            )

        self.assertNotIn("max_num_seqs", llm_cls.call_args.kwargs)
        self.assertNotIn("max_num_batched_tokens", llm_cls.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for the instance-scoped TllmRuntime and the legacy façade."""

from __future__ import annotations

import unittest
from unittest import mock
from types import SimpleNamespace

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

    def test_workloads_are_instance_scoped_and_legacy_consumers_are_adapted(self) -> None:
        consumer = _Consumer("workload")
        runtime = TllmRuntime(workloads=(consumer,))

        self.assertEqual(runtime.workloads, (consumer,))
        self.assertEqual(runtime.consumers, (consumer,))

    def test_declared_pure_workload_is_admitted_for_native_region_binding(self) -> None:
        from tllm.workload import Before, task, workload

        @workload(id="no.adapter")
        class NoAdapter:
            @task(timing=Before("step.end"))
            def run(self) -> None:
                return None

        runtime = TllmRuntime(workloads=(NoAdapter(),))
        llm = object()
        with mock.patch(
            "tllm.runtime.residual_runtime.make_llm", return_value=llm
        ) as make_llm, mock.patch(
            "tllm.runtime.mp.control.ControlPlane.try_bind", return_value=None
        ):
            result = runtime.make_llm(model_name="model")
        self.assertIs(result, llm)
        make_llm.assert_called_once_with(model_name="model")
        self.assertIsNotNone(runtime.workload_execution_bindings)

    def test_native_authority_rejects_an_unrelated_legacy_sidecar(self) -> None:
        """Mixed authority remains runtime-wide until resources declare ownership."""
        from tllm.consumers.esamp.workload import (
            ESampConfig,
            ESampWorkload,
        )

        native = ESampWorkload(ESampConfig(distiller_hidden_dim=3))
        sidecar = _Consumer("gtta")

        with self.assertRaisesRegex(RuntimeError, "native Region authority"):
            TllmRuntime(workloads=(native,), consumers=(sidecar,))

    def test_backend_cold_path_compiles_declared_workloads_before_forward(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.backends.vllm.workgraph_runtime import cold_compile_loaded_workloads
        from tllm.runtime import residual_runtime
        from tllm.runtime.graph_policy import GraphMode
        from tllm.workgraph.anchors import BackendResolution
        from tllm.workload import Between, task, workload

        @workload(id="cold.compile")
        class ColdWorkload:
            @task(timing=Between("model.execute", "step.end"))
            def run(self) -> None:
                return None

        runtime = TllmRuntime(
            workloads=(ColdWorkload(),), workload_memory_budget_bytes=64
        )
        previous_resolution = residual_runtime.RUNTIME.workgraph_backend_resolution
        previous_compilation = residual_runtime.RUNTIME.workload_compilation
        previous_execution = residual_runtime.RUNTIME.workload_execution
        previous_physical_executor = residual_runtime.RUNTIME.physical_executor
        try:
            residual_runtime.RUNTIME.workgraph_backend_resolution = BackendResolution(
                capabilities=V027MRv2Driver().capabilities(),
                graph_mode=GraphMode.EAGER,
            )
            with mock.patch(
                "tllm.runtime.runtime.active_runtime", return_value=runtime
            ):
                cold_compile_loaded_workloads(
                    core=residual_runtime,
                    runner=SimpleNamespace(device="cpu")
                )

            compilation = runtime.workload_compilation
            self.assertIsNotNone(compilation)
            self.assertIs(residual_runtime.RUNTIME.workload_compilation, compilation)
            assert compilation is not None
            self.assertEqual(compilation.timeline.runner_family, "mrv2")
            self.assertEqual(compilation.timeline.static_headroom_bytes, 64)
            self.assertIsNotNone(compilation.physical)
            self.assertIsNotNone(residual_runtime.RUNTIME.workload_execution)
        finally:
            current_execution = residual_runtime.RUNTIME.workload_execution
            if current_execution is not None and current_execution is not previous_execution:
                current_execution.close()
            residual_runtime.RUNTIME.workgraph_backend_resolution = previous_resolution
            residual_runtime.RUNTIME.workload_compilation = previous_compilation
            residual_runtime.RUNTIME.workload_execution = previous_execution
            residual_runtime.RUNTIME.physical_executor = previous_physical_executor

    def test_root_make_llm_routes_canonical_workloads_keyword(self) -> None:
        import tllm

        consumer = _Consumer("root")
        llm = object()
        with mock.patch("tllm.runtime.runtime.TllmRuntime.make_llm", return_value=llm) as make:
            result = tllm.make_llm(model_name="model", workloads=(consumer,))

        self.assertIs(result, llm)
        self.assertEqual(make.call_args.kwargs, {"model_name": "model"})

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
            "tllm.runtime.lifecycle.make_plain_llm", return_value=llm
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
        with mock.patch("vllm.LLM", return_value=llm) as llm_cls:
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

        with mock.patch("vllm.LLM") as llm_cls:
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
